# -*- coding: utf-8 -*-
import copy
import functools
import random
import math
from typing import List
from einops import rearrange

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer
from torchvision import transforms


class DEYO(BaseAdaptation):
    """
    Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors,
    https://arxiv.org/abs/2403.07366,
    https://whitesnowdrop.github.io/DeYO/
    """
    
    def __init__(self, meta_conf, model: nn.Module):
        super(DEYO, self).__init__(meta_conf, model)
        self.mem_records = []
        
        self._deyo_margin = self._meta_conf.deyo_margin
        self._filter_ent = self._meta_conf.filter_ent
        self._aug_type = self._meta_conf.aug_type
        self._occlusion_size = self._meta_conf.occlusion_size
        self._row_start = self._meta_conf.row_start
        self._column_start = self._meta_conf.column_start
        self._patch_len = self._meta_conf.patch_len
        self._filter_plpd = self._meta_conf.filter_plpd
        self._plpd_threshold = self._meta_conf.plpd_threshold
        self._reweight_ent = self._meta_conf.reweight_ent
        self._reweight_plpd = self._meta_conf.reweight_plpd
        self._margin = self._meta_conf.margin

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
        model.train()
        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # bn module always uses batch statistics, in both training and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        """
        Collect the affine scale + shift parameters from norm layers.

        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        self._adapt_module_names = []
        adapt_params = []
        adapt_param_names = []

        for name_module, module in self._model.named_modules():
            if isinstance(
                module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)
            ):  # only bn is used in the paper.
                self._adapt_module_names.append(name_module)
                for name_parameter, parameter in module.named_parameters():
                    if name_parameter in ["weight", "bias"]:
                        adapt_params.append(parameter)
                        adapt_param_names.append(f"{name_module}.{name_parameter}")

        assert (
            len(self._adapt_module_names) > 0
        ), "DEYO needs some adaptable model parameters."
        return adapt_params, adapt_param_names

    def one_adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        """adapt the model in one step."""
        with timer("forward"):
            x, targets = batch._x, batch._y
            with fork_rng_with_seed(random_seed):
                outputs = model(x)

            # calculate entropys
            entropys = adaptation_utils.softmax_entropy(outputs)
            x_prime = x.detach()

            if self._aug_type=='occ':
                first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
                final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
                occlusion_window = final_mean.expand(-1, -1, self._occlusion_size, self._occlusion_size)
                x_prime[:, :, self._row_start:self._row_start+self._occlusion_size,self._column_start:self._column_start+self._occlusion_size] = occlusion_window
            elif self._aug_type=='patch':
                resize_t = torchvision.transforms.Resize(((x.shape[-1]//self._patch_len)*self._patch_len,(x.shape[-1]//self._patch_len)*self._patch_len))
                resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
                x_prime = resize_t(x_prime)
                x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=self._patch_len, ps2=self._patch_len)
                perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
                x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
                x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=self._patch_len, ps2=self._patch_len)
                x_prime = resize_o(x_prime)
            elif self._aug_type=='pixel':
                x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
                x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
                x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
            with torch.no_grad():
                outputs_prime = model(x_prime)

            prob_outputs = outputs.softmax(1)
            prob_outputs_prime = outputs_prime.softmax(1)

            cls1 = prob_outputs.argmax(dim=1)

            plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
            plpd = plpd.reshape(-1)

            if self._filter_plpd:
                filter_ids_4 = torch.where((plpd > self._plpd_threshold) & (entropys < self._deyo_margin))
            else:
                filter_ids_4 = torch.where((plpd > -2.0) & (entropys < self._deyo_margin))
            entropys = entropys[filter_ids_4]
            final_backward = len(entropys)
            plpd = plpd[filter_ids_4]
            if final_backward != 0:
                if self._reweight_ent or self._reweight_plpd:
                    coeff = (self._reweight_ent * (1 / (torch.exp(((entropys.clone().detach()) - self._margin)))) +
                    self._reweight_plpd * (1 / (torch.exp(-1. * plpd.clone().detach())))
                    )
                    entropys = entropys.mul(coeff)
                loss = entropys.mean(0)
            else:
                loss = torch.tensor([0.], requires_grad=True)
                
            # apply fisher regularization when enabled
            if self.fishers is not None:
                ewc_loss = 0
                for name, param in model.named_parameters():
                    if name in self.fishers:
                        ewc_loss += (
                            self._meta_conf.fisher_alpha
                            * (
                                self.fishers[name][0]
                                * (param - self.fishers[name][1]) ** 2
                            ).sum()
                        )
                loss += ewc_loss
        
        with timer("backward"):
            loss.backward()
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            optimizer.step()
            optimizer.zero_grad()
        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
            "yhat": outputs,
        }

    def run_multiple_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: int = None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                batch,
                timer,
                random_seed=random_seed,
            )

            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(model).state_dict(),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **adaptation_result,
                },
                current_batch=batch,
            )

    def adapt_and_eval(
        self,
        episodic: bool,
        metrics: Metrics,
        model_selection_method: BaseSelection,
        current_batch: Batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
    ):
        """The key entry of test-time adaptation."""
        # some simple initialization.
        log = functools.partial(logger.log, display=self._meta_conf.debug)
        if episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        # evaluate the per batch pre-adapted performance. Different with no adaptation.
        if self._meta_conf.record_preadapted_perf:
            with timer("evaluate_preadapted_performance"):
                self._model.eval()
                with torch.no_grad():
                    yhat = self._model(current_batch._x)
                self._model.train()
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                )

        # adaptation.
        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            log(f"\tadapt the model for {nbsteps} steps with lr={self._meta_conf.lr}.")
            self.run_multiple_steps(
                model=self._model,
                optimizer=self._optimizer,
                batch=current_batch,
                model_selection_method=model_selection_method,
                nbsteps=nbsteps,
                timer=timer,
                random_seed=self._meta_conf.seed,
            )

        # select the optimal checkpoint, and return the corresponding prediction.
        with timer("select_optimal_checkpoint"):
            optimal_state = model_selection_method.select_state()
            log(
                f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
            )

            self._model.load_state_dict(optimal_state["model"])
            model_selection_method.clean_up()

            if self._oracle_model_selection:
                # oracle model selection needs to save steps
                self.oracle_adaptation_steps.append(optimal_state["step"])
                # update optimizer.
                self._optimizer.load_state_dict(optimal_state["optimizer"])

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, optimal_state["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    optimal_state["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )

        # stochastic restore part of model parameters if enabled.
        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    @property
    def name(self):
        return "deyo"
