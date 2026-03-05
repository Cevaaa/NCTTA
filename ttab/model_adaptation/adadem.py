# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer

class AdaDEMLoss(object):
    def __init__(self, pi=0.1, reduction="mean", mode="adadem"):
        """
        mode:
            - "adadem"       : AdaDEM
            - "adadem-norm"  : AdaDEM-Norm
            - "adadem-mec"   : AdaDEM-MEC
        """
        self.pi = pi
        self.reduction = reduction
        self.mode = mode
        self.avg_pred = None

    def reset(self):
        self.avg_pred = None

    def __call__(self, logits):
        """
        logits: (N, C)
        """
        p = F.softmax(logits, dim=1)

        # update marginal estimator
        self._step(p)

        # CADF gradient norm (delta)
        with torch.no_grad():
            T = -(p * logits).sum(1, keepdim=True)
            grad = (logits + T + 1) * p
            delta = grad.abs().sum(1, keepdim=True)

        pseudo_label = p.argmax(1)

        if self.mode == "adadem":
            p = (p - self.avg_pred[pseudo_label].detach()) / delta.detach()
        elif self.mode == "adadem-norm":
            p = (p - p.detach()) / delta.detach()
        elif self.mode == "adadem-mec":
            p = p - self.avg_pred[pseudo_label].detach()
        else:
            raise ValueError(f"Unknown AdaDEM mode: {self.mode}")

        loss = -(p * logits).sum(1)

        if self.reduction == "mean":
            return loss.mean(0)
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Reduction {self.reduction} not supported")

    @torch.no_grad()
    def _step(self, p):
        pseudo_label = p.argmax(1)

        if self.avg_pred is None:
            C = p.size(1)
            self.avg_pred = torch.ones((C, C), device=p.device) / C

        self.avg_pred = self.avg_pred.detach()

        for cls in torch.unique(pseudo_label):
            cls_mask = pseudo_label == cls
            self.avg_pred[cls] = (
                (1.0 - self.pi) * self.avg_pred[cls]
                + self.pi * p[cls_mask].mean(0).detach()
            )


class AdaDEM(BaseAdaptation):
    """
    Decoupled Entropy Minimization,
    https://arxiv.org/abs/2511.03256, 
    https://github.com/HAIV-Lab/DEM/
    """

    def __init__(self, meta_conf, model: nn.Module):
        super().__init__(meta_conf, model)

        # AdaDEM configuration
        self.adadem_loss = AdaDEMLoss(
            pi=getattr(meta_conf, "adadem_pi", 0.1),
            reduction="mean",
            mode=getattr(meta_conf, "adadem_mode", "adadem"),
        )

    def _initialize_model(self, model: nn.Module):
        model.train()
        model.requires_grad_(False)

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

        return model.to(self._meta_conf.device)

    def _initialize_trainable_parameters(self):
        adapt_params = []
        adapt_param_names = []
        self._adapt_module_names = []

        for name_module, module in self._model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                self._adapt_module_names.append(name_module)
                for name_param, param in module.named_parameters():
                    if name_param in ["weight", "bias"]:
                        adapt_params.append(param)
                        adapt_param_names.append(f"{name_module}.{name_param}")

        assert len(adapt_params) > 0, "AdaDEM-Tent requires norm parameters."
        return adapt_params, adapt_param_names

    def one_adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        timer: Timer,
        random_seed: int = None,
    ):
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                logits = model(batch._x)

            # AdaDEM loss
            loss = self.adadem_loss(logits)

            if self.fishers is not None:
                ewc_loss = 0.0
                for name, param in model.named_parameters():
                    if name in self.fishers:
                        fisher, mean = self.fishers[name]
                        ewc_loss += (
                            self._meta_conf.fisher_alpha
                            * (fisher * (param - mean) ** 2).sum()
                        )
                loss += ewc_loss

        with timer("backward"):
            loss.backward()

            grads = {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            }

            optimizer.step()
            optimizer.zero_grad()

        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
            "yhat": logits,
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
            result = self.one_adapt_step(
                model, optimizer, batch, timer, random_seed
            )

            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(model).state_dict(),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **result,
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
        log = functools.partial(logger.log, display=self._meta_conf.debug)

        if episodic:
            log("\t[Reset] episodic adaptation")
            self.reset()
            self.adadem_loss.reset()

        model_selection_method.initialize()

        if self._meta_conf.record_preadapted_perf:
            with timer("evaluate_preadapted_performance"):
                self._model.eval()
                with torch.no_grad():
                    yhat = self._model(current_batch._x)
                self._model.train()
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, "preadapted_accuracy_top1"
                )

        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            self.run_multiple_steps(
                self._model,
                self._optimizer,
                current_batch,
                model_selection_method,
                nbsteps,
                timer,
                random_seed=self._meta_conf.seed,
            )

        with timer("select_optimal_checkpoint"):
            optimal = model_selection_method.select_state()
            self._model.load_state_dict(optimal["model"])
            model_selection_method.clean_up()

            if self._oracle_model_selection:
                self.oracle_adaptation_steps.append(optimal["step"])
                self._optimizer.load_state_dict(optimal["optimizer"])

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, optimal["yhat"])

        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    @property
    def name(self):
        return "adadem"