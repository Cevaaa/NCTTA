# -*- coding: utf-8 -*-
import copy
import functools
from typing import List, Optional, Tuple

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

# Utility: stochastic depth style sub-network forward
def forward_with_stochastic_depth(model: nn.Module, x: torch.Tensor, drop_ratio: float):
    """
    A lightweight approximation of stochastic depth for TTA.
    Assumes dropout exists or BN affine update dominates.
    """
    def _set_dropout(m):
        if isinstance(m, nn.Dropout):
            m.p = drop_ratio

    model.apply(_set_dropout)
    return model(x)

class EATAC(BaseAdaptation):
    """
    Uncertainty-Calibrated Test-Time Model Adaptation without Forgetting, 
    https://arxiv.org/abs/2403.11491
    """

    def __init__(self, meta_conf, model: nn.Module):
        super().__init__(meta_conf, model)
        self.current_model_probs = None
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0

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
                for n, p in module.named_parameters():
                    if n in ["weight", "bias"]:
                        adapt_params.append(p)
                        adapt_param_names.append(f"{name_module}.{n}")

        assert len(adapt_params) > 0, "EATA-C needs norm affine parameters."
        return adapt_params, adapt_param_names

    @staticmethod
    def update_model_probs(old, new):
        if new is None or new.size(0) == 0:
            return old
        new = new.mean(0)
        if old is None:
            return new.detach()
        return (0.9 * old + 0.1 * new).detach()

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    def one_adapt_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        current_model_probs: Optional[torch.Tensor],
        timer: Timer,
        random_seed: Optional[int] = None,
    ) -> Optional[Tuple[dict, int, int, torch.Tensor]]:

        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                y_full = model(batch._x)

            prob_full = y_full.softmax(1)
            entropy_full = adaptation_utils.softmax_entropy(y_full)

            # Reliable samples
            ids_e = torch.where(
                entropy_full < self._meta_conf.eata_margin_e0
            )[0]
            if ids_e.numel() == 0:
                return None
            # Redundant samples
            if current_model_probs is not None:
                cosine = F.cosine_similarity(
                    current_model_probs.unsqueeze(0),
                    prob_full[ids_e],
                    dim=1,
                )
                ids_d = torch.where(
                    torch.abs(cosine) < self._meta_conf.eata_margin_d0
                )[0]
                if ids_d.numel() == 0:
                    return None
                ids = ids_e[ids_d]
            else:
                ids = ids_e

            if ids.numel() == 0:
                return None

            # Sub-network
            with fork_rng_with_seed(random_seed):
                y_sub = forward_with_stochastic_depth(
                    model,
                    batch._x[ids],
                    self._meta_conf.stochastic_depth_ratio,
                )

            prob_sub = y_sub.softmax(1)
            prob_full_sel = prob_full[ids].detach()

            # Consistency loss
            p = self._meta_conf.consistency_smooth_p
            prob_fuse = (prob_full_sel + (1 - p) * prob_sub) / (2 - p)

            consistency_loss = F.kl_div(
                prob_sub.log(),
                prob_fuse,
                reduction="batchmean",
            )

            # Data uncertainty (min-max entropy)
            agree = (
                prob_sub.argmax(1) == prob_full_sel.argmax(1)
            ).float()
            c = torch.where(agree > 0, 1.0, -1.0)

            entropy_sub = adaptation_utils.softmax_entropy(y_sub)
            entropy_loss = (c * entropy_sub).mean()

            loss = consistency_loss + self._meta_conf.entropy_alpha * entropy_loss

            # Fisher regularization
            if self.fishers is not None:
                ewc = 0.0
                for name, param in model.named_parameters():
                    if name in self.fishers:
                        fisher, ref = self.fishers[name]
                        ewc += (fisher * (param - ref) ** 2).sum()
                loss += self._meta_conf.fisher_alpha * ewc

        with timer("backward"):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        updated_probs = self.update_model_probs(
            current_model_probs, prob_full[ids]
        )

        return (
            {
                "loss": loss.item(),
                "yhat": y_full,
                "optimizer": copy.deepcopy(optimizer).state_dict(),
            },
            ids.numel(),
            ids_e.numel(),
            updated_probs,
        )

    def run_multiple_steps(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Batch,
        model_selection_method: BaseSelection,
        nbsteps: int,
        timer: Timer,
        random_seed: Optional[int] = None,
    ):
        for step in range(1, nbsteps + 1):
            result = self.one_adapt_step(
                model,
                optimizer,
                batch,
                self.current_model_probs,
                timer,
                random_seed,
            )
            if result is None:
                continue

            adaptation_result, n2, n1, updated_probs = result

            self.num_samples_update_2 += n2
            self.num_samples_update_1 += n1
            self.reset_model_probs(updated_probs)

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
        log = functools.partial(logger.log, display=self._meta_conf.debug)

        if episodic:
            self.reset()

        model_selection_method.initialize()

        # pre-adaptation eval
        if self._meta_conf.record_preadapted_perf:
            with torch.no_grad():
                y = self._model(current_batch._x)
            metrics.eval_auxiliary_metric(
                current_batch._y, y, "preadapted_accuracy_top1"
            )

        with timer("test_time_adaptation"):
            steps = self._get_adaptation_steps(len(previous_batches))
            self.run_multiple_steps(
                self._model,
                self._optimizer,
                current_batch,
                model_selection_method,
                steps,
                timer,
                self._meta_conf.seed,
            )


        with timer("select_optimal_checkpoint"):
            optimal_state = model_selection_method.select_state()

            if optimal_state is None:
                with torch.no_grad():
                    yhat = self._model(current_batch._x)
                model_selection_method.clean_up()
            else:
                self._model.load_state_dict(optimal_state["model"])
                yhat = optimal_state["yhat"]
                model_selection_method.clean_up()

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, yhat)

    @property
    def name(self):
        return "eata_c"