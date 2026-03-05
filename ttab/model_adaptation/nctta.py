# -*- coding: utf-8 -*-
import copy
import functools
import math
import warnings
from typing import List, Optional, Literal, Tuple
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.loads.define_model import load_pretrained_model
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer

def compute_nc_loss(
    weight: torch.Tensor,               # (K, D)
    features: torch.Tensor,             # (B, D)
    y_hat: torch.Tensor,                # (B, K) predicted scores (probs or logits)
    top_k: int = 3,
    type: Literal["infonce", "cosine_l2", "margin_triplet", "hinge_cos"] = "infonce",
    metric: Literal["cos", "euclid"] = "cos",
    tau_align: float = 0.07,            # temperature for InfoNCE alignment
    margin: float = 0.2,                # margin for triplet/hinge
    mix_prob_weight: float = 0.0,       # mix ratio to blend q_dist with q_prob
    y_hat_is_logits: bool = False,      # set True if y_hat are logits (will softmax)
    reduce: Literal["mean", "sum", "none"] = "mean",
    safe_eps: float = 1e-8,
) -> Tuple[torch.Tensor, dict]:
    assert weight.dim() == 2 and features.dim() == 2 and y_hat.dim() == 2
    B, D = features.shape
    K, Dw = weight.shape
    assert Dw == D and y_hat.shape == (B, K)
    device = features.device
    dtype = features.dtype

    # Normalize to unit sphere (geometry-friendly)
    w_hat = F.normalize(weight, p=2, dim=1)
    h_hat = F.normalize(features, p=2, dim=1)
    # Ensure y_hat is probability distribution
    if y_hat_is_logits:
        probs = F.softmax(y_hat, dim=1)
    else:
        probs = (y_hat + safe_eps).clamp_min(safe_eps)
        probs = probs / probs.sum(dim=1, keepdim=True)
    # Cosine similarities and distances
    sims = h_hat @ w_hat.t()
    if metric == "cos":
        dists = 1.0 - sims.clamp(-1.0, 1.0)
    elif metric == "euclid":
        dists = (2.0 - 2.0 * sims.clamp(-1.0, 1.0)).clamp_min(0.0).sqrt()
    else:
        raise ValueError(f"Unknown metric: {metric}")
    # Select top-k classes based on predicted probabilities
    topk_prob, topk_idx = torch.topk(probs, k=min(top_k, K), dim=1, largest=True)
    # Gather corresponding distances and similarities
    topk_dists = torch.gather(dists, dim=1, index=topk_idx)
    topk_sims  = torch.gather(sims,  dim=1, index=topk_idx)
    # Distance-based soft target q_dist
    d_mean = topk_dists.mean(dim=1, keepdim=True)
    d_std  = topk_dists.std(dim=1, keepdim=True, unbiased=False) + safe_eps
    norm_d = (topk_dists - d_mean) / d_std
    logits_q_dist = -norm_d
    q_dist = F.softmax(logits_q_dist, dim=1)
    # Probability-based soft target q_prob
    logits_q_prob = torch.log(topk_prob + safe_eps)
    q_prob = F.softmax(logits_q_prob, dim=1)
    # Blend two soft targets
    alpha = float(mix_prob_weight)
    if alpha <= 0:
        q = q_dist
    elif alpha >= 1:
        q = q_prob
    else:
        q = (1 - alpha) * q_dist + alpha * q_prob
        q = q + safe_eps
        q = q / q.sum(dim=1, keepdim=True)
    # Primary margin diagnostics
    sims_sorted, _ = sims.sort(dim=1, descending=True)
    primary_margin = sims_sorted[:, 0] - sims_sorted[:, 1].clamp_max(1e9)
    # nc Loss types
    if type == "infonce":
        exp_logits = torch.exp(sims / max(tau_align, safe_eps))
        exp_probs_topk = torch.gather(exp_logits, dim=1, index=topk_idx)
        loss_per = -torch.log(
            (q * exp_probs_topk).sum(dim=1) /
            torch.sum(exp_logits, dim=1)
        )
    elif type == "cosine_l2":
        w_soft = torch.einsum("bk,bkd->bd", q, w_hat[topk_idx])
        w_soft = F.normalize(w_soft, dim=1)
        diff = (h_hat - w_soft)
        loss_per = (diff * diff).sum(dim=1)
    elif type == "margin_triplet":
        w_pos = torch.einsum("bk,bkd->bd", q, w_hat[topk_idx])
        w_pos = F.normalize(w_pos, dim=1)
        sim_pos = (h_hat * w_pos).sum(dim=1, keepdim=True)
        mask_all = torch.ones_like(sims, dtype=torch.bool)
        mask_topk = torch.zeros_like(mask_all)
        mask_topk.scatter_(dim=1, index=topk_idx, src=torch.ones_like(topk_idx, dtype=torch.bool))
        mask_neg = mask_all & (~mask_topk)
        sims_neg = sims.masked_fill(~mask_neg, -1e9)
        sim_neg_hard, _ = sims_neg.max(dim=1, keepdim=True)
        loss_per = F.relu(margin - (sim_pos - sim_neg_hard)).squeeze(1)
    elif type == "hinge_cos":
        s_pos = (q * topk_sims).sum(dim=1, keepdim=True)
        mask_all = torch.ones_like(sims, dtype=torch.bool)
        mask_topk = torch.zeros_like(mask_all)
        mask_topk.scatter_(dim=1, index=topk_idx, src=torch.ones_like(topk_idx, dtype=torch.bool))
        mask_neg = mask_all & (~mask_topk)
        sims_neg = sims.masked_fill(~mask_neg, -1e9)
        s_neg, _ = sims_neg.max(dim=1, keepdim=True)
        loss_per = F.relu(margin - (s_pos - s_neg)).squeeze(1)
    else:
        raise ValueError(f"Unknown type: {type}")

    if reduce == "mean":
        loss = loss_per.mean()
    elif reduce == "sum":
        loss = loss_per.sum()
    elif reduce == "none":
        loss = loss_per
    else:
        raise ValueError(f"Unknown reduce: {reduce}")

    aux = {
        "q": q.detach(),
        "topk_idx": topk_idx.detach(),
        "topk_dists": topk_dists.detach(),
        "topk_sims": topk_sims.detach(),
        "sims": sims.detach(),
        "dists": dists.detach(),
        "primary_margin": primary_margin.detach(),
    }

    return loss, aux

class NCTTA(BaseAdaptation):
    """
    Neural Collapse in Test Time Adaptation,
    https://arxiv.org/abs/2512.10421,
    https://github.com/Cevaaa/NCTTA
    """

    def __init__(self, meta_conf, model: nn.Module):
        super(NCTTA, self).__init__(meta_conf, model)
        self.mem_records = []
        self._get_classifier_weights()
        self._thre_ent = self._meta_conf.thre_ent
        self._margin_ent = self._meta_conf.margin_ent
        self._reweight_ent = self._meta_conf.reweight_ent
        self._nu = self._meta_conf.nu
        self._eta = self._meta_conf.eta
        self._scale = self._meta_conf.scale
        self._top_k = self._meta_conf.top_k
        self._mix_prob_weight = self._meta_conf.mix_prob_weight
        avgpool_layer = self._get_feature_layer(self._model)
        self._cached_features = None  # init cache
        self._hook_handle = avgpool_layer.register_forward_hook(self._feature_hook)
    
    def _get_feature_layer(self, model: nn.Module) -> nn.Module:
        """
        Automatically find the feature extraction layer.
        """
        if hasattr(model, "fc_norm"):
            return model.fc_norm
        if hasattr(model, "global_pool"):
            return model.global_pool
        if hasattr(model, "avgpool"):
            return model.avgpool
        raise RuntimeError(
            "Cannot find feature layer automatically. "
            "Please specify it manually."
        )
    
    def _feature_hook(self, module, input, output):
        """
        Forward hook to capture normalized features.
        """
        batch_size = output.shape[0]
        feat = output.view(batch_size, -1)
        feat = F.normalize(feat, p=2, dim=-1)
        self._cached_features = feat

    def _initialize_model(self, model: nn.Module):
        """Configure model for adaptation."""
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
        ), "TENT needs some adaptable model parameters."
        return adapt_params, adapt_param_names
    
    def _get_classifier_weights(self):
        for attr in ["classifier", "fc", "head"]:
            module = getattr(self._model, attr, None)
            if module is not None:
                if hasattr(module, "weight"):
                    self.weight = module.weight.detach()
                    return
        raise ValueError("No classifier head found in model.")

    def compute_loss_function(self, y_hat, features):
        """Initialization"""
        batch_size = y_hat.size()[0]  # batch size
        h_normalized = F.normalize(features, p=2, dim=1) # Normalize feature vectors row-wise
        w_normalized = F.normalize(self.weight, p=2, dim=1) # Normalize classifier weights row-wise
        y_hat_indices = torch.argmax(y_hat, dim=1) # predicted class indices [batch_size]

        """Entropy loss"""
        loss_ent = adaptation_utils.softmax_entropy(y_hat)  # calculate entropy
        
        """NC loss"""
        loss_nc, aux = compute_nc_loss(
            weight=self.weight,
            features=features,
            y_hat=y_hat,                # (B, K)
            top_k=self._top_k,
            type="infonce",             # Alternatives: "cosine_l2"/"margin_triplet"/"hinge_cos"
            metric="cos",
            tau_align=1.0,
            margin=0.2,
            mix_prob_weight=self._mix_prob_weight,  # Blend confidence into q (remaining from geometric distance)
            y_hat_is_logits=True,       # y_hat are logits
            reduce="none"
        )

        """Weight"""
        w_selected = w_normalized[y_hat_indices]
        distance = torch.norm(h_normalized - w_selected, p=2, dim=1) # Compute Euclidean distance
        filter_ids = torch.where(loss_ent < self._thre_ent)

        # Entropy-based reweighting coefficient
        coeff_ent = self._reweight_ent * (
            1 / (torch.exp(((loss_ent.clone().detach()) - self._margin_ent)))
        )
        # Distance-based reweighting coefficient
        coeff_exp = self._nu / (1 + self._eta * distance.clone().detach())
        coeff = coeff_ent + coeff_exp

        """Total loss"""
        loss = (loss_ent + self._scale * loss_nc).mul(coeff)
        loss = loss[filter_ids].mean(0)
        return loss

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
            self._cached_features = None
            with fork_rng_with_seed(random_seed):
                y_hat = model(batch._x)

            if self._cached_features is None:
                raise RuntimeError("Feature hook did not capture features.")

            loss = self.compute_loss_function(
                y_hat,
                self._cached_features,
            )
            
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
            "yhat": y_hat,
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
                    "model": copy.deepcopy(model.state_dict()),
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
        return "nctta"
