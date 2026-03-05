# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import torch
import torch.nn as nn

import ttab.model_adaptation.utils as adaptation_utils
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer

# Feature Bank
class FeatureBank:
    def __init__(self, num_classes, feat_dim, momentum, device):
        self.num_classes = num_classes
        self.momentum = momentum
        self.device = device

        self.bank = torch.zeros(num_classes, feat_dim, device=device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, features, labels):
        for c in labels.unique():
            mask = labels == c
            centroid = features[mask].mean(0)
            if not self.initialized[c]:
                self.bank[c] = centroid
                self.initialized[c] = True
            else:
                self.bank[c] = (
                    self.momentum * self.bank[c]
                    + (1 - self.momentum) * centroid
                )

    def get(self):
        return self.bank

# Feature Regularizers
def feature_redundancy(z):
    z = z - z.mean(dim=0, keepdim=True)
    std = z.std(dim=0, keepdim=True) + 1e-5
    z = z / std
    cov = z.T @ z / z.shape[0]
    off_diag = cov - torch.diag(torch.diag(cov))
    D = z.shape[1]
    return (off_diag ** 2).sum() / (D - 1)

def feature_inequity(z, classifier):
    mu = z.mean(dim=0, keepdim=True)
    logits = classifier(mu)
    p = torch.softmax(logits, dim=1).squeeze(0)
    C = p.shape[0]
    return torch.log(torch.tensor(C, device=z.device)) + torch.sum(p * torch.log(p + 1e-12))

class SAR2(BaseAdaptation):
    """
    Adapt in the Wild: Test-Time Entropy Minimization with Sharpness and Feature Regularization,
    https://arxiv.org/abs/2509.04977
    """

    def __init__(self, meta_conf, model: nn.Module):
        super().__init__(meta_conf, model)
        self.ema = None
        self.feature_bank = None
        self._cached_features = None
        self._feature_hook = None
        self.classifier = self._get_classifier()

    def _get_classifier(self):
        for attr in ["classifier", "fc", "head"]:
            module = getattr(self._model, attr, None)
            if module is not None and isinstance(module, nn.Module):
                # ensure it's a linear-like layer
                if hasattr(module, "weight"):
                    return module
        raise ValueError(
            f"SAR2: No valid classifier head found in model "
            f"({self._model.__class__.__name__})."
        )

    def _register_feature_hook(self, model):

        if hasattr(model, "fc_norm"):
            target_layer = model.fc_norm
        elif hasattr(model, "global_pool"):
            target_layer = model.global_pool
        elif hasattr(model, "avgpool"):
            target_layer = model.avgpool
        else:
            raise RuntimeError("SAR2: cannot find classifier layer for feature hook.")

        def hook_fn(module, input, output):
            self._cached_features = output

        self._feature_hook = target_layer.register_forward_hook(hook_fn)

    def _initialize_model(self, model):
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

        model = model.to(self._meta_conf.device)

        # register hook ONCE
        self._register_feature_hook(model)

        return model

    def _initialize_trainable_parameters(self):
        adapt_params = []
        adapt_param_names = []
        self._adapt_module_names = []

        for name_module, module in self._model.named_modules():
            if "layer4" in name_module:
                continue
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                self._adapt_module_names.append(name_module)
                for n, p in module.named_parameters():
                    if n in ["weight", "bias"]:
                        adapt_params.append(p)
                        adapt_param_names.append(f"{name_module}.{n}")

        return adapt_params, adapt_param_names

    def _initialize_optimizer(self, params):
        base_optimizer = torch.optim.SGD
        optimizer = adaptation_utils.SAM(
            params,
            base_optimizer=base_optimizer,
            lr=self._meta_conf.lr,
            momentum=getattr(self._meta_conf, "momentum", 0.9),
        )
        return optimizer

    def reset(self):
        self._model.load_state_dict(self.model_state_dict)
        self._optimizer.load_state_dict(self._base_optimizer.state_dict())
        self.ema = None
        self.feature_bank = None

    @staticmethod
    def update_ema(ema, new):
        if ema is None:
            return new
        return 0.9 * ema + 0.1 * new

    def one_adapt_step(
        self,
        model,
        optimizer,
        batch,
        ema,
        timer,
        random_seed=None,
    ):
        optimizer.zero_grad()

        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                y_hat = model(batch._x)

            feats = self._cached_features.flatten(1)
            assert feats is not None, "SAR2: feature hook failed."

            ent = adaptation_utils.softmax_entropy(y_hat)
            mask = ent < self._meta_conf.sar_margin_e0
            loss = ent[mask].mean()

        # Feature Bank
        if self.feature_bank is None:
            self.feature_bank = FeatureBank(
                num_classes=y_hat.shape[1],
                feat_dim=feats.shape[1],
                momentum=self._meta_conf.feature_bank_momentum,
                device=feats.device,
            )

        pseudo = y_hat.argmax(dim=1)
        self.feature_bank.update(feats.detach(), pseudo.detach())
        
        valid_mask = self.feature_bank.initialized
        centroids = self.feature_bank.bank[valid_mask]

        if centroids.shape[0] > 1:
            r_loss = feature_redundancy(centroids)
            i_loss = feature_inequity(centroids, self.classifier)
        else:
            r_loss = torch.tensor(0., device=feats.device)
            i_loss = torch.tensor(0., device=feats.device)

        loss = (
            loss
            + self._meta_conf.alpha * r_loss
            + self._meta_conf.beta * i_loss
        )

        with timer("backward"):
            loss.backward()
            optimizer.first_step(zero_grad=True)
            y_hat2 = model(batch._x)
            feats2 = self._cached_features.flatten(1)
            ent2 = adaptation_utils.softmax_entropy(y_hat2)
            loss2 = ent2[mask].mean()
            valid_mask = self.feature_bank.initialized
            centroids2 = self.feature_bank.bank[valid_mask]
            if centroids2.shape[0] > 1:
                r_loss2 = feature_redundancy(centroids2)
                i_loss2 = feature_inequity(centroids2, self.classifier)
            else:
                r_loss2 = torch.tensor(0., device=feats2.device)
                i_loss2 = torch.tensor(0., device=feats2.device)
            loss2 = loss2 + self._meta_conf.alpha * r_loss2 + self._meta_conf.beta * i_loss2
            ema = self.update_ema(ema, loss2.item())
            loss2.backward()
            optimizer.second_step(zero_grad=True)

        reset_flag = ema is not None and ema < self._meta_conf.reset_constant_em

        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "yhat": y_hat,
        }, reset_flag, ema

    def run_multiple_steps(
        self,
        optimizer,
        batch,
        model_selection_method,
        nbsteps,
        timer,
        random_seed=None,
    ):
        for step in range(1, nbsteps + 1):
            result, reset_flag, self.ema = self.one_adapt_step(
                self._model,
                optimizer,
                batch,
                self.ema,
                timer,
                random_seed,
            )

            model_selection_method.save_state(
                {
                    "model": copy.deepcopy(self._model).state_dict(),
                    "step": step,
                    "lr": self._meta_conf.lr,
                    **result,
                },
                current_batch=batch,
            )

            if reset_flag:
                self.reset()

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

        with timer("test_time_adaptation"):
            nbsteps = self._get_adaptation_steps(index=len(previous_batches))
            self.run_multiple_steps(
                self._optimizer,
                current_batch,
                model_selection_method,
                nbsteps,
                timer,
                random_seed=self._meta_conf.seed,
            )

        with timer("select_optimal_checkpoint"):
            optimal_state = model_selection_method.select_state()
            self._model.load_state_dict(optimal_state["model"])
            model_selection_method.clean_up()

        with timer("evaluate_adaptation_result"):
            metrics.eval(current_batch._y, optimal_state["yhat"])

    @property
    def name(self):
        return "sar2"