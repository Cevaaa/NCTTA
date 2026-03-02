# -*- coding: utf-8 -*-
import copy
import functools
from typing import List

import PIL
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import ttab.loads.define_dataset as define_dataset
import ttab.model_adaptation.utils as adaptation_utils
from numpy import random
from torchvision.transforms import ColorJitter, Compose, Lambda
from ttab.api import Batch
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import Metrics
from ttab.utils.auxiliary import fork_rng_with_seed
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer

class ViDAInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, r2=64):
        super().__init__()

        self.linear_vida = nn.Linear(in_features, out_features, bias)
        self.vida_down = nn.Linear(in_features, r, bias=False)
        self.vida_up = nn.Linear(r, out_features, bias=False)
        self.vida_down2 = nn.Linear(in_features, r2, bias=False)
        self.vida_up2 = nn.Linear(r2, out_features, bias=False)
        self.scale1 = 1.0
        self.scale2 = 1.0

        nn.init.normal_(self.vida_down.weight, std=1 / r**2)
        nn.init.zeros_(self.vida_up.weight)

        nn.init.normal_(self.vida_down2.weight, std=1 / r2**2)
        nn.init.zeros_(self.vida_up2.weight)

    def forward(self, input):
        # 修正为加法形式
        return self.linear_vida(input) + self.vida_up(self.vida_down(input)) * self.scale1 + self.vida_up2(self.vida_down2(input)) * self.scale2
    
def inject_trainable_vida(
    model: nn.Module,
    r: int = 4,
    r2: int = 16,
    target_replace_module: list = None,
):
    """
    Inject vida into model, and returns vida parameter groups.
    """
    if target_replace_module is None:
        target_replace_module = ["CrossAttention", "Attention"]
    
    require_grad_params = []
    names = []

    for name_module, _module in model.named_modules():
        if not target_replace_module or _module.__class__.__name__ in target_replace_module:
            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear" and not "vida_" in name:
                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = ViDAInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        r2,
                    )
                    _tmp.linear_vida.weight = weight
                    if bias is not None:
                        _tmp.linear_vida.bias = bias

                    # 根据模块的层次结构适当设置
                    if hasattr(_module, name):
                        setattr(_module, name, _tmp)
                    elif '.' in name:
                        parent_name, child_name = name.rsplit('.', 1)
                        parent = _module.get_submodule(parent_name)
                        setattr(parent, child_name, _tmp)

                    require_grad_params.extend(
                        list(_tmp.vida_up.parameters())
                    )
                    require_grad_params.extend(
                        list(_tmp.vida_down.parameters())
                    )
                    _tmp.vida_up.weight.requires_grad = True
                    _tmp.vida_down.weight.requires_grad = True

                    require_grad_params.extend(
                        list(_tmp.vida_up2.parameters())
                    )
                    require_grad_params.extend(
                        list(_tmp.vida_down2.parameters())
                    )
                    _tmp.vida_up2.weight.requires_grad = True
                    _tmp.vida_down2.weight.requires_grad = True                    
                    names.append(name)

    return require_grad_params, names

class ViDA(BaseAdaptation):

    def __init__(self, meta_conf, model: nn.Module):
        super(ViDA, self).__init__(meta_conf, model)
        # 设置EMA更新参数
        self.alpha_teacher = getattr(meta_conf, "alpha_teacher", 0.99)
        self.alpha_vida = getattr(meta_conf, "alpha_vida", 0.99)
        self.unc_thr = getattr(meta_conf, "unc_thr", 0.2)
        self.rst_prob = getattr(meta_conf, "rst_prob", 0.001)

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
            else:
                m.requires_grad_(True)
        
        return model.to(self._meta_conf.device)

    # def _initialize_trainable_parameters(self):
    #     """select target params for adaptation methods."""
    #     self._adapt_module_names = []
    #     adapt_params = []
    #     adapt_param_names = []

    #     for name_module, module in self._model.named_modules():
    #         if "vida_" not in name_module:
    #             self._adapt_module_names.append(name_module)
    #             for name_parameter, parameter in module.named_parameters():
    #                 if name_parameter in ["weight", "bias"]:
    #                     adapt_params.append(parameter)
    #                     adapt_param_names.append(f"{name_module}.{name_parameter}")

    #     assert (
    #         len(self._adapt_module_names) > 0
    #     ), "ViDA needs some adaptable model parameters."
    #     return adapt_params, adapt_param_names
    
    # hjz modified, original is wrong.
    def _initialize_trainable_parameters(self):
        model_param, vida_param = self.collect_params(self._model)
        assert len(vida_param) > 0, "ViDA needs trainable ViDA parameters."
        return vida_param, ["vida_up", "vida_down", "vida_up2", "vida_down2"]


    def _post_safety_check(self):
        is_training = self._model.training
        assert is_training, "adaptation needs train mode: call model.train()."

        param_grads = [p.requires_grad for p in self._model.parameters()]
        has_any_params = any(param_grads)
        assert has_any_params, "adaptation needs some trainable params."
    
    def initialize(self, seed: int):
        """Initialize the algorithm."""
        if self._meta_conf.model_selection_method == "oracle_model_selection":
            self._oracle_model_selection = True
            self.oracle_adaptation_steps = []
        else:
            self._oracle_model_selection = False

        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))
        self._base_model = copy.deepcopy(self._model)  # update base model
        
        # 指定目标模块进行注入（可以为空以应用到所有线性层）
        target_modules = getattr(self._meta_conf, "target_modules", ["CrossAttention", "Attention"])
        vida_params, vida_name = inject_trainable_vida(
            self._model,
            self._meta_conf.vida_rank1,
            self._meta_conf.vida_rank2,
            target_modules
        )
        
        model_param, vida_param = self.collect_params(self._model)
        self._model.to(self._meta_conf.device)
        
        # 使用不同学习率
        self._optimizer = self._initialize_optimizer([
            {"params": model_param, "lr": self._meta_conf.lr},
            {"params": vida_param, "lr": self._meta_conf.ViDALR}
        ])
        
        self._base_optimizer = copy.deepcopy(self._optimizer)
        
        self._auxiliary_data_cls = define_dataset.ConstructAuxiliaryDataset(
            config=self._meta_conf
        )
        self.transform = self.get_aug_transforms(img_shape=self._meta_conf.img_shape)
        
        # Fisher正则化（如果使用）
        self.fishers = None
        
        # 保存模型状态
        self.model_state_dict, self.model_ema, self.model_anchor = self.copy_model_states(self._model)
        self.model_ema.to(self._meta_conf.device)
        self.model_anchor.to(self._meta_conf.device)
        
    def _preprocess_image(self, x, input_resolution=224):
        """将输入图像调整为期望的尺寸"""
        if x.shape[2] == input_resolution:
                return x # 尺寸相同时不调整！
        elif len(x.shape) == 3:
            x = x.unsqueeze(0)
            x = torch.nn.functional.interpolate(x, size=(input_resolution, input_resolution), 
                            mode='bilinear', align_corners=False)
            return x.squeeze(0)
        elif len(x.shape) == 4:
            return torch.nn.functional.interpolate(x, size=(input_resolution, input_resolution), 
                            mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected (B,C,H,W) or (C,H,W)")

    @staticmethod
    def collect_params(model):
        """收集所有可训练参数，区分vida参数和模型参数"""
        vida_params_list = []
        model_params_lst = []
        for name, param in model.named_parameters():
            if 'vida_' in name:
                vida_params_list.append(param)
            else:
                model_params_lst.append(param)     
        return model_params_lst, vida_params_list

    @staticmethod
    def get_aug_transforms(
        img_shape: tuple, gaussian_std: float = 0.005, soft: bool = False
    ):
        """Get augmentation transforms used at test time."""
        n_pixels = img_shape[0]

        clip_min, clip_max = 0.0, 1.0

        p_hflip = 0.5

        tta_transforms = transforms.Compose(
            [
                Clip(0.0, 1.0),
                ColorJitterPro(
                    brightness=[0.8, 1.2] if soft else [0.6, 1.4],
                    contrast=[0.85, 1.15] if soft else [0.7, 1.3],
                    saturation=[0.75, 1.25] if soft else [0.5, 1.5],
                    hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
                    gamma=[0.85, 1.15] if soft else [0.7, 1.3],
                ),
                transforms.Pad(padding=int(n_pixels / 2), padding_mode="edge"),
                transforms.RandomAffine(
                    degrees=[-8, 8] if soft else [-15, 15],
                    translate=(1 / 16, 1 / 16),
                    scale=(0.95, 1.05) if soft else (0.9, 1.1),
                    shear=None,
                    interpolation=PIL.Image.BILINEAR,
                    fill=None,
                ),
                transforms.GaussianBlur(
                    kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]
                ),
                transforms.CenterCrop(size=n_pixels),
                transforms.RandomHorizontalFlip(p=p_hflip),
                GaussianNoise(0, gaussian_std),
                Clip(clip_min, clip_max),
            ]
        )
        return tta_transforms

    @staticmethod
    def update_ema_variables(ema_model, model, alpha_teacher, alpha_vida=0.99):
        """区分不同类型参数的EMA更新"""
        for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
            if "vida_" in name:
                ema_param.data[:] = alpha_vida * ema_param[:].data[:] + (1 - alpha_vida) * param[:].data[:]
            else:
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        # for (ema_name, ema_param), (model_name, model_param) in zip(ema_model.named_parameters(), model.named_parameters()):
        #     if "vida_" in model_name:
        #         ema_param.data.mul_(alpha_vida).add_(model_param.data, alpha=1-alpha_vida)
        #     else:
        #         ema_param.data.mul_(alpha_teacher).add_(model_param.data, alpha=1-alpha_teacher)
        return ema_model

    @staticmethod
    def copy_model_states(model):
        """Copy the model states for resetting after adaptation."""
        model_state = copy.deepcopy(model.state_dict())
        model_anchor = copy.deepcopy(model)
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return model_state, ema_model, model_anchor

    @staticmethod
    def load_model_and_optimizer(
        model, optimizer, model_state, target_optimzer, 
        vida_optimizer=None, target_vida_optimzizer=None
    ):
        """Restore the model and optimizer states from copies."""
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(target_optimzer.state_dict())
        if vida_optimizer and target_vida_optimzizer:
            vida_optimizer.load_state_dict(target_vida_optimzizer.state_dict())
            
    def reset(self):
        """recover model and optimizer to their initial states."""
        self.load_model_and_optimizer(
            self._model, self._optimizer, 
            self.model_state_dict, 
            self._base_optimizer
        )
        # restore the teacher model.
        (
            self.model_state_dict,
            self.model_ema,
            self.model_anchor,
        ) = self.copy_model_states(self._model)

    def stochastic_restore(self, prob=None):
        """随机恢复部分参数为初始状态，增强适应的稳定性"""
        if prob is None:
            prob = self.rst_prob
            
        for nm, m in self._model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < prob).float().to(p.device)
                    with torch.no_grad():
                        p.data = self.model_state_dict[f"{nm}.{npp}"] * mask + p.data * (1. - mask)
                        
    def set_scale(self, update_model, high, low):
        """设置ViDA模块的比例因子"""
        for name, module in update_model.named_modules():
            if hasattr(module, 'scale1'):
                module.scale1 = low.item()
            if hasattr(module, 'scale2'):
                module.scale2 = high.item()

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
            with fork_rng_with_seed(random_seed):
                outputs = model(batch._x)
                
            # 使用数据增强进行多次前向传播以估计不确定性
            outputs_emas = []
            for i in range(self._meta_conf.aug_size):
                if "vit" in self._meta_conf.model_name.lower():
                    outputs_ = self.model_ema(self._preprocess_image(self.transform(batch._x))).detach()
                else:
                    outputs_ = self.model_ema(self.transform(batch._x)).detach()
                outputs_emas.append(outputs_)
                
            # 计算不确定性并调整缩放因子
            outputs_unc = torch.stack(outputs_emas)
            variance = torch.var(outputs_unc, dim=0)
            uncertainty = torch.mean(variance) * 0.1
            
            if uncertainty >= self.unc_thr:
                lambda_high = 1 + uncertainty
                lambda_low = 1 - uncertainty
            else:
                lambda_low = 1 + uncertainty
                lambda_high = 1 - uncertainty
                
            self.set_scale(update_model=model, high=lambda_high, low=lambda_low)
            self.set_scale(update_model=self.model_ema, high=lambda_high, low=lambda_low)
            
            # 教师模型推理
            outputs_ema = self.model_ema(batch._x)
            
            # 使用对称的softmax熵进行教师-学生学习
            loss = -0.5 * (outputs_ema.softmax(1) * outputs.log_softmax(1)).sum(1) - 0.5 * (outputs.softmax(1) * outputs_ema.log_softmax(1)).sum(1)
            loss = loss.mean(0)

            # 应用Fisher正则化（如果启用）
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
            
            # 使用区分vida参数和普通参数的EMA更新教师模型
            self.model_ema = self.update_ema_variables(
                ema_model=self.model_ema,
                model=self._model,
                alpha_teacher=self.alpha_teacher,
                alpha_vida=self.alpha_vida
            )
            
            # 随机恢复部分参数以增强稳定性
            self.stochastic_restore()
            
        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
            "yhat": outputs_ema,
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
            if hasattr(self, 'tta_loss_computer') and self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    optimal_state["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )
            
    @property
    def name(self):
        return "ViDA"


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class Clip(torch.nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + "(min_val={0}, max_val={1})".format(
            self.min_val, self.max_val
        )


class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, "gamma")

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                Lambda(lambda img: F.adjust_brightness(img, brightness_factor))
            )

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: F.adjust_contrast(img, contrast_factor))
            )

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(lambda img: F.adjust_saturation(img, saturation_factor))
            )

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = (
                    torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                )
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = (
                    torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                )
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = (
                    torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                )
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(
                    1e-8, 1.0
                )  # to fix Nan values in gradients, which happens when applying gamma
                # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        format_string += ", gamma={0})".format(self.gamma)
        return format_string