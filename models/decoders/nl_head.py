import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

# from mmcv.cnn import NonLocal2d
from collections import defaultdict
import functools
import numpy as np

from mmseg.ops import resize

# from ..builder import HEADS
from .decode_head import BaseDecodeHead
# from .aspp_head import ASPPHead, ASPPModule

from abc import ABCMeta
from typing import Dict, Optional

import torch
import torch.nn as nn

from mmcv.cnn.utils import constant_init, normal_init
# from .conv_module import ConvModule
# from .registry import PLUGIN_LAYERS

# def bmm_flops_count(input, output):
#     print('=============================================================')
#     input1, input2, *remain = input[0]
#     print(np.prod(input1.shape[1:]) * input2.shape[-1])
#     return np.prod(input1.shape[1:]) * input2.shape[-1]

# method_mapping = {
#     'bmm': bmm_flops_count,
# }
# flops_cnt = defaultdict(int)


# def flops_count_wrapper(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         output = func(*args, **kwargs)
#         name = func.__name__
#         flops_cnt[name] += method_mapping[name](input=(args, kwargs),
#                                                 output=output)
#         return output
#     return wrapper


# decorate wrapper
# bmm = flops_count_wrapper(torch.bmm)


class _NonLocalNd(nn.Module, metaclass=ABCMeta):
    """Basic Non-local module.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971
    Code reference: https://github.com/AlexHex7/Non-local_pytorch

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
            Default: True.
        conv_cfg (None | dict): The config dict for convolution layers.
            If not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (None | dict): The config dict for normalization layers.
            Default: None. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
    """

    def __init__(
        self,
        in_channels: int,
        reduction: int = 2,
        use_scale: bool = True,
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        mode: str = "embedded_gaussian",
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode

        if mode not in ["gaussian", "embedded_gaussian", "dot_product", "concatenation"]:
            raise ValueError(
                "Mode should be in 'gaussian', 'concatenation', "
                f"'embedded_gaussian' or 'dot_product', but got "
                f"{mode} instead."
            )

        # g, theta, phi are defaulted as `nn.ConvNd`.
        # Here we use ConvModule for potential usage.
        self.g = ConvModule(self.in_channels, self.inter_channels, kernel_size=1, conv_cfg=conv_cfg, act_cfg=None)  # type: ignore
        self.conv_out = ConvModule(
            self.inter_channels, self.in_channels, kernel_size=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None
        )

        if self.mode != "gaussian":
            self.theta = ConvModule(
                self.in_channels, self.inter_channels, kernel_size=1, conv_cfg=conv_cfg, act_cfg=None
            )
            self.phi = ConvModule(
                self.in_channels, self.inter_channels, kernel_size=1, conv_cfg=conv_cfg, act_cfg=None
            )

        if self.mode == "concatenation":
            self.concat_project = ConvModule(
                self.inter_channels * 2, 1, kernel_size=1, stride=1, padding=0, bias=False, act_cfg=dict(type="ReLU")
            )

        self.init_weights(**kwargs)

    def init_weights(self, std: float = 0.01, zeros_init: bool = True) -> None:
        if self.mode != "gaussian":
            for m in [self.g, self.theta, self.phi]:
                normal_init(m.conv, std=std)
        else:
            normal_init(self.g.conv, std=std)
        if zeros_init:
            if self.conv_out.norm_cfg is None:
                constant_init(self.conv_out.conv, 0)
            else:
                constant_init(self.conv_out.norm, 0)
        else:
            if self.conv_out.norm_cfg is None:
                normal_init(self.conv_out.conv, std=std)
            else:
                normal_init(self.conv_out.norm, std=std)

    def gaussian(self, theta_x: torch.Tensor, phi_x: torch.Tensor) -> torch.Tensor:
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x: torch.Tensor, phi_x: torch.Tensor) -> torch.Tensor:
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        # print(theta_x.shape,phi_x.shape)
        pairwise_weight = torch.bmm(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x: torch.Tensor, phi_x: torch.Tensor) -> torch.Tensor:
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x: torch.Tensor, phi_x: torch.Tensor) -> torch.Tensor:
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.size()
        pairwise_weight = pairwise_weight.view(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]

        return pairwise_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume `reduction = 1`, then `inter_channels = C`
        # or `inter_channels = C` when `mode="gaussian"`

        # NonLocal1d x: [N, C, H]
        # NonLocal2d x: [N, C, H, W]
        # NonLocal3d x: [N, C, T, H, W]

        shape = x.size()[2:]
        identity = x
        n = x.size(0)

        # NonLocal1d g_x: [N, H, C]
        # NonLocal2d g_x: [N, HxW, C]
        # NonLocal3d g_x: [N, TxHxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # x = F.avg_pool2d(x, kernel_size=(4,4))##########
        # x = nn.functional.interpolate(x, size=(16,16))
        # assert x.size()[2:]==(16,16)

        # NonLocal1d theta_x: [N, H, C], phi_x: [N, C, H]
        # NonLocal2d theta_x: [N, HxW, C], phi_x: [N, C, HxW]
        # NonLocal3d theta_x: [N, TxHxW, C], phi_x: [N, C, TxHxW]
        if self.mode == "gaussian":
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(x).view(n, self.in_channels, -1)
            else:
                phi_x = x.view(n, self.in_channels, -1)
        elif self.mode == "concatenation":
            theta_x = self.theta(x).view(n, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # NonLocal1d pairwise_weight: [N, H, H]
        # NonLocal2d pairwise_weight: [N, HxW, HxW]
        # NonLocal3d pairwise_weight: [N, TxHxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)
        # print(pairwise_weight.shape,'=================')
        # pairwise_weight = pairwise_weight.view(n,256,16,16)#########
        # pairwise_weight = nn.functional.interpolate(pairwise_weight, size=shape)
        # pairwise_weight = pairwise_weight.view(n,256,shape[0]*shape[1]).view(n,shape[0]*shape[1],16,16)#########
        # pairwise_weight = nn.functional.interpolate(pairwise_weight, size=shape)
        # pairwise_weight = pairwise_weight.view(n,shape[0]*shape[1],shape[0]*shape[1])
        # print(pairwise_weight.shape,'=================')

        # NonLocal1d y: [N, H, C]
        # NonLocal2d y: [N, HxW, C]
        # NonLocal3d y: [N, TxHxW, C]
        y = torch.bmm(pairwise_weight, g_x)
        # NonLocal1d y: [N, C, H]
        # NonLocal2d y: [N, C, H, W]
        # NonLocal3d y: [N, C, T, H, W]
        y = y.permute(0, 2, 1).contiguous().reshape(n, self.inter_channels, *shape)

        output = identity + self.conv_out(y)

        return output


class NonLocal1d(_NonLocalNd):
    """1D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv1d').
    """

    def __init__(self, in_channels: int, sub_sample: bool = False, conv_cfg: Dict = dict(type="Conv1d"), **kwargs):
        super().__init__(in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != "gaussian":
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


# @PLUGIN_LAYERS.register_module()
class NonLocal2d(_NonLocalNd):
    """2D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv2d').
    """

    _abbr_ = "nonlocal_block"

    def __init__(self, in_channels: int, sub_sample: bool = False, conv_cfg: Dict = dict(type="Conv2d"), **kwargs):
        super().__init__(in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != "gaussian":
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class NonLocal3d(_NonLocalNd):
    """3D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv3d').
    """

    def __init__(self, in_channels: int, sub_sample: bool = False, conv_cfg: Dict = dict(type="Conv3d"), **kwargs):
        super().__init__(in_channels, conv_cfg=conv_cfg, **kwargs)
        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            self.g = nn.Sequential(self.g, max_pool_layer)
            if self.mode != "gaussian":
                self.phi = nn.Sequential(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class NLHead(BaseDecodeHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    TODO:
        Add other MD models (Ham).
    """

    def __init__(self, **kwargs):
        super(NLHead, self).__init__(input_transform="multiple_select", **kwargs)
        # self.channels = channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.reduction = 2
        self.use_scale = True
        self.mode = "embedded_gaussian"
        self.nl_block = NonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode,
        )

        self.align = ConvModule(
            self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [
            resize(level, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
            for level in inputs
        ]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.nl_block(x)

        output = self.align(x)
        output = self.cls_seg(output)
        return output
