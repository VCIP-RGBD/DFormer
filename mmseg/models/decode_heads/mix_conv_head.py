import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .aspp_head import ASPPHead, ASPPModule

class MixConv(nn.Module):
    def __init__(self, dim, kernels=[1, 7, 11, 15]):
        super().__init__()

        self.branches = len(kernels)
        self.fc1 = nn.Conv2d(dim, dim * self.branches, 1)
        self.conv1 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 15, padding=7, groups=dim)
        self.fc2 = nn.Conv2d(dim * self.branches, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        xs = torch.chunk(x, 4, dim=1)
        x1 = xs[0]
        x2 = self.conv1(xs[1])
        x3 = self.conv2(xs[2])
        x4 = self.conv3(xs[3])
        x = self.act(torch.cat([x1, x2, x3, x4], dim=1))
        x = self.fc2(x)

        return x

@HEADS.register_module()
class MixConvHead(BaseDecodeHead):
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
        super(MixConvHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # self.channels = channels

        self.squeeze = ConvModule(
            sum(self.in_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv = MixConv(dim=self.channels)

        self.align = ConvModule(
            self.channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        inputs = [resize(
            level,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
        ) for level in inputs]

        inputs = torch.cat(inputs, dim=1)
        x = self.squeeze(inputs)

        x = self.conv(x)

        output = self.align(x)
        output = self.cls_seg(output)
        return output