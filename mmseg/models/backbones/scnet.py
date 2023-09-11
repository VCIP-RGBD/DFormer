# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
                         load_state_dict)
from mmcv.utils import to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES



# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg
import math

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        # self.norm = nn.BatchNorm2d(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.fc2(x)

        return x

class MemoryModule(nn.Module):
    def __init__(self, dim, num_head=8, window=7, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.num_head = num_head
        self.window = window

        self.q = nn.Linear(dim, dim)
        self.a = nn.Linear(dim, dim)
        self.l = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        

        self.proj = nn.Linear(dim, dim)
        if window != 0:
            # self.t = nn.Linear(dim, window*window*num_head)
            self.conv_att = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
            self.kv = nn.Linear(dim, dim*2)
            self.m = nn.Parameter(torch.zeros(1, window, window, dim), requires_grad=True)
            self.proj = nn.Linear(dim * 2, dim)

        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")

    def forward(self, x):
        B, H, W, C = x.size()
        x = self.norm(x)

        q = self.q(x)        
        x = self.l(x).permute(0, 3, 1, 2)
        x = self.act(x)
            
        a = self.conv(x)
        a = a.permute(0, 2, 3, 1)
        a = self.a(a)

        if self.window != 0:
            b = self.conv_att(x)
            b = b.permute(0, 2, 3, 1)
            kv = self.kv(b)
            kv = kv.reshape(B, H*W, 2, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            # t = self.t(a).reshape(B, H*W, self.num_head, -1).permute(0, 2, 1, 3).softmax(dim=-1)

            m = self.m.reshape(1, -1, self.num_head, C // self.num_head).permute(0, 2, 1, 3).expand(B, -1, -1, -1)
            attn = (m * (C // self.num_head) ** -0.5) @ k.transpose(-2, -1) 
            attn = attn.softmax(dim=-1)
            attn = (attn @ v).reshape(B, self.num_head, self.window, self.window, C // self.num_head).permute(0, 1, 4, 2, 3).reshape(B, C, self.window, self.window)
            attn = F.interpolate(attn, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            # attn = (t @ attn).permute(0, 2, 1, 3).reshape(B, H, W, C)
        
        x = q * a

        if self.window != 0:
            x = torch.cat([x, attn], dim=3)
        x = self.proj(x)

        return x

class MemoryModulev2(nn.Module):
    def __init__(self, dim, num_head=8, window=7, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.num_head = num_head
        self.window = window

        self.q = nn.Linear(dim, dim)
        self.a = nn.Linear(dim, dim)
        self.l = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        

        self.proj = nn.Linear(dim, dim)
        if window != 0:
            self.t = nn.Linear(dim, window*window*num_head)
            self.conv_att = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
            self.kv = nn.Linear(dim, dim*2)
            self.m = nn.Parameter(torch.zeros(1, window, window, dim), requires_grad=True)
            self.proj = nn.Linear(dim * 2, dim)

        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")

    def forward(self, x):
        B, H, W, C = x.size()
        x = self.norm(x)

        q = self.q(x)        
        x = self.l(x).permute(0, 3, 1, 2)
        x = self.act(x)
            
        a = self.conv(x)
        a = a.permute(0, 2, 3, 1)
        a = self.a(a)

        if self.window != 0:
            b = self.conv_att(x)
            b = b.permute(0, 2, 3, 1)
            kv = self.kv(b)
            kv = kv.reshape(B, H*W, 2, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            t = self.t(a).reshape(B, H*W, self.num_head, -1).permute(0, 2, 1, 3).softmax(dim=-1)

            m = self.m.reshape(1, -1, self.num_head, C // self.num_head).permute(0, 2, 1, 3).expand(B, -1, -1, -1)
            attn = (m * (C // self.num_head) ** -0.5) @ k.transpose(-2, -1) 
            attn = attn.softmax(dim=-1)
            attn = (attn @ v).reshape(B, self.num_head, self.window*self.window, C // self.num_head)
            # attn = F.interpolate(attn, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            attn = (t @ attn).permute(0, 2, 1, 3).reshape(B, H, W, C)
        
        x = q * a

        if self.window != 0:
            x = torch.cat([x, attn], dim=3)
        x = self.proj(x)

        return x

class Block(nn.Module):
    def __init__(self, index, dim, num_head, norm_cfg=dict(type='SyncBN', requires_grad=True), mlp_ratio=4., window=7, dropout_layer=None):
        super().__init__()
        
        self.index = index
        layer_scale_init_value = 1e-6  
        self.attn = MemoryModulev2(dim, num_head, window=window, norm_cfg=norm_cfg)
        self.mlp = MLP(dim, mlp_ratio, norm_cfg=norm_cfg)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
                 
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.dropout_layer(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * self.attn(x))
        x = x + self.dropout_layer(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x))
        return x

@BACKBONES.register_module()
class SCNet(BaseModule):
    def __init__(self, in_channels=3, depths=(2, 2, 8, 2), dims=(32, 64, 128, 256), out_indices=(0, 1, 2, 3), windows=[7, 7, 7, 7], norm_cfg=dict(type='SyncBN', requires_grad=True),
                 mlp_ratios=[8, 8, 4, 4], num_heads=(2, 4, 10, 16), drop_path_rate=0.1, init_cfg=None):
        super().__init__()
        self.depths = depths
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
                nn.Conv2d(in_channels, dims[0] // 2, kernel_size=3, stride=2, padding=1),
                build_norm_layer(norm_cfg, dims[0] // 2)[1],
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
                build_norm_layer(norm_cfg, dims[0])[1],
        )

        self.downsample_layers.append(stem)
        for i in range(len(dims)-1):
            stride = 2
            downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, dims[i])[1],
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(index=cur+j, 
                        dim=dims[i], 
                        window=windows[i],
                        dropout_layer=dict(type='DropPath', drop_prob=dp_rates[cur + j]), 
                        num_head=num_heads[i], 
                        norm_cfg=norm_cfg,
                        mlp_ratio=mlp_ratios[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Add a norm layer for each output
        for i in out_indices:
            layer = LayerNorm(dims[i], eps=1e-6, data_format="channels_first")
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        # self.apply(self.init_weights)

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # load state_dict
            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = x.permute(0, 2, 3, 1)
            x = self.stages[i](x)            
            x = x.permute(0, 3, 1, 2)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                outs.append(out)

        return outs