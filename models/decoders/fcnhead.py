import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F


class FCNHead(nn.Module):
    def __init__(self, in_channels=384, channels=None, kernel_size=3, dilation=1,
                 num_classes=40, norm_layer=nn.BatchNorm2d):
        super(FCNHead, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.channels = channels or in_channels // 4

        conv_padding = (kernel_size // 2) * dilation
        self.conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.channels, kernel_size, padding=conv_padding),
                norm_layer(self.channels),
                nn.ReLU(inplace=True)
                )

        self.classifier = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def forward(self, x):
        output = self.conv(x)
        output = self.classifier(output)
        return output