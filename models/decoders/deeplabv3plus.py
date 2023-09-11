import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=[256, 512, 1024, 2048], num_classes=40, norm_layer=nn.BatchNorm2d):
        super(DeepLabV3Plus, self).__init__()
        self.num_classes = num_classes
        
        self.aspp = ASPP(in_channels=in_channels[3], atrous_rates=[12, 24, 36], norm_layer=norm_layer)
        self.low_level = nn.Sequential(
                            nn.Conv2d(in_channels[0], 48, kernel_size=3, stride=1, padding=1),
                            norm_layer(48),
                            nn.ReLU(inplace=True)
                            )
        self.block = nn.Sequential(
                        nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
                        norm_layer(256),
                        nn.ReLU(inplace=True),
                        #nn.Dropout(0.5),
                        #nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        #norm_layer(256),
                        #nn.ReLU(inplace=True),
                        nn.Dropout(0.1),
                        nn.Conv2d(256, num_classes, 1))

    def forward(self, inputs):
        c1, _, _, c4 = inputs
        c1 = self.low_level(c1)
        c4 = self.aspp(c4)
        c4 = F.interpolate(c4, c1.size()[2:], mode='bilinear', align_corners=True)
        output = self.block(torch.cat([c4, c1], dim=1))
        return output


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer):
        super(ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer):
        super(ASPP, self).__init__()
        out_channels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer=norm_layer)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x