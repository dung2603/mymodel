import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvUnit(nn.Module):
    def __init__(self, features, use_bn=False):
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        return out + x

class FeatureFusionBlock(nn.Module):
    def __init__(self, features, use_bn=False):
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features, use_bn=use_bn)
        self.resConfUnit2 = ResidualConvUnit(features, use_bn=use_bn)

    def forward(self, x, residual=None):
        if residual is not None:
            res = self.resConfUnit1(residual)
            # Kiểm tra và điều chỉnh kích thước của res
            if res.shape[2:] != x.shape[2:]:
                res = F.interpolate(res, size=x.shape[2:], mode='bilinear', align_corners=True)
            x = x + res
        x = self.resConfUnit2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

def _make_scratch(in_channels, out_channels):
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=3, stride=1, padding=1, bias=False)
    return scratch


_make_scratch