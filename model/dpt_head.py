import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import FeatureFusionBlock, _make_scratch

class DPTHead(nn.Module):
    def __init__(self, in_channels, features=256, use_bn=False):
        super(DPTHead, self).__init__()

        # Tạo scratch layers
        self.scratch = _make_scratch(in_channels, [features, features, features, features])

        # Tạo các FeatureFusionBlock
        self.scratch.refinenet4 = FeatureFusionBlock(features, use_bn=use_bn)
        self.scratch.refinenet3 = FeatureFusionBlock(features, use_bn=use_bn)
        self.scratch.refinenet2 = FeatureFusionBlock(features, use_bn=use_bn)
        self.scratch.refinenet1 = FeatureFusionBlock(features, use_bn=use_bn)

        # Layer đầu ra
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        layer_1 = self.scratch.layer1_rn(features[0])
        layer_2 = self.scratch.layer2_rn(features[1])
        layer_3 = self.scratch.layer3_rn(features[2])
        layer_4 = self.scratch.layer4_rn(features[3])

        path_4 = self.scratch.refinenet4(layer_4)
        path_3 = self.scratch.refinenet3(path_4, layer_3)
        path_2 = self.scratch.refinenet2(path_3, layer_2)
        path_1 = self.scratch.refinenet1(path_2, layer_1)

        out = self.scratch.output_conv(path_1)
        return out