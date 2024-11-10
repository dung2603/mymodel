import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder import DPT_DINOv2
from model.unethead import UNetHead
from model.dpt_head import DPTHead

class CombinedModel(nn.Module):
    def __init__(self, n_classes, use_bn=False):
        super(CombinedModel, self).__init__()
        self.encoder = DPT_DINOv2()
        self.unet_head = UNetHead(n_classes, use_bn=use_bn)
        in_channels = [n_classes, 1, 256, 256]  # Điều chỉnh theo số kênh đầu ra từ UNet Head và encoder
        self.dpt_head = DPTHead(in_channels, features=256, use_bn=use_bn)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        seg_output, depth_output = self.unet_head(x1, x2, x3, x4)

        # Đảm bảo kích thước của depth_output phù hợp với seg_output
        depth_output = F.interpolate(depth_output, size=seg_output.shape[2:], mode='bilinear', align_corners=True)

        # Tạo danh sách các đặc trưng cho DPTHead
        features = [seg_output, depth_output, x3, x4]

        final_depth = self.dpt_head(features)

        # Upsample final_depth để khớp với kích thước của nhãn mục tiêu
        final_depth = F.interpolate(final_depth, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        print(f"Final depth output shape: {final_depth.shape}")  # Kỳ vọng: [batch_size, 1, 224, 224]

        return final_depth, seg_output
