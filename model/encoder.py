import torch
import torch.nn as nn
import math

class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', localhub=True):
        super(DPT_DINOv2, self).__init__()
        
        assert encoder in ['vits', 'vitb', 'vitl', 'vitg']
        
        if localhub:
            self.pretrained = torch.hub.load(
                r"C:\filemohinh\modelmoi\torchhub\facebookresearch_dinov2_main",
                'dinov2_{:}14'.format(encoder),
                source='local',
                pretrained=False
            )
        else:
            self.pretrained = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_{:}14'.format(encoder)
            )
        
        self.embed_dim = self.pretrained.embed_dim  # Kích thước embedding
        
        # Các layer convolution để điều chỉnh số kênh
        self.conv1 = nn.Conv2d(self.embed_dim, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(self.embed_dim, 256, kernel_size=3, padding=1, stride=2)  # Giảm kích thước không gian xuống 8x8
        self.conv3 = nn.Conv2d(self.embed_dim, 256, kernel_size=3, padding=1, stride=2)  # Giảm kích thước không gian xuống 4x4
        self.conv4 = nn.Conv2d(self.embed_dim, 256, kernel_size=3, padding=1, stride=2)  # Giảm kích thước không gian xuống 2x2
        self.pool1 = nn.Identity()  # Giữ nguyên cho x1
        self.pool2 = nn.MaxPool2d(2)  # Giảm kích thước cho x2 xuống 8x8
        self.pool3 = nn.MaxPool2d(2)  # Giảm kích thước cho x3 xuống 4x4
        self.pool4 = nn.MaxPool2d(2)  # Giảm kích thước cho x4 xuống 2x2
        
    def forward(self, x):
        features = self.pretrained.get_intermediate_layers(x, n=4, return_class_token=False)
        outputs = []
        for i, feature in enumerate(features):
            batch_size, num_patches, embed_dim = feature.shape
            patch_dim = int(math.sqrt(num_patches))
            patch_h = patch_w = patch_dim
            x_i = feature.reshape(batch_size, patch_h, patch_w, embed_dim).permute(0, 3, 1, 2)
            conv_layer = getattr(self, f'conv{i+1}')
            x_i = conv_layer(x_i)
            pool_layer = getattr(self, f'pool{i+1}')
            x_i = pool_layer(x_i)
            outputs.append(x_i)
        return outputs  # Trả về danh sách [x1, x2, x3, x4]
