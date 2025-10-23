import torch
import torch.nn as nn
import torch.nn.functional as F
from mlagents.torch_utils import Initialization
from mlagents.trainers.torch.layers import linear_layer
from mlagents.trainers.torch import exporting_to_onnx



class RADaugment(nn.Module):
    """Data augmentation for robust visual RL (grayscale navigation)."""
    def __init__(self, height=155, width=86):
        super().__init__()
        self.height = height
        self.width = width
        
    def forward(self, x):
       
        if not self.training:
            return x
        
        batch_size = x.size(0)
        
        # Random crop and resize
        crop_h = int(self.height * 0.95)
        crop_w = int(self.width * 0.95)
        
        top = torch.randint(0, self.height - crop_h + 1, (batch_size,))
        left = torch.randint(0, self.width - crop_w + 1, (batch_size,))
        
        cropped = []
        for i in range(batch_size):
            crop = x[i:i+1, :, top[i]:top[i]+crop_h, left[i]:left[i]+crop_w]
            crop = F.interpolate(crop, size=(self.height, self.width), 
                                mode='bilinear', align_corners=False)
            cropped.append(crop)
        x = torch.cat(cropped, dim=0)
        
        # Random brightness
        if torch.rand(1) > 0.5:
            brightness = 0.7 + torch.rand(1).to(x.device) * 0.6
            x = torch.clamp(x * brightness, 0, 1)
        
        # Random contrast
        if torch.rand(1) > 0.5:
            mean = x.mean(dim=[2, 3], keepdim=True)
            contrast = 0.8 + torch.rand(1).to(x.device) * 0.4
            x = torch.clamp((x - mean) * contrast + mean, 0, 1)
        
        # Random noise (fog variation)
        if torch.rand(1) > 0.7:
            noise = torch.randn_like(x) * 0.02
            x = torch.clamp(x + noise, 0, 1)
        
        return x


#  Main Encoder
class VisualEncoder(nn.Module):

    
    
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        
        # Data augmentation (training only)
        self.augment = RADaugment(height=height, width=width)
        
        # Visual feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(initial_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.layer1 = self._make_residual_block(32, 64, stride=2)
        self.layer2 = self._make_residual_block(64, 128, stride=2)
        self.layer3 = self._make_residual_block(128, 256, stride=2)
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(256, reduction=16)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
    
        self.target_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 3)  # [visible, left, right]
        )
        
        # Policy feature head
        final_h = height // 16
        final_w = width // 16
        flat_size = 256 * final_h * final_w
        
        self.policy_head = nn.Sequential(
            linear_layer(
                flat_size + 3,
                512,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,
            ),
            nn.ReLU(inplace=True),
            linear_layer(
                512,
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.0,
            ),
        )
    
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        return ResidualBlock(in_channels, out_channels, stride, downsample)
    
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # Convert NHWC → NCHW
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute(0, 3, 1, 2)
        
        # Normalize
        if visual_obs.max() > 1.0:
            visual_obs = visual_obs / 255.0
        
        # Augment (training only)
        if self.training:
            visual_obs = self.augment(visual_obs)
        
        # Feature extraction
        x = self.stem(visual_obs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        
        # Target visibility
        target_signals = torch.sigmoid(self.target_detector(x))
        
        # Combine features
        spatial_features = torch.flatten(x, start_dim=1)
        combined = torch.cat([spatial_features, target_signals], dim=1)
        
        # Policy features
        output = self.policy_head(combined)
        
        return output




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
    
    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                             padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        combined = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention