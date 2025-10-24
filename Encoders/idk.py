import torch
import torch.nn as nn
import torch.nn.functional as F
from mlagents.torch_utils import Initialization
from mlagents.trainers.torch.layers import linear_layer
from mlagents.trainers.torch import exporting_to_onnx
import kornia.augmentation as K


class RandomShiftsAug(nn.Module):
    """Random shift augmentation (core of RAD)"""
    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad,
                                device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2),
                              device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class RADEncoder(nn.Module):
    """
    RAD-enhanced encoder for Mouse vs AI Competition
    Based on the RAD paper + competition requirements
    """
    
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.height = height
        self.width = width
        
        # Data augmentation (applied during training only)
        self.aug = RandomShiftsAug(pad=4)
        
        # Alternative: Use Kornia for more augmentations
        self.aug_pipeline = nn.Sequential(
            K.RandomCrop((height, width), padding=4, pad_if_needed=True),
            K.RandomGrayscale(p=0.2),  # Since input is grayscale, simulate noise
            K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0, p=0.3),
        )
        
        # CNN architecture inspired by the PixelEncoder from RAD
        # Simplified for 155x86 input
        num_filters = 32
        
        self.convs = nn.ModuleList([
            nn.Conv2d(initial_channels, num_filters, 3, stride=2),  # 155x86 -> 77x43
            nn.Conv2d(num_filters, num_filters, 3, stride=2),        # 77x43 -> 38x21
            nn.Conv2d(num_filters, num_filters, 3, stride=2),        # 38x21 -> 18x10
            nn.Conv2d(num_filters, num_filters, 3, stride=1),        # 18x10 -> 16x8
        ])
        
        # Calculate flattened size
        # After convolutions: approximately 16x8 with 32 filters
        self.flatten_size = num_filters * 16 * 8
        
        # Feature projection with LayerNorm (more stable than BatchNorm in RL)
        self.fc = nn.Linear(self.flatten_size, output_size)
        self.ln = nn.LayerNorm(output_size)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, visual_obs: torch.Tensor, augment: bool = True) -> torch.Tensor:
        # Convert NHWC → NCHW if not exporting to ONNX
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute(0, 3, 1, 2)
        
        # Normalize to [0, 1]
        if visual_obs.max() > 1.0:
            visual_obs = visual_obs / 255.0
        
        
        if augment and self.training and not exporting_to_onnx.is_exporting():
            visual_obs = self.aug(visual_obs)
            # Alternative: use kornia pipeline
            # visual_obs = self.aug_pipeline(visual_obs)
        
        # Convolutional forward pass
        x = visual_obs
        for conv in self.convs:
            x = F.relu(conv(x))
        
        # Flatten and project
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.ln(x)
        x = torch.tanh(x)
        
        return x


class ImprovedRADEncoder(nn.Module):

    
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        
        # Augmentation
        self.aug = RandomShiftsAug(pad=4)
        
        # Deeper CNN 
        self.conv1 = nn.Conv2d(initial_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        # Approximate output size: 155x86 -> 78x43 -> 39x22 -> 20x11
        self.flatten_size = 128 * 20 * 11
        
        # Two-layer MLP head
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, output_size)
        self.ln = nn.LayerNorm(output_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, visual_obs: torch.Tensor, augment: bool = True) -> torch.Tensor:
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute(0, 3, 1, 2)
        
        if visual_obs.max() > 1.0:
            visual_obs = visual_obs / 255.0
        
        # Apply augmentation during training
        if augment and self.training and not exporting_to_onnx.is_exporting():
            visual_obs = self.aug(visual_obs)
        
        # Conv blocks with residual-style processing
        x = F.relu(self.conv1(visual_obs))
        x = F.relu(self.conv2(x) + x) 
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x) + x)  
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x) + x) 
        
        # MLP head
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.ln(x)
        x = torch.tanh(x)
        
        return x