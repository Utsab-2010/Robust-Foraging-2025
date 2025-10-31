import torch
import torch.nn as nn
import torch.nn.functional as F
from mlagents.torch_utils.globals import exporting_to_onnx

class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.initial_channels = initial_channels
        self.num_sections = 5
        self.section_width = width // self.num_sections
        
        # Main feature extraction pipeline - processes full image first
        self.main_conv = nn.Sequential(
            nn.Conv2d(initial_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),  # Down to 78x44
            nn.LeakyReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # Down to 39x22
            nn.LeakyReLU()
        )
        
        # Edge detection on full image - more effective than per-section
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # Sectional attention - learns to focus on important regions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),  # Channel reduction
            nn.LeakyReLU(),
            nn.Conv2d(32, self.num_sections, kernel_size=1),  # One channel per section
            nn.Softmax(dim=1)  # Attention weights across sections
        )
        
        # Global feature extraction after attention
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # Down to ~20x11
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Fixed spatial size
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size)
        )
        
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])
        
        # Extract main features from full image
        features = self.main_conv(visual_obs)  # [B, 64, 39, 22]
        
        # Apply edge detection on original image
        gray_img = visual_obs.mean(dim=1, keepdim=True)  # Convert to grayscale
        edges_x = F.conv2d(gray_img, self.sobel_x, padding=1)
        edges_y = F.conv2d(gray_img, self.sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edges_x.pow(2) + edges_y.pow(2) + 1e-6)
        
        # Resize edge map to match feature size
        edge_features = F.interpolate(edge_magnitude, size=features.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine features with edge information
        enhanced_features = features + 0.3 * edge_features  # Weighted edge enhancement
        
        # Generate spatial attention map
        attention_map = self.spatial_attention(enhanced_features)  # [B, 5, H, W]
        
        # Apply sectional attention - weight features by attention map
        weighted_features = enhanced_features.unsqueeze(1) * attention_map.unsqueeze(2)  # Broadcast and multiply
        attended_features = weighted_features.sum(dim=1)  # Sum across sections
        
        # Final processing
        final_features = self.final_conv(attended_features)
        output = self.final_proj(final_features)
        
        return output