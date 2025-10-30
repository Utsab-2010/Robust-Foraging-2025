import torch
import torch.nn as nn
import torch.nn.functional as F
from mlagents.torch_utils.globals import exporting_to_onnx

class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.initial_channels = initial_channels
        
        # Calculate output dimensions like neurips.py
        conv_1_hw = self.conv_output_shape((height, width), 4, 2)
        conv_2_hw = self.conv_output_shape(conv_1_hw, 3, 2)
        conv_3_hw = self.conv_output_shape(conv_2_hw, 3, 1)
        self.final_flat = conv_3_hw[0] * conv_3_hw[1] * 32
        
        # Simple edge detection (fixed Sobel, no learnable params)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
        # Simple conv layers like neurips.py but with edge features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(initial_channels + 2, 32, [4, 4], [2, 2]),  # +2 for edge channels
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, [3, 3], [2, 2]),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, [3, 3], [1, 1]),
            nn.LeakyReLU(),
        )
        
        # Simple dense layer
        self.dense = nn.Sequential(
            nn.Linear(self.final_flat, self.h_size),
            nn.LeakyReLU(),
        )
    
    def conv_output_shape(self, input_shape, kernel_size, stride):
        """Calculate output shape after convolution"""
        h, w = input_shape
        h_out = (h - kernel_size) // stride + 1
        w_out = (w - kernel_size) // stride + 1
        return (h_out, w_out)
    
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])
        
        # Simple edge detection (much faster than learnable)
        edges_x = F.conv2d(visual_obs.mean(dim=1, keepdim=True), self.sobel_x, padding=1)
        edges_y = F.conv2d(visual_obs.mean(dim=1, keepdim=True), self.sobel_y, padding=1)
        edges = torch.sqrt(edges_x.pow(2) + edges_y.pow(2) + 1e-6)
        
        # Combine original and edge features
        x = torch.cat([visual_obs, edges_x, edges_y], dim=1)
        
        # Simple forward pass
        hidden = self.conv_layers(x)
        hidden = hidden.reshape([-1, self.final_flat])
        return self.dense(hidden)

