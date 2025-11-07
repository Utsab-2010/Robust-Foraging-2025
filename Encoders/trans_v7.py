class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.initial_channels = initial_channels
        
        # Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(initial_channels, 24, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(6, 24),
            nn.GELU()
        )
        
        # Edge detection parameters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Create Sobel filters (12 channels for good edge detection)
        self.sobel_x = nn.Parameter(sobel_x.repeat(12, initial_channels, 1, 1), requires_grad=True)
        self.sobel_y = nn.Parameter(sobel_y.repeat(12, initial_channels, 1, 1), requires_grad=True)
        
        # Main feature extraction with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(24 + 12, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(6, 48),
            nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(6, 48),
            nn.GELU()
        )
        
        # Intensity-sensitive feature enhancement with spatial preservation
        self.intensity_branch = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final feature extraction
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        
        # ONNX-compatible pooling to handle variable input sizes
        self.global_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),  # Further reduce spatial size
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Project to output size with attention to spatial information
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 256),  # Updated for global pooling output
            nn.GELU(),
            nn.Linear(256, output_size)
        )
        
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])
            
        # Initial features
        x = self.stem(visual_obs)
        
        # Edge detection - apply for each input channel
        edges_x = F.conv2d(visual_obs, self.sobel_x, padding=1, groups=self.initial_channels)
        edges_y = F.conv2d(visual_obs, self.sobel_y, padding=1, groups=self.initial_channels)
        edges = torch.sqrt(edges_x.pow(2) + edges_y.pow(2) + 1e-6)
        x = torch.cat([x, edges], dim=1)
        
        # Main feature extraction with residual connection
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = x2 + x1  # Residual connection
        
        # Intensity-sensitive feature enhancement
        attention = self.intensity_branch(x2)
        x2 = x2 * attention
        
        # Final convolution and pooling
        x3 = self.conv3(x2)
        x3 = self.global_pool(x3)
        
        # Project to output size
        out = self.proj(x3)
        
        return out

