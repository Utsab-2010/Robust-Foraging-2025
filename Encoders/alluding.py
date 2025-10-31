class NatureVisualEncoder(nn.Module):
    """
    Encoder optimized for detecting and reaching targets at screen extremes.
    
    Key features:
    - Wider receptive field at edges (better peripheral detection)
    - Explicit "target out of view" detector
    - Strong rotation bias when target at extreme positions
    - Multi-resolution feature extraction
    """
    
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.height = height
        self.width = width
        
        # === Multi-Scale Feature Extraction ===
        # Extract features at different scales to catch targets at any position
        
        # Standard path
        self.stem = nn.Sequential(
            nn.Conv2d(initial_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.layer1 = self._make_conv_block(32, 64, stride=2)
        self.layer2 = self._make_conv_block(64, 128, stride=2)
        self.layer3 = self._make_conv_block(128, 256, stride=2)
        
        # === Edge-Aware Feature Extraction ===
        # Parallel branch with larger kernels to capture edge targets
        self.edge_branch = nn.Sequential(
            nn.Conv2d(initial_channels, 16, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Combine standard and edge features
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256 + 128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # === Horizontal Position Detector (5 regions) ===
        # More granular: FAR_LEFT | LEFT | CENTER | RIGHT | FAR_RIGHT
        self.horizontal_detector = HorizontalFiveRegionPooling()
        
        self.position_head = nn.Sequential(
            nn.Linear(256 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 7)  # More outputs for extreme positions
        )
        # Outputs:
        # [0]: target_visible
        # [1]: target_far_left (needs aggressive right rotation)
        # [2]: target_left
        # [3]: target_center
        # [4]: target_right  
        # [5]: target_far_right (needs aggressive left rotation)
        # [6]: target_distance
        
        # === Spatial Coordinates with Edge Boost ===
        final_h = height // 16
        final_w = width // 16
        self.spatial_softmax = SpatialSoftmaxWithEdgeBoost(
            height=final_h, 
            width=final_w,
            edge_boost=2.0  # Boost attention to edges
        )
        
       
        flat_size = 256 * final_h * final_w
        
        self.policy_head = nn.Sequential(
            linear_layer(
                flat_size + 7 + 2,  # features + position + coords
                512,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            linear_layer(
                512,
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.0,
            ),
        )
    
    def _make_conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # Convert NHWC → NCHW
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute(0, 3, 1, 2)
        
        # Normalize
        if visual_obs.max() > 1.0:
            visual_obs = visual_obs / 255.0
        
        # === Dual-Path Feature Extraction ===
        # Standard path (good for centered targets)
        x_standard = self.stem(visual_obs)
        x_standard = self.layer1(x_standard)
        x_standard = self.layer2(x_standard)
        x_standard = self.layer3(x_standard)
        
        # Edge path (good for extreme positions)
        x_edge = self.edge_branch(visual_obs)
        
        # Fuse features
        x_combined = torch.cat([x_standard, x_edge], dim=1)
        x = self.feature_fusion(x_combined)
        
        # === Horizontal Position Detection ===
        # Detect which of 5 horizontal regions contains target
        region_features = self.horizontal_detector(x)
        position_logits = self.position_head(region_features)
        position_signals = torch.sigmoid(position_logits)
        
        # === Precise Coordinates ===
        target_coords = self.spatial_softmax(x)
        
        # === Combine for Policy ===
        spatial_features = torch.flatten(x, start_dim=1)
        combined = torch.cat([
            spatial_features,
            position_signals,
            target_coords
        ], dim=1)
        
        output = self.policy_head(combined)
        
        return output


class HorizontalFiveRegionPooling(nn.Module):
    """
    Splits image into 5 horizontal regions for fine-grained position detection:
    [FAR_LEFT | LEFT | CENTER | RIGHT | FAR_RIGHT]
    
    This helps detect targets at extreme positions.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C*5] - features from 5 horizontal regions
        """
        B, C, H, W = x.shape
        
        # Define region boundaries
        fifth = W // 5
        
        # Split into 5 regions
        regions = [
            x[:, :, :, 0:fifth],              # Far left (0-20%)
            x[:, :, :, fifth:2*fifth],        # Left (20-40%)
            x[:, :, :, 2*fifth:3*fifth],      # Center (40-60%)
            x[:, :, :, 3*fifth:4*fifth],      # Right (60-80%)
            x[:, :, :, 4*fifth:W],            # Far right (80-100%)
        ]
        
        # Pool each region
        pooled = []
        for region in regions:
            pool = F.adaptive_avg_pool2d(region, 1).flatten(1)
            pooled.append(pool)
        
        # Concatenate all regions
        return torch.cat(pooled, dim=1)


class SpatialSoftmaxWithEdgeBoost(nn.Module):
    """
    Spatial softmax that gives extra weight to edge regions.
    Helps detect targets at screen extremes.
    """
    def __init__(self, height, width, temperature=1.0, edge_boost=2.0):
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = temperature
        self.edge_boost = edge_boost
        
        # Create coordinate grid
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, width),
            torch.linspace(-1, 1, height),
            indexing='xy'
        )
        
        # Create edge weight map (higher at edges)
        # Center has weight 1.0, edges have weight edge_boost
        dist_from_center_x = torch.abs(pos_x)
        dist_from_center_y = torch.abs(pos_y)
        edge_weight = 1.0 + (edge_boost - 1.0) * torch.maximum(
            dist_from_center_x, dist_from_center_y
        )
        
        self.register_buffer('pos_x', pos_x.reshape(1, height * width))
        self.register_buffer('pos_y', pos_y.reshape(1, height * width))
        self.register_buffer('edge_weight', edge_weight.reshape(1, height * width))
    
    def forward(self, x):
        """
        Args:
            x: [B, H, W] - feature map
        Returns:
            coords: [B, 2] - (x, y) in [-1, 1]
        """
        batch_size = x.size(0)
        
        # Flatten
        x_flat = x.view(batch_size, -1)
        
        # Apply edge boost
        x_boosted = x_flat * self.edge_weight
        
        # Softmax
        softmax_attention = F.softmax(x_boosted / self.temperature, dim=1)
        
        # Expected coordinates
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        
        coords = torch.cat([expected_x, expected_y], dim=1)
        
        return coords
    
    
# import matplotlib.pyplot as plt
# import numpy as np

# def visualize_encoder_output(encoder, image_tensor):
#     """
#     Visualize what the encoder detects in an image.
#     """
#     encoder.eval()
#     with torch.no_grad():
#         # Forward until position signals
#         visual_obs = image_tensor.permute(0, 3, 1, 2)
#         if visual_obs.max() > 1.0:
#             visual_obs = visual_obs / 255.0

#         # Extract intermediate features
#         x_standard = encoder.stem(visual_obs)
#         x_standard = encoder.layer1(x_standard)
#         x_standard = encoder.layer2(x_standard)
#         x_standard = encoder.layer3(x_standard)
#         x_edge = encoder.edge_branch(visual_obs)
#         x_combined = torch.cat([x_standard, x_edge], dim=1)
#         x = encoder.feature_fusion(x_combined)
#         region_features = encoder.horizontal_detector(x)
#         position_logits = encoder.position_head(region_features)
#         position_signals = torch.sigmoid(position_logits)[0].cpu()

#     # === Visualization ===
#     import matplotlib.pyplot as plt
#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#     # Image
#     axes[0].imshow(image_tensor[0, :, :, 0].cpu(), cmap='gray')
#     axes[0].set_title('Input Image')

#     # Bar chart of positions
#     positions = ['Far Left', 'Left', 'Center', 'Right', 'Far Right']
#     bars = axes[1].bar(positions, position_signals[1:6])
#     axes[1].set_ylim([0, 1])
#     axes[1].set_title('Target Position Detection')
#     axes[1].set_ylabel('Confidence')

#     max_idx = torch.argmax(position_signals[1:6])
#     bars[max_idx].set_color('red')

#     plt.tight_layout()
#     plt.savefig('encoder_debug.png')
#     plt.show()
#     print(f"Target detected in region: {positions[max_idx]}")

        
# encoder = VisualEncoder(height=86, width=155, initial_channels=1, output_size=256)

# # Load image
# from PIL import Image
# import torchvision.transforms as T

# image_path = "C:\\Users\\maila\\OneDrive\\Desktop\\mouse_vs_ai_windows\\initial_images\\RandomTrain\\episode_00252_agent_0.png"
# img = Image.open(image_path).convert("L")

# transform = T.Compose([
#     T.Resize((84, 84)),
#     T.ToTensor()
# ])
# img_tensor = transform(img).permute(1, 2, 0).unsqueeze(0)

# # Visualize
# visualize_encoder_output(encoder, img_tensor)