import cv2

class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.initial_channels = initial_channels
        
        # CONTRAST ADJUSTMENT BLOCK
        # No parameters needed - contrast adjustment is applied dynamically
        
        # SOBEL Y FILTER BLOCK (Non-trainable)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0).repeat(1, initial_channels, 1, 1))
        
        # MORPHOLOGICAL CLOSING BLOCK (Non-trainable)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.morph_kernel_size = kernel.shape
        
        # TRAINABLE WEIGHTING LAYER
        # This learns to weight 4 different processing paths based on input content
        # Input: original + contrast + sobel_y + morphological_closing = 4 channels
        self.adaptive_weighter = nn.Sequential(
            # Reduce spatial dimensions using regular pooling (Barracuda compatible)
            # Input: 86x155 -> 21x38 (stride 4) -> 10x19 (stride 2) -> 5x9 (stride 2)
            nn.Conv2d(4, 16, kernel_size=4, stride=4, padding=1),  # 86x155 -> ~21x38
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),   # ~21x38 -> ~10x19
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),    # ~10x19 -> ~5x9
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1),
            # Global average pooling using tensor.mean() (Barracuda compatible)
            # Will be applied in forward() as: tensor.mean(dim=[2, 3])
        )
        
        # MAIN FEATURE EXTRACTION (from trans_v7_3)
        self.section_stem = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, stride=1, padding=1),  # 4 weighted channels input
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1)
        )
        
        # MAIN CONVOLUTION PIPELINE (from trans_v7_3)
        self.section_conv = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        
        # FINAL PROJECTION (from trans_v7_3)
        self.final_proj = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_size)
        )
    
    def adjust_contrast(self, image, contrast=1.5):
        """
        Contrast adjustment block (non-trainable)
        Increases contrast to enhance edge visibility
        """
        return torch.clamp((image - 0.5) * contrast + 0.5, 0.0, 1.0)
    
    def apply_sobel_y(self, image):
        """
        Sobel Y filter block (non-trainable)
        Detects horizontal edges (good for oval detection)
        """
        edges = F.conv2d(image, self.sobel_y, padding=1, groups=self.initial_channels)
        edge_magnitude = torch.abs(edges)
        
        # Average across input channels to get single channel output
        if edge_magnitude.shape[1] > 1:
            edge_magnitude = edge_magnitude.mean(dim=1, keepdim=True)
        
        return edge_magnitude
    
    def morphological_closing(self, image):
        """
        Morphological closing block (non-trainable)
        Fills gaps in detected edges - good for connecting oval boundaries
        """
        # Dilation then erosion
        dilated = F.max_pool2d(image, kernel_size=self.morph_kernel_size, 
                             stride=1, padding=self.morph_kernel_size[0]//2)
        closed = -F.max_pool2d(-dilated, kernel_size=self.morph_kernel_size,
                             stride=1, padding=self.morph_kernel_size[0]//2)
        return closed
    
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # Input format conversion (Barracuda compatible)
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute(0, 3, 1, 2)
        
        # PROCESSING PIPELINE: Create 4 different representations
        
        
        # 2. Contrast-enhanced image
        contrast_enhanced = self.adjust_contrast(visual_obs, contrast=1.5)
        
        # 3. Sobel Y edge detection
        sobel_edges = self.apply_sobel_y(visual_obs)
        
        # 4. Morphological closing on Sobel edges
        morph_closed = self.morphological_closing(sobel_edges)
        
        # ADAPTIVE WEIGHTING: Stack all 4 channels for weight calculation
        stacked_features = torch.cat([visual_obs, contrast_enhanced, sobel_edges, morph_closed], dim=1)
        
        # Learn adaptive weights based on input content
        weight_features = self.adaptive_weighter(stacked_features)  # Shape: [batch, 4, H', W']
        
        # Global average pooling to get weights (Barracuda compatible)
        weights = weight_features.mean(dim=[2, 3])  # Shape: [batch, 4]
        
        # Apply softmax to ensure weights sum to 1
        weights = F.softmax(weights, dim=1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch, 4, 1, 1] for broadcasting
        
        # Apply learned weights to each channel
        weighted_features = stacked_features * weights
        
        # MAIN FEATURE EXTRACTION
        features = self.section_stem(weighted_features)
        
        # MAIN CONVOLUTION PIPELINE
        conv_features = self.section_conv(features)
        
        # GLOBAL POOLING (Barracuda compatible)
        pooled = conv_features.mean(dim=[2, 3], keepdim=False)  # [batch, 64]
        
        # FINAL PROJECTION
        output = self.final_proj(pooled)
        
        return output
