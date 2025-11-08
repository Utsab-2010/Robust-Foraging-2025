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
        
        # SIMPLIFIED TRAINABLE WEIGHTING LAYER
        # Takes raw input image and directly computes 4 weights for processing paths
        # Much simpler: just look at input to decide weighting strategy
        self.adaptive_weighter = nn.Sequential(
            # For small 86x155 image, directly process to get global weights
            nn.Conv2d(initial_channels, 8, kernel_size=5, stride=4, padding=2),  # ~86x155 -> ~22x39
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),   # ~22x39 -> ~11x20
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1),
            # Global average pooling + small MLP to get 4 weights
            # Pool to single values, then predict weights
        )
        
        # Small MLP to convert pooled features to weights
        self.weight_predictor = nn.Sequential(
            nn.Linear(4, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 4)  # Output 4 weights (will be softmaxed)
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
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, output_size)
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
        
        # SIMPLIFIED WEIGHTING: Use raw input to predict weights
        # Extract features from raw input for weight prediction
        weight_features = self.adaptive_weighter(visual_obs)  # [batch, 4, H', W']
        pooled_features = weight_features.mean(dim=[2, 3])    # [batch, 4] - global avg pool
        raw_weights = self.weight_predictor(pooled_features)  # [batch, 4] - predict weights
        weights = F.softmax(raw_weights, dim=1)               # [batch, 4] - normalize to sum=1
        weights = weights.unsqueeze(-1).unsqueeze(-1)         # [batch, 4, 1, 1] - for broadcasting
        
        # PROCESSING PIPELINE: Create 4 different representations
        
        
        # 2. Contrast-enhanced image
        contrast_enhanced = self.adjust_contrast(visual_obs, contrast=5)
        
        # 3. Sobel Y edge detection
        sobel_edges = self.apply_sobel_y(visual_obs)
        
        # 4. Morphological closing on Sobel edges
        morph_closed = self.morphological_closing(sobel_edges)
        
        # Stack and apply learned weights
        stacked_features = torch.cat([original, contrast_enhanced, sobel_edges, morph_closed], dim=1)
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
