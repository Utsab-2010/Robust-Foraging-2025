class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.initial_channels = initial_channels
        self.num_sections = 5
        self.section_width = width // self.num_sections
        
        # Edge detection parameters - smaller since we process sections
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # Reduced edge channels for efficiency (4 channels per section)
        self.sobel_x = nn.Parameter(sobel_x.repeat(4, initial_channels, 1, 1), requires_grad=True)
        self.sobel_y = nn.Parameter(sobel_y.repeat(4, initial_channels, 1, 1), requires_grad=True)
        
        # Section-wise feature extraction - smaller network per section
        self.section_stem = nn.Sequential(
            nn.Conv2d(initial_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU()
        )
        
        # Compact convolutional pipeline for each section
        self.section_conv = nn.Sequential(
            nn.Conv2d(16 + 4, 24, kernel_size=3, stride=2, padding=1),  # 16 features + 4 edge features
            nn.GroupNorm(4, 24),
            nn.LeakyReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU()
        )
        
        # Calculate approximate output size per section after convolutions
        # For 156x88 input: section width ≈ 31, after 2 stride-2 convs ≈ 8x22
        section_height_after_conv = height // 4  # Approximate after 2 stride-2 convs
        section_width_after_conv = self.section_width // 4
        
        # Global pooling for each section to get fixed-size features - ONNX compatible
        self.section_pool = nn.AdaptiveAvgPool2d((2, 2))  # Fixed 2x2 output
        self.features_per_section = 32 * 4  # 32 channels * 2*2 = 128 features per section
        
        # print(f"Section width: {self.section_width}")
        # print(f"Features per section: {self.features_per_section}")
        # print(f"Total features: {self.features_per_section * self.num_sections}")
        
        # Cross-section attention to focus on relevant sections
        self.section_attention = nn.Sequential(
            nn.Linear(self.features_per_section * self.num_sections, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.num_sections),
            nn.Softmax(dim=1)
        )
        
        # Final projection combining all sections
        self.final_proj = nn.Sequential(
            nn.Linear(self.features_per_section * self.num_sections, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size)
        )
        
    def extract_section_features(self, section_input):
        """Extract features from a single section with edge detection"""
        # Initial feature extraction for this section
        x = self.section_stem(section_input)
        
        # Edge detection for this section
        edges_x = F.conv2d(section_input, self.sobel_x, padding=1, groups=self.initial_channels)
        edges_y = F.conv2d(section_input, self.sobel_y, padding=1, groups=self.initial_channels)
        edges = torch.sqrt(edges_x.pow(2) + edges_y.pow(2) + 1e-6)
        
        # Combine features and edges
        combined = torch.cat([x, edges], dim=1)
        
        # Process through convolutions
        features = self.section_conv(combined)
        
        # Pool to fixed size - using fixed 2x2 output
        pooled = self.section_pool(features)
        # Use reshape instead of view for better compatibility
        batch_size = pooled.shape[0]
        flattened = pooled.reshape(batch_size, -1)
        # print(f"Section features shape: {flattened.shape}")
        return flattened
    
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])
        
        batch_size = visual_obs.shape[0]
        
        # Pre-calculate section indices for ONNX compatibility
        section_features = []
        
        # Process each section with fixed indices
        for i in range(5):  # Fixed number instead of self.num_sections
            start_idx = i * self.section_width
            if i == 4:  # Last section
                section = visual_obs[:, :, :, start_idx:]
            else:
                end_idx = (i + 1) * self.section_width
                section = visual_obs[:, :, :, start_idx:end_idx]
            
            features = self.extract_section_features(section)
            section_features.append(features)
        
        # Stack instead of cat for better ONNX support
        all_features = torch.cat(section_features, dim=1)
        
        # Apply cross-section attention
        attention_weights = self.section_attention(all_features)
        
        # Apply attention weights to each section - ONNX compatible approach
        weighted_section_features = []
        for i in range(5):  # Fixed range
            start_idx = i * self.features_per_section
            end_idx = (i + 1) * self.features_per_section
            section_feat = all_features[:, start_idx:end_idx]
            weight = attention_weights[:, i:i+1]
            # Use broadcasting for ONNX compatibility
            weighted_feat = section_feat * weight
            weighted_section_features.append(weighted_feat)
        
        # Concatenate weighted features instead of using index assignment
        weighted_features = torch.cat(weighted_section_features, dim=1)
        
        # Final projection
        output = self.final_proj(weighted_features)
        
        return output