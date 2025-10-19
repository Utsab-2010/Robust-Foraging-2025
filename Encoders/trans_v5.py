class NatureVisualEncoder(nn.Module):
    def __init__(
        self, 
        height: int, 
        width: int, 
        initial_channels: int, 
        output_size: int
    ):
        super().__init__()
        self.h_size = output_size
        self.embed_size = 64
        self.patch_size = 32
        
        # Edge detection branch
        self.edge_detector = nn.Sequential(
            nn.Conv2d(initial_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU()
        )
        
        # Texture analysis branch
        self.texture_analyzer = nn.Sequential(
            nn.Conv2d(initial_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, kernel_size=5, padding=2, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=7, padding=3, bias=False),
            nn.LeakyReLU()
        )
        
        # Contrast enhancement branch
        self.contrast_enhancer = nn.Sequential(
            nn.Conv2d(initial_channels, 8, kernel_size=5, padding=2, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU()
        )
        
        # Feature fusion
        total_feature_channels = 16 + 16 + 16
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(total_feature_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(32, self.embed_size, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.embed_size, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # Patch embedding
        self.patch_embeddings = nn.Conv2d(
            self.embed_size, self.embed_size,
            kernel_size=self.patch_size, stride=self.patch_size,
            bias=False
        )
        
        patch_h = height // self.patch_size
        patch_w = width // self.patch_size
        self.final_flat = self.embed_size * patch_h * patch_w
        
        # Patch processor
        self.patch_processor = nn.Sequential(
            nn.Linear(self.final_flat, self.final_flat // 2),
            nn.LeakyReLU(),
            nn.Linear(self.final_flat // 2, self.final_flat // 2), 
            nn.LeakyReLU(),
            nn.Linear(self.final_flat // 2, self.final_flat),
            nn.LeakyReLU()
        )
        
        # Feature enhancer
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.final_flat, self.embed_size * 4),
            nn.LeakyReLU(),
            nn.Linear(self.embed_size * 4, self.final_flat),
            nn.LeakyReLU()
        )
        
        # Final dense layer
        self.dense = nn.Sequential(
            nn.Linear(self.final_flat, self.h_size),
            nn.LeakyReLU()
        )
    
    def forward(self, visual_obs):
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])

        # Multi-branch feature extraction
        edge_features = self.edge_detector(visual_obs)
        texture_features = self.texture_analyzer(visual_obs)
        contrast_features = self.contrast_enhancer(visual_obs)
        
        # Feature fusion
        combined_features = torch.cat([
            edge_features, 
            texture_features, 
            contrast_features
        ], dim=1)
        
        fused_features = self.feature_fusion(combined_features)
        
        # Spatial attention
        attention_map = self.spatial_attention(fused_features)
        attended_features = fused_features * attention_map
        
        # Patch embedding
        hidden = self.patch_embeddings(attended_features)
        hidden = hidden.reshape([-1, self.final_flat])
        
        # Processing
        hidden = self.patch_processor(hidden)
        enhanced = self.feature_enhancer(hidden)
        hidden = hidden + enhanced
        
        return self.dense(hidden)

