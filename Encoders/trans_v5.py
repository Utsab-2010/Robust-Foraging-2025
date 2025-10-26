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
        self.embed_size = 32
        self.patch_size = 32
        
        # Simple feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(initial_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(16, self.embed_size, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU()
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
        
        # Final dense layer
        self.dense = nn.Sequential(
            nn.Linear(self.final_flat, self.h_size),
            nn.LeakyReLU()
        )
    
    def forward(self, visual_obs):
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])

        # Simple feature extraction
        features = self.feature_extractor(visual_obs)
        
        # Patch embedding
        hidden = self.patch_embeddings(features)
        hidden = hidden.reshape([-1, self.final_flat])
        
        return self.dense(hidden)