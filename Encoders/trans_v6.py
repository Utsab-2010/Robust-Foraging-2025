
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
        self.embed_size = 32  # Reduced from 64 since we're processing halves
        self.patch_size = 32
        self.width = width
        
        # Calculate split dimensions
        self.half_width = width // 2  # Split at middle: left and right halves
        
        # Left half feature extractor (reduced parameters for half-width processing)
        self.left_feature_extractor = nn.Sequential(
            nn.Conv2d(initial_channels, 4, kernel_size=3, padding=1, bias=False),  # Reduced from 8 to 4
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),                # Reduced from 16 to 8
            nn.LeakyReLU(),
            nn.Conv2d(8, self.embed_size, kernel_size=3, padding=1, bias=False),   # embed_size now 32
            nn.LeakyReLU()
        )
        
        # Right half feature extractor (reduced parameters for half-width processing)
        self.right_feature_extractor = nn.Sequential(
            nn.Conv2d(initial_channels, 4, kernel_size=3, padding=1, bias=False),  # Reduced from 8 to 4
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),                # Reduced from 16 to 8
            nn.LeakyReLU(),
            nn.Conv2d(8, self.embed_size, kernel_size=3, padding=1, bias=False),   # embed_size now 32
            nn.LeakyReLU()
        )
        
        # Patch embedding for left half
        self.left_patch_embeddings = nn.Conv2d(
            self.embed_size, self.embed_size,
            kernel_size=self.patch_size, stride=self.patch_size,
            bias=False
        )
        
        # Patch embedding for right half  
        self.right_patch_embeddings = nn.Conv2d(
            self.embed_size, self.embed_size,
            kernel_size=self.patch_size, stride=self.patch_size,
            bias=False
        )
        
        # Calculate dimensions for each half
        patch_h = height // self.patch_size
        left_patch_w = self.half_width // self.patch_size
        right_patch_w = (width - self.half_width) // self.patch_size
        
        self.left_flat = self.embed_size * patch_h * left_patch_w
        self.right_flat = self.embed_size * patch_h * right_patch_w
        self.combined_flat = self.left_flat + self.right_flat
        
        # Cross-hemisphere processing (more efficient with reduced dimensions)
        self.cross_processor = nn.Sequential(
            nn.Linear(self.combined_flat, self.combined_flat // 4),  # More aggressive reduction
            nn.LeakyReLU(),
            nn.Linear(self.combined_flat // 4, self.h_size),         # Direct to output size
            nn.Tanh()
        )
        
        # # Final dense layer
        # self.dense = nn.Sequential(
        #     nn.Linear(self.embed_size * 4, self.h_size),
        #     nn.LeakyReLU()
        # )
    
    def forward(self, visual_obs):
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])

        # Split input into left and right halves
        # visual_obs shape: (batch, channels, height, width)
        left_half = visual_obs[:, :, :, :self.half_width]      # Left half: [:, :, :, :width//2]
        right_half = visual_obs[:, :, :, self.half_width:]     # Right half: [:, :, :, width//2:]
        
        # Process left half
        left_features = self.left_feature_extractor(left_half)
        left_patches = self.left_patch_embeddings(left_features)
        left_flat = left_patches.reshape([-1, self.left_flat])
        
        # Process right half  
        right_features = self.right_feature_extractor(right_half)
        right_patches = self.right_patch_embeddings(right_features)
        right_flat = right_patches.reshape([-1, self.right_flat])
        
        # Concatenate left and right embeddings
        combined = torch.cat([left_flat, right_flat], dim=1)  # Shape: (batch, left_flat + right_flat)
        
        # Cross-hemisphere processing
        # cross_processed = self.cross_processor(combined)
        
        # Final embedding
        return self.cross_processor(combined)

