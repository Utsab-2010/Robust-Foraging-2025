
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
        
        # Calculate split dimensions - each part extends to 1.4x half width for overlap
        self.half_width = width // 2
        self.left_width = int(self.half_width * 1.4)   # Left extends to 1.4x half width
        self.right_start = width - int(self.half_width * 1.4)  # Right starts earlier for 1.4x coverage
        
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
        
        # Calculate dimensions for each extended part
        patch_h = height // self.patch_size
        left_patch_w = self.left_width // self.patch_size  # Left part extends to 1.4x half width
        right_patch_w = (width - self.right_start) // self.patch_size  # Right part from start to end
        
        self.left_flat = self.embed_size * patch_h * left_patch_w
        self.right_flat = self.embed_size * patch_h * right_patch_w
        self.combined_flat = self.left_flat + self.right_flat
        
        # Cross-hemisphere processing (more efficient with reduced dimensions)
        self.cross_processor = nn.Sequential(
            nn.Linear(self.combined_flat, self.combined_flat // 4),  # More aggressive reduction
            nn.LeakyReLU(),
            nn.Linear(self.combined_flat // 4, self.h_size),         # Direct to output size
            nn.LeakyReLU()
        )
        
        # # Final dense layer
        # self.dense = nn.Sequential(
        #     nn.Linear(self.embed_size * 4, self.h_size),
        #     nn.LeakyReLU()
        # )
    
    def forward(self, visual_obs):
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])

        # Split input into overlapping left and right parts (1.4x half width each)
        # visual_obs shape: (batch, channels, height, width)
        left_part = visual_obs[:, :, :, :self.left_width]           # Left part: extends to 1.4x half width
        right_part = visual_obs[:, :, :, self.right_start:]         # Right part: starts earlier for 1.4x coverage
        
        # Process left part (1.4x half width)
        left_features = self.left_feature_extractor(left_part)
        left_patches = self.left_patch_embeddings(left_features)
        left_flat = left_patches.reshape([-1, self.left_flat])
        
        # Process right part (1.4x half width)
        right_features = self.right_feature_extractor(right_part)
        right_patches = self.right_patch_embeddings(right_features)
        right_flat = right_patches.reshape([-1, self.right_flat])
        
        # Concatenate left and right embeddings
        combined = torch.cat([left_flat, right_flat], dim=1)  # Shape: (batch, left_flat + right_flat)
        
        # Cross-hemisphere processing
        # cross_processed = self.cross_processor(combined)
        
        # Final embedding
        return self.cross_processor(combined)

