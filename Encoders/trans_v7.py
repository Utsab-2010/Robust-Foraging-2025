class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.embed_size = 64
        self.head_size = 64
        self.patch_size = 16  # Smaller patches for better compatibility
        
        # Calculate dimensions (matching neurips pattern)
        patch_h = height // self.patch_size  # 86 // 16 = 5
        patch_w = width // self.patch_size   # 155 // 16 = 9
        self.num_patches = patch_h * patch_w  # 45 patches
        self.final_flat = self.embed_size * patch_h * patch_w  # 64 * 5 * 9 = 2880
        
        # Patch embedding (Conv2d with stride = patch_size)
        self.patch_embeddings = nn.Conv2d(
            initial_channels, self.embed_size,
            kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )
        
        # Attention components (simplified)
        self.query = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.key = nn.Linear(self.embed_size, self.head_size, bias=False) 
        self.value = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.scale = self.head_size ** -0.5
        
        # Output projection 
        self.out_proj = nn.Linear(self.head_size, self.embed_size, bias=False)
        
        # Final processing (like neurips)
        self.dense = nn.Sequential(
            nn.Linear(self.final_flat, self.h_size),
            nn.LeakyReLU()
        )
    
    def forward(self, visual_obs):
        # Standard ONNX preparation
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])
        
        # Patch extraction
        patches = self.patch_embeddings(visual_obs)  # (B, embed_size, H', W')
        
        # Flatten like neurips (single safe reshape)
        hidden = patches.reshape([-1, self.final_flat])  # (B, embed_size * H' * W')
        
        # MINIMAL ATTENTION: Work directly on flattened patches
        B = hidden.shape[0]
        
        # Reshape to patches for attention (static dimensions only)
        patches_2d = hidden.view(B, self.num_patches, self.embed_size)  # (B, N, E)
        
        # Compute attention (minimal operations)
        Q = self.query(patches_2d)  # (B, N, head_size)
        K = self.key(patches_2d)    # (B, N, head_size) 
        V = self.value(patches_2d)  # (B, N, head_size)
        
        # Attention computation (simplified)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, N, N)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, N, N)
        attn_out = torch.matmul(attn_weights, V)  # (B, N, head_size)
        
        # Project back to embed_size
        attn_proj = self.out_proj(attn_out)  # (B, N, embed_size)
        
        # Flatten back to original size (static dimensions)
        attended_flat = attn_proj.view(B, self.final_flat)  # (B, embed_size * N)
        
        # Final processing
        return self.dense(attended_flat)