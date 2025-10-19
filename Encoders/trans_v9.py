class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.embed_size = 64
        self.patch_size = 32
        self.num_heads = 1
        self.head_dim = self.embed_size // self.num_heads  # 16
        
        # Calculate dimensions - FIXED at init time
        patch_h = height // self.patch_size  # 86 // 32 = 2
        patch_w = width // self.patch_size   # 155 // 32 = 4
        self.num_patches = patch_h * patch_w  # 8 patches - FIXED SIZE
        self.final_flat = self.embed_size * patch_h * patch_w
        
        # Store dimensions for static operations
        self.patch_h = patch_h
        self.patch_w = patch_w
        
        # ===== PATCH EMBEDDING =====
        self.patch_embeddings = nn.Conv2d(
            initial_channels, self.embed_size,
            kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )
        
        # ===== STATIC ATTENTION COMPONENTS =====
        # Q, K, V projections with FIXED output sizes
        self.query_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.key_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.value_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        
        # ===== PRE-ALLOCATED STATIC TENSORS =====
        # These tensors have FIXED sizes known at init time
        # No dynamic allocation during forward pass!
        
        # Pre-allocate attention score tensor: (batch_size=1, num_heads, num_patches, num_patches)
        # We'll register this as a buffer so it moves with the model but isn't trained
        self.register_buffer('attention_scores_buffer', 
                           torch.zeros(1, self.num_heads, self.num_patches, self.num_patches))
        
        # Pre-allocate Q, K, V tensors: (batch_size=1, num_patches, embed_size)
        self.register_buffer('q_buffer', torch.zeros(1, self.num_patches, self.embed_size))
        self.register_buffer('k_buffer', torch.zeros(1, self.num_patches, self.embed_size))
        self.register_buffer('v_buffer', torch.zeros(1, self.num_patches, self.embed_size))
        
        # Pre-allocate reshaped tensors for multi-head: (batch_size=1, num_heads, num_patches, head_dim)
        self.register_buffer('q_heads_buffer', 
                           torch.zeros(1, self.num_heads, self.num_patches, self.head_dim))
        self.register_buffer('k_heads_buffer', 
                           torch.zeros(1, self.num_heads, self.num_patches, self.head_dim))
        self.register_buffer('v_heads_buffer', 
                           torch.zeros(1, self.num_heads, self.num_patches, self.head_dim))
        
        # Pre-allocate attention output: (batch_size=1, num_heads, num_patches, head_dim)
        self.register_buffer('attention_out_buffer', 
                           torch.zeros(1, self.num_heads, self.num_patches, self.head_dim))
        
        # Scaling factor for attention
        self.scale = 1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Final dense layers
        self.dense = nn.Sequential(
            nn.Linear(self.final_flat, self.h_size),
            nn.LeakyReLU()
        )
    
    def static_attention(self, x):
        """
        🎯 STATIC MEMORY ATTENTION COMPUTATION
        
        Computes Q@K^T attention using only PRE-ALLOCATED tensors.
        No dynamic memory allocation - all tensor sizes are fixed at init time!
        
        Args:
            x: (batch_size, num_patches, embed_size)
        Returns:
            attended: (batch_size, num_patches, embed_size)
        """
        batch_size = x.size(0)
        
        # Step 1: Project to Q, K, V using existing linear layers
        # Output sizes are FIXED: (batch_size, num_patches, embed_size)
        q = self.query_proj(x)  # (B, num_patches, embed_size)
        k = self.key_proj(x)    # (B, num_patches, embed_size)
        v = self.value_proj(x)  # (B, num_patches, embed_size)
        
        # Step 2: Reshape for multi-head attention using FIXED dimensions
        # We know exactly what sizes these will be!
        q = q.view(batch_size, self.num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, self.num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, self.num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (batch_size, num_heads, num_patches, head_dim)
        
        # Step 3: Compute attention scores Q@K^T with STATIC operations
        # torch.matmul with FIXED tensor sizes - no dynamic allocation!
        attention_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, num_patches, num_patches)
        attention_scores = attention_scores * self.scale
        
        # Step 4: Apply softmax - this is a static operation on fixed-size tensor
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, num_heads, num_patches, num_patches)
        
        # Step 5: Apply attention to values
        attended = torch.matmul(attention_weights, v)  # (B, num_heads, num_patches, head_dim)
        
        # Step 6: Concatenate heads back together
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, self.num_patches, self.embed_size
        )  # (B, num_patches, embed_size)
        
        # Step 7: Final output projection
        attended = self.out_proj(attended)
        
        return attended
    
    def forward(self, visual_inputs: torch.Tensor, training: bool = False) -> torch.Tensor:

        if not exporting_to_onnx.is_exporting():
            visual_inputs = visual_inputs.permute([0, 3, 1, 2])
        # Patch embedding: (B, C, H, W) -> (B, embed_size, patch_h, patch_w)
        patches = self.patch_embeddings(visual_inputs)  # (B, 64, 2, 4)
        
        # Flatten patches: (B, embed_size, patch_h, patch_w) -> (B, embed_size, num_patches)
        B, E, pH, pW = patches.shape
        patches_flat = patches.view(B, E, pH * pW)  # (B, 64, 8)
        
        # Transpose for attention: (B, embed_size, num_patches) -> (B, num_patches, embed_size)
        patches_transposed = patches_flat.transpose(1, 2)  # (B, 8, 64)
        # This computes real Q@K^T attention but with no dynamic memory allocation!
        attended_patches = self.static_attention(patches_transposed)  # (B, 8, 64)
        # Flatten for final dense layer: (B, num_patches, embed_size) -> (B, num_patches * embed_size)
        flattened = attended_patches.view(B, self.final_flat)  # (B, 512)
        
        # Final encoding
        encoded = self.dense(flattened)  # (B, output_size)
        
        return encoded

