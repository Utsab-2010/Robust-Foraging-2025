class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size  # Mirror neurips naming
        # print("Input image size:", height, "x", width)
        # print("Output size:", output_size)
        # Lightweight params: larger patches, small embed for <5MB
        self.kernel_size = 12
        self.embed_size = 32
        self.head_size = 32  # Single-head
        
        # Fixed patch grid (no dynamic ceiling; assume exact for static export)
        self.num_patches_h = height // self.kernel_size  # 155//12=12 (drop last if partial)
        self.num_patches_w = width // self.kernel_size   # 86//12=7
        self.num_patches = self.num_patches_h * self.num_patches_w  # 12x7=84 (static)
        # Calculate final flattened size like neurips/trans_v4 (CRITICAL!)
        patch_h = height // self.kernel_size  # 86 // 12 = 7
        patch_w = width // self.kernel_size   # 155 // 12 = 12
        self.final_flat = self.embed_size * patch_h * patch_w  # Fixed size calculation

        # Patch embedding (conv, like neurips conv_layers start)
        self.patch_embeddings = nn.Conv2d(
            initial_channels, self.embed_size,
            kernel_size=self.kernel_size, stride=self.kernel_size,
            bias=False  # Lightweight, no bias
        )

        # Fixed sinusoidal positional encoding (Barracuda-compatible: buffer + Add)
        # self.register_buffer('positional_enc', self._generate_positional_encoding())

        # Attention projections (sequential, no bias)
        self.query = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.key = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.value = nn.Linear(self.embed_size, self.head_size, bias=False)

        # Lightweight MLP with LeakyReLU (2x expansion)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, 2 * self.embed_size),
            nn.LeakyReLU(),  # Barracuda-supported
            nn.Linear(2 * self.embed_size, self.embed_size),
            nn.LeakyReLU()
        )

        # Dense (like neurips dense, input will be embed_size after mean pooling)
        self.dense = nn.Sequential(
            nn.Linear(self.embed_size, self.h_size),  # Changed from final_flat to embed_size
            nn.LeakyReLU()  # Match neurips
        )

        self.log_const = torch.log(torch.tensor(10000.0)).item()  # For fixed positional encoding

    # def _generate_positional_encoding(self) -> torch.Tensor:
    #     # Fixed sinusoidal positions (no Embedding; static buffer)
    #     pe = torch.zeros(self.num_patches, self.embed_size)
    #     position = torch.arange(0, self.num_patches, dtype=torch.float).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, self.embed_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embed_size))
    #     pe[:, 0::2] = torch.sin(position * div_term)
    #     pe[:, 1::2] = torch.cos(position * div_term)
    #     return pe.unsqueeze(0)  # (1, N, E) - broadcastable

    def attention(self, hidden: torch.Tensor) -> torch.Tensor:
        # Simplified sequential attention (no residuals for graph simplicity)
        query = self.query(hidden)
        query = torch.clamp(query, -1e2, 1e2)  # Prevent extreme values
        key = self.key(hidden)
        key = torch.clamp(key, -1e2, 1e2)
        value = self.value(hidden)
        value = torch.clamp(value, -1e2, 1e2)
        # Scaled dot-product (supported ops: MatMul, Div, Softmax)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float))
        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        # attn_output = torch.nan_to_num(attn_output, nan=0.0, posinf=1.0, neginf=-1.0)
        # attn_output = torch.clamp(attn_output, -1e6, 1e4)

        return attn_output

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # Mirror neurips: conditional permute for training (NHWC input)
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])  # To (B, C, H, W)

        # Patch extraction (like neurips conv_layers)
        hidden = self.patch_embeddings(visual_obs)  # (B, E, H/K, W/K)
        
        # SAFE APPROACH: Use neurips-style reshape, then properly apply attention
        hidden = hidden.reshape([-1, self.final_flat])  # (B, E*H*W) - FIXED SIZE
        batch_size = hidden.shape[0]
        
        # Reshape to proper attention format using FIXED dimensions
        # We know: final_flat = embed_size * patch_h * patch_w
        # So we can safely reshape to (B, num_patches, embed_size)
        num_patches = self.final_flat // self.embed_size  # This is fixed at init time
        
        # Safe reshape to patches for attention
        hidden_patches = hidden.reshape(batch_size, num_patches, self.embed_size)  # (B, N, E)
        
        # Apply attention mechanism (this was never the problem!)
        attended = self.attention(hidden_patches)  # (B, N, head_size)
        
        # Add residual connection
        residual = hidden_patches
        if attended.shape == residual.shape:  # Only if dimensions match
            hidden_patches = attended + residual
        else:
            hidden_patches = attended
            
        # Apply MLP
        hidden_patches = self.mlp(hidden_patches)  # (B, N, E)
        
        # SAFE aggregation: Simple mean over patches (this should be fine)
        hidden = hidden_patches.mean(dim=1)  # (B, E)
        
        # Reshape for final dense layer
        hidden = hidden.reshape([-1, self.embed_size])
        return self.dense(hidden)  # (B, output_size)
