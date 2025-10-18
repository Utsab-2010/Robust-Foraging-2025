class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size  # Mirror neurips naming
        
        # Lightweight params: larger patches, small embed for <5MB
        self.kernel_size = 12
        self.embed_size = 32
        self.head_size = 32  # Single-head
        
        # Compute patch grid (like neurips' conv_output_shape)
        self.num_patches_h = (height + self.kernel_size - 1) // self.kernel_size  # Ceiling for padding
        self.num_patches_w = (width + self.kernel_size - 1) // self.kernel_size
        self.num_patches = self.num_patches_h * self.num_patches_w  # e.g., 13x8=104
        self.final_flat = self.num_patches * self.embed_size  # Equivalent to neurips' final_flat

        # Patch embedding (conv, like neurips conv_layers start)
        self.patch_embeddings = nn.Conv2d(
            initial_channels, self.embed_size,
            kernel_size=self.kernel_size, stride=self.kernel_size,
            bias=False  # Lightweight, no bias
        )

        # Positional embeddings (small due to fewer patches)
        self.position_embeddings = nn.Embedding(self.num_patches, self.embed_size)

        # Attention projections (no bias, lightweight)
        self.query = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.key = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.value = nn.Linear(self.embed_size, self.head_size, bias=False)

        # Lightweight MLP (2x expansion, like neurips' simple dense but residual)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, 2 * self.embed_size),
            nn.GELU(),  # Similar to LeakyReLU in neurips
            nn.Linear(2 * self.embed_size, self.embed_size)
        )

        # Dense (like neurips' dense, but after mean pool for efficiency)
        self.dense = nn.Sequential(
            nn.Linear(self.embed_size, self.h_size),  # Post-mean: small input
            nn.LeakyReLU()  # Match neurips activation
        )

    def attention(self, hidden: torch.Tensor) -> torch.Tensor:
        # Scaled dot-product (lightweight single-head)
        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_size ** 0.5)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # Mirror neurips: conditional permute only during training (NHWC input)
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])  # To (B, C, H, W)

        # Patch extraction (like neurips conv_layers)
        hidden = self.patch_embeddings(visual_obs)  # (B, E, H/K, W/K)
        
        # Flatten and transpose (like neurips reshape, but to (B, N, E))
        hidden = hidden.flatten(2).transpose(1, 2)  # (B, N', E); N' <= num_patches

        # Add positional (expand to actual N')
        actual_n = hidden.size(1)
        position_ids = torch.arange(actual_n, device=hidden.device)
        position_ids = position_ids.expand((visual_obs.size(0), -1))  # (B, N')
        position_embeds = self.position_embeddings(position_ids)  # (B, N', E)
        hidden = hidden + position_embeds  # (B, N', E)

        # Residual attention (transformer block, lightweight)
        residual = hidden
        hidden = self.layer_norm(hidden) if hasattr(self, 'layer_norm') else hidden  # Optional, skipped for light
        hidden = self.attention(hidden)
        hidden = hidden + residual

        # Residual MLP
        residual = hidden
        hidden = self.mlp(hidden)
        hidden = hidden + residual

        # Mean pool (global avg, efficient flatten alternative to neurips' large reshape)
        hidden = hidden.mean(dim=1)  # (B, E) - avoids big final_flat like neurips

        # Dense output (mirror neurips dense)
        hidden = hidden.reshape([-1, self.embed_size])  # Ensure (B, E)
        return self.dense(hidden)