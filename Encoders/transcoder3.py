class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.kernel_size = 6
        self.embed_size = 128
        self.head_size = 128

        # Compute patch grid
        self.num_patches_h = height // self.kernel_size
        self.num_patches_w = width // self.kernel_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch embedding projection (similar to conv feature extraction)
        self.patch_embeddings = nn.Conv2d(
            in_channels=initial_channels,
            out_channels=self.embed_size,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            bias=False
        )

        # Positional embedding per patch
        self.position_embeddings = nn.Embedding(self.num_patches, self.embed_size)

        # Attention projections
        self.query = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.key = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.value = nn.Linear(self.embed_size, self.head_size, bias=False)

        # Normalization + MLP (transformer block)
        self.layer_norm = nn.LayerNorm(self.embed_size, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, 4 * self.embed_size),
            nn.GELU(),
            nn.Linear(4 * self.embed_size, self.embed_size),
        )

        # Dense projection matching CNN encoder’s output
        self.dense = nn.Sequential(
            nn.Linear(self.embed_size * self.num_patches, self.h_size),
            nn.LeakyReLU(),
        )

    def attention(self, hidden):
        # Scaled Dot-Product Attention
        query = self.query(hidden)
        key = self.key(hidden)
        value = self.value(hidden)
        attn = torch.matmul(query, key.transpose(-2, -1)) / (self.head_size ** 0.5)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, value)
        return out

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])  # (B,C,H,W)

        hidden = self.patch_embeddings(visual_obs)         # (B,E,H/K,W/K)
        hidden = hidden.flatten(2).transpose(1, 2)         # (B,N,E)

        # Add positional embeddings
        positions = torch.arange(self.num_patches, device=hidden.device).unsqueeze(0)
        hidden = hidden + self.position_embeddings(positions)

        # Transformer-style update
        residual = hidden
        hidden = self.layer_norm(hidden)
        hidden = self.attention(hidden) + residual

        residual = hidden
        hidden = self.layer_norm(hidden)
        hidden = self.mlp(hidden) + residual

        # Flatten all patch embeddings like CNN flatten
        hidden = hidden.reshape(hidden.size(0), -1)         # (B, N*E)

        return self.dense(hidden)
