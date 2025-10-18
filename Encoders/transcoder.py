class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.kernel_size = 6
        self.num_patches = (height // self.kernel_size) * (width // self.kernel_size)
        print("height:", height)
        print("width:", width)
        self.num_positions = self.num_patches
        self.embed_size = 128
        self.head_size = 128

        self.patch_embeddings = nn.Conv2d(
            initial_channels,
            self.embed_size,
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            bias=False
        )

        self.position_embeddings = nn.Embedding(
            self.num_positions,
            self.embed_size
        )

        self.query = nn.Linear(self.embed_size, self.head_size,bias=False)
        self.key = nn.Linear(self.embed_size, self.head_size,bias=False)
        self.value = nn.Linear(self.embed_size, self.head_size,bias=False)

        self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.embed_size, 4 * self.embed_size),
                    nn.GELU(),
                    nn.Linear(4 * self.embed_size, self.embed_size)
                )

        self.final_layer = nn.Sequential(
            nn.Linear(self.embed_size, 2*self.embed_size),
            nn.LeakyReLU(),
            nn.Linear(2*self.embed_size, output_size)
        )

    def attention(self, hidden) :
        query = self.query(hidden)  # (B,N,E)
        key = self.key(hidden)      # (B,N,E)
        value = self.value(hidden)  # (B,N,E)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_size ** 0.5)  # (B,N,N)
        attn_weights = attn_weights.softmax(dim=-1)  # (B,N,N)

        attn_output = torch.matmul(attn_weights, value)  # (B,N,E)
        return attn_output


        
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2]) #(B,E,H,W)

        hidden = self.patch_embeddings(visual_obs) #(B,E,H/K,W/K)
        hidden = hidden.flatten(2) # (B,E,N) where N = (H/K)*(W/K)
        hidden = hidden.transpose(1, 2) # (B,N,E)

        position_ids = torch.arange(self.num_positions, device=hidden.device)
        position_ids = position_ids.expand((visual_obs.size(0), -1))  # [B, N]
        position_embeds = self.position_embeddings(position_ids) # (B,N,E)
        hidden = hidden + position_embeds # (B,N,E)

        residual = hidden
        hidden = self.attention(hidden)  # (B,N,E)
        hidden = self.dropout(hidden)  # (B,N,E)
        hidden = hidden + residual

        residual = hidden
        hidden = self.mlp(hidden)
        hidden = self.dropout(hidden)
        hidden = hidden + residual # (B,N,E)
        hidden = hidden.max(dim=1).values # (B,E)

        output = self.final_layer(hidden)

        return output

