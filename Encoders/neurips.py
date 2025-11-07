class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        
        # Use global average pooling instead of adaptive pooling for ONNX compatibility
        self.final_flat = 32  # Global pooling outputs 32 features (number of channels)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(initial_channels, 64, [6, 6], [3, 3]),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, [4, 4], [2, 2]),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, [3, 3], [1, 1]),
            nn.LeakyReLU(),
        )
        
        # No separate pooling layer needed - using global average pooling in forward
        
        self.dense = nn.Sequential(
            linear_layer(
                self.final_flat,
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,
            ),
            nn.LeakyReLU(),
        )
        print(f"NatureVisualEncoder initialized with global average pooling, final_flat: {self.final_flat}")
        
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])
        hidden = self.conv_layers(visual_obs)
        print(f"Shape after conv layers: {hidden.shape}")
        
        # Use ONNX-compatible global average pooling instead of adaptive pooling
        hidden = hidden.mean(dim=[2, 3], keepdim=False)  # Global average pooling
        print(f"Shape after global average pooling: {hidden.shape}")
        
        # No reshape needed since we now have [batch, 32] directly
        return self.dense(hidden)