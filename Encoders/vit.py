class NatureVisualEncoder(nn.Module):

    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size

       
        self.cnn = nn.Sequential(
            nn.Conv2d(initial_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        
        h_out = height // 4
        w_out = width // 4

       
        
        self.vit = timm.create_model(
            "deit_tiny_patch16_224 ",
            pretrained=False,
            num_classes=0,          # no classification head
            img_size=(h_out, w_out),
            in_chans=64,
            embed_dim=128,
            depth=2,
            num_heads=4,
        )

       
        self.fc = nn.Sequential(
            linear_layer(
                128, 
                self.h_size,
                kernel_init=Initialization.KaimingHeNormal,
                kernel_gain=1.41,
            ),
            nn.LeakyReLU(),
        )

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute(0, 3, 1, 2)

        x = self.cnn(visual_obs)
        x = self.vit(x)
        x = self.fc(x)
        return x