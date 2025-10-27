class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        
        self.h_size = output_size
        weights_path = "D:\mouse_vs_ai_windows\dinov2_vits14_reg4_pretrain.pth"
        
        # Add channel mapping layer for grayscale to RGB
        self.channel_map = nn.Conv2d(
            in_channels=1, 
            out_channels=3, 
            kernel_size=1,
            bias=False
        )
        # Initialize to equal weights to maintain the grayscale information
        self.channel_map.weight.data.fill_(1/3)
        
        # Create DINOv2 backbone
        self.backbone = timm.create_model(
            'vit_small_patch14_dinov2.lvd142m',
            pretrained=False,
            num_classes=0  # Get features instead of logits
        )
        
        # Add projection to match required output size
        self.proj = nn.Sequential(
            nn.LayerNorm(384),  # DINOv2 ViT-S has 384 features
            nn.Linear(384, output_size)
        )
        
        # Load pretrained weights if provided
        if weights_path is not None:
            print(f"Loading pretrained weights from {weights_path}")
            self._load_pretrained_weights(weights_path)
            
            # Freeze backbone weights
            print("Freezing DINO backbone weights...")
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen - only projection layer will be trained")
            
        # Normalization parameters (ImageNet stats as used in DINOv2)
        # self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _load_pretrained_weights(self, weights_path: str):
        print(f"Loading pretrained weights from {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            # Filter only backbone weights
            model_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('head'):
                    model_state_dict[k] = v
            
            msg = self.backbone.load_state_dict(model_state_dict, strict=False)
            print(f"Loaded DINOv2 weights from {weights_path}")
            print(f"Missing keys: {msg.missing_keys}")
            print(f"Unexpected keys: {msg.unexpected_keys}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Continuing with random initialization")

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # Handle ONNX export format
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])
            
        # Convert grayscale to RGB using 1x1 convolution
        if visual_obs.shape[1] == 1:
            visual_obs = self.channel_map(visual_obs)
            
        # Resize input to 518x518 (DINOv2's expected size)
        visual_obs = torch.nn.functional.interpolate(
            visual_obs, 
            size=(518, 518), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Get features from backbone
        features = self.backbone(visual_obs)
        
        # Project to required output size
        x = self.proj(features)
        
        return x

    # @staticmethod
    # def create_input_processors(height: int, width: int) -> dict:
    #     """
    #     Creates input processors for the visual observation.
    #     Required for ML-Agents compatibility.
    #     """
    #     return {
    #         "visual": ModelUtils.get_image_encoder_processor(height, width),
    #     }
