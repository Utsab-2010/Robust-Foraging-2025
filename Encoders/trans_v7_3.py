import torch
import torch.nn as nn
import torch.nn.functional as F

class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.initial_channels = initial_channels
        self.num_sections = 5
        self.section_width = width // self.num_sections
        
        # Section importance weights - emphasize sides for better control
        # Higher weights for side sections to prevent marker from going out of sight
        section_weights = torch.tensor([1.5, 1.2, 1.0, 1.2, 1.5], dtype=torch.float32)  # Sides emphasized
        self.register_buffer('section_importance', section_weights)
        
        # COMMENTED OUT: Advanced edge detection for grey oval marker in foggy B&W environment
        # Multi-scale edge detection to handle different lighting conditions
        
        # # Classic Sobel filters
        # sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        # sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # # Prewitt filters - better for gradual transitions (good for 3D lighting)
        # prewitt_x = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32)
        # prewitt_y = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        
        # # Roberts cross-gradient - good for sharp edges
        # roberts_1 = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)
        # roberts_2 = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)
        
        # # Laplacian for second-order edge detection (good for oval shapes)
        # laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        
        # # Register all edge filters as buffers
        # self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        # self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))
        # self.register_buffer('prewitt_x', prewitt_x.unsqueeze(0).unsqueeze(0))
        # self.register_buffer('prewitt_y', prewitt_y.unsqueeze(0).unsqueeze(0))
        # self.register_buffer('roberts_1', F.pad(roberts_1.unsqueeze(0).unsqueeze(0), (0, 1, 0, 1)))
        # self.register_buffer('roberts_2', F.pad(roberts_2.unsqueeze(0).unsqueeze(0), (0, 1, 0, 1)))
        # self.register_buffer('laplacian', laplacian.unsqueeze(0).unsqueeze(0))
        
        # MULTIPLE EDGE DETECTION: Add back multiple filters (Barracuda-compatible)
        # Classic Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Prewitt filters - better for gradual transitions
        prewitt_x = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=torch.float32)
        prewitt_y = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        
        # Roberts cross-gradient - good for sharp edges (pad to 3x3 for consistency)
        roberts_1 = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=torch.float32)
        roberts_2 = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=torch.float32)
        
        # Register all as buffers - expand for all input channels
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0).repeat(1, initial_channels, 1, 1))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0).repeat(1, initial_channels, 1, 1))
        self.register_buffer('prewitt_x', prewitt_x.unsqueeze(0).unsqueeze(0).repeat(1, initial_channels, 1, 1))
        self.register_buffer('prewitt_y', prewitt_y.unsqueeze(0).unsqueeze(0).repeat(1, initial_channels, 1, 1))
        self.register_buffer('roberts_1', roberts_1.unsqueeze(0).unsqueeze(0).repeat(1, initial_channels, 1, 1))
        self.register_buffer('roberts_2', roberts_2.unsqueeze(0).unsqueeze(0).repeat(1, initial_channels, 1, 1))
        
        # Section-wise feature extraction with enhanced edge processing
        self.section_stem = nn.Sequential(
            nn.Conv2d(initial_channels, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),  # Better for lighting variations
            nn.LeakyReLU(0.1)
        )
        
        # MULTI-EDGE COMBINER: Process multiple edge detection results
        self.edge_combiner = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=1),  # 6 edge channels -> 16 features
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1)
        )
        
        # Main convolution pipeline - adjusted for multiple edge features
        self.section_conv = nn.Sequential(
            nn.Conv2d(24 + 16, 48, kernel_size=3, stride=2, padding=1),  # 24 features + 16 edge features = 40
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        
        # REMOVED: Complex sectioning and attention mechanisms for Barracuda compatibility
        # Original script had dynamic loops, slicing, and attention that Barracuda can't handle
        
        # SIMPLIFIED: Final projection for global pooled features (64 -> output_size)
        self.final_proj = nn.Sequential(
            nn.Linear(64, 256),  # 64 features from global average pooling
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_size)
            # Removed complex attention and sectioning for Barracuda compatibility
        )
        
   
    def multi_edge_detection(self, img):
        """Multi-filter edge detection - Barracuda compatible"""
        # Apply all 6 edge filters using F.conv2d
        sobel_x_edges = F.conv2d(img, self.sobel_x, padding=1, groups=self.initial_channels)
        sobel_y_edges = F.conv2d(img, self.sobel_y, padding=1, groups=self.initial_channels)
        prewitt_x_edges = F.conv2d(img, self.prewitt_x, padding=1, groups=self.initial_channels)
        prewitt_y_edges = F.conv2d(img, self.prewitt_y, padding=1, groups=self.initial_channels)
        roberts_1_edges = F.conv2d(img, self.roberts_1, padding=1, groups=self.initial_channels)
        roberts_2_edges = F.conv2d(img, self.roberts_2, padding=1, groups=self.initial_channels)
        
        # Take magnitude (absolute value) for each filter - more stable for ONNX than sqrt
        edge_list = [
            torch.abs(sobel_x_edges),
            torch.abs(sobel_y_edges), 
            torch.abs(prewitt_x_edges),
            torch.abs(prewitt_y_edges),
            torch.abs(roberts_1_edges),
            torch.abs(roberts_2_edges)
        ]
        
        # Average across input channels for each filter, then stack
        edge_channels = []
        for edges in edge_list:
            if edges.shape[1] > 1:
                edge_channel = edges.mean(dim=1, keepdim=True)
            else:
                edge_channel = edges
            edge_channels.append(edge_channel)
        
        # Concatenate all edge channels: [batch, 6, height, width]
        multi_edges = torch.cat(edge_channels, dim=1)
        return multi_edges
    
    def extract_section_features(self, section_input):
        """Extract features from a single section - simplified without edge detection"""
        # COMMENTED OUT: Apply advanced edge detection
        # edge_features = self.advanced_edge_detection(section_input)
        # edge_processed = self.edge_combiner(edge_features)
        
        # Regular feature extraction only
        regular_features = self.section_stem(section_input)
        
        # COMMENTED OUT: Combine regular and edge features
        # combined = torch.cat([regular_features, edge_processed], dim=1)
        
        # Process through main convolution pipeline (no edge features)
        conv_features = self.section_conv(regular_features)
        
        # BARRACUDA-COMPATIBLE: Use tensor.mean() instead of adaptive pooling
        # This performs global average pooling without ONNX export issues
        pooled = conv_features.mean(dim=[2, 3], keepdim=False)  # Global average pooling
        
        # No need for dynamic batch_size or reshape - pooled is already [batch, channels]
        return pooled  # Will be 64 features per section
    
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # BARRACUDA-COMPATIBLE: Always permute, no conditional behavior
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute(0, 3, 1, 2)
        
        # MULTI-EDGE DETECTION: Apply all 6 filters  
        edge_features = self.multi_edge_detection(visual_obs)
        edge_processed = self.edge_combiner(edge_features)
        
        # Regular feature extraction
        regular_features = self.section_stem(visual_obs)
        
        # Combine regular and edge features (simple concatenation)
        combined = torch.cat([regular_features, edge_processed], dim=1)  # 24 + 16 = 40 channels
        
        # Process through convolution pipeline
        conv_features = self.section_conv(combined)
        
        # Global average pooling - Barracuda friendly
        pooled = conv_features.mean(dim=[2, 3], keepdim=False)  # [batch, 64]
        
        # Simple projection to output size
        output = self.final_proj(pooled)
        
        return output