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
        
        # # Gaussian blur for noise reduction in foggy conditions
        # gaussian = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
        # self.register_buffer('gaussian', gaussian.unsqueeze(0).unsqueeze(0))
        
        # Section-wise feature extraction with enhanced edge processing
        self.section_stem = nn.Sequential(
            nn.Conv2d(initial_channels, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),  # Better for lighting variations
            nn.LeakyReLU(0.1)
        )
        
        # COMMENTED OUT: Edge feature processing - combines multiple edge detectors
        # self.edge_combiner = nn.Sequential(
        #     nn.Conv2d(7, 16, kernel_size=1),  # 7 edge channels -> 16 features
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.1)
        # )
        
        # Main convolution pipeline for each section - adjusted for no edge features
        self.section_conv = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),  # 24 features only (no edge features)
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        
        # Regular pooling for ONNX compatibility - not actually used, using F.adaptive_avg_pool2d instead
        self.section_pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Keep for compatibility
        # Fixed features per section with 2x2 global pooling
        self.features_per_section = 64  # 64 channels from global average pooling
        
        # Cross-section attention with section importance weighting - simplified for ONNX
        self.section_attention = nn.Sequential(
            nn.Linear(self.features_per_section * self.num_sections, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, self.num_sections)
            # Removed Dropout and Sigmoid for ONNX compatibility
        )
        
        # Final projection with section importance consideration - simplified for ONNX
        self.final_proj = nn.Sequential(
            nn.Linear(self.features_per_section * self.num_sections, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, output_size)
            # Removed Dropout for ONNX compatibility
        )
        
    # COMMENTED OUT: def advanced_edge_detection(self, img):
    #     """Multi-scale edge detection optimized for grey oval markers in foggy B&W"""
    #     # Convert to grayscale if needed
    #     if img.shape[1] > 1:
    #         gray = img.mean(dim=1, keepdim=True)
    #     else:
    #         gray = img
    #         
    #     # Apply Gaussian blur first to reduce fog noise
    #     denoised = F.conv2d(gray, self.gaussian, padding=1)
    #     
    #     # Apply multiple edge detectors
    #     sobel_x_edges = F.conv2d(denoised, self.sobel_x, padding=1)
    #     sobel_y_edges = F.conv2d(denoised, self.sobel_y, padding=1)
    #     sobel_magnitude = torch.sqrt(sobel_x_edges.pow(2) + sobel_y_edges.pow(2) + 1e-6)
    #     
    #     prewitt_x_edges = F.conv2d(denoised, self.prewitt_x, padding=1)
    #     prewitt_y_edges = F.conv2d(denoised, self.prewitt_y, padding=1)
    #     prewitt_magnitude = torch.sqrt(prewitt_x_edges.pow(2) + prewitt_y_edges.pow(2) + 1e-6)
    #     
    #     roberts_1_edges = F.conv2d(denoised, self.roberts_1, padding=1)
    #     roberts_2_edges = F.conv2d(denoised, self.roberts_2, padding=1)
    #     roberts_magnitude = torch.sqrt(roberts_1_edges.pow(2) + roberts_2_edges.pow(2) + 1e-6)
    #     
    #     # Laplacian for detecting oval/circular shapes
    #     laplacian_edges = torch.abs(F.conv2d(denoised, self.laplacian, padding=1))
    #     
    #     # Combine all edge features
    #     edge_stack = torch.cat([
    #         sobel_magnitude, prewitt_magnitude, roberts_magnitude, laplacian_edges,
    #         sobel_x_edges, sobel_y_edges, denoised  # Include original for context
    #     ], dim=1)
    #     
    #     return edge_stack
    
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
        
        # Use ONNX-compatible global pooling - replace adaptive pooling
        pooled = conv_features.mean(dim=[2, 3], keepdim=False)  # Global average pooling
        
        # No need for reshape since we already have [batch, channels]
        return pooled  # Will be 64 features per section
    
    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        # Always permute - remove dynamic ONNX check for compatibility
        visual_obs = visual_obs.permute(0, 3, 1, 2)
        
        # Process each section - use fixed indexing for ONNX compatibility
        section_0 = visual_obs[:, :, :, 0:self.section_width]
        section_1 = visual_obs[:, :, :, self.section_width:2*self.section_width]  
        section_2 = visual_obs[:, :, :, 2*self.section_width:3*self.section_width]
        section_3 = visual_obs[:, :, :, 3*self.section_width:4*self.section_width]
        section_4 = visual_obs[:, :, :, 4*self.section_width:]  # Last section gets remainder
        
        # Extract features from each section
        features_0 = self.extract_section_features(section_0)
        features_1 = self.extract_section_features(section_1)
        features_2 = self.extract_section_features(section_2)
        features_3 = self.extract_section_features(section_3)
        features_4 = self.extract_section_features(section_4)
        
        # Concatenate all section features - static operation for ONNX
        all_features = torch.cat([features_0, features_1, features_2, features_3, features_4], dim=1)
        
        # Generate attention weights for each section - apply softmax for normalization
        attention_logits = self.section_attention(all_features)  # [batch, num_sections]
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # Apply section importance weighting (emphasize sides) - ONNX compatible
        importance_weights = self.section_importance.unsqueeze(0)  # [1, num_sections]
        combined_weights = attention_weights * importance_weights
        
        # Normalize weights for ONNX compatibility
        weight_sum = combined_weights.sum(dim=1, keepdim=True)
        combined_weights = combined_weights / (weight_sum + 1e-8)
        
        # Apply weighted attention to each section
        weighted_features = []
        for i in range(self.num_sections):
            start_idx = i * self.features_per_section
            end_idx = (i + 1) * self.features_per_section
            section_feat = all_features[:, start_idx:end_idx]
            weight = combined_weights[:, i:i+1]  # [batch, 1]
            weighted_feat = section_feat * weight
            weighted_features.append(weighted_feat)
        
        # Combine weighted features
        final_features = torch.cat(weighted_features, dim=1)
        
        # Final projection to output
        output = self.final_proj(final_features)
        
        return output