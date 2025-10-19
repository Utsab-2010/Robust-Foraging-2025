import torch
import torch.nn as nn
from mlagents.torch_utils import torch_utils
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.settings import EncoderType, ScheduleType
from mlagents.trainers.torch_entities.utils import linear_layer, Initialization
from mlagents_envs.base_env import ActionSpec, BehaviorSpec
from mlagents.trainers.torch_entities.layers import LinearEncoder
from mlagents.trainers.exception import UnityTrainerException
from mlagents.torch_utils.globals import exporting_to_onnx

class NatureVisualEncoder(nn.Module):
    def __init__(
        self, 
        height: int, 
        width: int, 
        initial_channels: int, 
        output_size: int
    ):
        super().__init__()
        self.h_size = output_size  # Mirror neurips/trans_fixed naming
        self.embed_size = 64  # Smaller embed size for efficiency
        self.head_size = 64   # Single-head attention
        self.patch_size = 32
        
        # Compute expected num_patches based on init height/width (no padding)
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        self.num_patches = num_patches_h * num_patches_w
        # Calculate final flattened size like neurips does (CRITICAL!)
        patch_h = height // self.patch_size  # 86 // 8 = 10
        patch_w = width // self.patch_size   # 155 // 8 = 19
        self.final_flat = self.embed_size * patch_h * patch_w  # Fixed size calculation
        
        # Conv2d for patch embedding: projects to embed_size with stride=patch_size (no padding)
        self.patch_embeddings = nn.Conv2d(
            initial_channels, self.embed_size,
            kernel_size=self.patch_size, stride=self.patch_size,
            bias=False  # Lightweight, no bias like trans_fixed
        )
        
        # Separate Q, K, V projections (ONNX-compatible, like trans_fixed)
        self.query = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.key = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.value = nn.Linear(self.embed_size, self.head_size, bias=False)
        
        # Lightweight MLP with LeakyReLU (ONNX-compatible)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, 2 * self.embed_size),
            nn.LeakyReLU(),  # ONNX-compatible activation
            nn.Linear(2 * self.embed_size, self.embed_size),
            nn.LeakyReLU()
        )
        
        # Multi-layer patch processor (transformer-inspired but ONNX-safe)
        self.patch_processor = nn.Sequential(
            # First processing stage - dimensional reduction
            nn.Linear(self.final_flat, self.final_flat // 2),
            nn.LeakyReLU(),
            
            # Second processing stage - feature mixing
            nn.Linear(self.final_flat // 2, self.final_flat // 2), 
            nn.LeakyReLU(),
            
            # Third processing stage - dimensional restoration
            nn.Linear(self.final_flat // 2, self.final_flat),
            nn.LeakyReLU()
        )
        
        # Additional feature enhancement layer (like attention but safe)
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.final_flat, self.embed_size * 4),  # Expand
            nn.LeakyReLU(),
            nn.Linear(self.embed_size * 4, self.final_flat),  # Contract
            nn.LeakyReLU()
        )
        
        # Final dense layer (handles patch_processor output which is final_flat)
        self.dense = nn.Sequential(
            nn.Linear(self.final_flat, self.h_size),
            nn.LeakyReLU()
        )
        
        # Weights will be initialized by default PyTorch initialization
    
    def attention(self, hidden):
        """Extremely simplified attention - no fancy stuff, just basic operations."""
        B, N, E = hidden.shape
        
        query = self.query(hidden)  # (B, N, head_size)
        key = self.key(hidden)      # (B, N, head_size)  
        value = self.value(hidden)  # (B, N, head_size)
        
        # Scaled dot-product attention
        attn_scores = query @ key.transpose(-2, -1)  # (B, N, N)
        attn_scores = attn_scores / (self.head_size ** 0.5)  # Scale
        attn_weights = attn_scores.softmax(dim=-1)  # (B, N, N)
        attn_output = attn_weights @ value  # (B, N, head_size)
        
        return attn_output
    
    def forward(self, visual_obs):
        # Mirror neurips: conditional permute for training (NHWC input)
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])  # To (B, C, H, W)

        # Patch extraction (like neurips conv_layers)
        hidden = self.patch_embeddings(visual_obs)  # (B, E, H/K, W/K)
        
        # MINIMAL APPROACH: Just flatten and process like neurips does
        hidden = hidden.reshape([-1, self.final_flat])  # (B, E*H*W) - SAFE, same as neurips
        
        # Use simple patch_processor (like neurips dense layer)
        return self.dense(hidden)  # (B, output_size)
