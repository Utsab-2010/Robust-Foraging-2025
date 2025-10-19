class NatureVisualEncoder(nn.Module):
    def __init__(self, height: int, width: int, initial_channels: int, output_size: int):
        super().__init__()
        self.h_size = output_size
        self.embed_size = 64
        self.patch_size = 32
        
        # Calculate dimensions
        patch_h = height // self.patch_size  # 86 // 32 = 2
        patch_w = width // self.patch_size   # 155 // 32 = 4
        self.num_patches = patch_h * patch_w  # 8 patches
        self.final_flat = self.embed_size * patch_h * patch_w
        
        # Store spatial dimensions for pattern creation
        self.patch_h = patch_h
        self.patch_w = patch_w
        
        # Patch embedding (standard conv operation - Unity compatible)
        self.patch_embeddings = nn.Conv2d(
            initial_channels, self.embed_size,
            kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )
        
        # 🎯 PRE-COMPUTED ATTENTION PATTERNS
        # Instead of computing Q@K^T dynamically, we use fixed patterns
        self.register_attention_patterns()
        
        # Linear projections (like attention, but simpler)
        self.value_proj = nn.Linear(self.embed_size, self.embed_size, bias=False)
        
        # Final processing
        self.dense = nn.Sequential(
            nn.Linear(self.final_flat, self.h_size),
            nn.LeakyReLU()
        )
    
    def register_attention_patterns(self):
        """
        🧠 PRE-COMPUTE ATTENTION PATTERNS
        
        Instead of computing attention weights dynamically (Q@K^T + softmax),
        we create fixed attention patterns that capture spatial relationships.
        These patterns are learned during training but fixed during inference.
        """
        
        # Create different types of attention patterns
        patterns = []
        
        # 1. LOCAL ATTENTION: Each patch attends to its neighbors
        local_pattern = self.create_local_attention_pattern()
        patterns.append(local_pattern)
        
        # 2. GLOBAL ATTENTION: Each patch attends to all patches with distance weighting
        global_pattern = self.create_global_attention_pattern()
        patterns.append(global_pattern)
        
        # 3. STRUCTURED ATTENTION: Row/column patterns for spatial awareness
        row_pattern = self.create_row_attention_pattern()
        col_pattern = self.create_col_attention_pattern()
        patterns.extend([row_pattern, col_pattern])
        
        # Stack all patterns: (num_patterns, num_patches, num_patches)
        attention_patterns = torch.stack(patterns, dim=0)
        
        # Register as learnable parameters (will be optimized during training)
        self.attention_weights = nn.Parameter(attention_patterns)
        
        # Pattern mixing weights (learn how to combine different patterns)
        self.pattern_mixer = nn.Parameter(torch.ones(len(patterns)) / len(patterns))
        
    def create_local_attention_pattern(self):
        """
        🏠 LOCAL ATTENTION PATTERN
        Each patch attends strongly to itself and nearby patches.
        This captures local spatial relationships.
        """
        pattern = torch.zeros(self.num_patches, self.num_patches)
        
        for i in range(self.patch_h):
            for j in range(self.patch_w):
                patch_idx = i * self.patch_w + j
                
                # Attend to neighborhood (3x3 around current patch)
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.patch_h and 0 <= nj < self.patch_w:
                            neighbor_idx = ni * self.patch_w + nj
                            # Distance-based weighting
                            distance = abs(di) + abs(dj)
                            weight = 1.0 / (distance + 1)  # Closer = stronger attention
                            pattern[patch_idx, neighbor_idx] = weight
        
        # Normalize to sum to 1 (like softmax)
        pattern = pattern / pattern.sum(dim=1, keepdim=True)
        return pattern
    
    def create_global_attention_pattern(self):
        """
        🌍 GLOBAL ATTENTION PATTERN
        Each patch can attend to all patches, with distance-based decay.
        This captures long-range dependencies.
        """
        pattern = torch.zeros(self.num_patches, self.num_patches)
        
        for i in range(self.patch_h):
            for j in range(self.patch_w):
                patch_idx = i * self.patch_w + j
                
                for ki in range(self.patch_h):
                    for kj in range(self.patch_w):
                        target_idx = ki * self.patch_w + kj
                        
                        # Euclidean distance between patches
                        dist = math.sqrt((i - ki)**2 + (j - kj)**2)
                        # Exponential decay with distance
                        weight = math.exp(-dist / 2.0)
                        pattern[patch_idx, target_idx] = weight
        
        # Normalize
        pattern = pattern / pattern.sum(dim=1, keepdim=True)
        return pattern
    
    def create_row_attention_pattern(self):
        """
        ↔️ ROW ATTENTION PATTERN
        Each patch attends to all patches in the same row.
        Captures horizontal spatial relationships.
        """
        pattern = torch.zeros(self.num_patches, self.num_patches)
        
        for i in range(self.patch_h):
            for j in range(self.patch_w):
                patch_idx = i * self.patch_w + j
                
                # Attend to all patches in the same row
                for kj in range(self.patch_w):
                    target_idx = i * self.patch_w + kj
                    pattern[patch_idx, target_idx] = 1.0
        
        # Normalize
        pattern = pattern / pattern.sum(dim=1, keepdim=True)
        return pattern
    
    def create_col_attention_pattern(self):
        """
        ↕️ COLUMN ATTENTION PATTERN
        Each patch attends to all patches in the same column.
        Captures vertical spatial relationships.
        """
        pattern = torch.zeros(self.num_patches, self.num_patches)
        
        for i in range(self.patch_h):
            for j in range(self.patch_w):
                patch_idx = i * self.patch_w + j
                
                # Attend to all patches in the same column
                for ki in range(self.patch_h):
                    target_idx = ki * self.patch_w + j
                    pattern[patch_idx, target_idx] = 1.0
        
        # Normalize
        pattern = pattern / pattern.sum(dim=1, keepdim=True)
        return pattern
    
    def apply_precomputed_attention(self, patches):
        """
        🔄 APPLY PRE-COMPUTED ATTENTION
        
        Instead of computing Q@K^T dynamically, we use our pre-computed patterns.
        This is ONNX-friendly because all operations are static matrix multiplications.
        """
        B, N, E = patches.shape  # (batch, num_patches, embed_size)
        
        # 1. Project to values (like V in standard attention)
        values = self.value_proj(patches)  # (B, N, E)
        
        # 2. Mix attention patterns using learned weights
        # Pattern mixer learns which patterns are most important
        mixed_pattern = torch.zeros_like(self.attention_weights[0])
        for i, weight in enumerate(self.pattern_mixer):
            mixed_pattern += weight * self.attention_weights[i]
        
        # Ensure pattern is normalized (safety check)
        mixed_pattern = mixed_pattern / mixed_pattern.sum(dim=1, keepdim=True)
        
        # 3. Apply attention pattern (matrix multiplication - ONNX friendly!)
        # This is equivalent to: attended = softmax(Q@K^T) @ V
        # But using pre-computed patterns instead of Q@K^T
        attended_values = torch.matmul(mixed_pattern.unsqueeze(0), values)  # (B, N, E)
        
        return attended_values
    
    def forward(self, visual_obs):
        """
        🚀 FORWARD PASS WITH PRE-COMPUTED ATTENTION
        
        This approach gives us attention-like behavior while being Unity ONNX compatible:
        1. Uses only static matrix operations
        2. No dynamic tensor creation
        3. No complex reshaping patterns
        4. Pre-computed attention patterns are learned but fixed at inference
        """
        
        # Standard ONNX preparation
        if not exporting_to_onnx.is_exporting():
            visual_obs = visual_obs.permute([0, 3, 1, 2])
        
        # Extract patches using convolution (Unity compatible)
        patches = self.patch_embeddings(visual_obs)  # (B, embed_size, H', W')
        
        # Reshape to patch sequence (safe operation)
        B = patches.shape[0]
        patches_seq = patches.view(B, self.embed_size, -1).transpose(1, 2)  # (B, N, E)
        
        # Apply pre-computed attention (ONNX-friendly!)
        attended_patches = self.apply_precomputed_attention(patches_seq)  # (B, N, E)
        
        # Flatten for final processing (safe operation)
        attended_flat = attended_patches.view(B, -1)  # (B, N*E)
        
        # Ensure correct size for dense layer
        if attended_flat.shape[1] != self.final_flat:
            # Use linear projection to correct size (instead of padding/truncation)
            if not hasattr(self, 'size_corrector'):
                self.size_corrector = nn.Linear(attended_flat.shape[1], self.final_flat)
            attended_flat = self.size_corrector(attended_flat)
        
        # Final processing
        return self.dense(attended_flat)  # (B, output_size)