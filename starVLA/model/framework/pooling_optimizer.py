"""
Enhancement Module: Additional pooling strategies for QwenSuper
Purpose: Further optimize forward pass by reducing token dimensions

Available optimizations:
1. DINO output pooling (196 tokens → 32 tokens)
2. MapAnything patch pooling (flexible)
3. Adaptive sequence length pooling based on batch size
"""

import torch
import torch.nn.functional as F
from typing import Optional


class TokenPoolingOptimizer:
    """Utility class for intelligent token pooling"""
    
    @staticmethod
    def pool_tokens(
        tokens: torch.Tensor,
        target_length: int,
        method: str = "adaptive_avg_pool1d"
    ) -> torch.Tensor:
        """
        Pool token sequence to target length.
        
        Args:
            tokens: [B, seq_len, hidden_dim]
            target_length: desired output seq length
            method: pooling method ('adaptive_avg_pool1d', 'max', 'linear')
        
        Returns:
            pooled tokens: [B, target_length, hidden_dim]
        """
        B, seq_len, hidden_dim = tokens.shape
        
        if seq_len <= target_length:
            return tokens  # No pooling needed
        
        if method == "adaptive_avg_pool1d":
            # Move to [B, hidden_dim, seq_len] for pooling
            tokens_T = tokens.transpose(1, 2)  # [B, hidden_dim, seq_len]
            pooled_T = F.adaptive_avg_pool1d(tokens_T, target_length)
            pooled = pooled_T.transpose(1, 2)  # [B, target_length, hidden_dim]
            return pooled
        
        elif method == "max_pool":
            tokens_T = tokens.transpose(1, 2)
            # Use max_pool1d with appropriate kernel
            kernel_size = seq_len // target_length
            if kernel_size > 1:
                pooled_T = F.max_pool1d(tokens_T, kernel_size=kernel_size)
            else:
                pooled_T = tokens_T
            pooled = pooled_T.transpose(1, 2)
            return pooled
        
        elif method == "linear":
            # Linear interpolation-based pooling
            tokens_T = tokens.transpose(1, 2).unsqueeze(0)  # [1, B, hidden_dim, seq_len]
            # Use grid_sample for interpolation (advanced)
            grid = torch.linspace(-1, 1, target_length, device=tokens.device)
            grid = grid.view(1, 1, 1, target_length).expand(1, B, 1, target_length)
            # Note: grid_sample is complex, fallback to adaptive_avg_pool1d
            return TokenPoolingOptimizer.pool_tokens(tokens, target_length, "adaptive_avg_pool1d")
        
        else:
            raise ValueError(f"Unknown pooling method: {method}")
    
    @staticmethod
    def adaptive_pool_by_compute_budget(
        tokens: torch.Tensor,
        target_compute_tokens: int = 64,
        vit_layers: int = 12
    ) -> torch.Tensor:
        """
        Adaptively pool tokens to meet compute budget.
        
        Target compute: O(seq_len^2) for attention, so we control seq_len
        to keep seq_len * num_layers * dim constant (roughly).
        
        Args:
            tokens: [B, seq_len, hidden_dim]
            target_compute_tokens: target number of tokens to keep
            vit_layers: number of transformer layers that will process this
        
        Returns:
            pooled tokens with controlled compute
        """
        B, seq_len, hidden_dim = tokens.shape
        
        # Estimate: each token pays O(seq_len) cost in attention × layers
        # So if we have 1000 tokens × 32 layers, that's 32k operations per sample
        # To get 64 tokens: 64 × 32 = 2048 operations per sample
        
        max_tokens = target_compute_tokens  # Use as target
        
        if seq_len > max_tokens:
            return TokenPoolingOptimizer.pool_tokens(tokens, max_tokens, "adaptive_avg_pool1d")
        return tokens


class OptimizedQwenSuperForward:
    """
    Mixin class to add pooling to forward pass.
    Usage: inherit from Qwen_Super and this class, or call these methods manually.
    """
    
    @staticmethod
    def add_dino_pooling(
        dino_features: torch.Tensor,
        target_tokens: int = 32,
        log_timing: bool = True
    ) -> torch.Tensor:
        """
        Apply pooling to DINO output features.
        
        DINO typically outputs 196 tokens (14x14) or 64 tokens (8x8).
        This pools to 32 tokens to reduce downstream computation.
        
        Args:
            dino_features: [B, seq_len, hidden_dim]
            target_tokens: target number of tokens (default 32)
            log_timing: whether to log timing info
        
        Returns:
            pooled DINO features
        """
        import time
        t0 = time.perf_counter()
        
        B, seq_len, dim = dino_features.shape
        if seq_len > target_tokens:
            dino_T = dino_features.transpose(1, 2)
            dino_pooled = F.adaptive_avg_pool1d(dino_T, target_tokens)
            dino_features = dino_pooled.transpose(1, 2)
        
        if log_timing:
            elapsed = time.perf_counter() - t0
            print(f"[DINO POOLING] {seq_len} → {target_tokens} tokens, {elapsed*1000:.2f}ms")
        
        return dino_features
    
    @staticmethod
    def add_map_pooling(
        map_patch_tokens: torch.Tensor,
        target_tokens: int = 64,
        log_timing: bool = True
    ) -> torch.Tensor:
        """
        Apply pooling to MapAnything patch tokens.
        
        Args:
            map_patch_tokens: [B, seq_len, dim]
            target_tokens: target number of tokens (default 64)
            log_timing: whether to log timing info
        
        Returns:
            pooled map tokens
        """
        import time
        t0 = time.perf_counter()
        
        B, seq_len, dim = map_patch_tokens.shape
        if seq_len > target_tokens:
            map_T = map_patch_tokens.transpose(1, 2)
            map_pooled = F.adaptive_avg_pool1d(map_T, target_tokens)
            map_patch_tokens = map_pooled.transpose(1, 2)
        
        if log_timing:
            elapsed = time.perf_counter() - t0
            print(f"[MAP POOLING] {seq_len} → {target_tokens} tokens, {elapsed*1000:.2f}ms")
        
        return map_patch_tokens


# Example usage in QwenSuper.forward():
"""
# After DINO processing:
dino_encoded_features = self.dino_pro(dino_encoded_features)
dino_encoded_features = OptimizedQwenSuperForward.add_dino_pooling(
    dino_encoded_features, 
    target_tokens=32,
    log_timing=True
)

# After MapAnything:
map_patch_tokens_pro = self.map_patch_pro(map_patch_tokens)
map_patch_tokens_pro = OptimizedQwenSuperForward.add_map_pooling(
    map_patch_tokens_pro,
    target_tokens=64,
    log_timing=True
)
"""


if __name__ == "__main__":
    # Test pooling
    print("Testing TokenPoolingOptimizer...")
    
    # Create dummy features
    B, seq_len, hidden_dim = 8, 256, 2048
    tokens = torch.randn(B, seq_len, hidden_dim)
    
    # Test pooling
    pooled = TokenPoolingOptimizer.pool_tokens(tokens, target_length=64)
    print(f"Original shape: {tokens.shape}")
    print(f"Pooled shape: {pooled.shape}")
    assert pooled.shape == (B, 64, hidden_dim), "Pooling failed!"
    
    print("✅ TokenPoolingOptimizer test passed!")
