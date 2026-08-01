"""Longformer baseline implementation.

This module implements a simplified Longformer-style attention mechanism
with local attention windows and global attention tokens for baseline comparison.
"""

import torch
import torch.nn as nn

from efficient_longctx.utils.constants import FLASH_ATTN_AVAILABLE

from ..utils.gqa_kernels import gqa_attention

# Import FlashAttention if available
if FLASH_ATTN_AVAILABLE:
    from flash_attn import flash_attn_func
else:
    flash_attn_func = None  # pragma: no cover


class LongformerBlock(nn.Module):
    """Longformer-style attention block with local and global attention.

    Implements a simplified version of Longformer attention that uses:
    - Local attention within sliding windows
    - Global attention on special tokens (first/last tokens)

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Size of local attention window
        n_global_tokens: Number of global attention tokens (default: 2 for first/last)
        dropout: Dropout probability
        **kwargs: All other keyword arguments are ignored
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        n_global_tokens: int = 2,
        dropout: float = 0.1,
        num_kv_heads: int | None = None,
        **kwargs,
    ):
        super().__init__()

        # GQA setup: default to n_heads for backward compatibility
        if num_kv_heads is None:
            num_kv_heads = n_heads

        # Validate GQA parameters
        if num_kv_heads > n_heads:
            raise ValueError("num_kv_heads cannot be greater than n_heads")
        if n_heads % num_kv_heads != 0:
            raise ValueError("n_heads must be divisible by num_kv_heads for GQA")

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.n_global_tokens = n_global_tokens
        self.head_dim = d_model // n_heads
        self.dropout_rate = dropout

        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # GQA: K/V projections use num_kv_heads instead of n_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with Longformer-style attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state: Optional state tensor (ignored for Longformer)

        Returns:
            Tuple of (output tensor, state tensor)
            Output tensor of shape (batch_size, seq_len, d_model)
            State tensor is None (no state tracking for Longformer)
        """
        B, T, D = x.shape

        # Pre-LN
        h = self.ln1(x)

        # Compute QKV with proper GQA dimensions
        q = (
            self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        )  # (B, Hq, T, Dh)
        k = (
            self.k_proj(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        )  # (B, Hkv, T, Dh)
        v = (
            self.v_proj(h).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        )  # (B, Hkv, T, Dh)

        # Create attention mask for local + global attention
        attn_mask = self._create_attention_mask(T, x.device)

        # Use GQA-aware attention computation (no materialization)
        attn_out = gqa_attention(
            Q=q,
            K=k,
            V=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_rate if self.training else 0.0,
            is_causal=False,  # We handle causality in the mask
        )

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        attn_out = self.out_proj(attn_out)

        # Residual connection
        x = x + self.dropout(attn_out)

        # Feed-forward network
        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x, None  # Return output and None state

    def _create_attention_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create attention mask for local + global attention pattern.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Attention mask of shape (1, 1, seq_len, seq_len)
        """
        # Initialize mask with -inf (no attention)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)

        # Local attention: each token attends to its window
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0.0

        # Global attention: first and last tokens attend to all
        if self.n_global_tokens >= 1:
            mask[0, :] = 0.0  # First token attends to all
            mask[:, 0] = 0.0  # All tokens attend to first token

        if self.n_global_tokens >= 2 and seq_len > 1:
            mask[-1, :] = 0.0  # Last token attends to all
            mask[:, -1] = 0.0  # All tokens attend to last token

        # Add causal constraint: tokens can only attend to previous positions
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        mask = mask.masked_fill(causal_mask == 0, float("-inf"))

        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
