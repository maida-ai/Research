"""BigBird baseline implementation.

This module implements a simplified BigBird-style attention mechanism
with local, random, and global attention patterns for baseline comparison.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficient_longctx.utils.constants import FLASH_ATTN_AVAILABLE

# Import FlashAttention if available
if FLASH_ATTN_AVAILABLE:
    from flash_attn import flash_attn_func
else:
    flash_attn_func = None  # pragma: no cover


class BigBirdBlock(nn.Module):
    """BigBird-style attention block with local, random, and global attention.

    Implements a simplified version of BigBird attention that uses:
    - Local attention within sliding windows
    - Random attention on a subset of tokens
    - Global attention on special tokens (first/last tokens)

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Size of local attention window
        n_random_tokens: Number of random tokens to attend to (per token)
        n_global_tokens: Number of global attention tokens (default: 2 for first/last)
        dropout: Dropout probability
        **kwargs: All other keyword arguments are ignored
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        n_random_tokens: int = 4,
        n_global_tokens: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.n_random_tokens = n_random_tokens
        self.n_global_tokens = n_global_tokens
        self.head_dim = d_model // n_heads
        self.dropout_rate = dropout

        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
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
        """Forward pass with BigBird-style attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state: Optional state tensor (ignored for BigBird)

        Returns:
            Tuple of (output tensor, state tensor)
            Output tensor of shape (batch_size, seq_len, d_model)
            State tensor is None (no state tracking for BigBird)
        """
        B, T, D = x.shape

        # Pre-LN
        h = self.ln1(x)

        # Compute QKV
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Create attention mask for local + random + global attention
        attn_mask = self._create_attention_mask(T, x.device)

        # Apply attention
        # Note: FlashAttention doesn't support arbitrary attention masks,
        # so we always use PyTorch SDPA for baseline blocks
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
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
        """Create attention mask for local + random + global attention pattern.

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

        # Random attention: each token attends to n_random_tokens random positions
        if self.n_random_tokens > 0:
            for i in range(seq_len):
                # Generate random positions (excluding local window and global tokens)
                local_start = max(0, i - self.window_size // 2)
                local_end = min(seq_len, i + self.window_size // 2 + 1)

                # Available positions for random attention
                available_positions = []
                for j in range(seq_len):
                    if j < local_start or j >= local_end:
                        # Exclude global tokens from random attention
                        if self.n_global_tokens >= 1 and j == 0:
                            continue
                        if self.n_global_tokens >= 2 and j == seq_len - 1:
                            continue
                        available_positions.append(j)

                # Sample random positions
                if available_positions:
                    n_samples = min(self.n_random_tokens, len(available_positions))
                    random_positions = random.sample(available_positions, n_samples)
                    for pos in random_positions:
                        mask[i, pos] = 0.0

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
