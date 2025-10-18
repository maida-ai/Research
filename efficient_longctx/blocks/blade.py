"""BLADE (Block-Local Attention with Per-Block State) implementation.

This module implements the BLADE attention mechanism that processes sequences
in chunks with local attention within each chunk and passes compact state
information between chunks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficient_longctx.utils.constants import FLASH_ATTN_AVAILABLE

from ..utils.gqa_kernels import gqa_attention

# Import FlashAttention if available
if FLASH_ATTN_AVAILABLE:
    from flash_attn import flash_attn_func
else:
    flash_attn_func = None  # pragma: no cover


class BLADEBlock(nn.Module):
    """Block-Local Attention with Per-Block State (BLADE) implementation.

    Splits sequences into chunks and runs exact attention per chunk while
    passing compact state information between chunks to maintain cross-chunk
    information flow.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        chunk_size: Maximum size of each chunk
        state_dim: Dimension of the per-chunk state
        m_global: Number of global tokens (0 to disable)
        dropout: Dropout probability
        **kwargs: All other keyword arguments are ignored
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        chunk_size: int,
        state_dim: int,
        m_global: int = 0,
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
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.m_global = m_global
        self.dropout_rate = dropout

        # Validate parameters
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.head_dim = d_model // n_heads

        # State projection layers
        self.state_proj = nn.Linear(d_model, state_dim)
        self.state_inject = nn.Linear(state_dim, d_model)

        # Global tokens (optional)
        if m_global > 0:
            self.global_tokens = nn.Parameter(torch.randn(m_global, d_model))
            # Nicer init for globals
            nn.init.normal_(self.global_tokens, mean=0.0, std=0.02)

        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # GQA: K/V projections use num_kv_heads instead of n_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # Layer normalization for stackable design
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Cache for masks keyed by (T, window, device)
        self._mask_cache: dict[
            tuple[int, tuple[int, int] | None, torch.device], torch.Tensor
        ] = {}

    def _causal_attn(
        self, x: torch.Tensor, window_size: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """Compute causal attention within a chunk using modern PyTorch APIs.

        Args:
            x: Input tensor [B, T, D]
            window_size: Optional (left, right) for sliding window attention.
                Note: Under causality, right is redundant. Only left window is used.

        Returns:
            Output tensor [B, T, D]
        """
        B, T, D = x.shape

        H, Dh = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)  # [B,H,T,Dh]

        # GQA: K and V have fewer heads
        k = (
            self.k_proj(x).view(B, T, self.num_kv_heads, Dh).transpose(1, 2)
        )  # [B,Hkv,T,Dh]
        v = (
            self.v_proj(x).view(B, T, self.num_kv_heads, Dh).transpose(1, 2)
        )  # [B,Hkv,T,Dh]

        # Set dropout based on training mode (SDPA applies dropout even in eval if > 0)
        dropout_p = self.dropout.p if self.training else 0.0

        # Try modern PyTorch SDPA first (recommended primary path)
        try:
            if window_size is None:
                # Pure causal attention - use GQA-aware computation
                o = gqa_attention(
                    Q=q,
                    K=k,
                    V=v,
                    dropout_p=dropout_p,
                    is_causal=True,
                )
            else:
                # Windowed + causal attention - build boolean mask
                left, _ = window_size
                i = torch.arange(T, device=q.device)[:, None]
                j = torch.arange(T, device=q.device)[None, :]

                # Causal, left window only (right is redundant under causality)
                band = (j <= i) & (j >= i - left)

                # Cache the mask for efficiency
                key = (T, window_size, x.device)
                if key in self._mask_cache:
                    attn_mask = self._mask_cache[key]
                else:
                    attn_mask = band.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
                    self._mask_cache[key] = attn_mask

                o = gqa_attention(
                    Q=q,
                    K=k,
                    V=v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=False,  # Must be False when using attn_mask
                )
        except (RuntimeError, NotImplementedError):
            # Fallback to FlashAttention direct call for older PyTorch
            o = None

            # Optional FA direct path (ensure shapes [B,T,H,Dh])
            if (
                FLASH_ATTN_AVAILABLE
                and flash_attn_func is not None
                and x.device.type == "cuda"
            ):
                try:
                    qfa = q.transpose(1, 2).contiguous()  # [B,T,H,Dh]
                    kfa = k.transpose(1, 2).contiguous()
                    vfa = v.transpose(1, 2).contiguous()
                    # Cast to half/bfloat16 if running FP32
                    if qfa.dtype == torch.float32:
                        qfa = qfa.half()
                        kfa = kfa.half()
                        vfa = vfa.half()
                    if window_size is None:
                        ofa = flash_attn_func(
                            qfa, kfa, vfa, dropout_p=dropout_p, causal=True
                        )
                    else:
                        left, _ = window_size
                        ofa = flash_attn_func(
                            qfa,
                            kfa,
                            vfa,
                            dropout_p=dropout_p,
                            causal=True,
                            window_size=(left, 0),  # Only left window for causal
                        )
                    # back to [B,H,T,Dh] and original dtype
                    o = ofa.transpose(1, 2).contiguous()
                    if o.dtype != q.dtype:
                        o = o.to(q.dtype)
                except Exception:
                    o = None

        # Manual fallback (very old stacks): do math in float32 for stability
        if o is None:
            qf = q.to(torch.float32)
            kf = k.to(torch.float32)
            vf = v.to(torch.float32)
            scale = Dh**-0.5
            scores = torch.matmul(qf * scale, kf.transpose(-2, -1))  # [B,H,T,T]
            # causal mask
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            mask = causal
            if window_size is not None:
                left, _ = window_size
                i = torch.arange(T, device=x.device)[:, None]
                j = torch.arange(T, device=x.device)[None, :]
                band_keep = (j <= i) & (j >= i - left)  # Causal + left window only
                mask = mask | (~band_keep)
            scores = scores.masked_fill(mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            # output dropout only (keep it simple)
            attn = F.dropout(attn, p=dropout_p, training=self.training)
            o = torch.matmul(attn, vf).to(q.dtype)

        o = o.transpose(1, 2).contiguous().view(B, T, D)  # [B,T,D]
        return self.out_proj(self.dropout(o))

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of BLADE block.

        Args:
            x: Input tensor of shape [B, T, D]
            state: Optional state tensor of shape [B, state_dim]

        Returns:
            Tuple of (output_tensor, updated_state)
            - output_tensor: Shape [B, T, D]
            - updated_state: Shape [B, state_dim]
        """
        B, T, D = x.shape

        # Initialize state if not provided
        if state is None:
            state = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)

        outputs = []
        current_state = state

        # Process sequence in chunks
        for start in range(0, T, self.chunk_size):
            end = min(T, start + self.chunk_size)
            chunk = x[:, start:end, :]  # [B,Tc,D]

            # Pre-LN for attention branch
            h = self.ln_attn(chunk)
            # Inject previous state (bias) into normalized activations
            h = h + self.state_inject(current_state).unsqueeze(1)

            # Prepend globals (shared across chunks)
            if self.m_global > 0:
                g = self.global_tokens.unsqueeze(0).expand(B, -1, -1)
                h_in = torch.cat([g, h], dim=1)  # [B,M+Tc,D]
            else:
                h_in = h

            y = self._causal_attn(h_in)  # [B,M+Tc,D]
            if self.m_global > 0:
                y = y[:, self.m_global :, :]  # [B,Tc,D]

            # Residual 1 (attention)
            chunk = chunk + self.dropout(y)
            # FFN (Pre-LN) + Residual 2
            chunk = chunk + self.dropout(self.ffn(self.ln_ffn(chunk)))

            # Store output
            outputs.append(chunk)

            # Update state from post-attn features (mean pool)
            current_state = self.state_proj(chunk.mean(dim=1))

        # Concatenate outputs
        y_full = torch.cat(outputs, dim=1)  # [B,T,D]
        return y_full, current_state
