"""
GQA kernel capability detection and dispatch utilities.

This module provides utilities to detect available GQA-aware kernels
and dispatch to the appropriate implementation.
"""

import torch
import torch.nn.functional as F

from .constants import FLASH_ATTN_AVAILABLE


def detect_gqa_capabilities() -> dict:
    """
    Detect available GQA-aware kernel capabilities.

    Returns:
        dict: Capability flags for different GQA implementations
    """
    capabilities = {
        "sdpa_gqa": False,
        "flash_attn_gqa": False,
        "fallback_group_loop": True,  # Always available
    }

    # Check PyTorch SDPA GQA support
    # PyTorch 2.8+ with Flash kernel on CUDA supports GQA
    if torch.cuda.is_available():
        try:
            # Test if SDPA can handle different Q/K head counts
            B, Hq, Hkv, T, Dh = 1, 4, 2, 8, 64
            Q = torch.randn(B, Hq, T, Dh, device="cuda", dtype=torch.float16)
            K = torch.randn(B, Hkv, T, Dh, device="cuda", dtype=torch.float16)
            V = torch.randn(B, Hkv, T, Dh, device="cuda", dtype=torch.float16)

            # Try SDPA with different head counts
            _ = F.scaled_dot_product_attention(Q, K, V)
            capabilities["sdpa_gqa"] = True
        except Exception:
            capabilities["sdpa_gqa"] = False

    # Check FlashAttention availability
    capabilities["flash_attn_gqa"] = FLASH_ATTN_AVAILABLE

    return capabilities


def gqa_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    capabilities: dict | None = None,
) -> torch.Tensor:
    """
    Compute Grouped Query Attention using the best available kernel.

    Args:
        Q: Query tensor (B, Hq, T, Dh)
        K: Key tensor (B, Hkv, T, Dh)
        V: Value tensor (B, Hkv, T, Dh)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        capabilities: Optional capability dict (auto-detected if None)

    Returns:
        Output tensor (B, Hq, T, Dh)
    """
    if capabilities is None:
        capabilities = detect_gqa_capabilities()

    B, Hq, T, Dh = Q.shape
    Hkv = K.shape[1]
    G = Hq // Hkv

    assert Hq % Hkv == 0, f"n_heads ({Hq}) must be divisible by num_kv_heads ({Hkv})"
    assert Hkv >= 1, f"num_kv_heads ({Hkv}) must be >= 1"
    assert Hkv <= Hq, f"num_kv_heads ({Hkv}) cannot be greater than n_heads ({Hq})"

    # Prefer SDPA with GQA support
    if capabilities["sdpa_gqa"]:
        try:
            # Use SDPA kernel hints for optimal performance
            from torch.backends.cuda import sdp_kernel

            with sdp_kernel(
                enable_flash=True, enable_mem_efficient=False, enable_math=False
            ):
                return F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
        except Exception:
            # Fall back to group loop if SDPA fails
            pass

    # Try FlashAttention if available
    if capabilities["flash_attn_gqa"]:
        try:
            import flash_attn

            return flash_attn.flash_attn_func(
                Q,
                K,
                V,
                causal=is_causal,
                dropout_p=dropout_p,
            )
        except Exception:
            # Fall back to group loop if FlashAttention fails
            pass

    # Fallback: group loop without materialization
    outputs = []
    for g in range(Hkv):
        # Get query group
        qg = Q[:, g * G : (g + 1) * G]  # (B, G, T, Dh)

        # Broadcast K/V for this group (no materialization)
        kg = K[:, g : g + 1].expand(-1, G, -1, -1)  # (B, G, T, Dh) view/broadcast
        vg = V[:, g : g + 1].expand(-1, G, -1, -1)  # (B, G, T, Dh) view/broadcast

        # Compute attention for this group
        yg = F.scaled_dot_product_attention(
            qg,
            kg,
            vg,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )  # (B, G, T, Dh)

        outputs.append(yg)

    # Concatenate all groups
    return torch.cat(outputs, dim=1)  # (B, Hq, T, Dh)


def get_gqa_info() -> str:
    """Get information about available GQA capabilities."""
    caps = detect_gqa_capabilities()
    info = []

    if caps["sdpa_gqa"]:
        info.append("✓ PyTorch SDPA with GQA support")
    else:
        info.append("✗ PyTorch SDPA GQA not available")

    if caps["flash_attn_gqa"]:
        info.append("✓ FlashAttention with GQA support")
    else:
        info.append("✗ FlashAttention not available")

    info.append("✓ Group loop fallback available")

    return "\n".join(info)
