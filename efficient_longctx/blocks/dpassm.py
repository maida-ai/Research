import torch
import torch.nn as nn


class DPASSMBlock(nn.Module):
    """
    DP-ASSM Block: Dual-Path Attention + State Space Model

    Combines windowed local attention with a global state-space path,
    then fuses via a learnable gate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        ssm_state_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.ssm_state_dim = ssm_state_dim
        self.dropout_rate = dropout

        # Layer normalization layers (pre-LN for parallel paths)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # QKV and output projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Gate for fusing attention and SSM paths
        self.gate = nn.Linear(d_model, d_model)

        # SSM parameters
        self.A = nn.Parameter(torch.zeros(ssm_state_dim))  # Initialize small values
        with torch.no_grad():
            self.A.uniform_(-0.1, 0.1)

        self.B = nn.Linear(d_model, ssm_state_dim, bias=False)
        self.C = nn.Linear(ssm_state_dim, d_model, bias=False)

        # FFN/MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

        # Dropout
        self.drop = nn.Dropout(dropout)

    def _shape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape from (B, T, d) to (B, h, T, d_h) for multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Reshaped tensor of shape (batch_size, n_heads, seq_len, head_dim)
        """
        B, T, d = x.shape
        d_h = d // self.n_heads
        return x.view(B, T, self.n_heads, d_h).transpose(1, 2)

    def _build_window_mask(
        self, T: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build additive mask for windowed causal attention.

        Args:
            T: Sequence length
            W: Window size
            device: Device for tensor creation
            dtype: Data type for tensor creation

        Returns:
            Mask tensor of shape (T, T) with 0 for allowed positions, -inf for disallowed
        """
        # Create causal mask (j <= i)
        i_indices = torch.arange(T, device=device, dtype=torch.long)
        j_indices = torch.arange(T, device=device, dtype=torch.long)
        i_grid, j_grid = torch.meshgrid(i_indices, j_indices, indexing="ij")
        causal_mask = j_grid <= i_grid

        # Create window mask ((i - j) < W)
        window_mask = (i_grid - j_grid) < W

        # Combine masks
        combined_mask = causal_mask & window_mask

        # Create additive mask (0 for allowed, -inf for disallowed)
        mask = torch.zeros((T, T), device=device, dtype=dtype)
        mask = mask.masked_fill(~combined_mask, float("-inf"))

        return mask
