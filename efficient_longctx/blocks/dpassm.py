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

    def _compute_attention(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute windowed causal attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (seq_len, seq_len)

        Returns:
            Attention output tensor of shape (batch_size, seq_len, d_model)
        """
        B, T, d = x.shape
        d_h = d // self.n_heads

        # Project Q, K, V
        Q = self.W_Q(x).view(B, T, self.n_heads, d_h).transpose(1, 2)  # (B, h, T, d_h)
        K = self.W_K(x).view(B, T, self.n_heads, d_h).transpose(1, 2)  # (B, h, T, d_h)
        V = self.W_V(x).view(B, T, self.n_heads, d_h).transpose(1, 2)  # (B, h, T, d_h)

        # Scale by sqrt(d_h)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_h**0.5)

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores + mask

        # Softmax
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.drop(attention_weights)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, d)

        # Final linear projection
        attention_output = self.W_O(attention_output)
        attention_output = self.drop(attention_output)

        return attention_output

    def _compute_ssm(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SSM recurrent update.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state: Previous SSM state tensor of shape (batch_size, ssm_state_dim)

        Returns:
            Tuple of (ssm_output, new_state) where:
            - ssm_output: shape (batch_size, seq_len, d_model)
            - new_state: shape (batch_size, ssm_state_dim)
        """
        B, T, d = x.shape

        if state is None:
            # Initialize state to zeros
            state = torch.zeros(B, self.ssm_state_dim, device=x.device, dtype=x.dtype)

        # Compute input projections B(x_t) for SSM
        u = self.B(x)  # (B, T, ssm_state_dim)

        # Initialize output tensor
        outputs = []
        current_state = state

        # Sequential SSM update (more stable than parallel for training)
        for i in range(T):
            # SSM update: s_t = A * s_{t-1} + B * u_t
            current_state = self.A * current_state + u[:, i, :]
            outputs.append(current_state)

        # Stack outputs
        ssm_states = torch.stack(outputs, dim=1)  # (B, T, ssm_state_dim)

        # Compute SSM output C(s_t)
        ssm_output = self.C(ssm_states)  # (B, T, d_model)
        ssm_output = self.drop(ssm_output)

        return ssm_output, current_state

    def forward(
        self, x: torch.Tensor, ssm_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through DPASSM block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            ssm_state: Previous SSM state tensor of shape (batch_size, ssm_state_dim)

        Returns:
            Tuple of (output, new_ssm_state) where:
            - output: shape (batch_size, seq_len, d_model)
            - new_ssm_state: shape (batch_size, ssm_state_dim)
        """
        B, T, d = x.shape

        # Pre-layer normalization
        x_norm1 = self.ln1(x)

        # Build window mask for attention
        mask = self._build_window_mask(T, self.window_size, x.device, x.dtype)

        # Dual-path forward:
        # 1. Local attention path (windowed causal attention)
        attention_out = self._compute_attention(x_norm1, mask)

        # 2. SSM path (global state-space model)
        ssm_out, new_ssm_state = self._compute_ssm(x_norm1, ssm_state)

        # 3. Fuse attention and SSM outputs with learnable gate
        # Gate controls mixture: gate_norm * attention_out + (1 - gate_norm) * ssm_out
        gate_logits = self.gate(x_norm1)  # (B, T, d_model)
        gate_weights = torch.sigmoid(gate_logits)  # (B, T, d_model)

        # Fused output
        fused_out = gate_weights * attention_out + (1 - gate_weights) * ssm_out

        # Residual connection
        x = x + fused_out

        # Second sub-layer (post-layer norm + MLP)
        x_norm2 = self.ln2(x)
        mlp_out = self.mlp(x_norm2)
        mlp_out = self.drop(mlp_out)

        # Final residual connection
        output = x + mlp_out

        return output, new_ssm_state
