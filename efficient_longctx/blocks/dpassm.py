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
        self, x_in: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SSM recurrent update.

        Implementation follows specification:
        - Uses pre-normed input x_in from LN1(x)
        - Sequential loop over t = 0..T-1
        - s_t = alpha * s_{t-1} + B(x_in[:, t, :])
        - alpha = torch.tanh(self.A) to keep stable in (-1,1)
        - y_ssm[:, t, :] = C(s_t)

        Args:
            x_in: Pre-normalized input tensor of shape (batch_size, seq_len, d_model)
            state: Previous SSM state tensor of shape (batch_size, ssm_state_dim) or None

        Returns:
            Tuple of (ssm_output, new_state) where:
            - ssm_output: shape (batch_size, seq_len, d_model)
            - new_state: shape (batch_size, ssm_state_dim)
        """
        B, T, d = x_in.shape

        if state is None:
            # Initialize state to zeros as specified
            state = torch.zeros(
                B, self.ssm_state_dim, device=x_in.device, dtype=x_in.dtype
            )
        else:
            state = state.clone()

        # alpha = torch.tanh(self.A) to keep it stable in (-1,1) as specified
        alpha = torch.tanh(self.A)  # (d_s,)

        # Initialize output tensor
        outputs = []
        current_state = state

        # Sequential SSM update: loop over t = 0..T-1
        for t in range(T):
            # SSM update: s_t = alpha * s_{t-1} + B(x_in[:, t, :])
            current_state = alpha * current_state + self.B(x_in[:, t, :])  # (B, d_s)
            # SSM output: y_ssm[:, t, :] = C(s_t)
            outputs.append(self.C(current_state))  # (B, d_model)

        # Stack outputs to get (B, T, d_model)
        ssm_output = torch.stack(outputs, dim=1)

        # Apply dropout
        ssm_output = self.drop(ssm_output)

        return ssm_output, current_state

    def forward(
        self, x: torch.Tensor, ssm_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through DPASSM block.

        Implements the exact specification from issue #24:
        - Feature-wise gate: g = torch.sigmoid(self.gate(x_in))
        - Fuse: y = g * y_attn + (1.0 - g) * y_ssm
        - Residual 1: x = x + self.drop(y)
        - FFN with Pre-LN: x = x + self.drop(self.mlp(self.ln2(x)))

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            ssm_state: Previous SSM state tensor of shape (batch_size, ssm_state_dim)

        Returns:
            Tuple of (output, new_ssm_state) where:
            - output: shape (batch_size, seq_len, d_model)
            - new_ssm_state: shape (batch_size, ssm_state_dim)
        """
        B, T, d = x.shape

        # Keep x_in for gate computation (before normalization)
        x_in = x

        # Pre-layer normalization
        x_norm1 = self.ln1(x)

        # Build window mask for attention
        mask = self._build_window_mask(T, self.window_size, x.device, x.dtype)

        # Dual-path forward:
        # 1. Local attention path (windowed causal attention)
        y_attn = self._compute_attention(x_norm1, mask)

        # 2. SSM path (global state-space model)
        y_ssm, new_ssm_state = self._compute_ssm(x_norm1, ssm_state)

        # 3. Feature-wise gate: g = torch.sigmoid(self.gate(x_in))
        g = torch.sigmoid(self.gate(x_in))  # shape (B,T,d)

        # 4. Fuse: y = g * y_attn + (1.0 - g) * y_ssm
        y = g * y_attn + (1.0 - g) * y_ssm

        # 5. Residual 1: x = x + self.drop(y)
        x = x + self.drop(y)

        # 6. FFN with Pre-LN: x = x + self.drop(self.mlp(self.ln2(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))

        return x, new_ssm_state

    def forward_step(
        self, x_t: torch.Tensor, ssm_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Streaming inference step - process one token at a time.

        This method enables streaming inference by updating the model state
        incrementally one token at a time, without needing the full sequence.

        Args:
            x_t: Current token embedding of shape (batch_size, 1, d_model)
            ssm_state: Previous SSM state of shape (batch_size, ssm_state_dim)

        Returns:
            Tuple of (output_token, new_ssm_state) where:
            - output_token: shape (batch_size, 1, d_model)
            - new_ssm_state: shape (batch_size, ssm_state_dim)
        """
        B, T_in, d = x_t.shape
        assert T_in == 1, f"forward_step expects T=1, got T={T_in}"

        # Keep copy for residual connections
        x_in = x_t

        # Pre-layer normalization for the single token
        x_norm1 = self.ln1(x_t)  # (B, 1, d)

        # Handle attention path for single token
        # Since we have only 1 token, the window is effectively just itself
        # We still compute attention in case we want to query previous context
        # For now, we'll use self-attention over the single token
        Q = (
            self.W_Q(x_norm1)
            .view(B, 1, self.n_heads, d // self.n_heads)
            .transpose(1, 2)
        )  # (B, h, 1, d_h)
        K = (
            self.W_K(x_norm1)
            .view(B, 1, self.n_heads, d // self.n_heads)
            .transpose(1, 2)
        )  # (B, h, 1, d_h)
        V = (
            self.W_V(x_norm1)
            .view(B, 1, self.n_heads, d // self.n_heads)
            .transpose(1, 2)
        )  # (B, h, 1, d_h)

        d_h = d // self.n_heads
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            d_h**0.5
        )  # (B, h, 1, 1)

        # Apply dropout to attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (B, h, 1, 1)
        attention_weights = self.drop(attention_weights)

        # Apply attention to values
        y_attn = torch.matmul(attention_weights, V)  # (B, h, 1, d_h)
        y_attn = y_attn.transpose(1, 2).contiguous().view(B, 1, d)  # (B, 1, d)
        y_attn = self.W_O(y_attn)  # (B, 1, d)
        y_attn = self.drop(y_attn)

        # Handle SSM path for single token
        if ssm_state is None:
            ssm_state = torch.zeros(
                B, self.ssm_state_dim, device=x_t.device, dtype=x_t.dtype
            )
        else:
            ssm_state = ssm_state.clone()

        # SSM update: s_t = alpha * s_{t-1} + B(x_norm1[0])
        alpha = torch.tanh(self.A)  # (d_s,)
        current_state = alpha * ssm_state + self.B(x_norm1[:, 0, :])  # (B, d_s)

        # SSM output: y_ssm = C(s_t)
        y_ssm = self.C(current_state).unsqueeze(1)  # (B, 1, d_model)
        y_ssm = self.drop(y_ssm)

        # Feature-wise gate computation
        g = torch.sigmoid(self.gate(x_in))  # (B, 1, d)

        # Fuse attention and SSM outputs
        y = g * y_attn + (1.0 - g) * y_ssm  # (B, 1, d)

        # Apply residual connection
        x_out = x_in + self.drop(y)  # (B, 1, d)

        # Pre-LN for MLP
        x_norm2 = self.ln2(x_out)  # (B, 1, d)

        # Apply MLP and residual connection
        mlp_out = self.mlp(x_norm2)  # (B, 1, d)
        x_final = x_out + self.drop(mlp_out)  # (B, 1, d)

        return x_final, current_state
