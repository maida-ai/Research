import torch
import torch.nn as nn
import torch.nn.functional as F

# Enable TF32 for better performance on Ampere/Hopper GPUs
torch.set_float32_matmul_precision("medium")


class DPASSMBlock(nn.Module):
    """
    DP-ASSM Block: Dual-Path Attention + State Space Model

    Combines windowed local attention with a global state-space path,
    then fuses via a learnable gate.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Window size for attention
        ssm_state_dim: Dimension of the SSM state
        dropout: Dropout probability
        max_mask_T: Maximum sequence length for cached mask approach. Beyond this,
                   blockwise SDPA is used to avoid O(T×T) memory allocations.
        ssm_impl: SSM implementation ("naive", "tile_scan", or "auto")
        tile_size: Tile size for tile-scan SSM implementation
        threshold_tokens: Minimum batch*sequence tokens for auto mode to use tile-scan
        **kwargs: All other keyword arguments are ignored
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        ssm_state_dim: int,
        dropout: float = 0.1,
        max_mask_T: int = 4096,
        ssm_impl: str = "naive",
        tile_size: int = 256,
        threshold_tokens: int = 1024,
        **kwargs,
    ):
        super().__init__()

        # Head dimension validation
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.ssm_state_dim = ssm_state_dim
        self.dropout_rate = dropout
        self.max_mask_T = max_mask_T
        self.ssm_impl = ssm_impl
        self.tile_size = tile_size
        self.threshold_tokens = threshold_tokens

        # KV cache mode for streaming: "cat" (default), "ring_win", or "ring2x"
        # Kept behind kwargs for backward compatibility
        self.kv_cache_mode = kwargs.get("kv_cache_mode", "cat")

        # Mask cache for performance
        self._mask_cache = {}

        # Layer normalization layers (pre-LN for parallel paths)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Split projections with CUDA stream overlap for better performance
        # W_qkv: QKV projections (bias=False for better cublasLt algo selection)
        # W_ug: SSM input + gate projections (bias=True for gate initialization)
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_ug = nn.Linear(d_model, ssm_state_dim + d_model, bias=True)
        self.W_O = nn.Linear(d_model, d_model)

        # Keep B layer for backward compatibility (fallback when u is None)
        self.B = nn.Linear(d_model, ssm_state_dim, bias=False)

        # SSM parameters
        self.A = nn.Parameter(torch.zeros(ssm_state_dim))  # Initialize small values
        with torch.no_grad():
            self.A.uniform_(-0.1, 0.1)

        # Keep B layer for backward compatibility (fallback when u is None)
        self.B = nn.Linear(d_model, ssm_state_dim, bias=False)
        self.C = nn.Linear(ssm_state_dim, d_model, bias=False)

        # FFN/MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

        # Dropout
        self.drop = nn.Dropout(dropout)

        # Attention state type alias
        self.AttnState = tuple[torch.Tensor, torch.Tensor]  # (K_cache, V_cache)

    def _init_kv_buffers(
        self, B: int, H: int, W: int, Dh: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Initialize ring buffers and staging windows for ring_win mode."""
        K_buf = torch.empty(B, H, W, Dh, device=device, dtype=dtype)
        V_buf = torch.empty(B, H, W, Dh, device=device, dtype=dtype)
        K_win = torch.empty(B, H, W, Dh, device=device, dtype=dtype)
        V_win = torch.empty(B, H, W, Dh, device=device, dtype=dtype)
        idx = torch.zeros((), dtype=torch.int64, device=device)
        filled = torch.zeros((), dtype=torch.int64, device=device)
        return (K_buf, V_buf, K_win, V_win, idx, filled)

    def _init_kv_buffers_2x(
        self, B: int, H: int, W: int, Dh: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Initialize 2W rolling write buffers for ring2x mode.

        Keep write pointer and filled counters as Python ints to avoid GPU syncs.
        """
        K_buf = torch.empty(B, H, 2 * W, Dh, device=device, dtype=dtype)
        V_buf = torch.empty(B, H, 2 * W, Dh, device=device, dtype=dtype)
        wp: int = W - 1  # next write will go to wp+1
        filled: int = 0  # logical window length, capped at W
        return (K_buf, V_buf, wp, filled)

    def _get_window_mask(self, T: int, W: int, device: torch.device) -> torch.Tensor:
        """Get cached boolean window mask for SDPA.

        Args:
            T: Sequence length
            W: Window size
            device: Device for tensor creation

        Returns:
            Boolean mask of shape (1, 1, T, T) where True = allowed positions
        """
        key = (T, W, device)
        if key in self._mask_cache:
            return self._mask_cache[key]

        i = torch.arange(T, device=device)[:, None]
        j = torch.arange(T, device=device)[None, :]
        band = (j <= i) & ((i - j) < W)  # causal + left window
        mask = band.view(1, 1, T, T)  # bool mask: True=allowed
        self._mask_cache[key] = mask
        return mask

    def _compute_attention(self, x: torch.Tensor, _unused=None) -> torch.Tensor:
        """
        Compute windowed causal attention using optimized SDPA.

        Uses PyTorch's `scaled_dot_product_attention` for optimal performance.
        This provides 2-4x speedup and 30-50% memory reduction compared to
        manual attention computation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            _unused: Unused parameter for compatibility

        Returns:
            Attention output tensor of shape (batch_size, seq_len, d_model)
        """
        B, T, d = x.shape
        d_h = d // self.n_heads

        # Split projections for attention computation
        qkv = self.W_qkv(x)  # (B, T, 3*d)
        Q, K, V = qkv.split(d, dim=-1)  # Each: (B, T, d)

        # Reshape Q, K, V for attention
        Q = (
            Q.view(B, T, self.n_heads, d_h).transpose(1, 2).contiguous()
        )  # (B, h, T, d_h)
        K = (
            K.view(B, T, self.n_heads, d_h).transpose(1, 2).contiguous()
        )  # (B, h, T, d_h)
        V = (
            V.view(B, T, self.n_heads, d_h).transpose(1, 2).contiguous()
        )  # (B, h, T, d_h)

        # Use cached bool mask and causal fast-path
        if self.window_size >= T:
            attn_mask, is_causal = None, True
        elif T <= self.max_mask_T:
            attn_mask = self._get_window_mask(T, self.window_size, x.device)
            is_causal = False
        else:
            # Blockwise SDPA for large T to avoid O(T×T) memory allocations
            return self._compute_attention_blockwise(Q, K, V, x.device)

        # Use single source of truth for dropout
        p = self.drop.p if self.training else 0.0

        # Use PyTorch's optimized scaled_dot_product_attention
        attention_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, dropout_p=p, is_causal=is_causal
        )

        # Reshape back to (B, T, d)
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, d)

        # Final linear projection
        attention_output = self.W_O(attention_output)
        attention_output = self.drop(attention_output)

        return attention_output

    def _compute_attention_from_qkv(
        self, x: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute windowed causal attention using pre-computed Q, K, V tensors.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) - used for shape info
            Q: Query tensor of shape (batch_size, seq_len, d_model)
            K: Key tensor of shape (batch_size, seq_len, d_model)
            V: Value tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Attention output tensor of shape (batch_size, seq_len, d_model)
        """
        B, T, d = x.shape
        d_h = d // self.n_heads

        # Reshape Q, K, V for attention
        Q = (
            Q.view(B, T, self.n_heads, d_h).transpose(1, 2).contiguous()
        )  # (B, h, T, d_h)
        K = (
            K.view(B, T, self.n_heads, d_h).transpose(1, 2).contiguous()
        )  # (B, h, T, d_h)
        V = (
            V.view(B, T, self.n_heads, d_h).transpose(1, 2).contiguous()
        )  # (B, h, T, d_h)

        # Use cached bool mask and causal fast-path
        if self.window_size >= T:
            attn_mask, is_causal = None, True
        elif T <= self.max_mask_T:
            attn_mask = self._get_window_mask(T, self.window_size, x.device)
            is_causal = False
        else:
            # Blockwise SDPA for large T to avoid O(T×T) memory allocations
            return self._compute_attention_blockwise(Q, K, V, x.device)

        # Use single source of truth for dropout
        p = self.drop.p if self.training else 0.0

        # Use PyTorch's optimized scaled_dot_product_attention
        attention_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, dropout_p=p, is_causal=is_causal
        )

        # Reshape back to (B, T, d)
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, d)

        # Final linear projection
        attention_output = self.W_O(attention_output)
        attention_output = self.drop(attention_output)

        return attention_output

    def _compute_attention_blockwise(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Compute attention using blockwise SDPA to avoid O(T×T) memory allocations.

        This method processes attention in sliding blocks over the target time dimension,
        where each block attends to its left window only. This avoids creating a full
        T×T attention mask for large sequences.

        Args:
            Q: Query tensor of shape (B, h, T, d_h)
            K: Key tensor of shape (B, h, T, d_h)
            V: Value tensor of shape (B, h, T, d_h)
            device: Device for tensor operations

        Returns:
            Attention output tensor of shape (B, T, d_model)
        """
        B, h, T, d_h = Q.shape
        W = self.window_size

        # Use single source of truth for dropout
        p = self.drop.p if self.training else 0.0

        # Choose block size (e.g., BS = W or 2W)
        BS = min(2 * W, T)
        out_chunks = []

        # Process in sliding blocks over target time dimension
        for t0 in range(0, T, BS):
            t1 = min(T, t0 + BS)
            # Keys/values span [max(0,t1-W), t1)
            ks = max(0, t1 - W)

            Qb = Q[:, :, t0:t1, :]  # (B, h, BS, d_h)
            Kb = K[:, :, ks:t1, :]  # (B, h, KB, d_h)
            Vb = V[:, :, ks:t1, :]

            # Build a small causal mask inside the block:
            # target positions are t0..t1-1, allowed keys <= each position
            # local indices:
            Tb = t1 - t0
            Kb_len = t1 - ks
            i = torch.arange(Tb, device=device)[:, None]
            j = torch.arange(Kb_len, device=device)[None, :]

            # position of key j corresponds to global time ks+j
            # we need j_global <= i_global -> (ks+j) <= (t0+i) => j <= i + (t0-ks)
            shift = t0 - ks
            band = j <= (i + shift)
            mask_b = band.view(1, 1, Tb, Kb_len)

            out_b = F.scaled_dot_product_attention(
                Qb, Kb, Vb, attn_mask=mask_b, dropout_p=p, is_causal=False
            )
            out_chunks.append(out_b)

        # Concatenate all blocks
        out = torch.cat(out_chunks, dim=2)  # (B, h, T, d_h)

        # Reshape back to (B, T, d)
        attention_output = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Final linear projection
        attention_output = self.W_O(attention_output)
        attention_output = self.drop(attention_output)

        return attention_output

    def _compute_ssm(
        self,
        x_in: torch.Tensor,
        state: torch.Tensor | None = None,
        u: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SSM recurrent update with preallocated outputs.

        Implementation follows specification:
        - Uses pre-normed input x_in from LN1(x)
        - Sequential loop over t = 0..T-1
        - s_t = alpha * s_{t-1} + u_t (where u_t comes from fused projection)
        - alpha = torch.tanh(self.A) to keep stable in (-1,1)
        - y_ssm[:, t, :] = C(s_t)

        Args:
            x_in: Pre-normalized input tensor of shape (batch_size, seq_len, d_model)
            state: Previous SSM state tensor of shape (batch_size, ssm_state_dim) or None
            u: Pre-computed SSM input tensor of shape (batch_size, seq_len, ssm_state_dim) or None

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

        # Use provided u or compute it (for backward compatibility)
        if u is None:
            # Fallback: compute u using B(x_in) - this should not happen in normal fused flow
            u = self.B(x_in)  # (B, T, d_s)

        # Choose implementation based on ssm_impl and sequence length
        if self.ssm_impl == "naive" or T <= self.tile_size:
            # Use naive implementation for short sequences or when explicitly requested
            return self._compute_ssm_naive(x_in, state, u, alpha)
        elif self.ssm_impl == "auto":
            # Auto-select: use tile-scan if sequence is long enough and batch*seq >= threshold
            use_tile_scan = (T >= self.tile_size) and (B * T >= self.threshold_tokens)
            if use_tile_scan:
                return self._compute_ssm_tile_scan(x_in, state, u, alpha)
            else:
                return self._compute_ssm_naive(x_in, state, u, alpha)
        else:
            # Use tile-scan implementation for longer sequences
            return self._compute_ssm_tile_scan(x_in, state, u, alpha)

    def _compute_ssm_naive(
        self,
        x_in: torch.Tensor,
        state: torch.Tensor,
        u: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Naive SSM implementation with sequential loop."""
        B, T, d = x_in.shape

        # Preallocate output tensor to avoid Python list overhead
        y_ssm = torch.empty(B, T, self.d_model, device=x_in.device, dtype=x_in.dtype)
        current_state = state

        # Sequential SSM update: loop over t = 0..T-1
        for t in range(T):
            # SSM update: s_t = alpha * s_{t-1} + u_t
            current_state = alpha * current_state + u[:, t, :]  # (B, d_s)
            # SSM output: y_ssm[:, t, :] = C(s_t)
            y_ssm[:, t, :] = self.C(current_state)  # (B, d_model)

        # Apply dropout
        y_ssm = self.drop(y_ssm)

        return y_ssm, current_state

    def _compute_ssm_tile_scan(
        self,
        x_in: torch.Tensor,
        state: torch.Tensor,
        u: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized tile-scan SSM:
          - No per-tile weight construction in Python
          - Prefix-scan via cumprod/cumsum
          - Batched readout C(s) (once per tile or once total)
          - fp32 math for SSM for stability
        """
        B, T, _ = x_in.shape
        d_s = self.ssm_state_dim
        C = self.tile_size
        device = x_in.device
        dtype_model = x_in.dtype

        # ---- fp32 math inside SSM path for stability ----
        u32 = u.float()  # (B,T,d_s)
        alpha32 = alpha.float()  # (d_s,)
        s0 = state.float()  # (B,d_s)

        # ---- reshape into tiles (pad if needed) ----
        N = (T + C - 1) // C  # number of tiles
        pad = N * C - T
        if pad:
            u32 = F.pad(u32, (0, 0, 0, pad))  # (B, N*C, d_s)

        u_tiles = u32.view(B, N, C, d_s).contiguous()  # (B, N, C, d_s)

        # ---- per-tile weights & A_tile (vectorized; cache-friendly) ----
        # weights w[i] = alpha**(C-1-i) for full tiles in fp32
        # build using logs (numerically friendlier)
        idx = torch.arange(C - 1, -1, -1, device=device, dtype=torch.float32).unsqueeze(
            -1
        )  # (C,1)
        loga = torch.log(alpha32.clamp_min(1e-12))  # (d_s,)
        w_full = torch.exp(idx * loga.unsqueeze(0))  # (C, d_s) fp32

        # A_tile for full tiles; handle last (short) tile separately later
        A_full = torch.exp(C * loga)  # (d_s,)

        # ---- per-tile summaries (vectorized) ----
        # Full tiles first:
        if N > 1 or pad == 0:
            # Weighted sum across C: (B,N,C,d_s) * (C,d_s) -> (B,N,d_s)
            b_tiles = (u_tiles * w_full.view(1, 1, C, d_s)).sum(dim=2)  # (B,N,d_s)
            A_tiles = A_full.view(1, 1, d_s).expand(B, N, d_s).contiguous()  # (B,N,d_s)
        else:
            # N == 1 is handled below; this branch rarely triggers
            b_tiles = (u_tiles * w_full.view(1, 1, C, d_s)).sum(dim=2)
            A_tiles = A_full.view(1, 1, d_s).expand(B, N, d_s)

        # If the last tile is short (pad>0), correct A/b for that tile length L_last
        if pad:
            L_last = C - pad
            # Recompute weights for the last tile only (length L_last)
            idx_last = torch.arange(
                L_last - 1, -1, -1, device=device, dtype=torch.float32
            ).unsqueeze(-1)
            w_last = torch.exp(idx_last * loga.unsqueeze(0))  # (L_last, d_s)
            b_last = (u_tiles[:, -1, :L_last, :] * w_last.unsqueeze(0)).sum(
                dim=1
            )  # (B,d_s)
            b_tiles[:, -1, :] = b_last
            A_last = torch.exp(L_last * loga)  # (d_s,)
            A_tiles[:, -1, :] = A_last

        # ---- prefix-scan over tiles (associative combine) ----
        # suf[k] = prod_{j=k+1..N-1} A_tiles[j]  (elementwise)
        suf = torch.empty_like(A_tiles)  # (B,N,d_s)
        suf[:, -1, :] = 1.0
        if N > 1:
            # reverse cumprod on dim=1 then reverse back (excluding last position)
            cumprod_result = torch.cumprod(A_tiles[:, :-1, :], dim=1)
            suf[:, :-1, :] = torch.flip(cumprod_result, dims=[1])

        # b_prefix[k] = sum_{i=0..k} b_tiles[i] * prod_{j=i+1..k} A_tiles[j]
        b_weighted = b_tiles * suf  # (B,N,d_s)
        b_prefix = torch.cumsum(b_weighted, dim=1)  # (B,N,d_s)

        # A_prefix[k] = prod_{i=0..k} A_tiles[i]
        A_prefix = torch.cumprod(A_tiles, dim=1)  # (B,N,d_s)

        # carry-in state for each tile k:
        # s_in[k] = A_prefix[k-1]*s0 + b_prefix[k-1], with s_in[0]=s0
        s_in = torch.empty(B, N, d_s, device=device, dtype=torch.float32)
        s_in[:, 0, :] = s0
        if N > 1:
            s_in[:, 1:, :] = (A_prefix[:, :-1, :] * s0.unsqueeze(1)) + b_prefix[
                :, :-1, :
            ]

        # ---- materialize per-token states (short in-tile loop, but batch readout) ----
        # We store per-step states in fp32, then do ONE matmul per tile via self.C
        y_states = torch.empty(B, T, d_s, device=device, dtype=torch.float32)
        for k in range(N):
            start = k * C
            end = min(start + C, T)
            s_k = s_in[:, k, :]  # (B,d_s)
            # write a temporary buffer for states in this tile to do a single matmul
            tile_len = end - start
            s_buf = torch.empty(B, tile_len, d_s, device=device, dtype=torch.float32)
            for i in range(tile_len):
                s_k = alpha32 * s_k + u32[:, start + i, :]
                s_buf[:, i, :] = s_k
            # single readout for the whole tile
            y_states[:, start:end, :] = s_buf
            # keep final state for return
            if k == N - 1:
                s_last = s_k

        # ---- batched readout C(s) across the full sequence; cast back ----
        # One matmul instead of T matmuls:
        y_ssm = self.C(y_states.to(dtype_model))  # (B,T,d_model)
        y_ssm = self.drop(y_ssm)
        return y_ssm, s_last.to(dtype_model)

    def forward(
        self, x: torch.Tensor, ssm_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through DPASSM block.

        Implements the exact specification from issue #24:
        - Feature-wise gate: g = torch.sigmoid(self.gate(x_norm1))
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

        # Pre-layer normalization
        x_norm1 = self.ln1(x)

        # Split projections with CUDA stream overlap for better performance
        if torch.cuda.is_available():
            # Launch QKV and UG projections on separate streams for overlap
            s_qkv, s_ug = torch.cuda.Stream(), torch.cuda.Stream()

            with torch.cuda.stream(s_qkv):
                qkv = self.W_qkv(x_norm1)  # (B, T, 3*d)

            with torch.cuda.stream(s_ug):
                ug = self.W_ug(x_norm1)  # (B, T, ssm_state_dim + d)

            # Sync both streams before proceeding
            e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
            e1.record(s_qkv)
            e2.record(s_ug)
            torch.cuda.current_stream().wait_event(e1)
            torch.cuda.current_stream().wait_event(e2)
        else:
            # Fallback for CPU
            qkv = self.W_qkv(x_norm1)
            ug = self.W_ug(x_norm1)

        # Split outputs
        Q, K, V = qkv.split(d, dim=-1)  # Each: (B, T, d)
        u, gate_pre = ug.split(
            [self.ssm_state_dim, d], dim=-1
        )  # u: (B, T, ssm_state_dim), gate_pre: (B, T, d)

        # Dual-path forward with CUDA stream overlap for better performance
        if torch.cuda.is_available():
            # Create separate streams for attention and SSM computation
            s_attn, s_ssm = torch.cuda.Stream(), torch.cuda.Stream()

            with torch.cuda.stream(s_attn):
                # 1. Local attention path (windowed causal attention)
                y_attn = self._compute_attention_from_qkv(x_norm1, Q, K, V)

            with torch.cuda.stream(s_ssm):
                # 2. SSM path (global state-space model) - pass u directly
                y_ssm, new_ssm_state = self._compute_ssm(x_norm1, ssm_state, u)

            # Synchronize both streams before fusion
            e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
            e1.record(s_attn)
            e2.record(s_ssm)
            torch.cuda.current_stream().wait_event(e1)
            torch.cuda.current_stream().wait_event(e2)
        else:
            # Fallback for CPU: sequential execution
            # 1. Local attention path (windowed causal attention)
            y_attn = self._compute_attention_from_qkv(x_norm1, Q, K, V)

            # 2. SSM path (global state-space model) - pass u directly
            y_ssm, new_ssm_state = self._compute_ssm(x_norm1, ssm_state, u)

        # 3. Feature-wise gate: g = torch.sigmoid(gate_pre) - use pre-computed gate_pre
        g = torch.sigmoid(gate_pre)  # shape (B,T,d)

        # 4. Fuse: y = g * y_attn + (1.0 - g) * y_ssm
        y = g * y_attn + (1.0 - g) * y_ssm

        # 5. Residual 1: x = x + self.drop(y)
        x = x + self.drop(y)

        # 6. FFN with Pre-LN: x = x + self.drop(self.mlp(self.ln2(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))

        return x, new_ssm_state

    def forward_step(
        self,
        x_t: torch.Tensor,
        ssm_state: torch.Tensor | None = None,
        attn_state: tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, int]
        | tuple[torch.Tensor, torch.Tensor, int, int]
        | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, int]
        | tuple[torch.Tensor, torch.Tensor, int, int],
    ]:
        """
        Streaming inference step - process one token at a time with KV cache.

        This method enables streaming inference by updating the model state
        incrementally one token at a time, with proper attention to previous tokens
        via KV cache.

        Args:
            x_t: Current token embedding of shape (batch_size, 1, d_model)
            ssm_state: Previous SSM state of shape (batch_size, ssm_state_dim)
            attn_state: Previous attention state (K_cache, V_cache) of shape
                       (batch_size, n_heads, cache_len, head_dim)

        Returns:
            Tuple of (output_token, new_ssm_state, new_attn_state) where:
            - output_token: shape (batch_size, 1, d_model)
            - new_ssm_state: shape (batch_size, ssm_state_dim)
            - new_attn_state: (K_cache, V_cache) with updated cache
        """
        B, T_in, d = x_t.shape
        assert T_in == 1, f"forward_step expects T=1, got T={T_in}"
        d_h = d // self.n_heads

        # Pre-layer normalization for the single token
        x_norm1 = self.ln1(x_t)  # (B, 1, d)

        # Split projections for streaming inference
        qkv = self.W_qkv(x_norm1)  # (B, 1, 3*d)
        ug = self.W_ug(x_norm1)  # (B, 1, ssm_state_dim + d)

        # Split outputs
        Q, K, V = qkv.split(d, dim=-1)  # Each: (B, 1, d)
        u, gate_pre = ug.split(
            [self.ssm_state_dim, d], dim=-1
        )  # u: (B, 1, ssm_state_dim), gate_pre: (B, 1, d)

        # Reshape Q, K, V for attention
        Q = Q.view(B, 1, self.n_heads, d_h).transpose(1, 2)  # (B, h, 1, d_h)
        K_new = K.view(B, 1, self.n_heads, d_h).transpose(1, 2)
        V_new = V.view(B, 1, self.n_heads, d_h).transpose(1, 2)

        # Update/read KV cache
        if self.kv_cache_mode == "ring2x":
            # 2W rolling write buffer mode: attn_state is (K_buf, V_buf, wp, filled)
            W = self.window_size
            if attn_state is None or len(attn_state) != 4:
                K_buf, V_buf, wp, filled = self._init_kv_buffers_2x(
                    B, self.n_heads, W, d_h, x_t.device, x_t.dtype
                )
            else:
                K_buf, V_buf, wp, filled = attn_state  # type: ignore[misc]

            # Advance write pointer (Python int math; no .item())
            wp += 1
            twoW = 2 * W
            if wp >= twoW:
                # Bulk copy about once per W steps; no autograd
                with torch.no_grad():
                    K_buf[:, :, 0:W].copy_(K_buf[:, :, W:twoW])
                    V_buf[:, :, 0:W].copy_(V_buf[:, :, W:twoW])
                wp = W

            # Write new KV (no autograd)
            with torch.no_grad():
                K_buf[:, :, wp] = K_new[:, :, 0]
                V_buf[:, :, wp] = V_new[:, :, 0]

            # Logical window length (Python int; no tensor creation)
            filled = filled + 1 if filled + 1 < W else W
            L = filled

            # Contiguous slice for attention: [wp-L+1 : wp+1)
            start = wp - (L - 1)
            stop = wp + 1
            K_win = K_buf[:, :, start:stop]  # (B,H,L,Dh) contiguous view
            V_win = V_buf[:, :, start:stop]

            # Call SDPA with contiguous buffers (fast path preserved)
            p = self.drop.p if self.training else 0.0
            y_attn = F.scaled_dot_product_attention(
                Q, K_win, V_win, attn_mask=None, dropout_p=p, is_causal=False
            )  # (B,H,1,Dh)
            y_attn = y_attn.transpose(1, 2).contiguous().view(B, 1, d)
            y_attn = self.drop(self.W_O(y_attn))
        elif self.kv_cache_mode == "ring_win":
            # Ring buffer with staging windows: attn_state is (K_buf, V_buf, K_win, V_win, idx, filled)
            W = self.window_size
            if attn_state is None:
                # Initialize buffers on first step
                K_buf, V_buf, K_win, V_win, idx, filled = self._init_kv_buffers(
                    B, self.n_heads, W, d_h, x_t.device, x_t.dtype
                )
            else:
                # Unpack existing buffers
                if len(attn_state) == 2:
                    # Backward compatibility: re-init from cat-mode state
                    K_buf, V_buf, K_win, V_win, idx, filled = self._init_kv_buffers(
                        B, self.n_heads, W, d_h, x_t.device, x_t.dtype
                    )
                elif len(attn_state) == 6:
                    K_buf, V_buf, K_win, V_win, idx, filled = attn_state  # type: ignore[misc]
                else:
                    # Handle legacy 3-tuple or 4-tuple states
                    K_buf, V_buf, K_win, V_win, idx, filled = self._init_kv_buffers(
                        B, self.n_heads, W, d_h, x_t.device, x_t.dtype
                    )

            # Write new K/V to ring buffer (contiguous write)
            K_buf[:, :, idx] = K_new[:, :, 0]  # K_new: (B,H,1,Dh) -> (B,H,Dh)
            V_buf[:, :, idx] = V_new[:, :, 0]  # V_new: (B,H,1,Dh) -> (B,H,Dh)
            filled = torch.minimum(filled + 1, torch.as_tensor(W, device=filled.device))

            # Build contiguous window in staging buffers (two copy_ ops)
            L = int(filled.item())
            if L < W:
                # Contiguous prefix [0:L)
                K_win[:, :, :L].copy_(K_buf[:, :, :L])
                V_win[:, :, :L].copy_(V_buf[:, :, :L])
            else:
                # Logical order oldest..newest: [idx+1..W-1] then [0..idx]
                tail = W - 1 - int(idx.item())
                if tail > 0:
                    K_win[:, :, :tail].copy_(K_buf[:, :, idx + 1 : W])
                    V_win[:, :, :tail].copy_(V_buf[:, :, idx + 1 : W])
                head = int(idx.item()) + 1
                if head > 0:
                    K_win[:, :, tail : tail + head].copy_(K_buf[:, :, :head])
                    V_win[:, :, tail : tail + head].copy_(V_buf[:, :, :head])
                L = W  # full window

            # Call SDPA with contiguous staging buffers (fast path preserved)
            p = self.drop.p if self.training else 0.0
            y_attn = F.scaled_dot_product_attention(
                Q,
                K_win[:, :, :L],
                V_win[:, :, :L],
                attn_mask=None,
                dropout_p=p,
                is_causal=False,
            )  # (B,H,1,Dh)
            y_attn = y_attn.transpose(1, 2).contiguous().view(B, 1, d)
            y_attn = self.drop(self.W_O(y_attn))

            # Advance ring index
            if (W & (W - 1)) == 0:  # W is power of 2
                idx = (idx + 1) & (W - 1)
            else:
                idx = (idx + 1) % W
        else:
            # Default cat mode (backward compatible)
            if attn_state is None:
                K_cache, V_cache = K_new, V_new
            else:
                K_cache, V_cache = attn_state  # type: ignore[assignment]
                # Use rolling window approach to avoid full concat
                if K_cache.size(2) >= self.window_size:
                    # Shift left and append new token (more efficient than cat + slice)
                    K_cache = torch.cat([K_cache[:, :, 1:, :], K_new], dim=2)
                    V_cache = torch.cat([V_cache[:, :, 1:, :], V_new], dim=2)
                else:
                    # Still growing, just append
                    K_cache = torch.cat([K_cache, K_new], dim=2)
                    V_cache = torch.cat([V_cache, V_new], dim=2)

            # Attend to cache (no future present, so no mask needed)
            p = self.drop.p if self.training else 0.0
            y_attn = F.scaled_dot_product_attention(
                Q, K_cache, V_cache, attn_mask=None, dropout_p=p, is_causal=False
            )
            y_attn = y_attn.transpose(1, 2).contiguous().view(B, 1, d)
            y_attn = self.drop(self.W_O(y_attn))

        # SSM update using pre-computed u
        if ssm_state is None:
            ssm_state = torch.zeros(
                B, self.ssm_state_dim, device=x_t.device, dtype=x_t.dtype
            )
        alpha = torch.tanh(self.A)
        current_state = alpha * ssm_state + u[:, 0, :]  # Use pre-computed u
        y_ssm = self.drop(self.C(current_state).unsqueeze(1))  # (B, 1, d)

        # Use pre-computed gate_pre
        g = torch.sigmoid(gate_pre)  # (B, 1, d)
        y = g * y_attn + (1.0 - g) * y_ssm

        # Apply residual connection
        x_out = x_t + self.drop(y)  # (B, 1, d)

        # Pre-LN for MLP
        x_norm2 = self.ln2(x_out)  # (B, 1, d)

        # Apply MLP and residual connection
        mlp_out = self.mlp(x_norm2)  # (B, 1, d)
        x_final = x_out + self.drop(mlp_out)  # (B, 1, d)

        if self.kv_cache_mode == "ring2x":
            # Return 2W rolling buffers and updated state
            return x_final, current_state, (K_buf, V_buf, wp, filled)
        elif self.kv_cache_mode == "ring_win":
            # Return ring buffers, staging windows, and updated state
            return x_final, current_state, (K_buf, V_buf, K_win, V_win, idx, filled)
        else:
            return x_final, current_state, (K_cache, V_cache)
