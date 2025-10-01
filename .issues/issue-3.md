Implement DP-ASSM block (Dual-Path Attention + SSM)

Combine windowed local attention with a global state-space path, then fuse via a learnable gate.

## Description

Create a transformer block that:

1. Applies **causal windowed attention** over the most recent `W` tokens for precise local reasoning.
2. Maintains a compact **SSM state** to summarize long-range context with linear cost.
3. **Fuses** the two paths with a learned gate per token.

## Implementation Suggestions

* File: `efficient_longctx/blocks/dpassm.py`
* Skeleton:
  ```python
  import torch
  import torch.nn as nn

  class DPASSMBlock(nn.Module):
      def __init__(self, d_model, n_heads, window_size, ssm_state_dim, dropout=0.1, n_probes=0):
          super().__init__()
          # TODO: self.attn = FlashAttention-based windowed attention
          # TODO: self.ssm = simple SSM cell (see notes below)
          self.probes = nn.Parameter(torch.randn(n_probes, d_model)) if n_probes > 0 else None
          self.gate = nn.Linear(d_model, 1)
          self.proj = nn.Linear(d_model, d_model)
          self.dropout = nn.Dropout(dropout)

      def forward(self, x, ssm_state=None):
          # x: (B, T, d_model)
          y_attn = self._local_attn(x)                     # implement causal windowed attention
          y_ssm, new_ssm_state = self._ssm_path(x, ssm_state)  # recurrent update over T
          gate = torch.sigmoid(self.gate(x))
          y = gate * y_attn + (1 - gate) * y_ssm
          return self.proj(self.dropout(y)), new_ssm_state
  ```

* Local attention:
  * Use FlashAttention-2 causal kernel over sliding window `W`.
  * Mask tokens outside the window.
* SSM path (simple starter):
  * Implement a small gated 1D convolution or a diagonal-plus-low-rank recurrence:
    * `s_t = A * s_{t-1} + B * x_t`, `y_ssm_t = C * s_t` with learnable `A, B, C` (start with diagonal `A`).
* Add `tests/test_dpassm.py` to check shapes, gradients, and state continuity across two forward calls.

## Acceptance Criteria

* [ ] `DPASSMBlock` forward pass runs on CUDA with random input.
* [ ] Attention uses a causal window; outside-window tokens are not attended.
* [ ] SSM state is updated and returned; continuity verified by test.
