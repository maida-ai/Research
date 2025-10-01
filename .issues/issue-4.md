Implement BLADE block (Block-Local Attention with Per-Block State)

Split sequences into chunks; run exact attention per chunk; pass a compact state between chunks.

## Description

Chunking reduces attention cost and makes memory predictable. Passing a small per-chunk state improves cross-chunk information flow without carrying full K/V.

## Implementation Suggestions

* File: `efficient_longctx/blocks/blade.py`
* Skeleton:
  ```python
  import torch
  import torch.nn as nn

  class BLADEBlock(nn.Module):
      def __init__(self, d_model, n_heads, chunk_size, state_dim, m_global=0, dropout=0.1):
          super().__init__()
          self.chunk_size = chunk_size
          self.state_proj = nn.Linear(d_model, state_dim)
          self.state_inject = nn.Linear(state_dim, d_model)
          self.m_global = m_global
          if m_global > 0:
              self.global_tokens = nn.Parameter(torch.randn(m_global, d_model))
          # TODO: self.attn = FlashAttention-based causal attention
          self.proj = nn.Linear(d_model, d_model)
          self.dropout = nn.Dropout(dropout)

      def forward(self, x, state=None):
          B, T, D = x.shape
          if state is None:
              state = torch.zeros(B, self.state_inject.in_features, device=x.device, dtype=x.dtype)
          outputs = []
          for start in range(0, T, self.chunk_size):
              end = min(T, start + self.chunk_size)
              chunk = x[:, start:end, :]
              # Inject previous state (bias the chunk)
              chunk = chunk + self.state_inject(state).unsqueeze(1)
              # Optionally prepend global tokens
              if self.m_global > 0:
                  g = self.global_tokens.unsqueeze(0).expand(B, -1, -1)
                  chunk_in = torch.cat([g, chunk], dim=1)
              else:
                  chunk_in = chunk
              y_chunk = self._causal_attn(chunk_in)
              # Strip globals from output if used
              if self.m_global > 0:
                  y_chunk = y_chunk[:, self.m_global:, :]
              outputs.append(y_chunk)
              # Update state from this chunk
              state = self.state_proj(y_chunk.mean(dim=1))
          y = torch.cat(outputs, dim=1)
          return self.proj(self.dropout(y)), state
  ```

* Add `tests/test_blade.py`:
  * Confirms gradients flow across chunks, global tokens optional, and changing early chunks affects later outputs when state passing is on.

## Acceptance Criteria

* [ ] Processes arbitrary `T` via chunking without OOM at typical sizes.
* [ ] Per-chunk state is updated and influences later chunks (unit test).
* [ ] Optional global tokens are supported and trainable.
