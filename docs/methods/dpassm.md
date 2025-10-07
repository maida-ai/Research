# DP-ASSM — Dual-Path (Local Attention + SSM) with Gated Fusion

**Idea.** Keep exact local interactions via windowed attention and capture the long tail with a compact **state-space model (SSM)**; fuse both per token with a learned gate.

- **Local path:** windowed causal attention over last $W$ tokens
- **Global path:** linear-time **SSM** summarizing all history
- **Fusion:** feature-wise gate $\sigma(W_g x)$ blends paths
- Pre-LN + FFN for stable stacking

## Diagram (block)

```mermaid
flowchart LR
  X["x (B,T,d)"] --> LN1["LayerNorm"]
  LN1 -->|local| ATT["Windowed Causal Self-Attn (W)"]
  LN1 -->|global| SSM["SSM: s_t = α ⊙ s_{t-1} + Bx_t; y_ssm = C s_t"]

  ATT --> CAT["Fusion: y = g ⊙ y_attn + (1-g) ⊙ y_ssm"]
  SSM --> CAT

  CAT --> RES1["Residual Add"]
  RES1 --> LN2["LayerNorm"]
  LN2 --> FFN["FFN (GELU)"]
  FFN --> RES2["Residual Add → output"]
````

## Diagram (SSM recurrence)

```mermaid
flowchart LR
  s_prev["s_{t-1}"]
  x_t["x_t"]
  alpha["α (diag)"]
  B["B"]
  C["C"]
  s_prev -->|elementwise α| MUL["α ⊙ s_{t-1}"]
  x_t --> B --> ADD
  MUL --> ADD["s_t = α ⊙ s_{t-1} + B x_t"]
  ADD --> C --> y["y^{ssm}_t = C s_t"]
```

## Math (minimal)

* Local attention: standard causal attention over window $W$
* SSM:
  $$\begin{aligned}
  s_t &= \alpha \odot s_{t-1} + B x_t \\
  y^{ssm}_t &= C s_t
  \end{aligned}$$
  with $\alpha \in \mathbb{R}^{d_s}$ (use `tanh(alpha_raw)` to keep it in (-1,1))
* Fusion (feature-wise):
  $$\begin{aligned}
  g_t &= \sigma(W_g x_t) \in \mathbb{R}^d \\
  y_t &= g_t \odot y^{attn}_t + (1 - g_t) \odot y^{ssm}_t
  \end{aligned}$$

## Pseudocode (forward)

```python
u = LN1(x)
y_attn = windowed_causal_attn(u, W)              # O(L·W)
y_ssm, sT = ssm_scan(u, s0)                      # O(L)
g = sigmoid(Wg @ u)
y = g * y_attn + (1 - g) * y_ssm
y = x + Drop(y)                                  # residual 1
y = y + Drop(FFN(LN2(y)))                        # residual 2
return y, sT
```

## Complexity

* Local path: $O(LW)$ time, $O(W·d)$ KV memory (window-only)
* SSM path: $O(L)$ time, constant memory
* Total $\approx O(LW) + O(L)$ (W << L)

## Configuration Tips

* `window_size W`: 512–2048
* `ssm_state_dim d_s`: 128–512
* Start with linear SSM; add gating/nonlinearities if needed later
