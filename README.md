# Efficient Long-Context Blocks

A Python package for training and benchmarking efficient long-context attention blocks.

## Overview

This repository provides a clean, reproducible environment for developing and testing efficient long-context attention mechanisms. It includes GPU-ready dependencies (PyTorch + FlashAttention) and a comprehensive development setup.

### Why this exists (Research Direction)

Modern LLMs rely on attention that scales **quadratically** with context length $L$, which limits both speed and feasible context sizes. Two core observations motivate our approach:

1) **Local detail vs. global gist.** In natural text, tokens mostly need **dense, accurate attention locally** (recent context), while faraway context can often be represented as a **compressed summary** (“blurred” low-frequency content) without tracking every token-token interaction.

2) **Compute should match information value.** As tokens age, we can spend **less exact compute** on them while keeping their ideas accessible via **compact state** (summaries) and **lightweight connectivity**.

We explore two complementary blocks:

- **BLADE** - **B**lock-**L**ocal **A**ttention with **D** per-block **E**mbedding/state
  Split the sequence into chunks; compute exact attention **within** each chunk; pass a **small, learned state** between chunks to propagate information at **sub-quadratic** cost.

- **DP-ASSM** - **D**ual-**P**ath (local attention + **A**ttention-**S**SM fusion)
  Combine a local, **windowed causal** attention path with a global **state-space model (SSM)** path that summarizes long-range history linearly; fuse them with a **learned gate**.

Together, these reduce effective complexity while staying competitive on tasks requiring long-range reasoning.


## Features

- **GPU-Ready:** Pre-configured with CUDA support and FlashAttention
- **Long-Context Blocks:** Plug-and-play BLADE and DP-ASSM modules
- **Baselines & Ablations:** Longformer/BigBird-style baselines and an Infini-style mixed-path ablation (optional)
- **Evaluation Harness:** Synthetic long-context tasks, streaming curves, and long-document QA
- **Development Tools:** Black, Ruff, isort for code formatting and linting
- **Testing:** Pytest with coverage reporting
- **Pre-commit Hooks:** Automated code quality checks
- **Reproducible:** Locked dependencies and clear setup instructions

## Research Motivation & Hypotheses

**Motivation**
- Full attention has $O(L^2)$ time/memory. Even with kernel/IO optimizations, quadratic scaling from dense token–token interactions constrains long contexts.
- In long documents, **recent tokens** need precise interactions, while **older tokens** mostly contribute **coarse, semantic signals**.

**Hypotheses.**
- **H1 (Local precision):** Restricting exact attention to a **local window or chunk** captures most short-range dependencies and stabilizes generation.
- **H2 (Global compression):** Representing **long-tail context** via **compact state** (e.g., SSM, per-chunk embeddings, or global tokens) is sufficient for many tasks.
- **H3 (Dual-path fusion):** Mixing **dense local** and **cheap global** paths with a **learned gate** matches or exceeds quality of purely sparse attention at a lower cost.
- **H4 (Per-block state):** A **small state vector per block** can carry forward salient information and enable cross-block influence without full K/V caches.

---

## Methods

### 1) BLADE - Block-Local Attention with Per-Block State

See: [docs/methods/blade.md](docs/methods/blade.md)

**Intuition.** Treat the sequence as **local, dense clusters** (chunks) with a **compact state** flowing between them.

**Mechanics.**
- Split input into chunks of size $C$.
- In each chunk, run standard **causal self-attention** (can be full or windowed in-chunk).
- After a chunk, compute a **state vector** $h_i \in \mathbb{R}^{d_s}$ (e.g., mean-pooled features passed through an MLP). Inject it into the **next** chunk as a bias or conditioning signal.
- Optional **global tokens** can be prepended per chunk to provide lightweight global connectivity.

**Complexity.** With $k=\lceil L/C \rceil$ chunks, attention cost scales like $O(k \cdot C^2)$ (vs. $O(L^2)$). State passing is $O(k \cdot d_s \cdot d)$.

**API (simplified).**
```python
block = BLADEBlock(
   d_model, n_heads, chunk_size, state_dim, m_global=0
)
y, new_state = block(x, state=None)
```

**When to use.**

* You want a **drop-in block** with stable training that already gives **sub-quadratic** behavior.
* You prefer explicit control over **chunk size** and simple reasoning about compute.


### 2) DP-ASSM - Dual-Path (Local Attention + SSM) with Gated Fusion

See: [docs/methods/dpassm.md](docs/methods/dpassm.md)

**Intuition.** Keep exact local interactions; capture the long tail with a **linear-time SSM** and **blend** the two.

**Mechanics.**

* **Local path:** windowed causal attention over the last (W) tokens (FlashAttention/SDPA kernels).
* **SSM path:** a compact recurrent state $s_t \in \mathbb{R}^{d_s}$ updated per token:
  $s_t = \alpha \odot s_{t-1} + B x_t,~y^{ssm}_t = C s_t$
  with learnable $\alpha, B, C$. (Start linear; add gating/nonlinearities later if needed.)
* **Fusion:** feature-wise gate $g_t = \sigma(W_g x_t)$ mixing paths
  $y_t = g_t \odot y^{attn}_t + (1-g_t) \odot y^{ssm}_t$.

**Complexity.** Local attention is $O(L \cdot W)$; SSM is $O(L)$. Memory does **not** grow with $L$ in the SSM path.

**API (simplified).**

```python
block = DPASSMBlock(
    d_model, n_heads, window_size, ssm_state_dim
)
y, new_state = block(x, ssm_state=None)
```

**When to use.**

* You want **streaming-friendly** inference.
* You need robustness on very long contexts but still want **precise local** reasoning.


## Evaluation Plan

We include scripts to measure **quality**, **latency**, and **memory** under increasing context:

1. **Synthetic long-context tasks**

   * **Passkey retrieval:** plant a token far in the past; query at the end.
   * **Copy/recall:** reproduce a distant subsequence.
   * **Drift curves:** track performance vs. input length (e.g., 16K -> 1M) with streaming prefill.

2. **Long-document QA**

   * Public-domain books or converted PDFs; ask questions far from answers.
   * Evaluate EM/F1 and latency; compare BLADE/DP-ASSM vs. baselines.

3. **System profiling**

   * Tokens/sec (train/infer), prefill speed, and peak VRAM across blocks and lengths.
   * Output CSVs + plots suitable for papers/benchmarks.

**Baselines & Ablations**

* Longformer / BigBird-style (local+global) baselines.
* Infini-style mixed-path ablation (local exact + compressive memory).
* Fairness guidelines: equal parameter budgets, matched training steps, fixed evaluation prompts, report wall-clock and VRAM.


## Future Explorations

* **Small-World Attention (SWA):**
  Local dense clusters (chunks/windows) plus a few **random, weak long-range links** between blocks-reduces effective graph distance with $O(LW + Lr)$ edges (where $r \ll W$).

* **Frequency/Pyramid Memory:**
  A UNet/Laplacian-like **multi-scale** pathway: recent tokens keep high-frequency detail; older tokens are **downsampled/blurred** into low-frequency summaries.

* **Hybrid State Designs:**
  Mix **per-block state** (BLADE) with **global SSM** (DP-ASSM) to capture both block evolution and global dynamics.

* **State Compression & Quantization:**
  Learned compaction of per-block or SSM states (8-bit or lower), enabling **edge** deployment with tiny memory budgets.

* **Shifted windows & block boundaries:**
  Alternate chunk/window alignment between layers to improve cross-boundary mixing with negligible overhead.


## Quick Start

### Prerequisites

* Python 3.12+
* CUDA-compatible GPU (recommended)
* [uv](https://github.com/astral-sh/uv) for dependency management

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/maida-ai/research.git
   cd research
   ```

2. Set up the development environment:

   ```bash
   make setup
   ```

3. Verify installation:

   ```bash
   make test
   ```


## Development

### Available Commands

* `make setup` - Set up development environment
* `make test` - Run tests
* `make lint` - Run linting checks
* `make format` - Format code
* `make clean` - Clean up generated files

### Project Structure

```
efficient-longctx/
├── efficient_longctx/          # Main package
│   ├── blocks/                 # Attention blocks (BLADE, DP-ASSM, baselines)
│   ├── training/               # Training utilities
│   ├── evals/                  # Evaluation utilities (synthetic, long-doc, profiling)
│   └── utils/                  # Utility functions
├── tests/                      # Test suite
├── examples/                   # Example notebooks
├── docs/                       # Documentation (method cards, diagrams)
├── reports/                    # Analysis and benchmarks (CSV, plots)
└── scripts/                    # Utility scripts
```

### Code Quality

* **Black** - Code formatting
* **Ruff** - Fast Python linter
* **isort** - Import sorting
* **Pre-commit** - Git hooks for automated checks

### Testing

Run the test suite:

```bash
make test
```

The test suite includes:

* CUDA availability checks
* FlashAttention/SDPA attention smoke tests
* BLADE/DP-ASSM shape and state-continuity tests

---

## Dependencies

### Core Dependencies

* `torch` - PyTorch for deep learning
* `flash-attn` - FlashAttention for efficient attention
* `numpy` - Numerical computing
* `transformers` - Hugging Face transformers
* `datasets` - Dataset utilities
* `accelerate` - Training acceleration
* `einops` - Tensor operations
* `triton` - GPU kernel compilation

### Development Dependencies

* `pytest` - Testing framework
* `pytest-cov` - Coverage reporting
* `ruff` - Fast Python linter
* `black` - Code formatter
* `isort` - Import sorter
* `pre-commit` - Git hooks

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting: `make test && make lint`
5. Commit your changes
6. Submit a pull request

We welcome issues/PRs for:

* New long-context blocks or ablations
* Evaluation tasks and datasets
* Optimization (state compression, quantization, kernel tweaks)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* [FlashAttention](https://github.com/Dao-AILab/flash-attention) for efficient attention implementations
* [PyTorch](https://pytorch.org/) for the deep learning framework
* [Hugging Face](https://huggingface.co/) for transformer models and datasets
