# Design Doc A — Dual-Path Attention + State Space (DP-ASSM)

## 1) Goal & Thesis

Deliver a **single block** that combines:

* **Short-term, precise reasoning:** local causal **attention** over the last (W) tokens.
* **Long-term, compressed memory:** a recurrent **state-space model (SSM)** stream that keeps linear-time, flat-memory “low-frequency” context.

This directly builds on: **Compressive Transformer** (compressed long-term memory) ([arXiv][1]), **Infini-attention** (local exact + compressive/linear long path) ([arXiv][2]), and modern SSMs (**Mamba**) for linear scaling and stable long horizons ([arXiv][3]).

**Why now?** Evidence shows (i) keeping a tiny set of **attention-sink**/initial tokens plus a short window stabilizes streaming LLMs, but window-only breaks past cache size ([arXiv][4]); (ii) compressing older information helps if the compression is predictive (aux losses) ([arXiv][1]); (iii) SSMs provide linear cost but need help for content-addressed precision (attention fills that gap) ([arXiv][3]).

## 2) Block Architecture

**Inputs:** token chunk (x_{t-W+1:t}), previous SSM state (s_{t-W}).

**Parallel paths (per layer):**

1. **Local Attention Path**

   * Windowed causal self-attention over (W) recent tokens (FlashAttention kernels for training) ([arXiv][5]).
   * Optional: keep 2–8 **global/sink tokens** always visible to stabilize ultra-long streams ([arXiv][4]).

2. **SSM Path**

   * Recurrent update ($s_{t}\leftarrow\mathrm{SSM}(s_{t-1}, x_t)$) (Mamba-style **selective** parameters for discrete text) ([arXiv][3]).
   * Parallel training mode (as in RetNet/Mamba) for throughput; chunkwise recurrent for long inference ([arXiv][6]).

**Fusion & Routing:**

* Gate or FiLM-condition to **mix** attention output and SSM output per token; light cross-attention from queries to a **small learned “state probe”** (2–8 vectors) to read global SSM context.
* **Auxiliary compression loss** on the SSM stream (predict local summaries or reconstruct downsampled features) à la Compressive Transformer ([arXiv][1]).

**Complexity (per layer):**

* Attention (O(WHd)) (bounded by (W)).
* SSM (O(Hd^2)) per token (linear in length, fixed state), flat memory growth.

## 3) Training & Inference

* **Pretrain recipe:** mix long and short sequences; ramp (W) from 1–2K to 8–16K over curriculum.
* **Stability:** periodic state **reset** every N chunks if drift observed; selective gates control forgetting (per Mamba) ([arXiv][3]).
* **Inference:** KV cache only for the last (W); SSM keeps **O(d)** state per layer → near-constant VRAM w.r.t. context length.

## 4) Evaluation Plan

**Quality vs length**

* **StreamingLLM curves**: perplexity or QA EM vs input length up to 1–2M tokens; ablate sink tokens and window size (W) ([arXiv][4]).
* **Long-range retrieval / book tasks** (as in **Infini-attention**): passkey retrieval, book-level QA/summarization ([arXiv][2]).
* **Exact-recall stress**: code quote-back, long math proofs; show that small global tokens or a tiny kNN memory (Memorizing Transformer) preserve verbatim recall ([arXiv][7]).

**Systems**

* Wall-clock throughput, HBM traffic (FlashAttention IO accounting) ([arXiv][5]).
* Memory scaling vs context; multi-GPU **context parallelism** for >128K prefill (Ring/Context Parallel) ([OpenReview][8]).

## 5) Milestones (Month 1 OSS)

* **Week 1:** Minimal DP-ASSM block in PyTorch + FlashAttention kernels; SSM path via open-sourced Mamba-like ops. Unit tests, fp16/bf16.
* **Week 2:** Pretrain **150–350M** toy LM on SlimPajama / OpenWeb subsets; release **long-context synthetic** benchmarks (needle-in-a-haystack, passkey, copy/recall).
* **Week 3:** Paper draft v0 with **ablation table** (with/without SSM path, aux loss, sink tokens).
* **Week 4:** OSS release: code, weights of small model, and **evaluation harness** (streaming curves + book QA).

---

[1]: https://arxiv.org/abs/1911.05507?utm_source=chatgpt.com "Compressive Transformers for Long-Range Sequence ..."
[2]: https://arxiv.org/abs/2404.07143?utm_source=chatgpt.com "Efficient Infinite Context Transformers with Infini-attention"
[3]: https://arxiv.org/abs/2312.00752?utm_source=chatgpt.com "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
[4]: https://arxiv.org/abs/2309.17453?utm_source=chatgpt.com "Efficient Streaming Language Models with Attention Sinks"
[5]: https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com "Fast and Memory-Efficient Exact Attention with IO-Awareness"
[6]: https://arxiv.org/abs/2307.08621?utm_source=chatgpt.com "A Successor to Transformer for Large Language Models"
[7]: https://arxiv.org/pdf/2203.08913?utm_source=chatgpt.com "Memorizing Transformer"
[8]: https://openreview.net/forum?id=WsRHpHH4s0&utm_source=chatgpt.com "RingAttention with Blockwise Transformers for Near-Infinite ..."
[9]: https://arxiv.org/abs/2004.05150?utm_source=chatgpt.com "[2004.05150] Longformer: The Long-Document Transformer"
[10]: https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com "FlashAttention-2: Faster Attention with Better Parallelism ..."
[11]: https://aclanthology.org/P19-1285/?utm_source=chatgpt.com "Transformer-XL: Attentive Language Models beyond a ..."
[12]: https://arxiv.org/abs/2007.14062?utm_source=chatgpt.com "[2007.14062] Big Bird: Transformers for Longer Sequences"
[13]: https://arxiv.org/html/2411.01783v2?utm_source=chatgpt.com "Context Parallelism for Scalable Million-Token Inference"
[14]: https://huggingface.co/blog/infini-attention?utm_source=chatgpt.com "A failed experiment: Infini-Attention, and why we should ..."
