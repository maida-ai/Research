
# Design Doc B — Block-Local Attention + Per-Block State (BLADE)

## 1) Goal & Thesis

Partition a long sequence into (k) chunks (length (L/k)). Within each chunk we run **exact attention**; between chunks we pass a **compact recurrent state** per layer. Add a tiny global/latent mixing to prevent information islands.

This unifies **local/block attention** (Longformer/BigBird) with a **chunkwise recurrent conduit** (RetNet chunkwise mode), and optional **latent bottleneck** (Perceiver IO) for cross-chunk communication. ([arXiv][9])

## 2) Architecture

* **Intra-chunk:** causal attention with FlashAttention; complexity (k\cdot(L/k)^2 = O(L^2/k)) ([arXiv][10]).
* **Per-block state:** each layer maintains a small state vector (h_i^{(\ell)}) for chunk (i); when processing (i!+!1), we **inject** (h_i^{(\ell)}) via (a) additive bias, (b) cross-attention from chunk tokens to the state, or (c) gating into MLP. Analogous to **Transformer-XL**’s segment recurrence but far cheaper than carrying full K/V ([ACL Anthology][11]).
* **Global mixing (choose one):**

  1. **G tokens** (BigBird) always visible to all chunks;
  2. **Shifted chunking** every M layers;
  3. **Latent array** (m!\in![8,32]) (Perceiver IO) each chunk cross-attends to. ([arXiv][12])

**Complexity:** (O(L^2/k) + O(kdm)) (with tiny (m))—the reconstructor is light. This keeps your simple scaling story while beating plain local windows on long-range tasks.

## 3) Evaluation Plan

* **LRA-style** long-range benchmarks + book QA; compare against **Longformer/BigBird** and **H-Transformer-1D** (hierarchical baseline) ([arXiv][9]).
* **Cross-chunk transfer tests:** prompt info in chunk 1, ask a question in chunk N. Measure degradation vs chunk distance with/without latent/global mixing.
* **System metrics:** throughput vs (k), VRAM vs (L), flash vs non-flash kernels.
* **Distributed:** support million-token prefill with **Context Parallelism / RingAttention** for demos ([OpenReview][8]).

## 4) Milestones (Month 1 OSS)

* **Week 1:** BLADE kernels using FlashAttention; per-layer state API (save/restore).
* **Week 2:** Demo model **150–350M**; ablate (k), state size, and global/latent mixing.
* **Week 3:** Paper draft v0; reproducible scripts vs **Longformer/BigBird** baselines.
* **Week 4:** OSS release: code + small checkpoints + **chunk-transfer benchmark**.


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
