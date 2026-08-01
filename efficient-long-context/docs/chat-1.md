# Quick correctness checks (to sharpen the problem statement)

* **What's actually quadratic?** During *training*, the attention score matrix (QK^\top) is (L\times L) per head and drives (O(L^2)) compute/memory. During *autoregressive inference*, we don't materialize/store the full (L\times L) map; we store a **KV cache** of size (O(Ld)) and each new token costs (O(Ld)) to attend over the cache. This nuance matters because many “speedups” either target training IO/compute (e.g., FlashAttention) or inference memory (e.g., KV-cache compression/eviction). ([arXiv][1])

* **“Sparse or tiny probabilities ⇒ noisy generations.”** Empirically, long-context failures stem more from *recency bias* and *how caches are managed* than from “tiny probs” per se. For example, **StreamingLLM** shows that keeping a small set of “attention sink” tokens + recent tokens stabilizes generation over millions of tokens without fine-tuning. Framing the issue as *information routing and memory management* (not just probability mass) aligns better with prior evidence. ([arXiv][2])

# How your ideas line up with prior work

## A. “Blur distant context, keep recent in hi-res” (Solution 1)

* **Very close cousins.**

  * **Compressive Transformer**: retains a short uncompressed memory and *compresses* older activations with pooling/conv, trained with an auxiliary reconstruction loss so long-term info remains useful. That's almost the same “blur over time” intuition you propose. ([arXiv][3])
  * **Infini-attention**: fuses *local masked attention* with a **compressive memory + linear long-range pathway** in the same block, enabling effectively unbounded context at bounded cost. It's another principled realization of “dense local, cheap global.” ([arXiv][4])
  * **Hierarchical/Multiscale attention**: **H-Transformer-1D** builds a **multi-resolution hierarchy** to get linear complexity and does well on LRA; conceptually akin to your Laplacian/UNet framing. ([ACL Anthology][5])

* **What to borrow explicitly.**

  1. **Auxiliary reconstruction loss for the compressed stream** (à la Compressive Transformer) to keep the “blur” predictive. ([arXiv][3])
  2. **Two-path block** (local exact + global cheap) as in Infini-attention; your “low-/high-frequency” language can be implemented as *cross-scale mixing* between a short high-res buffer and a long compressed memory. ([arXiv][4])
  3. **Hierarchical schedule** for when/where to downsample and how info flows up/down the pyramid (H-Transformer-1D gives one workable template). ([ACL Anthology][5])

* **Pitfalls to pre-empt.**

  * **Tasks needing verbatim recall** (code, quotes, math): pure “blur” can destroy token-level fidelity. Mixed strategies help: keep a *tiny set of global tokens* (BigBird result) or attach a **kNN/associative memory** for exact lookups when needed (Memorizing Transformer). ([NeurIPS Papers][6])
  * **Where to compress (keys? values? states?)**: Compressive Transformer compresses *past hidden states* stored as memory rather than current-step keys/values; Infini-attention integrates the compressive path *inside* the attention block. Be concrete here. ([arXiv][3])

## B. “Split the sequence into k local groups + a reconstructor” (Solution 2)

* **Very well-trodden design space.**

  * **Local/block attention**: **Longformer** (sliding windows + optional global tokens) and **BigBird** (local + random + global) reduce cost to near-linear and prove strong theory (universal approximation / Turing completeness) *if* a few globals exist. Your (O(L^2/k)) within-block complexity matches this family. ([arXiv][7])
  * **Latent bottlenecks**: **Perceiver/Perceiver IO** compress tokens into a latent array via cross-attention, then decode—this is a powerful “reconstruction layer” pattern that avoids recomputing full global maps. ([arXiv][8])

* **What you'll need beyond pure blocking.**

  1. **Cross-block mixing** so information isn't trapped: e.g., **shifted windows** (vision analogy), a *few* **global tokens** (BigBird), or a small **latent set** (Perceiver IO). This keeps quality while preserving (O(L^2/k)+O(kL)). ([NeurIPS Papers][6])
  2. **IO-aware kernels**: if you still run exact attention within blocks, pair with **FlashAttention** during training to convert “theory wins” into wall-clock wins. ([arXiv][1])

# Context in the broader landscape (to position novelty)

* **Low-rank/linear approximations** (Linformer, Performer, Linear Transformers, Nyströmformer) trade exactness for linear scaling; many groups combine a few exact layers with mostly linear ones. Your Solution 1 can be pitched as a *structured, content-aware* approximation rather than purely kernelized. ([arXiv][9])

* **State-space / convolutional alternatives** (RetNet, Mamba, Hyena) give linear time with strong long-range behavior; they often keep a short local attention path too. Useful as **baselines** and as inspiration for the “low-frequency” channel in Solution 1. ([arXiv][10])

* **Recurrent or retrieval memory** (Transformer-XL, Compressive, Infini-attention, StreamingLLM, Memorizing Transformers) already validate: *dense short-term + compressed/associative long-term* beats naïve sliding windows. Your proposal is well aligned—just anchor it to these threads. ([arXiv][11])

# Concrete feedback on your two proposals

## 1) Frequency-split / UNet-style “blurred” memory

**Keep it, but specify:**

* **Where the pyramid lives:** (a) outside attention as an auxiliary memory (Compressive-style), or (b) *inside each block* with dual paths (Infini-attention). The latter simplifies integration and lets you train end-to-end. ([arXiv][3])
* **What gets downsampled:** keys, values, or hidden states? If you compress *values* (semantic content) but keep *keys* for addressing, you often preserve retrieval while saving memory.
* **Losses & routing:** add a **reconstruction/objective on compressed memory** and a **gating policy** to decide when a token is “promoted” to high-res or “demoted” to low-res. Compressive Transformer's auxiliary loss is a plug-and-play starting point. ([arXiv][3])
* **Safety valve for verbatim needs:** include *a few* **global tokens** or a tiny **latent bottleneck** to keep theoretical expressivity and practical recall. ([NeurIPS Papers][6])

## 2) k-way grouping + reconstruction

**Good baseline, but add two ingredients:**

* **Global mixing:** adopt **(i)** 1–4 global tokens (BigBird), **(ii)** periodic *shifted groups*, or **(iii)** a small latent array (Perceiver IO). This prevents “information islands.” ([NeurIPS Papers][6])
* **Implementation realism:** use **FlashAttention** for intra-group attention; for *very* long contexts or multi-GPU inference, consider **Ring Attention / context parallelism** so groups stream across devices. ([arXiv][1])

# Evaluation advice (what will convince reviewers)

* **Quality under very long contexts**: LRA (sanity check), 1M-token passkey retrieval / book-level summarization (as in Infini-attention), and **streaming stability** analyses from StreamingLLM (does quality drift as length grows?). ([arXiv][4])
* **Exact-recall stress tests**: math/code long-range dependencies and “quote-back” tasks to catch failures of excessive blurring (compare against a kNN/associative-memory ablation). ([arXiv][12])
* **System metrics**: wall-clock speed, HBM traffic, and memory footprints (FlashAttention style IO accounting) so gains aren't only asymptotic. ([arXiv][1])
* **Ablations**: pyramid depth, downsampling ratio, which tensors are compressed (K vs V vs states), number of global tokens/latent slots, and group size (L/k).

# Suggested reframes to strengthen novelty & clarity

1. **Rename Solution 1 as** *Hierarchical Dual-Path Attention with Compressive Memory* and explicitly connect to Compressive/Infini lineages while highlighting your **frequency-split routing** (UNet/Laplacian prior). ([arXiv][3])
2. **Position Solution 2 as** *Block-local Attention with Lightweight Global Reconstruction* and claim simplicity + strong baselines vs Longformer/BigBird/Perceiver-style bottlenecks. ([arXiv][7])
3. **State the bet:** *“Dense local, compressive global* outperforms *purely sparse* or *purely linear* approximations at equal compute.” Then design experiments that hold FLOPs/memory constant across baselines.

If you'd like, I can turn this into a one-page **“Related Work & Design Rationale”** section with crisp citations and a figure that situates your method among Compressive/Infini/BigBird/Perceiver families.

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
