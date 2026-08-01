# Open-Source Deliverables (Month 1)

1. **`efficient-longctx` repo**

   * `blocks/dual_path_assm.py`, `blocks/blade.py`
   * FlashAttention-2 integration; Triton kernels where helpful.
   * Training scripts + configs; evaluation harnesses (streaming curve plots, book QA, passkey).
   * Checkpoints for **150–350M** models to keep costs low.

2. **Benchmarks**

   * **StreamingLLM-style** harness (quality vs length), with switches for sink/global tokens. ([arXiv][4])
   * **Chunk-transfer** benchmark (info in chunk 1, question in chunk N).
   * **Verbatim recall** tasks (quote-back, code symbol lookup), optional kNN memory toggle. ([arXiv][7])

3. **Docs & Colab**

   * Minimal Colab demo for **1M-token prefill** using **Context Parallelism** (H100/4090 multi-GPU notes). ([arXiv][13])

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
