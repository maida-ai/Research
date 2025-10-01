Add synthetic long-context evals (passkey, copy/recall, drift curve)

Create toy tasks to validate long-range retention and streaming stability.

## Description

We need simple, fast-to-run benchmarks to verify the mechanics:

* **Passkey retrieval:** plant a random token early; query at end.
* **Copy/recall:** reproduce a distant subsequence.
* **Drift curve:** track metric vs. increasing sequence length.

## Implementation Suggestions

* File: `efficient_longctx/evals/synthetic.py`
* Implement dataset generators that yield `(input_ids, labels)` for each task.
* Example:
  ```
  python -m efficient_longctx.evals.synthetic \
    --task passkey --seq_len 32768 --max_len 131072 \
    --model <path_or_hf_id> --block dpassm
  ```
* Save plots to `reports/synthetic/` (use matplotlib).
* Provide a CPU-friendly mode for very short sequences to sanity-check without a GPU.

## Acceptance Criteria

* [ ] Commands run end-to-end and print task metrics (accuracy/EM).
* [ ] Drift curves (PNG) saved for lengths up to ≥128K on a GPU box.
* [ ] README includes exact commands to reproduce each task.
