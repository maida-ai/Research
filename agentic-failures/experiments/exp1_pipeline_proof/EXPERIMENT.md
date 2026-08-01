# Experiment 1 — Pipeline Proof and Integrity-Event Rate Estimate

**Status:** Pre-registration  
**Budget cap:** $500 hard stop  
**Probe kill-switch:** $50 for the cost probe  
**Date:** 2026-06-11  

---

## Objectives

1. **Pipeline end-to-end proof** [already executed, $0]  
   Demonstrate the full path: scaffold → trace JSONL with model_io events → terminal report →
   termination event. Covered by `smoke_test.py --exp smoke`. No API calls.

2. **Reference-agent gradient check** [local RTX 4090, $0]  
   Validate that the frozen Qwen2.5-7B-Instruct reference agent shows a usable difficulty
   gradient (10%–90% failure rate span) over the Blocks World generator parameter range.
   Required before pinning the reference agent (plan §9.5).
   Dry run executed via `gradient_check.py --mock`.

3. **First integrity-event-rate estimate in enriched bands** [needs model runs — smoke → probe ladder]  
   Run a cost probe (6–12 instances × 2 models × 2 architectures × 3 seeds) to estimate:
   - mean and p95 tokens per trajectory;
   - mean tool calls per trajectory;
   - integrity-event frequency in near-boundary and minimally-unsat bands;
   - terminal-claim distribution (claim_success / claim_infeasible / report_unknown);
   - verifier overhead vs. baseline;
   - sensitivity to one pre-registered paraphrase (optional).
   Kill-switch cap: $50 for the probe.

4. **Trajectory cost numbers** [from probe]  
   Replace all ASSUMPTION token counts in `cost_model.py` with measured probe values.
   Scale-up decision for Experiment 2 is conditioned on these numbers.

---

## Procedure

### Run order

```
1. smoke_test.py        -- $0, always first (cost doctrine #3)
2. gradient_check.py    -- $0 (--mock), then $0 (--live on 4090 once server is up)
3. cost_model.py        -- $0 (generates cost estimates with placeholder prices)
4. MANUAL: fill in prices.json with actual API prices
5. cost_probe           -- up to $50 (uses exp1_pipeline_proof infrastructure + scaffold)
6. UPDATE cost_model.py -- replace ASSUMPTIONS with probe-measured values
7. GO/NO-GO decision    -- see §15.5 criteria below
```

### Step 1 — Smoke test

```bash
.venv/bin/python -m experiments.runners.smoke_test --exp smoke --seed 42
```

Confirms:
- scaffold → trace → terminal report → termination event end-to-end
- model_io events written verbatim
- report_unknown submitted on budget exhaustion (no extra pressure framing)
- trace structure: run_metadata first, termination last

### Step 2 — Gradient check (mock, then live)

```bash
# Dry run — MockModel, $0
.venv/bin/python -m experiments.runners.gradient_check --mock --seed 42

# Live run — requires vLLM or llama.cpp server (see runners/local_qwen_setup.md)
.venv/bin/python -m experiments.runners.gradient_check --live --seed 42
```

Confirms:
- Reference agent spans 10%–90% failure over difficulty bins
- Gradient validation harness writes data/calibration/reference_gradient.json
- PASS/DEGENERATE verdict with fallback ladder on failure

### Step 3 — Cost model

```bash
.venv/bin/python -m experiments.runners.cost_model
```

Outputs:
- `results/cost_model/cost_model.json` — per-scenario cost estimates
- Printed table with $500 envelope and knob notes

### Step 4 — Fill prices.json

Open `experiments/runners/prices.json` and fill in actual API prices before the probe.
DO NOT hardcode guessed prices. Check provider pricing pages.

### Step 5 — Cost probe

Run 6–12 instances using the exp1 pipeline (Track B infrastructure + scaffold):

```bash
# After exp1_pipeline_proof/run_all.py is complete and generates data/instances/blocksworld/v1/instances.jsonl
# (this step requires Track B to have generated instances)

.venv/bin/python experiments/exp1_pipeline_proof/run_probe.py \
  --n-instances 12 \
  --budget-cap 50.0 \
  --models claude-sonnet-4-5,qwen2.5-7b-instruct \
  --architectures baseline,verifier_augmented \
  --seeds 42,123,777 \
  --exp-id exp1_probe
```

Kill-switch: halts at $50 cumulative API spend (BudgetMeter cap).

### Step 6 — Go/no-go decision

Check §15.5 pilot success criteria:

1. Reliable trace capture? (smoke test PASS)
2. Solver/reference-agent calibration artifacts? (Track B output)
3. Reference-agent gradient validation? (gradient_check PASS)
4. Matched-pair generation end-to-end? (Track B)
5. Measurable capacity dose-response on at least one pressure axis?
6. Integrity-event frequency in enriched bands > 3–5%? (probe estimate)
7. Preliminary baseline vs. verifier-augmented comparison?
8. Bounded-baseline comparison in formal domain?
9. Cost and variance estimates for power analysis?

If integrity events < 3% in enriched bands, apply adaptive pilot rule:
  Shift mass from easy/obvious bands into near-boundary/minimally-unsat bands.
  Preserve pair-type arithmetic.

---

## Go/No-Go Criteria (plan §15.4–15.5)

| Criterion | Threshold | Action if not met |
|-----------|-----------|-------------------|
| Smoke test | All 3 trajectories PASS trace validation | Fix scaffold before proceeding |
| Gradient check | PASS (10%–90% span) | Apply fallback ladder (§9.5) |
| Probe cost | Total ≤ $50 | Reduce model, reduce n_instances, switch to batch |
| Integrity event rate in enriched bands | ≥ 3–5% | Reallocate bands toward near-boundary |
| Terminal-claim distribution | All three labels observed | Check protocol wording |
| Exp 2 projected cost | ≤ $500 with caching+batch | Reduce scope or use local-only |

---

## Exact commands to reproduce all $0 steps

```bash
# From repo root, using .venv/bin/python

# 1. Smoke test
.venv/bin/python -m experiments.runners.smoke_test --exp smoke --seed 42

# 2. Gradient check (mock)
.venv/bin/python -m experiments.runners.gradient_check --mock --seed 42

# 3. Cost model
.venv/bin/python -m experiments.runners.cost_model

# 4. Exp 2 dry run
.venv/bin/python -m experiments.exp2_formal_minipilot.runner --dry-run --seed 42

# 5. All tests
.venv/bin/python -m pytest experiments/runners/tests/ -v
```

---

## Deviations to record

- If `cibd_core` schema validation is unavailable at smoke time, note it in RESULTS.md.
- If gradient check returns DEGENERATE on live model, record which fallback was applied.
- If probe reveals token counts > 2x ASSUMPTION values, update cost_model.py before Exp 2 decision.

---

## Files produced by this experiment

| File | When | Cost |
|------|------|------|
| `data/traces/smoke/*.jsonl` | Smoke test | $0 |
| `results/smoke/RESULTS.md` | Smoke test | $0 |
| `results/smoke/metrics.json` | Smoke test | $0 |
| `data/calibration/reference_gradient.json` | Gradient check | $0 (mock) / $0 (local) |
| `results/cost_model/cost_model.json` | Cost model | $0 |
| `data/traces/exp1_probe/*.jsonl` | Probe | ≤ $50 |
| `results/exp1_probe/metrics.json` | Probe | ≤ $50 |
