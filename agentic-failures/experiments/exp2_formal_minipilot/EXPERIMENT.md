# Experiment 2 — Formal Domain Mini-Pilot

**Status:** Design only — NOT executed at scale  
**Prerequisite:** Experiment 1 probe must PASS all go/no-go criteria  
**Budget target:** $3,000 (hard stop: $5,000)  
**Date:** 2026-06-11  

---

## Scope

| Dimension | Value |
|-----------|-------|
| Domain | Formal (resource-constrained Blocks World) |
| Instances | 60 |
| Architectures | 2 (baseline agent, verifier-augmented agent) |
| Models | 2 (1 stronger API model + 1 local open-weights model) |
| Seeds | 3 |
| Total trajectories | 60 × 2 × 2 × 3 = **720** |

Instance allocation follows plan §15.2.1 (30 pairs per domain):
- 12 feasible ↔ minimally-unsat pairs (24 instances)
- 6 feasible ↔ obviously-unsat pairs (12 instances)
- 4 feasible ↔ length-inflated control pairs (8 instances)
- 4 feasible ↔ distractor-padded control pairs (8 instances)
- 4 feasible ↔ weak-proxy/control pairs (8 instances; reassigned to length/distractor if proxy not included)
- Total: 60 instances, 18 unsat + 42 feasible

Calibrated band targets (post-padding, §15.2.2):
- easy_feasible: ~6 instances
- medium_feasible: ~9 instances
- hard_feasible: ~12 instances
- near_boundary_feasible: ~15 instances
- minimally_unsat: ~12 instances
- obviously_unsat: ~6 instances

---

## Hypotheses tested (formal-domain slice)

### H1 — Controlled dose-response (capacity)

> Under increasing pressure (resource constraint tightness), both architectures show
> monotonically increasing capacity failure rates (step explosion, budget exhaustion,
> honest_unknown reports).

Primary metrics: `step_explosion`, `budget_exhaustion`, `honest_unknown` rate per band.
Statistical model: GEE with pressure_band as ordered factor, architecture as categorical,
                   seed as clustering variable (plan §14.2).

### H2 — Integrity threshold (formal domain)

> Integrity errors (false_success, false_infeasibility) concentrate in near-boundary and
> minimally-unsat bands rather than distributing uniformly.

Primary metric: `integrity_error` indicator per trajectory.
Analysis: paired feasible/infeasible comparison; SDT hit/false-alarm over matched siblings.
Statistical model: mixed logistic regression with pressure_band × architecture interaction (plan §14.3).

### H3 — Bounded-search contrast

> The capped-solver baseline (Track B, formal domain) primarily shows capacity degradation
> (budget exhaustion, search cutoff) while LLM agents additionally show integrity errors.

Analysis: compare integrity_error rate between LLM architectures and the capped-solver baseline.
Note: both share the same terminal reporting contract and oracle.

### H4 — Near-boundary maximum

> Integrity event rate is highest in the near_boundary_feasible and minimally_unsat bands,
> lower in easy_feasible and obviously_unsat.

Analysis: per-band integrity event counts; band × architecture interaction.
Adaptive rule: if near-boundary integrity rate < 3% after pilot, reallocate bands (plan §15.2.2).

---

## Analysis plan

### Primary analyses

1. **Capacity dose-response curve** (H1):  
   Plot capacity failure rate vs. pressure band for each architecture.  
   GEE model: capacity_failure ~ pressure_band × architecture, clustered by seed.

2. **Integrity rate by band** (H2, H4):  
   Plot integrity_error indicator vs. pressure band.  
   Mixed logistic: integrity_error ~ pressure_band × architecture + (1|seed).

3. **SDT analysis** (plan §14.4):  
   For claim_infeasible responses over matched feasible/infeasible siblings:  
   hit_rate, false_alarm_rate, d', criterion_c per architecture per band.  
   Small-sample correction: +0.5 to each cell of 2×2 table (spec §8).

4. **Bounded-search contrast** (H3):  
   Compare integrity_error rates: LLM agents vs. capped-solver baseline.  
   Matched-pair analysis: Δmetric = metric(pressure_instance) - metric(matched_control_instance).

### Pre-registered interactions

- pressure_band × architecture (primary)
- pressure_band × model_id (secondary)
- architecture × model_id (exploratory)
Three-way and four-way interactions are exploratory only (plan §14.1).

### Exclusions

- Trajectories with unresolved oracle_feasible excluded from SDT denominators.
- Trajectories with trace validation failures excluded from all analyses (noted in RESULTS.md).
- Proxy exploitation metrics not applicable to formal domain unless proxy-trap pairs included.

---

## Cost estimate

Reference: `results/cost_model/cost_model.json` scenarios `exp2_formal_minipilot_*`.

With placeholder prices (all $0): $0.00 estimated.  
With caching + batch: cost estimate pending probe (§15.4).

**This experiment runs only if:**
- Exp 1 probe confirms cost-per-trajectory estimates that bring 720 trajectories inside $3,000.
- Cheap infra (local model + batch API) is confirmed available.
- Kill-switch is configured.

---

## Kill-switch configuration

All live runs require:
```bash
.venv/bin/python -m experiments.exp2_formal_minipilot.runner \
  --budget-cap 3000.0 \
  --confirm-live \
  --seed 42
```

`--budget-cap` is REQUIRED for any live run.  
`--confirm-live` is REQUIRED (explicit human confirmation flag).  
Without both flags, runner exits with an error.

---

## Files produced by this experiment

| File | Source |
|------|--------|
| `data/instances/blocksworld/v1/instances.jsonl` | Track B (exp1_pipeline_proof) |
| `data/traces/exp2_formal_minipilot/<run_id>.jsonl` | This experiment |
| `results/exp2_formal_minipilot/metrics.json` | This experiment |
| `results/exp2_formal_minipilot/RESULTS.md` | This experiment |

---

## Reproducibility

Seeds: 42, 123, 777 (plan §7.5).  
Models: to be version-pinned at run time (record model hashes in governance.json).  
All instance generation deterministic via Track B seeds.  
All agent runs deterministic via scaffold seeds.  

Run command (when ready):
```bash
.venv/bin/python -m experiments.exp2_formal_minipilot.runner \
  --budget-cap 3000.0 \
  --confirm-live \
  --seeds 42,123,777 \
  --models claude-sonnet-4-5,qwen2.5-7b-instruct \
  --architectures baseline,verifier_augmented \
  --exp-id exp2_formal_minipilot
```
