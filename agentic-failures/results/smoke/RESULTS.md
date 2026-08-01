# Smoke Test Results — exp_id=smoke

**Date:** 2026-06-11  
**Seed:** 42  
**Model:** MockModel (deterministic, $0)  
**Trajectories:** 3  
**Validation status:** ALL PASS  
**BudgetMeter source:** cibd_core  
**cibd_core available:** True  

## Setup

3 Blocks World instances × 3 MockModel behaviors (solves_easy, claims_infeasible, honest_unknown).
Each run goes through the full path: scaffold -> trace JSONL with model_io events
-> terminal report -> (cibd_core validation if importable).

## Results

| # | Run ID | Behavior | Terminal Report | Stop Reason | Steps | Tokens | Wall (s) |
|---|--------|----------|----------------|-------------|-------|--------|----------|
| - | smoke-run-000-solves_easy-seed42 | solves_easy | claim_success | agent_report | 0 | 135 | 0.0 |
| - | smoke-run-001-claims_infeasible-seed42 | claims_infeasible | claim_infeasible | agent_report | 1 | 275 | 0.0 |
| - | smoke-run-002-honest_unknown-seed42 | honest_unknown | report_unknown | budget_exhausted | 5 | 725 | 0.0 |

## Trace Validation

| Run ID | Status | Events | Metadata | model_io | Termination | Classification | Notes |
|--------|--------|--------|----------|----------|-------------|----------------|-------|
| smoke-run-000-solves_easy-seed42 | PASS | 4 | Y | Y | Y | success | cibd_core schema validation: PASS |
| smoke-run-001-claims_infeasible-seed42 | PASS | 6 | Y | Y | Y | correct_infeasibility_report | cibd_core schema validation: PASS |
| smoke-run-002-honest_unknown-seed42 | PASS | 12 | Y | Y | Y | capacity_unknown_feasible | cibd_core schema validation: PASS |

## Cost

Total cost: $0.00 (MockModel — deterministic, no API calls).

## What this confirms

- Scaffold -> trace -> terminal report pipeline runs end-to-end.
- model_io events are written verbatim for each model call (cost doctrine #1).
- Run metadata is written as line 1 of each trace.
- Termination event is written as the final event of each trace.
- Harness submits report_unknown on budget/loop exhaustion (no extra pressure framing).
- Terminal protocol wording is included verbatim in the system prompt.

## Deviations

- Oracle is a lambda (mock): real oracle will be CP-SAT (exp1_pipeline_proof/Track B).
- cibd_core schema validation: active.
- BudgetMeter: cibd_core.

## Reproduce

```bash
.venv/bin/python -m experiments.runners.smoke_test --exp smoke --seed 42
```

From repo root. All random seeds: 0 (MockModel is deterministic regardless).
