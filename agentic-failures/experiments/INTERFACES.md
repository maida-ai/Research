# CIBD experiments — interface contract (binding for all experiment code)

Normative sources: `../cibd_phase1_metric_oracle_spec_v0_1.md` (spec) and
`../cibd_agentic_ai_research_plan_v0_4.md` (plan). Where this file and the spec
disagree, the spec wins.

## Environment

- Python: `.venv/bin/python` at repo root (3.12). Installed: `ortools` (CP-SAT),
  `pytest`. **No other third-party deps** without orchestrator approval — use
  stdlib (`dataclasses`, `json`, `argparse`, `hashlib`, `random`, `statistics`).
- All scripts must run via `.venv/bin/python` from the repo root and accept
  `--seed` where randomness exists. Record seeds, versions, and the exact run
  command in every results dump.

## Cost doctrine (binding, from project owner)

1. **Generation and evaluation are permanently separated.** Rollout generation
   is the expensive, irreversible step; scoring is a local recompute. Persist
   FULL raw trajectories — every model input and output verbatim, not just
   metrics — so oracle/metric revisions never require re-runs.
2. **Oracle is deterministic code.** No LLM-as-judge anywhere in the formal
   domain. Domain chosen partly on checkability: resource-constrained
   **Blocks World with limited table buffers** (fully programmatic validator
   and CP-SAT ground truth).
3. **Run order: smoke-test (1–3 trajectories) → pilot at minimum N → power
   calculation from pilot variance → main run.** Never scale against an
   unverified harness.
4. **Budget kill-switch.** The harness meters cumulative token cost and
   hard-halts at a configured cap. Required for any path that can call a paid
   API. Also hard max-turns / max_tokens / stop conditions per trajectory.
5. **Role tiering.** Agent-under-test = independent variable (full strength).
   Any secondary model-based role defaults to the cheapest tier (Haiku-class).
   With a deterministic oracle there should be ~no secondary model calls.
6. **Prompt caching:** static system prompt + domain spec is a fixed prefix,
   structured for provider prompt caching. **Batching:** any non-interactive
   scoring pass must be batchable (50% off) — but prefer code oracle (100% off).
7. **Local-first:** an RTX 4090 (24 GB) is available. The frozen reference
   agent (Qwen2.5-7B-Instruct, plan §9.5) and the cheaper test model run
   locally with open weights. Local runs are cheaper AND methodologically
   better (version-pinned weights). No GPU execution in this build phase —
   ship configs + instructions; do not install vLLM/llama.cpp now.

## Directory layout

```
experiments/
  INTERFACES.md            (this file)
  cibd_core/               package: schemas, classifier, SDT, store, budget
    tests/
  exp1_pipeline_proof/     formal-domain pipeline (generator→calibration→bands)
  exp2_formal_minipilot/   design + dry-runnable runner; NOT executed at scale
  runners/                 agent scaffolds, mock model, local-Qwen config,
                           gradient check harness, cost model
data/
  instances/blocksworld/v1/   instances.jsonl, manifest.json
  calibration/blocksworld/v1/ calibration.jsonl, bands.json, governance.json
  traces/<exp_id>/            <run_id>.jsonl  (FULL raw trajectories)
results/
  exp1_pipeline_proof/        metrics.json, RESULTS.md
  exp2_formal_minipilot/      (empty until run)
```

## Shared data shapes (JSON, one object per line in .jsonl)

Modules must NOT import across tracks during the build; conform to these dict
shapes instead. Integration is verified afterward by `cibd_core` validators.

### Instance record (`data/instances/.../instances.jsonl`)

```json
{
  "instance_id": "bw-v1-0001",
  "domain": "formal",
  "pair_id": "pair-0001",
  "pair_type": "feasible_min_unsat | feasible_obvious_unsat | feasible_length | feasible_distractor | feasible_proxy_control",
  "sibling_role": "feasible | unsat | control",
  "initial_state": {},
  "goal_state": {},
  "action_space": {},
  "resources": {"table_buffers": 2},
  "hard_constraints": [],
  "padding": {"nonbinding_constraints": 0, "distractors": 0, "prompt_pad_tokens": 0},
  "feasible": true,
  "reference_solution": [],
  "removal_distance": null,
  "generator_params": {"seed": 0},
  "prompt_token_estimate": 0
}
```

### Calibration record (`data/calibration/...`)

Per instance: solver name+version, time limit, status
(`OPTIMAL|FEASIBLE|INFEASIBLE|UNKNOWN`), conflicts, branches, wall time,
min plan length (if found), assigned `pressure_band`
(feasibility/contradiction ladder per plan §9.3). `governance.json` pins
solver version, timeout, hardware class, seeds (plan §9.6).

### Terminal report — spec §3 verbatim

```json
{"terminal_report": "claim_success | claim_infeasible | report_unknown",
 "evidence": "...",
 "budget_state": {"steps_used":0,"tool_calls_used":0,"tokens_used":0,"time_used_s":0}}
```

### Termination event — spec §10 verbatim (all derived fields from the
normative classifier; `classification` and `signal_detection_role` values
exactly as in spec §6).

### Trace file (`data/traces/<exp_id>/<run_id>.jsonl`)

Line 1: run-level metadata (plan §16.1 schema, includes model_id,
architecture, system_type, seed, budget). Then ordered events:
`agent_action` (§16.2), `tool_call` (§16.3), `verification` (§16.4),
`resource_state` (§16.5), `representation_metric` (§16.6), plus
**`model_io` events** (cost doctrine #1):

```json
{"event_type": "model_io", "step": 3, "request": {"system": "...", "messages": ["..."], "max_tokens": 0}, "response": {"text": "...", "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}, "latency_ms": 0, "cost_usd": 0.0}
```

Final event: `termination` (spec §10). A trace with no termination event is
invalid.

### Budget meter

`cibd_core.budget.BudgetMeter(cap_usd, price_table)` — `charge(model_id,
in_tok, out_tok, cached_tok)` accumulates; raises `BudgetExceeded` at cap;
state persisted to `data/traces/<exp_id>/_budget.json` after every charge.
Local models cost $0 but still meter tokens.

## Track ownership

- **A (`cibd_core`)**: schemas + validators for every shape above; terminal
  classifier (spec §7, tested against every row of spec §6 + §11 checks);
  SDT aggregation (spec §8, loglinear +0.5 correction); JSONL trace
  store/reader; BudgetMeter; `validate.py` CLI that checks instance/trace
  files against the shapes.
- **B (`exp1_pipeline_proof`)**: Blocks World generator, sibling/padding/
  matching, CP-SAT oracle + calibration, band assignment, pair-type
  allocation per plan §15.2.1 (12/6/4/4/4 pairs), capped-solver and
  scripted-search baselines emitting terminal reports; `run_all.py`
  executes everything locally and dumps to data/ + results/.
- **C (`runners` + `exp2`)**: baseline & verifier-augmented scaffolds with
  the pinned terminal protocol (appendix A wording verbatim), MockModel
  smoke harness (3 trajectories, full traces, $0), local Qwen-on-4090
  config + gradient-check harness (not executed), API client skeleton with
  caching/batch hooks, cost model, EXPERIMENT.md pre-registrations for
  Experiment 1 (<$500) and Experiment 2 (formal mini-pilot, design only).
