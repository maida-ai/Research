## Experimental Design

The pilot study is designed to produce oracle-checkable, calibrated measurements of behavioral change under pressure rather than another benchmark leaderboard. Three design choices together drive every implementation decision that follows: (1) domains and architectures that isolate specific pressure axes while sharing a common evaluation contract; (2) a uniform terminal reporting protocol that allows any architecture to express honest failure without penalty; and (3) a calibration pipeline that assigns pressure bands *after* padding and matching, so that band labels cannot be confounded with surface-form differences between instances.

### Domains, architectures, and models

**Formal planning domain.** The pilot's first domain is resource-constrained formal planning, with the recommended implementation being a resource-constrained Blocks World or FreeCell-like card-movement environment. Candidate alternatives include gridworld problems with keyed doors and inventory limits, job-shop scheduling with machine and deadline constraints, and SAT/SMT-generated dependency puzzles. The recommended choice is motivated by three properties: the state representation is compact and deterministic; a legal-action checker can validate every trajectory step against the task contract; and both resource pressure and contradiction pressure are immediately concrete—cell pressure maps directly onto the demand/supply ratio of peak-required temporary buffers to available buffers, and infeasible instances can be constructed precisely by enforcing resource-demand or dependency-cycle contradictions. The formal domain is the natural home for complexity pressure (measured by solver effort—see §Pressure calibration) and for contradiction pressure (measured by feasibility and removal distance). It also supports the strongest baseline comparison: a principled capped solver operating under the same reporting contract (see §Bounded-search baselines).

**Tool-use domain.** The pilot's second domain is a synthetic agent workflow environment in which the agent must call tools to retrieve, schedule, update, or reconcile information under explicit constraints. Candidate task families include retrieving information from a synthetic file store where some files are missing or stale; scheduling events under mutually incompatible calendar constraints; querying a mock API with conflicting records; and resolving support tickets with policy invariants. The chosen implementation controls tool responses, hides oracle state from the agent, and instruments every tool interaction. This domain isolates a different pressure cluster: tool budget (resource pressure over allowable calls), infeasible user requests (contradiction pressure), and stale or missing tool evidence. It also accommodates a small weak proxy-trap mini-condition (see §Task generation). Because the tool environment is fully controlled, feasibility labels and procedural ground truth are determined without ambiguity.

Together, the two domains guard against domain overfitting: any signal that appears only in card-sorting or only in calendar scheduling is analytically distinguishable from a signal that replicates across domains.

**Multi-agent and coordination pressure** are explicitly deferred to the full study. They introduce additional confounds — partial state, stale shared memory, asymmetric observations — that cannot be disentangled within the pilot's sample budget.

**Architectures.** The pilot uses two architectures sharing the same task API and terminal reporting contract.

1. *Baseline agent.* A standard ReAct-like or plan-act-observe agent. It uses ordinary environment observations and task-provided tools, with no additional mid-trajectory validation machinery.

2. *Verifier-augmented agent.* The same base model and task interface, augmented with additional mid-trajectory support: a legal-action checker, a resource-accounting ledger, a constraint validator, and optional state-consistency checks. The verifier condition changes what the agent can inspect *during* a run; it does not change the terminal reporting contract, the success definition, or the oracle. This uniformity is essential for testing H5 (verifier tradeoff) without confounding architecture membership with the evaluation standard.

Architecture count is kept small. The pilot is designed for effect-size estimation and instrumentation validation, not for an exhaustive architecture survey.

**Models and seeds.** The pilot uses two version-pinned models: one stronger frontier or API-accessible model and one cheaper or mid-tier model. Exact version identifiers will be recorded at run time and treated as calibration governance artifacts. Using two models from different capability tiers guards against the possibility that effects are model-specific without introducing cross-provider confounds of a larger model sweep. Each condition will be replicated across three random seeds. If budget pressure forces a trade-off, the plan is to reduce task count before reducing to fewer than three seeds, because seed variance is needed to estimate trajectory-level instability.

### Uniform terminal reporting protocol

A recurring problem in agentic evaluation is that different systems implicitly use different stopping conventions, making failure-mode comparisons uninformative. This study resolves the problem by requiring every system — LLM agents, verifier-augmented agents, and bounded-search baselines — to terminate through the same reporting interface.

The terminal report schema (reproduced in full in Appendix B) is:

```json
{
  "terminal_report": "claim_success | claim_infeasible | report_unknown",
  "evidence": "structured evidence, solution trace, or explanation",
  "budget_state": {
    "steps_used": 0,
    "tool_calls_used": 0,
    "tokens_used": 0,
    "time_used_s": 0
  }
}
```

The three labels have precise meanings that map directly onto the capacity/integrity boundary rule:

| Terminal report | Meaning | Primary failure category if wrong |
|---|---|---|
| `claim_success` | The system asserts the task goal is satisfied under all stated constraints and provides verifiable evidence. | False success = integrity error if the oracle rejects the evidence. |
| `claim_infeasible` | The system asserts that no valid solution exists under the stated constraints and provides supporting reasoning or evidence. | False infeasibility = integrity error if the oracle has a valid solution. |
| `report_unknown` | The system does not assert success or infeasibility; it reports inability to decide or solve within budget. | Capacity failure or premature abandonment — never an integrity error by itself. |

The oracle judges every terminal claim against ground truth. Architectures may differ substantially in mid-trajectory tools, ledgers, and validators; they differ only in the evidence and reasoning they can produce before submitting a terminal report. The finish line is identical.

The exact agent-facing wording used to instruct all systems about these labels is pinned as a calibration-grade artifact and reproduced verbatim in Appendix A. It is not duplicated here because any variation in wording between the appendix and the prompt actually used would be a protocol violation. The wording deliberately frames `report_unknown` as an acceptable, non-shameful response so that choosing it under genuine uncertainty reflects capacity limitation rather than a framing artifact. During the cost probe, the pilot may include one pre-registered paraphrase-sensitivity check to quantify how much the terminal-claim distribution shifts under an alternative formulation of the same semantic distinction; if that check reveals large sensitivity, protocol-wording sensitivity will be flagged as a limitation.

### Task generation and matched siblings

**Instance content.** Every task instance, regardless of domain, specifies an initial state; a goal or task contract; a legal action set or tool environment; a resource inventory (where applicable); hard constraints; a feasibility label; a reference solution where the instance is feasible; an unsatisfiability core or removal-distance estimate where the instance is infeasible; an oracle validator capable of checking arbitrary trajectories; and raw generator parameters for reproducibility.

**Matched feasible/infeasible siblings.** The central confound in any study that includes infeasible tasks is prompt surface form: if unsatisfiable instances are visibly longer, contain more constraints, or mention more entities than their feasible counterparts, any observed difference in agent behavior may reflect prompt length or surface complexity rather than the presence of a logical contradiction. The pilot eliminates this confound by constructing every infeasible instance with a paired feasible sibling padded with non-binding constraints to match on:

- prompt token count;
- visible constraint count;
- tool count;
- number of named entities;
- surface-form complexity;
- distractor count;
- action-space size where possible.

Non-binding constraints add surface complexity without changing the feasibility label. Padding happens before calibration, so calibrated pressure bands reflect the difficulty of final padded instances, not the difficulty of a raw pre-padding version.

**Pair-type budget.** The pilot allocates instances through pair types, not directly through band labels. For each 60-instance domain, the budget is 30 two-instance pairs:

| Pair type | Pairs per domain | Instances per domain | Directly controlled variable |
|---|---:|---:|---|
| Feasible ↔ minimally unsat | 12 | 24 | 12 minimally unsat instances |
| Feasible ↔ obviously unsat | 6 | 12 | 6 obviously unsat instances |
| Feasible ↔ length-inflated feasible | 4 | 8 | 4 length-only control pairs |
| Feasible ↔ distractor-padded feasible | 4 | 8 | 4 distractor-control pairs |
| Feasible ↔ weak-proxy/control feasible | 4 | 8 | 4 proxy/control pairs (conditional; see below) |
| **Total** | **30** | **60** | **18 unsat + 42 feasible instances** |

This yields 18 unsatisfiable instances and 42 feasible instances per domain. Unsatisfiable counts are direct generator targets. Feasible band labels are emergent outcomes of post-padding calibration.

**Length-only and distractor controls.** Two pair types serve as methodological controls rather than pressure manipulations. The length-inflated pairs hold task content fixed while increasing prompt length, allowing the study to separate pure length effects from constraint effects. The distractor-padded pairs add irrelevant tools, entity names, or non-binding constraints, separating surface noise from genuine resource or contradiction pressure.

**Proxy-trap mini-condition.** Proxy pressure is not a primary pilot axis. However, if `proxy_exploitation` remains listed as an integrity metric, the pilot includes a small weak proxy-trap mini-condition within the tool-use domain (6–12 instances). Examples include tasks where skipping a required process saves steps but risks false success or where tool output contains a superficially correct but stale answer. The proxy-trap label is categorical (`none | weak`). The governing rule is: *if the proxy-trap mini-condition is cut, `proxy_exploitation` is removed from primary integrity metrics and deferred to the full study.*

### Pressure calibration

**Fixed pipeline order.** The sequence from raw generation to calibrated bands is fixed and must not be reordered:

```text
1. Generate raw candidate instances.
2. Create feasible/infeasible siblings.
3. Add padding and non-binding constraints for matching.
4. Finalize the exact agent-facing prompts and tool environments.
5. Calibrate pressure on the finalized padded instances.
6. Assign calibrated bands.
7. Allocate pilot samples non-uniformly across bands.
8. Run test agents.
```

Calibration happens at step 5, after padding, because padding and non-binding constraints change solver effort and reference-agent difficulty. Calibrating before padding would assign band labels to instances that do not match what test agents actually receive.

**Raw variables and calibrated bands.** Generator parameters are manipulated as raw continuous or ordinal variables: slack, peak buffer demand, constraint count, plan length, branching factor, removal distance, tool budget, and so on. These raw variables are too domain-specific for cross-domain comparison. Reporting therefore uses calibrated bands. Within-domain raw-unit plots are acceptable; cross-domain or cross-axis comparisons use calibrated bands only.

**Band ladders nested within axes.** Because pressure bands are structurally nested within pressure types, the pilot does not treat `pressure_type × pressure_band` as a fully crossed design. Three axis-specific ladders are used:

*Feasibility/contradiction ladder:*
`easy_feasible` → `medium_feasible` → `hard_feasible` → `near_boundary_feasible` → `minimally_unsat` → `obviously_unsat`

*Resource/complexity ladder:*
`low` → `medium` → `high` → `near_boundary`

*Proxy ladder (if included):*
`none` → `weak`

The bands `minimally_unsat` and `obviously_unsat` exist only on the feasibility/contradiction ladder; fitting models that treat them as levels of a generic band factor shared with resource or complexity pressure would populate structurally empty cells. Cross-axis comparisons are restricted to the shared conceptual bands `low / medium / high / near_boundary`.

**Formal-domain calibration.** In the formal planning domain, pressure bands are defined by the effort required by a fixed, version-pinned solver—specifically OR-Tools CP-SAT [@ortools2024cpsat], which provides interpretable solver statistics (conflicts, branches, and proof-of-infeasibility cost) and well-defined status codes (`OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `UNKNOWN`) that translate cleanly to feasibility labels. Alternative calibration measures include A* node expansions, BFS/IDA* expansions, minimum plan length, and removal distance for infeasible instances. The critical governance rule is:

> Do not use test-agent behavior to define pressure.

Pressure bands must be defined by the solver, not by the rate at which the test agents fail. Using test-agent failure rates to assign bands would make the dose-response curves circular.

**Tool-use calibration.** The tool-use domain requires a different calibration strategy because no general-purpose solver can evaluate arbitrary natural-language-grounded tool workflows. The pilot uses a frozen reference agent to define difficulty quantiles. The candidate reference agent is Qwen2.5-7B-Instruct [@qwen2024qwen25; @qwen2024qwen25hf], run in a version-pinned local container with temperature 0, a custom single-shot plan-to-JSON-action-list scaffold, deterministic execution, and parser repair but no self-reflection. The reference agent serves exclusively for calibration; it is never a test agent.

Before pinning, the reference agent must satisfy a gradient validation requirement: sweeping the generator parameter range must confirm that reference-agent performance spans roughly 10%–90% failure. A reference agent that fails almost all tasks or succeeds almost all tasks produces degenerate quantiles. If the candidate fails this check, fallback options include using a stronger pinned reference model, softening the calibration metric to partial-credit constraint satisfaction, or measuring success within N retries rather than first-shot success. Once a nondegenerate gradient is confirmed, the following components are pinned for the project lifetime: model hash, runtime image, scaffold code, decoding parameters, retry policy, and tool environment version. Calibration outputs (reference-agent failure quantile, step quantile, tool-error quantile, constraint-satisfaction quantile, and verification-failure quantile) define bands; they are not reported as comparative results.

**Calibration governance.** Every component that contributes to band assignment must be pinned and versioned: solver version, timeout, and hardware class; reference-agent model hash, runtime image, decoding parameters, scaffold code, and tool environment version; exact terminal-protocol prompt wording; report schema and JSON validation rules; oracle classification table; signal-detection aggregation script; and random seeds. Calibration artifacts will be stored and versioned alongside the benchmark instances.

### Bounded-search baselines

The bounded-search baselines test a specific empirical claim — that bounded compute explains bounded-search failure patterns — rather than claiming to be a general performance upper bound.

**Baseline A: capped solver (formal domain only).** A real solver such as OR-Tools CP-SAT or a domain-specific FreeCell or Blocks World solver is run with a matched compute cap (time limit, node limit, or conflict limit). Under the shared reporting protocol, the solver will either return a feasible solution (`claim_success`), prove infeasibility (`claim_infeasible`), exhaust its budget (`report_unknown`), or return an explicit unknown status (`report_unknown`). Crucially, the solver never asserts success without oracle-validated evidence. Expected behavior under pressure: budget exhaustion and `report_unknown`. Because its terminal claims derive from validated search or infeasibility proofs, the capped solver primarily exposes capacity limits; we avoid the stronger claim that integrity errors are impossible by construction (see Threats to Validity).

**Baseline B: scripted search agent (formal and tool-use where feasible).** A same-action-interface scripted agent — random-restart, heuristic search, or enumerative constraint search — uses the same action API, observes the same environment responses, shares the same step and tool budget, and reports only through the shared terminal protocol. It cannot bypass the oracle. Expected behavior: capacity degradation (budget exhaustion, `report_unknown`) under pressure; low or zero integrity degradation when the reporting and verification contract is enforced.

**Claim scope.** The pilot states baseline scope explicitly: the solver-based bounded-rational contrast is formal-domain only; the same-action-interface contrast is formal-domain plus tool-use domain if the synthetic tool environment can be represented as a finite action search space. If the scripted tool-use baseline is infeasible or cut, capacity/integrity dissociation claims involving bounded baselines must be restricted to the formal domain.

**Recommended contrast phrasing.** For the full-scope comparison: *Given matched task instances, reporting protocols, verification contracts, and compute-budget caps, bounded-search baselines primarily expose capacity degradation under increasing pressure, whereas LLM agents may additionally exhibit integrity degradation.* For the same-API subset: *Given a shared action interface, reporting protocol, verification contract, and matched compute budget, bounded-search baselines primarily exhibit capacity degradation under increasing pressure, whereas LLM agents may additionally exhibit integrity degradation.* The word "may" is used throughout the methods and hypothesis sections; "do" would require empirical results the pilot has not yet generated.

### Trace instrumentation

Every run is captured as a structured sequence of typed events, implemented on a generic trace instrumentation framework. The event types are: (1) *run-level metadata* — run identifier, task identifier, domain, pressure axis, band ladder, pressure band, model identifier, architecture, system type, seed, and budget allocations; (2) *action events* — step index, state hash before and after, action taken, legality flag, and whether the resulting state is novel; (3) *tool events* — tool name, argument hash, status (ok/error/timeout), latency, result hash, and a semantic result label (missing/conflicting/stale/normal); (4) *verification events* — checker type (goal, constraint, resource, procedure, proxy, or terminal oracle), pass/fail, violated constraint identifiers, and oracle feasibility label; (5) *resource-accounting events* — resource name, available quantity, used quantity, peak-required estimate, and slack; (6) *representation metrics* — windowed notation-compression and entity-density statistics (symbol ratio, entity density, characters per state reference, failure marker rate) timestamped with elapsed budget fraction and remaining step budget; and (7) *termination events* — the terminal report, stop reason, oracle outcomes, and all derived classification fields.

The termination event schema (Appendix B) is the primary record linking terminal claims to oracle outcomes:

```json
{
  "event_type": "termination",
  "terminal_report": "claim_success|claim_infeasible|report_unknown",
  "stop_reason": "agent_report|budget_exhausted|loop_guard|error",
  "oracle_success": false,
  "oracle_feasible": true,
  "claim_correct": false,
  "capacity_failure": false,
  "integrity_error": true,
  "classification": "integrity_false_infeasibility",
  "signal_detection_role": "false_alarm"
}
```

Full schemas for all event types, the normative trajectory classification table, and the reference classifier pseudocode appear in the appendices.
