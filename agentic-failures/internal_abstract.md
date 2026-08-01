## Measuring how AI agents fail under pressure — capacity vs. integrity

**Project:** Constraint-Induced Behavioral Destabilization (CIBD) — a controlled dose-response framework for trace-level failure modes in agentic AI systems.

We are looking to run the first experiments and complete the writeups for a study on agent reliability. The framework, formal definitions, metric dictionary, and a pre-specified pilot design are complete, and the task-generation, calibration, and bounded-baseline harness already runs end-to-end on a formal planning domain. What remains is the part that produces publishable results: the LLM pilot and its analysis. If you join now, you are generating the first data, not building infrastructure from scratch.

## The question

Agent evaluations almost universally report one number: did the task complete. That number cannot distinguish an agent that exhausted its budget and said so from one that fabricated a solution, silently relaxed a constraint, or declared a solvable task impossible. These are different failures with opposite fixes — more compute and better search for the first; verification and contract enforcement for the rest — and aggregate success rates hide which one you have.

CIBD treats task pressure as an independent, externally calibrated variable and measures how agent behavior changes at the trace level as pressure rises. The central pre-specified hypothesis is a dissociation: bounded-search baselines degrade in *capacity* (loops, exhaustion, honest "unknown" reports), while LLM agents may additionally degrade in *integrity* (false success, false infeasibility, constraint omission), and that this crossing concentrates near the feasibility boundary.

## What makes it methodologically clean

Three design choices are doing the load-bearing work, and they are what a careful collaborator will care about:

1. **Calibration without circularity.** Difficulty bands are defined by fixed solver effort or a frozen reference agent — never by the behavior of the systems under test. Matched feasible/infeasible sibling tasks control for prompt length and surface complexity.
2. **A uniform terminal contract.** Every system — LLM agents, verifier-augmented agents, capped solvers, scripted search — must end each run with exactly one of `claim_success`, `claim_infeasible`, or `report_unknown`, judged by an oracle. `report_unknown` is a first-class, non-penalized option, so honest uncertainty is never miscounted as an integrity error.
3. **Signal-detection scoring.** Infeasibility claims are scored jointly across feasible and infeasible siblings (hit rate vs. false-alarm rate), separating genuine discrimination from "claim impossible whenever stuck" guessing.

The framework is deliberately deflationary: pressure is a property of the task environment, destabilization is read entirely from execution traces, and no claims are made about subjective states.

## Current status (honest)

- Framework, definitions, metric dictionary, normative oracle classification, and pinned protocol artifacts: **complete.**
- Generation → sibling-matching → padding → calibration → banding → bounded-baseline pipeline: **implemented and running** on a formal planning domain (resource-constrained Blocks World).
- A pipeline-proof run has already surfaced and is fixing two infrastructure bugs (unsat-instance generation; cross-system compute-budget equivalence) — exactly the failures a probe is supposed to catch before any spend.
- LLM pilot: **not yet run.** This is the next milestone and the core of the collaboration.

## Contributions we're looking for

- **Experimental execution & infrastructure** — run the pilot across two domains, two architectures, two models (~1,400 trajectories); extend the harness and trace tooling.
- **Study design & statistics** — finalize the mixed-effects and signal-detection analysis, power analysis from pilot variance, and the OSF pre-registration.
- **Domain expansion** — add a tool-use domain and/or further formal domains (scheduling, routing, dependency puzzles) with their calibration pipelines.
- **Later phases (optional)** — multi-agent coordination pressure; mitigation experiments (unsat-core feedback, trace-gated early stopping, refusal calibration).
