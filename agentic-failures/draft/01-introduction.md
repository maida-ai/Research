## Abstract

Agentic AI systems are increasingly evaluated on end-to-end task completion, but
final success rates obscure how agents fail as tasks become resource-scarce,
search-intensive, or near the feasibility boundary. We introduce a controlled
dose-response framework for studying **constraint-induced behavioral
destabilization** (CIBD): trace-level changes in agent behavior as independently
calibrated task-pressure axes are increased. The framework generates matched
feasible and infeasible task instances in formal planning and tool-use domains,
calibrates pressure on final padded instances using fixed solver effort or
frozen-reference-agent quantiles, and measures dependent variables from execution
traces rather than from final answers alone. We distinguish **capacity
degradation**—loops, step explosion, budget exhaustion, abandonment, tool
overuse, and reasoning/action imbalance—from **integrity degradation**—false
success, false infeasibility, constraint omission, state inconsistency,
procedural violation, and proxy exploitation. All systems are bound to a uniform
terminal reporting protocol: `claim_success`, `claim_infeasible`, or
`report_unknown`. This shared contract separates honest bounded failure from
incorrect task-contract claims, enabling clean classification of outcomes without
inflating integrity-error counts. Infeasibility claims are further evaluated
through a signal-detection layer that reports hit rate, false-alarm rate,
discrimination, and response bias, so correct infeasibility reports are not
conflated with pressure-induced guessing. Given matched task instances, reporting
protocols, verification contracts, and compute-budget caps, bounded-search
baselines primarily expose capacity degradation under increasing pressure, whereas
LLM agents may additionally exhibit integrity degradation. We present the
framework, its formal definitions, and a pre-registered pilot design as
contributions; no empirical results are reported here. This deflationary
measurement layer supports future work on verifier-guided and
trace-gated agentic reliability without attributing subjective experience to the
systems under study.

---

## Introduction

### Motivation

Agentic AI systems are now routinely deployed on tasks that require multi-step
planning, tool use, and constraint satisfaction over extended horizons. The
dominant evaluation paradigm asks a single question: did the agent complete the
task? End-to-end success rates are easy to interpret and compare, but they
compress failure into a single bit. When a task is hard enough that an agent
fails, the success rate alone cannot distinguish an agent that exhausted its
budget honestly from one that fabricated a solution, silently relaxed a
constraint, or claimed infeasibility without grounds. As tasks become
resource-scarce, search-intensive, or positioned near the satisfiability boundary,
the internal structure of failure becomes substantively important: not just
whether the agent failed, but *how* it failed and *what contract-level
commitments it violated* on the way to failing.

Recent work has documented the texture of agentic failure beyond success
rate. Studies of LLM overthinking show that agents may expand reasoning without
improving outcomes [@cuadron2025overthinking]. Taxonomies of multi-agent failure
identify cooperation breakdown, state inconsistency, and cascading errors as
distinct failure modes [@cemri2025multiagent]. Procedure-aware evaluations
demonstrate that agents can achieve nominal task completion while violating the
procedural requirements that make completion meaningful
[@cao2026procedure]. Collectively this work establishes that failure mode matters,
not only failure frequency. What remains missing is a controlled methodology that
isolates how failure mode *changes* as a function of independently calibrated
task pressure—one that makes the pressure variable explicit, measurable, and
separable from test-agent behavior.

### The central question

This paper is organized around a single operational question:

> How does agent behavior change as task pressure is independently and
> systematically increased?

By "task pressure" we mean an externally imposed property of the task
environment: the ratio of required resources to available resources, the
complexity of the search required to find a valid plan, or proximity of the
task instance to the satisfiability boundary. By "behavior" we mean
trace-observable properties of agent execution: action counts, tool-call
patterns, termination decisions, and truth-value of terminal claims as judged
by an independent oracle. The question is behavioral and operational throughout.
No claims are made about internal states, subjective experience, or anything that
cannot be read from execution traces.

### Constraint-induced behavioral destabilization

We introduce the term **constraint-induced behavioral destabilization** (CIBD) to
name the phenomenon under study: a measurable increase in unstable, invalid, or
goal-divergent behaviors as externally imposed task pressure increases. The
definition is intentionally deflationary. CIBD is a behavioral construct measured
through traces, oracle outcomes, and validation events. It does not attribute
subjective stress, frustration, emotional instability, or any other mental-state
property to the model. "Destabilization" refers to a change in the statistical
distribution of trace-observable behaviors, not to any inner state of the system.
Pressure is a property of the task environment; destabilization is a property of
the resulting trajectory.

### The capacity/integrity dissociation

The central conceptual contribution of this paper is a distinction between two
separable forms of behavioral change under pressure.

**Capacity degradation** is failure caused by bounded search, limited budget, or
inability to maintain and explore the required state space while preserving the
task contract. Its canonical expressions include loops, step explosion, budget
exhaustion, honest `report_unknown` due to search failure, premature abandonment
without a false claim, tool overuse, and reasoning/action imbalance. Capacity
degradation is expected under pressure: any system with bounded compute will
eventually exhaust its resources on a sufficiently hard task. Bounded-search
baselines—capped solvers and scripted search agents—are expected to fail
primarily in this way.

**Integrity degradation** is failure in which the agent violates, relaxes, omits,
misreports, or bypasses the task contract. Its canonical expressions include false
success claims, false infeasibility claims, constraint omission, inconsistent
state reporting, procedural violations, and proxy exploitation. Integrity
degradation is not expected of any well-designed bounded-search system: a capped
solver that runs out of time should report failure or unknown, not fabricate a
solution. LLM agents, by contrast, generate their terminal claims from a learned
distribution and may produce incorrect contract-level assertions even when a
valid path exists or when no valid path was demonstrated.

The core empirical question is whether these two failure modes are empirically
dissociable under controlled pressure. The recommended contrast, using the
broader form applicable when baselines do not share the same action API, is:

> Given matched task instances, reporting protocols, verification contracts, and
> compute-budget caps, bounded-search baselines primarily expose capacity
> degradation under increasing pressure, whereas LLM agents may additionally
> exhibit integrity degradation.

This phrasing uses "may" deliberately. The dissociation is a pre-registered
hypothesis of the pilot study, not an established empirical result. The framework
provides the experimental and measurement infrastructure to test it.

The dissociation matters for reliability engineering. If LLM agents fail only by
honest exhaustion, the safety-relevant intervention is increased compute or better
search. If they fail additionally by integrity violation, the intervention requires
a different layer: contract enforcement, oracle-checked verification, or trace-gated
early stopping. Without a controlled measurement framework, the two failure modes are
conflated in aggregate success rates and remain invisible to system designers.

### Contributions

This paper makes the following contributions:

1. **Framework and formal definitions.** We introduce CIBD as a behavioral
   construct, define capacity and integrity degradation with a boundary rule based
   on terminal claims and oracle outcomes, and specify the pressure model—resource,
   complexity, and contradiction axes with demand/supply formalization—that
   allows independent variation of pressure type and level.

2. **Uniform terminal reporting protocol and oracle contract.** We specify a
   shared reporting interface (`claim_success`, `claim_infeasible`,
   `report_unknown`) that all systems—LLM agents and bounded-search baselines
   alike—must use to terminate. The oracle applies the same verification contract
   to all terminal claims. This design prevents integrity-error counts from being
   inflated by asymmetric evaluation and prevents capacity errors from being
   disguised as integrity success.

3. **Signal-detection analysis for infeasibility claims.** We introduce a
   signal-detection layer that analyzes `claim_infeasible` responses jointly
   across matched feasible and infeasible sibling instances. This separates
   discrimination—whether the system can distinguish feasible from infeasible
   tasks—from response bias—whether the system defaults to confident infeasibility
   claims under pressure. Lucky guessing is thereby distinguished from genuine
   infeasibility detection.



4. **Pre-registered pilot design with calibrated pressure and non-uniform
   allocation.** We present a 2×2×2×3 pilot (two domains, two architectures, two
   models, three seeds) with thirty matched pairs per domain and
   non-uniform allocation that concentrates samples near the hypothesized
   integrity-failure boundary. Pressure is calibrated using solver effort or
   frozen-reference-agent quantiles applied to final padded instances, so
   calibration does not depend on test-agent behavior. The pilot design—including
   research questions RQ1–RQ7 and hypotheses H1–H6—is pre-registered as a
   contribution; results are not yet available.

5. **Trace-level dose-response curves as primary outcome.** Rather than reporting
   aggregate success rates, the framework takes per-metric dose-response curves as
   primary results, enabling separate characterization of how capacity metrics and
   integrity metrics each respond to increasing pressure along each axis.

### Paper roadmap

Section 2 reviews related work on agent failure taxonomies, planning benchmarks,
procedure-aware evaluation, specification gaming, and signal detection theory,
positioning the CIBD framework relative to each. Section 3 presents the framework
in full: formal definitions of CIBD, capacity degradation, integrity degradation,
the boundary classification rule, and the pressure model. Section 4 describes the
experimental design, including domain selection, architectures, task generation,
the uniform terminal reporting protocol, and matched feasible/infeasible sibling
construction. Section 5 defines the metric suite and oracle classification
procedure, including the signal-detection layer for infeasibility claims. Section
6 presents the pre-registered pilot scope, allocation tables, statistical models,
and pilot success criteria. Section 7 addresses threats to validity. Section 8
discusses the framework's implications for verifier-guided and trace-gated
reliability, its limitations as a protocol paper, and directions for the full
study.
