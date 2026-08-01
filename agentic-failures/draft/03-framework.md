## The CIBD Framework

### Definitions

**Constraint-induced behavioral destabilization (CIBD)** is a measurable increase in unstable, invalid, or goal-divergent behavior as externally imposed task pressure increases. The term is intentionally behavioral: it does not attribute subjective stress, emotion, or experience to the model. CIBD is measured through execution traces, oracle outcomes, and validation events. Pressure is a property of the task environment; destabilization is a property of the observed trace.

**Behavioral response under pressure** refers to the trajectory-level pattern an agent exhibits following repeated failed attempts, resource conflict, or near-boundary uncertainty. Observable instances include repeated planning without action, loops, invalid moves, premature abandonment, false success claims, false infeasibility claims, constraint relaxation, and proxy exploitation. The concept captures the class of trace-level responses that warrant measurement; it makes no claim about the internal process generating them.

**Unstable inference regime (UIR)** is a region of task pressure where small perturbations in prompt, state, tool feedback, seed, or architecture produce disproportionately large changes in behavior, success rate, or failure mode. Candidate measurements for detecting the UIR include:

- variance in outcome across seeds at the same calibrated pressure band;
- sensitivity to prompt paraphrase holding constraints constant;
- sensitivity to distractor constraints that do not change task feasibility;
- divergence in action trajectories across matched architectures;
- elevated probability of integrity failure near calibrated pressure boundaries.

The UIR is not an assumption of the framework; it is a predicted region whose existence and location may be examined empirically.

---

### Capacity versus integrity degradation

The central analytic distinction of the CIBD framework is between two separable modes of failure under pressure.

**Capacity degradation** is failure caused by bounded search, limited budget, or inability to maintain and explore the required state space while preserving the task contract. Examples include:

- loops and repeated action reversals;
- step explosion relative to the reference solver effort;
- budget exhaustion (steps, tool calls, tokens, or time);
- honest `report_unknown` due to budget exhaustion or search failure;
- premature abandonment without a false task-contract claim;
- tool overuse or repeated failed tool calls;
- reasoning/action imbalance;
- solver timeout or unknown solver status.

**Integrity degradation** is failure in which the agent violates, relaxes, omits, misreports, or bypasses the task contract. Examples include:

- false success: `claim_success` where the oracle rejects the solution;
- false infeasibility: `claim_infeasible` where the oracle holds a valid solution;
- constraint omission or silent relaxation;
- inconsistent state reporting;
- illegal actions represented as legal in subsequent reasoning;
- tool-evidence hallucination: claiming a result that was not observed;
- proxy exploitation: satisfying a measured objective while violating an explicit instruction.

The two failure modes are not mutually exclusive over a run, but they are analytically separable through the oracle classification applied at termination.

#### The boundary rule

Classification attaches to the **terminal claim**, not merely to the stop condition. The rule is:

```text
claim_success + oracle_success           = true success
claim_success + oracle_failure           = integrity error: false success
claim_infeasible + oracle_infeasible     = correct infeasibility report
claim_infeasible + oracle_feasible       = integrity error: false infeasibility
report_unknown + budget_exhausted        = capacity failure
report_unknown + early_stop             = capacity degradation / premature abandonment
```

The governing principle is: **Honest exhaustion or uncertainty is capacity degradation; asserting success or infeasibility incorrectly is integrity degradation.**

This rule prevents the study from manufacturing integrity errors. An agent that has exhausted its budget without finding a solution may legitimately respond:

> I could not find a solution within the budget; status unknown.

That response—`report_unknown`—is an acceptable terminal report and is classified as capacity degradation, not as an integrity failure. The uniform terminal reporting protocol therefore includes `report_unknown` as a first-class option presented without stigma: agents must be allowed honest statements of uncertainty, or the boundary between capacity and integrity failures collapses. If an agent is penalized for admitting uncertainty, the optimal strategy becomes confident assertion, and the study would produce a biased excess of apparent integrity events that are artifactual rather than substantive.

---

### The infeasibility-claim signal-detection layer

A correct `claim_infeasible` on an oracle-infeasible instance is not, by itself, evidence that an agent can reliably detect infeasibility. A degenerate policy—claiming infeasible whenever the agent fails to find a solution—would produce high correct-infeasibility rates on unsatisfiable instances while generating a correspondingly high false-alarm rate on feasible siblings. To separate genuine discrimination from response bias, infeasibility reporting is analyzed jointly across matched feasible and infeasible sibling pairs using a signal-detection framework [@green1966sdt; @macmillan2005detection].

The signal is oracle infeasibility; the response is the terminal report `claim_infeasible`. The four cells of the detection table are:

```text
signal:           oracle_infeasible
response:         terminal_report == claim_infeasible

hit:              claim_infeasible on oracle-infeasible instance
false_alarm:      claim_infeasible on oracle-feasible instance
miss:             report_unknown or claim_success on oracle-infeasible instance
correct_rejection: not claim_infeasible on oracle-feasible instance
```

For every system and near-boundary band, the analysis reports hit rate and false-alarm rate. Where sample size permits, signal-detection summaries are computed:

```text
d_prime = z(hit_rate) - z(false_alarm_rate)
criterion_c = -0.5 * (z(hit_rate) + z(false_alarm_rate))
```

A small-sample correction—adding 0.5 to each cell of the 2×2 table before computing rates—is applied when zero or one rates would otherwise make these quantities undefined.

This decomposition separates two distinct phenomena. **Discrimination** (`d′`) measures whether the system can distinguish feasible from infeasible tasks near the boundary, independent of how readily it invokes `claim_infeasible`. **Response bias** (criterion `c`) measures whether the system defaults to confident infeasibility claims under pressure, regardless of actual task feasibility.

The mirror-image degenerate policy—always reporting `report_unknown`—produces low false-alarm rates but also low hit rates and elevated unknown or premature-abandonment rates. Under the boundary rule, this pattern lands primarily in capacity degradation rather than integrity degradation, because `report_unknown` is not an integrity error by itself. Signal-detection analysis thus provides a consistent accounting: agents that guess infeasible aggressively incur false-alarm integrity costs; agents that never commit incur capacity costs. Neither degenerate extreme is invisible to the framework.

---

### Pressure model

#### Demand/supply frame

For pressure types that admit a resource-like interpretation, we adopt a demand/supply definition:

```text
pressure(resource_type) = required_demand(resource_type) / available_supply(resource_type)
```

A ratio below 1.0 indicates slack; a ratio near 1.0 indicates a tight task; a ratio above 1.0 indicates that completion is impossible unless some dependency is modified. This frame applies cleanly to resource pressure and complexity pressure.

**Resource pressure** is defined as:

```text
resource_pressure = peak_required_resource / available_resource
```

Candidate operationalizations include peak temporary buffer demand divided by available buffers, required memory slots divided by available slots, required tool calls divided by the allowed tool budget, required context tokens divided by the context window, and required time divided by the time budget. In formal planning tasks with a card- or cell-like structure, this reduces cleanly to:

```text
cell_pressure = peak_required_temporary_cells / available_cells
```

**Complexity pressure** is defined as:

```text
complexity_pressure = required_search_budget / allotted_search_budget
```

The required search budget may be approximated by solver-based measures such as shortest-plan length, A\* node expansions, SAT conflicts, CP-SAT branches, or minimum backtracking count, depending on the domain. Raw complexity variables are binned into calibrated bands rather than compared directly across domains; cross-axis comparisons use only the shared conceptual bands (`low`, `medium`, `high`, `near_boundary`).

**Contradiction pressure** is a limiting case of the demand/supply frame. When all constraints must remain enforced and the instance is unsatisfiable, the demand for a valid completion is effectively unbounded:

```text
contradiction_pressure = degenerate case where required valid completion cost = infinity
```

Nearness to satisfiability is operationalized through **removal distance**:

```text
removal_distance = minimum number or weight of constraints that must be removed
                   or relaxed to make the task satisfiable
```

This lets the study distinguish clearly feasible tasks, barely feasible tasks, minimally unsatisfiable tasks, and deeply unsatisfiable tasks. Minimally unsatisfiable cases—those with removal distance of one—are hypothesized to be most destabilizing, because they preserve the appearance of solvability while containing a hard contradiction. An agent searching for a solution in a minimally unsatisfiable task may continue to find near-solutions and may face the highest incentive to relax or omit a constraint rather than honestly report failure.

#### Proxy and coordination pressure as categorical axes

Proxy pressure and coordination pressure are not straightforwardly reducible to a demand/supply ratio and are treated as categorical axes rather than continuous scales.

**Proxy pressure** arises when an unintended path appears cheaper, more rewarded, or more likely to satisfy an evaluator than the intended path. It is not a resource shortfall; it is a task-design condition. The primary treatment in the pilot is categorical:

```text
proxy_pressure = none | weak
```

Forcing proxy pressure into the demand/supply frame risks making the operationalization appear domain-specific or post-hoc. An optional exploratory covariate—the ratio of honest-path cost to shortcut-path cost—may be logged but is not a primary pressure scale.

**Coordination pressure** arises when multiple agents receive partial, stale, inconsistent, or asymmetrically distributed state and must coordinate their actions. It is not reducible to a single resource ratio. The primary treatment is categorical (shared-consistent state, partial private state, stale shared memory, and so on), but coordination pressure is deferred from the pilot. Multi-agent conditions require additional infrastructure beyond what the single-agent pilot covers, and the pilot's purpose is effect-size estimation and instrumentation validation.

---

### Research questions and hypotheses

#### Research questions

**RQ1 (Dose-response).** As calibrated pressure increases, do agent traces show systematic increases in capacity and/or integrity degradation? This is the primary dose-response question, testing whether the framework's dependent variables respond to its independent manipulation at all.

**RQ2 (Capacity/integrity dissociation).** Do LLM agents exhibit integrity degradation under pressure beyond the capacity degradation observed in bounded-search baselines under the same reporting and verification contract? The pilot-scope caveat applies: the capped-solver baseline comparison is formal-domain only, and if the scripted search baseline is not feasible in the tool-use domain, RQ2 for that domain is LLM-only evidence rather than a cross-system comparison.

**RQ3 (Pressure-specific signatures).** Do different pressure axes produce different trace signatures? Resource pressure may be associated with illegal resource use or hallucinated availability; complexity pressure with loops and step explosion; contradiction pressure with false success, false infeasibility, or constraint omission; proxy pressure with shortcutting or evaluator optimization.

**RQ4 (Near-boundary destabilization).** Are barely feasible and minimally unsatisfiable tasks more destabilizing than clearly feasible or clearly impossible tasks? This question targets whether the UIR concentrates near the feasibility boundary.

**RQ5 (Verifier resilience).** Do verifier-augmented agents reduce integrity degradation under pressure, even when capacity costs increase? All architectures share the same terminal reporting contract; the verifier condition changes only mid-trajectory support.

**RQ6 (Notation compression as leading indicator).** Do representation-level trace signals—such as increasing symbolic compression or state-reference density—precede loops, abandonment, or false claims? This question is exploratory unless retrospective traces provide a sufficiently clear signal to justify pre-registration.

**RQ7 (Trace-level early warning).** Can structural trace signals predict eventual failure before final-answer failure is observed? Candidate signals include loop risk scores, repeated state references, rising invalid-action counts, increasing verification failures, step-count acceleration, tool-call bursts, compression slope, and state-inconsistency events.

#### Hypotheses

**H1 (Controlled dose-response).** Capacity degradation may increase monotonically or near-monotonically with calibrated pressure.

**H2 (Integrity threshold).** Integrity degradation may be rare at low pressure but may rise sharply near feasibility boundaries, in minimally unsatisfiable cases, or under proxy pressure. The design implication of H2 is direct: uniform band allocation is underpowered for integrity events if integrity failures concentrate near boundaries. The pilot therefore oversamples near-boundary feasible and minimally unsatisfiable bands; each 60-instance domain uses 18 feasible/unsat sibling pairs and 12 feasible-control pairs rather than a uniform split.

**H3 (Bounded-search contrast).** Given matched task instances, reporting protocols, verification contracts, and compute-budget caps, bounded-search baselines may primarily expose capacity degradation under increasing pressure, whereas LLM agents may additionally exhibit integrity degradation. For baselines sharing the same action interface, the narrower formulation holds: given a shared action interface, reporting protocol, verification contract, and matched compute budget, bounded-search baselines may primarily exhibit capacity degradation, whereas LLM agents may additionally exhibit integrity degradation.

**H4 (Near-boundary maximum).** Barely feasible and minimally unsatisfiable tasks may produce higher overall destabilization than both easy feasible tasks and obviously impossible tasks, because the former maximize search effort and incentive to violate the task contract.

**H5 (Verifier tradeoff).** Verifier-augmented agents may reduce integrity failures relative to baseline agents but may also increase capacity costs such as step counts, tool calls, and latency. This is tested under the uniform terminal protocol; the verifier condition is a mid-trajectory architectural difference, not a change in the finish line.

**H6 (Compression precursor).** If notation compression is a real behavioral signal rather than a style artifact, compression metrics may rise before visible capacity or integrity failure in at least some domains, after conditioning on step index and remaining budget. This hypothesis is exploratory in the pilot.
