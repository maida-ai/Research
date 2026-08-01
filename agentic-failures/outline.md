# Paper outline — CIBD

**Working title:** Constraint-Induced Behavioral Destabilization in Agentic AI Systems: A Controlled Dose-Response Framework for Trace-Level Failure Modes

**Paper type:** Methodology / pre-registered protocol paper. No pilot results exist yet.
The paper presents the framework, the measurement and oracle contract, and the
pre-registered pilot design. Nothing in the paper may assert empirical results.
Hypotheses use "may"; the contrast claims use the §11.4 plan phrasing verbatim.

**Thesis (plan §24):** Agentic LLMs do not merely fail more often as tasks become
harder; under calibrated pressure and a shared reporting contract, they may shift
from bounded capacity failure into measurable integrity failure, and this shift
can be detected through controlled, oracle-checked execution traces.

**Framing rules (binding for all sections):**
- Deflationary measurement science. No claims about stress (subjective), emotion,
  panic, consciousness, AGI, emergence, suffering, meltdown.
- Pressure = externally imposed task-environment property. Destabilization =
  trace-observable behavior.
- Maida named at most once, generically, as "a trace instrumentation framework".
- Citations: ONLY the 14 sources in plan §22 (sources/ notes + references.bib).
  Recent arXiv works cited as preprints unless venue verified.
- Use "may" in methods/hypotheses; never "do" (no results yet).

## Section map → draft files

1. **Introduction** — `draft/01-introduction.md`
   Motivation (end-to-end success obscures how agents fail), the dose-response
   question, CIBD definition, capacity/integrity dissociation as the core
   contribution, contributions list (plan §2.3 novelty items), paper roadmap.
   Includes the abstract (adapted from plan §2.2, protocol-paper tense).

2. **Related work** — `draft/02-related-work.md`
   Agent overthinking/failure taxonomies (Cuadron, Cemri, Chen); planning and
   constrained-agent evaluation (Zhang, Sun); procedure-aware evaluation and
   specification gaming (Cao, Nishimura-Gasparian); signal detection theory
   (Green & Swets, Macmillan & Creelman); positioning: what's new (plan §2.3).

3. **The CIBD framework** — `draft/03-framework.md`
   Definitions: CIBD, behavioral response under pressure, unstable inference
   regime; capacity vs integrity degradation (plan §3.4–3.6 boundary rule);
   pressure model: demand/supply frame, resource/complexity/contradiction axes,
   removal distance, categorical proxy & coordination axes (plan §4); research
   questions RQ1–RQ7 and hypotheses H1–H6 (plan §5–6).

4. **Experimental design** — `draft/04-design.md`
   Domains (formal planning + tool-use), architectures (baseline vs
   verifier-augmented), models, seeds (plan §7); uniform terminal reporting
   protocol incl. pinned wording reference (plan §7.3); task generation and
   matched feasible/infeasible siblings, pair-type budgets, padding, length and
   distractor controls, weak proxy-trap mini-condition (plan §8, §10);
   calibration pipeline order, solver-based and frozen-reference-agent
   calibration, band ladders nested in axes, governance (plan §9);
   bounded-search baselines A and B with explicit claim scope (plan §11).

5. **Metrics and oracle classification** — `draft/05-metrics.md`
   Per-metric dose-response curves as primary results; capacity metric table;
   integrity metric table; metric-definition uniformity rule; trajectory-level
   classification table (oracle spec §6); signal-detection layer for
   infeasibility claims with hit/false-alarm/d′/criterion and small-sample
   correction (plan §12, §14.7; oracle spec §8); sparsity risk and enrichment;
   sub-indices secondary, no headline composite; notation compression as
   exploratory with length controls (plan §13).

6. **Pre-registered pilot and analysis plan** — `draft/06-pilot-analysis.md`
   Pilot scope (2×2×2×3, 30 pairs/domain, 1,440 trajectories), joint pair-type
   and band-target allocation (plan §15); cost probe and adaptive rule;
   statistical models: mixed-effects mains + hypothesis two-way interactions,
   nested-band treatment, rare-event methods, matched-pair tests, multiple
   comparisons (plan §14); pilot success criteria (plan §15.5).

7. **Threats to validity** — `draft/07-threats.md`
   Plan §19 items: anthropomorphism, sparsity, calibration circularity,
   degenerate reference agent, domain overfitting, prompt-length confounds,
   metric multiplicity, trajectory-length confound, solver mismatch,
   integrity-by-construction caveat, asymmetric verification, preprint
   uncertainty.

8. **Discussion and conclusion** — `draft/08-discussion.md`
   What the framework buys (deflationary stress-testing layer); implications
   for verifier-guided/trace-gated reliability and early-warning signals;
   limitations as protocol paper; future work (full study extensions, plan §7.6,
   §17.3).

**Appendices** — `draft/09-appendix.md`
   A: pinned terminal-protocol wording (verbatim, oracle spec §2);
   B: terminal report + termination event schemas (oracle spec §3, §10);
   C: normative trajectory classification table + reference classifier
      pseudocode (oracle spec §6–7);
   D: pair-type budget and calibrated band-target tables (plan §15.2).

**Assembly:** `paper.md` = title/abstract + sections 1–8 + appendices + references.
