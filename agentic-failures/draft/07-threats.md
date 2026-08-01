## Threats to Validity

Any controlled study of behavioral degradation under task-environment pressure
faces validity threats arising from measurement, calibration, design scope, and
the framing of the constructs themselves. The twelve threats enumerated here
apply to the framework as specified; they are not post-hoc observations but
pre-registered acknowledgements that shape both the pilot design and the
interpretation of results.

### Anthropomorphism risk

**Threat.** The terms "pressure" and "destabilization" carry connotations of
subjective experience that the framework does not claim and cannot measure.
**Mitigation.** Throughout this paper, pressure is defined as an externally
imposed task-environment property—a measurable feature of the task instance
such as resource tightness, constraint count, or feasibility status—and
destabilization is defined as trace-observable behavior. The framing
deliberately excludes claims about internal states, affect, or awareness. Any
passage that might invite a subjective reading should be read against these
definitions.

### Integrity-event sparsity

**Threat.** Integrity events — false successes, false infeasibility claims,
procedural violations — may occur too rarely under uniform task allocation for
the pilot to produce reliable rate estimates or to support dose-response
modelling.
**Mitigation.** The pilot uses non-uniform band allocation that concentrates
instances in near-boundary feasible and minimally unsatisfiable bands, where
H2 predicts integrity events are most likely. Event-frequency estimation is
specified as an explicit success criterion (criterion 6), and the adaptive
reallocation rule allows further enrichment if the cost probe reveals
unacceptably low rates. If enrichment is insufficient, the paper reports the
rate estimate as an upper bound rather than fitting dose-response curves on
sparse counts.

### Calibration circularity

**Threat.** If the pressure bands are defined using the same models and
scaffolds that the study subsequently tests, band assignment becomes circular:
a model that performs poorly will appear to encounter harder tasks simply
because its own failures shaped the band ladder.
**Mitigation.** Calibration uses either solver effort (for formal-domain
tasks via CP-SAT [@ortools2024cpsat]) or a frozen reference agent that is
pinned before any test runs and excluded from the test set. Neither
calibration source is the object of study, so the band labels they produce
are independent of the test-phase outcomes.

### Degenerate reference-agent calibration

**Threat.** A reference agent that is too weak may fail all tool-use tasks,
and one that is too strong may succeed at nearly all of them. Either extreme
produces a flat gradient that assigns almost all instances to a single band,
destroying the coverage the calibration procedure is meant to provide.
**Mitigation.** The candidate reference agent must be validated before it is
frozen: a sweep across the generator parameter range must confirm that
reference-agent performance spans roughly 10%–90% failure. If the gradient is
degenerate, pre-specified fallbacks apply — a stronger pinned reference model,
a slightly stronger non-test scaffold, a partial-credit calibration metric, or
success-within-N-retries — before any test runs begin.

### Domain overfitting

**Threat.** A behavioral signal found in formal combinatorial tasks (such as
card-scheduling or Blocks World variants) may not generalize to tool-use
tasks with open-ended action spaces, different failure modes, and different
calibration pipelines.
**Mitigation.** The pilot includes one formal domain and one tool-use domain.
Cross-domain comparisons are reported separately, and any claim about
generality is scoped to the two domains tested. Claims about broader
generalization are deferred to a full study with additional domains.

### Prompt-length confounds

**Threat.** Near-boundary and unsatisfiable instances may be systematically
longer than easier feasible instances — both because specifying tight
constraints requires more text and because padding operations add further
length. Any behavioral difference between band levels could then be explained
by prompt length rather than pressure.
**Mitigation.** The pair-type design includes length-only control pairs in
which a feasible instance is paired with a length-inflated feasible sibling
that has identical structure but additional neutral text. Band-label
assignment occurs after padding so that calibrated difficulty reflects the
padded prompt, not a pre-padding estimate. Matched-pair analyses condition on
prompt length within pair type.

### Metric multiplicity

**Threat.** Tracking many trace-level metrics across multiple conditions and
band levels creates substantial multiple-comparisons exposure. With enough
metrics, spurious signals are inevitable.
**Mitigation.** The analysis plan (Section 6) distinguishes primary,
secondary, and exploratory metrics before any data are collected. Primary
metrics are limited and tied directly to the capacity and integrity
hypotheses; only these carry evidentiary weight. Exploratory metrics —
including notation-compression indices — are reported descriptively and
cannot confirm or disconfirm hypotheses. This tiering is part of the
pre-registration and cannot be revised post-hoc.

### Trajectory-length confound for compression

**Threat.** Notation-compression metrics (symbol ratio, entity density,
characters per state reference) may correlate with failure not because
compression predicts failure but because both increase with trajectory length:
longer trajectories contain more state references and more opportunities to
observe any trace event.
**Mitigation.** Compression analyses condition on step index and remaining
budget fraction. Lagged analyses use windows anchored to elapsed-budget
fraction rather than absolute step count, so that early and late trajectory
windows are comparable across systems with different step totals.
Compression metrics are classified as exploratory for precisely this reason.

### Solver interface mismatch

**Threat.** Classical solvers and LLM agents operate over structurally
different interfaces: a CP-SAT solver receives a constraint-satisfaction
problem in a formal language, while an LLM agent receives a natural-language
task description and acts through a tool interface. Comparing their outputs
as if they were solving the same problem risks comparing incommensurable
quantities.
**Mitigation.** Two bounded baselines are used: a principled capped solver
that receives the same formal representation as the oracle, and a
same-interface scripted search baseline that operates through the same action
interface as the LLM agent. Claims about the capacity/integrity dissociation
are scoped to the action interface over which all compared systems operate,
and the domain is stated explicitly in any cross-system comparison
[@ortools2024cpsat].

### Integrity-by-construction caveat

**Threat.** Bounded-search baselines may appear to produce no integrity
events not because they are genuinely integrity-preserving but because the
verifier or reporting protocol makes certain integrity errors structurally
impossible for those systems.
**Mitigation.** The paper does not claim that bounded baselines literally
cannot make integrity errors; it states that verifier-enforced bounded
baselines primarily expose capacity limits, and it qualifies the
architectural comparison accordingly. If the bounded baseline's reporting
protocol or verifier makes a particular error class impossible, that
constraint is stated explicitly so readers can assess what the comparison
does and does not show.

### Asymmetric verification risk

**Threat.** If procedural-violation or skipped-verification metrics are
defined in a way that is only detectable for verifier-augmented agents (e.g.,
a metric that requires verifier log entries that baseline agents never
produce), then apparent architecture differences may be artifacts of
asymmetric instrumentation.
**Mitigation.** The terminal reporting protocol is uniform across all system
types: every agent—baseline, verifier-augmented, and bounded baseline—
uses the same terminal labels and is subject to the same oracle-judging
contract. Metrics that depend on verifier-specific log events are marked as
not applicable for non-verifier systems and excluded from cross-architecture
comparisons.

### Preprint uncertainty

**Threat.** Several related works cited in this paper are recent arXiv
preprints whose findings have not yet been independently replicated and whose
peer-review status at time of writing is uncertain
[@cuadron2025overthinking; @cemri2025multiagent;
@zhang2026deepplanning; @sun2026apb; @cao2026procedure;
@nishimura2026specgaming]. Among the recent works cited, only Chen et
al. has verified venue acceptance [@chen2026elephant].
**Mitigation.** All arXiv works are cited explicitly as preprints unless
venue acceptance has been verified. Claims that depend on a single preprint
finding are presented as provisional. The paper's core methodological
contributions — the CIBD framework, the capacity/integrity dissociation, and
the pre-registered pilot design — do not depend on any single external
empirical result and are not invalidated if preprint findings fail to
replicate.
