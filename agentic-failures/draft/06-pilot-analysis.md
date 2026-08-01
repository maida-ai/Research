## Pre-Registered Pilot Design and Analysis Plan

This section specifies the pilot study in sufficient detail to support
pre-registration. The design is not yet executed; all statements about what
the pilot "will" measure or "is designed to" detect are forward-looking
descriptions of protocol, not reports of results. The pilot serves two
separable purposes: validating that the instrumentation and task-generation
pipeline function end-to-end, and producing preliminary effect-size and
event-rate estimates that determine whether a full study is warranted and
how large it must be.

### Pilot scope

The minimal viable pilot crosses the following factors:

```text
domains:        2  (one formal, one tool-use)
architectures:  2  (baseline agent, verifier-augmented agent)
models:         2  (one stronger API model, one cheaper/open or mid-tier model)
seeds:          3
matched pairs:  30 formal pairs + 30 tool-use pairs
instances:      120 total instances if each pair has two siblings
trajectories:   120 * 2 models * 2 architectures * 3 seeds = 1,440
```

At 1,440 trajectories the pilot is large enough to validate instrumentation
and obtain first-order effect-size estimates, yet small enough to control
token cost before committing to full-study scale.

An important constraint governs how the 30-pair-per-domain budget is
interpreted. The 30 pairs per domain are not equivalent to independently
assigning 60 band labels from a flat distribution. Pair types are
determined first by the task-generation procedure; calibrated pressure-band
labels are assigned only after padding and matching are complete. The band
label for the feasible sibling in any pair may shift as a result of
padding operations, so feasible-band counts are partly emergent outcomes of
the calibration pipeline rather than free design parameters. The
statistical models and pre-registration must reflect this dependency.

### Joint pair-type and band-target allocation

Uniform allocation across pressure bands would produce too few integrity
events per cell to support the core hypotheses. The pilot therefore uses a
non-uniform allocation that concentrates instances where hypothesis H2
predicts integrity failures: near-boundary feasible bands and minimally
unsatisfiable instances. The allocation is specified across two interdependent
tables — one controlling pair-type counts directly, one specifying calibrated
band-target shares as post-calibration objectives.

#### Pair-type budget per 60-instance domain

The following table governs the directly controlled pair-type composition
within each 60-instance domain:

| Pair type | Pairs | Instances | Directly controlled variable |
|---|---:|---:|---|
| Feasible ↔ minimally unsat | 12 | 24 | 12 minimally unsat instances |
| Feasible ↔ obviously unsat | 6 | 12 | 6 obviously unsat instances |
| Feasible ↔ length-inflated feasible | 4 | 8 | 4 length-only control pairs |
| Feasible ↔ distractor-padded feasible | 4 | 8 | 4 distractor-control pairs |
| Feasible ↔ weak-proxy/control feasible | 4 | 8 | 4 proxy/control pairs if proxy condition is retained |
| **Total** | **30** | **60** | **18 unsat + 42 feasible instances** |

If a domain does not support weak proxy traps, the four proxy pairs are
reassigned to length-only or distractor controls and proxy-exploitation
metrics are marked not applicable for that domain.

#### Calibrated band-target allocation per 60-instance domain

The following table specifies target band shares after post-padding
calibration. These are objectives for the calibration pipeline, not
independently assigned labels:

| Band | Target share | Target instances per 60-instance domain | Control status |
|---|---:|---:|---|
| easy_feasible | 10% | 6 | Emergent calibrated target over the 42 feasible instances |
| medium_feasible | 15% | 9 | Emergent calibrated target over the 42 feasible instances |
| hard_feasible | 20% | 12 | Emergent calibrated target over the 42 feasible instances |
| near_boundary_feasible | 25% | 15 | Emergent calibrated target over the 42 feasible instances |
| minimally_unsat | 20% | 12 | Direct unsat generation target |
| obviously_unsat | 10% | 6 | Direct unsat generation target |

The implementation iterates generator parameters until the calibrated
feasible instances approximately hit the 6/9/12/15 target split. The
acceptable tolerance must be pre-specified before pilot execution — for
example, ±1 instance per feasible band in a 60-instance domain. This
iterate-until-targets-hit rule preserves coverage while concentrating
statistical power where integrity effects are hypothesized to emerge.

Because near-boundary feasible and minimally unsatisfiable instances
are the primary sites of predicted integrity events, a pilot that fails
to populate these bands adequately cannot test the central claims. The
following adaptive rule governs reallocation if the cost probe reveals
insufficiently low event rates:

> If the cost probe shows integrity events below roughly 3–5% in the
> enriched bands, shift further mass from easy/obvious bands into
> near-boundary/minimally-unsat bands or add a weak proxy-trap subset.
> Preserve the pair-type arithmetic when reallocating.

This rule must be pre-registered with its trigger threshold so that
post-hoc reallocation cannot be mistaken for result-driven steering.

### Cost and event-rate probe

Before the full pilot launches, a cost-probe run covers 5–10% of the total
instance budget (approximately 6–12 task instances, two models, two
architectures, three seeds where affordable). The probe's role is diagnostic,
not confirmatory: it must produce estimates of all major cost and
event-rate parameters before the team commits to the full 1,440-trajectory
budget.

The probe estimates the following quantities at minimum:

- mean and p95 tokens per trajectory;
- mean tool calls per trajectory;
- retry and hard-failure rate;
- verifier overhead (additional latency and token cost introduced by
  the verifier-augmented architecture);
- storage volume per trajectory;
- annotation time per trajectory;
- integrity-event frequency in enriched bands;
- terminal-claim distribution across `claim_success`, `claim_infeasible`,
  and `report_unknown`;
- sensitivity of the terminal-claim distribution to one pre-registered
  terminal-protocol paraphrase, if the paraphrase check is included;
- cost per condition.

If the paraphrase check reveals large swings in terminal-claim
distributions, the team must report protocol-wording sensitivity as a
limitation and keep the original pinned wording in place for the full pilot.

Budget planning defaults are:

```text
target pilot budget:        $3,000
hard stop including reruns: $5,000
```

These figures are recommended planning ceilings, not scientific requirements.
The final task count and band allocation should be rescaled after the probe
rather than fixed in advance. The team must not commit to trajectories
in the tens of thousands before cost-per-trajectory is measured; the probe
exists precisely to prevent such commitments from being made on untested
assumptions.

### Statistical analysis plan

#### Avoiding saturated models

With 1,440 trajectories distributed non-uniformly across bands and crossed
with two architectures, two models, two domains, and three seeds, the pilot
cannot support fully saturated four-way interaction terms such as:

```text
pressure_type * pressure_band * architecture * model_id
```

This is particularly acute after non-uniform allocation, where some bands
are intentionally sparse. The pre-registered analysis specifies main effects
and hypothesis-relevant two-way interactions only. Three-way and four-way
interactions are treated as exploratory and must be clearly labeled as such
in the analysis document.

#### Primary models for capacity outcomes

For continuous capacity metrics (e.g., step count, tool-call count, loop
rate):

```text
metric ~ band_level + architecture + model_id + domain + pressure_axis +
         band_level:architecture +
         (1 | task_id)
```

For count outcomes modelled with negative-binomial error:

```text
negative_binomial(metric_count) ~ band_level + architecture + model_id + domain + pressure_axis +
                                  band_level:architecture +
                                  (1 | task_id)
```

The key interaction for hypothesis H5 is `band_level × architecture`. This
term tests whether verifier augmentation changes the slope of capacity
degradation across pressure levels — that is, whether the architecture
difference grows, shrinks, or stays constant as pressure increases.

#### Primary models for integrity outcomes

For binary integrity outcomes (e.g., false-success rate, false-infeasibility
rate):

```text
logit(P(integrity_error)) ~ band_level + architecture + model_id + domain + pressure_axis +
                            band_level:architecture +
                            (1 | task_id)
```

If bounded baselines are included in the same model alongside LLM agents:

```text
logit(P(integrity_error)) ~ band_level + system_type + domain + pressure_axis +
                            band_level:system_type +
                            (1 | task_id)
```

where `system_type = LLM_agent | bounded_search`. The key interaction for
hypothesis H3 is `band_level × system_type`. This tests whether the
capacity/integrity dissociation predicted by H3 — LLM agents showing rising
integrity errors while bounded-search baselines do not — is consistent with
the data.

For rare integrity events, the pre-registration specifies one or more of the
following methods, selected depending on observed cell counts: exact tests
for matched pairs where appropriate; Firth-corrected logistic regression;
Bayesian logistic regression with weakly informative priors; or
pre-registered pooling across adjacent near-boundary bands. The choice among
these methods must be specified before the pilot executes.

#### Nested-band treatment

Pressure bands are nested within pressure axes: the band ladder for the
feasibility-contradiction axis differs from that of the resource-complexity
axis. Fitting a fully crossed `pressure_axis × pressure_band` interaction
over structurally empty cells is not appropriate. The analysis uses one of
two approaches:

1. **Axis-specific models.** Separate models fit within each pressure axis
   using that axis's own band ladder.
2. **Common-band models.** Cross-axis comparisons are restricted to bands
   shared across axes (e.g., `low / medium / high / near_boundary`).

Three-way interactions such as `pressure_axis × band_level × architecture`
are exploratory in the pilot and must be pre-labelled accordingly.

#### Fixed versus random effects

`model_id` and `architecture` are treated as fixed effects in the pilot. With
only two levels per factor, random-effects estimation for these variables is
not well-justified and would produce unstable variance components. The
random-effects structure is limited to `(1 | task_id)` to account for
repeated matched variants and seeds per task.

#### Matched-pair tests

The pair-type design supports a set of pre-registered matched contrasts. For
each sibling pair, the analysis computes a within-pair delta:

```text
paired_delta_metric ~ condition + architecture + model_id
```

Matched deltas are reported for the following contrasts:

- feasible versus minimally unsatisfiable siblings;
- normal versus prompt-length-inflated siblings;
- no-distractor versus distractor-padded siblings;
- no-proxy versus weak-proxy-trap siblings.

These contrasts isolate the effect of specific task-environment properties
(feasibility, prompt length, distractor presence, proxy trap) while
controlling for all other task-level variation shared within the pair.

#### Signal-detection analysis

The signal-detection analysis for infeasibility claims is defined in the
Metrics section and applied per system × domain × band cell. The analysis
treats `claim_infeasible` as the response variable and `oracle_infeasible`
as the signal. At minimum, the pilot reports hit rate, false-alarm rate, and
unknown rate; where cell counts permit, `d′` and criterion `c` are also
reported, using a pre-registered loglinear correction (add 0.5 to each cell
of the 2×2 table) to handle zero or near-zero rates. The four diagnostic
patterns — good discrimination, response bias, conservative capacity
limitation, and boundary-region integrity risk — are assessed over matched
sibling sets, not isolated unsatisfiable instances
[@green1966sdt; @macmillan2005detection].

#### Multiple comparisons

Because the study tracks multiple capacity and integrity metrics across
several conditions, false-discovery risk is non-trivial. The pre-registration
distinguishes three tiers:

- **Primary metrics**, limited in number and tied directly to capacity and
  integrity hypotheses; these carry the study's evidentiary weight.
- **Secondary metrics**, pre-specified but not the principal basis for
  hypothesis evaluation.
- **Exploratory metrics** (including notation-compression indices), reported
  descriptively with no adjustment claims.

Primary metrics are subject to a pre-specified multiple-comparisons
correction. Exploratory metrics are not used to confirm or disconfirm
hypotheses.

### Pilot success criteria

The pilot is successful if it satisfies all nine of the following criteria:

1. Reliable trace capture: every trajectory produces a complete, parseable
   trace with no missing terminal events.
2. Calibration artifacts: solver-based and frozen-reference-agent
   calibration runs complete and produce pressure-band label assignments for
   the full instance set.
3. Reference-agent gradient validation: the candidate reference agent for
   tool-use calibration produces a nondegenerate success-rate gradient across
   bands before being frozen.
4. Matched-pair generation: the end-to-end pair-generation, padding, and
   matching pipeline completes without structural failures and satisfies the
   pair-type budget table.
5. Capacity dose-response: at least one pressure axis produces a measurable
   monotonic capacity dose-response across band levels.
6. Integrity-event frequency estimation: the enriched bands yield sufficient
   integrity events to estimate whether full-study scaling is worthwhile —
   even if the absolute count is low, the rate estimate must be informative
   enough to support a power calculation.
7. Architecture comparison: a preliminary comparison between baseline and
   verifier-augmented agents under a shared reporting protocol is
   interpretable.
8. Bounded-baseline comparison: a bounded-baseline comparison is complete in
   the formal domain; a tool-use scripted baseline comparison is included if
   feasible.
9. Cost and variance estimates: the pilot produces the token cost, latency,
   and within-cell variance estimates needed for a full-study power analysis.

Integrity-event adequacy is a design criterion, not an assumed result. The
non-uniform allocation and adaptive reallocation rule are designed to
maximize the chance of observing a sufficient number of integrity events, but
the pilot cannot guarantee it. If the enriched pilot still yields too few
integrity events to support dose-response modeling, the appropriate response
is transparent reporting and redesign of the task-generation procedure—
not extrapolation of dose-response curves beyond what the data support. A
pilot that returns an informative null on integrity-event rate is still a
successful pilot by criterion 6, provided it yields a credible upper-bound
estimate.
