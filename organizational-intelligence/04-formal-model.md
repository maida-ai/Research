# Light Formal Model: Main Text and Appendix Scaffold

Status of this section: this is a conceptual formalization for theory-building. The definitions below are proposed constructs, not established formulas in the literature. For the first paper, the main text should use only the compact model in Sections 1-2. Sections 3-8 are appendix material or working scaffolding for later formal development.

## 1. Minimal Model for the Main Text

### 1.1 Decision Episode

A high-stakes organizational decision episode is a bounded sequence of communications, commitments, reviews, escalations, and actions through which an organization converts conflicting pressures into an ordered course of action. The unit of analysis is the episode, not the whole organization.

Let:

$$
x_t \in X
$$

represent the episode state at time $t$. $X$ may include product maturity, operational capacity, codebase quality, revenue pressure, customer trust, legal exposure, customer expectations, employee morale, technical debt, regulatory posture, market position, organizational memory, and commitments already created.

This state representation is deliberately broad but episode-bounded. It represents what matters for the focal decision: what is known, what has been promised, what resources are available, what risks are visible, what constraints have appeared, and what commitments are already difficult to undo.

In empirical settings, transitions are not caused solely by internal action. They also include external responses and shocks: customer reactions, market movement, regulatory changes, media attention, competitor moves, and capital-market pressure. If compact notation is useful, write:

$$
x_{t+1} = F_a(x_t,\epsilon_t),
$$

where $a$ is an organizational action and $\epsilon_t$ summarizes external events and responses during the episode.

### 1.2 Ordered Commitments and Sequence Sensitivity

Let:

$$
a ; b
$$

mean "do $a$, then do $b$." The key diagnostic point is that organizational actions are often sequence-sensitive. Formally, this is analogous to non-commutativity:

$$
F_{a;b}(x) \not\equiv_D F_{b;a}(x),
$$

where $\not\equiv_D$ means "not materially equivalent for the decision episode." Exact equality is too strong for most organizational settings because episode states are noisy, partially observed, and interpreted through organizational records. The relevant question is whether the two sequences produce materially different technical, financial, legal, reputational, social, or strategic consequences.

Examples:

| Sequence pair | Diagnostic contrast |
|---|---|
| `ship ; review` vs. `review ; ship` | Post-release review can create customer exposure and remediation pressure before constraint information is incorporated. |
| `sell_enterprise ; build_compliance` vs. `build_compliance ; sell_enterprise` | Commercial commitments before compliance work can create promise-risk and trust exposure. |
| `hire_rapidly ; define_culture` vs. `define_culture ; hire_rapidly` | Rapid hiring before norm formation can lock in practices that are difficult to reshape. |

We define sequence sensitivity, written $\kappa(s,t)$, as the degree to which performing sequence $s$ before sequence $t$ produces a materially different episode state from performing $t$ before $s$. $\kappa$ is a diagnostic variable, not a welfare score. It says that order matters; it does not say which order is better.

### 1.3 Practical Irreversibility

We define practical irreversibility, written $\rho(s)$, as the difficulty or cost of returning to a materially similar prior state after action sequence $s$.

An action may be repairable without being reversible. A public launch can be patched, but customers have already experienced it. An enterprise promise can be renegotiated, but trust and contractual expectations may persist. A reorganization can be revised, but social networks, morale, and authority patterns may not return to the prior state.

In this paper, $\rho$ is a diagnostic variable and potential tuning knob. It should not be presented as a validated quantitative metric. Later work can make $\rho$ a quantitative centerpiece by specifying state variables, recovery actions, loss functions, and measurement rules.

### 1.4 Pressure Families

The model defines three pressure families:

$$
I = \text{appetite}, \quad
E = \text{feasibility}, \quad
S = \text{constraint}.
$$

These are not literal mental agencies, departments, or fixed actor groups. They are analytic families of pressure, claims, evaluations, actors, and actions. In simplified notation, they may be linked to actions, but the core construct is pressure-linked interpretation.

| Layer | Example |
|---|---|
| Pressure | "We need revenue this quarter." |
| Actor | Sales, investor, founder, customer, engineer, lawyer, security reviewer, employee group |
| Claim | "The enterprise deal requires SOC 2." |
| Evaluation | "This trajectory is good for growth but bad for reliability." |
| Action | Ship, delay, review, disclose, refactor, renegotiate |

The same action can be proposed for different reasons. `harden_security` can be feasibility-driven when the system is brittle, constraint-driven when compliance or trust is at stake, or appetite-driven when enterprise sales require it.

Appetite names the pressure toward growth, revenue, opportunity capture, resource acquisition, speed, expansion, and competitive advantage. Feasibility names contact with implementational reality: technical, operational, socio-technical, infrastructural, temporal, and coordination constraints. Constraint names the plural set of legitimacy-preserving pressures: law, compliance, ethics, reputation, stakeholder claims, customer trust, employee legitimacy, and social permission.

### 1.5 Evaluators Without Scalar Overclaim

For notational convenience, we may write $U_I$, $U_E$, and $U_S$ as trajectory evaluators. They need not be scalar utilities in empirical applications. They may be qualitative assessments, partial orders, constraints, thresholds, or multi-dimensional evaluations.

The point is partial conflict, not permanent zero-sum conflict. A trajectory may be attractive to appetite but risky to feasibility or constraint. Another trajectory may improve all three. The model should not assume that any one evaluator alone defines organizational intelligence.

### 1.6 Communication and Power

Each pressure family is carried by actors with partial information, incentives, authority, and exposure. We define the following diagnostic parameters:

| Parameter | Diagnostic meaning |
|---|---|
| $B_j$ | Bandwidth: how much information pressure family $j$ can communicate during the episode. |
| $\ell_j$ | Latency: how late that information arrives relative to commitment. |
| $c_j^{\mathrm{truth}}$ | Truthful-disclosure cost: political, career, reputational, or operational cost of saying what the actor knows. |
| $p_j$ | Power: authority, incentive weight, resource control, dependency position, and escalation capacity. |
| $v_j$ | Verifiability: how easily others can test the claim before action. |

These parameters are proposed model definitions. They are motivated by communication and team-theory sources, but the specific organizational parameters are part of this paper's framework.

### 1.7 Mediation Protocol

The central object of the model is the mediation protocol:

$$
\Phi =
\left(
\phi^{\mathrm{surf}},
\phi^{\mathrm{comm}},
\phi^{\mathrm{weight}},
\phi^{\mathrm{seq}},
\phi^{\mathrm{resolve}}
\right).
$$

A mediation protocol is the recurring communicative and procedural structure through which appetite, feasibility, and constraint are surfaced, communicated, weighted, sequenced, translated, and resolved into commitments.

| Protocol slot | Function in the episode |
|---|---|
| $\phi^{\mathrm{surf}}$ | Surfaces pressure claims from local observations. |
| $\phi^{\mathrm{comm}}$ | Translates claims into messages under bandwidth, latency, truthful-disclosure cost, verifiability, and power conditions. |
| $\phi^{\mathrm{weight}}$ | Assigns provisional decision weight to the surfaced pressures. |
| $\phi^{\mathrm{seq}}$ | Identifies candidate action orderings and flags high-$\rho$ or high-$\kappa$ decisions. |
| $\phi^{\mathrm{resolve}}$ | Selects, authorizes, revises, or escalates an action sequence. |

We define organizational intelligence provisionally as the quality of $\Phi$: its capacity to generate action sequences that remain viable across growth, implementability, and legitimacy pressures over time. This is a theory-building definition, not an empirical finding.

## 2. Main Propositions

The following propositions are testable conjectures for future empirical work, not findings.

### Proposition 1: Sequence Sensitivity

The greater the sequence sensitivity of recurring organizational action pairs, the more organizational performance depends on mediation protocols that identify and manage action order.

**This suggests:** when the same actions produce materially different results depending on order, intelligence depends less on having a single correct department and more on having a protocol that detects order-sensitive commitments.

### Proposition 2: Practical Irreversibility

The greater the practical irreversibility of a decision episode, the more valuable early mediation among appetite, feasibility, and constraint becomes.

**This suggests:** late review may be adequate for low-$\rho$ actions but dangerous when commitments are hard to undo.

### Proposition 3: Communication and Power

When communication bandwidth is low, information latency is high, truthful disclosure is costly, or power asymmetry is large, decision episodes are more likely to reflect the dominant pressure family rather than mediated intelligence.

**This suggests:** communication design and power distribution shape which pressure claims enter the action sequence before commitment.

### Proposition 4: Appetite-First Accumulation

Repeated appetite-first sequencing increases future recovery burden when feasibility and constraint are systematically delayed or incorporated only after commitment.

**This suggests:** growth pressure can remain adaptive when mediated, but repeated `sell ; build`, `announce ; verify`, or `ship ; review` patterns can accumulate technical, legal, reputational, and morale debt.

### Proposition 5: Protocol Learning

Organizations reduce recurrent sequence failures when post-episode learning updates mediation protocols rather than merely producing retrospective blame or local repair.

**This suggests:** organizational memory matters when it changes future surfacing, communication, weighting, sequencing, or resolution rules.

Plural constraint and protocol libraries are important implications rather than main propositions in the first paper. Constraint should be decomposed where legal, compliance, ethical, reputational, customer-trust, and employee-legitimacy pressures diverge. Recurrent high-$\kappa$ action pairs may also justify reusable protocol libraries, but that claim can be developed in future research.

## 3. Diagnostic Constructs

| Construct | Diagnostic question |
|---|---|
| $\rho(s)$ | How difficult or costly is this action sequence to undo or repair? |
| $\kappa(s,t)$ | How much does the order of these two action sequences matter? |
| $U_I, U_E, U_S$ | Which pressure-family evaluation is favored, suppressed, or left unresolved? |
| $B_j$ | Does the relevant actor have enough bandwidth to explain what it knows? |
| $\ell_j$ | Does relevant information arrive before or after commitment? |
| $c_j^{\mathrm{truth}}$ | What does it cost an actor to disclose inconvenient information? |
| $p_j$ | Which actors can force, delay, suppress, or reframe action? |
| $v_j$ | Which claims can be verified before action? |
| $\Phi$ | How are pressure claims surfaced, communicated, weighted, sequenced, and resolved? |

These constructs can be used qualitatively before they are quantified. A case study could code them ordinally. A later quantitative study could operationalize them through decision records, issue trackers, launch postmortems, compliance artifacts, customer escalations, incident reports, employee surveys, or process-mining data.

## 4. Assumptions

1. **Episode boundary.** Analysts can identify a bounded high-stakes decision episode.
2. **State relevance.** The selected state variables capture enough of the episode condition to support meaningful comparison.
3. **Material equivalence.** Analysts can define when two resulting states are materially similar for the decision episode.
4. **Partial observability.** No pressure-carrying actor observes the full episode state.
5. **Partial goal conflict.** Appetite, feasibility, and constraint evaluations are partly but not wholly conflicting.
6. **Unequal power.** The relative power distribution $p$ among actors carrying the pressure families can be provisionally described.
7. **Recoverability judgment.** Analysts can describe the practical repair or recovery options available after commitment.
8. **Protocol stability.** The mediation protocol is stable enough to be identified, described, and compared across episodes.

## 5. Limitations

1. **Conceptual status.** The model is a proposed formalization, not a theorem or validated empirical instrument.
2. **Measurement burden.** $X$, $\rho$, $\kappa$, $p$, and $\Phi$ require domain-specific operationalization.
3. **Causal identification.** It may be hard to distinguish protocol effects from leadership quality, market conditions, regulation, firm age, or product complexity.
4. **State ambiguity.** Episode states are high-dimensional, partially latent, and interpreted through incomplete records.
5. **Plurality within families.** Appetite, feasibility, and constraint each contain internal disagreements. Constraint is especially plural.
6. **Normative risk.** Quantifying legitimacy, ethics, trust, morale, or social permission can create false precision or encourage gaming.
7. **Power and politics.** The model includes $p$ but does not yet fully theorize domination, coercion, ideology, or resource dependence.
8. **External environment.** Competitors, regulators, customers, capital markets, and media can change the episode state independently of internal action.
9. **Not a replacement framework.** The model should complement systems psychodynamics, paradox theory, organizational learning, communication theory, legitimacy theory, and cybernetics rather than claim to supersede them.

## 6. Appendix Scaffold: Optional Formal Details

The following details should not all appear in the main text of the first paper. They can support an appendix, reviewer response, or later formal paper.

### 6.1 Expanded Episode State

For a product/strategy decision episode, one expanded state representation is:

$$
x_t = (P_t, q_t, r_t, T_t, L_t, C_t, \mathrm{Mor}_t, D_t, R_t, Z_t, H_t, W_t).
$$

| Component | Possible interpretation |
|---|---|
| $P_t$ | Product maturity and feature readiness |
| $q_t$ | Codebase quality, architecture, maintainability, and security posture |
| $r_t$ | Revenue, pipeline, pricing power, and customer concentration |
| $T_t$ | Customer trust, public trust, enterprise trust, and credibility |
| $L_t$ | Legal exposure, contractual exposure, and regulatory risk |
| $C_t$ | Customer expectations and commitments already created |
| $\mathrm{Mor}_t$ | Employee morale, retention risk, and internal legitimacy |
| $D_t$ | Technical debt and operational debt |
| $R_t$ | Regulatory posture, compliance readiness, and auditability |
| $Z_t$ | Market position, competitive timing, and strategic optionality |
| $H_t$ | Organizational memory, routines, precedents, and decision records |
| $W_t$ | Work capacity, staffing, infrastructure, and execution bandwidth |

### 6.2 Action Sequences

Let $A$ be a set of episode action symbols and let $\Sigma$ be the set of finite action sequences over $A$. For a sequence $\sigma = a_1 ; a_2 ; \cdots ; a_n$, a compact deterministic representation is:

$$
F_\sigma(x)
=
F_{a_n}(\cdots F_{a_2}(F_{a_1}(x))\cdots).
$$

This notation is only a convenience. In empirical settings, $F_a$ may be stochastic, partially observed, relational, or estimated from case evidence.

### 6.3 Sequence Sensitivity

Let $\equiv_D$ denote material equivalence for decision episode class $D$. Two sequences $s,t \in \Sigma$ are diagnostically sequence-sensitive at state $x$ when:

$$
F_{s;t}(x) \not\equiv_D F_{t;s}(x).
$$

If a distance or loss function $d$ and reference distribution $\mu$ are available, a possible sequence-sensitivity score is:

$$
\kappa_{\mu,d}(s,t)
=
\mathbb{E}_{x \sim \mu}\left[
d\!\left(F_{s ; t}(x), F_{t ; s}(x)\right)
\right].
$$

This is a proposed construct. It is not a known standard measure.

### 6.4 Practical Irreversibility

Let $\mathcal{R} \subseteq \Sigma$ be a set of practically available recovery, repair, mitigation, or reversal sequences. Let $c(r,y)\ge 0$ be the practical cost of applying recovery sequence $r$ after the episode has reached state $y$, and let $\lambda \ge 0$ be a modeling weight. A possible detailed irreversibility measure is:

$$
\rho_{\mathcal{R},\mu,d,c,\lambda}(s)
=
\inf_{r \in \mathcal{R}}
\mathbb{E}_{x \sim \mu}\left[
d\!\left(F_{s ; r}(x), x\right)
+ \lambda c(r,F_s(x))
\right].
$$

This is also a proposed construct. In the first paper, write $\rho(s)$ and explain it in prose unless the appendix needs the full expression.

### 6.5 Pressure-Family Evaluators

Let $\mathcal{T}$ be the set of modeled trajectories. For notation, we may write:

$$
U_I,\ U_E,\ U_S
$$

as appetite, feasibility, and constraint evaluators over trajectories. They need not be real-valued scalar utilities. They may be qualitative assessments, partial orders, constraints, thresholds, or multi-dimensional evaluations.

If a trajectory evaluator $W$ has been chosen, a directional order comparison can be written:

$$
\Delta_W(s,t)
=
\mathbb{E}_{x_0 \sim \mu}\left[
W(\tau(x_0, s ; t)) - W(\tau(x_0, t ; s))
\right].
$$

This comparison is distinct from $\kappa(s,t)$. $\kappa$ measures sequence sensitivity; $\Delta_W$ evaluates which order is preferred under a chosen evaluator.

### 6.6 Detailed Protocol Mapping

Let $\Pi_t$ denote surfaced pressure claims, $\mathcal{M}_t$ messages available at time $t$, $w_t$ provisional weights assigned to pressure families, $\Omega_t$ candidate ordered action sequences, $\mathcal{H}_t$ organizational memory, and $\Theta$ communication and power parameters. A detailed representation of the mediation protocol is:

$$
\begin{aligned}
\Pi_t &= \phi^{\mathrm{surf}}(x_t,y_I,y_E,y_S,\Theta),\\
\mathcal{M}_t &= \phi^{\mathrm{comm}}(\Pi_t,\Theta),\\
w_t &= \phi^{\mathrm{weight}}(\mathcal{M}_t,\mathcal{H}_t,\Theta),\\
\Omega_t &= \phi^{\mathrm{seq}}(x_t,\mathcal{M}_t,w_t,\mathcal{H}_t,\Theta),\\
\sigma_t &= \phi^{\mathrm{resolve}}(\Omega_t,x_t,\mathcal{M}_t,w_t,\mathcal{H}_t,\Theta).
\end{aligned}
$$

Equivalently:

$$
\Phi: X \times Y_I \times Y_E \times Y_S \times \mathcal{H}_t \times \Theta \to \Sigma.
$$

This should remain appendix material unless the paper needs a more explicit bridge between the diagnostic vocabulary and later formal work.

### 6.7 What Later Mathematical Rigor Would Require

Later formal work would need to specify:

1. episode boundaries and inclusion rules,
2. the structure of $X$,
3. whether $F_a$ is deterministic, stochastic, relational, partial, or empirical,
4. the material equivalence relation $\equiv_D$ or loss function $d$,
5. the information conditions for $y_I$, $y_E$, and $y_S$,
6. the communication and power model behind $B$, $\ell$, $c^{\mathrm{truth}}$, $p$, and $v$,
7. the recovery set $\mathcal{R}$ and recovery cost $c$,
8. the properties and measurement rules for $\rho$ and $\kappa$,
9. the status of $U_I$, $U_E$, and $U_S$ as scalar evaluators, partial orders, thresholds, constraints, or vectors,
10. how organizational memory updates $\Phi$ after outcomes,
11. empirical proxies for episode states, pressure claims, messages, weights, action sequences, and outcomes.

The paper should not introduce heavier formal frameworks unless these objects are specified.

## 7. Theory Ledger

| Construct | Revised canonical definition |
|---|---|
| Decision episode | A bounded sequence of communications, commitments, reviews, escalations, and actions through which an organization converts conflicting pressures into an ordered course of action. |
| Appetite | The pressure family oriented toward growth, revenue, opportunity capture, resource acquisition, speed, expansion, and competitive advantage. |
| Feasibility | The pressure family oriented toward implementational reality: technical, operational, socio-technical, infrastructural, temporal, and coordination constraints. |
| Constraint | The plural pressure family oriented toward legitimacy, law, compliance, ethics, reputation, stakeholder claims, trust, and social permission. |
| Practical irreversibility | The degree to which an action sequence cannot be undone or repaired without residual state change or unacceptable cost. |
| Sequence sensitivity | The degree to which changing the order of two actions materially changes the resulting organizational state. |
| Power distribution | The relative authority, incentive weight, resource control, dependency position, and escalation capacity of actors carrying appetite, feasibility, and constraint claims. |
| Mediation protocol | The recurring communicative and procedural structure through which pressure claims are surfaced, communicated, weighted, sequenced, translated, and resolved into commitments. |
| Organizational intelligence | The provisional quality of a mediation protocol: its capacity to generate action sequences that remain viable across growth, implementability, and legitimacy pressures over time. |

## 8. Claim Ledger

| Claim | Category | Supporting source or model element | Confidence | Can this appear in the paper? |
|---|---|---|---|---|
| Organization theory already studies persistent contradictory demands. | Directly Sourced | Smith and Lewis (2011). | High | Yes |
| Organizational learning can be discussed through information acquisition, distribution, interpretation, and memory. | Directly Sourced | Huber (1991). | High | Yes |
| Exploration/exploitation is an established organizational-learning tension. | Directly Sourced | March (1991). | High | Yes |
| Team theory treats organizational form in relation to information and decision functions. | Directly Sourced | Marschak and Radner (1972). | High | Yes |
| Management cybernetics is an important predecessor for diagnostic organizational models. | Directly Sourced | Beer (1985). | Medium | Yes, as positioning |
| Communication under partially conflicting preferences may be strategic or limited. | Directly Sourced | Crawford and Sobel (1982); Farrell and Rabin (1996). | High | Yes |
| Constraint is not a single unified force; organizations can face multiple incompatible institutional prescriptions. | Directly Sourced | Greenwood et al. (2011); Thornton et al. (2012). | High | Yes |
| The high-stakes decision episode is the unit of analysis for this model. | Model-Definition | Proposed decision-episode definition. | High as model definition | Yes |
| Appetite, feasibility, and constraint are pressure families that may appear as pressures, actors, claims, evaluations, or actions. | Model-Definition | Proposed triadic pressure-family definition. | High as model definition | Yes |
| $\rho(s)$ is a diagnostic variable for practical irreversibility. | Model-Definition | Proposed construct. | High as model definition | Yes |
| $\kappa(s,t)$ is a diagnostic variable for sequence sensitivity, not a directional welfare score. | Model-Definition | Proposed construct. | High as model definition | Yes |
| The model includes $p$, the relative power distribution among actors carrying appetite, feasibility, and constraint claims. | Model-Definition | Proposed communication and power parameter. | High as model definition | Yes |
| Organizational intelligence can be defined provisionally as the quality of a mediation protocol $\Phi$. | Model-Definition | Proposed protocol model. | Medium | Yes, as theory-building proposition |
| $\rho$ and $\kappa$ are discussion variables and possible tuning knobs in this paper, with later work needed before treating them as quantitative metric centerpieces. | Speculation | Proposed research program. | Medium | Yes |
| The five propositions in this section are testable conjectures for future research. | Speculation | Proposed model. | Medium | Yes |

## References Verified for This Section

Beer, S. (1985). *Diagnosing the System for Organizations*. Chichester: Wiley. ISBN: 0471906751. Stable record: https://openlibrary.org/books/OL2862901M/Diagnosing_the_system_for_organizations.

Crawford, V. P., & Sobel, J. (1982). "Strategic Information Transmission." *Econometrica*, 50(6), 1431-1451. https://doi.org/10.2307/1913390.

Farrell, J., & Rabin, M. (1996). "Cheap Talk." *Journal of Economic Perspectives*, 10(3), 103-118. https://doi.org/10.1257/jep.10.3.103.

Greenwood, R., Raynard, M., Kodeih, F., Micelotta, E. R., & Lounsbury, M. (2011). "Institutional Complexity and Organizational Responses." *Academy of Management Annals*, 5(1), 317-371. https://doi.org/10.5465/19416520.2011.590299.

Huber, G. P. (1991). "Organizational Learning: The Contributing Processes and the Literatures." *Organization Science*, 2(1), 88-115. https://doi.org/10.1287/orsc.2.1.88.

March, J. G. (1991). "Exploration and Exploitation in Organizational Learning." *Organization Science*, 2(1), 71-87. https://doi.org/10.1287/orsc.2.1.71.

Marschak, J., & Radner, R. (1972). *Economic Theory of Teams*. Cowles Foundation Monograph 22. Yale University Press. Stable record: https://cowles.yale.edu/research/cfm-22-economic-theory-teams.

Smith, W. K., & Lewis, M. W. (2011). "Toward a Theory of Paradox: A Dynamic Equilibrium Model of Organizing." *Academy of Management Review*, 36(2), 381-403. https://doi.org/10.5465/amr.2009.0223.

Thornton, P. H., Ocasio, W., & Lounsbury, M. (2012). *The Institutional Logics Perspective: A New Approach to Culture, Structure and Process*. Oxford University Press. https://doi.org/10.1093/acprof:oso/9780199601936.001.0001.
