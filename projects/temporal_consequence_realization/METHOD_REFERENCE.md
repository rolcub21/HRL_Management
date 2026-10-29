# Canonical method reference: law–condition consequence realization

**Authority:** normative reference for every implementation in this project

**Revision:** `TCR-METHOD-2026-10-29.8`

**Status:** living specification

**Stable path:** `projects/temporal_consequence_realization/METHOD_REFERENCE.md`

This document defines the method. Experiment protocols freeze particular
implementations of it; reports record evidence about those implementations.
Neither a protocol nor a report changes the method unless the change is also
recorded here.

The authority order is this reference, then a revision-pinned experiment
protocol, then its implementation, then its report. Older design notes preserve
the history of the idea but are not normative when they disagree with this
file.

The words **MUST**, **SHOULD**, and **MAY** distinguish required mechanisms,
preferred practice, and optional extensions. An experiment that violates a
MUST requirement is a diagnostic or ablation, not a canonical implementation,
and must say so explicitly.

## 1. Mechanism in one paragraph

The persistent environmental representation is a shared consequence law plus a
belief over conditions attached to places or entities. A state estimator
extracts the currently observable physical situation, and dynamic memory tracks
changing hidden physical processes when the current observation is not Markov.
A condition says which persistent mechanism acts, its instance-specific
attributes, and its pose under applicable world symmetries. The law reads the
current physical state, dynamic state, action, and only the conditions involved
in that action; it expresses them in a canonical frame and predicts the full
next consequence—including dynamic-state changes and any conditions the action
writes. Estimator, memory/filter, law, conditions, prior, capacity, and
vocabulary are charged by one description-length objective. Values, plans,
successor-like readouts, and exploration targets are derived from this law and
current belief; they are not separate world knowledge.

### 1.1 Architectural principles

1. **Shared generative law.** Learn action-consequence mechanisms independently
   of downstream objectives.
2. **Explicit persistent conditions.** Represent enduring environmental
   conditions through a structured categorical-attribute-pose field.
3. **Predictive responsibility separation.** State estimation represents the
   current physical situation, dynamic memory represents changing physical
   processes, the persistent field maintains enduring condition knowledge, and
   the shared law explains how they generate consequences. Persistent knowledge
   acquired from history must not be duplicated in an uncontrolled memory that
   bypasses the field.
4. **Coherent belief inference.** Maintain and update uncertainty over the
   relevant persistent and changing hidden conditions.

Exploration is an optional acquisition mechanism, not a fifth representational
principle.

## 2. What is stored

The canonical learned architecture stores:

1. a state estimator `E_omega` and, when needed, parameters for a dynamic-state
   filter;
2. a shared law (or calibrated belief over shared laws) `Phi`;
3. a prior over persistent fields and changing latent state;
4. a joint posterior belief `b_t(U,m)` over the current world's persistent
   conditions `U` and changing hidden physical state `m`. When no hidden dynamic
   state is needed, this reduces to `q_t(U)`.

The estimator and filter are legitimate learned components, but they are not an
independent store of persistent world knowledge. A currently visible wall may
naturally be represented in estimator features, and an equipped tool may be
part of the current physical state. What is forbidden is retaining historical
knowledge of an unobserved persistent condition in a separate learned memory so
that prediction no longer depends on the designated field belief.

Planning values, transition graphs, SR-like matrices, and route caches MAY be
stored as derived computational caches. They MUST be recomputed or updated when
the law or condition belief changes. They are not an additional representation
of how the world works.

## 3. Formal objects

### 3.1 Dynamic state

The state

\[
s_t \in \mathcal S
\]

contains the current physical variables needed for consequences: position,
velocity, posture, inventory, status, and other relevant variables. An
observation adapter may produce

\[
z_t=E_\omega(o_t,\hat m_t),
\]

where `z_t` is the estimated currently observable situation and `m_t` denotes
changing hidden physical state. Examples of `m_t` include an interaction-local
block-breaking process, a delayed pickup process, contact mode, or another
transient process whose value cannot be recovered from the current observation
alone. If a necessary variable is not observed, the agent MUST maintain a
belief over it. A non-Markov predictive state is an implementation failure, not
evidence against the law–condition factorization.

Persistence duration alone does not determine component membership. An equipped
tool can remain in current state for hundreds of steps; a block-breaking process
can be hidden yet last only a few steps. Assignment follows predictive role:
current physical situation, changing hidden process, or enduring carrier-level
condition.

The classification rule for a hidden quantity attached during interaction is:

- if it remains attached to a carrier after the interaction ends and can affect
  a later return, it is a persistent field attribute that the law reads and
  writes;
- if it lapses when the interaction ends, it is changing dynamic state.

Thus persistent block damage belongs to the block's field condition, whereas
standard Minecraft-style break progress that resets when mining stops belongs
to `m`. The adapter MUST declare this lifecycle rather than classifying the
quantity from duration alone.

### 3.2 Predictive responsibility separation

Let `U` be the persistent condition field and `m` changing hidden physical
state. The predictive distribution may be written

\[
P(o_{t+1}\mid h_t,a_t)
=\int P_\Phi(o_{t+1}\mid z_t,m,a_t,U)\,
b_t(U,m)\,dU\,dm.
\]

The decomposition need not be statistically independent: `b_t(U,m)` MAY retain
dependence between persistent and changing hidden variables. Its explanatory
responsibilities MUST nevertheless be explicit:

- `E_omega` determines what current physical information is available;
- `b_t` maintains uncertainty informed by history;
- `U` carries enduring place-, object-, or entity-specific conditions;
- `m` carries changing hidden physical processes;
- `Phi` generates and writes consequences from these quantities.

An encoder feature may reveal a currently visible persistent condition. This is
not leakage. A history-aware estimator or memory MUST NOT independently retain
that condition after it is unobserved in a way that makes the field causally
irrelevant to prediction. Learned estimator and memory capacity MUST be charged
by the same total objective as the law and field.

#### 3.2.1 Responsibility audits

Architectures capable of bypassing the field MUST be tested with the following
audits when the required matched cases exist:

1. **History leakage.** Compare histories with the same current observation and
   equivalent changing state but different earlier evidence about a persistent
   condition. Undeclared persistent evidence must not survive outside the
   designated condition belief.
2. **Field intervention.** Hold estimated current physical and changing state
   fixed while varying the persistent-field belief. Predictions must change
   when the field alternatives imply different consequences.
3. **Field ablation.** Neutralize the persistent belief and measure degradation
   on predictions that require previously acquired condition knowledge.

A successful probe of persistent information from encoder features is not by
itself a failure because the condition may be currently visible. Conversely,
field ablation need not hurt when the current observation already determines
the condition. Protocols MUST state which matched cases make each audit
informative and report `not_applicable` with evidence when none exist.

### 3.3 Condition carriers and field support

Let `X` be the addressable carriers of persistent conditions. A carrier may be a
spatial place, object, potion appearance, inventory slot, relation, terrain
patch, or another entity on which a persistent fact can sit.

The condition field is

\[
U = \{u(x): x \in X\}.
\]

The field MUST cover every carrier that a one-step consequence can read or
write, including reachable places outside the range visited by the offline data.
Unseen carriers begin at the prior. They MUST NOT be omitted, treated as
impassable, or silently clipped onto a visited carrier.

For continuous spatial fields, support SHOULD include the observed range plus a
margin covering the maximum one-step footprint and body/contact radius. Adding
support must preserve the resolution and alignment of existing cells when
possible.

### 3.4 A condition

Each carrier has

\[
u(x) = \big(c(x),\theta(x),g(x)\big).
\]

- `c(x)` is a discrete, anonymous category selecting a mechanism.
- `theta(x)` is a category-conditional attribute, possibly empty. Examples are
  boundary offset, strength, color, direction, or effect magnitude.
- `g(x)` is a pose in a symmetry group `G`, with the identity pose when no
  symmetry applies.

Categories have no semantic labels during learning. A category is justified only
when the consequences it explains save more description length than it costs.
Continuous attributes MUST have charged precision; arbitrarily tiny codes that a
high-gain network can amplify are not valid cheap facts.

Compositional conditions MAY contain several discrete parts when consequences
combine independent mechanisms. The same description-length accounting applies.

### 3.5 Beliefs

The agent stores a posterior, not merely a best map. Without changing hidden
dynamic state this is

\[
q_t(U)=P(U\mid h_t).
\]

More generally it is the joint belief

\[
b_t(U,m)=P(U,m\mid h_t).
\]

Unseen conditions retain the prior. A categorical condition requires categorical
or mixture uncertainty capable of representing “possibly category A, possibly
category B.” A single smooth point estimate is insufficient when alternatives
have distinct consequences.

Uncertainty about the world and uncertainty about the shared law are distinct.
If several laws `Phi_m` are retained, their weights MUST depend on predictive
evidence or description length and SHOULD update after real consequences. Equal
weights are not retained when one law explains the evidence worse.

## 4. The shared canonical law

### 4.1 Local read and write sets

For physical state `s`, changing hidden state `m`, action `a`, and field `U`,
the law identifies:

- a read set `R(s,m,a,U)` containing carriers whose conditions can affect the
  consequence;
- a write set `W(s,m,a,U)` containing carriers whose conditions the action may
  change.

Examples include a swept motion path, source and destination cells, the targeted
object, or inventory slots involved in an action. The read/write construction
MUST follow the action's causal footprint, not dataset support.

An environment adapter MAY supply a general locality envelope or interaction
interface. Any more specific supplied attachment—such as “read the destination
cell”—must be declared. When attachment is claimed as learned, selection or
attention within the envelope must be fitted without grading labels and charged
by the objective.

The involved conditions `U_R` form a set, not an arbitrary list of fixed input
slots. The shared law MUST read them through a permutation-invariant set reader,
or through a permutation-equivariant construction followed by invariant
aggregation. Causal roles, relative poses, and action-footprint coordinates MAY
distinguish members and preserve relational or path structure; changing only
the enumeration order MUST NOT change the predicted consequence. Set-reader
capacity and any cardinality limit are charged by the same objective.

When one carrier is responsible for a mutually exclusive consequence, the law
MUST compute that carrier's consequence in the carrier's own canonical frame
before combining alternatives. It MUST NOT first pool differently posed carrier
tokens in an unrelated frame and ask one decoder to reconstruct which carrier
acted. A canonical responsibility mixture has the form

\[
\begin{aligned}
P_\Phi(y\mid s,m,a,U_{\mathcal R})
&=\pi_0\,P_{\Phi,0}(y\mid s,m,a)\\
&\quad+\sum_{x\in\mathcal R}\pi_\Phi(x\mid s,m,a,U_{\mathcal R})
\,(T_{g_x}^{-1})_{\#}
P_\Phi^{\mathrm{canon}}\!\left(
T_{g_x}y\mid T_{g_x}(s,m,a),c_x,\theta_x
\right),
\end{aligned}
\]

where `y` is the full consequence, `pi_0` is the no-carrier/default
responsibility, and the nonnegative responsibilities sum to one. The
responsibility construction MUST be permutation-equivariant over carriers, so
the resulting mixture is permutation-invariant. It MAY depend on condition
content when responsibility genuinely depends on what occupies a carrier; this
is distinct from a content-independent candidate-relevance envelope used to
decide which beliefs receive evidence.

Posterior responsibility is also the attribution distribution for condition
updates. If several carriers act sequentially—such as two wall contacts during
one swept move—the law SHOULD recursively compose canonical single-carrier
consequences over the remaining motion. If mechanisms genuinely act jointly or
additively, an interaction law or canonical sufficient-statistic aggregation is
allowed, but the combination rule and its symmetry must be declared and learned;
raw posed effects cannot be averaged merely for convenience.

When mutually exclusive carriers are encountered in a known temporal or path
order, responsibility MUST preserve that order rather than average unordered
effects. Let `tau_x` be the first point at which the action footprint reaches
carrier `x`, and let `h_x` be its conditional probability of acting once reached.
The canonical first-event construction is

\[
\pi_x=h_x\prod_{y:\tau_y<\tau_x}(1-h_y),
\qquad
\pi_0=\prod_x(1-h_x),
\]

or an equivalent permutation-equivariant survival/hazard model. The adapter
MUST declare how `tau_x` is obtained and how simultaneous events are composed.
This is a general sequential composition rule for swept interactions, not a
domain label: it applies whenever the earliest contacted carrier determines the
immediate consequence. If later events can act on remaining motion, the law then
recurses from the first event.

### 4.2 Canonicalization by symmetry

Let `T_g` transform states, actions, conditions, and consequences into the
canonical frame induced by the relevant poses. The law is

\[
\begin{aligned}
&P_\Phi(s_{t+1},m_{t+1},U'_{\mathcal W}
\mid s_t,m_t,a_t,U_{\mathcal R})\\
&\quad = (T_g^{-1})_{\#}\,
P_\Phi^{\mathrm{canon}}\!\left(
T_g(s_{t+1},m_{t+1},U'_{\mathcal W})
\mid T_g(s_t,m_t,a_t),c_{\mathcal R},\theta_{\mathcal R}
\right),
\end{aligned}
\]

where pushforward by `T_g^{-1}` returns the predicted consequence to the world
frame.

The law MUST NOT see absolute carrier identity or address unless that dependence
is itself selected and charged. Applicable symmetries may include translation,
rotation, reflection, permutation, or relabeling. Their role is pooling: posed
instances of one mechanism train one canonical consequence rule.

The symmetry group, its precision, and how poses are obtained MUST be stated.
They may be supplied as a general physical assumption, learned, or selected by
description length; these are different claims.

### 4.3 One likelihood over full consequences

The law predicts the complete consequence, including the next observable or
physical state, the transition of changing hidden state, and any persistent
conditions written by the action. “Do nothing” or “no intervention” is an
action of this same law. A background predictor MUST NOT be fitted separately
and subtracted from outcomes as if effects were necessarily additive.

Structured branches—such as free motion gated against contact—MAY exist inside
one conditional likelihood. They are still one law and must be fitted under the
same objective. A default branch MUST NOT depend on conditions that, by
definition, cannot affect it.

## 5. Learning by one description length

The normative objective is

\[
L_{\mathrm{total}}
=L(E_\omega,\Phi,\psi,\xi)
+D_{\mathrm{KL}}\!\left(b(U,m)\,\|\,P_\psi(U,m)\right)
+\mathbb E_{(U,m)\sim b}
\left[-\log P_{E_\omega,\Phi,\xi}(\mathcal D\mid U,m)\right],
\]

where `xi` denotes any learned dynamic-state filter. An implementation MAY use a
factorized prior or variational family, but it must charge every learned
component and declare the approximation.

When no changing hidden state is needed, `b(U,m)` reduces to `q(U)` and the
dynamic-filter terms disappear. The expectation MUST be trained with samples
from `b` (or `q` in that special case), rather than replacing the belief by one
best latent, when calibrated uncertainty is claimed.

The objective selects, jointly or through a declared model-selection schedule:

- the state estimator and dynamic-filter family and capacity;
- the number of categories;
- attribute families and precision;
- the symmetry family and pose precision;
- the law's capacity;
- the prior's capacity and structure.

The law and condition vocabulary are coupled parts of this selection. Whenever
a retrained law materially changes predictive sharpness or likelihood, the
condition attribute family and its precision MUST be reselected under the same
total cost; inheriting a vocabulary selected for an earlier law is diagnostic,
not conforming condition-learning evidence. A sharper law can make previously
adequate quantization or fixed support predict precise false consequences, so
additional attribute precision must be allowed to pay for itself in likelihood.

Held-out predictive likelihood MUST participate in capacity and temperature
selection. Labels may grade recovered meanings but MUST NOT choose kinds,
features, capacities, stopping points, or control actions.

A weight penalty MAY be used as a provisional approximation to
`L(E_omega,Phi,psi,xi)`, but must be declared as an approximation. Fixed
penalties are not evidence that the full description-length selection has been
implemented.

### 5.1 Preferred alternating fitting procedure

When exact joint fitting is not used, implementations SHOULD alternate the
following two responsibilities:

1. **Learn the law—what and when.** Pool events across carriers and worlds in
   the canonical frame, integrating or sampling the current carrier beliefs.
   Fit the shared full-consequence likelihood without grading labels.
2. **Infer the field—where.** With the current law fixed, update each carrier's
   enumerable posterior from every action footprint that informs it. Attribute
   observed events and count informative non-events. Use exact enumeration when
   feasible; otherwise declare and charge the variational, particle, or other
   approximation.

Repeat until the declared total-cost and held-out predictive criteria converge.
The prior and any shared attributes MAY update in the law phase or a separately
declared step. A different joint or alternating schedule is allowed, but its
responsibility assignment and evidence must be stated.

### 5.2 Optimization requirements

The following distinguish optimization failures from objective failures:

- compare learned solutions with corrected or oracle-coded alternatives under
  the same total cost;
- retain the cheapest converged restart;
- record gradient clipping and non-finite updates;
- calibrate likelihood variances so rare events cannot be absorbed solely by
  arbitrary uncertainty;
- require specialized components to improve their own conditional means, not
  merely mixture likelihood;
- keep training, selection, and final evaluation data separate.

Rare-event sampling is allowed. If it changes the sampling distribution, exact
importance weights MUST restore the declared objective.

When a rare structured branch initially assigns essentially zero likelihood to
the events it should explain, a **default-plus-correction** fit MAY bootstrap it
without consequence labels. First fit the common default under the same full
likelihood. Then introduce a broad correction distribution whose posterior
responsibility captures consequences the default cannot explain. Finally replace
or refine that broad correction with the structured, carrier-frame consequence
law, using the broad responsibilities only as a label-free continuation signal.
The handoff, final model, and all selection remain under the unchanged total
description length; the broad correction is not a separately subtracted
background model.

Low-dimensional structural scalars or finite alternatives that are difficult to
learn by gradient—for example a footprint radius—MAY be enumerated on a declared
grid and selected by untouched total description length. Their search range,
precision, and code cost MUST be declared. Grading truth must not choose the
grid point.

### 5.3 Learn first, then prune; restoration audit

When a charged read set, mixture component, category, or other learned part can
eliminate its own training signal, an implementation MAY use a fixed,
predeclared continuation schedule without changing the model or final score:

1. fit the explanatory components with their pruning/read charge at zero and a
   broad, label-free initialization;
2. ramp that charge to its declared final value over a fixed number of updates;
3. jointly refine at the full charge.

The final checkpoint and all hyperparameters are still selected by untouched
total description length. Grading labels, true attributes, and oracle component
assignments MUST NOT control the schedule. A scalar or finite structural choice
MAY instead be enumerated and selected by the same untouched total cost.

Every applicable fit MUST perform a **restoration audit**. If a learned part is
unused or nearly unused, restore it to a plausible label-free setting—such as
its pre-pruning checkpoint or another predeclared initialization—and rescore it
under the same final total description length. If restoration is cheaper, the
reported outcome is an optimization failure, not evidence that the objective or
data reject that part. The restoration candidate, unused threshold, complete
score decomposition, and verdict MUST be retained in the report.

## 6. Belief updating

After a real transition, the joint posterior updates by the same law:

\[
b_{t+1}(U',m')\propto
\sum_{U,m} b_t(U,m)
P_\Phi(s_{t+1},U',m'\mid s_t,m,a_t,U)^\eta.
\]

When no changing hidden state is required, this reduces to

\[
q_{t+1}(U')\propto
\sum_U q_t(U)
P_\Phi(s_{t+1},U'\mid s_t,a_t,U)^\eta.
\]

The likelihood temperature `eta` is chosen on held-out predictive calibration.
It is not a hand-set confidence guarantee.

The update MUST touch every condition in the action's read or write footprint
that the observation informs, and every dynamic latent informed by the
transition. In a deterministic world, a consequence impossible under one
category should eliminate that category up to the learned observation noise.
Exact discrete updates are preferred when the alternatives can be enumerated.

Persistent conditions remain fixed across imagined future steps unless the law
predicts that an action writes them. Changing hidden state instead evolves under
the law. Resampling a new world independently at every imagined step destroys
persistence and is nonconforming.

Calibration MUST be measured by comparing stated probabilities with realized
frequencies on held-out events.

## 7. Derived consequences and readouts

One-step consequences, multi-step rollouts, transition graphs, reachability,
values, and successor-like readouts are computed from the law and a joint
persistent/dynamic-state sample or belief.

For finite abstractions, a default policy induced by the law may define

\[
G=(e^\beta I-P_0)^{-1},
\]

with goal-specific absorbing corrections and low-rank updates after local field
changes. This is a computational readout of current knowledge, not a separately
learned world representation.

## 8. Control

For goal set `G*`, a complete sampled world `U`, changing hidden state `m`, and
a law sample `Phi_j`, values are computed from the consequence law:

\[
V(s,m,U)=
\begin{cases}
1,&(s,m,U)\in G^*,\\
\gamma\displaystyle\max_a
\mathbb E_{(s',m',U')\sim P_\Phi}
[V(s',m',U')],&\text{otherwise}.
\end{cases}
\]

Goals are absorbing. When speed matters, utility SHOULD use discounted first
hitting time rather than “touches the goal anywhere in the horizon,” which can
create oscillating ties.

Decision making averages or applies a declared risk functional over complete
world and law samples:

\[
a^*=\arg\max_a\;\rho_{\Phi,U,m}
\left(
\mathbb E_{(s',m',U')\sim P_\Phi}
[V(s',m',U')]
\right).
\]

`rho` may be the posterior mean, a lower quantile, CVaR, or another stated risk
attitude. A reliability threshold MUST apply to a success probability, not to a
discounted value. Complete persistent-condition samples remain fixed for the
plan except where simulated actions write them; changing hidden state evolves
under the law.

After every real consequence, the agent updates its belief and replans. Frozen
and online-adapting evaluations must be reported separately, and comparison
agents receive their natural equivalent adaptation budget.

## 9. Exploration and data selection

The optional active objective chooses reachable experiments that reduce
uncertainty about the shared law and local conditions:

\[
\text{target}^*=arg\max_{\text{target}}
\frac{
\mathbb E[I((\Phi,U,m);o\mid h,\text{target})]
}{
\text{steps to reach}+\text{steps to test}
}.
\]

Information gain MUST use the learned likelihood, including confusable outcomes;
an unknown condition cannot be assumed perfectly identified by one probe.
Planning SHOULD include travel cost or imagined belief updates so useful nearby
tests are batched.

On a fixed dataset, the analogous mechanism is label-free surprise selection:
prioritize transitions the current law explains poorly, then use exact importance
weights. Surprise identifies evidence for missing causes; grading labels do not
enter selection.

## 10. Environment adapter contract

Every implementation MUST document the following adapter, even when an item is
the identity map:

| Adapter element | Required declaration |
|---|---|
| State | Variables or posterior needed for Markov consequences |
| State estimator | Observation inputs, recurrence, output state, and how current persistent evidence is represented |
| Dynamic memory | Changing hidden processes, prior/update, and any coupling with the persistent field |
| Carriers | What can hold a persistent condition |
| Support | How all reachable read/write carriers are represented |
| Condition family | Categories, attributes, poses, and their priors |
| Read set | How the prospective action footprint is obtained |
| Write set | Which conditions an action may change |
| Symmetry | Group, canonicalization, and pose source |
| Law likelihood | Full predicted consequence and observation noise |
| Belief update | Exact, variational, particle, or other inference method |
| Law uncertainty | Posterior/ensemble construction and evidence weighting |
| Control | Goal, utility, horizon/readout, and risk functional |
| Exploration | None, passive surprise, or active information objective |
| Oracle inputs | Any supplied structure used only for diagnosis |

## 11. Canonical conformance checklist

An implementation is canonical only if all applicable items pass.

### Representation

- [ ] One shared law is reused across carriers and worlds.
- [ ] The shared law reads each involved condition set permutation-invariantly;
      arbitrary enumeration order cannot change a consequence.
- [ ] Mutually exclusive carrier effects are computed in each carrier's
      canonical frame and combined by permutation-equivariant responsibility;
      jointly acting or sequential effects use an explicitly declared
      composition rule.
- [ ] State estimator, changing dynamic memory, persistent field, and shared law
      have explicitly declared predictive responsibilities.
- [ ] Persistent condition knowledge acquired from history cannot bypass the
      designated field through an independent learned estimator or memory.
- [ ] Applicable history-leakage, field-intervention, and field-ablation audits
      verify responsibility routing; non-applicability is justified.
- [ ] Conditions are persistent carrier-level facts, not transition outcomes.
- [ ] Categories are anonymous and attributes/poses are charged.
- [ ] The field covers reachable unseen carriers, initialized from the prior.
- [ ] Absolute addresses are absent from the shared law unless selected and
      charged.

### Learning

- [ ] Full consequences come from one conditional likelihood.
- [ ] Law, conditions, and prior are fitted jointly or by a declared alternating
      procedure; departures from the preferred pooled-law/per-carrier-inference
      schedule are justified.
- [ ] Belief samples, not only point estimates, enter training when uncertainty
      is claimed.
- [ ] Capacity/vocabulary choices use total cost and held-out likelihood.
- [ ] Labels are grading-only.
- [ ] Any biased sampler has exact importance correction.
- [ ] Any learned part that becomes unused receives a restoration audit under
      the same final total description length; a cheaper restored model is
      reported as an optimization failure.

### Inference and control

- [ ] Conditions persist across the imagined plan unless written.
- [ ] Beliefs update after every real consequence.
- [ ] Law alternatives are evidence-weighted.
- [ ] Calibration is measured on held-out data.
- [ ] Goals are absorbing and utility/risk quantities are not conflated.
- [ ] Values and plans are derived from the current law and belief.

### Reproducibility

- [ ] Protocol records this method revision.
- [ ] Data, split, implementation, protocol, and checkpoint hashes are stored.
- [ ] Diagnostic oracle inputs and deviations are explicit.
- [ ] Optimization diagnostics and failed gates are retained.

### Required implementation declaration

Every new protocol or run manifest MUST include the equivalent of:

```yaml
method_reference: projects/temporal_consequence_realization/METHOD_REFERENCE.md
method_revision: TCR-METHOD-2026-10-29.8
claim_level: expressibility | law-learning | condition-learning | belief-control | full
declared_deviations: []
oracle_inputs: []
```

The declaration records what was actually run. It must not be changed
retroactively when this reference advances; reruns against a later revision get
a new manifest and output directory.

### Enforced run and report contract

Every new or rerun experiment MUST validate its protocol with
`method_conformance.py` before loading data, training, or control. The protocol
must also contain a `conformance_audit` entry for every checklist item above.
Each entry has a status (`pass`, `fail`, or `not_applicable`) and one line of
evidence.

Every report MUST begin with the rendered conformance audit, before results. A
failed item or a shortcut from Section 12 is part of the run's claim level, not
a footnote. In particular, `claim_level: full` requires every checklist item to
pass and is invalid when the protocol declares any oracle input, deviation, or
nonconforming shortcut. Diagnostic and partial runs remain useful at their
accurately limited claim level.

## 12. Nonconforming shortcuts and what they test instead

| Shortcut | What it actually tests |
|---|---|
| One predictor per kind | A model library, not one shared law with small codes |
| Background subtraction | An additive-effect assumption, not the full law |
| Best latent only | Point-estimate control, not belief-aware prediction |
| Visited-only graph/field | Supported-only planning, not prior-filled unseen places |
| Independent world sample each step | Marginal uncertainty without persistence |
| Equal permanent law weights | Unupdated model averaging |
| Labels mapped to useful actions | Supervised control, not law-derived control |
| Perfect-information probes | An oracle exploration heuristic |
| Uncorrected rare-event oversampling | A different training objective |
| Fixed absolute-position input | Address memorization unless explicitly selected |
| Shrinkable continuous facts with unbounded decoder gain | Free memorization, not charged conditions |

Such shortcuts may be useful diagnostics. Their results must not be attributed to
the canonical method.

## 13. Claim levels

To keep implementations and evidence separate, reports should use these levels:

1. **Expressibility diagnostic:** oracle conditions/poses show that the law can
   represent a mechanism.
2. **Law-learning gate:** conditions or geometry may be supplied, but the law is
   learned without consequence labels.
3. **Condition-learning gate:** anonymous conditions, attributes, and poses are
   inferred from consequences.
4. **Belief/control gate:** calibrated online inference drives successful control.
5. **Full method:** law, conditions, belief, control, and any exploration satisfy
   this reference without undeclared oracle structure.

Passing one level does not imply the next.

## 14. Current boundary of evidence at this revision

- Deterministic FourRooms supports shared discrete condition kinds, pooled
  consequence learning, online condition updates, and law-derived control. Some
  attachment and translation structure was supplied in specific runs and is
  reported as such.
- PointMaze establishes label-free learning of a precise wall-frame contact law
  when geometry attributes are supplied. In the anonymous-field audit, 12 of 14
  directly evidenced single-line carriers were recovered (85.7%); all committed
  boundaries were real, and recovered geometry was precise. A pooled action-frame
  set head failed despite oracle geometry: its contact branch was worse than free
  motion. On the same panel the frozen carrier-frame law reached contact AUROC
  0.9997, recall 0.816, specificity 0.9991, and reduced contact error by 53.9%.
  Multi-carrier footprints therefore require canonical per-carrier consequences
  combined by responsibility, not raw token pooling. An oracle first-wall
  responsibility interface subsequently matched the frozen reference exactly
  on AUPRC, recall, and contact error, covered all 49 contacts, and was invariant
  to carrier enumeration. A subsequent learned-responsibility pilot preserved
  the carrier-frame interface and exact permutation invariance but collapsed to
  the no-wall component: it recalled 0 of 49 contacts and left radius near 0.507.
  A label-free held-out loss-surface audit located a substantially cheaper
  solution near radius 0.7 (mean NLL 3.301 versus 3.422 for the selected model),
  with likelihood savings far exceeding the extra read code. Learned
  responsibility is therefore still an optimization frontier, not evidence
  that description length rejects contact. A subsequent predeclared
  learn-first/prune-later Stage-A pilot also collapsed: radius remained 0.508,
  contact recall was 0/49, and the contact branch was worse than free motion.
  Its nominal restoration candidate retained near-zero responsibility and
  therefore failed the substantive restoration requirement on post-run review.
  A final one-attempt default-plus-correction protocol then cleanly separated
  the remaining problem: its broad correction captured all 49/49 contacts with
  only 0.00135 mean responsibility on free moves, and held-out description
  length selected the grading-true radius 0.7 from a six-value grid. The
  structured handoff nevertheless failed: AUPRC was 0.951, but recall was only
  0.306 and the selected carrier contact branch was 41.2% worse than free
  motion. This is evidence that broad event capture and scalar radius selection
  work, while attribution to the correct canonical carrier consequence remains
  the PointMaze bottleneck. A subsequent evaluation-only audit replaced only
  the unordered carrier weights by analytic first swept-body contact. It covered
  all 49 contacts and was exactly invariant to carrier enumeration, but the
  conditional contact RMSE remained 0.0550, identical to the unordered reader
  and above the 0.0255 wall-frame reference. Thus ordering alone cannot repair
  the frozen checkpoint: the canonical per-carrier stop predictor must be
  refitted under ordered responsibility. The final Stage-A fit did exactly that,
  without contact labels: AUPRC reached 0.972, recall 1.000, contact-branch RMSE
  0.00713, and expected contact RMSE 0.00680, while free expected RMSE remained
  0.000402 and carrier-permutation differences were exactly zero. The selected
  radius remained the label-free total-cost choice 0.7. This passes the
  supplied-carrier law-learning gate and shows that broad event capture,
  finite structural selection, temporal first-event responsibility, and joint
  refitting of responsibility with the canonical consequence branch solve the
  Stage-A contact problem. Anonymous inference and control did not run.
  Three subsequent conforming Stage-B condition-learning attempts did run and
  failed. The strongest version used a label-free global
  default-versus-broad-correction posterior before structured attribution. It
  proposed all 159/159 training contacts with 0.837 precision, so event
  discovery was no longer the bottleneck. Nevertheless it committed 151
  anonymous wall segments, only 39 of which matched supported physical
  segments (precision 0.258; evidenced recall 0.661). Correct matches had exact
  C4 pose and 0.053-cell projection error, but phantom carrier/pose assignments
  made the held-out field unusable: contact AUPRC was 0.081, recall 0.286, and
  expected contact error was 26.6% worse than free motion. This is a failed
  condition-learning gate. It locates the remaining frontier at persistent
  carrier attribution and rejection of geometrically plausible aliases, not
  rare-event discovery or the supplied-carrier law. An evaluation-only audit
  then reconstructed the exact capped non-event sample. Sixty-five of 112
  phantom segments were contradicted by at least five free passages in the full
  40,000-transition fit set, but only 23 of those 65 received zero crossing
  contradictions in the capped sample. Thus capping contributes but is not the
  primary cause: most strongly contradicted phantoms survived evidence already
  present in the factorized update. Whole-map or joint-block explaining-away
  remains the principal Stage-B inference frontier. A subsequent deterministic
  complete-map coordinate test confirmed both sides of that diagnosis. It
  reduced 151 segments to 32, removed 91 of 112 phantom segments, and lowered
  held-out free-motion RMSE from 0.01986 to 0.00241. But it also reduced matched
  segments from 39 to 11 and evidenced-wall recall from 0.661 to 0.186; contact
  recall was 0.184 and expected contact error remained worse than free motion.
  Complete-map competition therefore performs explaining-away, but this single
  MAP approximation resolves unresolved location uncertainty by over-pruning.
  It is not evidence that a calibrated posterior over complete maps has been
  implemented, and Stage B remains unpassed. A grading-only restoration audit
  then scored the 59-segment physical map restricted to directly evidenced
  boundaries under the identical complete-map objective. It cost 79,329.55
  nats versus 66,369.84 for the selected 32-segment map, so the fit objective
  itself prefers the sparse map by 12,959.71 nats. Only 165.16 nats of that
  gap are extra map code; the rest is likelihood. The null branch is not
  absorbing bumps (mean non-null contact responsibility was 0.936 for the
  selected map and 0.983 for the physical map). Instead, the projected physical
  map fires on free fit moves (mean non-null responsibility 0.0947; free RMSE
  0.0553). On the untouched selection panel the physical map was narrowly
  cheaper, revealing additional split overfitting, but not reversing the full
  fit deficit. This is a scoring/representation failure, so more powerful map
  search is not yet justified; the physical boundary vocabulary or likelihood
  must first be made compatible with observed free passage. A subsequent
  grading-only frozen-law vocabulary audit reproduced that score exactly and
  isolated the missing attribute. Increasing normal-offset precision from 9 to
  33 bins while retaining fixed one-cell extents still left the evidenced
  physical map 11,710.50 nats more expensive than the selected sparse map.
  Adding charged finite start/end attributes reduced the deficit to 63.53 nats
  at 9-offset/17-endpoint precision; 17/17 and 33/33 vocabularies made the
  evidenced physical map cheaper by 221.03 and 536.73 fit nats, respectively.
  Expected free-motion RMSE fell from 0.05526 to 0.00104 and 0.000507. The
  audit's first presentation charged the complete map code again on the
  untouched panel. A revision-2026-10-29.8 accounting audit corrected that
  double charge: held-out predictive NLL favored all finite-extent candidates,
  and the combined fit-plus-selection description length with code charged once
  favored 9/17, 17/17, and 33/33 by 194.3, 516.5, and 858.4 nats. The
  identification diagnostic therefore passes. This remains oracle grading
  evidence that finite extent is the missing representational attribute, not an
  anonymous Stage-B pass. A subsequent anonymous Stage-B run used four
  finite-chain initializations per vocabulary and reversible complete-map
  toggle, offset, endpoint, and joint-attribute proposals. It selected 17/17,
  but its 54-segment MAP cost 87,851.65 combined nats, 1,143.46 more than the
  86,708.19-nat directly evidenced physical restoration under the identical
  vocabulary and code-once objective. Only 4 segments matched (precision 0.074;
  evidenced recall 0.068); contact AUPRC was 0.027, contact recall 0.122, and
  free-motion RMSE 0.01035. All 59 evidenced true carrier/pose slots were in the
  693-slot proposal universe. However, within-vocabulary best-chain objectives
  differed by 3,940--5,329 nats, and the selected chain accepted only 8 births;
  each birth drew uniformly from 2,312 offset/endpoint alternatives. The
  retained samples therefore do not establish a converged or calibrated map
  posterior. This is a complete-map proposal/mixing failure, not a renewed
  scoring or expressibility failure, and anonymous Stage B remains unpassed.
  Untouched walls remain at the prior, so anonymous
  geometry is condition-learning evidence on evidenced single-line carriers,
  not complete-maze recovery.
- Symbolic Alchemy provides non-spatial evidence for law/condition separation in
  tested slices; full-task comparison remains a separate benchmark claim.
- The NetHack canonical potion-v2 run is evidence for the canonical
  law/condition architecture at its recorded revision: one deterministic
  full-consequence law with no per-drink latent improves second-use prediction
  by about 25% at 4x, and its selected checkpoints are on par with or ahead of
  the selected GRU at 1x, 2x, and 4x. Restart ranges overlap, so this is not a
  statistical-superiority claim. Restricted NetHack pipelines remain excluded.
- In the symbolic hard-world exploration gate, pooled kind-action information
  crossed 90% prediction after 44 causal probes; the direct appearance-action
  model never crossed 90% despite about 295 probes at the common 360-step
  budget. This is controlled symbolic evidence for information-per-effort
  exploration, not pixel or partial-observation evidence.
- ProcGen exposed useful collection, perception, and conformance failures;
  restricted pipelines from that line are not evidence for the full method.
- Minecraft stepwise diagnostics show that dynamic event state can improve a
  high-precision subset of delayed writes when routed to intervention streams,
  while broad event memory is needlessly costly. The flat law still wins total
  description length and Gate 1 remains open; this motivates responsibility
  separation but is not a law-learning pass.

The current PointMaze sequence has passed Stage A at the law-learning claim
level: supplied physical wall carriers, a label-free selected body radius, and
temporally ordered carrier-frame responsibility yield a precise shared contact
law. Stage B has now been attempted and failed: event discovery works, but the
posterior does not yet assign consequences to the correct persistent boundary
carriers without creating phantom segments. Evidence acquisition for untouched
walls and belief-updating room-holdout control therefore remain downstream
Stage-B/Stage-C claims; control must not be reported from this checkpoint.

## 15. Maintenance and change procedure

This file is the stable consultation point. Update it when a result changes the
mechanism, not merely when a new benchmark is added.

For every normative change:

1. increment the revision at the top;
2. add a revision-history entry below;
3. state the evidence or failure that motivated the change;
4. identify which adapters or implementations are affected;
5. update protocols to record the new revision;
6. preserve old reports under the revision they actually tested.

Clarifications that do not change required behavior may keep the same major
revision but still receive a dated history entry. Environment-specific choices
belong in protocols, not in this reference, unless they reveal a general
principle.

### Revision history

| Revision | Date | Change |
|---|---|---|
| `TCR-METHOD-2026-10-29.8` | 2026-10-29 | Required condition attribute families and precision to be reselected whenever a materially changed or sharpened law changes their likelihood tradeoff. Motivated by the PointMaze frozen-law vocabulary audit: fixed one-cell wall extents remained at least 11,710.50 fit nats worse than the sparse map even with 33 offset bins, while charged finite endpoints plus 17/17 or 33/33 precision made the evidenced physical map 221.03 or 536.73 fit nats cheaper and reduced free-motion RMSE from 0.05526 to 0.00104 or 0.000507. A same-revision accounting correction charged the map once across fit and held-out data; combined margins favored 9/17, 17/17, and 33/33 finite extents by 194.3, 516.5, and 858.4 nats. Affects every future condition-learning protocol that inherits a vocabulary after retraining its law; historical runs remain pinned to their recorded revisions. |
| `TCR-METHOD-2026-10-29.7` | 2026-10-29 | Added label-free default-plus-correction bootstrapping for rare structured branches, finite total-cost selection for difficult low-dimensional structural choices, and survival/hazard responsibility for carriers encountered in temporal or swept-path order. Motivated by the PointMaze default-plus-correction pilot: the broad correction captured 49/49 held-out contacts with 0.00135 mean free-event responsibility, untouched description length selected radius 0.7, and the remaining failure was unordered attribution among several canonical wall carriers. Affects future rare-event fits and every adapter whose first encountered carrier determines a consequence; historical runs remain pinned to their recorded revisions. |
| `TCR-METHOD-2026-10-28.6` | 2026-10-28 | Added a declared learn-first/prune-later continuation schedule for components that can erase their own learning signal, and made restoration audits under the unchanged final description length mandatory. Motivated by the PointMaze learned-responsibility pilot, where simultaneous fitting collapsed to the no-wall branch even though a label-free held-out radius audit found a substantially cheaper contact-using solution. Affects future mixture, read-set, category, and sparse-component fits; historical runs remain pinned to their recorded revisions. |
| `TCR-METHOD-2026-10-28.5` | 2026-10-28 | Required mutually exclusive carrier effects to be predicted in each carrier's canonical frame and combined by permutation-equivariant responsibility; distinguished responsibility from a content-independent evidence envelope; specified recursive composition for sequential contacts. Motivated by the PointMaze matched-panel regression: a pooled action-frame set head failed to specialize with oracle geometry, while the frozen wall-frame law on the same 10,000 transitions achieved AUROC 0.9997, recall 0.816, specificity 0.9991, and 53.9% contact-error improvement. Affects future multi-carrier readers and belief-attribution updates; historical runs remain pinned to their recorded revisions. |
| `TCR-METHOD-2026-10-28.4` | 2026-10-28 | Clarified persistent-field versus changing-state classification by lifecycle rather than duration; made permutation-invariant reading of involved condition sets explicit; added the preferred pooled-law/per-carrier-posterior alternating fit and consolidated the joint-belief objective; updated NetHack, PointMaze, and symbolic exploration evidence. Motivated by the PointMaze attribution/set-reader results, the NetHack canonical potion-v2 scaling result, and the symbolic hard-world exploration gate. Affects future adapters with carrier-attached processes, multi-carrier footprints, or learned fields; historical protocols remain pinned to their recorded revisions. |
| `TCR-METHOD-2026-10-28.3` | 2026-10-28 | Added explicit predictive responsibility separation among state estimation, changing dynamic memory, persistent condition belief, and the shared law. Replaced any strict timescale interpretation with a functional-role criterion; added history-leakage, field-intervention, and field-ablation audits; expanded the adapter and joint-belief/objective declarations. Motivated by prediction-equivalent FourRooms law-field/history representations and Minecraft event-memory factorization. Affects all future protocols, especially partially observed or recurrent adapters; historical evidence remains pinned to its recorded revision. |
| `TCR-METHOD-2026-10-28.2` | 2026-10-28 | Made conformance operational: every new or rerun protocol must pass the shared validator, declare evidence for every checklist item, place the audit before results, and limit its claim level when it uses deviations, oracle inputs, shortcuts, or failed items. |
| `TCR-METHOD-2026-10-28.1` | 2026-10-28 | Established the canonical reference after the FourRooms, Alchemy, NetHack, ProcGen, and PointMaze lines. Consolidated discrete categories with attributes and poses; canonical-frame symmetry; full-support fields; one full-consequence likelihood; sampled beliefs; evidence-weighted law uncertainty; law-derived control; and information-per-effort exploration. |
