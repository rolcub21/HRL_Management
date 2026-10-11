# Kim et al. (2020) A3C spatial baseline adaptation

## Status and claim boundary

This baseline adapts the spatial-placement idea in Kim, Jeong, and Shin,
“[Spatial arrangement using deep reinforcement learning to minimise
rearrangement in ship block stockyards](https://doi.org/10.1080/00207543.2020.1748247),”
*International Journal of Production Research* 58(16), 5062–5076, 2020.
Its project name is `kim2020_a3c_spatial_adapted`.

This is an **adaptation, not an exact reproduction**. No public implementation
or original schedules were located, and the article does not specify every
network, optimizer, and training detail needed for a bit-for-bit
reimplementation. Results from this repository must therefore be described as
“Kim et al.-inspired” or “adapted from Kim et al. (2020).” They must not be
reported as reproduced results from the paper.

The comparison is deliberately narrow. It tests the paper's closest relevant
idea—a learned spatial arrangement policy—inside this project's common online
yard, masking, and execution contracts. It does not replace or modify REG-v5,
the deterministic scheduler, or the fully learned hierarchy.

## Paper-level method and results

The following high-level properties come from the paper:

- The method separates transport from storage-location choice. A Transporting
  Agent is trained first; that agent is then fixed while a Locating Agent is
  trained. The agents are not trained jointly.
- Both agents are presented as asynchronous advantage actor-critic (A3C)
  agents.
- The Transporting Agent uses a categorical yard-grid state and primitive
  movement plus load/unload actions. It is trained for 200,000 episodes on a
  `5 x 5` yard with between 1 and 20 randomly placed blocks.
- The Locating Agent chooses one of the yard's storage locations. Its input is
  an `m x (n + 1)` representation: empty cells are `-1`, occupied cells encode
  a relative remaining-time value clipped to `[0, 2]`, and the additional
  row/column encodes the road direction.
- Locating-Agent reward is delayed until the selected block is shipped and is
  larger when that placement requires fewer block movements. Training episodes
  contain schedules of 50–100 blocks; the paper reports 11,000 training
  episodes and approximate convergence after 6,000.
- In the paper's `5 x 5`, one-month, 43-block case, five stochastic evaluations
  averaged 41 rearrangements for A3C, versus 47 for PSLAP, 48 for bottom-left
  fill, and 54 for historical operation. In a virtual `5 x 10` doubled-schedule
  case, the corresponding values were 96, 114, and 132 for A3C, PSLAP, and
  bottom-left fill.

Those numbers characterize the paper's own layouts, schedules, simulator, and
trained policy. They are context only and are not targets that this adaptation
is expected to reproduce numerically.

## Explicit adaptation choices

The repository environment differs from the paper's simulator. The following
choices are consequently part of the method definition rather than inferred
paper details:

- The project keeps its own walls, pickup point, exit gate, storage cells,
  arrival process, timing semantics, and variable-duration operations.
- A shared, safety-checked candidate mask defines the actions available at an
  assignment epoch. The learned policy cannot place a block outside that mask.
- The policy learns only storage-cell selection. Existing deterministic
  navigation and atomic storage/retrieval executors replace the paper's learned
  Transporting Agent so that spatial methods share identical physical control.
- The model is a geometry-aware convolutional actor-critic with a masked
  per-cell policy head and a scalar value head. This architecture is an explicit
  repository choice because the paper does not expose enough detail to recover
  its exact neural network.
- Training uses a reproducible, single-process A3C-style episodic actor-critic
  update. It retains the actor, critic, entropy, delayed-reward, and discounted
  return structure, but it is not a claim to reproduce the paper's
  asynchronous worker implementation.
- The checkpoint records the feature schema, architecture version, geometry,
  training seed, effective training device, deterministic cuBLAS workspace
  configuration, optimizer state, model state, and deployment mode. A
  checkpoint whose geometry or schema is incompatible with an episode must
  fail explicitly rather than silently remap cells.

These choices isolate the scientific comparison: learned dwell-aware spatial
placement versus the project's existing spatial assignment sources, with the
same admissible cells and the same consequences after a cell is chosen.

### Shared retrieval admission (v3)

Track-B comparison runs use `retrieve_deliver_option_v3` with
`named_atomic_retrieval_executor_v3`. The stored-yard planner still constructs
one complete canonical relocation-and-delivery plan. Before the scheduler may
admit that atomic macro, v3 augments the immutable planning yard with every
other currently visible, undelivered block and validates all relocation legs
and the final exit leg. The executor then commits the validated tail instead of
discarding it and choosing a new route after the first relocation.

This validation is read-only and certifies the complete **current-live** plan;
it does not inspect future arrivals. The canonical-only v1 and first-live-leg
v2 executors remain separate historical ablations. Results produced under v1,
v2, and v3 are therefore different execution protocols and must not be pooled.

## Adapted policy

### Observation

At each inbound assignment epoch the source builds a fixed-geometry,
five-channel tensor from the immutable yard snapshot and incoming block:

1. **Relative dwell:** `-1` for an empty cell; for an occupied cell, the
   stored block's nonnegative remaining dwell divided by the incoming block's
   dwell denominator and clipped to `[0, 2]`.
2. **Occupancy:** one for an occupied cell and zero otherwise.
3. **Storage layout:** one for a declared storage cell and zero otherwise.
4. **Traversability:** one for a traversable cell and zero for a wall or other
   non-traversable geometry.
5. **Exit proximity:** normalized inverse Manhattan distance to the nearest
   exit on traversable cells.

The repository-authoritative legal-candidate mask is separate from those five
channels. Empty space and zero remaining dwell remain distinguishable through
the occupancy channel, while the storage and traversability channels separate
walls, roads, and usable storage geometry. The boolean candidate mask is
applied to policy logits, so an illegal cell has exactly zero selection
probability rather than merely a low learned score.

This is the closest information-preserving mapping of the paper's relative
remaining-time grid to the repository's richer geometry. The source receives
only information available at the current online decision epoch; it does not
inspect unarrived blocks or future schedule events.

### Action

An action is one exact cell from the ordered `valid_candidates` set. The actor
produces one logit per yard cell, masks all non-candidates, and forms a
categorical distribution over the remaining cells.

Training samples from that categorical distribution using the source's local
training RNG. Frozen deployment has two declared modes:

- `stochastic`: derive a stateless uniform variate from a cryptographic hash,
  then apply the masked categorical inverse CDF;
- `map`: choose the legal cell with the largest policy logit, with stable
  deterministic tie breaking.

The frozen stochastic hash binds the deployment-checkpoint digest, policy seed,
episode-instance identity, decision time step, block and source coordinates,
ordered candidates, and observation digest. It has no mutable RNG state and
requires no hidden preview cache. Repeated inspection of the identical proposal
therefore returns the same cell, while a different declared policy seed creates
a distinct rollout. If accepted, the ordinary reservation token still binds
that chosen cell through execution.

### Delayed reward

For a placement decision concerning block `b`, let `R_b` be the number of
other blocks relocated while retrieving `b` for delivery. The adapted learning
reward is

\[
r_b = -R_b.
\]

The delivery move of `b` itself is not a rearrangement. The source keeps the
placement decision pending by block label. It counts relocation events during
the atomic retrieval that ends with a delivery event and attaches that count
to the delivered block's earlier placement decision. This is possible because
the common executor completes one retrieval atomically before another delivery
starts.

When an episode is eligible for an update, the source sorts its completed
outcome records by placement epoch. Eligibility unconditionally requires a
complete, uncensored, integrity-valid episode, so every selected placement
reward has resolved. For placements `t = 1, ..., T`, the source computes the
terminal Monte Carlo return

\[
G_t = r_t + \gamma G_{t+1}, \qquad G_T = r_T.
\]

The default is `gamma=0.99`, and valid configured values lie in `(0, 1]`.
`gamma` discounts successive **placement decisions**, not primitive travel
steps or elapsed option durations. The critic target is the scaled `G_t`; the
actor advantage is that target minus the value predicted from the placement
observation. Since every reward included in the return is already resolved
before the backward pass, there is no value bootstrap at episode end.

Checkpoints identify this rule as
`kim2020_episodic_discounted_placement_return_v1`, separately from the
per-block delayed relocation reward contract.

A selected placement whose storage or later retrieval is not observed by
episode end has a **censored outcome**: its unknown relocation cost is not
imputed as zero. A truncated episode records truncation as the censoring reason.
Any censored placement causes the entire episode update to be skipped. The same
is true if the episode fails, truncates, has unmatched store or delivery events,
leaves a retrieval relocation count open, or credits a relocation total that
differs from the observed relocation events. Partial episodic Monte Carlo
updates are not configurable. Completed, censored, unmatched, and
skipped-episode counts remain auditable.

This reward intentionally excludes general environment return, travel-time
bonuses, timing error, and scheduler reward shaping. It trains the spatial
source on the same physical quantity used in the paper-level rearrangement
claim.

### Learning lifecycle

One episode follows this lifecycle:

1. Reset the source, its pending-decision ledger, and episode counters.
2. Admit an inbound decision only after the live agent can reach the pickup
   block. A temporarily sealed approach yields `WAIT` without calling the
   assignment source, consuming policy randomness, or creating a learning
   record.
3. At each admitted inbound epoch, encode the current online observation, mask the
   logits, sample one cell, and record the observation, mask, chosen action,
   block label, and placement epoch. Log probability, value, and entropy are
   recomputed from the current network when the episodic batch is updated.
   An exceptional retry of the same unresolved macro reuses that provisional
   action rather than sampling or recording a second decision.
4. Execute storage and all later operations through the common environment and
   deterministic executors. The source observes relocation and delivery events
   but does not control them.
5. On delivery, resolve that block's delayed relocation outcome.
6. On episode termination, check outcome integrity. If the episode is complete,
   uncensored, and integrity-valid, sort its completed records by placement
   epoch, form their discounted terminal Monte Carlo returns, and update.
   Otherwise skip the entire episode and report its censored or unmatched
   records.

The trainable object is only the spatial source. The schedule generator,
transport controller, retrieval policy, candidate mask, and temporal manager
are not optimized by this training run.

Retrieval planning uses stored inventory as relocatable obstacles and arrived
unstored inventory as hard transient obstacles. This keeps the detached route
graph consistent with live A* execution: a route may not pass through a block
waiting at pickup or the external queue. If inbound inventory temporarily
seals every due retrieval route, the executor stores that inbound block first
and retries retrieval from the resulting live state. If no inbound block can be
cleared, or a reachable inbound block has no admissible storage cell, stored
work is overdue, and the greedy dispatcher still has no plan, the common
executor runs the policy-independent exact recovery verifier and executes the
first transition of its deterministic strict-macro witness. This fallback
applies identically to every Track-A assignment source; it does not alter the
learned candidate mask or forgive the relocation cost it incurs.
Validation rows expose `retrieval_live_plan_failure_count`,
`inbound_approach_defer_count`, and the `exact_recovery_*` counters. An
executable run should have zero live-plan and exact-recovery failures; a
positive exact-recovery fallback count is a valid common-executor event.

## Training contract

`train_kim2020_a3c_spatial.py` trains the source in the Track-A assignment
setting. Track A gives all compared assignment sources the same schedule,
geometry, legal-cell mask, storage executor, and neutral retrieval executor.
Training checkpoints must identify at least:

- method `kim2020_a3c_spatial_adapted`;
- checkpoint schema version `2`;
- adaptation/architecture and observation-schema versions;
- delayed-reward and episodic-return contract identifiers;
- exact-recovery fallback contract and search-node bound;
- training seed and completed episode;
- effective training device and `CUBLAS_WORKSPACE_CONFIG`;
- environment geometry and candidate-cell ordering;
- model and optimizer state;
- hyperparameters, including the placement-decision discount `gamma` (default
  `0.99`), reward scale, value coefficient, and entropy coefficient; and
- completed, censored, and unmatched delayed outcomes.

The best checkpoint is selected on held-out **schedule seeds**, not on the
evaluation schedules later used for the final comparison. Validation evaluates
MAP once and stochastic deployment over five fixed policy-seed rolls on every
held-out instance. A checkpoint must have strict success on every stochastic
roll before it is eligible; eligible checkpoints are ranked by the primary
stochastic mean obstructive-move count and then stochastic return. MAP remains
a separately reported secondary diagnostic and neither rejects nor ranks the
paper-facing stochastic A3C checkpoint. This matters because A3C trains the
categorical policy, whereas its deterministic argmax realization is a distinct
policy that the objective does not directly optimize. No Track-B metric is used
to update the spatial model.

A standard training command is:

```bash
PYTHONHASHSEED=0 .venv/bin/python -u train_kim2020_a3c_spatial.py \
  --episodes 11000 \
  --seed 0 \
  --gamma 0.99 \
  --output-dir results/kim2020-a3c-spatial-seed0
```

Deterministic PyTorch algorithms are enabled by default. Before importing
PyTorch, the trainer supplies the CUDA-compatible default
`CUBLAS_WORKSPACE_CONFIG=:4096:8` when the variable is absent. A caller's
explicit supported value (`:4096:8` or `:16:8`) is preserved. Deterministic
CUDA training rejects any other value before constructing the policy, rather
than failing later in a cuBLAS matrix operation. The effective device and
workspace value are written to every checkpoint and the training summary.

An interrupted run can resume only from its own `latest.pth` and original
output directory. `--episodes` is the new total target, not the number of
additional episodes:

```bash
PYTHONHASHSEED=0 .venv/bin/python -u train_kim2020_a3c_spatial.py \
  --resume results/kim2020-a3c-spatial-seed0/latest.pth \
  --episodes 20000 \
  --seed 0 \
  --gamma 0.99 \
  --output-dir results/kim2020-a3c-spatial-seed0
```

Resume requires the same full model/training configuration, training seed,
validation seeds and policy-seed settings, geometry, workload regime, fixed
`PYTHONHASHSEED`, deterministic-algorithm setting, effective training device,
and cuBLAS workspace configuration as the original run. The trainer restores
model and optimizer state, its local training RNG, counters, and diagnostic
histories; verifies that `training-history.json` ends at the checkpoint's
completed episode; and preserves the best eligible validation checkpoint found
before interruption. The new total must exceed the completed episode count.
Any non-default original arguments and externally selected cuBLAS workspace
value must be repeated exactly.

For implementation smoke tests, reduce `--episodes`; such runs are engineering
checks and not reportable experiments. The command-line interface is defined
by the script's `--help`, and the saved manifest is the authoritative record
of all defaults used by a run.

## Track-B evaluation contract

Evaluation loads a frozen checkpoint into the production duration-aware
reserved-cell hierarchy:

```text
duration-aware deterministic temporal manager
├── AcceptStore(block, exact Kim-policy cell)
│   └── reserved atomic inbound executor
├── StrictRetrieveDeliver(block)
│   └── common strict retrieval executor and neutral relocation rule
└── DeferUntilEvent
    └── common bounded event-driven defer executor
```

The source previews one exact legal cell at the scheduler decision epoch. The
duration-aware manager evaluates the corresponding
`AcceptStore(block, cell)` macro. If acceptance is selected, the proposal token
is bound and the executor commits that same cell after pickup, without a
second policy call or substitution. If another macro is selected, the proposal
is discarded without changing the yard.

This reserved-cell contract matters especially for a stochastic policy: the
cell whose duration and counterfactual effect the manager evaluated is exactly
the cell that is executed. The source is frozen throughout Track B. It receives
no Track-B gradient or parameter update.

Every compared spatial source uses the same:

- immutable episode instance and arrived-information regime;
- candidate-cell mask and candidate ordering;
- duration-aware temporal manager and lookahead settings;
- exact-cell reservation and validation contract;
- navigation, storage, strict retrieval, relocation, and defer executors; and
- failure, truncation, and censoring rules.

Thus a result measures the consequences of the spatial proposal, not a private
transport policy or a different feasibility rule.

## Stochastic and deterministic reporting

The primary result follows the paper's stochastic deployment. For every
schedule seed, evaluate five distinct policy seeds and average the five rolls:

\[
\bar y_j = \frac{1}{5}\sum_{k=1}^{5} y_{j,k},
\]

where `j` identifies a fixed schedule/episode instance and `k` identifies only
the policy seed used by stateless categorical sampling. All methods see the
same set of schedule seeds. The stochastic Kim policy uses the same five
declared policy seeds for every rerun, and those seeds are written to the
output manifest.

The secondary result uses `map` deployment once per schedule seed. MAP is
useful for operational repeatability and diagnoses whether categorical
sampling helps, but it must not replace the five-roll stochastic primary result
or be pooled with it.

Schedule seeds and policy seeds are separate experimental factors:

- a **schedule seed** fixes arrivals, durations, geometry-dependent episode
  identity, and all exogenous workload data;
- a **policy seed** controls only the frozen Kim policy's stateless sampling;
- deterministic baselines and Kim-MAP are run once for each schedule seed;
- paired stochastic contrasts are computed within schedule seed after
  averaging policy rolls, so one schedule with five rolls does not count as
  five independent workloads.

Final comparison schedule seeds must also be disjoint from every learned
checkpoint's training instances and checkpoint-selection validation instances.
Before evaluation, the runner reconstructs the Kim checkpoint's training-seed
range and validation-seed set and checks the requested `--schedule-seeds`
against both. When a REG-v5 checkpoint is supplied, it performs the same audit
for REG. Missing or otherwise unverifiable seed provenance fails this check,
just as an observed overlap does.

The default is to stop before any evaluation when disjointness cannot be
verified. `--allow-seed-overlap` bypasses that stop only for an explicitly
diagnostic run; it must not be used for a confirmatory final comparison. The
output records each checkpoint's verification status, training range,
validation seeds, observed overlaps, and disjointness result in
`seed_disjointness_audit`, together with whether the diagnostic override was
used.

The comparison runner evaluates both stochastic and MAP deployment under this
contract. A standard command is:

```bash
PYTHONHASHSEED=0 .venv/bin/python -u compare_kim2020_a3c_spatial.py \
  --kim-checkpoint results/kim2020-a3c-spatial-seed0/best.pth \
  --reg-checkpoint results/reg-selector-v5-seed0/best.pth \
  --lambda 0.5 \
  --mu 50 \
  --schedule-seeds 1000 1001 1002 1003 1004 \
  --stochastic-rollouts 5 \
  --policy-seed-base 20000000 \
  --output results/kim2020-a3c-spatial-comparison.json
```

`--reg-checkpoint` is optional; omit it when REG-v5 is not part of the requested
comparison. Supply the complete final schedule-seed list explicitly. The runner
derives the fixed stochastic policy-seed rolls from `--policy-seed-base` and
records both seed namespaces in its output. The example intentionally omits
`--allow-seed-overlap`; the chosen seeds must pass the checkpoint-provenance
audit.

## Baselines and comparisons

The registered matched comparator includes:

- `kim2020_a3c_spatial_adapted` with stochastic deployment;
- the same checkpoint with MAP deployment as a within-method secondary
  comparison;
- `dynamic_pslap`;
- `nearest_free`; and
- `reg_selector_v5` with deterministic deployment when `--reg-checkpoint` is
  supplied.

It runs those methods on the exact same immutable episode instance, averages
the Kim stochastic rolls within that instance, and only then creates
cross-schedule summaries and paired contrasts. A rolling
`pslap_ga_2009_rolling` result may be added through a separate matched
evaluation when computational budget permits; it is not currently executed by
`compare_kim2020_a3c_spatial.py`.

`pslap_ga_2009_offline` may be included only as an explicitly
information-advantaged reference because it sees the complete schedule. It is
not a like-for-like online baseline. The paper's reported PSLAP, bottom-left
fill, and historical numbers must not be inserted as if they were evaluations
on this repository's schedules.

The cleanest attribution uses two views of the same frozen source:

1. **Track A, spatial-only:** compare assignment sources with the temporal and
   execution context held fixed.
2. **Track B, system-level:** compare the same sources inside the common
   duration-aware reserved hierarchy.

Any comparison against a jointly learned hierarchy is a whole-controller
comparison and should be labeled separately from the spatial-source ablation.

## Required metrics and audit

The primary operational metric is actual `obstructive_moves` produced by the
common executor. The registered comparator retains every raw run. Its cost and
performance means and sample standard deviations use only protocol-valid
schedule rows for:

- actual `obstructive_moves`;
- `obstructive_moves_per_delivered_block`;
- environment `return`;
- `mean_absolute_error`; and
- primitive `steps`.

Its protocol audit also reports success, strict-method success, reservation
integrity, invalid assignments, fallbacks, illegal drops, and an aggregate
`protocol_valid` flag. A run is protocol-valid only if strict method success and
reservation integrity both hold and invalid assignments, fallbacks, and
illegal drops are all zero. Paired deltas use
`kim_stochastic_minus_comparator`, so a negative relocation delta favors Kim.

Every method summary reports `schedule_count`, `eligible_schedule_count`, and
`censored_schedule_count`. Here, an evaluation-censored schedule is retained in
the feasibility and failure audit but excluded from that method's cost summary;
its raw cost is not admitted as an eligible observation and no replacement is
imputed. For a paired contrast, a schedule is eligible only when both
Kim-stochastic and the comparator are protocol-valid. Pairwise cost deltas and
win/tie/loss counts use only those jointly valid schedules, and the contrast
separately reports its eligible and censored counts.

For the stochastic primary analysis, the runner first averages the five policy
rolls within each schedule seed, then compares those schedule-level values
against the paired baseline result. It retains the individual raw rolls so
policy variability remains visible and repeated policy rolls never masquerade
as independent schedules. A Kim-stochastic schedule row is protocol-valid only
when every constituent rollout is protocol-valid. Training manifests
separately retain completed, censored, unmatched, and skipped delayed-outcome
counts; frozen Track-B inference does not manufacture a delayed-reward training
ledger.

Failure handling is part of the result, not a preprocessing choice. Do not
discard failed, truncated, or otherwise protocol-invalid schedules from
success-rate denominators. `success_rate` and `strict_method_success_rate`
remain unconditional over all requested schedules even though costs are
conditional on protocol validity. Metrics whose terminal value is unknown
after truncation are **censored** and must be reported with their observed
count; they must not be silently treated as zero, as successful, or as
completed deliveries. Cost findings must therefore always be shown beside
unconditional failure and eligibility/censoring rates, especially when methods
have different eligible counts.

Executor-v3 validation uses a two-stage seed protocol. Seeds `44000..44049`,
which exposed eight v2 route failures, are regression-only and cannot support a
new final performance claim. The exact failure tuples are checked by
`verify_kim2020_v3_retrieval_regression.py`. Seeds `45000..45049` were then
evaluated with the seed-0 model before the project adopted a three-training-seed
paper design, so that artifact is retained as a preliminary single-model
result rather than promoted post hoc.

The frozen paper adaptation trains independent model seeds `0`, `1`, and `2`
under the same v7 configuration and evaluates all three on the preregistered,
previously unused schedules `97000..97049`. The five stochastic policy rolls
are first averaged within each model/schedule pair. Paper tables report every
model separately, mean and sample standard deviation across the three trained
models, and paired schedule contrasts after giving the three models equal
weight. Repeated deterministic-baseline runs are checked for equality and are
not counted as three independent observations. See
`experiments/kim2020_paper_three_seed/README.md` for the locked commands and
`aggregate_kim2020_three_seed.py` for the registered estimand.

## Registered three-seed paper result

The frozen three-model evaluation uses training seeds `0`, `1`, and `2`, five
stochastic policy rolls per model/schedule, and 50 common paper schedules
`97000..97049`. All 1,200 raw executions were protocol-valid under retrieval
executor v3: there were no censored schedules, strict-method failures, invalid
assignments, fallbacks, reservation violations, or illegal drops. Repeated
deterministic-baseline rows were identical across the three artifacts.

After averaging policy rolls within model/schedule and giving the three models
equal weight, the schedule means are:

| Method | Rearrangements | Timing MAE | Return | Primitive steps |
|---|---:|---:|---:|---:|
| Kim stochastic | 1.117 | 10.986 | 352.540 | 348.240 |
| Kim MAP | 1.373 | 12.808 | 342.529 | 338.900 |
| Dynamic PSLAP | 2.160 | 23.150 | 287.124 | 264.400 |
| Nearest free | 0.000 | 64.443 | 212.165 | 274.280 |

Kim stochastic rearrangements were `1.117 +/- 0.069` (mean and sample standard
deviation across the three independently trained models). Relative to dynamic
PSLAP, it reduced rearrangements by 1.043 per 12-block schedule (48.3%),
improved timing MAE by 12.164, and increased return by 65.416, while requiring
83.840 additional primitive steps. All three model-specific rearrangement
deltas favored Kim (`-0.980`, `-1.032`, and `-1.116`; Kim minus dynamic). The
training-seed t interval for the mean rearrangement delta is
`[-1.213, -0.872]` with `df=2`; it should be presented with the three raw
model-seed effects because only three independent models were trained.

Nearest free remains the best rearrangement-only baseline, with zero
rearrangements on every paper schedule, but has much worse timing MAE and
return. The defensible claim is therefore that the stochastic Kim-inspired
source improves rearrangement, timing, and return over dynamic PSLAP under the
shared online hierarchy—not that it minimizes rearrangements among every
baseline. MAP is a secondary ablation: its rearrangements vary sharply across
training seeds (`2.780`, `0.020`, and `1.320`), so the average stochastic-versus-
MAP difference is not robust to model-seed uncertainty.

The authoritative aggregate is
`results/kim2020-a3c-spatial-6x8-stress-v7-paper-three-seed-summary.json`.

## Interpretation

A favorable result supports a limited claim: under the repository's online
information and common execution contract, a Kim et al.-inspired dwell-aware
actor-critic can learn useful storage-cell placement. It does not establish
reproduction of the paper, superiority of learned transport, or superiority
under the paper's undisclosed schedules.

A neutral or unfavorable result is still informative. It distinguishes the
benefit of the paper's spatial representation and stochastic policy from
differences in transport, simulator, schedule, and action admissibility—all of
which are controlled here rather than inherited from an unavailable reference
implementation.
