# Fully learned parameterized Track-B hierarchy

The current curriculum is v4.1. It keeps the reserved macro interface and the
cardinality-invariant composite SMDP from v3, but repairs the transition from
locked-cell temporal learning to unrestricted cell selection. The repair adds
a full-cell spatial-calibration phase, retains decaying teacher supervision in
joint learning, reduces early joint perturbations, and rejects checkpoints
whose deployment behavior falls materially below the calibrated baseline.

The failed v2 temporal run and the v3 joint collapse remain useful internal
ablations. Historical checkpoints are never reinterpreted as v4.1 checkpoints.

## Scope and version identity

This controller learns every high-level Track-B decision:

- temporal mode: accept/store, retrieve/deliver, or defer;
- the retrieval target inside retrieve mode;
- the exact storage cell inside accept mode.

Feasibility masks, proposal validation, path planning, reservations, and option
execution remain deterministic. They define and safely execute the learned
action set; they are not competing policies.

The v4.1 identity is:

- checkpoint schema: `4`;
- trainer: `4`;
- method: `fully_learned_reserved_macro_hierarchy_v4_1`;
- controller: `relational_parameterized_mode_regularized_smdp_v4`;
- action interface: `relational_parameterized_reserved_macro_v2`;
- network: `shared_deepset_candidate_q_v2`;
- replay: `variable_candidate_mode_balanced_common_continuation_smdp_v4`;
- backup: `nested_logmeanexp_raw_q_common_smdp_v2`;
- curriculum:
  `spatial_calibration_replay_warmup_joint_retention_quality_gated_v4_1`.

The network shape is intentionally unchanged from v3. This allows an
authenticated v3 `temporal-end.pth` to initialize v4.1 without importing its
optimizer, replay, exploration clocks, or random state.

## Decision and execution contract

At a hands-free macro decision epoch, the controller constructs

\[
\mathcal C(s)=\mathcal C_A(s)\cup\mathcal C_R(s)\cup\mathcal C_D(s),
\]

where each feasible storage cell is an explicit
`AcceptStore(block, cell)` control, each executable stored block is a named
retrieval control, and defer is present only when it cannot hide a capacity
failure.

Candidate construction is read-only. Only the selected accept candidate is
converted into a proposal token and bound to `ReservedAcceptStoreOption`. The
bound, committed, and executed block, cell, and proposal IDs must agree. There
is no fallback or post-pickup rescoring.

## Cardinality-invariant common continuation

For the live controls in temporal mode \(g\), the controller uses the
uniform-reference within-mode value

\[
U_g(s)=\tau_g\left[
\log\sum_{c\in\mathcal C_g(s)}e^{Q(s,c)/\tau_g}
-\log|\mathcal C_g(s)|
\right].
\]

Across the currently available modes,

\[
V(s)=\tau_M\left[
\log\sum_{g\in G(s)}e^{U_g(s)/\tau_M}
-\log|G(s)|
\right].
\]

Every completed macro uses the same continuation semantics:

\[
y=\eta R^{(k)}+\gamma^k(1-d)V^-(s').
\]

`R^(k)` is the raw discounted environment reward stream accumulated exactly
once. The delivery reward and timing bonus are not added a second time. The
training reward scale `eta` is applied only inside the learner. Environment
termination, time-limit truncation, and method failure remain distinct
boundaries and never bootstrap from a mid-option state. An unsuccessful
time-limit cut receives one discounted training-only completion penalty; this
does not alter the reported environment return.

Teacher scores never enter either side of the TD loss. Bellman predictions and
continuations use raw shared-Q values. Evaluation uses hierarchical MAP:
maximize `U_g` over live modes and then maximize `Q(s,c)` within the selected
mode. Epsilon exploration samples a mode first, so a mode does not receive more
exploration probability merely because it contains more controls.

## Why v3 joint learning collapsed

V3 temporal learning itself succeeded. At episode 200 its preserved
`temporal-end.pth` obtained validation return `1491.24`, strict success `1.0`,
reservation integrity `1.0`, and mean absolute timing error about `3.48`.
After only 25 unrestricted joint episodes, validation return fell to `547.69`
and mean absolute error rose to about `34.33`, while strict success and
reservation integrity remained `1.0`.

This separates the failure from option execution or safety. The temporal mode
counts remained broadly stable, but the selected storage region moved from an
effective lower-right area toward poor upper-left cells. The accept-cell
ranking correlation with REG changed from approximately `+0.66` at the joint
boundary to `-0.67`. The copied spatial encoder itself moved only about
`0.19%`, whereas the shared fusion and Q-head parameters moved about `4.28%`
and `3.81%`. A small TD loss therefore indicated self-consistency on the new
replay distribution, not preservation of a useful cell ranking.

The cause was the action-set discontinuity. Temporal training exposed only one
REG-selected cell inside accept mode. Joint training suddenly exposed roughly
the whole free-cell set, removed teacher retention, restarted epsilon at
`0.20`, and trained shared scoring layers on fresh exploratory replay. The
controller had never been required to rank the non-teacher cells before that
point.

## Training-only teacher and deployment boundary

The frozen REG-v5 checkpoint has only these roles:

1. provide teacher actions in imitation and locked-cell temporal learning;
2. initialize the controller-owned geometric candidate encoder;
3. provide full-cell scores for spatial distillation and joint retention while
   training.

In supervised spatial or joint training, REG scores all feasible accept cells.
Let \(z_c\) be its standardized scores and let \(q_c\) be the learned accept
values. In addition to the hard teacher-mode and teacher-control losses, the
learner minimizes a listwise term of the form

\[
\mathrm{KL}\!\left(
\operatorname{softmax}(z_c/\tau_T)
\;\middle\|\;
\operatorname{softmax}(q_c/\tau_A)
\right).
\]

Standardizing the teacher scores makes this supervision insensitive to their
arbitrary affine scale while preserving their ordering.

REG is not a deployment component. Validation and strict evaluation expose
all feasible cells with `teacher_supervision=false`; the candidate builder does
not call REG's scoring policy. It reuses the fixed geometric feature definition
and the controller-owned encoder initialized from REG, then chooses solely
from learned-Q values. REG replay, optimizer state, and assignment-interval TD
targets are never enabled. Checkpoint metadata records
`deployment_reg_policy_query=false` and, for joint deployment artifacts,
`frozen_reg_runtime_policy_query=false`.

## Four training phases

A full run uses `50/150/50/250` episodes by default. The recommended v4.1 run
below imports the already authenticated v4 spatial endpoint and therefore runs
only `0/0/0/250`.

1. **Imitation.** The repaired duration-aware reserved hierarchy executes the
   teacher action. Hierarchical behavior cloning learns the teacher mode and
   its exact cell or target. Teacher prior logits are excluded from the loss
   and from validation, so neither low loss nor high validation return can be
   produced by leaking the label.
2. **Temporal.** Only the REG teacher cell is exposed inside accept mode. The
   learned controller chooses modes and retrieval targets under the common
   SMDP target while the copied spatial encoder remains frozen. Temporal
   epsilon anneals from `0.25` to `0.05`; teacher-action mixing anneals from
   `0.50` to zero; auxiliary BC anneals from `0.50` to `0.05`. These schedules
   advance on phase-local macro decisions.
3. **Spatial calibration.** Every feasible accept cell is exposed, but the
   duration-aware teacher still executes the trajectory. The copied spatial
   encoder remains frozen while the shared state, candidate, fusion, and Q
   layers learn the hard teacher action and the complete REG cell ordering.
   This phase performs supervised calibration rather than TD learning. Its
   greedy validation is deployment-mode validation: all cells are visible and
   REG is not queried.
4. **Joint.** Every live control remains exposed and TD learning resumes under
   the common SMDP objective. Early joint optimization is deliberately
   conservative:

   - epsilon is fixed at `0.05` instead of restarting at `0.20`;
   - the first 2,000 decisions collect broad replay without TD updates;
   - teacher-action mixing holds at `0.25` for 2,000 decisions, then anneals to
     zero over 8,000 decisions;
   - auxiliary BC/listwise retention holds at `0.20` for 5,000 decisions, then
     anneals to `0.02` over 20,000 decisions;
   - exploratory accept actions are sampled only from the teacher's top eight
     feasible cells while training supervision is active;
   - the copied spatial encoder is frozen for the first 5,000 joint macro
     decisions;
   - joint TD uses one update per eligible macro, rather than the failed run's
     two;
   - main and spatial learning-rate groups are both multiplied by `0.10`,
     giving effective rates `5e-6` and `5e-7` for the recommended base rates.

Teacher action mixing and listwise retention are optimization aids only. The
optional behavior-prior coefficient stays at zero and teacher values never
enter the Bellman backup.

Replay is cleared whenever the candidate-set contract changes, Adam moments
are reset, and the target network is synchronized to the retained local
policy. TD minibatches are balanced across live accept, retrieve, and defer
modes. Uniform replay remains an explicit ablation.

## Validation gates and checkpoint artifacts

The v4.1 import authenticates the v4 spatial source's schema, trainer and
curriculum identity, passing spatial endpoint, selector deployment digest,
geometry, reward/discount contract, model dimensions, and validation protocol.
It copies only `Q_local` weights into the new local and target networks.
Optimizer, replay, phase clocks, histories, and RNG state start fresh. The
source path and SHA-256 digest are retained as provenance.

The quality gates use greedy deployment-mode validation:

- the spatial endpoint must retain at least `90%` of the temporal reference
  return and must not exceed `2x` its mean absolute timing error;
- every joint validation must retain at least `90%` of the immutable spatial
  endpoint return and must not exceed `2x` its mean absolute error;
- strict-success or reservation-integrity regression stops joint training
  immediately, including during warmup;
- return and MAE failures are diagnostic for the first 3,000 joint decisions,
  then require two consecutive failed validations before stopping;
- `best.pth` can be selected only in joint phase, after its quality gate passes,
  using strict success and then mean return.

The output directory contains:

- `latest.pth`: resumable state, including optimizer, replay, RNG, phase clocks,
  complete training history, and validation history;
- `spatial-end.pth`: immutable resumable endpoint and the calibrated reference
  for joint learning;
- `joint-init.pth`: non-resumable diagnostic snapshot at the unrestricted joint
  boundary;
- `best.pth`: non-resumable deployment artifact, emitted only by a gate-passing
  joint validation;
- `training-summary.json`: configuration, histories, validations, best score,
  and artifact paths.

A from-scratch run additionally emits `imitation-end.pth` and
`temporal-end.pth`. A warm-start repair does not fabricate zero-length phase
artifacts.

If the process is interrupted normally, resume from `latest.pth`. If a quality
gate stops the run, its latest checkpoint is explicitly blocked from advancing
the curriculum; preserve the failed directory as diagnostic evidence. To
change the repair settings, create a new output directory from the authenticated
v4 spatial endpoint; do not overwrite or reinterpret the failed run.

## Recommended v4.1 repair run

The spatial endpoint already passed deployment-mode validation at return
`1484.77`, MAE `3.658`, strict success `1.0`, and reservation integrity `1.0`.
Importing it avoids repeating the 50 calibration episodes.

```bash
cd /home/ai_diagnosis/HRL_Management

PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u train_fully_learned_track_b.py \
  --selector-checkpoint results/reg-selector-v5-seed0-local-500ep/best.pth \
  --warm-start-spatial results/fully-learned-v4-seed0-from-v3-temporal-300ep/spatial-end.pth \
  --lambda 0.5 \
  --mu 50 \
  --seed 0 \
  --imitation-episodes 0 \
  --temporal-episodes 0 \
  --spatial-episodes 0 \
  --joint-episodes 250 \
  --max-steps 4000 \
  --batch-size 128 \
  --buffer-size 20000 \
  --update-every 4 \
  --updates-per-macro 2 \
  --joint-updates-per-macro 1 \
  --joint-td-warmup-decisions 2000 \
  --learning-rate 5e-5 \
  --spatial-learning-rate 5e-6 \
  --joint-main-lr-scale 0.10 \
  --joint-spatial-lr-scale 0.10 \
  --joint-spatial-freeze-decisions 5000 \
  --gamma 0.99 \
  --target-tau 1e-3 \
  --grad-clip 5.0 \
  --huber-delta 1.0 \
  --reward-scale 0.01 \
  --failure-penalty -50 \
  --replay-sampling mode_balanced \
  --joint-epsilon-start 0.05 \
  --joint-epsilon-end 0.05 \
  --joint-epsilon-warmup-decisions 0 \
  --joint-epsilon-decay-decisions 20000 \
  --joint-teacher-mixture-start 0.25 \
  --joint-teacher-mixture-end 0.0 \
  --joint-teacher-mixture-warmup-decisions 2000 \
  --joint-teacher-mixture-decay-decisions 8000 \
  --joint-bc-start 0.20 \
  --joint-bc-end 0.02 \
  --joint-bc-warmup-decisions 5000 \
  --joint-bc-decay-decisions 20000 \
  --spatial-distillation-weight 1.0 \
  --teacher-score-temperature 1.0 \
  --joint-accept-exploration-top-k 8 \
  --spatial-min-return-ratio 0.90 \
  --spatial-max-mae-ratio 2.0 \
  --joint-min-return-ratio 0.90 \
  --joint-max-mae-ratio 2.0 \
  --joint-gate-warmup-decisions 3000 \
  --joint-collapse-patience 2 \
  --max-defer-steps 10 \
  --tau-accept 0.1 \
  --tau-retrieve 0.1 \
  --tau-defer 0.1 \
  --tau-mode 1.0 \
  --teacher-coefficient 0.0 \
  --lookahead-margin-steps 2 \
  --validation-seeds 10000 10001 10002 \
  --validation-steps 4000 \
  --eval-every 10 \
  --log-every 5 \
  --device cuda \
  --output-dir results/fully-learned-v4-1-seed0-from-v4-spatial-250ep
```

The first log line should report the authenticated spatial validation and state
that optimizer, replay, clocks, history, and RNG are fresh. Joint starts at
episode 1. `Loss` remains `None` during the first 2,000 macro decisions while
replay and BC supervision accumulate; `Mix` and `BCw` initially remain at
`0.25` and `0.20`.

## Resume an interrupted v4.1 run

Repeat the exact command above, remove `--warm-start-spatial ...`, and add:

```bash
  --resume results/fully-learned-v4-1-seed0-from-v4-spatial-250ep/latest.pth
```

Keep the same output directory and every other argument unchanged. Resume
authentication rejects changes to the phase schedule, geometry, optimizer and
replay contracts, validation protocol, selector digest, or `PYTHONHASHSEED`.

## Strict evaluation

Only a gate-passing `deployment_best` from joint phase is accepted:

```bash
cd /home/ai_diagnosis/HRL_Management

PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u track_b_fully_learned_evaluate.py \
  --checkpoint results/fully-learned-v4-1-seed0-from-v4-spatial-250ep/best.pth \
  --lambda 0.5 \
  --mu 50 \
  --eval-seeds 30000 30001 30002 \
  --max-steps 4000 \
  --device cuda \
  --output results/fully-learned-v4-1-seed0-from-v4-spatial-3seed.json
```

The evaluator rejects legacy schemas, resumable or non-joint artifacts,
teacher-dependent metadata, regime or geometry mismatches, invalid/fallback
assignments, and reservation-integrity violations. Final confirmation should
reuse identical saved `EpisodeInstance` objects for this controller and every
deterministic baseline.
