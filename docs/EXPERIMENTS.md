# Experiments and evaluation

The current unified VCG experiment chain—including exact VCG 1.1 nesting, the
five-level handling-weight family, unseen 89k confirmation, and matched 89k
comparators—is indexed separately in [Unified VCG](UNIFIED_VCG.md). The
sections below document the broader repository and historical experiment
families.

## Terminology

- `lambda` / `arrival_rate`: rate of the exponential arrival process; the mean
  inter-arrival interval is `1 / lambda`.
- `mu` / `proc_mean`: mean of the Poisson distribution used for required
  storage duration.
- training seed: seed used while fitting an HRL or GA-assisted HRL checkpoint.
- evaluation seed: independently controls a common evaluation episode.
- episode instance: immutable arrival times, storage durations, and exact yard
  geometry consumed by every method in a matched comparison.
- success: all 40 blocks were delivered before the step limit.
- delivery error: mean recorded delivery-time error for an episode. Existing
  files can contain signed values; some plots use its absolute interpretation.

## Entry-point matrix

| Entry point | Method | Purpose | Status in this copy |
|---|---|---|---|
| `example/run_small_rooms_example_hrl.py` | HRL | train learned-selector sensitivity runs | code and output convention present |
| `example/run_small_rooms_example_ga.py` | GA-assisted HRL | train with GA storage assignment | assignment file missing |
| `example/run_small_rooms_example_pslap.py` | PSLaP | heuristic sensitivity grid | self-contained in principle |
| `common_evaluate.py` | all three | one seed-controlled evaluation row | usable when required artifacts exist |
| `run_learned_common_evaluation.sh` | HRL and GA | batch learned-policy comparison | paths refer to another machine |
| `plot_revision_sensitivity.py` | all three | aggregate sensitivity plots | referenced result directories missing |

## Training experiments as currently configured

### Learned selector HRL

The script uses diagonal parameter pairs:

```text
(lambda=0.2, mu=20)
(lambda=0.5, mu=50)
(lambda=1.0, mu=80)
```

It trains seeds 3 and 4. The active `is_quick_test=True` branch sets 2,500
episodes, 4,000 steps per episode, and a 100-episode plotting window. The
inactive branch sets only 2,000 episodes, so the name "quick test" is currently
misleading.

### GA-assisted HRL

The grid, seeds, episode count, and agent hyperparameters match the learned
selector experiment. It requires:

```text
results/ga_20260311-093907-20 Gen/best_assignment.pkl
```

That file and directory are absent from this repository.

### Dynamic and legacy PSLAP sensitivity

The active configuration evaluates the Cartesian product:

```text
lambda in {0.2, 0.5, 1.0}
mu     in {20, 50, 80}
seed   in {0, 1, 2}
```

For each combination it runs five episodes with a 4,000-step limit. This is
9 settings times 3 seeds times 5 episodes, or 135 episodes in total.

The repaired baseline is the default. Select the frozen ablation explicitly:

```bash
python -m example.run_small_rooms_example_pslap \
  --baseline dynamic_pslap

python -m example.run_small_rooms_example_pslap \
  --baseline legacy_adapted_pslap
```

## Common evaluator

`common_evaluate.py` appends one row to a CSV and accepts:

```text
--method {hrl,ga,pslap,dynamic_pslap,pslap_ga_2009_offline,
          pslap_ga_2009_rolling,pslap_ga_2009,legacy_adapted_pslap}
--lambda FLOAT
--mu FLOAT
--eval-seed INT
--instance PATH          optional exact EpisodeInstance JSON to replay
--save-instance PATH     optional path for the exact replay JSON
--train-seed INT          optional metadata, default -1
--checkpoint PATH         required for HRL and GA
--ga-assignment PATH      required for GA
--allow-legacy-checkpoint explicitly permit an unverified old checkpoint
--max-steps INT           default 4000
--output PATH             required
```

Examples, after setting `PYTHONPATH` as shown in the main README:

```bash
# The repaired dynamic baseline does not need a learned checkpoint.
python common_evaluate.py \
  --method dynamic_pslap --lambda 0.2 --mu 20 --eval-seed 0 \
  --output results/evaluation.csv

# Frozen internal ablation; keep its rows explicitly labeled.
python common_evaluate.py \
  --method legacy_adapted_pslap --lambda 0.2 --mu 20 --eval-seed 0 \
  --output results/legacy-evaluation.csv

# Information-advantaged full-schedule GA reference.
python common_evaluate.py \
  --method pslap_ga_2009_offline --lambda 0.2 --mu 20 --eval-seed 0 \
  --output results/pslap-ga-offline-evaluation.csv

# Deployable rolling GA using arrived information only.
python common_evaluate.py \
  --method pslap_ga_2009_rolling --lambda 0.2 --mu 20 --eval-seed 0 \
  --output results/pslap-ga-rolling-evaluation.csv

# HRL needs a compatible checkpoint.
python common_evaluate.py \
  --method hrl --lambda 0.2 --mu 20 \
  --train-seed 3 --eval-seed 0 \
  --checkpoint /path/to/model_HRL.pth \
  --output results/evaluation.csv

# The older GA-assisted HRL method additionally needs a saved assignment.
python common_evaluate.py \
  --method ga --lambda 0.2 --mu 20 \
  --train-seed 3 --eval-seed 0 \
  --checkpoint /path/to/model_HRL.pth \
  --ga-assignment /path/to/best_assignment.pkl \
  --output results/evaluation.csv
```

`--method pslap` is accepted as a compatibility alias but is written to the CSV
as `dynamic_pslap`. The CSV schema includes `obstructive_moves` and
`illegal_drops` and `assignment_fallbacks`. Every row also includes an
`information_regime`. `pslap_ga_2009_offline` is the full-schedule reference;
`pslap_ga_2009_rolling` is the arrived-information baseline. The old
`pslap_ga_2009` name canonicalizes to the offline method. Both remain distinct
from `dynamic_pslap` and the older learned `ga` method.

The evaluator samples one immutable `EpisodeInstance` before constructing a
controller. This removes the former hidden resampling caused by using
`env.reset()` to infer a learned network's input size. `env.reset(instance=...)`
then installs the same arrivals, storage durations, exits, storage cells, and
room geometry for every method. The CSV's content-derived `instance_id`, not
the seed label alone, is the pairing key. For an auditable replay artifact:

```bash
python common_evaluate.py \
  --method dynamic_pslap --lambda 0.5 --mu 50 --eval-seed 100 \
  --save-instance results/instance-l05-m50-s100.json \
  --output results/evaluation.csv

python common_evaluate.py \
  --method pslap_ga_2009_offline --lambda 0.5 --mu 50 --eval-seed 100 \
  --instance results/instance-l05-m50-s100.json \
  --output results/evaluation.csv
```

An instance whose recorded seed, environment rates, or geometry differs from
the command fails validation before policy execution. Because the isolated
generator uses NumPy's current `default_rng`, historical seed-only rows may map
to different schedules; they must not be mixed with rows from this protocol.

For a small matched-seed assignment comparison, including a geometry that
actually activates the obstruction objective, run:

```bash
python compare_pslap_baselines.py \
  --methods dynamic_pslap pslap_ga_2009_rolling pslap_ga_2009_offline \
  --seeds 100 101 102 \
  --lambda 1.0 --mu 80 --exit-width 3 \
  --max-steps 2000 \
  --output results/pslap-narrow-gate.csv
```

The environment default gate should be evaluated separately. Changing only
arrival rate and processing duration did not guarantee that obstructive moves
occurred, so zero-obstruction settings cannot establish relative obstruction
management quality.

The evaluator forces greedy selection for learned policies. It reconstructs
the networks, loads manager and worker weights, and reconstructs the learned
selector architecture from its checkpoint when evaluating HRL. It validates
the saved controller ordering, selector feature version, return definition, and
event-return discount before loading.
Checkpoints without that metadata fail by default; the legacy override emits a
warning and should not be used for result comparisons.

## Track A assignment-isolation evaluator

`track_a_evaluate.py` compares assignment sources under the same strict
interface and neutral downstream stack. Supported methods are:

```text
reg_selector
reg_selector_v4
reg_selector_v5
nearest_free
dynamic_pslap
pslap_ga_2009_rolling
pslap_ga_2009_offline
```

Use one saved EpisodeInstance for every row:

```bash
python track_a_evaluate.py \
  --method nearest_free \
  --lambda 0.5 --mu 50 --episode-seed 100 --ga-seed 3100 \
  --save-instance results/track-a-l05-m50-s100.json \
  --audit-output results/track-a-nearest-s100.json \
  --output results/track-a.csv

python track_a_evaluate.py \
  --method pslap_ga_2009_rolling \
  --lambda 0.5 --mu 50 --episode-seed 100 --ga-seed 3100 \
  --instance results/track-a-l05-m50-s100.json \
  --audit-output results/track-a-rolling-s100.json \
  --output results/track-a.csv
```

A focused information-safe REG selector checkpoint can be trained with:

```bash
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python train_reg_selector.py \
  --lambda 0.5 --mu 50 --seed 0 \
  --episodes 20 --max-steps 4000 \
  --output-dir results/reg-selector-v3-seed0-20ep
```

`PYTHONHASHSEED` is mandatory because controllers are held in sets; fixing it
before interpreter startup makes their iteration order reproducible. The
trainer validates the final checkpoint schema, selector feature version,
selector weights, and return definition, then writes `training-summary.json`.
The 20-episode setting is a behavioral pilot, not a converged confirmation run.

Evaluate that checkpoint with:

```bash
python track_a_evaluate.py \
  --method reg_selector \
  --lambda 0.5 --mu 50 --episode-seed 100 --ga-seed 3100 \
  --instance results/track-a-l05-m50-s100.json \
  --checkpoint results/reg-selector-v3-seed0-20ep/models/DATE_HRL.pth \
  --audit-output results/track-a-reg-s100.json \
  --output results/track-a.csv
```

`episode_seed` controls only instance generation. GA search uses its own local
`ga_seed`; routing and fallback tie-breaking are deterministic. The CSV stores
a compact audit with candidate-mask hashes. `--audit-output` stores every full
candidate tuple, raw proposal, validated proposal, executed cell, invalidity
reason, and fallback flag.

Interpret the two result streams separately:

- strict results stop attribution at the first invalid method proposal;
- operational results allow the shared emergency fallback and explicitly count
  contaminated steps and deliveries.

`no_feasible_candidates` is downstream infeasibility, not an invalid method
proposal. Offline GA rows remain labeled `offline_full_schedule` and must be
reported separately from online methods.

The prior version-2 selector used required durations of unarrived blocks for
feature ordering, which violated the online information contract. Version 3
removes that leakage. Track A rejects old checkpoints rather than silently
evaluating them under the changed observation definition, so the REG selector
must be retrained once before this comparison.

### Train REG-v4 under Track A

REG-v4 is trained through the same neutral downstream executor and shared mask
used for comparison:

```bash
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  python train_reg_selector_v4.py \
  --lambda 0.5 --mu 50 --seed 0 \
  --episodes 100 --max-steps 4000 \
  --validation-seeds 10000 10001 10002 \
  --eval-every 20 --device cpu \
  --output-dir results/reg-selector-v4-seed0-100ep
```

The validation seeds select `best.pth`; they are not final test episodes. Use a
separate untouched seed set for manuscript evaluation. `latest.pth`,
`best.pth`, `training-history.json`, and `training-summary.json` are written to
the output directory. All selector optimization controls—including replay,
batch size, update count, Huber loss, gradient clipping, and exploration—are
explicit CLI arguments.

Evaluate the selected model on an independently saved EpisodeInstance:

```bash
python track_a_evaluate.py \
  --method reg_selector_v4 \
  --lambda 0.5 --mu 50 --episode-seed 20000 --ga-seed 3100 \
  --instance results/track-a-l05-m50-s20000.json \
  --checkpoint results/reg-selector-v4-seed0-100ep/best.pth \
  --audit-output results/track-a-reg-v4-s20000.json \
  --output results/track-a-v4.csv
```

### Frozen Track B transfer check

`track_b_evaluate.py` reconstructs the controller interface recorded by a
checkpoint and inserts the exact frozen REG-v5 assignment network. The primary
v5 architecture is `atomic_inbound_scheduler_v5`. It has one active temporal
mode and 42 versioned manager outputs: `AcceptStore`, `DeferUntilEvent`, and 40
manifest-indexed `RetrieveDeliver` jobs. Seven primitive outputs remain inert
base-class/checkpoint baggage; every primitive mask is false and the worker
replay buffer remains empty.

At an inbound epoch, `AcceptStore` is the sole admissible macro. It navigates to
the inbound block, picks it up once, invokes frozen REG once with the Track A
shared candidate mask, and immediately begins storage movement without an
artificial assignment WAIT. Outside inbound epochs, the learned scheduler
chooses among feasible block-specific retrievals and Defer. Forced singleton
decisions do not consume the epsilon clock. Exploration is cardinality-safe:
it first chooses the Retrieve or Defer intent with equal probability and only
then chooses uniformly among feasible retrieval blocks.

The corresponding shared targets are

```text
V(s) = max_{c in C_macro(s)} Q_M(s,c)
y_M  = R^(k) + gamma^k (1-d) V^-(s')
```

Train the primary atomic scheduler around the frozen REG-v5 selector with:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u train_track_b.py \
  --selector-checkpoint results/reg-selector-v5-seed0-local-500ep/best.pth \
  --controller-variant atomic_inbound_scheduler \
  --lambda 0.5 --mu 50 --seed 0 \
  --episodes 500 --max-steps 4000 \
  --batch-size 128 --buffer-size 100000 --update-every 100 \
  --lr-manager 5e-5 --lr-worker 3e-5 \
  --gamma 0.99 --target-tau 0.001 --grad-clip 5.0 \
  --reward-clip 100 \
  --training-policy map --validation-policy map \
  --epsilon-start 0.90 --epsilon-end 0.05 \
  --epsilon-warmup-decisions 500 \
  --epsilon-decay-decisions 20000 \
  --validation-seeds 10000 10001 10002 10003 10004 \
  --validation-steps 4000 --eval-every 25 --log-every 5 \
  --device cuda \
  --output-dir results/track-b-atomic-inbound-reg-v5-seed0-500ep
```

The split-inbound v4 scheduler remains reproducible with
`--controller-variant mode_only_scheduler`; the earlier fully regularized
controller remains reproducible with
`--controller-variant full_regularized --training-policy regularized
--validation-policy regularized --tau-option 0.1 --tau-primitive 0.1`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python track_b_evaluate.py \
  --controller-checkpoint results/GATED_CONTROLLER.pth \
  --selector-checkpoint results/reg-selector-v5-seed0-500ep/best.pth \
  --lambda 0.5 --mu 50 --eval-seeds 100 101 102 \
  --max-steps 4000 --policy map \
  --save-instances-dir results/track-b-instances \
  --output results/track-b-v5-frozen.json
```

Use `--allow-legacy-controller` only for the earlier gated pilot checkpoints,
which predate controller-architecture metadata. Also provide their known
`--controller-training-lambda` and `--controller-training-mu`; the output flags
whether controller and selector training regimes match evaluation. A legacy or
regime-mismatched run is a smoke/transfer audit and must not be reported as the
final comparative result. The JSON records every selector decision, gate mode
counts, timing metrics, illegal drops, obstructions, and exact `instance_id`.

### Repaired urgency-first Track B candidate

Before another long controller-training run, evaluate whether the decision
structure itself is the bottleneck. `track_b_urgency_evaluate.py` freezes the
same REG-v5 assignment checkpoint and replaces learned macro dispatch with a
deterministic due-first rule. It uses `RetrieveDeliver` v2 / retrieval executor
v2, which admits a canonical plan only when its first complete live leg is
currently executable. Canonical complete-plan ETA—not a live-adjusted ETA—is
used for slack ranking. The strict protocol aborts on an unrecoverable macro or
a blocked due job instead of silently retrying through WAIT.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  track_b_urgency_evaluate.py \
  --selector-checkpoint results/reg-selector-v5-seed0-local-500ep/best.pth \
  --lambda 0.5 --mu 50 \
  --eval-seeds 30000 30001 30002 \
  --max-steps 4000 --max-defer-steps 10 \
  --device cuda \
  --output results/track-b-urgency-first-3seed.json
```

This is a repaired learning-augmented scheduling candidate, not a pure
scheduler ablation against an existing learned v5 checkpoint: those checkpoints
retain retrieval option/executor v1 for reproducibility. The result records the
selector digest, v2 retrieval ABI, neutral relocation version, decision trace,
strict failures, and atomic-option audit.

### Duration-aware Track B candidate

`reg_v5_duration_aware_atomic` is a separately versioned extension of the
repaired urgency scheduler. At an inbound epoch with no already-due executable
job, it considers only the first retrieval in the established ordering. Let
`s0` be that job's current complete-plan slack. A noncommitting frozen REG-v5
preview supplies an estimated full AcceptStore duration and a projected
post-accept yard; replanning the same job produces projected slack `s1`. The
predicted signed delivery errors are

```text
e_now    = -s0
e_accept = -s1
```

and the scheduler retrieves when

```text
abs(e_now) <= abs(e_accept) + lookahead_margin_steps.
```

The current-head restriction makes this a transparent one-step rule, not a
full schedule optimizer. Due-job priority, capacity-release retrieval,
event-driven deferral, strict retrieval initiation, and explicit failure are
unchanged from urgency-first v1.

The margin candidates were predeclared as `{0, 2, 4}` and evaluated on the
fixed calibration seeds `10000` through `10004`. Selection was lexicographic:
higher strict success, then higher mean return, then lower mean absolute error,
then the smaller margin. The calibration results were:

| Margin | Strict success | Mean return | Return SD | Mean steps | Mean absolute error |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.000 | 1481.718 | 40.479 | 1079.8 | 3.705 |
| **2** | **1.000** | **1486.278** | **13.157** | **1095.8** | **3.585** |
| 4 | 1.000 | 1457.466 | 21.938 | 1112.6 | 4.040 |

This protocol therefore selects `lookahead_margin_steps=2`. For that candidate,
all five episodes delivered 40 blocks, the within-window rate was 1.0, and no
invalid assignment, infeasible epoch, or method failure occurred. Its mean
signed deviation was `-1.265`, so it had a small early bias rather than zero
bias.

The preview limitation is material and must accompany the result. It does not
reserve its proposed cell: AcceptStore-v1 runs REG again after pickup. Only
48/186 executed preview-accept decisions chose the same cell (`25.81%`). The
nominal duration overestimated actual duration by `1.882` steps on average
(actual minus estimate ranged from `-6` to `0`). The audit also records
arrival-to-completed-storage flow time because a block's delivery clock begins
only when it is stored; for margin 2 its calibration mean, p90, and maximum
were `492.4`, `898`, and `1006` steps. These are total flow times, not pure
queue waiting. The shared metric contract below prevents improved delivery
timing from hiding increased inbound delay.

The nominal placement and projected retrieval paths use the stored-yard
canonical planner. Actual execution still checks the complete live route and
strict live first leg. Consequently, the projected accept-first cost can be
optimistic if other live inventory changes route feasibility; this is a
documented approximation, not a guarantee of feasibility in a modified yard.

Run the selected configuration once on the fresh, previously uninspected
holdout seeds `42000` through `42009`:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u track_b_urgency_evaluate.py \
  --selector-checkpoint results/reg-selector-v5-seed0-local-500ep/best.pth \
  --lambda 0.5 --mu 50 \
  --eval-seeds 42000 42001 42002 42003 42004 \
    42005 42006 42007 42008 42009 \
  --max-steps 4000 --max-defer-steps 10 \
  --scheduler-variant duration_aware \
  --lookahead-margin-steps 2 \
  --device cuda \
  --output results/track-b-duration-aware-holdout-m2-10seed.json
```

Do not retune the margin on these seeds. Use a new seed namespace if the
holdout is inspected before another design change.

### Duration-aware assignment-source ablation

`compare_track_b_assignment_sources.py` holds the complete duration-aware
scheduler and executor fixed and changes only the storage-assignment source.
The three variants use REG-v5, nearest-free, or dynamic PSLAP for both the
current-epoch noncommitting preview and the post-pickup assignment. Every
source receives the same physical candidate mask. A missing, malformed, or
out-of-mask proposal is an explicit method failure; this Track-B experiment
never applies Track A's operational nearest-free continuation fallback.

The runner samples or loads each immutable `EpisodeInstance` once and passes
the same object to all three variants. It also hashes the fixed scheduler
contract and aborts if the architecture, margin, macro interface, strict
retrieval executor, relocation rule, defer behavior, or timing objective
changes across sources. Margin `m=2` is fixed rather than retuned per source.

The `42000` holdout informed the duration-aware design, and seeds `43000` and
`43001` were used for implementation smoke testing. Use a fresh namespace for
the confirmation experiment:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u compare_track_b_assignment_sources.py \
  --selector-checkpoint results/reg-selector-v5-seed0-local-500ep/best.pth \
  --assignment-sources reg_selector_v5 nearest_free dynamic_pslap \
  --reference-assignment-source reg_selector_v5 \
  --seeds 44000 44001 44002 44003 44004 \
    44005 44006 44007 44008 44009 \
  --lambda 0.5 --mu 50 \
  --max-steps 4000 --max-defer-steps 10 \
  --lookahead-margin-steps 2 --target-window 20 \
  --bootstrap-samples 10000 --device cuda \
  --output-dir results/track-b-duration-aware-assignment-source-10seed
```

Outputs are `assignment-source-results.csv`,
`assignment-source-summary.json`, `assignment-source-audit.json`, and the
exact replayable instances. Paired deltas are joined by `instance_id` and are
oriented so positive values favor REG-v5.

This estimates each source's total coupled contribution to preview and final
placement under the fixed scheduler. It does not separately identify preview
accuracy from post-placement layout quality. If all variants still have zero
relocations and obstructions, the result cannot support a congestion or layout
robustness claim; that requires the subsequent narrow-gate/contention test.

### Paired egress-constrained assignment experiment

`compare_track_b_contention.py` runs the required paired `2 x 3` design:
ordinary four-cell egress versus a right-aligned two-cell egress, crossed with
REG-v5, nearest-free, and dynamic PSLAP assignment. This is an
inventory-obstruction experiment in a single-agent open yard; it is not a
multi-vehicle traffic-contention experiment. REG-v5 remains frozen from the
ordinary geometry, so the constrained condition also measures zero-shot
geometry generalization.

For a seed, both geometries receive identical arrivals and storage durations.
Within one geometry, methods pair by the geometry-dependent `instance_id`.
Across geometries, the difference-in-differences pairs by `schedule_id`, which
deliberately excludes geometry. Positive interactions mean the constrained
geometry increased REG-v5's advantage over the comparator. The primary
comparison is dynamic PSLAP; nearest-free is secondary.

Seeds `45000` through `45004` were used only for the predeclared activation and
integrity screen. Width 2 passed without using return to select the geometry:
all 30 cells completed strictly, and constrained dynamic PSLAP incurred five
audited relocations in three of five episodes, versus zero in the ordinary
condition. The frozen width and untouched confirmation namespace are therefore
`2` and `46000` through `46009`:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u compare_track_b_contention.py \
  --selector-checkpoint results/reg-selector-v5-seed0-local-500ep/best.pth \
  --assignment-sources reg_selector_v5 nearest_free dynamic_pslap \
  --reference-assignment-source reg_selector_v5 \
  --primary-comparison-source dynamic_pslap \
  --seeds 46000 46001 46002 46003 46004 \
    46005 46006 46007 46008 46009 \
  --lambda 0.5 --mu 50 \
  --max-steps 4000 --max-defer-steps 10 \
  --lookahead-margin-steps 2 --target-window 20 \
  --constrained-exit-width 2 \
  --bootstrap-samples 10000 --device cuda \
  --output-dir results/track-b-egress-width2-confirmation-10seed
```

The runner writes per-episode CSV, full JSON audit, exact instances for both
geometries, within-geometry paired source contrasts, cross-geometry paired
interactions, and an explicit relocation/integrity manipulation check. As in
the ordinary source ablation, a source supplies both preview and commitment;
the result is therefore a source-scheduler interaction, not a pure intervention
on final layout alone.

### Compact contention and capacity-generalization stress panel

`compare_track_b_generalization.py --suite stress_v1` evaluates complete
systems in four compact 5x5 open-yard profiles. There are eight admissible
storage cells. This is a challenge panel, not a one-factor causal ablation;
the paired ordinary-versus-constrained protocol above remains the estimator of
the egress-width effect.

| Profile | lambda | mu | Blocks | Exit width |
|---|---:|---:|---:|---:|
| `load_5x5_b24` | 0.8 | 100 | 24 | 1 |
| `count_5x5_b52` | 0.1 | 100 | 52 | 1 |
| `egress_5x5_b40` | 0.4 | 100 | 40 | 1 |
| `compound_5x5_b64` | 0.8 | 100 | 64 | 1 |

The profiles were accepted on seeds 68000--68002 without consulting return or
timing quality. Under the fixed dynamic-PSLAP reference, every episode had to
complete strictly, reach at least 50% peak physical occupancy, expose no more
than three feasible placement cells at some assignment epoch, and execute at
least one relocation. All four profiles passed. A 4x4 candidate reached full
occupancy but deadlocked; several 6x6/7x7 candidates produced strict
infeasibility; and the earlier 9x9/10x10 candidates peaked at only 6--10%
occupancy. Those candidates were excluded before confirmatory comparison.

The evaluator records one pressure sample after every completed environment
transition. Physical occupancy counts a live stored block only while it is
actually in a storage cell; historical `storage_location` values and cumulative
placement counters are not occupancy. Outputs include active and physical peak
occupancy, minimum free cells, time at or above 80% occupancy, maximum waiting
and pickup queue sizes, and decision-epoch feasible-cell minima. The learned
controller's hierarchy-wide candidate count and the shared assignment-cell
count remain separate fields.

Because profiles contain different block counts, raw return is retained for
within-profile comparisons but is not the cross-profile primary metric. The
macro report uses return per manifest block together with completion rate. It
also reports return and steps per delivered block, which must never be read
without completion rate because an early failure can otherwise look
artificially efficient.

The completed three-seed diagnostic is reproducible with:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u compare_track_b_generalization.py \
  --checkpoint \
    results/fully-learned-mixed-v2-seed0-definitive-160ep/best.pth \
  --zero-shot-checkpoint \
    results/fully-learned-v4-1-seed0-from-v4-spatial-250ep/best.pth \
  --suite stress_v1 --stress-stage calibration \
  --max-steps 5000 --device cuda \
  --output-dir results/track-b-stress-v1-calibration-v3-seed0-3seed
```

The calibration revealed a genuine extrapolation failure. Dynamic PSLAP and
nearest-free completed all 12 episodes strictly. The mixed fully learned
controller completed only 4.97% of manifest deliveries on average across the
four profiles, authenticated v4.1 completed 1.87%, and duration-aware REG-v5
completed 19.50%; all three learned variants had zero strict success. The two
fully learned controllers commonly filled 87.5--100% of storage and then
reported `no executable strict macro`. REG-v5 usually failed with
`delivery_route_failed`. In contrast, dynamic PSLAP held peak occupancy at
62.5% while preserving exit access, and nearest-free held it at 25%.

This is evidence that the current learned spatial policies did not extrapolate
from their 9x9--11x11 training domain to compact capacity management. It is not
a confirmatory performance ranking and does not show that a learned hierarchy
cannot solve the task. The appropriate next experiment is to train the same
variable-cardinality architecture over a distribution that includes compact
contention, with an access-preservation safety mask or auxiliary feasibility
objective, then evaluate once on the untouched 69000--69009 namespace. Do not
open that holdout before the training recipe and checkpoint-selection rule are
frozen. The eventual command is identical except for:

```text
--stress-stage holdout
--output-dir results/track-b-stress-v1-holdout-10seed
```

### Relational residual Track B experiment

The relational experiment preserves both atomic v5 and the urgency-first
candidate. It uses the same frozen REG-v5 selector and audited macros as the
repaired candidate, but scores a variable feasible action set with a shared
Deep-Set/action encoder. A deterministic urgency score supplies the initial
policy; the learned residual estimates longer-term scheduling effects, and a
separate head predicts TD-error risk for conservative selection. Recent macro
history is included without exposing future arrivals.

Train it with its isolated entry point:

```bash
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u train_relational_track_b.py \
  --selector-checkpoint results/reg-selector-v5-seed0-local-500ep/best.pth \
  --lambda 0.5 --mu 50 --seed 0 \
  --episodes 500 --max-steps 4000 \
  --batch-size 128 --buffer-size 100000 --update-every 100 \
  --learning-rate 5e-5 --gamma 0.99 --target-tau 0.001 \
  --epsilon-start 0.90 --epsilon-end 0.05 \
  --epsilon-warmup-decisions 500 --epsilon-decay-decisions 20000 \
  --override-margin 1.0 --uncertainty-penalty 0.05 \
  --validation-seeds 10000 10001 10002 10003 10004 \
  --validation-steps 4000 --eval-every 25 --device cuda \
  --output-dir results/track-b-relational-reg-v5-seed0-500ep
```

Evaluate the saved checkpoint under either the learned residual policy or its
embedded baseline ablation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python \
  track_b_relational_evaluate.py \
  --checkpoint results/track-b-relational-reg-v5-seed0-500ep/best.pth \
  --lambda 0.5 --mu 50 --eval-seeds 30000 30001 30002 \
  --policy residual_map --max-steps 4000 --device cuda \
  --output results/track-b-relational-3seed.json
```

Use `--policy baseline` on the same checkpoint and instances to measure the
incremental effect of learned overrides. Use `--policy kind_safe_residual_map`
to permit learned reordering within a macro kind while retaining the urgency
baseline's choice between `AcceptStore`, retrieval, and Defer; cross-kind
proposals are recorded as `cross_kind_override_rejected`. Relational
checkpoints have their own schema and cannot be loaded by
`track_b_evaluate.py`; conversely, historical atomic-v5 checkpoints cannot be
loaded by the relational evaluator.

### Paired complete-system Track B comparison

Track B compares complete deployed policies, not storage-assignment calls in
isolation. `compare_track_b.py` gives every method the same immutable episode
schedule and yard geometry. The learned method executes its gated option and
primitive controller with frozen REG-v5. Each assignment baseline executes its
assignment through the neutral deterministic pickup, routing, retrieval, and
delivery stack. Thus the comparison measures end-to-end return while preserving
the exact EpisodeInstance contract.

The online primary table contains the learned controller, repaired urgency-first
candidate, duration-aware candidate when explicitly requested, nearest-free,
dynamic PSLAP, and rolling-horizon GA. The full-schedule GA has future
information and is therefore labeled `offline_reference`; it must not be pooled
into the online ranking. Invalid proposals are never silently credited. The
shared operational fallback may finish a baseline episode, but the row records
both the invalid proposal and `fallback_contaminated=1`, while
`strict_method_success` fails.

After Track B training completes, run the selected validation checkpoint on an
untouched test seed set:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python compare_track_b.py \
  --checkpoint results/track-b-atomic-inbound-reg-v5-seed0-500ep/best.pth \
  --methods gated_regularized_reg_v5_frozen \
    reg_v5_urgency_first_atomic reg_v5_duration_aware_atomic \
    nearest_free dynamic_pslap \
    pslap_ga_2009_rolling pslap_ga_2009_offline \
  --seeds 42000 42001 42002 42003 42004 \
    42005 42006 42007 42008 42009 \
  --lambda 0.5 --mu 50 --max-steps 4000 \
  --policy map \
  --lookahead-margin-steps 2 \
  --rolling-population 16 --rolling-generations 10 \
  --offline-population 30 --offline-generations 30 \
  --device cuda \
  --output-dir results/track-b-comparison-holdout-m2
```

The output directory contains the replayable instances,
`track-b-results.csv`, the full `track-b-audit.json`, and
`track-b-summary.json`. Paired differences are joined by `instance_id`, use a
deterministic paired bootstrap interval, and are oriented so positive values
always favor the learned method: learned-minus-baseline for return and success,
and baseline-minus-learned for steps, absolute error, tardiness, and storage
flow time. Delivery and storage-flow metrics have their own paired sample
counts. Storage completion-rate advantages use reference-minus-comparison;
unfinished-count and flow-time reductions use comparison-minus-reference. The
summary also emits
`repaired_candidate_paired_comparisons`, anchored on urgency-first and oriented
so positive values favor that candidate against each deterministic baseline.
When requested, `duration_aware_candidate_paired_comparisons` analogously
anchors the selected duration-aware scheduler. The protocol records its margin,
noncommitting preview contract, and full per-episode preview audit.

Fixed model construction is excluded for both neural deployments; episode
runtime includes online planning, inference, and physical execution.

### Shared storage-flow contract

Every maintained Track A and Track B operational episode defines block `i`'s
storage flow time as

```text
F_i = S_i - A_i,
```

where `A_i` is the immutable instance arrival step and `S_i` is the first
successful storage PUTDOWN step. It includes queueing, scheduling, both travel
legs, pickup, and putdown. It must not be labeled queue wait and is distinct
from the required storage duration, delivery deviation, and episode makespan.
Outputs identify this definition as
`arrival_to_first_completed_storage_right_censor_v1`.

At an incomplete observation ending at step `T`, arrived unstored blocks are
right-censored at age `T-A_i`. Not-yet-arrived manifest blocks are counted
separately and receive no censor age. Neither value is imputed as a completed
flow time. Consequently, `mean_storage_flow_time`,
`median_storage_flow_time`, `p90_storage_flow_time`,
`p95_storage_flow_time`, and `max_storage_flow_time` are finite only when the
entire manifest has completed storage. Otherwise they serialize as `null`, but
the completed raw times, right-censor ages, and full manifest-aligned records
remain in the audit.

Per-episode output includes manifest, arrived, completed, right-censored,
not-yet-arrived, and unfinished counts; manifest and arrived completion rates;
the observation-end step; and the fully-observed flag. Method summaries pool
all block-level flow times for mean/median/p90/p95/max only when every included
episode is fully observed. They always aggregate completion and censor counts
and report the number and rate of fully observed episodes.

Paired output follows the global positive-favors-reference convention:

- `storage_flow_completion_advantage` is reference completion rate minus
  comparison completion rate;
- `unfinished_storage_flow_reduction` is comparison unfinished count minus
  reference unfinished count; and
- every mean/median/p90/p95/max `storage_flow_time_reduction` is comparison
  time minus reference time.

Completion and unfinished comparisons use every matched `instance_id`.
Flow-time reductions admit an instance only when both methods stored the whole
manifest; each metric therefore records its own `n`, instance IDs, per-instance
differences, and episode-level paired-bootstrap interval. Unfinished blocks can
never make a truncated policy appear faster through survivor-only averaging.

## Output schemas

### Common-evaluation CSV

```text
method, information_regime, lambda, mu, train_seed, eval_seed, instance_id,
checkpoint_id,
return, delivery_error, success, steps, decision_seconds,
obstructive_moves, illegal_drops, assignment_fallbacks,
delivery_count, target_window, mean_signed_deviation, mean_absolute_error,
mean_tardiness, mean_earliness, within_target_window_rate,
tardy_delivery_rate, mean_tardiness_when_tardy, p90_tardiness,
p90_absolute_error, signed_deviation_std, absolute_error_std,
delivery_deviations
```

`delivery_error` is retained only as a backward-compatible alias for
`mean_signed_deviation`; it must not be described as timing accuracy. The raw
JSON array in `delivery_deviations` preserves block-level observations.
`instance_id` is the required key for paired comparisons. Existing CSVs without
that column use the old schema, so the evaluator requires a new output path.
Aggregate multiple evaluation rows, including across-episode variability, with:

```bash
python summarize_evaluations.py \
  --input results/evaluation.csv \
  --output results/evaluation-summary.json
```

The Track B comparison CSV additionally includes:

```text
storage_flow_metric_contract, storage_flow_observation_end_step,
storage_flow_manifest_count, storage_flow_arrived_count,
storage_flow_completed_count, storage_flow_right_censored_count,
storage_flow_not_yet_arrived_count, storage_flow_unfinished_count,
storage_flow_completion_rate, storage_flow_arrived_completion_rate,
storage_flow_fully_observed, mean_storage_flow_time,
median_storage_flow_time, p90_storage_flow_time, p95_storage_flow_time,
max_storage_flow_time, completed_storage_flow_times,
right_censored_storage_flow_ages
```

The full JSON audit also retains `storage_flow_records` in manifest order. Each
record contains the block/index, arrival and completion steps, observed flow
time, status (`completed`, `right_censored`, or `not_yet_arrived`), and censor
step/age where applicable. Commands above require no additional flag; current
evaluators emit this contract automatically.

### Training NPZ

```text
episode_returns
episode_avg_error
episode_mean_signed_deviation
episode_mean_absolute_error
episode_mean_tardiness
episode_mean_earliness
episode_within_target_window_rate
episode_tardy_delivery_rate
episode_p90_tardiness
episode_obstructive_moves
episode_illegal_drops
episode_assignment_fallbacks
episode_success
manager_losses
worker_losses
```

`episode_avg_error` is also a legacy signed-deviation field. Negative values
mean early bias; they are not intrinsically better than positive values.

### Training JSON

The JSON log records the same logical arrays in a human-readable form.

### Checkpoint

Every current checkpoint contains:

```text
checkpoint_schema_version
manager_option_ids
primitive_action_ids
manager_state_dict
worker_state_dict
manager_opt_state
worker_opt_state
epsilon
step_count
```

Learned-selector checkpoints also contain keys prefixed with `selector_`.
That includes `selector_feature_version`, which prevents weights trained under
different feature semantics from being loaded silently, and
`selector_return_definition`, which distinguishes the complete environment
event return from historical checkpoints whose delayed-reward semantics cannot
be verified. `selector_gamma` records the event-return discount and must match
the agent discount. The current checkpoint schema version is `3`.

The learned-selector training entry point uses the complete environment return
by default. The algebraically equivalent split form is available for internal
audits with:

```bash
HRL_SELECTOR_RETURN_MODE=explicit_terminal \
  python example/run_small_rooms_example_hrl.py
```

## Existing evidence in this copy

- `artifacts/legacy/evaluation/common_eval_20260623-142857.csv`: 15 PSLaP rows across three diagonal
  settings and five evaluation seeds.
- `artifacts/legacy/evaluation/common_eval_20260623-143142.csv`: one HRL evaluation row.
- `artifacts/legacy/evaluation/evaluation_smoke.csv` and
  `artifacts/legacy/evaluation/evaluation_smoke_gpu.csv`: duplicate HRL smoke
  evaluations on two execution environments.
- `artifacts/legacy/evaluation/ga_smoke.csv`: one successful GA-assisted evaluation.
- `artifacts/legacy/evaluation/hrl_old_arch_smoke.csv`: one unsuccessful old-architecture HRL evaluation.
- `example/models/`: 85 historical checkpoints dated from July 2025 through
  February 2026.
- `runs/hrl/`: TensorBoard event files for several runs.
- root-level NPZ files: legacy reward arrays whose provenance is only partially
  encoded in their key names.

## Reproducibility cautions

Do not compare result files solely by filename. A defensible comparison should
record method, code revision, environment parameters, training seed, evaluation
seed, step limit, checkpoint identity, GA assignment identity, dependency
versions, selector return definition, and whether delivery error is treated as
signed or absolute.

The existing training scripts do not store all of that metadata in one manifest.
Adding an experiment manifest is part of the proposed reorganization.

The workspace's default `python3` interpreter currently cannot import PyTorch,
so none of the learned-policy entry points should be assumed runnable until a
project environment has been created or activated.
