# Robust recovery on authenticated 5x5 snapshots

This is a deterministic, no-training bridge from the frozen 89k VCG result to
the finite robust-recovery construction. It compares nominal, one-step robust,
and recursively robust completion certificates on 30 authenticated 5x5
decision-boundary snapshots.

The source is the passed experiment in
`results/vcg-v1-1-nested-lambda-frontier-confirmation-89k`. Preparation checks
the source contract, source hashes, activation, instance manifest, checkpoint,
cost head, per-instance ledgers, and final report. It then exactly replays the
frozen model-seed-0, lambda-zero trajectories on CPU and captures one
exact-SAFE snapshot per EpisodeInstance. Assigned target occupancies cycle
through 2, 4, 6, and 8 stored blocks, with assigned quotas 8, 8, 7, and 7. If
an episode never reaches its assigned target at an eligible boundary, the
predeclared fallback chooses the nearest available target, preferring the
higher occupancy on a tie. The report records the resulting strata and every
substitution. This selection uses source-state availability only, before any
robustness result is computed.

If an unaccepted inbound block is waiting at pickup, preparation records it
explicitly, closes admission, and retains the pickup/queue cells as fixed
physical obstacles; the inbound is not silently counted as completed recovery
work.

There is no policy training, retraining, checkpoint selection, lambda
selection, or learning in this experiment. The lambda values 0 and 0.2 only
select a root action from the same certified set using frozen historical
operational and handling-value inputs; they cannot change the certificate.

## Certification comparison

For every frozen snapshot, the three methods use the same legal strict
`Deliver` and `Reconfigure` recovery macros and the same finite completion
horizon:

- **Nominal:** recursively completes the admitted workload under the nominal
  macro outcome only.
- **One-step robust:** requires every declared realization of the first macro
  to be nominally completable afterward.
- **Recursive robust:** requires robust completion under the declared
  realization set at every subsequent macro.

The finite disturbance envelope is the product of:

- zero or one extra primitive duration step; and
- the nominal post-macro agent stop or a clear traversable, unoccupied,
  orthogonally adjacent stop, with one additional primitive step for the
  adjacent stop.

The completion horizon is fixed before certification from the source nominal
witness: its primitive steps plus three steps per witness macro. Search is
bounded to 1,000 expanded nodes per method and snapshot. A proven `WINNING`
result is admitted, a proven `LOSING` result is rejected, and compute-budget
exhaustion is `UNKNOWN` and therefore rejected fail-closed.

Certification is resumable at snapshot granularity. Each completed row is
atomically written under `snapshot-results/` and authenticated before reuse, so
rerunning `evaluate` or `run` continues with missing rows. Panel preparation is
idempotent, but a preparation interrupted before `snapshot-panel.json` is
committed replays the panel from the beginning.

## Run

From the repository root:

```bash
bash experiments/vcg_robust_recovery_snapshot_panel_5x5/run.sh prepare
bash experiments/vcg_robust_recovery_snapshot_panel_5x5/run.sh evaluate
```

Or run both stages sequentially:

```bash
bash experiments/vcg_robust_recovery_snapshot_panel_5x5/run.sh run
```

The output directory is
`results/vcg-robust-recovery-snapshot-panel-5x5/`. Its principal artifacts are:

- `bridge-contract.json`: authenticated protocol and source binding;
- `snapshot-panel.json`: the 30 frozen physical-recovery snapshots;
- `snapshot-results/instance-<seed>.json`: resumable certificates; and
- `bridge-report.json`: aggregate comparison and per-snapshot rows.

## Observed frozen-panel result

The authenticated run passed on all 30 source EpisodeInstances. The selected
occupancy counts were 2:8, 4:8, 6:9, and 8:5; two assigned 8-block snapshots
used the declared 6-block fallback. Twenty-five snapshots contained a current
unaccepted inbound, handled under the explicit closed-admission convention
above.

| Certificate | WINNING | UNKNOWN | LOSING | Mean certified root actions | Snapshots reaching the compute cap |
|---|---:|---:|---:|---:|---:|
| Nominal | 30 | 0 | 0 | 12.13 | 11 |
| One-step robust | 28 | 2 | 0 | 10.03 | 11 |
| Recursive robust | 23 | 7 | 0 | 3.80 | 24 |

A `WINNING` state has a concrete certified witness even if other root actions
reached the cap. In those capped rows, the reported certified action set is a
sound lower bound, not a claim that the complete frontier was enumerated.
`UNKNOWN` is rejected and is not interpreted as `LOSING`.

Recursive WINNING counts by occupancy were 8/8 at occupancy 2, 3/8 at
occupancy 4, 7/9 at occupancy 6, and 5/5 at occupancy 8. The non-monotonic
pattern reflects branching: these 8-block states had only three certified root
actions on average, whereas the 4-block states exposed a much larger search
frontier.

Changing the frozen candidate-level merit from lambda 0 to 0.2 changed the
nominal and one-step selected root action in 6/30 snapshots, but changed none
of the 23 recursively WINNING selections. Recursive certification reduced the
choice set enough that the robustness constraint dominated this local
preference diagnostic. This is not a full mode-aggregated policy rollout or a
timing/rehandling performance comparison.

## Claim boundary

The exact claim is finite, bounded-disturbance, closed-admission **physical
recovery** for actions and snapshots proved `WINNING` within the declared
primitive horizon. The state is a frozen repository `RecoveryState`: there is
no new admission and no future arrival schedule. Completion means removing
the already accepted/stored blocks through strict recovery macros. A current
unaccepted inbound may remain outside that workload at the fixed pickup
obstacle, so this is not a certificate for completing every block physically
present in the live episode.

The block transition and macro route are nominal. Duration uncertainty only
consumes the completion budget, and stopping uncertainty only changes the
post-macro agent position. This experiment does not perturb interior route
states, advance block timing inside the recovery abstraction, inject
disturbances through the authoritative dynamic environment, or evaluate full
episodes. It therefore does **not** establish full dynamic-episode robustness,
arrival robustness, continuous or unbounded execution robustness, or hardware
robustness.
