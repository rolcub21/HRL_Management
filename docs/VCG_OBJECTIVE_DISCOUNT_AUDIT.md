# VCG objective--discount development audit

## Scope

This is a development-only mechanism experiment.  It does not modify or
replace the canonical VCG-v1 replications and it must not open either sealed
test panel.  Its purpose is to separate two explanations for the early
delivery behavior:

1. the zero-gradient tail of the legacy timing bonus; and
2. the mismatch between discounted training and undiscounted evaluation.

The graph network, exact-SAFE candidate interface, liveness guard, replay
scheme, optimizer, exploration schedule, environment instances, and model
initialization are held fixed across arms.

## Factorial arms

| Arm | Delivery objective | Gamma |
|---|---|---:|
| `legacy_g099` | legacy clipped bonus | 0.99 |
| `legacy_g100` | legacy clipped bonus | 1.00 |
| `dense_g099` | dense piecewise timing cost | 0.99 |
| `dense_g100` | dense piecewise timing cost | 1.00 |

The legacy delivery reward is

\[
r_{\mathrm{legacy}}(e)
=10+30\max\left(0,1-\frac{|e|}{20}\right).
\]

The dense reward is

\[
r_{\mathrm{dense}}(e)
=B-\lambda_1|e|-\lambda_2\max(0,|e|-20),
\]

with development-screen defaults

\[
B=40,\qquad \lambda_1=1.5,\qquad \lambda_2=0.5.
\]

These defaults make the two rewards exactly equal for every
\( |e|\leq20 \).  Outside the window, the legacy derivative is zero and the
dense derivative with respect to absolute error is `-2.0`.  The audit
therefore changes only the legacy dead zone.  Coefficients remain explicit in
every command and checkpoint; they are not hidden environment constants.

## Development protocol

- One paired model seed (`0`) for the mechanism screen.
- 200 training episodes per arm.
- Identical training EpisodeInstance seeds beginning at `40000000`.
- Validation every 25 episodes on development seeds `76000--76009`.
- A two-episode-per-arm smoke test precedes the screen.
- Epsilon decays from 0.90 to 0.05 over 3,000 macro decisions, leaving a
  material low-epsilon consolidation interval inside the 200-episode screen.
- The canonical VCG-v1 architecture and exact verifier remain authoritative.
- No baseline query, teacher, critic pruning, fallback, or future-arrival
  schedule is permitted.

Two hundred episodes are not a performance confirmation.  They are enough to
decide whether objective/discount changes visibly alter the delivery-position
failure.  Only a prespecified winning configuration should receive a full
three-training-seed confirmation.

## Measurements

Every trajectory is rescored under both reward definitions, independently of
the reward used for learning.  Native return must not be compared directly
between unlike reward definitions.

Primary common measurements are:

- strict completion and method failures;
- signed and absolute error by delivery position, especially positions 1--2;
- total MAE, tardiness, and within-window rate;
- primitive steps;
- defer, delivery, acceptance, and reconfiguration counts;
- guard activation and forced-decision rates;
- relocations per 100 deliveries.

The audit additionally records:

- undiscounted legacy-rescored return;
- undiscounted dense-rescored return;
- discounted return under the arm's gamma;
- delivery, storage, and primitive-step reward components.

The safety gate is unchanged: an arm with less than 100% strict completion on
the development validation panel is not a deployment candidate regardless of
its timing result.

## Gamma-one interpretation

Gamma one is valid here only as a finite-horizon episodic audit with an
enforced terminal/truncation boundary.  It is not covered by the existing
strict-contraction proposition.  If the gamma-one arm is retained for the
final method, the controller observation must explicitly include time or
remaining horizon and the manuscript must use a finite-horizon, time-indexed
Bellman argument rather than claiming a gamma contraction.

The mechanism screen deliberately leaves the VCG-v1 observation unchanged in
all four arms so that reward and gamma are the only factors.  The episode step
limit must be reported, and whether it binds must be audited.

The shortened exploration schedule is shared by all four arms and belongs to
the audit protocol.  Consequently, `legacy_g099` is the internal factorial
control; it is not presented as another canonical VCG-v1 replication.

## Fresh paired complete-system development gate

After selecting `dense_g099` in the mechanism screen, its episode-100
checkpoint was evaluated on fresh development seeds `78000--78019`.  These
instances were not used for training or checkpoint selection.  Both sealed
panels (`69000--69009` and `77000--77049`) remained unopened.

Every method consumed the same saved `EpisodeInstance`.  VCG used its learned
Accept/Recover/Defer controller over the complete exact-SAFE frontier.  The
baselines used the common duration-aware reserved-cell scheduler and differed
only in assignment source.  Rollouts used the legacy environment reward, and
every fixed trajectory was exactly rescored under the selected dense
objective by replacing only its realized delivery terms.

| Complete system | Strict | Legacy return | Dense return | MAE | First-2 MAE | Within window | Relocations/100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense VCG, gamma 0.99 | 1.00 | 226.37 | 192.87 | 12.06 | 16.55 | 0.850 | 18.13 |
| Duration-aware + nearest-free | 1.00 | 162.57 | -352.73 | 47.38 | 63.75 | 0.250 | 0.00 |
| Duration-aware + dynamic PSLAP | 1.00 | 215.22 | 19.22 | 23.05 | 49.98 | 0.519 | 4.38 |
| Duration-aware + rolling GA | 0.90 | 223.54 | 128.34 | 14.81* | 33.33* | 0.681* | 10.42 |
| Duration-aware + enhanced rolling GA | 0.90 | 205.07 | 114.27 | 16.18* | 34.33* | 0.583* | 13.89 |

`*` Timing summaries for the GA variants condition on their 18 completed
episodes.  Each GA variant failed two episodes under the strict no-fallback
protocol; VCG, nearest-free, and dynamic PSLAP completed all 20.

Paired against dynamic PSLAP, dense VCG achieved:

- MAE reduction `10.99`, bootstrap 95% interval `[7.82, 14.13]`;
- first-two MAE reduction `33.43`, interval `[27.20, 39.25]`;
- within-window advantage `0.331`, interval `[0.238, 0.419]`;
- dense-return advantage `173.65`, interval `[126.81, 219.98]`; and
- 12.25 fewer primitive steps, interval `[2.85, 20.10]`.

The trade-offs were 2.62 greater mean tardiness and 13.75 more relocations per
100 deliveries than dynamic PSLAP.  Legacy-return advantage was `11.15`, but
its interval `[-6.37, 30.45]` crossed zero.

The three completed canonical VCG replicas cannot be pooled with dense VCG:
they used the legacy objective, 500 episodes, a 10,000-decision epsilon decay,
and validation seeds `75000--75019`.  On the same fresh development panel,
their MAEs were `22.51`, `31.85`, and `26.51`, and their first-two MAEs were
`60.55`, `51.78`, and `55.93`.  This confirms that the old first-delivery
sacrifice was reproducible across model seeds.  Dense seed 0 obtained MAE
`12.06` and first-two MAE `16.55` on those instances.

This gate does not justify a sealed-test performance claim because dense VCG
still has one model seed and came from a development mechanism screen.  It
does remove the immediate case for an architectural V2: the unchanged VCG-v1
network became competitive or better on timing accuracy after the objective
repair.  The next step is a dedicated dense/gamma-0.99 three-seed training
protocol.  Architectural V2 work is conditional on replication failure or on
a separately specified requirement to reduce tardiness, relocation, or exact
frontier computation.

## Frozen proper-training confirmation

`train_vcg_dense_proper.py` promotes the selected `dense_g099` condition into
a single-condition confirmation protocol.  The public CLI exposes no reward,
gamma, architecture, environment, exploration, validation-panel, or episode-
budget switches.  The frozen contract is:

- dense reward `B=40`, `lambda_abs=1.5`, `lambda_outside=0.5`, window `20`;
- SMDP gamma `0.99` and the selected 3,000-decision epsilon decay;
- 500 episodes per model seed;
- seed-specific training namespaces `50000000`, `51000000`, and `52000000`;
- common checkpoint-selection instances `79000--79019`, every 25 episodes;
- exact-SAFE verification remains authoritative, with no critic pruning,
  teacher, baseline query, fallback, or future-arrival information.

Intermediate validation winners are authenticated in alternating transaction
slots (`best-candidate-slot-a.pth` and `-slot-b.pth`); the slot referenced by
the committed `latest.pth` is never overwritten.  Candidates are explicitly
ineligible for deployment.  The trainer creates
`best.pth` only after episode 500, when checkpoint selection over the fixed
budget has been finalized.  A stopped run resumes exactly from `latest.pth`
with its optimizer, replay, local RNG, global RNG, and liveness-guard state.

Seed 1:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=1 \
PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u train_vcg_dense_proper.py \
  --output-dir results/vcg-dense-v1-1-seed1-500ep \
  --model-seed 1 \
  --device cuda
```

Seed 2 uses the same command with `PYTHONHASHSEED=2`, `--model-seed 2`, and
output directory `results/vcg-dense-v1-1-seed2-500ep`.  Add
`--resume-existing` without changing any other argument after an intentional
or incidental stop.  Seed 0 from the 200-episode mechanism screen is not a
homogeneous replicate; it must eventually be retrained with model seed 0 and
this same 500-episode protocol before a three-seed claim.

Artifacts:

- `results/vcg-dense-g099-baselines-development-20seed/comparison-summary.json`
- `results/vcg-legacy-seed0-dynamic-development-20seed/comparison-summary.json`
- `results/vcg-legacy-seed1-dynamic-development-20seed/comparison-summary.json`
- `results/vcg-legacy-seed2-dynamic-development-20seed/comparison-summary.json`
