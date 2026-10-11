# E19 — Service tails and operational-cost sensitivity

E19 is a post-hoc reanalysis of closed experiments. It launches no training
and no policy rollouts. Its primary question is whether the established mean
timing/handling results also have a useful service-tail and operating-cost
interpretation.

## Evidence and scope

- E1 supplies signed job-level delivery deviations, rehandles, and episode
  steps for all 1,860 saved rows. It is the primary tail and cost panel.
- E11 supplies episode-level earliness, tardiness, within-window rate, and
  achieved handling at the same four frozen preferences across seven regimes.
- E13 supplies the corresponding episode-level summaries across 15 completed
  scale/occupancy configurations.
- E4/E5(c) are authenticated as supporting parents, but their rows do not add
  job-level timing traces beyond the final E1 method points.

Dynamic PSLAP and Kim2020 failed E1's method-level strict-completion gate.
Their recorded delivered/unmet counts and failures remain visible, but their
success-looking job-tail and cost aggregates are suppressed.

## Service analysis

For every eligible method point, E19 reports early/on-time/late proportions,
within-20 performance, and type-7 p50/p90/p95/p99 quantiles for absolute error
and tardiness. Pooled jobs describe the service distribution; they are not
treated as independent training replications. Episode and model-seed
replication remains the inferential level.

## Cost sensitivity

The hypothetical loss per required delivery is

```text
mean earliness
+ tardiness weight * mean tardiness
+ rehandle equivalent * physical rehandles / required deliveries
```

Tardiness weights are `{1, 2, 4}` and one physical rehandle is assigned
`{0, 10, 25, 50}` equivalent timing steps. All ten predeclared frozen VCG
operating points are reported in every scenario; no new preference is selected
from the test data.

Primitive steps are reported separately and excluded from the combined loss
because they overlap with delivery timing and physical handling. Travel
distance, loading/unloading time, transporter occupancy, energy, and monetary
cost were not separately measured. E19 therefore supports a transparent
exchange-rate sensitivity analysis—not energy or financial savings claims.

## Run

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E19_service_tails_cost_sensitivity/run.sh run
```

The command should take only a short analysis pass. It writes an authenticated
JSON report, three Markdown tables, and PDF/PNG service-tail and cost figures
under `results/vcg-conditioned-e19-service-cost/`.

## Completed result

The authenticated analysis completed over 1,860 E1 rows, 28 E11
regime/preference coordinates, and 15 E13 scale coordinates, with zero new
training or rollout runs.

All ten VCG points passed the 90/90 strict gate. Among VCG points, lambda=0
had the strongest service tail (82.8% within 20 steps and absolute-error p95
of 37), while lambda=.20 reduced rehandles from 24.58 to 1.67 per 100 at 76.4%
within-window and p95 of 46. Historical VCG 2.3 had a p95 of 43 and an observed
worst delay of 128. GA had a short tardiness tail but a much heavier earliness
tail: its late-job p95 was 19, while absolute-error p95 was 50.

The cost surface therefore has no universal winner. With symmetric timing
cost and no handling charge, lambda=0 has the lowest descriptive loss. As the
handling exchange rate rises, middle/high VCG preferences become favorable.
When tardiness is weighted heavily, GA is favored over much of the tested
surface because it rarely delivers very late, despite its poorer absolute
error. These are point-estimate sensitivity findings, not a selected operating
policy or a monetary-benefit claim.

E11 further shows that the same lambda generally reduces handling as lambda
increases, but its achieved handling rate varies substantially by regime.
Thus lambda is a portable preference input, not a hard handling-budget value.
