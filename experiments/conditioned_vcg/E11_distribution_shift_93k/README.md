# E11 — Frozen-policy distribution-shift generalization

E11 tests whether the final frozen VCG family preserves completion and useful
timing--handling operating points when the 5×5, eight-job deployment
distribution changes. It performs no training, checkpoint selection,
normalization update, verifier change, or regime-specific preference tuning.

The predeclared policies are frozen Qop (`lambda=0`), conditioned VCG at
`lambda in {.05,.10,.20}`, and heuristic-safe. Learned policies use all three
final model seeds; every policy is paired on the exact same serialized
EpisodeInstance within a regime.

## Regimes

| Regime | Changed factor | Fixed factors / qualification |
|---|---|---|
| `reference` | none | 5×5, 8 jobs, arrival rate 10, Poisson dwell mean 80 |
| `arrival_spread` | mean interarrival `.1 -> 10` | reference durations and geometry |
| `dwell_short` | Poisson dwell mean `80 -> 40` | reference arrival schedule and geometry |
| `dwell_long` | Poisson dwell mean `80 -> 120` | reference arrival schedule and geometry |
| `dwell_bimodal` | four Poisson-40 plus four Poisson-120 jobs | nominal mean 80; reference arrivals and geometry |
| `mirrored_entry` | robot start and entrance/pickup reflected horizontally | dimensions, exits, topology, and 8-cell capacity fixed; storage set is reflected |
| `combined_shift` | spread arrivals + bimodal dwell + mirrored entrance | fixed 5×5 dimensions and 8 jobs |

The simulator has no job-class field. E11 therefore uses a declared bimodal
dwell mixture as a workload-composition shift and does not invent class labels.
The reference is already strongly arrival-compressed, so the one-factor
arrival test is a spread-out deployment rather than a nominally redundant
"more compressed" level.

Difficulty is recorded rather than assumed equal: arrival span, simultaneous
arrival peak, nominal peak concurrency, duration mean/CV, actual peak active
jobs, peak storage occupancy, and peak arrived-but-unstored jobs. Unfinished
runs retain failure/admission counts; timing metrics are suppressed for an
entire coordinate if any required row is incomplete.

## Run order

Prepare and run the 60-row seed-0 mechanism pilot first:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E11_distribution_shift_93k/run.sh run-pilot
```

The full experiment contains 2,730 resumable rows. It can be launched one
regime at a time:

```bash
bash experiments/conditioned_vcg/E11_distribution_shift_93k/run.sh run-regime reference
bash experiments/conditioned_vcg/E11_distribution_shift_93k/run.sh run-regime arrival_spread
```

or as one resumable command:

```bash
bash experiments/conditioned_vcg/E11_distribution_shift_93k/run.sh run-all
```

After modular runs complete:

```bash
bash experiments/conditioned_vcg/E11_distribution_shift_93k/run.sh analyze
```

Render the two paper-facing figures and paired uncertainty table after the full
report is complete:

```bash
bash experiments/conditioned_vcg/E11_distribution_shift_93k/run.sh figures
```

The operating-point figure uses shared axes across all seven regimes. Bold
paths are the three-model-seed means; light paths are the separate model-seed
means. Lines connect the four prespecified deployment preferences and are not
temporal trajectories. The second figure reports shift-minus-reference paired
effects with pointwise hierarchical 95% bootstrap intervals over model seeds
and matched instances. With only three model seeds, these intervals are
descriptive rather than high-powered population-level uncertainty estimates.

Outputs are written to
`results/vcg-conditioned-e11-distribution-shift-93k/`.
