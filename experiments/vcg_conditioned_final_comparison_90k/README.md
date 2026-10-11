# Prospective conditioned-VCG final comparison on 90k (CPU v3)

Canonical paper entry point: `experiments/conditioned_vcg/E01_benchmark_90k/`.
This directory remains in place because the completed protocol authenticates
its original source paths.

This experiment opens a new panel of 30 serialized `EpisodeInstance` objects
only after freezing all policies, lambda coordinates, nuisance replications,
and aggregation rules.

The original v1 execution contract was aborted before its first policy row
because CUDA was unavailable. CPU v2 was also aborted with zero policy rows:
its contract incorrectly treated host-visible runtime metadata (such as CUDA
visibility) as semantic identity, so the same CPU command could not
authenticate across shell contexts. CPU v3 predeclares CPU execution for all
methods, binds software versions and sources, and records host metadata only as
descriptive activation provenance. The policies, lambda grid, instance seeds,
and analysis were unchanged; no outcome was observed before either correction.

Compared operating points and methods:

- conditioned VCG at `lambda = 0, .025, .0375, .05, .075, .1, .125, .15, .175, .2`;
- historical VCG 2.3;
- duration-aware Dynamic PSLAP;
- capacity-aware rolling GA;
- Kim2020 adaptation.

The conditioned controller uses its three authenticated converged terminals.
`lambda=0` delegates exactly to VCG 1.1, so VCG 1.1 is not redundantly rerun as
a separate method. Historical VCG 2.3 retains its authenticated 622m
action-RNG grid by panel position; this reuses stochastic streams, not the
86k EpisodeInstances or outcomes.

The evaluation contains 1,860 ledgered rows: 900 conditioned VCG, 360 VCG 2.3,
30 Dynamic PSLAP, 120 GA, and 450 Kim2020. Every command resumes by loading
completed ledgers. A failed cell is retained and suppresses that complete
method/lambda aggregate; complete-case averaging is forbidden.

## Completed result

CPU v3 completed all 1,860 rows. Every conditioned operating point was
strict-safe-complete (`900/900`). Representative aggregates are:

| Method | Dense return | MAE | Steps | Rehandles/100 | Safe rows |
|---|---:|---:|---:|---:|---:|
| Conditioned VCG, `lambda=0` | **185.36** | **12.38** | 188.49 | 24.58 | 90/90 |
| Conditioned VCG, `lambda=.05` | 168.02 | 13.72 | 162.72 | 8.47 | 90/90 |
| Conditioned VCG, `lambda=.10` | 161.34 | 14.17 | 153.24 | 3.06 | 90/90 |
| Conditioned VCG, `lambda=.20` | 157.10 | 14.48 | **152.09** | **1.67** | 90/90 |
| Historical VCG 2.3 | 108.92 | 17.80 | 230.96 | 6.25 | 360/360 |
| Capacity-aware GA | 141.73 | 14.91 | 167.77 | 13.75 | 120/120 |
| Dynamic PSLAP | -- | -- | -- | -- | 28/30 |
| Kim2020 adaptation | -- | -- | -- | -- | 448/450 |

Nine of ten conditioned points are on the point-estimate MAE--rehandles
frontier; `.075` is dominated by `.10`. Relative to `lambda=0`, `lambda=.20`
reduces rehandles by 22.92/100 while increasing MAE by 2.10. Relative to
historical VCG 2.3, `lambda=.20` improves return, MAE, steps, and rehandles,
with all four paired nominal 95% intervals excluding zero.

Dynamic PSLAP did not complete instances 90010 and 90012. Kim2020 blocked on
two model-seed-0 stochastic rolls (instances 90004 and 90023); their
whole-method aggregates are correctly suppressed.

The authenticated CPU-v3 identities are:

```text
contract  6f70a33f8f5d15a776c661edca8708c94ab46afa2f24cd325d81beed74203ca0
manifest  26a964a4b62cfb8b724857c5c3baf08560b072fd969bcfef03fa731b444f6cb1
report    75b7bf1b294486690f19e123ddb30b2db6061c7a60b888c21c208dbbd20f629f
```

Run everything sequentially:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/vcg_conditioned_final_comparison_90k/run.sh run-all
```

For visibility or separate terminals, run:

```bash
bash experiments/vcg_conditioned_final_comparison_90k/run.sh prepare
bash experiments/vcg_conditioned_final_comparison_90k/run.sh open-panel
bash experiments/vcg_conditioned_final_comparison_90k/run.sh run-vcg-seed0
bash experiments/vcg_conditioned_final_comparison_90k/run.sh run-vcg-seed1
bash experiments/vcg_conditioned_final_comparison_90k/run.sh run-vcg-seed2
bash experiments/vcg_conditioned_final_comparison_90k/run.sh run-v23
bash experiments/vcg_conditioned_final_comparison_90k/run.sh run-baselines
bash experiments/vcg_conditioned_final_comparison_90k/run.sh run-kim
bash experiments/vcg_conditioned_final_comparison_90k/run.sh analyze
bash experiments/vcg_conditioned_final_comparison_90k/run.sh plot
```

The reference `run-all` path is sequential. The three VCG seed commands are
individually resumable, but concurrent CPU runs can contend for memory and do
not reduce the predeclared row count. Baseline commands may be run separately
after the panel has been opened.
