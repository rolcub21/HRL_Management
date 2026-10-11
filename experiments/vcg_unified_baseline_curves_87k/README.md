# Unified VCG matched evaluation curves (87k)

This additive experiment compares the two frozen unified-VCG operating points
with four baselines on the exact same 30 serialized EpisodeInstances
(`87000..87029`). It performs no training and does not regenerate the panel.

Methods:

- VCG, lambda = 0 (existing 360 rows)
- VCG + handling cost, lambda = 0.05 (existing 360 rows)
- VCG 1.1 legacy comparator (90 new rows)
- Duration-aware Dynamic PSLAP (30 new rows)
- Capacity-aware rolling GA (120 new rows)
- Kim2020-inspired spatial A3C (450 new rows)

Run sequentially:

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/vcg_unified_baseline_curves_87k/run.sh run-all
```

Or run/resume the phases independently:

```bash
bash experiments/vcg_unified_baseline_curves_87k/run.sh prepare
bash experiments/vcg_unified_baseline_curves_87k/run.sh run-v11
bash experiments/vcg_unified_baseline_curves_87k/run.sh run-baselines
bash experiments/vcg_unified_baseline_curves_87k/run.sh run-kim
bash experiments/vcg_unified_baseline_curves_87k/run.sh plot
```

VCG 1.1 requires CUDA. The other three new baseline evaluations run on CPU.
Every phase is ledger-resumable. A failed/unsafe method is never averaged over
only its successful cases: its numerical curve stops at the first failed
EpisodeInstance and is marked with an X.

The figure has three cumulative-evaluation panels, in the frozen manifest
order: dense return, MAE, and physical rehandles per 100 required deliveries.
Shading is descriptive min-max variability across model seeds (or GA optimizer
realizations), not a confidence interval. These are frozen-policy evaluation
curves, not training learning curves.
