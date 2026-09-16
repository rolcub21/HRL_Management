# VCG-Dense V1.2 constrained checkpoint-selection diagnostic

This development-only experiment asks whether VCG V1.1 already learned a
better storage-layout policy than its dense-return checkpoint selector chose.
It changes no network weights and opens no new EpisodeInstances.

## Frozen selection rule

For each model seed:

1. retain only validation checkpoints with strict 100% completion, all 160
   deliveries, and no method failures;
2. use the finalized V1.1 selected checkpoint as the timing reference;
3. keep checkpoints whose validation MAE is no more than 2.0 steps above the
   reference;
4. minimize relocations per 100 deliveries;
5. break ties by greater dense return, lower MAE, then earlier episode.

The 2.0-step tolerance was frozen for the preceding Pareto diagnostic before
the 80000--80029 panel was opened. It is 10% of the +/-20 target window. The
selection rule itself is a post-hoc V1.2 development proposal, not a
confirmatory or deployment claim.

The rule selects:

- model seed 0: episode 475;
- model seed 1: episode 425;
- model seed 2: episode 500.

The first two preserved checkpoint files are evaluated on the already-opened
paired panel. Seed 2 episode 500 is byte-authenticated against its prior run
and reuses those rows. Current selected-best and enhanced-complete-GA rows are
also reused; no baseline is rerun.

Every source artifact, source ledger, source instance, and training/selection
record is hash-checked. Outputs remain explicitly post-hoc,
development-only, diagnostic, and deployment-ineligible.

The first post-hoc choice is intentionally not the final proposed rule. Its
seed-1 checkpoint exceeded the fixed timing tolerance on the already-opened
development panel. `derive_vcg_dense_v1_2_guarded.py` applies the frozen
validation-A ranking with a sequential panel-B timing and positive-relocation-
saving guard and selects episodes `475/500/500`. See `RESULTS.md` for the
complete interpretation.

## Run

```bash
bash experiments/vcg_dense_v1_2_posthoc/run.sh
```

After an interruption, append `--resume-existing` to the wrapper command.
