# Nested VCG confirmation on unseen 88k instances

This is an inference-only confirmation of the fixed nested method selected on
the 85k development panel:

- VCG 1.1 with `lambda = 0` (the exact original selector), and
- the same frozen VCG 1.1 controller with the independently trained handling
  predictor at the already selected `lambda = 0.025`.

The run creates 30 new EpisodeInstances (`88000..88029`) once and evaluates the
same two arms and three frozen model seeds on every instance: 180 deterministic
rows total. It performs no training, checkpoint selection, or lambda search.

Run from the repository root:

```bash
bash experiments/vcg_v11_nested_handling_confirmation_88k/run.sh prepare
bash experiments/vcg_v11_nested_handling_confirmation_88k/run.sh run
```

`prepare` authenticates and freezes the protocol without generating or loading
the 88k panel. `run` opens the panel, evaluates it, analyzes the 30 paired
EpisodeInstance contrasts, and writes `confirmation-curves.png` and `.pdf`.
If execution is interrupted, use `resume`; completed row ledgers are not rerun.

The confirmation passes only when all 180 rows are strict-safe-complete and
simultaneous familywise-95% bounds show fewer physical rehandles/100 while MAE
is noninferior within +2 and dense return is noninferior within -20. The plot is
a cumulative frozen-panel curve, not a learning curve. It contains only the two
matched nested VCG arms; baselines require their own runs on these same 88k
instances before they can be added.
