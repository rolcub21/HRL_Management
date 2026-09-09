# Kim2020 matched supplement: authenticated evaluation phase

This additive phase consumes the immutable completed training tree at
`results/vcg-v2-3-kim2020-supplement-85k` and writes only to the separate
`results/vcg-v2-3-kim2020-supplement-evaluation-85k` root.

The prospective grid is fixed at three model seeds, twelve already-opened
development instances, and five stochastic policy rolls (180 rows).  Policy
seeds are

```text
632000000 + 1000*model_seed + 10*panel_index + rollout
```

The commands, after installation in the project root, are:

```bash
bash experiments/vcg_v2_3_kim2020_supplement_evaluation_85k/run.sh prepare
bash experiments/vcg_v2_3_kim2020_supplement_evaluation_85k/run.sh evaluate_seed 0
bash experiments/vcg_v2_3_kim2020_supplement_evaluation_85k/run.sh evaluate_seed 1
bash experiments/vcg_v2_3_kim2020_supplement_evaluation_85k/run.sh evaluate_seed 2
bash experiments/vcg_v2_3_kim2020_supplement_evaluation_85k/run.sh analyze
bash experiments/vcg_v2_3_kim2020_supplement_evaluation_85k/run.sh validate
```

`prepare` must run before any Kim evaluation.  It authenticates and pins every
training artifact, all twelve serialized EpisodeInstances, the completed V2.3
stability result, the existing baseline report, and the evaluator sources.
Every later command reconstructs that trust state and fails closed on drift.

The evaluation is CPU-fixed and uses the common source-neutral duration-aware
reserved Track-B executor.  The environment return is retained as legacy
diagnostic data; the reported dense return is an exact fixed-trajectory
counterfactual rescore.  Rehandles are total physical storage-to-storage
relocations, not only a controller-specific obstruction subset.

No baseline is rerun.  No MAP Kim rollout, 86xxx EpisodeInstance, or 622m
policy seed is permitted.  The result remains development-only and does not
mutate the original V2.3 stability verdict.
