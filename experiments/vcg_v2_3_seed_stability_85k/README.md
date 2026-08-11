# Frozen VCG V2.3 seed stability (opened 85xxx development panel)

This workflow runs exact V2.3 gamma-1 training replicas for model seeds
`11`, `12`, and `13`. Seed `10` remains labeled development and is excluded
from the stability aggregate. The capacity-aware GA repair does not retrain
V2.3.

`prepare` is read-only with respect to environments and refuses to freeze the
run unless the fresh repair-V2 output is complete, authenticated, whole-safe,
and still leaves V2.3 non-dominated on MAE plus total physical rehandles. It
writes the source- and result-bound run manifest but executes no rollout:

```bash
bash experiments/vcg_v2_3_seed_stability_85k/run.sh prepare
```

The three long jobs are intentionally separate so the user can run them. Run
each exactly once (sequentially is the conservative GPU choice):

```bash
bash experiments/vcg_v2_3_seed_stability_85k/run.sh run_seed 11
bash experiments/vcg_v2_3_seed_stability_85k/run.sh run_seed 12
bash experiments/vcg_v2_3_seed_stability_85k/run.sh run_seed 13
```

Each uses 200 episodes, seven candidate looks, the original V2.3 gates and
fail-closed selected-best rule, its frozen fresh training/behavior/replay RNG
namespaces, and the common `85000..85011 x 620000000..620000047` opened
validation grid. A partial non-resumable seed directory is never overwritten.

After all three jobs finish, authenticate and aggregate without new rollouts:

```bash
bash experiments/vcg_v2_3_seed_stability_85k/run.sh analyze
```

Aggregation is equal action RNG within instance, equal instance within
training seed, then equal training seed. Baseline summaries from repair V2 are
reused once. Passing requires `3/3` eligible checkpoints, all `144` selected
rows strict-safe-complete, an aggregate not dominated by any whole-safe
baseline on MAE and total physical rehandles, and at least `2/3` individually
non-dominated seeds (`3/3` is the predeclared strong result).

This is still a development seed-stability screen, not a confirmatory or
deployment claim. The `86xxx` instances and `622000xxx` action RNGs remain
hard-rejected and sealed.
