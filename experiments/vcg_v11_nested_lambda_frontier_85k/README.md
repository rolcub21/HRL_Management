# Nested VCG fixed-lambda frontier development sweep

This additive experiment tests whether one frozen nested VCG model can expose
multiple operational–handling operating points through its lambda input.

Grid: model seeds 0–2, the exact serialized 85000–85011 EpisodeInstances, and
`lambda in {0, .025, .05, .10, .20}`. Of the 180 total rows, 108 are reused
from authenticated prior experiments; only 72 missing seed-1/2 rows execute.
There is no training or policy selection.

```bash
bash experiments/vcg_v11_nested_lambda_frontier_85k/run.sh run-all
```

This is development analysis. It advances to a new, unseen 89k confirmation
only if at least three distinct aggregate nondominated points are supported by
at least two of three model seeds and rehandles are broadly nonincreasing. If
not, the next architectural step is a lambda-conditioned cost critic
`Q_N(s,c,lambda)` trained on lambda-stratified rollouts while Qop remains
frozen and lambda=0 still delegates exactly to VCG 1.1.
