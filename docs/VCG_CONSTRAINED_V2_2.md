# Constrained VCG V2.2

V2.2 is the prospective development protocol that repairs the policy/operator
mismatch diagnosed in V2.1. It retains the exact-safe candidate frontier,
certified Hold option, shared graph encoder, raw operational and physical-
rehandle critic heads, vector Expected-SARSA SMDP backup, and projected dual.
It changes only policy realization:

1. training behavior samples the induced nested Lagrangian policy;
2. both continuation heads are evaluated under that same policy;
3. validation samples that policy with fixed action-only RNG streams; and
4. deployment is defined as the same stochastic policy, never raw MAP.

The external `training` flag controls learning and option evaluation mode. It
does not select a different controller policy. Training action RNG is reset
for every episode (`610000000 + episode - 1`), while replay sampling uses a
separate persistent stream seeded at `610100010`.

## Frozen development protocol

- Model initialization seed: 10.
- Training: 200 EpisodeInstances, seeds 61000000–61000199.
- Environment: 5×5, eight blocks, arrival rate 10, processing mean 80,
  2,000-step horizon.
- Dual: 20-episode critic warm-up is at lambda zero; then complete 10-episode
  updates with alpha 0.01, budget 20 physical rehandles per 100 required
  deliveries, and maximum lambda 20.
- Temperatures: blocks 1–3 use within/group 0.1/1.0; blocks 4–7 use the V2.1
  geometric schedule; blocks 8–20 use 0.01/0.05.
- Validation: episodes 20, 40, …, 200, always before a dual update at a shared
  boundary.
- Candidate looks: episodes 80, 100, 120, 140, 160, 180, and 200.

Each validation look crosses the same 12 EpisodeInstances (85000–85011) with
four action-only RNG streams:

`620000000 + 4 * instance_index + rng_index`.

Every row uses a fresh frozen weight-only clone and must be strict, complete,
exact-safe, learning-free, and mutation-free. The trainer freezes the full
canonical EpisodeInstance SHA-256, instance ID, and schedule ID at the first
look and rejects later drift.

## Expected-cost gate

For EpisodeInstance cluster `i`, with four policy realizations `j`, define

`x_i = 100 * sum_j N_phys(i,j) / sum_j N_required(i,j)`.

The point cost is the mean of the 12 cluster rates. The predeclared
Bonferroni-adjusted upper bound is

`mean(x) + 2.906203359932373 * sd(x) / sqrt(12)`.

Both the point estimate and bound must be at most 20. This is a model-based
development/FWER screen over the fixed panel, not a distribution-free or
deployment guarantee. Eligibility also requires MAE at most 20, a floor-
temperature candidate look, no current positive-residual dual saturation,
and all-row safety/integrity success.

The unopened final panel is 86000–86029 crossed with action RNGs
622000000–622000119. V2.2 development code refuses both namespaces. Earlier
83xxx instances have prior usage and are not claimed untouched.

Artifacts are development-only and non-resumable. Only
`best-development-candidate.pth` can be candidate-eligible; `latest.pth`
always remains audit-only. V1, V2, V2.1, and V2.2 checkpoint families reject
one another.

