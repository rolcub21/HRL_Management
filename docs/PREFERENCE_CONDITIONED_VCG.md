# Preference-conditioned handling VCG

This page specifies the current paper-facing VCG controller. It advanced from
an opened-panel seed-0 screen through three-seed convergence and then completed
a prospective 30-instance final comparison at seeds `90000..90029`.

## Motivation

The first jointly trained preference-conditioned vector critic changed both
operational and handling values and failed to preserve the strong VCG 1.1
operational endpoint. A later operational-residual experiment made lambda zero
exact, but its seed-0 high-lambda policy increased handling. Those experiments
remain development diagnostics.

The final architecture keeps the empirically established operational
controller fixed and learns only the quantity whose continuation policy
depends on the handling preference.

## Controller

The exact verifier and recovery-witness guard construct the admissible
frontier before learned values are consulted. For a certified candidate `c`,
the frozen VCG 1.1 encoder produces

\[
h(s,c)=[z(s),z(F(s,c)),z(F(s,c))-z(s),e(c)].
\]

The operational value is the authenticated frozen VCG 1.1 critic. Future
handling is conditioned on the deployment preference:

\[
\boxed{
Q_N(s,c,\lambda)
=\mathbf 1_{\{c=\mathrm{Reconfigure}\}}
+\operatorname{softplus} f_N(h(s,c),\lambda)
}
\]

and selection uses

\[
\boxed{
M_\lambda(s,c)=Q_{\mathrm{op}}^{\mathrm{V1.1}}(s,c)
-\lambda Q_N(s,c,\lambda).
}
\]

The structural term encodes the known immediate physical cost of a
Reconfigure macro. The learned nonnegative term predicts only subsequent
rehandles, so it cannot erase the immediate cost or predict a negative
handling consequence.

At `lambda=0`, the implementation directly invokes the original VCG 1.1
selector. It does not reconstruct the merit or query the handling network.
Certification, liveness, mode aggregation, tie-breaking, graph
representation, and operational discounting are shared across the family.

## Complete-episode target

For a completed trajectory with realized physical-rehandle indicators `n_t`,
the target at decision `t` is

\[
\boxed{
Y^{\mathrm{future}}_{N,t}=\sum_{j=t+1}^{T-1}n_j.
}
\]

The current `n_t` is excluded because it is represented by the structural
action term. The target is an undiscounted count, matching the reported
rehandles-per-100 metric. There is no handling Bellman bootstrap, handling
discount, dual multiplier update, or evaluation-time learning.

## Policy-consistent fitting and convergence

Future handling depends on the policy induced by lambda, and that policy
depends on the handling predictor. Training therefore uses bounded fitted
policy evaluation:

```text
warm-start the handling head
        |
freeze the current policy and collect complete trajectories across lambda
        |
compute exact post-current-action Monte Carlo labels
        |
fit only the future-handling network
        |
repeat to the predeclared convergence rule
```

Collection uses complete episodes with one fixed preference per episode. The
operational controller is immutable, whole episodes define the fit/validation
split, and terminals are fixed round endpoints rather than performance-picked
checkpoints.

The final authenticated terminals are:

| Model seed | Terminal round | Status |
|---:|---:|---|
| 0 | 8 | converged |
| 1 | 10 | converged after the predeclared continuation |
| 2 | 8 | converged |

A fixed-bank diagnostic records `Qop`, immediate and predicted future `QN`,
`lambda*QN`, merit, and selected candidate. It found smoothly adapting
handling estimates rather than erratic neighboring-lambda reversals; it was
diagnostic only and could not select a terminal.

## Prospective 90k final comparison

The CPU-v3 protocol froze the three terminals, ten preference coordinates,
four comparators, nuisance replications, and analysis before executing the
common 30-instance panel. It contains 1,860 rows and performs no training or
checkpoint/lambda selection.

All 900 conditioned rows were strict-safe-complete:

| Lambda | Dense return | MAE | Steps | Rehandles/100 |
|---:|---:|---:|---:|---:|
| 0 | **185.36** | **12.38** | 188.49 | 24.58 |
| .025 | 175.99 | 13.12 | 174.02 | 15.56 |
| .0375 | 172.32 | 13.39 | 169.77 | 12.92 |
| .05 | 168.02 | 13.72 | 162.72 | 8.47 |
| .075 | 160.97 | 14.24 | 157.39 | 5.00 |
| .10 | 161.34 | 14.17 | 153.24 | 3.06 |
| .125 | 157.78 | 14.44 | 153.43 | 2.78 |
| .15 | 157.19 | 14.47 | 152.44 | 2.08 |
| .175 | 157.17 | 14.48 | 152.98 | 1.81 |
| .20 | 157.10 | 14.48 | **152.09** | **1.67** |

Nine coordinates are on the point-estimate MAE--rehandles frontier. `.075`
is the only exception because `.10` improves both primary coordinates. Moving
from `lambda=0` to `.20` reduces rehandles by 22.92/100 (93.2%) while
increasing MAE by 2.10; both paired nominal 95% intervals exclude zero.

At `lambda=.20`, the same controller also improves on historical VCG 2.3 in
return, MAE, steps, and rehandles, with all four paired nominal 95% intervals
excluding zero. Capacity-aware GA is point-estimate dominated; its MAE and
return differences are not individually resolved by the paired intervals.
Dynamic PSLAP completed 28/30 instances and Kim2020 completed 448/450
stochastic rows, so their whole-method numeric aggregates are suppressed.

## Implementation and reproduction

- Controller: `vcg_v11_conditioned_handling.py`
- Iterative trainer: `train_vcg_v11_conditioned_handling_iterative.py`
- Seed-0 screen: `experiments/vcg_v11_conditioned_handling_seed0_85k/`
- Seed-1/2 training: `experiments/vcg_v11_conditioned_handling_two_phase_seeds12/`
- Seed-1 continuation: `experiments/vcg_v11_conditioned_handling_seed1_convergence_continuation/`
- Fixed-bank diagnostic: `experiments/vcg_v11_conditioned_handling_fixed_merit_bank/`
- Final runner and plot: `run_vcg_conditioned_final_comparison_90k.py` and
  `plot_vcg_conditioned_final_comparison_90k.py`
- Final experiment: `experiments/vcg_conditioned_final_comparison_90k/`
- Focused tests: `tests/test_vcg_v11_conditioned_handling*.py` and
  `tests/test_vcg_conditioned_final_comparison_90k.py`

Generated checkpoints, ledgers, reports, and figures live under ignored
`results/` paths. A clean clone needs the authenticated external artifacts
listed by the experiment contracts before completed runs can be reproduced.

## Scope

The result supports a sampled timing--handling operating frontier for the
tested 5x5 yard, eight blocks, arrival rate 10, Poisson stay mean 80, and three
frozen operational model seeds. It does not establish invariance to yard size,
block count, arrival intensity, storage duration, or unmodeled execution
disturbances; those require separate generalization and robustness panels.
