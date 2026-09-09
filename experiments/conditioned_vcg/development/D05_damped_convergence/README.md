# D5 — Damped seed-0 convergence

Status: passed. A fixed `rho=.25` parameter update converged at the declared
round-8 terminal without evaluation-based checkpoint selection.

```bash
bash experiments/conditioned_vcg/development/D05_damped_convergence/run.sh show
```

The wrapper delegates to
`experiments/vcg_v11_conditioned_handling_damped_convergence/`.
