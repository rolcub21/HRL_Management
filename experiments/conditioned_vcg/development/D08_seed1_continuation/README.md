# D8 — Seed-1 convergence continuation

Status: passed at round 10. The stopping rule used two consecutive stable
training-probe transitions and never queried evaluation performance.

```bash
bash experiments/conditioned_vcg/development/D08_seed1_continuation/run.sh analyze
```

The wrapper delegates to
`experiments/vcg_v11_conditioned_handling_seed1_convergence_continuation/`.
