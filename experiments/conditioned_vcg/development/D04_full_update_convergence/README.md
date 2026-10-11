# D4 — Full-update convergence extension

Status: rejected convergence diagnostic. Two additional undamped fitted-policy
evaluation rounds did not satisfy the predeclared stability rule. This result
motivated the fixed damped update in D5.

```bash
bash experiments/conditioned_vcg/development/D04_full_update_convergence/run.sh show
```

The wrapper delegates to
`experiments/vcg_v11_conditioned_handling_convergence_extension/`.
