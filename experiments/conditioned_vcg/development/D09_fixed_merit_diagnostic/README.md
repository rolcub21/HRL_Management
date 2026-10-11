# D9 — Fixed merit-bank diagnostic

Status: diagnostic complete. It records `Qop`, immediate/future `QN`,
`lambda*QN`, final merit, and selected candidates on a common bank without
training or terminal selection.

```bash
bash experiments/conditioned_vcg/development/D09_fixed_merit_diagnostic/run.sh
```

The wrapper delegates to
`experiments/vcg_v11_conditioned_handling_fixed_merit_bank/`.
