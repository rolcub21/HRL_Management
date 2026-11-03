# E212 — E204 heavy-day lambda-band mechanism audit

E212 runs no training. It replays E204's exact frozen seed-0 Stage-B head on
the E205 heavy instance where lambda 20 produced 309 physical rehandles.

The first arm reruns learned lambda 20 with a full decision trace, then runs
the E207 immediate-only controller only until its first different choice. The
saved divergence records every certified candidate's frozen E191 value,
immediate rehandle indicator, predicted future rehandles, corrected merit,
and the realized remaining rehandles after the learned choice.

The second arm reuses E205's lambda 20 and 40 results and runs the missing
lambda 15, 25, and 30 points on the same EpisodeInstance. This distinguishes
an isolated lambda-20 failure from a neighboring failure band.

```bash
cd /home/ai_diagnosis/HRL_Management
set -o pipefail

MPLCONFIGDIR=/tmp/e212-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E212_e204_heavy_band_mechanism_audit_271k_v1/run.sh \
  run --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e212-e204-heavy-band-mechanism-audit-271k-v1/run.log
```

The sealed test panel remains closed.
