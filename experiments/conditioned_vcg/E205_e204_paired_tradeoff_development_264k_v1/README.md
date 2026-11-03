# E205 — paired E204 trade-off development panel

E205 evaluates the completed E204 seed-0 terminal head at lambda values
`0, 5, 10, 20, 40`.  Every lambda receives the exact same EpisodeInstance at
each of the 20%, 50%, and 80% workloads.  This removes the instance mismatch
in E204's online collection and makes changes relative to lambda=10 paired.

The primary question is whether increasing lambda reduces physical rehandles.
The panel also records how that change affects timing, waiting, equipment
work, episode length, occupancy, `J10`, and `J_lambda`.  It is a small
development mechanism screen, not final paper evidence.

The runner prints a heartbeat every two minutes during long episodes.

```bash
cd /home/ai_diagnosis/HRL_Management
set -o pipefail

MPLCONFIGDIR=/tmp/e205-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E205_e204_paired_tradeoff_development_264k_v1/run.sh \
  prepare

MPLCONFIGDIR=/tmp/e205-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E205_e204_paired_tradeoff_development_264k_v1/run.sh \
  run-all --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e205-e204-paired-tradeoff-development-264k-v1/run.log

MPLCONFIGDIR=/tmp/e205-mpl \
bash experiments/conditioned_vcg/E205_e204_paired_tradeoff_development_264k_v1/run.sh \
  analyze
```

`run-all` is resumable by row.  Completed lambda/instance pairs are reused.
