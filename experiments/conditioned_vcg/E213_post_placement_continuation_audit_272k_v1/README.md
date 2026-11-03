# E213 — post-placement continuation audit

E213 runs no training and keeps the sealed test panel closed. It replays the
exact E212 heavy development instance through the six shared decisions and
forces E212's learned decision-seven placement `accept:B3:6:6`. It verifies an
identical post-placement state digest, then hands control to either:

- the immediate-only centered dial at lambda 20; or
- plain frozen E191 at its exact lambda-10 delegation point.

The natural anchors are retained from E205/E207: plain E191 made 28 rehandles,
immediate-only lambda 20 made 35, and learned Stage B lambda 20 collapsed to
309. The two continuations determine whether the placement itself caused the
damage or the learned lambda-20 dial continued making bad choices afterwards.

```bash
cd /home/ai_diagnosis/HRL_Management
set -o pipefail

MPLCONFIGDIR=/tmp/e213-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E213_post_placement_continuation_audit_272k_v1/run.sh \
  run --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e213-post-placement-continuation-audit-272k-v1/run.log
```

Expect two complete frozen episodes and no optimizer updates.
