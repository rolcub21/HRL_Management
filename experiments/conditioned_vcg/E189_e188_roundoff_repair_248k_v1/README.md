# E189 — E188 numerical audit repair

E189 continues E188 without changing its policy, guard, objective, rewards,
instances, exploration, replay, optimizer, RNG state, or validation schedule.
It replaces only the inherited fixed `1e-9` end-of-episode comparison between
two differently grouped sums of the same primitive rewards.  The tolerance is
an episode-specific floating-point bound derived from the recorded rewards,
using the validated E79-v2 construction.  Step-level reward checks remain
unchanged, and any discrepancy larger than the derived bound still fails.

Migrate the authenticated E188 episode-1 checkpoint into the distinct E189
output:

```bash
bash experiments/conditioned_vcg/E189_e188_roundoff_repair_248k_v1/run.sh \
  migrate-seed --model-seed 0
```

Then resume at episode 2:

```bash
cd /home/ai_diagnosis/HRL_Management
set -o pipefail

MPLCONFIGDIR=/tmp/e189-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E189_e188_roundoff_repair_248k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 300 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e189-e188-roundoff-repair-248k-v1/seed-0-run.log
```

Validation remains every 20 episodes.  The E188-versus-E187 paired comparison
command is unchanged apart from using the E189 runner/output:

```bash
bash experiments/conditioned_vcg/E189_e188_roundoff_repair_248k_v1/run.sh compare
```
