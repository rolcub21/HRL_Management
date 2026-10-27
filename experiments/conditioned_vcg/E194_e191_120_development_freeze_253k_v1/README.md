# E194 — E191 through-120 development checkpoint freeze

E194 snapshots E191 seed 0's selected development checkpoint after the
episode-120 validation. It does not change E191 training, selection, weights,
or authenticated sources.

E191 already ranks checkpoints lexicographically by strict completion,
completion, objective return per block, storage-clock MAE, rehandles, and then
earlier episode. E194 verifies that the through-120 winner is episode 100 or
120, verifies the candidate file and SHA-256 recorded by E191, and copies that
small candidate byte-for-byte into an immutable result artifact. The 2 GB
resumable `latest.pth` and both rotating E191 candidate slots are untouched.

The frozen artifact remains development-only and is not deployment-eligible:
E191's authenticated total budget is 300 episodes.

First let the existing E191 seed reach episode 120:

```bash
cd /home/ai_diagnosis/HRL_Management
set -o pipefail

MPLCONFIGDIR=/tmp/e191-mpl \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 120 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e191-fresh-long-stay-crowding-250k-v1/seed-0-run.log
```

Then inspect and freeze the selected episode-100/120 candidate:

```bash
bash experiments/conditioned_vcg/E194_e191_120_development_freeze_253k_v1/run.sh inspect
bash experiments/conditioned_vcg/E194_e191_120_development_freeze_253k_v1/run.sh freeze
```

The output is written beneath:

```text
results/vcg-conditioned-e194-e191-120-development-freeze-253k-v1/seed-0/
  selected-development-checkpoint.pth
  freeze.json
```

Result files are ignored by Git. Commit the E194 code and tests, not checkpoint
binaries, logs, JSON results, or plots.
