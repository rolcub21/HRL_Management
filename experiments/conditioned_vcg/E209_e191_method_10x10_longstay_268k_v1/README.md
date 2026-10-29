# E209 — fresh E191-method Stage A on scale-matched 10×10

E209 changes only physical scale and workload normalization. It keeps E191's
network, eight public queue/progress inputs, objective
`2T + E + 0.5W + 0.1L + 10R`, gamma 1, certification, mask-only progress
guard, checkpointing, and exact acceleration paths. Every learned tensor,
optimizer, replay buffer, counter, guard state, and certificate cache starts
fresh; no 8×8 or historical 10×10 checkpoint is transferred.

The 10×10 yard has 63 storage cells and 64 jobs. E191's expected stay per
cell is `700 / 35 = 20`, so E209 uses Poisson-1,260 stays. Release gaps 100,
40, and 25 induce nominal 20%, 50%, and 80% steady-state occupancy. Episodes
start empty, and validation uses three paired-duration cases per workload.

The protocol permits 300 episodes, retains every checkpoint, and validates
every 20 episodes. The first remote run targets episode 120 and is resumable.
Exploration retains E191's linear 0.90-to-0.05 schedule, with decay scaled by
the fixed job ratio: `18,500 × 64 / 36 = 32,889` decisions. The 16,000-step
limit is an error cap, not an action rule.

Prepare and run the episode-free checks before moving to the GPU server:

```bash
cd /home/ai_diagnosis/HRL_Management
MPLCONFIGDIR=/tmp/e209-mpl \
bash experiments/conditioned_vcg/E209_e191_method_10x10_longstay_268k_v1/run.sh \
  self-check
```

On the remote server, run a three-episode pilot first:

```bash
set -o pipefail
MPLCONFIGDIR=/tmp/e209-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E209_e191_method_10x10_longstay_268k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 3 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e209-e191-method-10x10-longstay-268k-v1/seed-0-run.log
```

If all three episodes complete strictly, resume to episode 120:

```bash
MPLCONFIGDIR=/tmp/e209-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E209_e191_method_10x10_longstay_268k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 120 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e209-e191-method-10x10-longstay-268k-v1/seed-0-run.log
```

This is development evidence. The sealed test panel remains closed, and Stage
B is not authorized until Stage A checkpoint selection is complete.
