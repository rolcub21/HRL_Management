# E80: three-seed Stage B for the original 8×8 base-80 VCG

E80 trains **only the preference-conditioned future-rehandling head** on the
three frozen episode-112 Stage-A critics from E78 (seed 0 is physically stored
under E70; seeds 1 and 2 under E78). It retains the original VCG graph critic,
mode hierarchy, exact certification, progress guard, 8×8/36-job empty-start
workload, base-80 operational reward, and γ = 0.99. The Stage-A critic is
hash-checked before collection and checked again after each fit. At λ = 0 the
conditioned controller delegates to the original Stage-A policy exactly.

This is a **development-only** Stage-B experiment. It does not open a fresh
evaluation panel, establish operating-profile superiority, or authorize a
paper claim. The terminal round-8 handling head is the declared checkpoint;
there is no best-round selection. Any comparison with baselines needs a
separately frozen, paired evaluation after the three seeds finish.

The collection budget is **8 rounds × 14 complete episodes = 112 episodes per
seed** (336 total). This is a deliberately work-matched, smaller exposure than
the older generic target-size Stage-B recipe of 8 × 50 episodes. Each round
uses 11 whole episodes to fit and 3 to measure handling calibration, a fresh
positive λ in (0, .20] per episode, eight fit epochs, and the older recipe's
0.25 late-round update fraction. The new collection seeds do not overlap E78
training, development validation, or the reserved test seeds. A collection
episode must finish strictly safely to supply its Monte Carlo target; a
failure stops the run instead of silently fitting incomplete labels.

Each completed collection episode is cached separately. If interrupted
partway through a round, rerunning the same command reuses its finished
episodes; a round's head fit commits only after all 14 episodes complete.
Run seeds sequentially if sharing one GPU. `--through-round N` lets you stop
at a round boundary. A one-episode pilot is available; it is cached and will
be reused by the full seed-0 run.

From `/home/ai_diagnosis/HRL_Management`:

```bash
bash experiments/conditioned_vcg/E80_original_vcg_base80_stage_b_8x8_145k_v1/run.sh prepare
bash experiments/conditioned_vcg/E80_original_vcg_base80_stage_b_8x8_145k_v1/run.sh verify-base --model-seed 0 --device cpu
bash experiments/conditioned_vcg/E80_original_vcg_base80_stage_b_8x8_145k_v1/run.sh pilot-collect --model-seed 0 --device cuda
```

`prepare` and `verify-base` execute **zero episodes**; `pilot-collect` runs
one full episode, so it may still be slow. Then train:

```bash
for model_seed in 0 1 2; do
  bash experiments/conditioned_vcg/E80_original_vcg_base80_stage_b_8x8_145k_v1/run.sh \
    train-seed --model-seed "$model_seed" --through-round 8 --device cuda
done
bash experiments/conditioned_vcg/E80_original_vcg_base80_stage_b_8x8_145k_v1/run.sh analyze
```

The output is `results/vcg-conditioned-e80-original-base80-stage-b-8x8-145k-v1`.
`e80-report.json` summarizes completion and the terminal checkpoints. The
round histories and Monte Carlo collection records remain available for
diagnosis, but their online returns are **not** a held-out quality comparison:
the conditioned policy changes after each round, and λ varies by episode.
