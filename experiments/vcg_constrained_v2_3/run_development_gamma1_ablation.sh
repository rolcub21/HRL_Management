#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHONUNBUFFERED=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=. \
nice -n 10 .venv/bin/python -u train_vcg_constrained_v2_3.py \
  --output-dir results/vcg-constrained-v2-3-gamma1-ablation-seed10-200ep \
  --model-seed 10 \
  --episodes 200 \
  --train-seed-base 61000000 \
  --validation-seeds 85000 85001 85002 85003 85004 85005 85006 85007 85008 85009 85010 85011 \
  --validation-every 20 \
  --max-steps 2000 \
  --grid-rows 5 \
  --grid-cols 5 \
  --number-blocks 8 \
  --arrival-rate 10 \
  --proc-mean 80 \
  --gamma-operational 1.0 \
  --reward-scale 0.01 \
  --rehandle-budget-per-100 20 \
  --dual-lr 0.01 \
  --lambda-initial 0 \
  --lambda-max 20 \
  --max-hold-steps 10 \
  --max-idle-steps 20 \
  --device cuda
