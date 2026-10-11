#!/usr/bin/env bash
set -euo pipefail

cd /home/ai_diagnosis/HRL_Management

PYTHONUNBUFFERED=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=. \
.venv/bin/python -u train_vcg_constrained_v2.py \
  --output-dir results/vcg-constrained-v2-development-seed0-25ep \
  --model-seed 0 \
  --episodes 25 \
  --train-seed-base 60000000 \
  --validation-seeds 84000 84001 84002 \
  --validation-every 5 \
  --max-steps 2000 \
  --grid-rows 5 \
  --grid-cols 5 \
  --number-blocks 8 \
  --arrival-rate 10 \
  --proc-mean 80 \
  --gamma-operational 0.99 \
  --reward-scale 0.01 \
  --rehandle-budget-per-100 20 \
  --dual-lr 0.05 \
  --lambda-initial 0 \
  --lambda-max 20 \
  --dual-update-episodes 1 \
  --max-hold-steps 10 \
  --max-idle-steps 20 \
  --device cuda
