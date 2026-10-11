#!/usr/bin/env bash
set -euo pipefail

cd /home/ai_diagnosis/HRL_Management

PYTHONUNBUFFERED=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=. \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
.venv/bin/python -u compare_vcg_dense_v1_2_posthoc.py \
  --training-dirs \
    results/vcg-dense-v1-1-seed0-500ep \
    results/vcg-dense-v1-1-seed1-500ep \
    results/vcg-dense-v1-1-seed2-500ep \
  --source-pareto-dir \
    results/vcg-dense-v1-1-pareto-development-30seed \
  --device cuda \
  --output-dir results/vcg-dense-v1-2-posthoc-development-30seed \
  "$@"
