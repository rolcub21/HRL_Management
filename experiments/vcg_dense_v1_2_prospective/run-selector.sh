#!/usr/bin/env bash
set -euo pipefail

cd /home/ai_diagnosis/HRL_Management

PYTHONUNBUFFERED=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=. \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
.venv/bin/python -u select_vcg_dense_v1_2_prospective.py \
  --training-dirs \
    results/vcg-dense-v1-2-prospective-seed3-500ep \
    results/vcg-dense-v1-2-prospective-seed4-500ep \
    results/vcg-dense-v1-2-prospective-seed5-500ep \
  --device cuda \
  --output-dir results/vcg-dense-v1-2-prospective-selection-60seed \
  "$@"
