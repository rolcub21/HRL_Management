#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHONUNBUFFERED=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=. \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
.venv/bin/python -u evaluate_vcg_dense_v1_1_v2_2_panel_control.py \
  --training-dirs \
    results/vcg-dense-v1-1-seed0-500ep \
    results/vcg-dense-v1-1-seed1-500ep \
    results/vcg-dense-v1-1-seed2-500ep \
  --v2-2-source-dir \
    results/vcg-constrained-v2-2-development-seed10-200ep \
  --device cuda \
  --output-dir \
    results/vcg-dense-v1-1-v2-2-panel-control-12instance \
  "$@"
