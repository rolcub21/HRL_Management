#!/usr/bin/env bash
set -euo pipefail

model_seed="${1:?usage: run_seed.sh MODEL_SEED [trainer arguments]}"
shift
case "${model_seed}" in
  3|4|5) ;;
  *)
    echo "MODEL_SEED must be 3, 4, or 5" >&2
    exit 2
    ;;
esac

cd /home/ai_diagnosis/HRL_Management

PYTHONUNBUFFERED=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=. \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
.venv/bin/python -u train_vcg_dense_v1_2_prospective.py \
  --model-seed "${model_seed}" \
  --device cuda \
  --output-dir "results/vcg-dense-v1-2-prospective-seed${model_seed}-500ep" \
  "$@"

