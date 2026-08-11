#!/usr/bin/env bash
set -euo pipefail

cd /home/ai_diagnosis/HRL_Management

PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  .venv/bin/python -u compare_vcg_v2_3_matched_baselines.py \
  --device cpu \
  --execute-baselines \
  --resume-existing \
  --output-dir results/vcg-v2-3-matched-baselines-85k
