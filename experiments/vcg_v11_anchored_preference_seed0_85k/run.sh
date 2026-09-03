#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ai_diagnosis/HRL_Management}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/vcg-v1-1-anchored-preference-seed0-85k-development}"
TRAINING_DIR="${OUTPUT_ROOT}/training/seed-0"
COMMAND="${1:-help}"

runner() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/run_vcg_v11_anchored_preference_seed0_85k.py" \
    "$1" \
    --project-root "${PROJECT_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --device "${DEVICE}"
}

train_seed0() {
  if [[ -f "${TRAINING_DIR}/terminal.pth" ]]; then
    echo "Seed-0 terminal checkpoint already exists; training is complete."
  elif [[ -f "${TRAINING_DIR}/latest.pth" ]]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/train_vcg_v11_anchored_preference.py" \
      --output-dir "${TRAINING_DIR}" \
      --episodes 200 \
      --resume-existing \
      --device "${DEVICE}"
  else
    "${PYTHON_BIN}" "${PROJECT_ROOT}/train_vcg_v11_anchored_preference.py" \
      --output-dir "${TRAINING_DIR}" \
      --episodes 200 \
      --device "${DEVICE}"
  fi
}

case "${COMMAND}" in
  prepare)
    runner prepare
    ;;
  parity)
    runner parity
    ;;
  train)
    train_seed0
    ;;
  evaluate)
    runner evaluate
    ;;
  analyze)
    runner analyze
    ;;
  run-analysis)
    runner run-analysis
    ;;
  run-all|run-seed0)
    runner prepare
    runner parity
    train_seed0
    runner evaluate
    runner analyze
    ;;
  *)
    echo "Usage: $0 {prepare|parity|train|evaluate|analyze|run-analysis|run-all|run-seed0}"
    exit 2
    ;;
esac
