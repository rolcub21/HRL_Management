#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/vcg-v1-1-conditioned-handling-seed0-85k-development}"
TRAINING_DIR="$OUTPUT_ROOT/training/seed-0"

screen() {
  "$PYTHON_BIN" "$ROOT/run_vcg_v11_conditioned_handling_seed0_85k.py" \
    "$1" \
    --project-root "$ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --device "$DEVICE"
}

train() {
  local resume=()
  if [[ -f "$TRAINING_DIR/terminal.pth" ]]; then
    echo "Fixed terminal checkpoint already exists; training is complete."
    return 0
  fi
  if [[ -f "$TRAINING_DIR/latest.pth" ]]; then
    resume=(--resume-existing)
  fi
  "$PYTHON_BIN" "$ROOT/train_vcg_v11_conditioned_handling_iterative.py" \
    --output-dir "$TRAINING_DIR" \
    --device "$DEVICE" \
    "${resume[@]}"
}

case "${1:-}" in
  prepare)
    screen prepare
    ;;
  parity)
    screen parity
    ;;
  train)
    train
    ;;
  evaluate)
    screen evaluate
    ;;
  analyze)
    screen analyze
    ;;
  run-analysis)
    screen run-analysis
    ;;
  run-seed0)
    screen prepare
    screen parity
    train
    screen evaluate
    screen analyze
    ;;
  *)
    echo "Usage: $0 {prepare|parity|train|evaluate|analyze|run-analysis|run-seed0}" >&2
    exit 2
    ;;
esac
