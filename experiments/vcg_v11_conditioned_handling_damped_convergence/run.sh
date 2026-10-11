#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/vcg-v1-1-conditioned-handling-seed0-damped-convergence}"

run_experiment() {
  local resume=()
  if [[ -f "$OUTPUT_DIR/terminal.pth" ]]; then
    echo "Fixed damped-convergence terminal already exists."
    "$PYTHON_BIN" -m json.tool "$OUTPUT_DIR/damped-convergence-summary.json"
    return 0
  fi
  if [[ -f "$OUTPUT_DIR/latest.pth" ]]; then
    resume=(--resume-existing)
  fi
  "$PYTHON_BIN" "$ROOT/train_vcg_v11_conditioned_handling_damped_convergence.py" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    "${resume[@]}"
}

case "${1:-}" in
  run)
    run_experiment
    ;;
  show)
    "$PYTHON_BIN" -m json.tool "$OUTPUT_DIR/damped-convergence-summary.json"
    ;;
  *)
    echo "Usage: $0 {run|show}" >&2
    exit 2
    ;;
esac
