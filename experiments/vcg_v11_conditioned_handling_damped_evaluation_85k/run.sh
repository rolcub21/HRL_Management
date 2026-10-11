#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/vcg-v1-1-conditioned-handling-seed0-damped-convergence}"

evaluate() {
  "$PYTHON_BIN" "$ROOT/run_vcg_v11_conditioned_handling_damped_evaluation_85k.py" \
    "$1" \
    --project-root "$ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --device "$DEVICE"
}

case "${1:-}" in
  prepare)
    evaluate prepare
    ;;
  evaluate)
    evaluate evaluate
    ;;
  analyze)
    evaluate analyze
    ;;
  run)
    evaluate run
    ;;
  *)
    echo "Usage: $0 {prepare|evaluate|analyze|run}" >&2
    exit 2
    ;;
esac
