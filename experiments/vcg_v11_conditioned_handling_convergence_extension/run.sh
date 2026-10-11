#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/vcg-v1-1-conditioned-handling-seed0-convergence-extension}"

run_extension() {
  local resume=()
  if [[ -f "$OUTPUT_DIR/terminal.pth" ]]; then
    echo "Fixed convergence-extension terminal already exists."
    "$PYTHON_BIN" -m json.tool "$OUTPUT_DIR/convergence-extension-summary.json"
    return 0
  fi
  if [[ -f "$OUTPUT_DIR/latest.pth" ]]; then
    resume=(--resume-existing)
  fi
  "$PYTHON_BIN" "$ROOT/train_vcg_v11_conditioned_handling_convergence_extension.py" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    "${resume[@]}"
}

case "${1:-}" in
  run)
    run_extension
    ;;
  show)
    "$PYTHON_BIN" -m json.tool "$OUTPUT_DIR/convergence-extension-summary.json"
    ;;
  *)
    echo "Usage: $0 {run|show}" >&2
    exit 2
    ;;
esac
