#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/vcg-unified-frozen-lambda-sweep-seed14-85k-development}"
PARENT_DIR="${PARENT_DIR:-${PROJECT_ROOT}/results/vcg-unified-budget-sweep-seed14-300ep-85k-development/lambda0}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_action() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/evaluate_vcg_unified_frozen_lambda_sweep.py" "$1" \
    --output-dir "${RESULT_ROOT}" \
    --parent-dir "${PARENT_DIR}" \
    --device "${DEVICE}"
}

case "${1:-}" in
  prepare) run_action prepare ;;
  run) run_action run ;;
  analyze) run_action analyze ;;
  *) echo "usage: $0 {prepare|run|analyze}" >&2; exit 2 ;;
esac
