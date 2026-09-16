#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/vcg-unified-frozen-lambda-seed-stability-85k-development}"
PARENT_ROOT="${PARENT_ROOT:-${PROJECT_ROOT}/results/vcg-unified-lambda-pair-seed-stability-85k}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_action() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/evaluate_vcg_unified_frozen_lambda_seed_stability.py" "$1" \
    --output-dir "${RESULT_ROOT}" \
    --parent-root "${PARENT_ROOT}" \
    --device "${DEVICE}"
}

case "${1:-}" in
  prepare) run_action prepare ;;
  run) run_action run ;;
  analyze) run_action analyze ;;
  *) echo "usage: $0 {prepare|run|analyze}" >&2; exit 2 ;;
esac
