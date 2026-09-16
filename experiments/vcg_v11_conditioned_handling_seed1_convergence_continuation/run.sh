#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/vcg-v1-1-conditioned-handling-seed1-convergence-continuation}"
MAX_STEPS="${MAX_STEPS:-2000}"
LOG_EVERY="${LOG_EVERY:-5}"

run_command() {
  "${PYTHON_BIN}" "${PROJECT_ROOT}/train_vcg_v11_conditioned_handling_seed1_convergence_continuation.py" \
    "$1" \
    --project-root "${PROJECT_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --max-steps "${MAX_STEPS}" \
    --log-every "${LOG_EVERY}"
}

case "${1:-run-all}" in
  prepare)
    run_command prepare
    ;;
  run)
    run_command run
    ;;
  analyze)
    run_command analyze
    ;;
  run-all)
    run_command prepare
    run_command run
    run_command analyze
    ;;
  *)
    echo "usage: $0 {prepare|run|analyze|run-all}" >&2
    exit 2
    ;;
esac
