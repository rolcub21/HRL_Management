#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
RUNNER="${PROJECT_ROOT}/run_vcg_v11_nested_lambda_frontier_85k.py"
PLOTTER="${PROJECT_ROOT}/plot_vcg_v11_nested_lambda_frontier_85k.py"
OUTPUT="${PROJECT_ROOT}/results/vcg-v1-1-nested-lambda-frontier-85k-development"

case "${1:-}" in
  prepare)
    exec "${PYTHON}" "${RUNNER}" prepare --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}"
    ;;
  run-missing)
    exec "${PYTHON}" "${RUNNER}" run-missing --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}" --device cuda
    ;;
  analyze)
    "${PYTHON}" "${RUNNER}" analyze --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}"
    exec env OMP_NUM_THREADS=1 "${PYTHON}" "${PLOTTER}" --output-dir "${OUTPUT}"
    ;;
  run-all)
    "${PYTHON}" "${RUNNER}" prepare --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}"
    "${PYTHON}" "${RUNNER}" run-missing --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}" --device cuda
    "${PYTHON}" "${RUNNER}" analyze --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}"
    exec env OMP_NUM_THREADS=1 "${PYTHON}" "${PLOTTER}" --output-dir "${OUTPUT}"
    ;;
  *)
    echo "usage: $0 {prepare|run-missing|analyze|run-all}" >&2
    exit 2
    ;;
esac
