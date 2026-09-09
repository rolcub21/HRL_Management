#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
RUNNER="${PROJECT_ROOT}/run_vcg_v11_nested_lambda_frontier_confirmation_89k.py"
PLOTTER="${PROJECT_ROOT}/plot_vcg_v11_nested_lambda_frontier_confirmation_89k.py"
OUTPUT="${PROJECT_ROOT}/results/vcg-v1-1-nested-lambda-frontier-confirmation-89k"

case "${1:-}" in
  prepare)
    exec "${PYTHON}" "${RUNNER}" prepare --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}"
    ;;
  run)
    "${PYTHON}" "${RUNNER}" run --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}" --device cuda --confirm OPEN_NESTED_LAMBDA_FRONTIER_89XXX
    exec env OMP_NUM_THREADS=1 "${PYTHON}" "${PLOTTER}" --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}"
    ;;
  resume)
    "${PYTHON}" "${RUNNER}" evaluate --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}" --device cuda
    "${PYTHON}" "${RUNNER}" analyze --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}"
    exec env OMP_NUM_THREADS=1 "${PYTHON}" "${PLOTTER}" --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}"
    ;;
  analyze)
    "${PYTHON}" "${RUNNER}" analyze --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}"
    exec env OMP_NUM_THREADS=1 "${PYTHON}" "${PLOTTER}" --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}"
    ;;
  *)
    echo "Usage: $0 {prepare|run|resume|analyze}" >&2
    exit 2
    ;;
esac
