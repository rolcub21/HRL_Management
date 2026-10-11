#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
RUNNER="${PROJECT_ROOT}/run_vcg_v11_nested_handling_confirmation_88k.py"
PLOTTER="${PROJECT_ROOT}/plot_vcg_v11_nested_handling_confirmation_88k_compat.py"
OUTPUT="${PROJECT_ROOT}/results/vcg-v1-1-nested-handling-confirmation-88k"
COMMAND="${1:-}"

case "${COMMAND}" in
  prepare)
    exec "${PYTHON}" "${RUNNER}" prepare --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}"
    ;;
  run)
    "${PYTHON}" "${RUNNER}" run --project-root "${PROJECT_ROOT}" --output-dir "${OUTPUT}" --device cuda --confirm OPEN_NESTED_HANDLING_88XXX
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
