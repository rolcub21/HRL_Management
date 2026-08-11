#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
RUNNER="${PROJECT_ROOT}/run_vcg_v11_nested_all_baselines_89k.py"
PLOTTER="${PROJECT_ROOT}/plot_vcg_v11_nested_all_baselines_89k.py"

command="${1:-}"
case "${command}" in
  prepare)
    exec env OMP_NUM_THREADS=1 "${PYTHON}" "${RUNNER}" prepare
    ;;
  run-v23)
    exec env OMP_NUM_THREADS=1 "${PYTHON}" "${RUNNER}" run-v23
    ;;
  run-baselines)
    exec env OMP_NUM_THREADS=1 "${PYTHON}" "${RUNNER}" run-baselines
    ;;
  run-kim)
    exec env OMP_NUM_THREADS=1 "${PYTHON}" "${RUNNER}" run-kim
    ;;
  inspect)
    exec env OMP_NUM_THREADS=1 "${PYTHON}" "${RUNNER}" inspect
    ;;
  plot)
    exec env OMP_NUM_THREADS=1 "${PYTHON}" "${PLOTTER}"
    ;;
  run-all)
    env OMP_NUM_THREADS=1 "${PYTHON}" "${RUNNER}" prepare
    env OMP_NUM_THREADS=1 "${PYTHON}" "${RUNNER}" run-v23
    env OMP_NUM_THREADS=1 "${PYTHON}" "${RUNNER}" run-baselines
    env OMP_NUM_THREADS=1 "${PYTHON}" "${RUNNER}" run-kim
    exec env OMP_NUM_THREADS=1 "${PYTHON}" "${PLOTTER}"
    ;;
  *)
    echo "usage: $0 {prepare|run-v23|run-baselines|run-kim|inspect|plot|run-all}" >&2
    exit 2
    ;;
esac
