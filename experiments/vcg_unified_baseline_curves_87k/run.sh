#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
RUNNER="${PROJECT_ROOT}/run_vcg_unified_baseline_curves_87k.py"
PLOTTER="${PROJECT_ROOT}/plot_vcg_unified_evaluation_curves_87k.py"

command="${1:-}"
case "${command}" in
  prepare)
    exec "${PYTHON}" "${RUNNER}" prepare
    ;;
  run-v11)
    exec "${PYTHON}" "${RUNNER}" run-v11
    ;;
  run-baselines)
    exec "${PYTHON}" "${RUNNER}" run-baselines
    ;;
  run-kim)
    exec "${PYTHON}" "${RUNNER}" run-kim
    ;;
  inspect)
    exec "${PYTHON}" "${RUNNER}" inspect
    ;;
  plot)
    exec "${PYTHON}" "${PLOTTER}"
    ;;
  run-all)
    "${PYTHON}" "${RUNNER}" prepare
    "${PYTHON}" "${RUNNER}" run-v11
    "${PYTHON}" "${RUNNER}" run-baselines
    "${PYTHON}" "${RUNNER}" run-kim
    exec "${PYTHON}" "${PLOTTER}"
    ;;
  *)
    echo "usage: $0 {prepare|run-v11|run-baselines|run-kim|inspect|plot|run-all}" >&2
    exit 2
    ;;
esac
