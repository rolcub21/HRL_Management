#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
RUNNER="${PROJECT_ROOT}/run_vcg_dynamic_budgeted_robust_filter_panel_5x5.py"
PILOT_OUTPUT="${PROJECT_ROOT}/results/vcg-dynamic-budgeted-robust-filter-panel-5x5-pilot-v2"
FULL_OUTPUT="${PROJECT_ROOT}/results/vcg-dynamic-budgeted-robust-filter-panel-5x5-v2"

usage() {
  echo "usage: $0 {prepare-pilot|run-pilot|summarize-pilot|prepare-full|run-full|summarize-full}" >&2
  exit 2
}

if (($# != 1)); then
  usage
fi

case "$1" in
  prepare-pilot)
    COMMAND="prepare"
    PANEL="pilot"
    OUTPUT_DIR="${PILOT_OUTPUT}"
    ;;
  run-pilot)
    COMMAND="run"
    PANEL="pilot"
    OUTPUT_DIR="${PILOT_OUTPUT}"
    ;;
  summarize-pilot)
    COMMAND="summarize"
    PANEL="pilot"
    OUTPUT_DIR="${PILOT_OUTPUT}"
    ;;
  prepare-full)
    COMMAND="prepare"
    PANEL="full"
    OUTPUT_DIR="${FULL_OUTPUT}"
    ;;
  run-full)
    COMMAND="run"
    PANEL="full"
    OUTPUT_DIR="${FULL_OUTPUT}"
    ;;
  summarize-full)
    COMMAND="summarize"
    PANEL="full"
    OUTPUT_DIR="${FULL_OUTPUT}"
    ;;
  *)
    usage
    ;;
esac

exec env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  "${PYTHON_BIN}" -u "${RUNNER}" "${COMMAND}" \
  --panel "${PANEL}" \
  --project-root "${PROJECT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}"
