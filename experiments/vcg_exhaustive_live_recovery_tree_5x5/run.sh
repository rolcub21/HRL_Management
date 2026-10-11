#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
RUNNER="${PROJECT_ROOT}/run_vcg_exhaustive_live_recovery_tree_5x5.py"

usage() {
  echo "usage: $0 {prepare-smoke|run-smoke|summarize-smoke|prepare-pilot|run-pilot|summarize-pilot|prepare-full|run-full|summarize-full}" >&2
  exit 2
}

if (($# != 1)); then
  usage
fi

case "$1" in
  prepare-smoke|run-smoke|summarize-smoke)
    PANEL="smoke"
    OUTPUT_DIR="${PROJECT_ROOT}/results/vcg-exhaustive-live-recovery-tree-5x5-smoke-v1"
    MAX_UNIQUE_NODES="${MAX_UNIQUE_NODES:-256}"
    ;;
  prepare-pilot|run-pilot|summarize-pilot)
    PANEL="pilot"
    OUTPUT_DIR="${PROJECT_ROOT}/results/vcg-exhaustive-live-recovery-tree-5x5-pilot-v1"
    MAX_UNIQUE_NODES="${MAX_UNIQUE_NODES:-20000}"
    ;;
  prepare-full|run-full|summarize-full)
    PANEL="full"
    OUTPUT_DIR="${PROJECT_ROOT}/results/vcg-exhaustive-live-recovery-tree-5x5-v1"
    MAX_UNIQUE_NODES="${MAX_UNIQUE_NODES:-100000}"
    ;;
  *)
    usage
    ;;
esac

case "$1" in
  prepare-*) COMMAND="prepare" ;;
  run-*) COMMAND="run" ;;
  summarize-*) COMMAND="summarize" ;;
  *) usage ;;
esac

exec env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  "${PYTHON_BIN}" -u "${RUNNER}" "${COMMAND}" \
  --panel "${PANEL}" \
  --project-root "${PROJECT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --max-unique-nodes "${MAX_UNIQUE_NODES}"
