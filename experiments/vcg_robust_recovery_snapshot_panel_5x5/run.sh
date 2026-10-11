#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
RUNNER="${PROJECT_ROOT}/run_vcg_robust_recovery_snapshot_panel_5x5.py"
OUTPUT_DIR="${PROJECT_ROOT}/results/vcg-robust-recovery-snapshot-panel-5x5"

if (($# != 1)); then
  echo "usage: $0 {prepare|evaluate|run}" >&2
  exit 2
fi

case "$1" in
  prepare|evaluate|run)
    exec env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      "${PYTHON_BIN}" -u "${RUNNER}" "$1" \
      --project-root "${PROJECT_ROOT}" \
      --output-dir "${OUTPUT_DIR}"
    ;;
  *)
    echo "usage: $0 {prepare|evaluate|run}" >&2
    exit 2
    ;;
esac
