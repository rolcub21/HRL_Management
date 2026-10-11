#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/ai_diagnosis/HRL_Management}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

case "${1:-}" in
  prepare|run|analyze)
    exec "${PYTHON_BIN}" "${PROJECT_ROOT}/run_kim2020_final86_comparison.py" "$1"
    ;;
  *)
    echo "usage: $0 {prepare|run|analyze}" >&2
    exit 2
    ;;
esac

