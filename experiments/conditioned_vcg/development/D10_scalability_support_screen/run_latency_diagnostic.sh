#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUNNER="${ROOT}/experiments/conditioned_vcg/development/D10_scalability_support_screen/latency_diagnostic.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-d10-10x10-latency-diagnostic-95k}"
COMMAND="${1:-}"
common=(--output-dir "${OUTPUT_DIR}")

case "${COMMAND}" in
  prepare)
    "${PYTHON_BIN}" "${RUNNER}" prepare "${common[@]}"
    ;;
  run-medium)
    "${PYTHON_BIN}" "${RUNNER}" run-medium "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" --allow-partial
    ;;
  run-high)
    "${PYTHON_BIN}" "${RUNNER}" run-high "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" --allow-partial
    ;;
  run-both)
    "${PYTHON_BIN}" "${RUNNER}" run-both "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}"
    ;;
  analyze)
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" "${@:2}"
    ;;
  *)
    echo "usage: $0 {prepare|run-medium|run-high|run-both|analyze}" >&2
    exit 2
    ;;
esac
