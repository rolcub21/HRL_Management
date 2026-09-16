#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUNNER="${ROOT}/experiments/conditioned_vcg/development/D10_scalability_support_screen/run.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-d10-scalability-support-screen-95k-v2}"
COMMAND="${1:-}"

common=(--output-dir "${OUTPUT_DIR}" --device cpu)

case "${COMMAND}" in
  prepare)
    "${PYTHON_BIN}" "${RUNNER}" prepare "${common[@]}"
    ;;
  run-small)
    "${PYTHON_BIN}" "${RUNNER}" run "${common[@]}" --instance-limit 1 \
      --scale reference_5x5_n8 --scale geometry_6x6_n8 --scale workload_6x6_n12
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" --allow-partial
    ;;
  run-scale)
    [[ $# -eq 2 ]] || { echo "usage: $0 run-scale SCALE_ID" >&2; exit 2; }
    "${PYTHON_BIN}" "${RUNNER}" run "${common[@]}" --instance-limit 1 --scale "$2"
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" --allow-partial
    ;;
  run-pilot)
    "${PYTHON_BIN}" "${RUNNER}" run-pilot "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" --allow-partial
    ;;
  run-all)
    "${PYTHON_BIN}" "${RUNNER}" run "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}"
    ;;
  analyze)
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" "${@:2}"
    ;;
  *)
    echo "usage: $0 {prepare|run-small|run-scale SCALE_ID|run-pilot|run-all|analyze}" >&2
    exit 2
    ;;
esac
