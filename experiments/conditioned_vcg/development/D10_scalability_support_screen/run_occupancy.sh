#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUNNER="${ROOT}/experiments/conditioned_vcg/development/D10_scalability_support_screen/occupancy_extension.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-d10-occupancy-extension-95k}"
COMMAND="${1:-}"
common=(--output-dir "${OUTPUT_DIR}" --device cpu)

case "${COMMAND}" in
  prepare)
    "${PYTHON_BIN}" "${RUNNER}" prepare "${common[@]}"
    ;;
  run-core-pilot)
    "${PYTHON_BIN}" "${RUNNER}" run-core-pilot "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" --allow-partial
    ;;
  run-grid-pilot)
    "${PYTHON_BIN}" "${RUNNER}" run-grid-pilot "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" --allow-partial
    ;;
  run-pilot)
    "${PYTHON_BIN}" "${RUNNER}" run-pilot "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" --allow-partial
    ;;
  run-scenario)
    [[ $# -ge 2 && $# -le 3 ]] || { echo "usage: $0 run-scenario SCENARIO_ID [INSTANCE_LIMIT]" >&2; exit 2; }
    limit="${3:-1}"
    "${PYTHON_BIN}" "${RUNNER}" run "${common[@]}" --scenario "$2" --instance-limit "${limit}"
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" --allow-partial
    ;;
  run-grid)
    "${PYTHON_BIN}" "${RUNNER}" run "${common[@]}" \
      --scenario size_5x5_occ_low --scenario size_5x5_occ_medium --scenario size_5x5_occ_high \
      --scenario size_8x8_occ_low --scenario size_8x8_occ_medium --scenario size_8x8_occ_high \
      --scenario size_10x10_occ_low --scenario size_10x10_occ_medium --scenario size_10x10_occ_high
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
    echo "usage: $0 {prepare|run-core-pilot|run-grid-pilot|run-pilot|run-scenario SCENARIO_ID [INSTANCE_LIMIT]|run-grid|run-all|analyze}" >&2
    exit 2
    ;;
esac
