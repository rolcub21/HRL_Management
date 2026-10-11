#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e14-budget-scalability-95k}"
MODULE="experiments.conditioned_vcg.E14_certification_scalability_95k.budget_panel"
RENDER_MODULE="experiments.conditioned_vcg.E14_certification_scalability_95k.render_block6"
COMMAND="${1:-}"

scenarios=(
  size_5x5_occ_low size_5x5_occ_medium size_5x5_occ_high
  size_8x8_occ_low size_8x8_occ_medium size_8x8_occ_high
  size_10x10_occ_low size_10x10_occ_medium size_10x10_occ_high
)
executed_budgets=(2 4 8 16)
common=(--output-dir "${OUTPUT_DIR}")

run_coordinate() {
  local max_nodes="$1"
  local scenario="$2"
  "${PYTHON_BIN}" -m "${MODULE}" run-one "${common[@]}" \
    --max-nodes "${max_nodes}" --scenario "${scenario}"
}

refresh_report() {
  "${PYTHON_BIN}" -m "${MODULE}" analyze "${common[@]}" "$@"
}

cd "${ROOT}"
case "${COMMAND}" in
  prepare)
    "${PYTHON_BIN}" -m "${MODULE}" prepare "${common[@]}"
    ;;
  authenticate)
    "${PYTHON_BIN}" -m "${MODULE}" authenticate "${common[@]}"
    ;;
  run-one)
    run_coordinate "${2:?max-nodes required}" "${3:?scenario required}"
    refresh_report --allow-partial
    ;;
  run-pilot)
    for max_nodes in "${executed_budgets[@]}"; do
      run_coordinate "${max_nodes}" size_5x5_occ_low
    done
    refresh_report --allow-partial
    ;;
  run-all)
    for max_nodes in "${executed_budgets[@]}"; do
      for scenario in "${scenarios[@]}"; do
        run_coordinate "${max_nodes}" "${scenario}"
      done
    done
    refresh_report
    ;;
  analyze)
    refresh_report "${@:2}"
    ;;
  render)
    "${PYTHON_BIN}" -m "${RENDER_MODULE}"
    ;;
  *)
    echo "usage: $0 {prepare|authenticate|run-one MAX_NODES SCENARIO|run-pilot|run-all|analyze|render}" >&2
    exit 2
    ;;
esac
