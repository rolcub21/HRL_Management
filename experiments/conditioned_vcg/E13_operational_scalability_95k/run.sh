#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e13-scalability-95k-v2}"
MODULE="experiments.conditioned_vcg.E13_operational_scalability_95k.program"
COMMAND="${1:-}"

main_scenarios=(
  size_5x5_occ_low size_5x5_occ_medium size_5x5_occ_high
  size_8x8_occ_low size_8x8_occ_medium size_8x8_occ_high
  size_10x10_occ_low size_10x10_occ_medium size_10x10_occ_high
)
companion_scenarios=(
  fixed_workload_8x8_k6_n8 fixed_workload_10x10_k6_n8
  episode_8x8_medium_n36 episode_8x8_medium_n54
  rectangle_6x10_medium rectangle_10x6_medium
)
seeds=(95100 95101 95102)
common=(--output-dir "${OUTPUT_DIR}")

run_coordinate() {
  local scenario="$1"
  local seed="$2"
  "${PYTHON_BIN}" -m "${MODULE}" run-one "${common[@]}" \
    --scenario "${scenario}" --seed "${seed}"
}

run_group() {
  local group_name="$1"
  local seed_limit="$2"
  local -a scenarios
  if [[ "${group_name}" == "main" ]]; then
    scenarios=("${main_scenarios[@]}")
  else
    scenarios=("${companion_scenarios[@]}")
  fi
  local scenario seed index=0
  for scenario in "${scenarios[@]}"; do
    for seed in "${seeds[@]}"; do
      if (( index < seed_limit )); then
        run_coordinate "${scenario}" "${seed}"
      fi
      index=$((index + 1))
    done
  done
}

refresh_report() {
  "${PYTHON_BIN}" -m "${MODULE}" analyze "${common[@]}" "$@" >/dev/null
  echo "E13 report refreshed: ${OUTPUT_DIR}/e13-report.json"
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
    run_coordinate "${2:?scenario required}" "${3:?seed required}"
    refresh_report --allow-partial
    ;;
  run-main-pilot)
    for scenario in "${main_scenarios[@]}"; do
      run_coordinate "${scenario}" 95100
    done
    refresh_report --allow-partial
    ;;
  run-main)
    for scenario in "${main_scenarios[@]}"; do
      for seed in "${seeds[@]}"; do
        run_coordinate "${scenario}" "${seed}"
      done
    done
    refresh_report --allow-partial
    ;;
  run-companions)
    for scenario in "${companion_scenarios[@]}"; do
      for seed in "${seeds[@]}"; do
        run_coordinate "${scenario}" "${seed}"
      done
    done
    refresh_report
    ;;
  run-all)
    "$0" run-main
    "$0" run-companions
    ;;
  analyze)
    refresh_report "${@:2}"
    ;;
  *)
    echo "usage: $0 {prepare|authenticate|run-one SCENARIO SEED|run-main-pilot|run-main|run-companions|run-all|analyze}" >&2
    exit 2
    ;;
esac
