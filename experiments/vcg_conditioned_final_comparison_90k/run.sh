#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
DEVICE="cpu"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/vcg-conditioned-final-comparison-90k-cpu-v3}"
RUNNER="${PROJECT_ROOT}/run_vcg_conditioned_final_comparison_90k.py"

run_command() {
  command="$1"
  shift
  "${PYTHON_BIN}" "${RUNNER}" "${command}" \
    --project-root "${PROJECT_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    "$@"
}

case "${1:-run-all}" in
  prepare)
    run_command prepare
    ;;
  open-panel)
    run_command open-panel --confirm OPEN_CONDITIONED_FINAL_90K_CPU_V3
    ;;
  run-vcg)
    run_command run-vcg
    ;;
  run-vcg-seed0|run-vcg-seed1|run-vcg-seed2)
    seed="${1##*seed}"
    run_command run-vcg --model-seed "${seed}"
    ;;
  run-v23)
    run_command run-v23
    ;;
  run-baselines)
    run_command run-baselines
    ;;
  run-kim)
    run_command run-kim
    ;;
  inspect)
    run_command inspect
    ;;
  analyze)
    run_command analyze
    ;;
  plot)
    "${PYTHON_BIN}" "${PROJECT_ROOT}/plot_vcg_conditioned_final_comparison_90k.py" \
      --project-root "${PROJECT_ROOT}" \
      --output-dir "${OUTPUT_DIR}"
    ;;
  run-all)
    run_command prepare
    run_command open-panel --confirm OPEN_CONDITIONED_FINAL_90K_CPU_V3
    run_command run-vcg
    run_command run-v23
    run_command run-baselines
    run_command run-kim
    run_command analyze
    "${PYTHON_BIN}" "${PROJECT_ROOT}/plot_vcg_conditioned_final_comparison_90k.py" \
      --project-root "${PROJECT_ROOT}" \
      --output-dir "${OUTPUT_DIR}"
    ;;
  *)
    echo "usage: $0 {prepare|open-panel|run-vcg|run-vcg-seed0|run-vcg-seed1|run-vcg-seed2|run-v23|run-baselines|run-kim|inspect|analyze|plot|run-all}" >&2
    exit 2
    ;;
esac
