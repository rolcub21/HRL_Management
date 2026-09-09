#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUNNER="${ROOT}/experiments/conditioned_vcg/E05_handling_model_ablation_92k/run_future_consequence.py"
RENDERER="${ROOT}/experiments/conditioned_vcg/E05_handling_model_ablation_92k/render_future_consequence.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e05c-future-consequence-ablation-92k}"
E04_OUTPUT="${E04_OUTPUT:-${ROOT}/results/vcg-conditioned-e04-ranking-ablation-92k}"
E05_OUTPUT="${E05_OUTPUT:-${ROOT}/results/vcg-conditioned-e05-handling-ablation-92k}"
COMMAND="${1:-}"

common=(--project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --e04-output "${E04_OUTPUT}" --e05-output "${E05_OUTPUT}")

case "${COMMAND}" in
  prepare)
    "${PYTHON_BIN}" "${RUNNER}" prepare "${common[@]}"
    ;;
  run-pilot)
    "${PYTHON_BIN}" "${RUNNER}" prepare "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" run "${common[@]}" --seed 0 --instance-limit 5
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" --allow-partial
    ;;
  run-seed)
    if [[ $# -ne 2 ]]; then
      echo "usage: $0 run-seed {0|1|2}" >&2
      exit 2
    fi
    "${PYTHON_BIN}" "${RUNNER}" run "${common[@]}" --seed "$2"
    ;;
  run-all)
    "${PYTHON_BIN}" "${RUNNER}" prepare "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" run-all "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}"
    ;;
  analyze)
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}"
    ;;
  figures)
    "${PYTHON_BIN}" "${RENDERER}" "${common[@]}"
    ;;
  *)
    echo "usage: $0 {prepare|run-pilot|run-seed {0|1|2}|run-all|analyze|figures}" >&2
    exit 2
    ;;
esac
