#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUNNER="${ROOT}/experiments/conditioned_vcg/E05_handling_model_ablation_92k/run.py"
MECHANISM="${ROOT}/experiments/conditioned_vcg/E05_handling_model_ablation_92k/diagnose_conditioning_mechanism.py"
E04_RUNNER="${ROOT}/experiments/conditioned_vcg/E04_safe_frontier_ranking_92k/run.sh"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e05-handling-ablation-92k}"
E04_OUTPUT="${E04_OUTPUT:-${ROOT}/results/vcg-conditioned-e04-ranking-ablation-92k}"
COMMAND="${1:-}"

case "${COMMAND}" in
  prepare)
    "${PYTHON_BIN}" "${RUNNER}" prepare --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --e04-output "${E04_OUTPUT}"
    ;;
  run-pilot)
    OUTPUT_DIR="${E04_OUTPUT}" bash "${E04_RUNNER}" run-pilot
    "${PYTHON_BIN}" "${RUNNER}" prepare --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --e04-output "${E04_OUTPUT}"
    "${PYTHON_BIN}" "${RUNNER}" run --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --e04-output "${E04_OUTPUT}" --seed 0 --instance-limit 5
    "${PYTHON_BIN}" "${RUNNER}" analyze --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --e04-output "${E04_OUTPUT}" --allow-partial
    ;;
  run-seed)
    if [[ $# -ne 2 ]]; then
      echo "usage: $0 run-seed {0|1|2}" >&2
      exit 2
    fi
    "${PYTHON_BIN}" "${RUNNER}" run --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --e04-output "${E04_OUTPUT}" --seed "$2"
    ;;
  run-all)
    "${PYTHON_BIN}" "${RUNNER}" prepare --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --e04-output "${E04_OUTPUT}"
    "${PYTHON_BIN}" "${RUNNER}" run-all --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --e04-output "${E04_OUTPUT}"
    "${PYTHON_BIN}" "${RUNNER}" analyze --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --e04-output "${E04_OUTPUT}"
    ;;
  analyze)
    "${PYTHON_BIN}" "${RUNNER}" analyze --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --e04-output "${E04_OUTPUT}"
    ;;
  mechanism)
    "${PYTHON_BIN}" "${MECHANISM}" --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --e04-output "${E04_OUTPUT}"
    ;;
  *)
    echo "usage: $0 {prepare|run-pilot|run-seed {0|1|2}|run-all|analyze|mechanism}" >&2
    exit 2
    ;;
esac
