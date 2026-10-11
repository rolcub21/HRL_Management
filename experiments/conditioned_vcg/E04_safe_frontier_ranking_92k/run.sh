#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUNNER="${ROOT}/experiments/conditioned_vcg/E04_safe_frontier_ranking_92k/run.py"
RENDERER="${ROOT}/experiments/conditioned_vcg/E04_safe_frontier_ranking_92k/render_results.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e04-ranking-ablation-92k}"
COMMAND="${1:-}"

case "${COMMAND}" in
  prepare)
    "${PYTHON_BIN}" "${RUNNER}" prepare --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    ;;
  run-pilot)
    "${PYTHON_BIN}" "${RUNNER}" prepare --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    "${PYTHON_BIN}" "${RUNNER}" run --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --seed 0 --instance-limit 5 --pilot
    "${PYTHON_BIN}" "${RUNNER}" analyze --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --allow-partial
    ;;
  run-seed)
    if [[ $# -ne 2 ]]; then
      echo "usage: $0 run-seed {0|1|2}" >&2
      exit 2
    fi
    "${PYTHON_BIN}" "${RUNNER}" run --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --seed "$2"
    ;;
  run-all)
    "${PYTHON_BIN}" "${RUNNER}" prepare --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    "${PYTHON_BIN}" "${RUNNER}" run-all --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    "${PYTHON_BIN}" "${RUNNER}" analyze --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    ;;
  analyze)
    "${PYTHON_BIN}" "${RUNNER}" analyze --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    ;;
  figures)
    "${PYTHON_BIN}" "${RENDERER}" --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    ;;
  *)
    echo "usage: $0 {prepare|run-pilot|run-seed {0|1|2}|run-all|analyze|figures}" >&2
    exit 2
    ;;
esac
