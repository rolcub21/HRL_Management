#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUNNER="${ROOT}/experiments/conditioned_vcg/E03_certification_ablation_90k/run.py"
TRACER="${ROOT}/experiments/conditioned_vcg/E03_certification_ablation_90k/trace_companion.py"
RENDERER="${ROOT}/experiments/conditioned_vcg/E03_certification_ablation_90k/render_companion.py"
SUMMARY_RENDERER="${ROOT}/experiments/conditioned_vcg/E03_certification_ablation_90k/render_summary_figures.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e03-certification-ablation-90k-v2}"
COMMAND="${1:-}"

case "${COMMAND}" in
  prepare)
    "${PYTHON_BIN}" "${RUNNER}" prepare --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    ;;
  run-pilot)
    "${PYTHON_BIN}" "${RUNNER}" prepare --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    "${PYTHON_BIN}" "${RUNNER}" run --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --seed 0 --instance-limit 5
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
    "${PYTHON_BIN}" "${RUNNER}" run-all --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    ;;
  analyze)
    "${PYTHON_BIN}" "${RUNNER}" analyze --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    ;;
  figure)
    "${PYTHON_BIN}" "${TRACER}" --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --output "${OUTPUT_DIR}/e03-companion-trace.json"
    "${PYTHON_BIN}" "${RENDERER}" --trace "${OUTPUT_DIR}/e03-companion-trace.json" --output-dir "${OUTPUT_DIR}"
    ;;
  figures)
    "${PYTHON_BIN}" "${SUMMARY_RENDERER}" --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    "${PYTHON_BIN}" "${TRACER}" --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --output "${OUTPUT_DIR}/e03-companion-trace.json"
    "${PYTHON_BIN}" "${RENDERER}" --trace "${OUTPUT_DIR}/e03-companion-trace.json" --output-dir "${OUTPUT_DIR}"
    ;;
  *)
    echo "usage: $0 {prepare|run-pilot|run-seed {0|1|2}|run-all|analyze|figure|figures}" >&2
    exit 2
    ;;
esac
