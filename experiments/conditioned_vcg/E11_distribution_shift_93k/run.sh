#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUNNER="${ROOT}/experiments/conditioned_vcg/E11_distribution_shift_93k/run.py"
RENDERER="${ROOT}/experiments/conditioned_vcg/E11_distribution_shift_93k/render_results.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e11-distribution-shift-93k}"
COMMAND="${1:-}"
common=(--project-root "${ROOT}" --output-dir "${OUTPUT_DIR}" --device "${DEVICE:-cpu}")

case "${COMMAND}" in
  prepare)
    "${PYTHON_BIN}" "${RUNNER}" prepare "${common[@]}"
    ;;
  run-pilot)
    "${PYTHON_BIN}" "${RUNNER}" prepare "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" run "${common[@]}" --model-seed 0 --instance-limit 3 \
      --regime reference --regime arrival_spread --regime dwell_long --regime mirrored_entry
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}" --allow-partial
    ;;
  run-regime)
    if [[ $# -ne 2 ]]; then
      echo "usage: $0 run-regime {reference|arrival_spread|dwell_short|dwell_long|dwell_bimodal|mirrored_entry|combined_shift}" >&2
      exit 2
    fi
    "${PYTHON_BIN}" "${RUNNER}" prepare "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" run "${common[@]}" --regime "$2"
    ;;
  run-seed)
    if [[ $# -ne 2 ]]; then
      echo "usage: $0 run-seed {0|1|2}" >&2
      exit 2
    fi
    "${PYTHON_BIN}" "${RUNNER}" prepare "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" run "${common[@]}" --model-seed "$2"
    ;;
  run-all)
    "${PYTHON_BIN}" "${RUNNER}" prepare "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" run "${common[@]}"
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}"
    ;;
  analyze)
    "${PYTHON_BIN}" "${RUNNER}" analyze "${common[@]}"
    ;;
  figures)
    "${PYTHON_BIN}" "${RENDERER}" --project-root "${ROOT}" --output-dir "${OUTPUT_DIR}"
    ;;
  *)
    echo "usage: $0 {prepare|run-pilot|run-regime REGIME|run-seed {0|1|2}|run-all|analyze|figures}" >&2
    exit 2
    ;;
esac
