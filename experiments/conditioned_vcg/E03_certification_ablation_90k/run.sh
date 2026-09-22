#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
RUNNER="${ROOT}/experiments/conditioned_vcg/E03_certification_ablation_90k/run.py"
TRACER="${ROOT}/experiments/conditioned_vcg/E03_certification_ablation_90k/trace_companion.py"
RENDERER="${ROOT}/experiments/conditioned_vcg/E03_certification_ablation_90k/render_companion.py"
SUMMARY_RENDERER="${ROOT}/experiments/conditioned_vcg/E03_certification_ablation_90k/render_summary_figures.py"
BASIN_RENDERER="${ROOT}/experiments/conditioned_vcg/E03_certification_ablation_90k/render_recovery_basin.py"
ATLAS_RUNNER="${ROOT}/experiments/conditioned_vcg/E03_certification_ablation_90k/full_state_recoverability_atlas.py"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e03-certification-ablation-90k-v2}"
BASIN_OUTPUT_DIR="${BASIN_OUTPUT_DIR:-${ROOT}/results/vcg-compact-recovery-basin-5x5}"
ATLAS_OUTPUT_DIR="${ATLAS_OUTPUT_DIR:-${ROOT}/results/vcg-full-state-recoverability-atlas-90k}"
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
  basin-figure)
    "${PYTHON_BIN}" "${BASIN_RENDERER}" \
      --source "${ROOT}/results/viability-kernel-5x5-all-goal.json" \
      --output-dir "${BASIN_OUTPUT_DIR}"
    ;;
  atlas-pilot)
    "${PYTHON_BIN}" "${ATLAS_RUNNER}" run-pilot \
      --project-root "${ROOT}" \
      --output-dir "${ATLAS_OUTPUT_DIR}"
    ;;
  atlas-seed0)
    "${PYTHON_BIN}" "${ATLAS_RUNNER}" run-seed0 \
      --project-root "${ROOT}" \
      --output-dir "${ATLAS_OUTPUT_DIR}"
    ;;
  atlas-render)
    "${PYTHON_BIN}" "${ATLAS_RUNNER}" render \
      --project-root "${ROOT}" \
      --output-dir "${ATLAS_OUTPUT_DIR}" \
      --model-seed 0
    ;;
  *)
    echo "usage: $0 {prepare|run-pilot|run-seed {0|1|2}|run-all|analyze|figure|figures|basin-figure|atlas-pilot|atlas-seed0|atlas-render}" >&2
    exit 2
    ;;
esac
