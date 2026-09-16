#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vcg-matplotlib}"
OUTPUT="${OUTPUT:-${ROOT}/results/vcg-conditioned-e16-prediction-ranking-audit}"
PROGRAM="${ROOT}/experiments/conditioned_vcg/E16_prediction_ranking_audit/analyze_offline.py"
CONTINUATIONS="${ROOT}/experiments/conditioned_vcg/E16_prediction_ranking_audit/continuation_bank.py"
DECOMPOSITION="${ROOT}/experiments/conditioned_vcg/E16_prediction_ranking_audit/decompose_signal.py"
COMPONENT_ROLLOUTS="${ROOT}/experiments/conditioned_vcg/E16_prediction_ranking_audit/component_rollout_ablation.py"
RENDERER="${ROOT}/experiments/conditioned_vcg/E16_prediction_ranking_audit/render_results.py"
CONTINUATION_OUTPUT="${CONTINUATION_OUTPUT:-${ROOT}/results/vcg-conditioned-e16-continuation-bank-96k}"
COMPONENT_OUTPUT="${COMPONENT_OUTPUT:-${ROOT}/results/vcg-conditioned-e16-component-rollout-ablation-92k}"
FIGURE_OUTPUT="${FIGURE_OUTPUT:-${ROOT}/results/vcg-conditioned-e16-mechanism-figure}"
DEVICE="${DEVICE:-cpu}"

case "${1:-}" in
  analyze-offline)
    "${PYTHON_BIN}" "${PROGRAM}" --output "${OUTPUT}"
    ;;
  prepare-continuations)
    "${PYTHON_BIN}" "${CONTINUATIONS}" prepare --output "${CONTINUATION_OUTPUT}"
    ;;
  run-continuation-pilot)
    "${PYTHON_BIN}" "${CONTINUATIONS}" run-pilot --output "${CONTINUATION_OUTPUT}" --device "${DEVICE}"
    "${PYTHON_BIN}" "${CONTINUATIONS}" analyze-partial --output "${CONTINUATION_OUTPUT}"
    ;;
  run-continuations)
    "${PYTHON_BIN}" "${CONTINUATIONS}" run-all --output "${CONTINUATION_OUTPUT}" --device "${DEVICE}"
    "${PYTHON_BIN}" "${CONTINUATIONS}" analyze --output "${CONTINUATION_OUTPUT}"
    ;;
  analyze-continuations)
    "${PYTHON_BIN}" "${CONTINUATIONS}" analyze --output "${CONTINUATION_OUTPUT}"
    ;;
  analyze-components)
    "${PYTHON_BIN}" "${DECOMPOSITION}" --output "${CONTINUATION_OUTPUT}" --device "${DEVICE}"
    ;;
  prepare-component-rollouts)
    "${PYTHON_BIN}" "${COMPONENT_ROLLOUTS}" prepare --output "${COMPONENT_OUTPUT}"
    ;;
  run-component-pilot)
    "${PYTHON_BIN}" "${COMPONENT_ROLLOUTS}" run-pilot --output "${COMPONENT_OUTPUT}" --device "${DEVICE}"
    "${PYTHON_BIN}" "${COMPONENT_ROLLOUTS}" analyze-pilot --output "${COMPONENT_OUTPUT}"
    ;;
  run-component-confirmation)
    "${PYTHON_BIN}" "${COMPONENT_ROLLOUTS}" run-confirmation --output "${COMPONENT_OUTPUT}" --device "${DEVICE}"
    "${PYTHON_BIN}" "${COMPONENT_ROLLOUTS}" analyze --output "${COMPONENT_OUTPUT}"
    ;;
  analyze-component-rollouts)
    "${PYTHON_BIN}" "${COMPONENT_ROLLOUTS}" analyze --output "${COMPONENT_OUTPUT}"
    ;;
  render)
    "${PYTHON_BIN}" "${RENDERER}" --output "${FIGURE_OUTPUT}"
    ;;
  run-all)
    "${PYTHON_BIN}" "${PROGRAM}" --output "${OUTPUT}"
    "${PYTHON_BIN}" "${CONTINUATIONS}" prepare --output "${CONTINUATION_OUTPUT}"
    "${PYTHON_BIN}" "${CONTINUATIONS}" run-all --output "${CONTINUATION_OUTPUT}" --device "${DEVICE}"
    "${PYTHON_BIN}" "${CONTINUATIONS}" analyze --output "${CONTINUATION_OUTPUT}"
    ;;
  *)
    echo "usage: $0 {analyze-offline|prepare-continuations|run-continuation-pilot|run-continuations|analyze-continuations|analyze-components|prepare-component-rollouts|run-component-pilot|run-component-confirmation|analyze-component-rollouts|render|run-all}" >&2
    exit 2
    ;;
esac
