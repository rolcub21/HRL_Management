#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results/vcg-conditioned-e12-representation-ablation-94k}"
TRAIN_DEVICE="${DEVICE:-auto}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
PROGRAM="${ROOT}/experiments/conditioned_vcg/E12_representation_ablation_94k/program.py"
OPERATIONAL="${ROOT}/experiments/conditioned_vcg/E12_representation_ablation_94k/train_operational.py"
HANDLING="${ROOT}/experiments/conditioned_vcg/E12_representation_ablation_94k/train_handling.py"
EVALUATE="${ROOT}/experiments/conditioned_vcg/E12_representation_ablation_94k/evaluate.py"
FROZEN_PARENT_EVALUATE="${ROOT}/experiments/conditioned_vcg/E12_representation_ablation_94k/frozen_parent_evaluate.py"
RENDERER="${ROOT}/experiments/conditioned_vcg/E12_representation_ablation_94k/render_results.py"
EXECUTION_AUDIT="${ROOT}/experiments/conditioned_vcg/E12_representation_ablation_94k/audit_execution_consistency.py"
EXECUTION_AUDIT_OUTPUT="${EXECUTION_AUDIT_OUTPUT:-${ROOT}/results/vcg-conditioned-e12-execution-consistency-audit-94k}"
EXECUTION_FIX="${ROOT}/experiments/conditioned_vcg/E12_representation_ablation_94k/confirm_execution_fix.py"
EXECUTION_FIX_OUTPUT="${EXECUTION_FIX_OUTPUT:-${ROOT}/results/vcg-conditioned-e12-execution-fix-confirmation-94k}"
PARITY_SCREEN="${ROOT}/experiments/conditioned_vcg/E12_representation_ablation_94k/screen_corrected_executor_parity.py"
PARITY_SCREEN_OUTPUT="${PARITY_SCREEN_OUTPUT:-${ROOT}/results/vcg-conditioned-e12-executor-parity-screen-94k}"
COMMAND="${1:-}"
variants=(full_relational_successor nonrelational_successor relational_current_candidate)

prepare() {
  "${PYTHON_BIN}" "${PROGRAM}" prepare --output-dir "${OUTPUT_DIR}"
}

train_operational() {
  "${PYTHON_BIN}" "${OPERATIONAL}" --output-dir "${OUTPUT_DIR}" \
    --variant "$1" --model-seed "$2" --device "${TRAIN_DEVICE}"
}

train_handling() {
  "${PYTHON_BIN}" "${HANDLING}" --output-dir "${OUTPUT_DIR}" \
    --variant "$1" --model-seed "$2" --device "${TRAIN_DEVICE}"
}

case "${COMMAND}" in
  prepare)
    prepare
    ;;
  train-operational-arm)
    [[ $# -eq 3 ]] || { echo "usage: $0 train-operational-arm VARIANT SEED" >&2; exit 2; }
    prepare
    train_operational "$2" "$3"
    ;;
  train-operational-pilot)
    prepare
    for variant in "${variants[@]}"; do train_operational "${variant}" 0; done
    ;;
  train-handling-arm)
    [[ $# -eq 3 ]] || { echo "usage: $0 train-handling-arm VARIANT SEED" >&2; exit 2; }
    prepare
    train_handling "$2" "$3"
    ;;
  train-handling-pilot)
    prepare
    for variant in "${variants[@]}"; do train_handling "${variant}" 0; done
    ;;
  evaluate-pilot)
    prepare
    "${PYTHON_BIN}" "${EVALUATE}" run-pilot --output-dir "${OUTPUT_DIR}" --device "${EVAL_DEVICE}"
    "${PYTHON_BIN}" "${EVALUATE}" analyze-pilot --output-dir "${OUTPUT_DIR}"
    ;;
  run-pilot)
    prepare
    for variant in "${variants[@]}"; do train_operational "${variant}" 0; done
    for variant in "${variants[@]}"; do train_handling "${variant}" 0; done
    "${PYTHON_BIN}" "${EVALUATE}" run-pilot --output-dir "${OUTPUT_DIR}" --device "${EVAL_DEVICE}"
    "${PYTHON_BIN}" "${EVALUATE}" analyze-pilot --output-dir "${OUTPUT_DIR}"
    ;;
  train-confirmation)
    prepare
    for seed in 1 2; do
      for variant in "${variants[@]}"; do train_operational "${variant}" "${seed}"; done
      for variant in "${variants[@]}"; do train_handling "${variant}" "${seed}"; done
    done
    ;;
  evaluate-confirmation)
    prepare
    "${PYTHON_BIN}" "${FROZEN_PARENT_EVALUATE}" --output-dir "${OUTPUT_DIR}" --device "${EVAL_DEVICE}"
    "${PYTHON_BIN}" "${EVALUATE}" analyze-confirmation --output-dir "${OUTPUT_DIR}"
    ;;
  evaluate-confirmation-seed)
    [[ $# -eq 2 ]] || { echo "usage: $0 evaluate-confirmation-seed SEED" >&2; exit 2; }
    prepare
    "${PYTHON_BIN}" "${FROZEN_PARENT_EVALUATE}" --output-dir "${OUTPUT_DIR}" \
      --device "${EVAL_DEVICE}" --model-seed "$2"
    "${PYTHON_BIN}" "${EVALUATE}" analyze-confirmation --output-dir "${OUTPUT_DIR}" --allow-partial
    ;;
  analyze-pilot)
    "${PYTHON_BIN}" "${EVALUATE}" analyze-pilot --output-dir "${OUTPUT_DIR}" "${@:2}"
    ;;
  analyze-confirmation)
    "${PYTHON_BIN}" "${EVALUATE}" analyze-confirmation --output-dir "${OUTPUT_DIR}" "${@:2}"
    ;;
  figures)
    "${PYTHON_BIN}" "${RENDERER}" --output-dir "${OUTPUT_DIR}"
    ;;
  audit-execution-prepare)
    "${PYTHON_BIN}" "${EXECUTION_AUDIT}" prepare \
      --e12-output "${OUTPUT_DIR}" --output "${EXECUTION_AUDIT_OUTPUT}"
    ;;
  audit-execution-case)
    [[ $# -eq 2 ]] || { echo "usage: $0 audit-execution-case CASE_ID" >&2; exit 2; }
    "${PYTHON_BIN}" "${EXECUTION_AUDIT}" run \
      --e12-output "${OUTPUT_DIR}" --output "${EXECUTION_AUDIT_OUTPUT}" \
      --device "${EVAL_DEVICE}" --case "$2"
    "${PYTHON_BIN}" "${EXECUTION_AUDIT}" analyze \
      --e12-output "${OUTPUT_DIR}" --output "${EXECUTION_AUDIT_OUTPUT}" \
      --allow-partial
    ;;
  audit-execution-analyze)
    "${PYTHON_BIN}" "${EXECUTION_AUDIT}" analyze \
      --e12-output "${OUTPUT_DIR}" --output "${EXECUTION_AUDIT_OUTPUT}" "${@:2}"
    ;;
  audit-execution)
    "${PYTHON_BIN}" "${EXECUTION_AUDIT}" run \
      --e12-output "${OUTPUT_DIR}" --output "${EXECUTION_AUDIT_OUTPUT}" \
      --device "${EVAL_DEVICE}"
    ;;
  confirm-execution-fix-prepare)
    "${PYTHON_BIN}" "${EXECUTION_FIX}" prepare \
      --e12-output "${OUTPUT_DIR}" --audit-output "${EXECUTION_AUDIT_OUTPUT}" \
      --output "${EXECUTION_FIX_OUTPUT}"
    ;;
  confirm-execution-fix-case)
    [[ $# -eq 2 ]] || { echo "usage: $0 confirm-execution-fix-case CASE_ID" >&2; exit 2; }
    "${PYTHON_BIN}" "${EXECUTION_FIX}" run \
      --e12-output "${OUTPUT_DIR}" --audit-output "${EXECUTION_AUDIT_OUTPUT}" \
      --output "${EXECUTION_FIX_OUTPUT}" --device "${EVAL_DEVICE}" --case "$2"
    ;;
  confirm-execution-fix-analyze)
    "${PYTHON_BIN}" "${EXECUTION_FIX}" analyze \
      --e12-output "${OUTPUT_DIR}" --audit-output "${EXECUTION_AUDIT_OUTPUT}" \
      --output "${EXECUTION_FIX_OUTPUT}" "${@:2}"
    ;;
  confirm-execution-fix)
    "${PYTHON_BIN}" "${EXECUTION_FIX}" run \
      --e12-output "${OUTPUT_DIR}" --audit-output "${EXECUTION_AUDIT_OUTPUT}" \
      --output "${EXECUTION_FIX_OUTPUT}" --device "${EVAL_DEVICE}"
    ;;
  parity-screen-prepare)
    "${PYTHON_BIN}" "${PARITY_SCREEN}" prepare \
      --e12-output "${OUTPUT_DIR}" --output "${PARITY_SCREEN_OUTPUT}"
    ;;
  parity-screen-seed)
    [[ $# -eq 2 ]] || { echo "usage: $0 parity-screen-seed SEED" >&2; exit 2; }
    "${PYTHON_BIN}" "${PARITY_SCREEN}" run \
      --e12-output "${OUTPUT_DIR}" --output "${PARITY_SCREEN_OUTPUT}" \
      --device "${EVAL_DEVICE}" --model-seed "$2"
    ;;
  parity-screen-analyze)
    "${PYTHON_BIN}" "${PARITY_SCREEN}" analyze \
      --e12-output "${OUTPUT_DIR}" --output "${PARITY_SCREEN_OUTPUT}" "${@:2}"
    ;;
  parity-screen)
    "${PYTHON_BIN}" "${PARITY_SCREEN}" run \
      --e12-output "${OUTPUT_DIR}" --output "${PARITY_SCREEN_OUTPUT}" \
      --device "${EVAL_DEVICE}"
    ;;
  status)
    "${PYTHON_BIN}" "${PROGRAM}" status --output-dir "${OUTPUT_DIR}"
    ;;
  *)
    echo "usage: $0 {prepare|train-operational-arm VARIANT SEED|train-operational-pilot|train-handling-arm VARIANT SEED|train-handling-pilot|evaluate-pilot|run-pilot|train-confirmation|evaluate-confirmation|evaluate-confirmation-seed SEED|analyze-pilot|analyze-confirmation|figures|audit-execution-prepare|audit-execution-case CASE_ID|audit-execution-analyze|audit-execution|confirm-execution-fix-prepare|confirm-execution-fix-case CASE_ID|confirm-execution-fix-analyze|confirm-execution-fix|parity-screen-prepare|parity-screen-seed SEED|parity-screen-analyze|parity-screen|status}" >&2
    exit 2
    ;;
esac
