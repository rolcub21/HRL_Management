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
    "${PYTHON_BIN}" "${EVALUATE}" run-confirmation --output-dir "${OUTPUT_DIR}" --device "${EVAL_DEVICE}"
    "${PYTHON_BIN}" "${EVALUATE}" analyze-confirmation --output-dir "${OUTPUT_DIR}"
    ;;
  analyze-pilot)
    "${PYTHON_BIN}" "${EVALUATE}" analyze-pilot --output-dir "${OUTPUT_DIR}" "${@:2}"
    ;;
  analyze-confirmation)
    "${PYTHON_BIN}" "${EVALUATE}" analyze-confirmation --output-dir "${OUTPUT_DIR}" "${@:2}"
    ;;
  status)
    "${PYTHON_BIN}" "${PROGRAM}" status --output-dir "${OUTPUT_DIR}"
    ;;
  *)
    echo "usage: $0 {prepare|train-operational-arm VARIANT SEED|train-operational-pilot|train-handling-arm VARIANT SEED|train-handling-pilot|evaluate-pilot|run-pilot|train-confirmation|evaluate-confirmation|analyze-pilot|analyze-confirmation|status}" >&2
    exit 2
    ;;
esac
