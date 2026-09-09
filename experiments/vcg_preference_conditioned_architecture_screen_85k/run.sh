#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/vcg-preference-conditioned-architecture-screen-85k-development}"
RUNNER="${ROOT}/run_vcg_preference_conditioned_architecture_screen_85k.py"
INTERIM_B_RUNNER="${ROOT}/run_vcg_preference_conditioned_B_interim_85k.py"
TRAINER="${ROOT}/train_vcg_preference_conditioned.py"
COMMAND="${1:-}"
SEED="${2:-}"

usage() {
  echo "usage: $0 {prepare|smoke|train-conditioned|train-unconditioned|train-all|evaluate-conditioned|analyze-conditioned|run-conditioned-analysis|evaluate|analyze|run-all} [seed]"
  echo ""
  echo "The optional seed (0, 1, or 2) limits a train-* command to one paired model seed."
}

validate_seed() {
  local value="$1"
  case "${value}" in
    0|1|2) ;;
    *)
      echo "model seed must be 0, 1, or 2" >&2
      exit 2
      ;;
  esac
}

run_prepare() {
  "${PYTHON_BIN}" "${RUNNER}" prepare \
    --project-root "${ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --device "${DEVICE}"
}

train_one() {
  local architecture="$1"
  local model_seed="$2"
  local output_dir="${OUTPUT_ROOT}/training/${architecture}/seed-${model_seed}"
  local terminal="${output_dir}/terminal.pth"
  local latest="${output_dir}/latest.pth"
  if [[ -f "${terminal}" ]]; then
    echo "terminal checkpoint already exists: ${terminal}"
    return
  fi
  local resume=()
  if [[ -f "${latest}" ]]; then
    resume=(--resume-existing)
  fi
  "${PYTHON_BIN}" "${TRAINER}" \
    --output-dir "${output_dir}" \
    --architecture "${architecture}" \
    --model-seed "${model_seed}" \
    --episodes 500 \
    --preference-relabels 1 \
    --device "${DEVICE}" \
    "${resume[@]}"
}

train_arm() {
  local architecture="$1"
  local only_seed="$2"
  if [[ -n "${only_seed}" ]]; then
    validate_seed "${only_seed}"
    train_one "${architecture}" "${only_seed}"
    return
  fi
  local model_seed
  for model_seed in 0 1 2; do
    train_one "${architecture}" "${model_seed}"
  done
}

run_smoke() {
  run_prepare
  local architecture
  for architecture in conditioned unconditioned; do
    local output_dir="${OUTPUT_ROOT}/smoke/${architecture}/seed-0"
    if [[ -f "${output_dir}/terminal.pth" ]]; then
      echo "smoke checkpoint already exists: ${output_dir}/terminal.pth"
      continue
    fi
    local resume=()
    if [[ -f "${output_dir}/latest.pth" ]]; then
      resume=(--resume-existing)
    fi
    "${PYTHON_BIN}" "${TRAINER}" \
      --output-dir "${output_dir}" \
      --architecture "${architecture}" \
      --model-seed 0 \
      --preference-relabels 1 \
      --device "${DEVICE}" \
      --smoke \
      "${resume[@]}"
  done
}

run_evaluate() {
  "${PYTHON_BIN}" "${RUNNER}" evaluate \
    --project-root "${ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --device "${DEVICE}"
}

run_analyze() {
  "${PYTHON_BIN}" "${RUNNER}" analyze \
    --project-root "${ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --device "${DEVICE}"
}

run_evaluate_conditioned() {
  "${PYTHON_BIN}" "${INTERIM_B_RUNNER}" evaluate \
    --project-root "${ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --device "${DEVICE}"
}

run_analyze_conditioned() {
  "${PYTHON_BIN}" "${INTERIM_B_RUNNER}" analyze \
    --project-root "${ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --device "${DEVICE}"
}

case "${COMMAND}" in
  prepare)
    [[ -z "${SEED}" ]] || { usage >&2; exit 2; }
    run_prepare
    ;;
  smoke)
    [[ -z "${SEED}" ]] || { usage >&2; exit 2; }
    run_smoke
    ;;
  train-conditioned)
    train_arm conditioned "${SEED}"
    ;;
  train-unconditioned)
    train_arm unconditioned "${SEED}"
    ;;
  train-all)
    train_arm conditioned "${SEED}"
    train_arm unconditioned "${SEED}"
    ;;
  evaluate-conditioned)
    [[ -z "${SEED}" ]] || { usage >&2; exit 2; }
    run_evaluate_conditioned
    ;;
  analyze-conditioned)
    [[ -z "${SEED}" ]] || { usage >&2; exit 2; }
    run_analyze_conditioned
    ;;
  run-conditioned-analysis)
    [[ -z "${SEED}" ]] || { usage >&2; exit 2; }
    run_evaluate_conditioned
    run_analyze_conditioned
    ;;
  evaluate)
    [[ -z "${SEED}" ]] || { usage >&2; exit 2; }
    run_evaluate
    ;;
  analyze)
    [[ -z "${SEED}" ]] || { usage >&2; exit 2; }
    run_analyze
    ;;
  run-all)
    run_prepare
    train_arm conditioned "${SEED}"
    train_arm unconditioned "${SEED}"
    run_evaluate
    run_analyze
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
