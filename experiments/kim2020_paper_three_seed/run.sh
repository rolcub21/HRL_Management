#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
REFERENCE_DIR="${PROJECT_ROOT}/results/kim2020-a3c-spatial-6x8-stress-v7-pilot1000"
SEED1_DIR="${PROJECT_ROOT}/results/kim2020-a3c-spatial-6x8-stress-v7-seed1-1000"
SEED2_DIR="${PROJECT_ROOT}/results/kim2020-a3c-spatial-6x8-stress-v7-seed2-1000"
PAPER_SEEDS=({97000..97049})
AGGREGATE_OUTPUT="${PROJECT_ROOT}/results/kim2020-a3c-spatial-6x8-stress-v7-paper-three-seed-summary.json"

seed_dir() {
    case "$1" in
        0) echo "${REFERENCE_DIR}" ;;
        1) echo "${SEED1_DIR}" ;;
        2) echo "${SEED2_DIR}" ;;
        *) echo "training seed must be 0, 1, or 2" >&2; exit 2 ;;
    esac
}

paper_output() {
    echo "${PROJECT_ROOT}/results/kim2020-a3c-spatial-6x8-stress-v7-paper-seed$1-final50.json"
}

train_seed() {
    local seed="$1"
    local resume_mode="${2:-fresh}"
    local output_dir
    local resume_args=()
    output_dir="$(seed_dir "${seed}")"
    if [[ "${seed}" == "0" ]]; then
        echo "seed 0 is the frozen reference and will not be retrained" >&2
        exit 2
    fi
    if [[ "${resume_mode}" == "resume" ]]; then
        if [[ ! -f "${output_dir}/latest.pth" ]]; then
            echo "cannot resume without ${output_dir}/latest.pth" >&2
            exit 2
        fi
        resume_args=(--resume "${output_dir}/latest.pth")
    elif [[ -e "${output_dir}" ]]; then
        echo "refusing to overwrite existing training directory: ${output_dir}" >&2
        exit 2
    fi
    PYTHONHASHSEED="${seed}" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        "${PYTHON_BIN}" "${PROJECT_ROOT}/train_kim2020_a3c_spatial.py" \
        --lambda 1.5 \
        --mu 100 \
        --grid-rows 6 \
        --grid-cols 8 \
        --exit-width 1 \
        --number-blocks 12 \
        --seed "${seed}" \
        "${resume_args[@]}" \
        --episodes 1000 \
        --max-steps 3000 \
        --learning-rate 0.0001 \
        --weight-decay 0 \
        --gamma 0.99 \
        --reward-scale 1 \
        --entropy-coef 0.01 \
        --value-coef 0.5 \
        --grad-clip 5 \
        --hidden-channels 32 \
        --updates-per-episode 1 \
        --eval-every 100 \
        --validation-seeds 34000 34001 34002 34003 34004 \
        --stochastic-rollouts 5 \
        --validation-policy-seed-base 10000000 \
        --training-instance-seed-base 200000 \
        --deterministic-algorithms \
        --device cuda \
        --output-dir "${output_dir}"
    verify_seed "${seed}"
}

verify_seed() {
    local seed="$1"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/verify_kim2020_training_replication.py" \
        --reference-dir "${REFERENCE_DIR}" \
        --candidate-dir "$(seed_dir "${seed}")" \
        --expected-seed "${seed}"
}

evaluate_seed() {
    local seed="$1"
    local checkpoint_dir output digest
    checkpoint_dir="$(seed_dir "${seed}")"
    output="$(paper_output "${seed}")"
    if [[ -e "${output}" ]]; then
        echo "refusing to overwrite existing paper result: ${output}" >&2
        exit 2
    fi
    if [[ "${seed}" == "0" ]]; then
        digest="73119b952fcec9d9769522b7ad9caf74c9e29112298e3f6407b92eeb8c9dc293"
    else
        digest="$("${PYTHON_BIN}" \
            "${PROJECT_ROOT}/verify_kim2020_training_replication.py" \
            --reference-dir "${REFERENCE_DIR}" \
            --candidate-dir "${checkpoint_dir}" \
            --expected-seed "${seed}" \
            --digest-only)"
    fi
    "${PYTHON_BIN}" "${PROJECT_ROOT}/compare_kim2020_a3c_spatial.py" \
        --kim-checkpoint "${checkpoint_dir}/best.pth" \
        --expected-kim-deployment-digest "${digest}" \
        --lambda 1.5 \
        --mu 100 \
        --schedule-seeds "${PAPER_SEEDS[@]}" \
        --stochastic-rollouts 5 \
        --policy-seed-base 20000000 \
        --grid-rows 6 \
        --grid-cols 8 \
        --exit-width 1 \
        --number-blocks 12 \
        --max-steps 3000 \
        --max-defer-steps 10 \
        --lookahead-margin-steps 0 \
        --target-window 20 \
        --device cpu \
        --output "${output}"
}

aggregate_results() {
    if [[ -e "${AGGREGATE_OUTPUT}" ]]; then
        echo "refusing to overwrite existing aggregate: ${AGGREGATE_OUTPUT}" >&2
        exit 2
    fi
    "${PYTHON_BIN}" "${PROJECT_ROOT}/aggregate_kim2020_three_seed.py" \
        --inputs \
        "$(paper_output 0)" \
        "$(paper_output 1)" \
        "$(paper_output 2)" \
        --output "${AGGREGATE_OUTPUT}"
}

case "${1:-}" in
    train1) train_seed 1 ;;
    train2) train_seed 2 ;;
    train-all) train_seed 1; train_seed 2 ;;
    resume1) train_seed 1 resume ;;
    resume2) train_seed 2 resume ;;
    verify1) verify_seed 1 ;;
    verify2) verify_seed 2 ;;
    eval0) evaluate_seed 0 ;;
    eval1) evaluate_seed 1 ;;
    eval2) evaluate_seed 2 ;;
    eval-all) evaluate_seed 0; evaluate_seed 1; evaluate_seed 2 ;;
    aggregate) aggregate_results ;;
    *)
        echo "usage: $0 {train1|train2|train-all|resume1|resume2|verify1|verify2|eval0|eval1|eval2|eval-all|aggregate}" >&2
        exit 2
        ;;
esac
