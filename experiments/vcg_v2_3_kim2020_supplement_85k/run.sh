#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
PROTOCOL="${PROJECT_ROOT}/vcg_v2_3_kim2020_supplement_protocol.py"
OUTPUT_ROOT="${PROJECT_ROOT}/results/vcg-v2-3-kim2020-supplement-85k"

prepare() {
    "${PYTHON_BIN}" "${PROTOCOL}" prepare \
        --project-root "${PROJECT_ROOT}" --output-root "${OUTPUT_ROOT}"
}

validate() {
    "${PYTHON_BIN}" "${PROTOCOL}" validate \
        --project-root "${PROJECT_ROOT}" --output-root "${OUTPUT_ROOT}"
}

train_seed() {
    local seed="${1:?model seed 0, 1, or 2 is required}"
    case "${seed}" in 0|1|2) ;; *) echo "seed must be 0, 1, or 2" >&2; exit 2 ;; esac
    validate >/dev/null
    mkdir -p "${OUTPUT_ROOT}/training"
    local seed_dir="${OUTPUT_ROOT}/training/seed-${seed}"
    if ! mkdir "${seed_dir}"; then
        echo "refusing to overwrite, resume, or race for ${seed_dir}" >&2
        exit 2
    fi
    PYTHONHASHSEED="${seed}" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        "${PYTHON_BIN}" "${PROJECT_ROOT}/train_kim2020_a3c_spatial.py" \
        --lambda 10 --mu 80 --grid-rows 5 --grid-cols 5 --number-blocks 8 \
        --seed "${seed}" --episodes 1000 --max-steps 2000 \
        --learning-rate 0.0001 --weight-decay 0 --gamma 0.99 \
        --reward-scale 1 --entropy-coef 0.01 --value-coef 0.5 \
        --grad-clip 5 --hidden-channels 32 --updates-per-episode 1 \
        --eval-every 100 \
        --validation-seeds 73000000 73000001 73000002 73000003 73000004 \
        --stochastic-rollouts 5 --validation-policy-seed-base 74000000 \
        --training-instance-seed-base 70000000 \
        --deterministic-algorithms --device cuda --output-dir "${seed_dir}"
}

case "${1:-}" in
    prepare) prepare ;;
    validate) validate ;;
    train_seed) train_seed "${2:-}" ;;
    *) echo "usage: $0 {prepare|validate|train_seed {0|1|2}}" >&2; exit 2 ;;
esac
