#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
CHECKPOINT="${PROJECT_ROOT}/results/kim2020-a3c-spatial-6x8-stress-v7-pilot1000/best.pth"
V2_ARTIFACT="${PROJECT_ROOT}/results/kim2020-a3c-spatial-6x8-stress-v7-pilot1000-heldout50.json"
REGRESSION_OUTPUT="${PROJECT_ROOT}/results/kim2020-a3c-spatial-6x8-stress-v7-v3-regression50.json"
FINAL_OUTPUT="${PROJECT_ROOT}/results/kim2020-a3c-spatial-6x8-stress-v7-v3-final50.json"
DIGEST="73119b952fcec9d9769522b7ad9caf74c9e29112298e3f6407b92eeb8c9dc293"

REGRESSION_SEEDS=({44000..44049})
# Preregistered here before inspection; never use these for executor tuning.
FINAL_SEEDS=({45000..45049})

run_comparison() {
    local output="$1"
    shift
    if [[ -e "${output}" ]]; then
        echo "refusing to overwrite existing result: ${output}" >&2
        exit 2
    fi
    "${PYTHON_BIN}" "${PROJECT_ROOT}/compare_kim2020_a3c_spatial.py" \
        --kim-checkpoint "${CHECKPOINT}" \
        --expected-kim-deployment-digest "${DIGEST}" \
        --lambda 1.5 \
        --mu 100 \
        --schedule-seeds "$@" \
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

verify_regression() {
    "${PYTHON_BIN}" \
        "${PROJECT_ROOT}/verify_kim2020_v3_retrieval_regression.py" \
        --before "${V2_ARTIFACT}" \
        --after "${REGRESSION_OUTPUT}"
}

case "${1:-}" in
    regression)
        run_comparison "${REGRESSION_OUTPUT}" "${REGRESSION_SEEDS[@]}"
        verify_regression
        ;;
    verify)
        verify_regression
        ;;
    final)
        verify_regression
        run_comparison "${FINAL_OUTPUT}" "${FINAL_SEEDS[@]}"
        ;;
    *)
        echo "usage: $0 {regression|verify|final}" >&2
        exit 2
        ;;
esac

