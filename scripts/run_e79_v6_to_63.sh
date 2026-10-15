#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/bukuj/hrl-e85-audit
PYTHON="$ROOT/.venv/bin/python"
RUNNER="$ROOT/experiments/conditioned_vcg/target_size_dense_stage_a_v6/resumable_continuation.py"
OUTPUT="$ROOT/results/vcg-e79-v6-direct-legal-overdue-progress-full-10x10-v1"
SUMMARY="$OUTPUT/training-summary.json"
TARGET_EPISODE=63

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/bukuj/hrl-eval

mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/overnight.lock"
if ! flock -n 9; then
    echo "Another E79-v6 overnight loop already holds $OUTPUT/overnight.lock" >&2
    exit 1
fi

while true; do
    completed=$("$PYTHON" -c \
        'import json, sys; print(json.load(open(sys.argv[1]))["through_episode"])' \
        "$SUMMARY")

    if (( completed >= TARGET_EPISODE )); then
        exit 0
    fi

    "$PYTHON" "$RUNNER" train-next \
        --output "$OUTPUT" \
        --device cuda
done
