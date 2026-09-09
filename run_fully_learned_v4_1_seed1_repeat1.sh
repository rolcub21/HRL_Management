#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

OUTPUT_DIR="results/fully-learned-v4-1-confirm-seed1-repeat1-500ep"
LOG_FILE="${OUTPUT_DIR}.console.log"

if [[ -e "$LOG_FILE" || -e "$OUTPUT_DIR/latest.pth" || -e "$OUTPUT_DIR/best.pth" ]]; then
  echo "Refusing to overwrite an existing seed-1 repeat run: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

env \
  PYTHONHASHSEED=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=. \
  nice -n 10 .venv/bin/python -u train_fully_learned_track_b.py \
  --selector-checkpoint results/reg-selector-v5-seed0-local-500ep/best.pth \
  --lambda 0.5 \
  --mu 50 \
  --grid-rows 10 \
  --grid-cols 10 \
  --number-blocks 40 \
  --seed 1 \
  --imitation-episodes 50 \
  --temporal-episodes 150 \
  --spatial-episodes 50 \
  --joint-episodes 250 \
  --max-steps 4000 \
  --batch-size 128 \
  --buffer-size 20000 \
  --update-every 4 \
  --updates-per-macro 2 \
  --joint-updates-per-macro 1 \
  --joint-td-warmup-decisions 2000 \
  --learning-rate 5e-5 \
  --spatial-learning-rate 5e-6 \
  --gamma 0.99 \
  --target-tau 0.001 \
  --grad-clip 5 \
  --huber-delta 1 \
  --reward-scale 0.01 \
  --failure-penalty -50 \
  --replay-sampling mode_balanced \
  --temporal-epsilon-start 0.25 \
  --temporal-epsilon-end 0.05 \
  --temporal-epsilon-decay-decisions 10000 \
  --joint-epsilon-start 0.05 \
  --joint-epsilon-end 0.05 \
  --joint-epsilon-decay-decisions 20000 \
  --temporal-teacher-mixture-start 0.5 \
  --temporal-teacher-mixture-end 0 \
  --temporal-teacher-mixture-decay-decisions 10000 \
  --joint-teacher-mixture-start 0.25 \
  --joint-teacher-mixture-end 0 \
  --joint-teacher-mixture-warmup-decisions 2000 \
  --joint-teacher-mixture-decay-decisions 8000 \
  --temporal-bc-start 0.5 \
  --temporal-bc-end 0.05 \
  --temporal-bc-decay-decisions 10000 \
  --joint-bc-start 0.2 \
  --joint-bc-end 0.02 \
  --joint-bc-warmup-decisions 5000 \
  --joint-bc-decay-decisions 20000 \
  --joint-main-lr-scale 0.1 \
  --joint-spatial-lr-scale 0.1 \
  --joint-spatial-freeze-decisions 5000 \
  --teacher-coefficient 0 \
  --lookahead-margin-steps 2 \
  --validation-seeds 10000 10001 10002 \
  --validation-steps 4000 \
  --eval-every 10 \
  --log-every 5 \
  --training-instance-seed-base 600000 \
  --target-window 20 \
  --device cuda \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"
