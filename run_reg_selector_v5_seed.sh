#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {1|2}" >&2
  exit 2
fi

if [[ "$1" != "1" && "$1" != "2" ]]; then
  echo "Usage: $0 {1|2}" >&2
  exit 2
fi

SEED="$1"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

OUTPUT_DIR="results/reg-selector-v5-seed${SEED}-local-500ep"
LOG_FILE="$OUTPUT_DIR/train.log"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite an existing REG-v5 seed run: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

nice -n 10 env \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONHASHSEED="$SEED" \
  PYTHONPATH=. \
  .venv/bin/python -u train_reg_selector_v5.py \
  --lambda 0.5 \
  --mu 50 \
  --seed "$SEED" \
  --episodes 500 \
  --max-steps 4000 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --batch-size 128 \
  --min-replay-size 256 \
  --replay-size 20000 \
  --updates-per-episode 20 \
  --gamma 1.0 \
  --reward-scale 0.01 \
  --grad-clip 5.0 \
  --huber-delta 1.0 \
  --truncation-penalty 0.0 \
  --target-update-every 250 \
  --ema-tau 0.01 \
  --epsilon-start 0.90 \
  --epsilon-end 0.05 \
  --epsilon-warmup 300 \
  --epsilon-decay 10000 \
  --block-embedding-dim 64 \
  --candidate-embedding-dim 64 \
  --context-dim 128 \
  --validation-seeds 10000 10001 10002 10003 10004 \
  --eval-every 25 \
  --device cuda \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"
