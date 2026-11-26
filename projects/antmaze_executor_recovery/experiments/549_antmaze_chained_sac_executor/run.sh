#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-verify}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ai_diagnosis/SG/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
BANK="$HERE/artifacts/command_distance_bank_r47.npz"
OUTPUT="$PROJECT_ROOT/results/experiments/549_antmaze_chained_sac_executor"
CHECKPOINTS="$OUTPUT/checkpoints"
NAME="GoalAntMazeChained_model"
export PYTHONPATH="$HERE/farama_modified:$HERE:$PROJECT_ROOT/../..${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/e549-matplotlib}"

prepare() {
  "$PYTHON_BIN" "$HERE/prepare_distance_bank.py" --output "$BANK"
}

compile() {
  "$PYTHON_BIN" -m py_compile \
    "$HERE/prepare_distance_bank.py" \
    "$HERE/verify_protocol.py" \
    "$HERE/smoke.py" \
    "$HERE/evaluate.py" \
    "$HERE/evaluate_adaptive.py" \
    "$HERE/farama_modified/reach_goal_ant.py" \
    "$HERE/farama_modified/train_ant.py"
}

verify() {
  prepare
  compile
  "$PYTHON_BIN" "$HERE/verify_protocol.py"
  "$PYTHON_BIN" "$HERE/smoke.py" --command-distance-bank "$BANK"
}

train() {
  local target="$1"
  shift
  verify
  "$PYTHON_BIN" "$HERE/farama_modified/train_ant.py" \
    --command-distance-bank "$BANK" \
    --output "$OUTPUT" \
    --target-timesteps "$target" \
    --device "$DEVICE" \
    "$@"
}

evaluate_checkpoint() {
  local label="$1"
  local checkpoint="$2"
  local destination="$3"
  verify
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing E549 checkpoint: $checkpoint" >&2
    exit 1
  fi
  "$PYTHON_BIN" "$HERE/evaluate.py" \
    --command-distance-bank "$BANK" \
    --candidate "$label=$checkpoint" \
    --output "$destination" \
    --device "$DEVICE"
}

case "$MODE" in
  prepare) prepare ;;
  verify) verify ;;
  source-diff)
    diff -u "$HERE/farama_original/train_ant.py" "$HERE/farama_modified/train_ant.py" || true
    diff -u "$HERE/farama_original/reach_goal_ant.py" "$HERE/farama_modified/reach_goal_ant.py" || true
    ;;
  train-500k) train 500000 ;;
  evaluate-500k)
    evaluate_checkpoint \
      "e549_500k" \
      "$CHECKPOINTS/${NAME}_500000_steps.zip" \
      "$OUTPUT/evaluation_500k"
    ;;
  resume-1m)
    train 1000000 \
      --resume-checkpoint "$CHECKPOINTS/${NAME}_500000_steps.zip"
    ;;
  train-1m) train 1000000 ;;
  evaluate-1m)
    evaluate_checkpoint \
      "e549_1m" \
      "$CHECKPOINTS/${NAME}_1000000_steps.zip" \
      "$OUTPUT/evaluation_1m"
    ;;
  resume-5m)
    if [[ ! -f "$CHECKPOINTS/${NAME}_1000000_steps.zip" ]]; then
      echo "Run the 1M gate before the 5M continuation." >&2
      exit 1
    fi
    train 5000000 \
      --resume-checkpoint "$CHECKPOINTS/${NAME}_1000000_steps.zip"
    ;;
  evaluate-5m)
    evaluate_checkpoint \
      "e549_5m" \
      "$CHECKPOINTS/${NAME}_5000000_steps.zip" \
      "$OUTPUT/evaluation_5m"
    ;;
  evaluate-5m-adaptive)
    verify
    if [[ ! -f "$CHECKPOINTS/${NAME}_5000000_steps.zip" ]]; then
      echo "Missing E549 5M checkpoint: $CHECKPOINTS/${NAME}_5000000_steps.zip" >&2
      exit 1
    fi
    "$PYTHON_BIN" "$HERE/evaluate_adaptive.py" \
      --checkpoint "$CHECKPOINTS/${NAME}_5000000_steps.zip" \
      --command-distance-bank "$BANK" \
      --output "$OUTPUT/evaluation_5m_adaptive" \
      --device "$DEVICE"
    ;;
  resume-10m)
    if [[ ! -f "$CHECKPOINTS/${NAME}_5000000_steps.zip" ]]; then
      echo "Run the 5M gate before the 10M continuation." >&2
      exit 1
    fi
    if [[ ! -f "$CHECKPOINTS/${NAME}_replay_buffer_5000000_steps.pkl" ]]; then
      echo "Missing the 5M replay buffer required for exact continuation." >&2
      exit 1
    fi
    train 10000000 \
      --resume-checkpoint "$CHECKPOINTS/${NAME}_5000000_steps.zip"
    ;;
  evaluate-10m)
    evaluate_checkpoint \
      "e549_10m" \
      "$CHECKPOINTS/${NAME}_10000000_steps.zip" \
      "$OUTPUT/evaluation_10m"
    ;;
  resume-30m)
    if [[ ! -f "$CHECKPOINTS/${NAME}_1000000_steps.zip" ]]; then
      echo "Run the 1M gate before the 30M continuation." >&2
      exit 1
    fi
    train 30000000 \
      --resume-checkpoint "$CHECKPOINTS/${NAME}_1000000_steps.zip"
    ;;
  resume)
    if [[ "$#" -ne 3 ]]; then
      echo "Usage: $0 resume TARGET_TOTAL_STEPS /path/to/checkpoint.zip" >&2
      exit 2
    fi
    train "$2" --resume-checkpoint "$3"
    ;;
  *)
    echo "Usage: $0 {prepare|verify|source-diff|train-500k|evaluate-500k|resume-1m|train-1m|evaluate-1m|resume-5m|evaluate-5m|evaluate-5m-adaptive|resume-10m|evaluate-10m|resume-30m|resume TARGET CHECKPOINT}" >&2
    exit 2
    ;;
esac

