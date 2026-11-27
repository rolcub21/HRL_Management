#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-help}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ai_diagnosis/SG/.venv/bin/python}"
STATIC_PYTHON="${STATIC_PYTHON:-/home/ai_diagnosis/SG/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
VENDOR="${OGBENCH_VENDOR:-/home/ai_diagnosis/SG/sn/third_party/ogbench-e344-v1.2.1}"
DATASET_DIR="${OGBENCH_DATASET_DIR:-/home/ai_diagnosis/SG/sn/datasets/ogbench}"
PARENT="$ROOT/projects/tfsn_consequence_value_iteration/ogbench_antmaze_large"
PROTOCOL="$HERE/protocol.json"
DEVELOPMENT_PROTOCOL="$PARENT/DEVELOPMENT_PROTOCOL.json"
DATASET_AUDIT="$ROOT/projects/tfsn_consequence_value_iteration/results/ogbench_antmaze_large_r18/audit/dataset.json"
TRAIN_DATASET="$DATASET_DIR/antmaze-large-navigate-v0.npz"
VALIDATION_DATASET="$DATASET_DIR/antmaze-large-navigate-v0-val.npz"
TRAINING_BANK="$HERE/artifacts/ogbench_training_reach_bank.npz"
PREFLIGHT="$HERE/artifacts/preflight.json"
R21_ROOT="$ROOT/projects/tfsn_consequence_value_iteration/results/ogbench_antmaze_large_r18/executor/seed0/adaptive_reach_audit_r21"
R21_BANK="$R21_ROOT/bank.npz"
R21_BANK_META="$R21_ROOT/bank.json"
OUTPUT="$ROOT/projects/tfsn_consequence_value_iteration/results/ogbench_antmaze_large_r22_e549_executor/seed42"
CHECKPOINTS="$OUTPUT/checkpoints"
CHECKPOINT_NAME="GoalOGBenchAntMazeLargeChained_model"
E549_ROOT="$ROOT/projects/antmaze_executor_recovery/experiments/549_antmaze_chained_sac_executor"
E549_TRAIN="$E549_ROOT/farama_modified/train_ant.py"
E549_ENVIRONMENT="$E549_ROOT/farama_modified/reach_goal_ant.py"
E549_PROTOCOL="$E549_ROOT/protocol.json"

export PYTHONPATH="$VENDOR:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/r22-matplotlib}"
# R22 is state-only and never renders. An inherited OSMesa override conflicts
# with this installed MuJoCo/OpenGL pair during import.
unset MUJOCO_GL
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

prepare() {
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.e549_executor_r22.prepare_bank \
    --protocol "$PROTOCOL" --dataset-audit "$DATASET_AUDIT" \
    --training-dataset "$TRAIN_DATASET" --output "$TRAINING_BANK"
}

compile() {
  "$STATIC_PYTHON" -m json.tool "$PROTOCOL" >/dev/null
  "$STATIC_PYTHON" -m py_compile \
    "$HERE/prepare_bank.py" "$HERE/environment.py" "$HERE/train.py" \
    "$HERE/verify.py" "$HERE/audit.py" "$HERE/test_r22.py"
  bash -n "$HERE/run.sh"
  PYTHONPATH="$VENDOR:$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$STATIC_PYTHON" -m unittest \
    projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.e549_executor_r22.test_r22
}

verify() {
  prepare
  compile
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.e549_executor_r22.verify \
    --protocol "$PROTOCOL" --command-bank "$TRAINING_BANK" \
    --e549-train "$E549_TRAIN" --e549-environment "$E549_ENVIRONMENT" \
    --e549-protocol "$E549_PROTOCOL" --output "$PREFLIGHT"
}

checkpoint_path() {
  local steps="$1"
  echo "$CHECKPOINTS/${CHECKPOINT_NAME}_${steps}_steps.zip"
}

summary_path() {
  local steps="$1"
  echo "$OUTPUT/training_summary_${steps}_steps.json"
}

audit_path() {
  local steps="$1"
  echo "$OUTPUT/audit_${steps}_steps.json"
}

require_failed_gate() {
  local steps="$1"
  local report
  report="$(audit_path "$steps")"
  [[ -f "$report" ]] || {
    echo "Audit the ${steps}-step checkpoint before continuing." >&2
    exit 2
  }
  "$STATIC_PYTHON" - "$report" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if report["gate"]["executor_accepted"]:
    raise SystemExit("The preceding checkpoint passed; the frozen stopping rule forbids more training.")
PY
}

train_target() {
  local target="$1"
  local resume="${2:-}"
  verify
  local arguments=(
    --protocol "$PROTOCOL" --command-bank "$TRAINING_BANK"
    --output "$OUTPUT" --target-timesteps "$target" --device "$DEVICE"
  )
  if [[ -n "$resume" ]]; then
    arguments+=(--resume-checkpoint "$resume")
  fi
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.e549_executor_r22.train "${arguments[@]}"
}

audit_target() {
  local steps="$1"
  verify
  local summary
  summary="$(summary_path "$steps")"
  [[ -f "$summary" && -f "$R21_BANK" && -f "$R21_BANK_META" ]] || {
    echo "R22 training summary and frozen R21 bank are required." >&2
    exit 2
  }
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.e549_executor_r22.audit \
    --protocol "$PROTOCOL" --development-protocol "$DEVELOPMENT_PROTOCOL" \
    --training-summary "$summary" --training-bank "$TRAINING_BANK" \
    --r21-bank "$R21_BANK" --r21-bank-meta "$R21_BANK_META" \
    --validation "$VALIDATION_DATASET" --output "$(audit_path "$steps")" \
    --device "$DEVICE"
}

case "$MODE" in
  prepare) prepare ;;
  verify) verify ;;
  source-diff)
    diff -u "$E549_TRAIN" "$HERE/train.py" || true
    diff -u "$E549_ENVIRONMENT" "$HERE/environment.py" || true
    ;;
  train-1m) train_target 1000000 ;;
  audit-1m) audit_target 1000000 ;;
  resume-2m)
    require_failed_gate 1000000
    train_target 2000000 "$(checkpoint_path 1000000)"
    ;;
  audit-2m) audit_target 2000000 ;;
  resume-5m)
    require_failed_gate 2000000
    train_target 5000000 "$(checkpoint_path 2000000)"
    ;;
  audit-5m) audit_target 5000000 ;;
  *)
    echo "Usage: $0 {prepare|verify|source-diff|train-1m|audit-1m|resume-2m|audit-2m|resume-5m|audit-5m}" >&2
    exit 2
    ;;
esac
