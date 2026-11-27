#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-help}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PYTHON_BIN="${OGBENCH_PYTHON:-/home/ai_diagnosis/anaconda3/envs/e344-ogbench/bin/python}"
STATIC_PYTHON="${STATIC_PYTHON:-/home/ai_diagnosis/SG/.venv/bin/python}"
VENDOR="${OGBENCH_VENDOR:-/home/ai_diagnosis/SG/sn/third_party/ogbench-e344-v1.2.1}"
IMPLS="${OGBENCH_IMPLS_ROOT:-$VENDOR/impls}"
DATASET_DIR="${OGBENCH_DATASET_DIR:-/home/ai_diagnosis/SG/sn/datasets/ogbench}"
OUT="${OGBENCH_TFSN_LARGE_OUT:-$ROOT/projects/tfsn_consequence_value_iteration/results/ogbench_antmaze_large_r18}"
PROTOCOL="$HERE/DEVELOPMENT_PROTOCOL.json"
FINETUNE_PROTOCOL="$HERE/EXECUTOR_FINETUNE_PROTOCOL.json"
MIXED_FINETUNE_PROTOCOL="$HERE/EXECUTOR_FINETUNE_MIXED_PROTOCOL.json"
ADAPTIVE_REACH_AUDIT_PROTOCOL="$HERE/EXECUTOR_ADAPTIVE_REACH_AUDIT_PROTOCOL.json"
AUDIT="$OUT/audit"
RUN_ROOT="$OUT/executor/seed0/reference_runs"
GROUP="TF_SN_R74_OGBench_AntMazeLarge_CRL_seed0"
TRAIN_DATASET="$DATASET_DIR/antmaze-large-navigate-v0.npz"
VALIDATION_DATASET="$DATASET_DIR/antmaze-large-navigate-v0-val.npz"
EXECUTOR_RECORD="$OUT/executor/seed0/executor.json"
CHAIN_ROOT="$OUT/executor/seed0/chained_audit"
CHAIN_BANK="$CHAIN_ROOT/bank.npz"
CHAIN_BANK_META="$CHAIN_ROOT/bank.json"
CHAIN_REPORT="$CHAIN_ROOT/audit.json"
FINETUNE_ROOT="$OUT/executor/seed0/chained_finetune_r19"
FINETUNE_500_RECORD="$FINETUNE_ROOT/steps_0500000/executor.json"
FINETUNE_1000_RECORD="$FINETUNE_ROOT/steps_1000000/executor.json"
FINETUNE_500_AUDIT="$FINETUNE_ROOT/steps_0500000/chained_audit.json"
FINETUNE_1000_AUDIT="$FINETUNE_ROOT/steps_1000000/chained_audit.json"
MIXED_FINETUNE_ROOT="$OUT/executor/seed0/chained_mixed_finetune_r20"
MIXED_FINETUNE_500_RECORD="$MIXED_FINETUNE_ROOT/steps_0500000/executor.json"
MIXED_FINETUNE_1000_RECORD="$MIXED_FINETUNE_ROOT/steps_1000000/executor.json"
MIXED_FINETUNE_500_AUDIT="$MIXED_FINETUNE_ROOT/steps_0500000/chained_audit.json"
MIXED_FINETUNE_1000_AUDIT="$MIXED_FINETUNE_ROOT/steps_1000000/chained_audit.json"
ADAPTIVE_REACH_ROOT="$OUT/executor/seed0/adaptive_reach_audit_r21"
ADAPTIVE_REACH_BANK="$ADAPTIVE_REACH_ROOT/bank.npz"
ADAPTIVE_REACH_BANK_META="$ADAPTIVE_REACH_ROOT/bank.json"
ADAPTIVE_REACH_REPORT="$ADAPTIVE_REACH_ROOT/audit.json"

export PYTHONPATH="$IMPLS:$VENDOR:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OGBENCH_IMPLS_ROOT="$IMPLS"
export OGBENCH_DATASET_DIR="$DATASET_DIR"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

verify() {
  "$STATIC_PYTHON" -m json.tool "$PROTOCOL" >/dev/null
  "$STATIC_PYTHON" -m json.tool "$FINETUNE_PROTOCOL" >/dev/null
  "$STATIC_PYTHON" -m json.tool "$MIXED_FINETUNE_PROTOCOL" >/dev/null
  "$STATIC_PYTHON" -m json.tool "$ADAPTIVE_REACH_AUDIT_PROTOCOL" >/dev/null
  "$STATIC_PYTHON" -m py_compile \
    "$HERE/preflight.py" "$HERE/run_official_main.py" \
    "$HERE/record_executor.py" "$HERE/audit_executor.py" \
    "$HERE/audit_executor_adaptive_reach.py" \
    "$HERE/finetune_executor.py" "$HERE/gate_finetuned_executor.py"
  bash -n "$HERE/run.sh"
  PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$STATIC_PYTHON" -m unittest \
    projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.test_executor_audit \
    projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.test_finetune_executor
  "$STATIC_PYTHON" "$HERE/preflight.py" verify --output "$AUDIT/static.json"
}

runtime() {
  JAX_PLATFORMS=cpu "$PYTHON_BIN" "$HERE/preflight.py" runtime \
    --vendor "$VENDOR" --output "$AUDIT/runtime.json"
}

gpu() {
  "$PYTHON_BIN" "$HERE/preflight.py" runtime --require-gpu \
    --vendor "$VENDOR" --output "$AUDIT/gpu.json"
}

download() {
  "$PYTHON_BIN" -c "import ogbench; ogbench.download_datasets(['antmaze-large-navigate-v0'], dataset_dir='$DATASET_DIR')"
}

dataset() {
  "$PYTHON_BIN" "$HERE/preflight.py" dataset --dataset-dir "$DATASET_DIR" --output "$AUDIT/dataset.json"
}

train_executor_seed0() {
  [[ -f "$AUDIT/static.json" && -f "$AUDIT/runtime.json" && -f "$AUDIT/gpu.json" && -f "$AUDIT/dataset.json" ]] || {
    echo "Run preflight before executor training." >&2
    exit 2
  }
  mkdir -p "$RUN_ROOT"
  (
    cd "$IMPLS"
    "$PYTHON_BIN" "$HERE/run_official_main.py" \
      --env_name=antmaze-large-navigate-v0 \
      --agent=agents/crl.py --agent.alpha=0.1 \
      --seed=0 --train_steps=1000000 --log_interval=5000 --save_interval=1000000 \
      --eval_interval=1000001 --eval_tasks=0 --eval_episodes=0 --video_episodes=0 \
      --save_dir="$RUN_ROOT" --run_group="$GROUP"
  )
  "$PYTHON_BIN" "$HERE/record_executor.py" \
    --run-root "$RUN_ROOT" --protocol "$PROTOCOL" \
    --output "$EXECUTOR_RECORD"
}

freeze_executor_audit_seed0() {
  [[ -f "$EXECUTOR_RECORD" && -f "$TRAIN_DATASET" && -f "$VALIDATION_DATASET" ]] || {
    echo "Executor record and both audited datasets are required." >&2
    exit 2
  }
  "$PYTHON_BIN" "$HERE/audit_executor.py" freeze \
    --protocol "$PROTOCOL" --dataset-audit "$AUDIT/dataset.json" \
    --validation "$VALIDATION_DATASET" \
    --bank "$CHAIN_BANK" --bank-meta "$CHAIN_BANK_META"
}

audit_executor_seed0() {
  freeze_executor_audit_seed0
  "$PYTHON_BIN" "$HERE/audit_executor.py" evaluate \
    --protocol "$PROTOCOL" --dataset-audit "$AUDIT/dataset.json" \
    --executor "$EXECUTOR_RECORD" \
    --train-dataset "$TRAIN_DATASET" --validation "$VALIDATION_DATASET" \
    --bank "$CHAIN_BANK" --bank-meta "$CHAIN_BANK_META" \
    --output "$CHAIN_REPORT"
}

audit_executor_adaptive_reach_seed0() {
  [[ -f "$EXECUTOR_RECORD" && -f "$TRAIN_DATASET" && -f "$VALIDATION_DATASET" ]] || {
    echo "The official executor record and both audited datasets are required." >&2
    exit 2
  }
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.audit_executor_adaptive_reach freeze \
    --protocol "$ADAPTIVE_REACH_AUDIT_PROTOCOL" \
    --development-protocol "$PROTOCOL" --dataset-audit "$AUDIT/dataset.json" \
    --validation "$VALIDATION_DATASET" --bank "$ADAPTIVE_REACH_BANK" \
    --bank-meta "$ADAPTIVE_REACH_BANK_META"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.audit_executor_adaptive_reach evaluate \
    --protocol "$ADAPTIVE_REACH_AUDIT_PROTOCOL" \
    --development-protocol "$PROTOCOL" --dataset-audit "$AUDIT/dataset.json" \
    --executor "$EXECUTOR_RECORD" --train-dataset "$TRAIN_DATASET" \
    --validation "$VALIDATION_DATASET" --bank "$ADAPTIVE_REACH_BANK" \
    --bank-meta "$ADAPTIVE_REACH_BANK_META" --output "$ADAPTIVE_REACH_REPORT"
}

finetune_executor_500k() {
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.finetune_executor \
    --protocol "$FINETUNE_PROTOCOL" --development-protocol "$PROTOCOL" \
    --dataset-audit "$AUDIT/dataset.json" --executor "$EXECUTOR_RECORD" \
    --train-dataset "$TRAIN_DATASET" --target-steps 500000 \
    --output-root "$FINETUNE_ROOT"
}

finetune_executor_1m() {
  [[ -f "$FINETUNE_500_RECORD" ]] || {
    echo "The 500k fine-tuning checkpoint is required before the 1M continuation." >&2
    exit 2
  }
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.finetune_executor \
    --protocol "$FINETUNE_PROTOCOL" --development-protocol "$PROTOCOL" \
    --dataset-audit "$AUDIT/dataset.json" --executor "$EXECUTOR_RECORD" \
    --train-dataset "$TRAIN_DATASET" --target-steps 1000000 \
    --output-root "$FINETUNE_ROOT" --resume-record "$FINETUNE_500_RECORD"
}

finetune_mixed_executor_500k() {
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.finetune_executor \
    --protocol "$MIXED_FINETUNE_PROTOCOL" --development-protocol "$PROTOCOL" \
    --dataset-audit "$AUDIT/dataset.json" --executor "$EXECUTOR_RECORD" \
    --train-dataset "$TRAIN_DATASET" --target-steps 500000 \
    --output-root "$MIXED_FINETUNE_ROOT"
}

finetune_mixed_executor_1m() {
  [[ -f "$MIXED_FINETUNE_500_RECORD" ]] || {
    echo "The mixed 500k checkpoint is required before the 1M continuation." >&2
    exit 2
  }
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.finetune_executor \
    --protocol "$MIXED_FINETUNE_PROTOCOL" --development-protocol "$PROTOCOL" \
    --dataset-audit "$AUDIT/dataset.json" --executor "$EXECUTOR_RECORD" \
    --train-dataset "$TRAIN_DATASET" --target-steps 1000000 \
    --output-root "$MIXED_FINETUNE_ROOT" \
    --resume-record "$MIXED_FINETUNE_500_RECORD"
}

audit_finetuned_executor() {
  local finetune_protocol="$1"
  local record="$2"
  local candidate="$3"
  local gate_output="$4"
  [[ -f "$record" && -f "$CHAIN_REPORT" ]] || {
    echo "The fine-tuned executor record and frozen baseline audit are required." >&2
    exit 2
  }
  freeze_executor_audit_seed0
  "$PYTHON_BIN" "$HERE/audit_executor.py" evaluate \
    --protocol "$PROTOCOL" --dataset-audit "$AUDIT/dataset.json" \
    --executor "$record" --train-dataset "$TRAIN_DATASET" \
    --validation "$VALIDATION_DATASET" --bank "$CHAIN_BANK" \
    --bank-meta "$CHAIN_BANK_META" --output "$candidate"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.ogbench_antmaze_large.gate_finetuned_executor \
    --protocol "$finetune_protocol" --baseline "$CHAIN_REPORT" \
    --candidate "$candidate" --executor "$record" --output "$gate_output"
}

case "$MODE" in
  verify) verify ;;
  runtime) verify; runtime ;;
  gpu) verify; gpu ;;
  download) verify; runtime; download ;;
  preflight) verify; runtime; gpu; dataset ;;
  train-executor-seed0) verify; runtime; gpu; dataset; train_executor_seed0 ;;
  record-executor-seed0)
    "$PYTHON_BIN" "$HERE/record_executor.py" \
      --run-root "$RUN_ROOT" --protocol "$PROTOCOL" \
      --output "$EXECUTOR_RECORD"
    ;;
  freeze-executor-audit-seed0) verify; runtime; dataset; freeze_executor_audit_seed0 ;;
  audit-executor-seed0) verify; runtime; gpu; dataset; audit_executor_seed0 ;;
  audit-executor-adaptive-reach-seed0)
    verify; runtime; gpu; dataset; audit_executor_adaptive_reach_seed0
    ;;
  finetune-executor-500k) verify; runtime; gpu; dataset; finetune_executor_500k ;;
  audit-finetuned-executor-500k)
    verify; runtime; gpu; dataset
    audit_finetuned_executor "$FINETUNE_PROTOCOL" "$FINETUNE_500_RECORD" "$FINETUNE_500_AUDIT" "$FINETUNE_ROOT/steps_0500000/gate.json"
    ;;
  finetune-executor-1m) verify; runtime; gpu; dataset; finetune_executor_1m ;;
  audit-finetuned-executor-1m)
    verify; runtime; gpu; dataset
    audit_finetuned_executor "$FINETUNE_PROTOCOL" "$FINETUNE_1000_RECORD" "$FINETUNE_1000_AUDIT" "$FINETUNE_ROOT/steps_1000000/gate.json"
    ;;
  finetune-mixed-executor-500k)
    verify; runtime; gpu; dataset; finetune_mixed_executor_500k
    ;;
  audit-mixed-finetuned-executor-500k)
    verify; runtime; gpu; dataset
    audit_finetuned_executor "$MIXED_FINETUNE_PROTOCOL" "$MIXED_FINETUNE_500_RECORD" "$MIXED_FINETUNE_500_AUDIT" "$MIXED_FINETUNE_ROOT/steps_0500000/gate.json"
    ;;
  finetune-mixed-executor-1m)
    verify; runtime; gpu; dataset; finetune_mixed_executor_1m
    ;;
  audit-mixed-finetuned-executor-1m)
    verify; runtime; gpu; dataset
    audit_finetuned_executor "$MIXED_FINETUNE_PROTOCOL" "$MIXED_FINETUNE_1000_RECORD" "$MIXED_FINETUNE_1000_AUDIT" "$MIXED_FINETUNE_ROOT/steps_1000000/gate.json"
    ;;
  *)
    echo "Usage: $0 {verify|runtime|gpu|download|preflight|train-executor-seed0|record-executor-seed0|freeze-executor-audit-seed0|audit-executor-seed0|audit-executor-adaptive-reach-seed0|finetune-executor-500k|audit-finetuned-executor-500k|finetune-executor-1m|audit-finetuned-executor-1m|finetune-mixed-executor-500k|audit-mixed-finetuned-executor-500k|finetune-mixed-executor-1m|audit-mixed-finetuned-executor-1m}" >&2
    exit 2
    ;;
esac
