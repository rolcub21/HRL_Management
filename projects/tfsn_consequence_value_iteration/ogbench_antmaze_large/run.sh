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
AUDIT="$OUT/audit"
RUN_ROOT="$OUT/executor/seed0/reference_runs"
GROUP="TF_SN_R74_OGBench_AntMazeLarge_CRL_seed0"

export PYTHONPATH="$IMPLS:$VENDOR:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OGBENCH_IMPLS_ROOT="$IMPLS"
export OGBENCH_DATASET_DIR="$DATASET_DIR"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

verify() {
  "$STATIC_PYTHON" -m json.tool "$PROTOCOL" >/dev/null
  "$STATIC_PYTHON" -m py_compile "$HERE/preflight.py" "$HERE/run_official_main.py" "$HERE/record_executor.py"
  bash -n "$HERE/run.sh"
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
    --output "$OUT/executor/seed0/executor.json"
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
      --output "$OUT/executor/seed0/executor.json"
    ;;
  *)
    echo "Usage: $0 {verify|runtime|gpu|download|preflight|train-executor-seed0|record-executor-seed0}" >&2
    exit 2
    ;;
esac
