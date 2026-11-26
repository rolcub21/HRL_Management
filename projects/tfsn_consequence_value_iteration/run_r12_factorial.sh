#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ai_diagnosis/SG/.venv/bin/python}"
DEVICE="${E600_DEVICE:-cpu}"
RUNTIME_ROOT="${E377_ROOT:-/home/ai_diagnosis/SG/e377}"
MODE="${1:-help}"
PROJECT="$ROOT/projects/tfsn_consequence_value_iteration"
RESULTS="$PROJECT/results/latent_detection_passage_r12"
ANT_BASE="$PROJECT/results/antmaze/r39_r46_rule_removal_r48_full15"
ANT_LATENT8_OUTPUT="$PROJECT/results/antmaze/latent_radius_nearest_miss8_r50"
MIN_TRAIN_FREE_KIB=$((5 * 1024 * 1024))

export PYTHONPATH="$ROOT:/home/ai_diagnosis/SG/sn:${PYTHONPATH:-}"
cd "$ROOT"

log() {
  printf '[r49] %s\n' "$*"
}

require_training_space() {
  local available_kib
  available_kib="$(df -Pk "$PROJECT" | awk 'NR == 2 {print $4}')"
  if [[ ! "$available_kib" =~ ^[0-9]+$ ]]; then
    log "could not determine free filesystem capacity; refusing expensive training"
    exit 1
  fi
  if (( available_kib < MIN_TRAIN_FREE_KIB )); then
    log "only $((available_kib / 1024)) MiB free; passage training requires at least 5 GiB free"
    exit 1
  fi
  log "filesystem capacity check passed: $((available_kib / 1024 / 1024)) GiB free"
}

train_ant() {
  require_training_space
  mkdir -p "$RESULTS/models/antmaze"
  exec 9>"$RESULTS/models/antmaze/.train.lock"
  if ! flock -n 9; then
    log "another AntMaze passage trainer is already running; refusing a duplicate"
    exit 1
  fi
  log "training AntMaze passage model (validation early stopping enabled)"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.shared.prepare_r12_passage \
    --domain antmaze --device "$DEVICE" \
    --graph "$ANT_BASE/recorded_transition_graph.npz" \
    --output "$RESULTS/models/antmaze"
  log "passage training complete"
}

preflight() {
  log "running AntMaze artifact and method preflight"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.shared.r12_preflight \
    --ant-graph "$ANT_BASE/recorded_transition_graph.npz" \
    --ant-passage-metadata "$RESULTS/models/antmaze/passage.json" \
    --output "$RESULTS/preflight.json"
  log "preflight complete"
}

evaluate_ant() {
  local observation passage
  for observation in sigma_kernel latent_radius; do
    for passage in nearest_recorded shared_learned; do
      log "evaluating arm observation=$observation passage=$passage"
      extra=()
      if [[ "$passage" == "shared_learned" ]]; then
        extra=(--passage-artifact "$RESULTS/models/antmaze/passage.pt")
      fi
      "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.antmaze.r49_latent_passage_factorial \
        --root "$RUNTIME_ROOT" --base-output "$ANT_BASE" \
        --output "$RESULTS/antmaze" --device "$DEVICE" evaluate \
        --observation-level "$observation" --passage-level "$passage" \
        "${extra[@]}"
      log "completed arm observation=$observation passage=$passage"
    done
  done
}

evaluate_ant_latent8_misses() {
  log "auditing AntMaze latent-radius nearest-passage miss extension"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.antmaze.r50_latent8_miss_extension audit
  log "evaluating AntMaze tasks 0 and 2 for eight episodes each"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.antmaze.r50_latent8_miss_extension \
    --root "$RUNTIME_ROOT" --base-output "$ANT_BASE" \
    --output "$ANT_LATENT8_OUTPUT" --device "$DEVICE" evaluate \
    --observation-level latent_radius --passage-level nearest_recorded \
    --episodes 8
  log "AntMaze latent-radius eight-episode miss extension complete"
}

summarize() {
  log "summarizing AntMaze 2x2"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.shared.summarize_r12_factorial \
    --root "$RESULTS" --output "$RESULTS/summary"
  log "summary complete"
}

case "$MODE" in
  audit-code)
    "$PYTHON_BIN" -m unittest \
      projects.tfsn_consequence_value_iteration.shared.test_r12_latent_passage
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.antmaze.r49_latent_passage_factorial audit
    ;;
  train-passage)
    train_ant
    ;;
  train-passages)
    log "mode 'train-passages' was retired with PointMaze; use 'train-passage'"
    exit 2
    ;;
  preflight)
    preflight
    ;;
  evaluate-2x2)
    preflight
    evaluate_ant
    summarize
    ;;
  evaluate-ant-latent8-misses)
    evaluate_ant_latent8_misses
    ;;
  summarize)
    summarize
    ;;
  *)
    echo "usage: $0 {audit-code|train-passage|preflight|evaluate-2x2|evaluate-ant-latent8-misses|summarize}" >&2
    exit 2
    ;;
esac
