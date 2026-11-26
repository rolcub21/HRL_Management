#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ai_diagnosis/SG/.venv/bin/python}"
DEVICE="${E600_DEVICE:-cpu}"
MODE="${1:-help}"
PROJECT="$ROOT/projects/tfsn_consequence_value_iteration"
BASE="$PROJECT/results/pointmaze"
MODELS="$BASE/situation_density_aligned_factorized_r6"
ROADMAP="$BASE/continuous_reference_roadmap_r11"
PLANNER="$BASE/continuous_reference_planner_r11"
ALIGNMENT="$BASE/current_planner_alignment_r2"
RESULTS="$BASE/latent_detection_passage_factorial_r12"
LATENT5_RESULTS="$BASE/latent_radius_five_command_smoke_r12"
LATENT8_RESULTS="$BASE/latent_radius_nearest_corner8_r12"
REGION_RESULTS="$BASE/command_region_corner8_r13"
R14_RESULTS="$BASE/sigma_vs_latent_passage_gain_goal4_r14"
PASSAGE="$RESULTS/models/passage.pt"
MIN_TRAIN_FREE_KIB=$((5 * 1024 * 1024))

export PYTHONPATH="$ROOT:/home/ai_diagnosis/SG/sn:${PYTHONPATH:-}"
cd "$ROOT"

log() {
  printf '[point-r12] %s\n' "$*"
}

require_training_space() {
  local available_kib
  available_kib="$(df -Pk "$PROJECT" | awk 'NR == 2 {print $4}')"
  if [[ ! "$available_kib" =~ ^[0-9]+$ ]]; then
    log "could not determine free capacity; refusing expensive training"
    exit 1
  fi
  if (( available_kib < MIN_TRAIN_FREE_KIB )); then
    log "only $((available_kib / 1024)) MiB free; training requires at least 5 GiB"
    exit 1
  fi
  log "capacity check passed: $((available_kib / 1024 / 1024)) GiB free"
}

train_passage() {
  require_training_space
  mkdir -p "$RESULTS/models"
  exec 9>"$RESULTS/models/.train.lock"
  if ! flock -n 9; then
    log "another PointMaze passage trainer is running; refusing a duplicate"
    exit 1
  fi
  log "training shared passage class on PointMaze reward-free data"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_factorial_passage_r12 \
    --roadmap "$ROADMAP" --alignment "$ALIGNMENT" \
    --output "$RESULTS/models" --device "$DEVICE"
  log "passage training complete"
}

preflight() {
  log "running isolated PointMaze preflight"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.preflight_factorial_r12 \
    --models "$MODELS" --roadmap "$ROADMAP" --planner "$PLANNER" \
    --passage-metadata "$RESULTS/models/passage.json" \
    --output "$RESULTS/preflight.json"
  log "preflight complete"
}

evaluate() {
  local observation passage
  for observation in sigma_kernel latent_radius; do
    for passage in nearest_recorded shared_learned; do
      extra=()
      if [[ "$passage" == "shared_learned" ]]; then
        extra=(--passage-artifact "$PASSAGE")
      fi
      log "evaluating observation=$observation passage=$passage"
      "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_factorial_r12 \
        --models "$MODELS" --roadmap "$ROADMAP" --planner "$PLANNER" \
        --output "$RESULTS/evaluation" --device "$DEVICE" \
        --observation-level "$observation" --passage-level "$passage" \
        "${extra[@]}"
      log "completed observation=$observation passage=$passage"
    done
  done
}

summarize() {
  log "summarizing PointMaze 2x2"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.summarize_factorial_r12 \
    --root "$RESULTS/evaluation" --output "$RESULTS/summary"
  log "summary complete"
}

evaluate_latent5() {
  log "running isolated latent-radius nearest-passage five-command smoke"
  mkdir -p "$LATENT5_RESULTS"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.preflight_factorial_r12 \
    --models "$MODELS" --roadmap "$ROADMAP" --planner "$PLANNER" \
    --passage-metadata "$RESULTS/models/passage.json" \
    --output "$LATENT5_RESULTS/preflight.json"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_latent5_r12 \
    --models "$MODELS" --roadmap "$ROADMAP" --planner "$PLANNER" \
    --output "$LATENT5_RESULTS/evaluation" --device "$DEVICE" \
    --observation-level latent_radius --passage-level nearest_recorded \
    --command-cap 5
  log "latent-radius five-command smoke complete"
}

evaluate_latent8_corners() {
  log "running latent-radius nearest-passage eight-episode corner extension"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_latent8_corners_r12 \
    --models "$MODELS" --roadmap "$ROADMAP" --planner "$PLANNER" \
    --output "$LATENT8_RESULTS/evaluation" --device "$DEVICE" \
    --observation-level latent_radius --passage-level nearest_recorded \
    --command-cap 20
  log "latent-radius eight-episode corner extension complete"
}

evaluate_command_regions() {
  log "auditing data-derived PointMaze command regions"
  mkdir -p "$REGION_RESULTS"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.audit_command_regions_r13 \
    --roadmap "$ROADMAP" --planner "$PLANNER" \
    --output "$REGION_RESULTS/preflight.json"
  log "running latent-radius command-region corner screen"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_command_regions_r13 \
    --models "$MODELS" --roadmap "$ROADMAP" --planner "$PLANNER" \
    --output "$REGION_RESULTS/evaluation" --device "$DEVICE" \
    --observation-level latent_radius --passage-level command_region \
    --command-cap 20
  log "command-region corner screen complete"
}

evaluate_sigma_latent5_goal4() {
  local observation
  log "preflighting fixed-sigma versus latent-radius five-command comparison"
  mkdir -p "$R14_RESULTS"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.preflight_sigma_latent5_goal4_r14 \
    --models "$MODELS" --roadmap "$ROADMAP" --planner "$PLANNER" \
    --output "$R14_RESULTS/preflight.json"
  for observation in sigma_kernel latent_radius; do
    log "R14 observation=$observation passage=nearest_recorded goals=4 episodes=8 commands=5"
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_sigma_latent5_goal4_r14 \
      --models "$MODELS" --roadmap "$ROADMAP" --planner "$PLANNER" \
      --output "$R14_RESULTS/evaluation" --device "$DEVICE" \
      --observation-level "$observation" --passage-level nearest_recorded \
      --command-cap 5
    log "R14 completed observation=$observation"
  done
  log "summarizing R14 memory comparison"
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.summarize_sigma_latent5_goal4_r14 \
    --root "$R14_RESULTS/evaluation" --output "$R14_RESULTS/summary"
  log "R14 comparison complete"
}

case "$MODE" in
  audit-code)
    "$PYTHON_BIN" -m unittest \
      projects.tfsn_consequence_value_iteration.shared.test_r12_latent_passage \
      projects.tfsn_consequence_value_iteration.shared.test_command_regions
    "$PYTHON_BIN" -m unittest \
      projects.tfsn_consequence_value_iteration.pointmaze.test_pointmaze \
      projects.tfsn_consequence_value_iteration.pointmaze.test_factorial_r12 \
      projects.tfsn_consequence_value_iteration.pointmaze.test_command_regions_r13 \
      projects.tfsn_consequence_value_iteration.pointmaze.test_sigma_latent5_goal4_r14
    ;;
  train-passage)
    train_passage
    ;;
  preflight)
    preflight
    ;;
  evaluate-2x2)
    preflight
    evaluate
    summarize
    ;;
  evaluate-latent5)
    evaluate_latent5
    ;;
  evaluate-latent8-corners)
    evaluate_latent8_corners
    ;;
  evaluate-command-regions)
    evaluate_command_regions
    ;;
  evaluate-sigma-latent5-goal4)
    evaluate_sigma_latent5_goal4
    ;;
  summarize)
    summarize
    ;;
  *)
    echo "usage: $0 {audit-code|train-passage|preflight|evaluate-2x2|evaluate-latent5|evaluate-latent8-corners|evaluate-command-regions|evaluate-sigma-latent5-goal4|summarize}" >&2
    exit 2
    ;;
esac
