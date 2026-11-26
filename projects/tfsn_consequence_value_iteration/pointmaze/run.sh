#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ai_diagnosis/SG/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
MODE="${1:-help}"
RESULTS="$ROOT/projects/tfsn_consequence_value_iteration/results/pointmaze"

cd "$ROOT"
export PYTHONPATH="$ROOT:/home/ai_diagnosis/SG/sn:${PYTHONPATH:-}"

reference_audit() {
  "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.shared.manifest_audit \
    --domain pointmaze
  "$PYTHON_BIN" -m unittest \
    projects.tfsn_consequence_value_iteration.shared.test_shared \
    projects.tfsn_consequence_value_iteration.pointmaze.test_pointmaze
}

case "$MODE" in
  freeze-fresh-confirmation-panel)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_fresh_confirmation_panel
    ;;
  collect-bootstrap-seed1|collect-bootstrap-seed2)
    preparation_seed="${MODE#collect-bootstrap-seed}"
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.collect_multiseed_bootstrap \
      --preparation-seed "$preparation_seed" --profile full --device "$DEVICE" \
      --output "$RESULTS/independent_multiseed_bootstrap_r1/seed${preparation_seed}/full"
    ;;
  collect-bootstrap-seed1-smoke|collect-bootstrap-seed2-smoke)
    preparation_seed="${MODE#collect-bootstrap-seed}"
    preparation_seed="${preparation_seed%-smoke}"
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.collect_multiseed_bootstrap \
      --preparation-seed "$preparation_seed" --profile smoke --device "$DEVICE" \
      --output "$RESULTS/independent_multiseed_bootstrap_r1/seed${preparation_seed}/smoke"
    ;;
  audit)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.audit \
      --output "$RESULTS/part_a_audit.json"
    ;;
  train-factorized-smoke)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.audit \
      --output "$RESULTS/part_a_audit.json"
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.train_factorized \
      --schedule smoke --device "$DEVICE" \
      --output "$RESULTS/initial_factorized_r1_smoke"
    ;;
  train-factorized)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.audit \
      --output "$RESULTS/part_a_audit.json"
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.train_factorized \
      --schedule full --device "$DEVICE" \
      --output "$RESULTS/initial_factorized_r1"
    ;;
  references-smoke)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.audit \
      --output "$RESULTS/part_a_audit.json"
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.references \
      --device "$DEVICE" --max-tasks 1 --max-episodes-per-task 1 \
      --output "$RESULTS/development_references_r1_smoke/summary.json"
    ;;
  references-development)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.audit \
      --output "$RESULTS/part_a_audit.json"
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.references \
      --device "$DEVICE" \
      --output "$RESULTS/development_references_r1/summary.json"
    ;;
  consequence-acceptance)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.acceptance \
      --device "$DEVICE" \
      --references "$RESULTS/development_references_r1/summary.json" \
      --models "$RESULTS/initial_factorized_r1" \
      --output "$RESULTS/initial_factorized_r1/oracle_acceptance.json"
    ;;
  prepare-roadmap-smoke)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_roadmap \
      --schedule smoke \
      --models "$RESULTS/initial_factorized_r1_smoke" \
      --output "$RESULTS/initial_roadmap_r1_smoke"
    ;;
  prepare-roadmap)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_roadmap \
      --schedule full \
      --models "$RESULTS/initial_factorized_r1" \
      --output "$RESULTS/initial_roadmap_r1"
    ;;
  prepare-planner-smoke)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges \
      all --schedule smoke --device "$DEVICE" \
      --models "$RESULTS/initial_factorized_r1_smoke" \
      --roadmap "$RESULTS/initial_roadmap_r1_smoke" \
      --output "$RESULTS/initial_planner_r1_smoke"
    ;;
  prepare-mobility)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges \
      mobility --schedule full --device "$DEVICE" \
      --models "$RESULTS/initial_factorized_r1" \
      --roadmap "$RESULTS/initial_roadmap_r1" \
      --output "$RESULTS/initial_planner_r1"
    ;;
  prepare-bridges)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges \
      bridges --schedule full --device "$DEVICE" \
      --models "$RESULTS/initial_factorized_r1" \
      --roadmap "$RESULTS/initial_roadmap_r1" \
      --output "$RESULTS/initial_planner_r1"
    ;;
  collect-alignment-smoke)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.collect_alignment \
      --phase both --profile smoke --device "$DEVICE" \
      --models "$RESULTS/initial_factorized_r1_smoke" \
      --roadmap "$RESULTS/initial_roadmap_r1_smoke" \
      --planner "$RESULTS/initial_planner_r1_smoke" \
      --output "$RESULTS/current_planner_alignment_r2_smoke"
    ;;
  collect-alignment-full-gate)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.collect_alignment \
      --phase both --profile smoke --device "$DEVICE" \
      --models "$RESULTS/initial_factorized_r1" \
      --roadmap "$RESULTS/initial_roadmap_r1" \
      --planner "$RESULTS/initial_planner_r1" \
      --output "$RESULTS/current_planner_alignment_r2_full_gate"
    ;;
  collect-alignment-acquisition)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.collect_alignment \
      --phase acquisition --profile full --device "$DEVICE" \
      --models "$RESULTS/initial_factorized_r1" \
      --roadmap "$RESULTS/initial_roadmap_r1" \
      --planner "$RESULTS/initial_planner_r1" \
      --output "$RESULTS/current_planner_alignment_r2"
    ;;
  collect-alignment-return)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.collect_alignment \
      --phase return --profile full --device "$DEVICE" \
      --models "$RESULTS/initial_factorized_r1" \
      --roadmap "$RESULTS/initial_roadmap_r1" \
      --planner "$RESULTS/initial_planner_r1" \
      --output "$RESULTS/current_planner_alignment_r2"
    ;;
  train-aligned-factorized-smoke)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.train_aligned_factorized \
      --schedule smoke --device "$DEVICE" \
      --alignment "$RESULTS/current_planner_alignment_r2" \
      --base-models "$RESULTS/initial_factorized_r1" \
      --output "$RESULTS/current_planner_aligned_factorized_r3_smoke"
    ;;
  train-aligned-factorized)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.train_aligned_factorized \
      --schedule full --device "$DEVICE" \
      --alignment "$RESULTS/current_planner_alignment_r2" \
      --base-models "$RESULTS/initial_factorized_r1" \
      --output "$RESULTS/current_planner_aligned_factorized_r3"
    ;;
  train-density-aligned-factorized-smoke)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.train_aligned_factorized \
      --schedule smoke --device "$DEVICE" --situation-density-weighting \
      --alignment "$RESULTS/current_planner_alignment_r2" \
      --base-models "$RESULTS/initial_factorized_r1" \
      --output "$RESULTS/situation_density_aligned_factorized_r6_smoke"
    ;;
  train-density-aligned-factorized)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.train_aligned_factorized \
      --schedule full --device "$DEVICE" --situation-density-weighting \
      --alignment "$RESULTS/current_planner_alignment_r2" \
      --base-models "$RESULTS/initial_factorized_r1" \
      --output "$RESULTS/situation_density_aligned_factorized_r6"
    ;;
  density-consequence-acceptance)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.acceptance \
      --device "$DEVICE" \
      --references "$RESULTS/development_references_r1/summary.json" \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --output "$RESULTS/situation_density_aligned_factorized_r6/oracle_acceptance.json"
    ;;
  prepare-movement-roadmap-smoke)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_roadmap \
      --schedule smoke --movement-conditioned-reach \
      --models "$RESULTS/situation_density_aligned_factorized_r6_smoke" \
      --alignment "$RESULTS/current_planner_alignment_r2" \
      --output "$RESULTS/movement_conditioned_reach_roadmap_r7_smoke"
    ;;
  prepare-movement-roadmap)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_roadmap \
      --schedule full --movement-conditioned-reach \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --alignment "$RESULTS/current_planner_alignment_r2" \
      --output "$RESULTS/movement_conditioned_reach_roadmap_r7"
    ;;
  prepare-movement-planner-smoke)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges \
      all --schedule smoke --device "$DEVICE" \
      --models "$RESULTS/situation_density_aligned_factorized_r6_smoke" \
      --roadmap "$RESULTS/movement_conditioned_reach_roadmap_r7_smoke" \
      --output "$RESULTS/movement_conditioned_reach_planner_r7_smoke"
    ;;
  prepare-movement-planner)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges \
      all --schedule full --device "$DEVICE" \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/movement_conditioned_reach_roadmap_r7" \
      --output "$RESULTS/movement_conditioned_reach_planner_r7"
    ;;
  prepare-movement-mobility)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges \
      mobility --schedule full --device "$DEVICE" \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/movement_conditioned_reach_roadmap_r7" \
      --output "$RESULTS/movement_conditioned_reach_planner_r7"
    ;;
  prepare-movement-bridges)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges \
      bridges --schedule full --device "$DEVICE" \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/movement_conditioned_reach_roadmap_r7" \
      --output "$RESULTS/movement_conditioned_reach_planner_r7"
    ;;
  evaluate-movement-development-gate2)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --advance-weighting \
      --max-tasks 2 --max-episodes-per-task 5 \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/movement_conditioned_reach_roadmap_r7" \
      --planner "$RESULTS/movement_conditioned_reach_planner_r7" \
      --output "$RESULTS/density_movement_reach_development_r8_gate2"
    ;;
  evaluate-movement-development)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --advance-weighting \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/movement_conditioned_reach_roadmap_r7" \
      --planner "$RESULTS/movement_conditioned_reach_planner_r7" \
      --output "$RESULTS/density_movement_reach_development_r8"
    ;;
  evaluate-detection-footprint-misses)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --advance-weighting \
      --acquisition-detection-radius-from-evaluator \
      --goal-name spatial_diagnostic_native_goal_r01_c06 \
      --goal-name spatial_diagnostic_native_goal_r06_c06 \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/movement_conditioned_reach_roadmap_r7" \
      --planner "$RESULTS/movement_conditioned_reach_planner_r7" \
      --output "$RESULTS/exact_detection_footprint_misses_r9"
    ;;
  evaluate-detection-footprint-development)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --advance-weighting \
      --acquisition-detection-radius-from-evaluator \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/movement_conditioned_reach_roadmap_r7" \
      --planner "$RESULTS/movement_conditioned_reach_planner_r7" \
      --output "$RESULTS/exact_detection_footprint_development_r9"
    ;;
  evaluate-consumptive-footprint-misses)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --advance-weighting \
      --acquisition-detection-radius-from-evaluator \
      --consumptive-lookahead \
      --goal-name spatial_diagnostic_native_goal_r01_c06 \
      --goal-name spatial_diagnostic_native_goal_r06_c06 \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/movement_conditioned_reach_roadmap_r7" \
      --planner "$RESULTS/movement_conditioned_reach_planner_r7" \
      --output "$RESULTS/exact_consumptive_lookahead_misses_r10"
    ;;
  evaluate-consumptive-footprint-gate2)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --advance-weighting \
      --acquisition-detection-radius-from-evaluator \
      --consumptive-lookahead --max-episodes-per-task 3 \
      --goal-name spatial_diagnostic_native_goal_r01_c06 \
      --goal-name spatial_diagnostic_native_goal_r06_c06 \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/movement_conditioned_reach_roadmap_r7" \
      --planner "$RESULTS/movement_conditioned_reach_planner_r7" \
      --output "$RESULTS/exact_consumptive_lookahead_gate2_r10"
    ;;
  evaluate-consumptive-footprint-r01-c06)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --advance-weighting \
      --acquisition-detection-radius-from-evaluator \
      --consumptive-lookahead \
      --goal-name spatial_diagnostic_native_goal_r01_c06 \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/movement_conditioned_reach_roadmap_r7" \
      --planner "$RESULTS/movement_conditioned_reach_planner_r7" \
      --output "$RESULTS/exact_consumptive_lookahead_r01_c06_r10"
    ;;
  evaluate-consumptive-footprint-development)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --advance-weighting \
      --acquisition-detection-radius-from-evaluator \
      --consumptive-lookahead \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/movement_conditioned_reach_roadmap_r7" \
      --planner "$RESULTS/movement_conditioned_reach_planner_r7" \
      --output "$RESULTS/exact_consumptive_lookahead_development_r10"
    ;;
  prepare-aligned-roadmap-smoke)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_roadmap \
      --schedule smoke \
      --models "$RESULTS/current_planner_aligned_factorized_r3_smoke" \
      --alignment "$RESULTS/current_planner_alignment_r2" \
      --output "$RESULTS/current_planner_aligned_roadmap_r4_smoke"
    ;;
  prepare-aligned-planner-smoke)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges \
      all --schedule smoke --device "$DEVICE" \
      --models "$RESULTS/current_planner_aligned_factorized_r3_smoke" \
      --roadmap "$RESULTS/current_planner_aligned_roadmap_r4_smoke" \
      --output "$RESULTS/current_planner_aligned_planner_r4_smoke"
    ;;
  collect-alignment-round2-smoke)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.collect_alignment \
      --phase both --profile smoke --alignment-round 2 --seed 2201 \
      --device "$DEVICE" \
      --models "$RESULTS/current_planner_aligned_factorized_r3_smoke" \
      --roadmap "$RESULTS/current_planner_aligned_roadmap_r4_smoke" \
      --planner "$RESULTS/current_planner_aligned_planner_r4_smoke" \
      --output "$RESULTS/current_planner_alignment_round2_r4_smoke"
    ;;
  prepare-aligned-roadmap)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_roadmap \
      --schedule full \
      --models "$RESULTS/current_planner_aligned_factorized_r3" \
      --alignment "$RESULTS/current_planner_alignment_r2" \
      --output "$RESULTS/current_planner_aligned_roadmap_r4"
    ;;
  prepare-aligned-mobility)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges \
      mobility --schedule full --device "$DEVICE" \
      --models "$RESULTS/current_planner_aligned_factorized_r3" \
      --roadmap "$RESULTS/current_planner_aligned_roadmap_r4" \
      --output "$RESULTS/current_planner_aligned_planner_r4"
    ;;
  prepare-aligned-bridges)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges \
      bridges --schedule full --device "$DEVICE" \
      --models "$RESULTS/current_planner_aligned_factorized_r3" \
      --roadmap "$RESULTS/current_planner_aligned_roadmap_r4" \
      --output "$RESULTS/current_planner_aligned_planner_r4"
    ;;
  evaluate-aligned-development-gate2)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --max-tasks 2 --max-episodes-per-task 5 \
      --models "$RESULTS/current_planner_aligned_factorized_r3" \
      --roadmap "$RESULTS/current_planner_aligned_roadmap_r4" \
      --planner "$RESULTS/current_planner_aligned_planner_r4" \
      --output "$RESULTS/current_planner_aligned_development_r5_gate2"
    ;;
  evaluate-aligned-development)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" \
      --models "$RESULTS/current_planner_aligned_factorized_r3" \
      --roadmap "$RESULTS/current_planner_aligned_roadmap_r4" \
      --planner "$RESULTS/current_planner_aligned_planner_r4" \
      --output "$RESULTS/current_planner_aligned_development_r5"
    ;;
  collect-alignment-round2-acquisition)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.collect_alignment \
      --phase acquisition --profile full --alignment-round 2 --seed 2201 \
      --device "$DEVICE" \
      --models "$RESULTS/current_planner_aligned_factorized_r3" \
      --roadmap "$RESULTS/current_planner_aligned_roadmap_r4" \
      --planner "$RESULTS/current_planner_aligned_planner_r4" \
      --output "$RESULTS/current_planner_alignment_round2_r4"
    ;;
  collect-alignment-round2-return)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.collect_alignment \
      --phase return --profile full --alignment-round 2 --seed 2201 \
      --device "$DEVICE" \
      --models "$RESULTS/current_planner_aligned_factorized_r3" \
      --roadmap "$RESULTS/current_planner_aligned_roadmap_r4" \
      --planner "$RESULTS/current_planner_aligned_planner_r4" \
      --output "$RESULTS/current_planner_alignment_round2_r4"
    ;;
  prepare-reference-roadmap-smoke)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_roadmap \
      --schedule smoke --continuous-reach-weighting \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --alignment "$RESULTS/current_planner_alignment_r2" \
      --output "$RESULTS/continuous_reference_roadmap_r11_smoke"
    ;;
  prepare-reference-planner-smoke)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges \
      all --schedule smoke --device "$DEVICE" \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/continuous_reference_roadmap_r11_smoke" \
      --output "$RESULTS/continuous_reference_planner_r11_smoke"
    ;;
  evaluate-reference-smoke)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --reference-method \
      --max-tasks 1 --max-episodes-per-task 1 \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/continuous_reference_roadmap_r11_smoke" \
      --planner "$RESULTS/continuous_reference_planner_r11_smoke" \
      --output "$RESULTS/continuous_reference_development_r11_smoke"
    ;;
  prepare-reference-roadmap)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_roadmap \
      --schedule full --continuous-reach-weighting \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --alignment "$RESULTS/current_planner_alignment_r2" \
      --output "$RESULTS/continuous_reference_roadmap_r11"
    ;;
  prepare-reference-planner)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.prepare_bridges \
      all --schedule full --device "$DEVICE" \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/continuous_reference_roadmap_r11" \
      --output "$RESULTS/continuous_reference_planner_r11"
    ;;
  evaluate-reference-gate2)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --reference-method \
      --max-tasks 2 --max-episodes-per-task 3 \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/continuous_reference_roadmap_r11" \
      --planner "$RESULTS/continuous_reference_planner_r11" \
      --output "$RESULTS/continuous_reference_development_r11_gate2"
    ;;
  evaluate-reference-development)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --reference-method \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/continuous_reference_roadmap_r11" \
      --planner "$RESULTS/continuous_reference_planner_r11" \
      --output "$RESULTS/continuous_reference_development_r11"
    ;;
  evaluate-reference-hidden-development15)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --reference-method \
      --panel-protocol "$ROOT/projects/tfsn_consequence_value_iteration/pointmaze/hidden_goal_protocol_panel15.json" \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/continuous_reference_roadmap_r11" \
      --planner "$RESULTS/continuous_reference_planner_r11" \
      --output "$RESULTS/continuous_reference_hidden_goal_r11_disjoint15_e120"
    ;;
  evaluate-reference-fresh-confirmation-hidden-seed0)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_development \
      --device "$DEVICE" --reference-method \
      --panel-protocol "$ROOT/projects/tfsn_consequence_value_iteration/pointmaze/hidden_goal_protocol_fresh_confirmation_seed0.json" \
      --models "$RESULTS/situation_density_aligned_factorized_r6" \
      --roadmap "$RESULTS/continuous_reference_roadmap_r11" \
      --planner "$RESULTS/continuous_reference_planner_r11" \
      --output "$RESULTS/fresh_confirmation_r11_seed0/hidden"
    ;;
  evaluate-supplied-smoke)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_supplied_goal \
      --environment-seed 40 --max-goals 1 --max-episodes 1 \
      --device "$DEVICE" \
      --output "$RESULTS/continuous_reference_supplied_goal_r11_smoke_development72_protocol/seed40"
    ;;
  evaluate-supplied-development)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_supplied_goal \
      --environment-seed 41 --development-subset --device "$DEVICE" \
      --output "$RESULTS/continuous_reference_supplied_goal_r11_development72"
    ;;
  evaluate-supplied-development15)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_supplied_goal \
      --protocol "$ROOT/projects/tfsn_consequence_value_iteration/pointmaze/supplied_goal_protocol_panel15.json" \
      --environment-seed 41 --development-subset --device "$DEVICE" \
      --output "$RESULTS/continuous_reference_supplied_goal_r11_disjoint15_e120"
    ;;
  evaluate-reference-fresh-confirmation-supplied-seed0)
    reference_audit
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_supplied_goal \
      --protocol "$ROOT/projects/tfsn_consequence_value_iteration/pointmaze/supplied_goal_protocol_fresh_confirmation_seed0.json" \
      --environment-seed 61 --device "$DEVICE" \
      --output "$RESULTS/fresh_confirmation_r11_seed0/supplied"
    ;;
  evaluate-supplied-seed40|evaluate-supplied-seed41|evaluate-supplied-seed42|evaluate-supplied-seed43|evaluate-supplied-seed44)
    reference_audit
    environment_seed="${MODE#evaluate-supplied-seed}"
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_supplied_goal \
      --environment-seed "$environment_seed" --device "$DEVICE" \
      --output "$RESULTS/continuous_reference_supplied_goal_r11/seed${environment_seed}"
    ;;
  evaluate-supplied-full)
    reference_audit
    for environment_seed in 40 41 42 43 44; do
      "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.evaluate_supplied_goal \
        --environment-seed "$environment_seed" --device "$DEVICE" \
        --output "$RESULTS/continuous_reference_supplied_goal_r11/seed${environment_seed}"
    done
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.summarize_supplied_goal \
      --root "$RESULTS/continuous_reference_supplied_goal_r11"
    ;;
  summarize-supplied)
    "$PYTHON_BIN" -m projects.tfsn_consequence_value_iteration.pointmaze.summarize_supplied_goal \
      --root "$RESULTS/continuous_reference_supplied_goal_r11"
    ;;
  *)
    echo "usage: $0 {freeze-fresh-confirmation-panel|collect-bootstrap-seed1|collect-bootstrap-seed2|audit|prepare-reference-roadmap-smoke|prepare-reference-planner-smoke|evaluate-reference-smoke|prepare-reference-roadmap|prepare-reference-planner|evaluate-reference-gate2|evaluate-reference-development|evaluate-reference-hidden-development15|evaluate-reference-fresh-confirmation-hidden-seed0|evaluate-reference-fresh-confirmation-supplied-seed0|evaluate-supplied-smoke|evaluate-supplied-development|evaluate-supplied-development15|evaluate-supplied-seed40|evaluate-supplied-seed41|evaluate-supplied-seed42|evaluate-supplied-seed43|evaluate-supplied-seed44|evaluate-supplied-full|summarize-supplied|...historical modes...}" >&2
    exit 2
    ;;
esac
