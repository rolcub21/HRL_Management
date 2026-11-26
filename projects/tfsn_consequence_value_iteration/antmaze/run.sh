#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT="${E377_ROOT:-/home/ai_diagnosis/SG/e377}"
SN_ROOT="${SN_ROOT:-/home/ai_diagnosis/SG/sn}"
MANAGEMENT_ROOT="${MANAGEMENT_ROOT:-/home/ai_diagnosis/HRL_Management}"
PYTHON_BIN="${PYTHON_BIN:-/home/ai_diagnosis/SG/.venv/bin/python}"
DEVICE="${E600_DEVICE:-cuda}"
MODE="${1:-help}"
R48_FULL15_OUTPUT="${R48_FULL15_OUTPUT:-$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/r39_r46_rule_removal_r48_full15}"
R52_E549_OUTPUT="${R52_E549_OUTPUT:-$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/e549_5m_preparation_r52}"
R52_E549_SMOKE_OUTPUT="${R52_E549_SMOKE_OUTPUT:-$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/e549_5m_preparation_r52_smoke}"
R54_E549_10M_OUTPUT="${R54_E549_10M_OUTPUT:-$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/e549_10m_preparation_r54}"
R54_E549_10M_SMOKE_OUTPUT="${R54_E549_10M_SMOKE_OUTPUT:-$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/e549_10m_preparation_r54_smoke}"

cd "$SN_ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT"

run() {
  "$PYTHON_BIN" -m scripts.antmaze.current_tfsn.model_planning \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_recorded() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/recorded_transition_planning.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_oracle_diagnostic() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/oracle_contract_diagnostic.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_factorized_consequence() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/factorized_consequence.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r29_alignment() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r29_alignment.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r30_aligned_consequence() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/aligned_factorized_consequence_r30.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r31_aligned_hidden_goal() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r31_aligned_hidden_goal.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r32_consumptive_lookahead() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r32_consumptive_lookahead.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r33_posterior_sampling() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r33_posterior_sampled_acquisition.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r34_posterior_consistency() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r34_posterior_consistent_acquisition.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r35_detection_footprint() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r35_detection_footprint_acquisition.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r36_reachable_posterior() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r36_reachable_posterior_acquisition.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r37_exact_detection_posterior() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r37_exact_detection_posterior.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r38_route_posterior_mass() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r38_route_posterior_mass.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r39_live_posterior_passage() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r39_live_posterior_passage.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r40_continuous_episode_diagnostic() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r40_continuous_episode_diagnostic.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r41_stall_posture_replay() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r41_stall_posture_replay.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r42_posture_risk_audit() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r42_posture_risk_audit.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r43_orientation_factorized() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/orientation_factorized_consequence_r43.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r44_orientation_risk_audit() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r44_orientation_risk_audit.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r45_orientation_deployment() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r45_orientation_deployment.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r46_online_consequence_correction() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r46_online_consequence_correction.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r47_current_behavior_alignment() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r47_current_behavior_alignment.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r48_rule_free_r39() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r48_rule_free_r39.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r51_destination_arrival_correction() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r51_destination_arrival_correction.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r52_e549_preparation() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r52_e549_preparation.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r53_e549_r51_evaluation() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r53_e549_r51_evaluation.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r54_e549_10m_preparation() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r54_e549_10m_preparation.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r55_e549_10m_r51_evaluation() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r55_e549_10m_r51_evaluation.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r56_e549_supplied_goal() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r56_e549_supplied_goal.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r57_e549_task13_posture() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r57_e549_task13_posture_replay.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r58_e549_supplied_goal_all9() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r58_e549_supplied_goal_all9.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r59_e549_hidden_goal_all9() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r59_e549_hidden_goal_all9.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r60_e549_latent_rho_all9() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r60_e549_latent_rho_all9.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r61_e549_phase_separated_latent_rho() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r61_e549_phase_separated_latent_rho.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r62_e549_10m_phase_separated_latent_rho() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r62_e549_10m_phase_separated_latent_rho.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r63_e549_10m_fixed_sigma() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r63_e549_10m_fixed_sigma.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r67_e549_hub_routed_exploration() {
  cd "$MANAGEMENT_ROOT"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r67}" \
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r67_e549_hub_routed_exploration.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r68_e549_dynamic_hubs() {
  cd "$MANAGEMENT_ROOT"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r68}" \
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r68_e549_dynamic_hubs.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r69_e549_dynamic_hub_long_budget() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r69_e549_dynamic_hub_long_budget.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r70_corrected_hub_reachability() {
  cd "$MANAGEMENT_ROOT"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r70}" \
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r70_corrected_hub_reachability.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r71_sigma_generalized_correction() {
  cd "$MANAGEMENT_ROOT"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r71}" \
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r71_sigma_generalized_correction.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r72_hub_route_correction() {
  cd "$MANAGEMENT_ROOT"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r72}" \
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r72_hub_route_correction.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r73_fixed_sigma_hub_route() {
  cd "$MANAGEMENT_ROOT"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r73}" \
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r73_fixed_sigma_hub_route.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r74_reach_scale_route_correction() {
  cd "$MANAGEMENT_ROOT"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r74}" \
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r74_reach_scale_route_correction.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r75_r74_twenty_episode() {
  cd "$MANAGEMENT_ROOT"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r75}" \
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r75_r74_twenty_episode.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r76_return_route_correction() {
  cd "$MANAGEMENT_ROOT"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r76}" \
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r76_return_route_correction.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r77_online_map_growth() {
  cd "$MANAGEMENT_ROOT"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r77}" \
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r77_online_map_growth.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_r78_soft_return_route_audit() {
  cd "$MANAGEMENT_ROOT"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-r78}" \
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r78_soft_return_route_audit.py \
    "$@"
}

run_r48_passage_gain_cause_audit() {
  cd "$MANAGEMENT_ROOT"
  "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r48_passage_gain_cause_audit.py \
    --ant-evaluation "$R48_FULL15_OUTPUT/evaluation_all15" "$@"
}

run_r48_memory_scale_cause_audit() {
  cd "$MANAGEMENT_ROOT"
  "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/r48_memory_scale_cause_audit.py \
    --evaluation "$R48_FULL15_OUTPUT/evaluation_all15" "$@"
}

run_factorized_recorded() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/factorized_recorded_transition.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_mobility_stitched() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/mobility_stitched_planning.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_shared_actions() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/shared_action_planning.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_expanded() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/expanded_development.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_lazy_alias() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/lazy_alias_advance.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_lazy_alias_long3() {
  local pids=()
  local failed=0
  for task_id in 9 13 14; do
    run_lazy_alias task --task-id "$task_id" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "one or more lazy-alias long-route tasks failed" >&2
    return 1
  fi
  run_lazy_alias summarize
}

run_partitioned_alias() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/alias_partition_advance.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_partitioned_alias_long3() {
  local pids=()
  local failed=0
  for task_id in 9 13 14; do
    run_partitioned_alias task --task-id "$task_id" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "one or more partitioned-alias long-route tasks failed" >&2
    return 1
  fi
  run_partitioned_alias summarize
}

run_partitioned_alias_all15() {
  local failed=0
  local start
  for start in 0 3 6 9 12; do
    local pids=()
    local task_id
    for task_id in "$start" "$((start + 1))" "$((start + 2))"; do
      run_partitioned_alias task --task-id "$task_id" &
      pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    if [[ "$failed" -ne 0 ]]; then
      echo "one or more partitioned-alias development tasks failed" >&2
      return 1
    fi
  done
  run_partitioned_alias summarize-all
}

run_extended_budget() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/extended_budget_diagnostic.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_extended_budget_long3() {
  local pids=()
  local failed=0
  for task_id in 9 13 14; do
    run_extended_budget task --task-id "$task_id" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "one or more extended-budget long-route tasks failed" >&2
    return 1
  fi
  run_extended_budget summarize
}

run_bellman_consistent() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/bellman_consistent_planning.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_bellman_long3() {
  local task_id
  for task_id in 9 13 14; do
    run_bellman_consistent task --task-id "$task_id"
  done
  run_bellman_consistent summarize-long3
}

run_bellman_all15() {
  local task_id
  for task_id in $(seq 0 14); do
    run_bellman_consistent task --task-id "$task_id"
  done
  run_bellman_consistent summarize
}

run_integrated_bellman() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/integrated_bellman_planning.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_integrated_bellman_long3() {
  local task_id
  for task_id in 9 13 14; do
    run_integrated_bellman task --task-id "$task_id"
  done
  run_integrated_bellman summarize-long3
}

run_integrated_bellman_all15() {
  local task_id
  for task_id in $(seq 0 14); do
    run_integrated_bellman task --task-id "$task_id"
  done
  run_integrated_bellman summarize
}

run_first_occupancy() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/first_occupancy_bellman.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_first_occupancy_gate() {
  local task_id
  for task_id in 0 4 8; do
    run_first_occupancy task --task-id "$task_id"
  done
  run_first_occupancy summarize-gate
}

run_first_occupancy_all15() {
  local task_id
  for task_id in $(seq 0 14); do
    run_first_occupancy task --task-id "$task_id"
  done
  run_first_occupancy summarize
}

run_cached_first_occupancy() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/cached_first_occupancy.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_pessimistic_first_occupancy() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/pessimistic_first_occupancy.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_factorized_sigma_r14() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/factorized_sigma_r14.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_reach_based_r14() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/reach_based_r14.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_averaged_successor_r23() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/averaged_successor_r23.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_gpi_successor_r24() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/gpi_successor_r24.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_hidden_goal_r22() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/hidden_goal_r22.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_hidden_goal_passage_r26() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/hidden_goal_passage_r26.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_hidden_goal_empirical_passage_r27() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/hidden_goal_empirical_passage_r27.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_hidden_goal_bounded_consumption_r28() {
  cd "$MANAGEMENT_ROOT"
  PYTHONPATH="${PYTHONPATH:-}:$SN_ROOT:$MANAGEMENT_ROOT" \
    "$PYTHON_BIN" \
    projects/tfsn_consequence_value_iteration/antmaze/hidden_goal_bounded_consumption_r28.py \
    --root "$RUNTIME_ROOT" --device "$DEVICE" "$@"
}

run_expanded_arm() {
  local arm="$1"
  local pids=()
  local failed=0
  run_expanded prepare-panel
  for task_id in 9 10 11 12 13 14; do
    run_expanded "${arm}-task" --task-id "$task_id" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "one or more $arm extension tasks failed" >&2
    return 1
  fi
  run_expanded summarize-arm --arm "$arm"
}

case "$MODE" in
  prepare) run prepare --round 2 ;;
  evaluate) run evaluate --round 2 ;;
  all)
    run prepare --round 2
    run evaluate --round 2
    ;;
  prepare-recorded)
    run_recorded prepare --round 2
    ;;
  evaluate-recorded)
    run_recorded evaluate --round 2
    ;;
  all-recorded)
    run_recorded prepare --round 2
    run_recorded evaluate --round 2
    ;;
  oracle-fixed-contract)
    run_oracle_diagnostic oracle-contract
    ;;
  diagnose-oracle-commands)
    run_oracle_diagnostic model-diagnostic --round 2
    ;;
  oracle-contract-diagnostic)
    run_oracle_diagnostic oracle-contract
    run_oracle_diagnostic model-diagnostic --round 2
    ;;
  train-factorized-consequence)
    run_factorized_consequence train --round 2
    ;;
  diagnose-factorized-consequence)
    run_factorized_consequence diagnose --round 2
    ;;
  factorized-consequence-diagnostic)
    run_factorized_consequence train --round 2
    run_factorized_consequence diagnose --round 2
    ;;
  evaluate-factorized-recorded)
    run_factorized_recorded
    ;;
  prepare-stitched-mobility)
    run_mobility_stitched prepare-mobility
    ;;
  prepare-stitched-bridges)
    run_mobility_stitched prepare-bridges
    ;;
  evaluate-mobility-stitched)
    run_mobility_stitched evaluate
    ;;
  mobility-stitched-all)
    run_mobility_stitched prepare-mobility
    run_mobility_stitched prepare-bridges
    run_mobility_stitched evaluate
    ;;
  prepare-shared-actions)
    run_shared_actions prepare
    ;;
  evaluate-shared-actions)
    run_shared_actions evaluate
    ;;
  shared-actions-all)
    run_shared_actions prepare
    run_shared_actions evaluate
    ;;
  expanded-r10)
    run_expanded_arm r10
    ;;
  expanded-oracle)
    run_expanded_arm oracle
    ;;
  expanded-direct)
    run_expanded_arm direct
    ;;
  expanded-all)
    run_expanded_arm r10
    run_expanded_arm oracle
    run_expanded_arm direct
    run_expanded summarize-combined
    ;;
  summarize-expanded)
    run_expanded summarize-combined
    ;;
  lazy-alias-long3)
    run_lazy_alias_long3
    ;;
  partitioned-alias-long3)
    run_partitioned_alias_long3
    ;;
  partitioned-alias-all15)
    run_partitioned_alias_all15
    ;;
  extended-budget-long3)
    run_extended_budget_long3
    ;;
  bellman-long3)
    run_bellman_long3
    ;;
  bellman-all15)
    run_bellman_all15
    ;;
  integrated-bellman-long3)
    run_integrated_bellman_long3
    ;;
  integrated-bellman-all15)
    run_integrated_bellman_all15
    ;;
  first-occupancy-gate)
    run_first_occupancy_gate
    ;;
  first-occupancy-all15)
    run_first_occupancy_all15
    ;;
  cached-first-occupancy-gate)
    run_cached_first_occupancy evaluate-set --set-name gate
    ;;
  cached-first-occupancy-all15)
    run_cached_first_occupancy evaluate-set --set-name all15
    ;;
  pessimistic-first-occupancy-gate)
    run_pessimistic_first_occupancy evaluate-set --set-name gate
    ;;
  pessimistic-first-occupancy-all15)
    run_pessimistic_first_occupancy evaluate-set --set-name all15
    ;;
  factorized-sigma-r14-prepare)
    run_factorized_sigma_r14 prepare
    ;;
  factorized-sigma-r14-all15)
    run_factorized_sigma_r14 evaluate-all15
    ;;
  reach-based-r14-prepare)
    run_reach_based_r14 prepare
    ;;
  reach-based-r14-all15)
    run_reach_based_r14 evaluate-all15
    ;;
  averaged-successor-r23-contrast)
    run_averaged_successor_r23 reset-contrast
    ;;
  averaged-successor-r23-all15)
    run_averaged_successor_r23 evaluate-all15
    ;;
  gpi-successor-r24-prepare)
    run_gpi_successor_r24 prepare
    ;;
  gpi-successor-r24-all15)
    run_gpi_successor_r24 prepare
    run_gpi_successor_r24 evaluate-all15
    ;;
  hidden-r22-gate2)
    run_hidden_goal_r22 evaluate-set --set-name gate2 --episodes 5 \
      --output-name hidden_goal_frozen_r22_gate2
    ;;
  hidden-r22-all15)
    run_hidden_goal_r22 evaluate-set --set-name all15 --episodes 5 \
      --output-name hidden_goal_frozen_r22_all15
    ;;
  hidden-r22-passage-gate2)
    run_hidden_goal_passage_r26 prepare
    run_hidden_goal_passage_r26 evaluate-set --set-name gate2 --episodes 5 \
      --output-name hidden_goal_passage_r26_gate2
    ;;
  hidden-r22-passage-all15)
    run_hidden_goal_passage_r26 prepare
    run_hidden_goal_passage_r26 evaluate-set --set-name all15 --episodes 5 \
      --output-name hidden_goal_passage_r26_all15
    ;;
  hidden-r22-empirical-passage-gate2)
    run_hidden_goal_empirical_passage_r27 prepare
    run_hidden_goal_empirical_passage_r27 evaluate-set \
      --set-name gate2 --episodes 5 \
      --output-name hidden_goal_empirical_passage_r27_gate2
    ;;
  hidden-r22-empirical-passage-all15)
    run_hidden_goal_empirical_passage_r27 prepare
    run_hidden_goal_empirical_passage_r27 evaluate-set \
      --set-name all15 --episodes 5 \
      --output-name hidden_goal_empirical_passage_r27_all15
    ;;
  hidden-r22-bounded-search-gate2)
    run_hidden_goal_empirical_passage_r27 prepare
    run_hidden_goal_bounded_consumption_r28 evaluate-set \
      --set-name gate2 --episodes 5 \
      --output-name hidden_goal_bounded_consumption_r28_gate2
    ;;
  hidden-r22-bounded-search-all15)
    run_hidden_goal_empirical_passage_r27 prepare
    run_hidden_goal_bounded_consumption_r28 evaluate-set \
      --set-name all15 --episodes 5 \
      --output-name hidden_goal_bounded_consumption_r28_all15
    ;;
  r28-alignment-smoke)
    run_r29_alignment --profile smoke \
      --output "$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/r28_alignment_r29_smoke"
    ;;
  collect-r28-alignment)
    run_r29_alignment --profile full \
      --output "$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/r28_alignment_r29"
    ;;
  train-r29-aligned-consequence-smoke)
    run_r30_aligned_consequence --profile smoke \
      --output "$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/r29_aligned_consequence_r30_smoke"
    ;;
  train-r29-aligned-consequence)
    run_r30_aligned_consequence --profile full \
      --output "$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/r29_aligned_consequence_r30"
    ;;
  audit-r30-hidden-deployment)
    run_r31_aligned_hidden_goal audit
    ;;
  prepare-r30-hidden-graph)
    run_r31_aligned_hidden_goal prepare-graph
    ;;
  prepare-r30-hidden-mobility)
    run_r31_aligned_hidden_goal prepare-mobility
    ;;
  prepare-r30-hidden-bridges)
    run_r31_aligned_hidden_goal prepare-bridges
    ;;
  prepare-r30-hidden-passage)
    run_r31_aligned_hidden_goal prepare-passage
    ;;
  prepare-r30-hidden-all)
    run_r31_aligned_hidden_goal prepare-all
    ;;
  hidden-r30-gate2)
    run_r31_aligned_hidden_goal evaluate-set --set-name gate2 --episodes 5
    ;;
  hidden-r30-all15)
    run_r31_aligned_hidden_goal evaluate-set --set-name all15 --episodes 5
    ;;
  hidden-r32-consumptive-gate2)
    run_r32_consumptive_lookahead evaluate-gate2 --episodes 5
    ;;
  hidden-r33-posterior-gate2)
    run_r33_posterior_sampling audit
    run_r33_posterior_sampling evaluate-set --set-name gate2 --episodes 5
    ;;
  hidden-r33-posterior-all15)
    run_r33_posterior_sampling audit
    run_r33_posterior_sampling evaluate-set --set-name all15 --episodes 5
    ;;
  hidden-r34-posterior-consistent-gate2)
    run_r34_posterior_consistency audit
    run_r34_posterior_consistency evaluate-set --set-name gate2 --episodes 5
    ;;
  hidden-r34-posterior-consistent-all15)
    run_r34_posterior_consistency audit
    run_r34_posterior_consistency evaluate-set --set-name all15 --episodes 5
    ;;
  hidden-r35-detection-footprint-gate2)
    run_r35_detection_footprint audit
    run_r35_detection_footprint evaluate-set --set-name gate2 --episodes 5
    ;;
  hidden-r35-detection-footprint-all15)
    run_r35_detection_footprint audit
    run_r35_detection_footprint evaluate-set --set-name all15 --episodes 5
    ;;
  hidden-r36-reachable-posterior-gate2)
    run_r36_reachable_posterior audit
    run_r36_reachable_posterior evaluate-set --set-name gate2 --episodes 5
    ;;
  hidden-r36-reachable-posterior-all15)
    run_r36_reachable_posterior audit
    run_r36_reachable_posterior evaluate-set --set-name all15 --episodes 5
    ;;
  hidden-r36-reachable-posterior-all15-ep10)
    run_r36_reachable_posterior audit
    run_r36_reachable_posterior evaluate-set --set-name all15_ep10 --episodes 10
    ;;
  hidden-r37-exact-detection-gate2)
    run_r37_exact_detection_posterior audit
    run_r37_exact_detection_posterior evaluate-set --set-name gate2 --episodes 5
    ;;
  hidden-r37-exact-detection-all15)
    run_r37_exact_detection_posterior audit
    run_r37_exact_detection_posterior evaluate-set --set-name all15 --episodes 5
    ;;
  hidden-r38-route-posterior-mass-gate2)
    run_r38_route_posterior_mass audit
    run_r38_route_posterior_mass evaluate-set --set-name gate2 --episodes 5
    ;;
  hidden-r38-route-posterior-mass-all15)
    run_r38_route_posterior_mass audit
    run_r38_route_posterior_mass evaluate-set --set-name all15 --episodes 5
    ;;
  hidden-r39-live-posterior-passage-gate2)
    run_r39_live_posterior_passage audit
    run_r39_live_posterior_passage evaluate-set --set-name gate2 --episodes 5
    ;;
  hidden-r39-live-posterior-passage-all15)
    run_r39_live_posterior_passage audit
    run_r39_live_posterior_passage evaluate-set --set-name all15 --episodes 5
    ;;
  hidden-r39-continuous50-gate2)
    run_r40_continuous_episode_diagnostic audit
    run_r40_continuous_episode_diagnostic evaluate
    ;;
  diagnose-r40-task3-posture)
    run_r41_stall_posture_replay audit
    run_r41_stall_posture_replay replay
    ;;
  audit-r42-posture-risk)
    run_r42_posture_risk_audit audit
    run_r42_posture_risk_audit run
    ;;
  train-r43-orientation-factorized)
    run_r43_orientation_factorized audit
    run_r43_orientation_factorized --profile full train
    ;;
  audit-r44-orientation-risk)
    run_r44_orientation_risk_audit audit
    run_r44_orientation_risk_audit run
    ;;
  prepare-r45-orientation-bridges)
    run_r45_orientation_deployment audit
    run_r45_orientation_deployment prepare-bridges
    ;;
  hidden-r45-orientation-gate2)
    run_r45_orientation_deployment audit
    run_r45_orientation_deployment evaluate-gate2 --episodes 5
    ;;
  hidden-r46-online-correction-gate2)
    run_r46_online_consequence_correction audit
    run_r46_online_consequence_correction evaluate-gate2 --episodes 5
    ;;
  hidden-r46-online-correction-task3-audit)
    run_r46_online_consequence_correction audit
    run_r46_online_consequence_correction evaluate-single \
      --task-id 3 --episodes 5 --output-name audit_task003
    "$PYTHON_BIN" \
      projects/tfsn_consequence_value_iteration/antmaze/plot_r39_gate2_trajectories.py \
      --results "$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/online_consequence_correction_r46/audit_task003" \
      --output "$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/online_consequence_correction_r46/audit_task003/r46_task003_behavior.png"
    ;;
  collect-r47-current-alignment-smoke)
    run_r47_current_behavior_alignment --profile smoke \
      --output "$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/current_behavior_alignment_r47_smoke" audit
    run_r47_current_behavior_alignment --profile smoke \
      --output "$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/current_behavior_alignment_r47_smoke" collect
    ;;
  collect-r47-current-alignment)
    run_r47_current_behavior_alignment --profile full \
      --output "$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/current_behavior_alignment_r47" audit
    run_r47_current_behavior_alignment --profile full \
      --output "$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/current_behavior_alignment_r47" collect
    ;;
  audit-r48-rule-removal)
    run_r48_rule_free_r39 audit
    ;;
  prepare-r48-rule-removal-graph)
    run_r48_rule_free_r39 audit
    run_r48_rule_free_r39 prepare-graph
    ;;
  prepare-r48-rule-removal-mobility)
    run_r48_rule_free_r39 audit
    run_r48_rule_free_r39 prepare-mobility
    ;;
  prepare-r48-rule-removal-bridges)
    run_r48_rule_free_r39 audit
    run_r48_rule_free_r39 prepare-bridges
    ;;
  prepare-r48-rule-removal-passage)
    run_r48_rule_free_r39 audit
    run_r48_rule_free_r39 prepare-passage
    ;;
  prepare-r48-rule-removal-all)
    run_r48_rule_free_r39 audit
    run_r48_rule_free_r39 prepare-all
    ;;
  hidden-r48-rule-removal-gate2)
    run_r48_rule_free_r39 audit
    run_r48_rule_free_r39 evaluate-gate2 --episodes 5
    ;;
  prepare-r48-rule-removal-full15)
    run_r48_rule_free_r39 --output "$R48_FULL15_OUTPUT" audit
    run_r48_rule_free_r39 --output "$R48_FULL15_OUTPUT" prepare-all
    ;;
  audit-r48-passage-matching-full15)
    run_r48_rule_free_r39 --output "$R48_FULL15_OUTPUT" audit
    run_r48_rule_free_r39 --output "$R48_FULL15_OUTPUT" audit-passage-matching
    ;;
  audit-r48-passage-gain-cause-full15)
    run_r48_passage_gain_cause_audit
    ;;
  audit-r48-memory-scale-full15)
    run_r48_memory_scale_cause_audit
    ;;
  hidden-r48-rule-removal-all15)
    run_r48_rule_free_r39 --output "$R48_FULL15_OUTPUT" audit
    run_r48_rule_free_r39 --output "$R48_FULL15_OUTPUT" \
      evaluate-set --set-name all15 --episodes 5
    ;;
  audit-r51-destination-arrival)
    run_r51_destination_arrival_correction audit
    ;;
  hidden-r51-destination-arrival-task3)
    run_r51_destination_arrival_correction audit
    run_r51_destination_arrival_correction evaluate-task3 --episodes 5
    ;;
  hidden-r51-destination-arrival-stalled-pair)
    run_r51_destination_arrival_correction audit
    run_r51_destination_arrival_correction \
      --output "$MANAGEMENT_ROOT/projects/tfsn_consequence_value_iteration/results/antmaze/destination_arrival_correction_r51/audit_stalled_pair_farama" \
      evaluate-stalled-pair --episodes 5
    ;;
  audit-r52-e549-preparation)
    run_r52_e549_preparation --output "$R52_E549_OUTPUT" audit
    ;;
  smoke-r52-e549-preparation)
    run_r52_e549_preparation --profile smoke \
      --output "$R52_E549_SMOKE_OUTPUT" audit
    run_r52_e549_preparation --profile smoke \
      --output "$R52_E549_SMOKE_OUTPUT" collect
    run_r52_e549_preparation --profile smoke \
      --output "$R52_E549_SMOKE_OUTPUT" train-consequence
    run_r52_e549_preparation --profile smoke \
      --output "$R52_E549_SMOKE_OUTPUT" prepare-graph
    ;;
  collect-r52-e549-preparation)
    run_r52_e549_preparation --profile full \
      --output "$R52_E549_OUTPUT" audit
    run_r52_e549_preparation --profile full \
      --output "$R52_E549_OUTPUT" collect
    ;;
  train-r52-e549-consequence)
    run_r52_e549_preparation --profile full \
      --output "$R52_E549_OUTPUT" train-consequence
    ;;
  prepare-r52-e549-graph)
    run_r52_e549_preparation --profile full \
      --output "$R52_E549_OUTPUT" prepare-graph
    ;;
  prepare-r52-e549-mobility)
    run_r52_e549_preparation --profile full \
      --output "$R52_E549_OUTPUT" prepare-mobility
    ;;
  prepare-r52-e549-bridges)
    run_r52_e549_preparation --profile full \
      --output "$R52_E549_OUTPUT" prepare-bridges
    ;;
  prepare-r52-e549-passage)
    run_r52_e549_preparation --profile full \
      --output "$R52_E549_OUTPUT" prepare-passage
    ;;
  prepare-r52-e549-runtime)
    run_r52_e549_preparation --profile full \
      --output "$R52_E549_OUTPUT" prepare-all
    ;;
  hidden-r53-e549-r51-stalled-pair)
    run_r53_e549_r51_evaluation \
      --preparation "$R52_E549_OUTPUT" audit
    run_r53_e549_r51_evaluation \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  audit-r54-e549-10m-preparation)
    run_r54_e549_10m_preparation \
      --output "$R54_E549_10M_OUTPUT" audit
    ;;
  smoke-r54-e549-10m-preparation)
    run_r54_e549_10m_preparation --profile smoke \
      --output "$R54_E549_10M_SMOKE_OUTPUT" audit
    run_r54_e549_10m_preparation --profile smoke \
      --output "$R54_E549_10M_SMOKE_OUTPUT" collect
    run_r54_e549_10m_preparation --profile smoke \
      --output "$R54_E549_10M_SMOKE_OUTPUT" train-consequence
    run_r54_e549_10m_preparation --profile smoke \
      --output "$R54_E549_10M_SMOKE_OUTPUT" prepare-graph
    ;;
  collect-r54-e549-10m-preparation)
    run_r54_e549_10m_preparation --profile full \
      --output "$R54_E549_10M_OUTPUT" audit
    run_r54_e549_10m_preparation --profile full \
      --output "$R54_E549_10M_OUTPUT" collect
    ;;
  train-r54-e549-10m-consequence)
    run_r54_e549_10m_preparation --profile full \
      --output "$R54_E549_10M_OUTPUT" train-consequence
    ;;
  prepare-r54-e549-10m-runtime)
    run_r54_e549_10m_preparation --profile full \
      --output "$R54_E549_10M_OUTPUT" prepare-all
    ;;
  hidden-r55-e549-10m-r51-stalled-pair)
    run_r55_e549_10m_r51_evaluation \
      --preparation "$R54_E549_10M_OUTPUT" audit
    run_r55_e549_10m_r51_evaluation \
      --preparation "$R54_E549_10M_OUTPUT" evaluate
    ;;
  supplied-r56-e549-r51-stalled-pair)
    run_r56_e549_supplied_goal \
      --preparation "$R52_E549_OUTPUT" audit
    run_r56_e549_supplied_goal \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  replay-r57-e549-task13-posture)
    run_r57_e549_task13_posture audit
    run_r57_e549_task13_posture replay
    ;;
  supplied-r58-e549-r51-original9)
    run_r58_e549_supplied_goal_all9 \
      --preparation "$R52_E549_OUTPUT" audit
    run_r58_e549_supplied_goal_all9 \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  hidden-r59-e549-r51-original9)
    run_r59_e549_hidden_goal_all9 \
      --preparation "$R52_E549_OUTPUT" audit
    run_r59_e549_hidden_goal_all9 \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  hidden-r60-e549-r51-latent-rho-original9)
    run_r60_e549_latent_rho_all9 \
      --preparation "$R52_E549_OUTPUT" audit
    run_r60_e549_latent_rho_all9 \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  hidden-r61-e549-r51-phase-separated-latent-rho-original9)
    run_r61_e549_phase_separated_latent_rho \
      --preparation "$R52_E549_OUTPUT" audit
    run_r61_e549_phase_separated_latent_rho \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  hidden-r62-e549-10m-r51-phase-separated-latent-rho-original9)
    run_r62_e549_10m_phase_separated_latent_rho \
      --preparation "$R54_E549_10M_OUTPUT" audit
    run_r62_e549_10m_phase_separated_latent_rho \
      --preparation "$R54_E549_10M_OUTPUT" evaluate
    ;;
  hidden-r63-e549-10m-r51-fixed-sigma-original9)
    run_r63_e549_10m_fixed_sigma \
      --preparation "$R54_E549_10M_OUTPUT" audit
    run_r63_e549_10m_fixed_sigma \
      --preparation "$R54_E549_10M_OUTPUT" evaluate
    ;;
  timing-r67-e549-5m-hub-routed-tasks-1-2-8)
    run_r67_e549_hub_routed_exploration \
      --preparation "$R52_E549_OUTPUT" audit
    run_r67_e549_hub_routed_exploration \
      --preparation "$R52_E549_OUTPUT" timing
    ;;
  hidden-r67-e549-5m-hub-routed-tasks-1-2-8)
    run_r67_e549_hub_routed_exploration \
      --preparation "$R52_E549_OUTPUT" audit
    run_r67_e549_hub_routed_exploration \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  prepare-r68-e549-5m-dynamic-hubs)
    run_r68_e549_dynamic_hubs \
      --preparation "$R52_E549_OUTPUT" audit
    run_r68_e549_dynamic_hubs \
      --preparation "$R52_E549_OUTPUT" prepare
    ;;
  timing-r68-e549-5m-dynamic-hubs)
    run_r68_e549_dynamic_hubs \
      --preparation "$R52_E549_OUTPUT" timing
    ;;
  hidden-r68-e549-5m-dynamic-hubs-tasks-1-2-8)
    run_r68_e549_dynamic_hubs \
      --preparation "$R52_E549_OUTPUT" audit
    run_r68_e549_dynamic_hubs \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  prepare-r69-e549-5m-dynamic-hubs-long-budget)
    run_r69_e549_dynamic_hub_long_budget \
      --preparation "$R52_E549_OUTPUT" audit
    run_r69_e549_dynamic_hub_long_budget \
      --preparation "$R52_E549_OUTPUT" prepare
    ;;
  hidden-r69-e549-5m-dynamic-hubs-long-budget-tasks-1-2-8)
    run_r69_e549_dynamic_hub_long_budget \
      --preparation "$R52_E549_OUTPUT" audit
    run_r69_e549_dynamic_hub_long_budget \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  plot-r69-e549-5m-dynamic-hubs-long-budget-tasks-1-2-8)
    run_r69_e549_dynamic_hub_long_budget \
      --preparation "$R52_E549_OUTPUT" plot
    ;;
  replay-r70-e549-5m-corrected-hub-reachability-task8)
    run_r70_corrected_hub_reachability \
      --preparation "$R52_E549_OUTPUT" audit
    run_r70_corrected_hub_reachability \
      --preparation "$R52_E549_OUTPUT" run
    ;;
  audit-r71-e549-5m-sigma-generalized-correction-original9-long-budget)
    run_r71_sigma_generalized_correction \
      --preparation "$R52_E549_OUTPUT" audit
    ;;
  hidden-r71-e549-5m-sigma-generalized-correction-original9-long-budget)
    run_r71_sigma_generalized_correction \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  audit-r72-e549-5m-hub-route-correction-original9-long-budget)
    run_r72_hub_route_correction \
      --preparation "$R52_E549_OUTPUT" audit
    ;;
  equivalence-r72-e549-5m-hub-route-correction-original9-long-budget)
    run_r72_hub_route_correction \
      --preparation "$R52_E549_OUTPUT" equivalence
    ;;
  hidden-r72-e549-5m-hub-route-correction-original9-long-budget)
    run_r72_hub_route_correction \
      --preparation "$R52_E549_OUTPUT" equivalence
    run_r72_hub_route_correction \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  audit-r73-e549-5m-fixed-sigma-hub-route-original9-long-budget)
    run_r73_fixed_sigma_hub_route \
      --preparation "$R52_E549_OUTPUT" audit
    ;;
  hidden-r73-e549-5m-fixed-sigma-hub-route-original9-long-budget)
    run_r73_fixed_sigma_hub_route \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  audit-r74-e549-5m-reach-scale-route-original9-long-budget)
    run_r74_reach_scale_route_correction \
      --preparation "$R52_E549_OUTPUT" audit
    ;;
  hidden-r74-e549-5m-reach-scale-route-original9-long-budget)
    run_r74_reach_scale_route_correction \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  audit-r75-e549-5m-r74-original9-twenty-episodes)
    run_r75_r74_twenty_episode \
      --preparation "$R52_E549_OUTPUT" audit
    ;;
  hidden-r75-e549-5m-r74-original9-twenty-episodes)
    run_r75_r74_twenty_episode \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  audit-r76-e549-5m-task6-return-route)
    run_r76_return_route_correction \
      --preparation "$R52_E549_OUTPUT" audit
    ;;
  replay-r76-e549-5m-task6-return-route)
    run_r76_return_route_correction \
      --preparation "$R52_E549_OUTPUT" audit
    run_r76_return_route_correction \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  audit-r77-e549-5m-task6-online-map-return)
    run_r77_online_map_growth \
      --preparation "$R52_E549_OUTPUT" audit
    ;;
  replay-r77-e549-5m-task6-online-map-return)
    run_r77_online_map_growth \
      --preparation "$R52_E549_OUTPUT" audit
    run_r77_online_map_growth \
      --preparation "$R52_E549_OUTPUT" evaluate
    ;;
  audit-r78-e549-5m-task6-soft-return-route)
    run_r78_soft_return_route_audit \
      --preparation "$R52_E549_OUTPUT" audit
    ;;
  *)
    echo "usage: $0 {...|audit-r52-e549-preparation|smoke-r52-e549-preparation|collect-r52-e549-preparation|train-r52-e549-consequence|prepare-r52-e549-{graph,mobility,bridges,passage,runtime}|hidden-r53-e549-r51-stalled-pair|audit-r54-e549-10m-preparation|smoke-r54-e549-10m-preparation|collect-r54-e549-10m-preparation|train-r54-e549-10m-consequence|prepare-r54-e549-10m-runtime|hidden-r55-e549-10m-r51-stalled-pair|supplied-r56-e549-r51-stalled-pair|replay-r57-e549-task13-posture|supplied-r58-e549-r51-original9|hidden-r59-e549-r51-original9|hidden-r60-e549-r51-latent-rho-original9|hidden-r61-e549-r51-phase-separated-latent-rho-original9|hidden-r62-e549-10m-r51-phase-separated-latent-rho-original9|hidden-r63-e549-10m-r51-fixed-sigma-original9|timing-r67-e549-5m-hub-routed-tasks-1-2-8|hidden-r67-e549-5m-hub-routed-tasks-1-2-8|prepare-r68-e549-5m-dynamic-hubs|timing-r68-e549-5m-dynamic-hubs|hidden-r68-e549-5m-dynamic-hubs-tasks-1-2-8|prepare-r69-e549-5m-dynamic-hubs-long-budget|hidden-r69-e549-5m-dynamic-hubs-long-budget-tasks-1-2-8|plot-r69-e549-5m-dynamic-hubs-long-budget-tasks-1-2-8|replay-r70-e549-5m-corrected-hub-reachability-task8|audit-r71-e549-5m-sigma-generalized-correction-original9-long-budget|hidden-r71-e549-5m-sigma-generalized-correction-original9-long-budget|audit-r72-e549-5m-hub-route-correction-original9-long-budget|equivalence-r72-e549-5m-hub-route-correction-original9-long-budget|hidden-r72-e549-5m-hub-route-correction-original9-long-budget|audit-r73-e549-5m-fixed-sigma-hub-route-original9-long-budget|hidden-r73-e549-5m-fixed-sigma-hub-route-original9-long-budget|audit-r74-e549-5m-reach-scale-route-original9-long-budget|hidden-r74-e549-5m-reach-scale-route-original9-long-budget|audit-r75-e549-5m-r74-original9-twenty-episodes|hidden-r75-e549-5m-r74-original9-twenty-episodes|audit-r76-e549-5m-task6-return-route|replay-r76-e549-5m-task6-return-route|audit-r77-e549-5m-task6-online-map-return|replay-r77-e549-5m-task6-online-map-return|audit-r78-e549-5m-task6-soft-return-route}" >&2
    exit 2
    ;;
esac
