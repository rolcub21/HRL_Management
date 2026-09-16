import copy
import hashlib
import unittest

import vcg_v2_1_policy_realization_analysis as analysis


def _source_audit():
    source = {
        "audit_protocol": analysis.PROTOCOL,
        "development_only": True,
        "cannot_create_or_select_checkpoint": True,
        "source_checkpoint_sha256": analysis.FROZEN_CHECKPOINT_SHA256,
        "source_training_contract_sha256": analysis.FROZEN_TRAINING_CONTRACT_SHA256,
        "source_checkpoint_role": analysis.FROZEN_CHECKPOINT_ROLE,
        "source_checkpoint_eligible": False,
        "source_completed_episodes": 100,
        "source_q_local_sha256": "a" * 64,
        "complete_case_filtering_used": False,
        "panel": {
            "episode_instance_seeds": analysis.FROZEN_INSTANCE_SEEDS,
            "reused_already_open_v2_1_development_validation_panel": True,
            "prospective_83xxx_panel_opened": False,
            "sealed_or_new_panel_opened": False,
        },
        "arms": {
            "map": {"runs_per_instance": 1, "policy_rng_used": False},
            "induced_soft": {
                "runs_per_instance": analysis.SOFT_REPLICATES_PER_INSTANCE,
                "replicate_indices": tuple(
                    range(analysis.SOFT_REPLICATES_PER_INSTANCE)
                ),
            },
        },
        "frozen_policy": {
            "dual_lambda": analysis.FROZEN_DUAL_LAMBDA,
            "within_group_temperatures": analysis.FROZEN_WITHIN_GROUP_TEMPERATURES,
            "group_temperature": analysis.FROZEN_GROUP_TEMPERATURE,
            "epsilon": 0.0,
            "latest_q_local_weights_only": True,
            "same_weights_lambda_and_temperatures_across_arms": True,
            "option_evaluation_mode": True,
            "evaluation_learning": False,
            "fresh_agent_clone_per_row": True,
            "fresh_certificate_cache_per_row": True,
            "replay_forbidden": True,
            "optimizer_step_forbidden": True,
            "target_update_forbidden": True,
        },
    }
    source["audit_contract_sha256"] = analysis._canonical_sha256(source)
    return source


def _manifest():
    records = {}
    for index, seed in enumerate(analysis.FROZEN_INSTANCE_SEEDS):
        records[str(seed)] = {
            "instance_index": index,
            "instance_seed": seed,
            "instance_id": f"instance-{seed}",
            "schedule_id": f"schedule-{seed}",
            "number_blocks": 8,
            "serialized_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
        }
    manifest = {
        "schema_version": 1,
        "audit_protocol": analysis.PROTOCOL,
        "source_training_contract_sha256": analysis.FROZEN_TRAINING_CONTRACT_SHA256,
        "episode_instance_seeds": analysis.FROZEN_INSTANCE_SEEDS,
        "reused_development_panel": True,
        "prospective_83xxx_panel_opened": False,
        "instances": records,
    }
    manifest["manifest_sha256"] = analysis._canonical_sha256(manifest)
    return manifest


def _row(*, policy, instance_index, replicate=None):
    seed = analysis.FROZEN_INSTANCE_SEEDS[instance_index]
    manifest = _manifest()["instances"][str(seed)]
    map_returns = (-100.0, -110.0, -117.99)
    map_mae = (30.0, 32.0, 34.0)
    map_holds = (12, 12, 13)
    if policy == analysis.MAP_POLICY:
        rng_index = None
        rng_seed = seed
        rng_used = False
        dense_return = map_returns[instance_index]
        mae = map_mae[instance_index]
        holds = map_holds[instance_index]
        reconfigures = 0
        physical = 0
    else:
        rng_index = replicate
        rng_seed = analysis.expected_soft_policy_rng_seed(instance_index, replicate)
        rng_used = True
        centered = replicate - 15.5
        dense_return = map_returns[instance_index] + 20.0 + centered / 10.0
        mae = map_mae[instance_index] - 2.0 + centered / 100.0
        holds = 4
        reconfigures = 1
        physical = replicate % 2
    actions = {
        "accept": 8,
        "deliver": 8,
        "defer": holds,
        "reconfigure": reconfigures,
    }
    macros = sum(actions.values())
    return {
        "audit_protocol": analysis.PROTOCOL,
        "audit_schema_version": 1,
        "policy": policy,
        "policy_realization": (
            "deterministic_nested_lagrangian_map"
            if policy == analysis.MAP_POLICY
            else "induced_nested_regularized_lagrangian_sample"
        ),
        "instance_index": instance_index,
        "instance_seed": seed,
        "instance_id": manifest["instance_id"],
        "schedule_id": manifest["schedule_id"],
        "episode_instance_serialized_sha256": manifest["serialized_sha256"],
        "policy_rng_index": rng_index,
        "policy_rng_seed": rng_seed,
        "policy_rng_used": rng_used,
        "policy_rng_scope": "action_sampling_only",
        "source_checkpoint_sha256": analysis.FROZEN_CHECKPOINT_SHA256,
        "source_checkpoint_role": analysis.FROZEN_CHECKPOINT_ROLE,
        "source_q_local_sha256": "a" * 64,
        "source_training_contract_sha256": analysis.FROZEN_TRAINING_CONTRACT_SHA256,
        "policy_schedule_protocol": analysis.FROZEN_POLICY_SCHEDULE_PROTOCOL,
        "schedule_episode_number": 100,
        "schedule_block_number": 10,
        "schedule_phase": "low_temperature_stabilization",
        "schedule_policy_mode": (
            "map" if policy == analysis.MAP_POLICY else "regularized_sample"
        ),
        "controller_select_internal_training_flag": (
            policy == analysis.INDUCED_SOFT_POLICY
        ),
        "stochastic_selection_only": policy == analysis.INDUCED_SOFT_POLICY,
        "dual_lambda": analysis.FROZEN_DUAL_LAMBDA,
        "within_group_temperatures": analysis.FROZEN_WITHIN_GROUP_TEMPERATURES,
        "group_temperature": analysis.FROZEN_GROUP_TEMPERATURE,
        "epsilon": 0.0,
        "strict_method_success": True,
        "success": True,
        "completion_rate": 1.0,
        "delivery_count": 8,
        "required_deliveries": 8,
        "dense_return": dense_return,
        "mean_absolute_error": mae,
        "delivery_deviations": (mae,) * 8,
        "steps": 200 + (0 if replicate is None else replicate),
        "macro_decisions": macros,
        "physical_rehandles": physical,
        "physical_rehandles_per_100_required_deliveries": 12.5 * physical,
        "selected_action_counts": actions,
        "hold_decisions": holds,
        "hold_decision_share": holds / macros,
        "reconfigure_decisions": reconfigures,
        "reconfigure_decision_share": reconfigures / macros,
        "method_failure_reason": None,
        "illegal_drops": 0,
        "fallbacks": 0,
        "witness_mismatches": 0,
        "all_selected_candidates_exact_safe": True,
        "policy_diagnostic_distribution": analysis.FROZEN_POLICY_DIAGNOSTIC_DISTRIBUTION,
        "policy_diagnostic_decisions": macros,
        "mean_outer_policy_entropy": 0.5,
        "mean_outer_map_probability": 0.7,
        "mean_selected_within_policy_entropy": 0.4,
        "mean_selected_within_map_probability": 0.8,
        "option_evaluation_mode": True,
        "evaluation_learning": False,
        "q_local_training_mode": False,
        "loss_updates": 0,
        "replay_size_before": 0,
        "replay_size_after": 0,
        "gradient_steps_before": 0,
        "gradient_steps_after": 0,
        "target_updates_before": 0,
        "target_updates_after": 0,
        "optimizer_state_entries_before": 0,
        "optimizer_state_entries_after": 0,
        "clone_learning_state_unchanged": True,
        "source_agent_unchanged": True,
        "training_agent_unchanged": True,
        "source_checkpoint_file_unchanged": True,
        "policy_frozen": True,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "fresh_certificate_cache": True,
        "exact_safe_frontier_authoritative": True,
    }


def _rows():
    rows = []
    for instance_index in range(3):
        rows.append(_row(policy=analysis.MAP_POLICY, instance_index=instance_index))
        rows.extend(
            _row(
                policy=analysis.INDUCED_SOFT_POLICY,
                instance_index=instance_index,
                replicate=replicate,
            )
            for replicate in range(analysis.SOFT_REPLICATES_PER_INSTANCE)
        )
    return rows


def _analyze(rows):
    return analysis.analyze_policy_realization(
        rows, source_audit=_source_audit(), instance_manifest=_manifest()
    )


class PolicyRealizationAnalysisTests(unittest.TestCase):
    def test_rng_namespace_is_exact_and_instance_disjoint(self):
        self.assertEqual(analysis.expected_soft_policy_rng_seed(0, 0), 910_000)
        self.assertEqual(analysis.expected_soft_policy_rng_seed(1, 0), 910_100)
        self.assertEqual(analysis.expected_soft_policy_rng_seed(2, 31), 910_231)

    def test_valid_grid_uses_conditional_bundle_inference(self):
        result = _analyze(_rows())
        self.assertEqual(result["design"]["total_rows"], 99)
        self.assertTrue(result["integrity"]["strict_integrity_gate"])
        self.assertTrue(result["integrity"]["map_source_reproduction_gate"])
        returned = result["metrics"]["dense_return"]
        self.assertAlmostEqual(returned["soft_minus_map"], 20.0)
        self.assertEqual(returned["mc_degrees_of_freedom"], 31)
        self.assertTrue(returned["conditional_mc_only"])
        self.assertTrue(result["budget"]["authenticated_budget_feasibility_gate"])
        self.assertTrue(result["conclusion"]["authenticated_comparison"])
        self.assertEqual(
            result["conclusion"]["dense_return"],
            "conditional_soft_improvement_supported",
        )
        self.assertEqual(
            result["conclusion"]["mean_absolute_error"],
            "conditional_soft_improvement_supported",
        )

    def test_missing_or_duplicate_row_fails_exact_grid(self):
        rows = _rows()
        with self.assertRaisesRegex(analysis.PolicyRealizationAnalysisError, "99 rows"):
            _analyze(rows[:-1])
        rows[-1] = copy.deepcopy(rows[-2])
        with self.assertRaisesRegex(analysis.PolicyRealizationAnalysisError, "duplicate"):
            _analyze(rows)

    def test_swapped_rng_seed_and_index_is_rejected(self):
        rows = _rows()
        soft = [row for row in rows if row["policy"] == analysis.INDUCED_SOFT_POLICY]
        soft[0]["policy_rng_seed"], soft[1]["policy_rng_seed"] = (
            soft[1]["policy_rng_seed"],
            soft[0]["policy_rng_seed"],
        )
        with self.assertRaisesRegex(analysis.PolicyRealizationAnalysisError, "RNG seed mismatch"):
            _analyze(rows)

    def test_manifest_and_source_authentication_are_required(self):
        manifest = _manifest()
        manifest["instances"]["84000"]["schedule_id"] = "tampered"
        with self.assertRaisesRegex(analysis.PolicyRealizationAnalysisError, "manifest SHA"):
            analysis.analyze_policy_realization(
                _rows(), source_audit=_source_audit(), instance_manifest=manifest
            )
        source = _source_audit()
        source["source_checkpoint_eligible"] = True
        source["audit_contract_sha256"] = analysis._canonical_sha256(
            {
                key: value
                for key, value in source.items()
                if key != "audit_contract_sha256"
            }
        )
        with self.assertRaisesRegex(analysis.PolicyRealizationAnalysisError, "eligible"):
            analysis.analyze_policy_realization(
                _rows(), source_audit=source, instance_manifest=_manifest()
            )

    def test_unsafe_row_is_retained_and_invalidates_authentication(self):
        rows = _rows()
        rows[1]["strict_method_success"] = False
        rows[1]["success"] = False
        rows[1]["method_failure_reason"] = "synthetic_failure"
        result = _analyze(rows)
        self.assertFalse(result["integrity"]["strict_integrity_gate"])
        self.assertEqual(result["integrity"]["failure_count"], 1)
        self.assertTrue(result["integrity"]["all_99_rows_retained"])
        self.assertTrue(result["metrics"]["dense_return"]["estimable"])
        self.assertFalse(result["conclusion"]["authenticated_comparison"])

    def test_missing_mae_never_triggers_complete_case_filtering(self):
        rows = _rows()
        row = rows[1]
        row["strict_method_success"] = False
        row["success"] = False
        row["completion_rate"] = 0.0
        row["delivery_count"] = 0
        row["delivery_deviations"] = ()
        row["mean_absolute_error"] = None
        row["method_failure_reason"] = "synthetic_failure"
        result = _analyze(rows)
        self.assertFalse(result["metrics"]["mean_absolute_error"]["estimable"])
        self.assertFalse(result["conclusion"]["authenticated_comparison"])
        self.assertTrue(result["design"]["no_complete_case_filtering"])

    def test_budget_ucb_can_fail_when_point_estimate_passes(self):
        rows = _rows()
        for row in rows:
            if row["policy"] == analysis.INDUCED_SOFT_POLICY:
                physical = 6 if row["policy_rng_index"] < 6 else 0
                row["physical_rehandles"] = physical
                row["physical_rehandles_per_100_required_deliveries"] = 12.5 * physical
        result = _analyze(rows)
        self.assertTrue(result["budget"]["soft_point_gate"])
        self.assertFalse(result["budget"]["soft_upper_bound_gate"])
        self.assertFalse(result["budget"]["authenticated_budget_feasibility_gate"])

    def test_map_reproduction_failure_is_fail_closed(self):
        rows = _rows()
        rows[0]["dense_return"] += 1.0
        result = _analyze(rows)
        self.assertFalse(result["integrity"]["map_source_reproduction_gate"])
        self.assertFalse(result["integrity"]["strict_integrity_gate"])
        self.assertFalse(result["conclusion"]["authenticated_comparison"])

    def test_action_shares_pool_counts_over_macro_decisions(self):
        result = _analyze(_rows())
        mapped = result["action_realization"][analysis.MAP_POLICY]
        self.assertEqual(mapped["counts"]["defer"], 37)
        self.assertEqual(mapped["macro_decisions"], 85)
        self.assertAlmostEqual(mapped["decision_weighted_shares"]["defer"], 37 / 85)


if __name__ == "__main__":
    unittest.main()
