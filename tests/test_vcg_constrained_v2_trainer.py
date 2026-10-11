import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from PSLAP.viability_candidates_hold_v2 import CERTIFIED_HOLD_INTERFACE_V2
from train_vcg_constrained_v2 import (
    BACKUP_VERSION,
    CHECKPOINT_FAMILY,
    CONTROLLER_ARCHITECTURE,
    COST_DEFINITION,
    ConstrainedV2TrainerError,
    DualConfig,
    ProjectedEpisodeDual,
    build_checkpoint_metadata,
    build_parser,
    build_training_contract,
    episode_budget_residual,
    load_default_runtime,
    normalize_v2_run,
    physical_rehandle_cost,
    primitive_physical_rehandle_cost,
    run_smoke_training,
    select_better_validation,
    summarize_v2_validation,
    validation_score,
)


def _args(output_dir, *extra):
    return build_parser().parse_args(["--output-dir", str(output_dir), *extra])


def _run(
    *,
    physical=0,
    required=2,
    delivered=None,
    dense=100.0,
    mae=3.0,
    strict=True,
    failure=None,
    exact=True,
):
    return {
        "strict_method_success": strict,
        "physical_rehandles": physical,
        "required_deliveries": required,
        "delivery_count": required if delivered is None else delivered,
        "dense_return": dense,
        "mean_absolute_error": mae,
        "method_failure_reason": failure,
        "illegal_drops": 0,
        "fallbacks": 0,
        "witness_mismatches": 0,
        "all_selected_candidates_exact_safe": exact,
    }


class _FakeRuntime:
    def __init__(self):
        self.core_contract = {
            "controller": CONTROLLER_ARCHITECTURE,
            "checkpoint_family": CHECKPOINT_FAMILY,
            "backup_version": BACKUP_VERSION,
            "hold_frontier_interface": CERTIFIED_HOLD_INTERFACE_V2,
            "cost_definition": COST_DEFINITION,
            "scalarized_reward_stored_in_replay": False,
            "training_policy": "induced_nested_regularized_lagrangian_sample",
            "deployment_policy": "nested_lagrangian_map",
            "mixed_discount_policy_merit": (
                "gamma_operational_pow_elapsed_times_qop_minus_lambda_qphys"
            ),
            "operational_gamma": 0.99,
            "rehandle_gamma": 1.0,
            "operational_reward_scale": 0.01,
            "dual_update_authority": "trainer_projected_episode_residual",
            "exact_safe_frontier_authoritative": True,
            "baseline_teacher": False,
            "baseline_policy_query": False,
        }
        self.lambda_values = []
        self.dual_lambda = 0.0
        self.training_seeds = []
        self.evaluation_seeds = []

    def set_dual_lambda(self, value):
        self.lambda_values.append(float(value))
        self.dual_lambda = float(value)

    def run_episode(self, *, instance_seed, training, max_steps):
        self.asserted_max_steps = max_steps
        if training:
            self.training_seeds.append(instance_seed)
            return _run(physical=1, dense=80.0)
        self.evaluation_seeds.append(instance_seed)
        return _run(physical=0, dense=90.0)

    def checkpoint_state(self, *, include_replay):
        return {"fake": True, "include_replay": bool(include_replay)}


class ConstrainedV2TrainerTests(unittest.TestCase):
    def test_forced_witness_completion_rebuilds_hold_before_next_decision(self):
        """A completed witness must not leave the old Hold suppression live.

        B1 is already stored and has a one-macro recovery witness.  B2 is
        still outside but arrives one step after that delivery.  The forced
        B1 delivery therefore has a nonterminal successor whose only live
        control is Hold.  Building that frontier before advancing the guard
        used to produce a false empty-frontier method failure.
        """

        with tempfile.TemporaryDirectory() as parent:
            args = _args(
                Path(parent) / "forced-witness-hold",
                "--episodes",
                "1",
                "--validation-seeds",
                "84000",
                "--max-steps",
                "200",
                "--number-blocks",
                "2",
                "--device",
                "cpu",
            )
            runtime = load_default_runtime(args, build_training_contract(args))
            # Trigger the retained exact witness at the first recovery epoch.
            runtime.agent.recovery_witness_guard.max_nonprogress_recovery_decisions = 0
            original_reset = runtime.env.reset

            def reset_to_witness_completion_case(instance=None):
                state = original_reset(instance=instance)
                stored, future = runtime.env.blocks
                stored.position = (2, 2)
                stored.storage_location = (2, 2)
                stored.carrying = False
                stored.stored = True
                stored.delivered = False
                stored.stored_time_step = 0
                stored.storage_steps_needed = 100
                stored.storage_steps_elapsed = 0
                future.position = None
                future.storage_location = None
                future.carrying = False
                future.stored = False
                future.delivered = False
                # Direct delivery takes seven steps in this compact layout.
                # The next WAIT exposes the future block as an observed event.
                future.arrival_step = 8
                runtime.env.current_state = (1, 1)
                runtime.env.time_steps = 0
                return state

            runtime.env.reset = reset_to_witness_completion_case
            run = runtime.run_episode(
                instance_seed=99_123,
                training=True,
                max_steps=200,
            )

            self.assertTrue(run["strict_method_success"])
            self.assertIsNone(run["method_failure_reason"])
            self.assertEqual(run["delivery_count"], 2)
            self.assertEqual(run["required_deliveries"], 2)
            self.assertEqual(run["witness_mismatches"], 0)
            self.assertEqual(run["hold_outcome_counts"], {"observed_event": 1})
            decisions = runtime.agent.decision_log
            self.assertTrue(decisions[0]["liveness_forced"])
            self.assertEqual(decisions[0]["selected_action_type"], "deliver")
            # This is the decisive regression assertion: Hold is rebound under
            # the post-observation (now inactive) guard instead of disappearing.
            self.assertEqual(decisions[1]["selected_action_type"], "defer")

    def test_default_runtime_hook_retains_zero_delivery_failure(self):
        with tempfile.TemporaryDirectory() as parent:
            args = _args(
                Path(parent) / "real-runtime",
                "--episodes",
                "1",
                "--validation-seeds",
                "84000",
                "--max-steps",
                "1",
                "--number-blocks",
                "1",
                "--device",
                "cpu",
            )
            contract = build_training_contract(args)
            runtime = load_default_runtime(args, contract)
            self.assertEqual(
                runtime.core_contract["checkpoint_family"], CHECKPOINT_FAMILY
            )
            self.assertEqual(
                runtime.core_contract["backup_version"], BACKUP_VERSION
            )
            self.assertEqual(
                runtime.core_contract["cost_definition"], COST_DEFINITION
            )

            failed = runtime.run_episode(
                instance_seed=84042,
                training=False,
                max_steps=1,
            )
            self.assertFalse(failed["strict_method_success"])
            self.assertEqual(failed["delivery_count"], 0)
            self.assertEqual(failed["required_deliveries"], 1)
            self.assertEqual(failed["completion_rate"], 0.0)
            self.assertIsNone(failed["mean_absolute_error"])
            self.assertIsNotNone(failed["method_failure_reason"])

            normalized = normalize_v2_run(failed)
            self.assertIsNone(normalized["mean_absolute_error"])
            validation = summarize_v2_validation(
                [failed],
                DualConfig(**contract["dual"]),
                checkpoint_episode=1,
            )
            self.assertFalse(validation["strict_safety_completion_gate"])
            self.assertFalse(validation["development_candidate_eligible"])
            self.assertIsNone(validation["mean_absolute_error"])
            self.assertIsNone(validation_score(validation))
            self.assertEqual(validation["total_required_deliveries"], 1)
            self.assertFalse(validation["complete_case_filtering_used"])

    def test_primitive_cost_uses_actual_relocation_event_not_action_name(self):
        self.assertEqual(primitive_physical_rehandle_cost({}), 0)
        self.assertEqual(
            primitive_physical_rehandle_cost(
                {"relocated_block": False, "action_type": "reconfigure"}
            ),
            0,
        )
        self.assertEqual(
            primitive_physical_rehandle_cost(
                {"relocated_block": "B7", "action_type": "deliver"}
            ),
            1,
        )
        self.assertEqual(
            physical_rehandle_cost(
                [
                    {"relocated_block": False},
                    {"relocated_block": "B7"},
                    {"relocated_block": "B7"},
                ]
            ),
            2,
        )
        with self.assertRaisesRegex(
            ConstrainedV2TrainerError, "relocated_block"
        ):
            primitive_physical_rehandle_cost({"relocated_block": 2})

    def test_dual_residual_update_projection_and_round_trip(self):
        config = DualConfig(
            budget_per_100_required_deliveries=20.0,
            learning_rate=0.5,
            lambda_initial=0.2,
            lambda_max=1.0,
        )
        self.assertAlmostEqual(episode_budget_residual(1, 2, config), 0.6)
        dual = ProjectedEpisodeDual(config)
        positive = dual.update(
            [{"physical_rehandles": 1, "required_deliveries": 2}]
        )
        self.assertAlmostEqual(positive["lambda_before"], 0.2)
        self.assertAlmostEqual(positive["lambda_after"], 0.5)

        negative = dual.update(
            [{"physical_rehandles": 0, "required_deliveries": 10}]
        )
        self.assertEqual(negative["lambda_after"], 0.0)
        self.assertTrue(negative["projected"])

        saturated = ProjectedEpisodeDual(
            config, lambda_value=0.9
        ).update([{"physical_rehandles": 4, "required_deliveries": 1}])
        self.assertEqual(saturated["lambda_after"], 1.0)
        self.assertTrue(saturated["saturated_with_positive_residual"])

        restored = ProjectedEpisodeDual.from_state_dict(dual.state_dict(), config)
        self.assertEqual(restored.state_dict(), dual.state_dict())
        changed = DualConfig(
            budget_per_100_required_deliveries=10.0,
            learning_rate=0.5,
            lambda_initial=0.2,
            lambda_max=1.0,
        )
        with self.assertRaisesRegex(
            ConstrainedV2TrainerError, "configuration mismatch"
        ):
            ProjectedEpisodeDual.from_state_dict(dual.state_dict(), changed)

    def test_validation_is_strict_then_budget_gated_without_survivor_filtering(self):
        config = DualConfig(budget_per_100_required_deliveries=25.0)
        eligible = summarize_v2_validation(
            [_run(physical=0), _run(physical=1, required=4)],
            config,
            checkpoint_episode=5,
        )
        self.assertTrue(eligible["strict_safety_completion_gate"])
        self.assertTrue(eligible["budget_gate"])
        self.assertTrue(eligible["development_candidate_eligible"])
        self.assertFalse(eligible["deployment_checkpoint_eligible"])
        self.assertAlmostEqual(
            eligible["physical_rehandles_per_100_required_deliveries"],
            100.0 / 6.0,
        )
        self.assertFalse(eligible["complete_case_filtering_used"])

        budget_fail = summarize_v2_validation(
            [_run(physical=1), _run(physical=1)],
            config,
            checkpoint_episode=10,
        )
        self.assertTrue(budget_fail["strict_safety_completion_gate"])
        self.assertFalse(budget_fail["budget_gate"])
        self.assertIsNone(validation_score(budget_fail))

        strict_fail = summarize_v2_validation(
            [_run(), _run(strict=False, delivered=1, failure="time_limit")],
            config,
            checkpoint_episode=15,
        )
        self.assertFalse(strict_fail["strict_safety_completion_gate"])
        self.assertEqual(strict_fail["run_count"], 2)
        self.assertEqual(len(strict_fail["safety_issues"]), 1)

        missing_exact_audit = _run()
        missing_exact_audit.pop("all_selected_candidates_exact_safe")
        with self.assertRaisesRegex(
            ConstrainedV2TrainerError, "all_selected_candidates_exact_safe"
        ):
            summarize_v2_validation(
                [missing_exact_audit], config, checkpoint_episode=20
            )

    def test_checkpoint_selection_uses_dense_then_mae_rehandles_and_earlier(self):
        config = DualConfig(budget_per_100_required_deliveries=100.0)
        early = summarize_v2_validation(
            [_run(dense=100.0, mae=4.0)], config, checkpoint_episode=5
        )
        later_better = summarize_v2_validation(
            [_run(dense=101.0, mae=9.0)], config, checkpoint_episode=10
        )
        self.assertIs(select_better_validation(early, later_better), later_better)

        lower_mae = summarize_v2_validation(
            [_run(dense=101.0, mae=2.0)], config, checkpoint_episode=15
        )
        self.assertIs(
            select_better_validation(later_better, lower_mae), lower_mae
        )

        ineligible = copy.deepcopy(lower_mae)
        ineligible["development_candidate_eligible"] = False
        self.assertIs(select_better_validation(lower_mae, ineligible), lower_mae)

    def test_contract_refuses_83xxx_and_checkpoint_metadata_is_v2_only(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(directory)
            contract = build_training_contract(args)
            self.assertEqual(contract["checkpoint_family"], CHECKPOINT_FAMILY)
            self.assertEqual(contract["cost_definition"], COST_DEFINITION)
            self.assertFalse(contract["prospective_83xxx_panel_opened"])
            self.assertEqual(
                contract["certified_hold"]["frontier_interface"],
                CERTIFIED_HOLD_INTERFACE_V2,
            )
            self.assertFalse(
                contract["certified_hold"]["base_v1_defer_frontier_allowed"]
            )

            protected = _args(directory, "--validation-seeds", "83000")
            with self.assertRaisesRegex(
                ConstrainedV2TrainerError, "protected prospective"
            ):
                build_training_contract(protected)

            config = DualConfig(**contract["dual"])
            dual = ProjectedEpisodeDual(config)
            eligible = summarize_v2_validation(
                [_run(physical=0)], config, checkpoint_episode=1
            )
            metadata = build_checkpoint_metadata(
                contract=contract,
                dual=dual,
                completed_episodes=1,
                checkpoint_role="best_development_candidate",
                validation_summary=eligible,
                core_contract={"controller": "v2"},
            )
            self.assertEqual(metadata["checkpoint_family"], CHECKPOINT_FAMILY)
            self.assertFalse(metadata["scalarized_reward_stored_in_replay"])

            self.assertFalse(metadata["deployment_checkpoint_eligible"])
            ineligible = dict(eligible, development_candidate_eligible=False)
            with self.assertRaisesRegex(
                ConstrainedV2TrainerError, "requires eligible validation"
            ):
                build_checkpoint_metadata(
                    contract=contract,
                    dual=dual,
                    completed_episodes=1,
                    checkpoint_role="best_development_candidate",
                    validation_summary=ineligible,
                    core_contract={"controller": "v2"},
                )

    def test_contract_only_cli_path_opens_no_core_or_training_seed(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "contract-only"
            args = _args(output, "--contract-only")
            summary = run_smoke_training(args)
            self.assertEqual(summary["status"], "contract_only")
            self.assertEqual(summary["episodes_executed"], 0)
            payload = json.loads(
                (output / "training-summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(payload["prospective_83xxx_panel_opened"])
            self.assertFalse((output / "latest.pth").exists())

    def test_injected_runtime_executes_short_smoke_and_writes_gated_checkpoints(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "smoke"
            args = _args(
                output,
                "--episodes",
                "2",
                "--validation-seeds",
                "84000",
                "84001",
                "--rehandle-budget-per-100",
                "20",
            )
            runtime = _FakeRuntime()
            summary = run_smoke_training(args, runtime=runtime)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["completed_training_episodes"], 2)
            self.assertTrue(summary["development_candidate_eligible"])
            self.assertFalse(summary["deployment_checkpoint_eligible"])
            self.assertEqual(runtime.training_seeds, [60_000_000, 60_000_001])
            self.assertEqual(
                runtime.evaluation_seeds, [84_000, 84_001, 84_000, 84_001]
            )
            self.assertGreater(runtime.lambda_values[-1], runtime.lambda_values[0])
            best = torch.load(
                output / "best-development-candidate.pth",
                map_location="cpu",
                weights_only=False,
            )
            latest = torch.load(
                output / "latest.pth", map_location="cpu", weights_only=False
            )
            self.assertEqual(
                best["checkpoint_role"], "best_development_candidate"
            )
            self.assertFalse(best["deployment_checkpoint_eligible"])
            self.assertFalse(best["agent_state"]["include_replay"])
            self.assertEqual(
                latest["checkpoint_role"], "latest_development_state"
            )
            self.assertFalse(latest["resumable"])
            self.assertTrue(latest["agent_state"]["include_replay"])


if __name__ == "__main__":
    unittest.main()
