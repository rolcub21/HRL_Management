import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from PSLAP.viability_candidates_hold_v2 import (
    CERTIFIED_HOLD_INTERFACE_V2,
    CERTIFIED_HOLD_RULE_V2,
    HIDDEN_SCHEDULE_CONTRACT_V2,
    ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
)
from train_vcg_constrained_v2 import COST_DEFINITION
from train_vcg_constrained_v2_1 import (
    ANNEAL_LAST_BLOCK,
    BACKUP_VERSION,
    BLOCK_EPISODES,
    CHECKPOINT_FAMILY,
    CONTROLLER_ARCHITECTURE,
    METHOD_VERSION,
    POLICY_SCHEDULE_PROTOCOL,
    POLICY_DIAGNOSTIC_DISTRIBUTION,
    FROZEN_VIABILITY_SEARCH_CONFIG,
    TOTAL_BLOCKS,
    CompleteBlockProjectedDual,
    ConstrainedV21TrainerError,
    V21DualConfig,
    build_parser,
    build_training_contract,
    frozen_agent_config,
    load_default_runtime,
    run_development_calibration,
    schedule_for_episode,
    summarize_validation,
    temperatures_for_block,
)


def _args(output_dir, *extra):
    return build_parser().parse_args(["--output-dir", str(output_dir), *extra])


def _base_run(*, physical=0, dense=100.0, mae=3.0):
    return {
        "strict_method_success": True,
        "physical_rehandles": int(physical),
        "required_deliveries": 2,
        "delivery_count": 2,
        "dense_return": float(dense),
        "mean_absolute_error": float(mae),
        "method_failure_reason": None,
        "illegal_drops": 0,
        "fallbacks": 0,
        "witness_mismatches": 0,
        "all_selected_candidates_exact_safe": True,
    }


class _FakeV21Runtime:
    def __init__(self, args):
        self.core_contract = {
            "checkpoint_family": CHECKPOINT_FAMILY,
            "controller": CONTROLLER_ARCHITECTURE,
            "backup_version": BACKUP_VERSION,
            "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
            "hold_frontier_interface": CERTIFIED_HOLD_INTERFACE_V2,
            "hold_rule": CERTIFIED_HOLD_RULE_V2,
            "robust_stutter_contract": ROBUST_STUTTER_CERTIFICATION_CONTRACT_V2,
            "hidden_schedule_contract": HIDDEN_SCHEDULE_CONTRACT_V2,
            "base_v1_defer_frontier_allowed": False,
            "cost_definition": COST_DEFINITION,
            "scalarized_reward_stored_in_replay": False,
            "same_temperatures_for_behavior_and_backup": True,
            "training_policy": (
                "block_annealed_nested_regularized_lagrangian_sample"
            ),
            "deployment_policy": "nested_lagrangian_map",
            "mixed_discount_policy_merit": (
                "gamma_operational_pow_elapsed_times_qop_minus_lambda_qphys"
            ),
            "operational_gamma": 0.99,
            "rehandle_gamma": 1.0,
            "operational_reward_scale": 0.01,
            "dual_update_authority": "trainer_complete_active_block_residual",
            "exact_safe_frontier_authoritative": True,
            "baseline_teacher": False,
            "baseline_policy_query": False,
            "agent_config": frozen_agent_config(args),
            "policy_diagnostic_distribution": POLICY_DIAGNOSTIC_DISTRIBUTION,
            "viability_search_config": dict(FROZEN_VIABILITY_SEARCH_CONFIG),
            "fresh_certificate_cache_per_episode": True,
        }
        self.dual_lambda = 0.0
        self.schedule_state = {}
        self.events = []

    def set_dual_lambda(self, value):
        self.dual_lambda = float(value)
        self.events.append(("set_lambda", self.dual_lambda))

    def set_schedule_state(self, state):
        self.schedule_state = dict(state)
        self.events.append(
            (
                "set_schedule",
                self.schedule_state["episode_number"],
                self.schedule_state["policy_mode"],
                self.dual_lambda,
            )
        )

    def run_episode(self, *, instance_seed, training, max_steps):
        del max_steps
        schedule = dict(self.schedule_state)
        self.events.append(
            (
                "run",
                bool(training),
                schedule["episode_number"],
                schedule["block_number"],
                schedule["policy_mode"],
                self.dual_lambda,
                int(instance_seed),
            )
        )
        run = _base_run(
            physical=1 if training else 0,
            dense=80.0 if training else 100.0,
            mae=3.0,
        )
        run.update(
            {
                "method_version": METHOD_VERSION,
                "policy_schedule_protocol": POLICY_SCHEDULE_PROTOCOL,
                "policy_mode": schedule["policy_mode"],
                "within_group_temperatures": tuple(
                    schedule["within_group_temperatures"]
                ),
                "group_temperature": schedule["group_temperature"],
                "dual_lambda": self.dual_lambda,
                "macro_decisions": 1,
                "selected_action_counts": {"deliver": 1},
                "hold_outcome_counts": {},
                "policy_diagnostic_decisions": 1,
                "mean_outer_policy_entropy": 0.1,
                "mean_outer_map_probability": 0.9,
                "mean_selected_within_policy_entropy": 0.05,
                "mean_selected_within_map_probability": 0.95,
                "evaluation_learning": False if not training else None,
                "training_agent_unchanged": True if not training else None,
                "policy_diagnostic_distribution": POLICY_DIAGNOSTIC_DISTRIBUTION,
            }
        )
        return run

    def checkpoint_state(self, *, include_replay):
        return {
            "fake": True,
            "include_replay": bool(include_replay),
            "dual_lambda": self.dual_lambda,
            "schedule_state": dict(self.schedule_state),
        }


class V21TrainerTests(unittest.TestCase):
    def test_block_temperature_schedule_has_frozen_boundaries(self):
        within1, group1 = temperatures_for_block(1)
        within3, group3 = temperatures_for_block(3)
        within4, group4 = temperatures_for_block(4)
        within8, group8 = temperatures_for_block(8)
        within10, group10 = temperatures_for_block(10)
        self.assertEqual(within1, (0.1,) * 4)
        self.assertEqual(group1, 1.0)
        self.assertEqual((within3, group3), (within1, group1))
        self.assertLess(within4[0], within3[0])
        self.assertLess(group4, group3)
        self.assertEqual(within8, (0.01,) * 4)
        self.assertEqual(group8, 0.05)
        self.assertEqual((within10, group10), (within8, group8))

        warmup = schedule_for_episode(20)
        anneal = schedule_for_episode(21)
        floor = schedule_for_episode(81)
        validation = schedule_for_episode(80, validation=True)
        self.assertEqual(warmup.phase, "critic_warmup")
        self.assertFalse(warmup.dual_updates_enabled)
        self.assertEqual(anneal.phase, "primal_dual_anneal")
        self.assertTrue(anneal.dual_updates_enabled)
        self.assertEqual(floor.phase, "low_temperature_stabilization")
        self.assertEqual(validation.phase, "primal_dual_anneal")
        self.assertEqual(validation.policy_mode, "map")
        self.assertFalse(validation.critic_updates_enabled)

    def test_dual_accepts_only_complete_active_blocks_and_terminal_is_proposal(self):
        dual = CompleteBlockProjectedDual(V21DualConfig())
        episodes = [_base_run(physical=1) for _ in range(BLOCK_EPISODES)]
        with self.assertRaisesRegex(ConstrainedV21TrainerError, "exactly one complete"):
            dual.update_complete_block(episodes[:-1], block_number=3)
        with self.assertRaisesRegex(ConstrainedV21TrainerError, "expected active block 3"):
            dual.update_complete_block(episodes, block_number=4)

        for block in range(3, TOTAL_BLOCKS):
            update = dual.update_complete_block(episodes, block_number=block)
            self.assertEqual(update["block_number"], block)
        before = dual.lambda_value
        proposal = dual.propose_complete_block(episodes, block_number=TOTAL_BLOCKS)
        self.assertGreater(proposal["lambda_after"], before)
        self.assertEqual(dual.lambda_value, before)
        with self.assertRaisesRegex(ConstrainedV21TrainerError, "must not be applied"):
            dual.update_complete_block(episodes, block_number=TOTAL_BLOCKS)
        self.assertEqual(dual.lambda_value, before)

    def test_candidate_requires_floor_temperature_budget_and_mae_noncollapse(self):
        config = V21DualConfig()
        block7 = summarize_validation(
            [_base_run(physical=0, mae=3.0)],
            config,
            checkpoint_episode=70,
            validation_lambda=0.0,
            schedule_state=schedule_for_episode(70, validation=True).to_dict(),
        )
        self.assertTrue(block7["strict_integrity_gate"])
        self.assertTrue(block7["aggregate_map_budget_gate"])
        self.assertTrue(block7["timing_noncollapse_gate"])
        self.assertFalse(block7["floor_temperature_gate"])
        self.assertFalse(block7["development_candidate_eligible"])

        block8 = summarize_validation(
            [_base_run(physical=0, mae=20.0)],
            config,
            checkpoint_episode=80,
            validation_lambda=0.0,
            schedule_state=schedule_for_episode(80, validation=True).to_dict(),
        )
        self.assertTrue(block8["floor_temperature_gate"])
        self.assertTrue(block8["timing_noncollapse_gate"])
        self.assertTrue(block8["development_candidate_eligible"])

        collapsed = summarize_validation(
            [_base_run(physical=0, mae=20.0001)],
            config,
            checkpoint_episode=80,
            validation_lambda=0.0,
            schedule_state=schedule_for_episode(80, validation=True).to_dict(),
        )
        self.assertFalse(collapsed["timing_noncollapse_gate"])
        self.assertFalse(collapsed["development_candidate_eligible"])

    def test_contract_is_new_nonresumable_and_refuses_protected_panel(self):
        with tempfile.TemporaryDirectory() as parent:
            args = _args(Path(parent) / "contract")
            contract = build_training_contract(args)
            self.assertEqual(contract["checkpoint_family"], CHECKPOINT_FAMILY)
            self.assertEqual(contract["episodes"], 100)
            self.assertEqual(contract["critic_warmup"]["episodes"], 20)
            self.assertFalse(contract["resumable"])
            self.assertTrue(
                contract["temperature_schedule"][
                    "same_temperatures_for_behavior_and_backup"
                ]
            )
            self.assertFalse(
                contract["certified_hold"]["base_v1_defer_frontier_allowed"]
            )
            self.assertEqual(
                contract["certified_hold"]["rule"], CERTIFIED_HOLD_RULE_V2
            )
            with self.assertRaisesRegex(
                ConstrainedV21TrainerError, "protected prospective"
            ):
                build_training_contract(
                    _args(Path(parent) / "bad", "--validation-seeds", "83000")
                )
            with self.assertRaisesRegex(ConstrainedV21TrainerError, "100 episodes"):
                build_training_contract(
                    _args(Path(parent) / "short", "--episodes", "25")
                )

    def test_runtime_loader_has_no_v2_compatibility_fallback(self):
        with tempfile.TemporaryDirectory() as parent:
            args = _args(Path(parent) / "missing")
            contract = build_training_contract(args)
            # Until the isolated core hook is installed, this must fail rather
            # than finding viability_graph_constrained_v2.build_smoke_runtime.
            try:
                runtime = load_default_runtime(args, contract)
            except RuntimeError as error:
                self.assertIn("V2.1", str(error))
            else:
                self.assertEqual(
                    runtime.core_contract["checkpoint_family"], CHECKPOINT_FAMILY
                )

    def test_contract_only_writes_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "contract-only"
            summary = run_development_calibration(
                _args(output, "--contract-only")
            )
            self.assertEqual(summary["status"], "contract_only")
            self.assertEqual(summary["episodes_executed"], 0)
            self.assertFalse((output / "latest.pth").exists())
            persisted = json.loads(
                (output / "training-summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(persisted["deployment_checkpoint_eligible"])

    def test_full_fake_protocol_validates_before_update_and_does_not_apply_terminal(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "full-fake"
            args = _args(output, "--validation-seeds", "84000")
            runtime = _FakeV21Runtime(args)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run_development_calibration(
                    args,
                    runtime=runtime,
                )
            self.assertEqual(summary["completed_training_episodes"], 100)
            self.assertEqual(summary["dual_state"]["update_count"], 7)
            self.assertEqual(len(summary["dual_update_history"]), 8)
            self.assertFalse(summary["dual_update_history"][-1]["applied"])
            self.assertIsNotNone(summary["terminal_dual_proposal"])
            actual_lambda = summary["dual_state"]["lambda_value"]
            proposed_lambda = summary["terminal_dual_proposal"]["lambda_after"]
            self.assertGreater(proposed_lambda, actual_lambda)

            validation_runs = [
                event
                for event in runtime.events
                if event[0] == "run" and not event[1]
            ]
            self.assertEqual(len(validation_runs), 10)
            # Block 3 validation sees lambda=0; only its subsequent proposal
            # can affect block 4.
            self.assertEqual(validation_runs[2][3], 3)
            self.assertEqual(validation_runs[2][5], 0.0)
            self.assertGreater(validation_runs[3][5], 0.0)

            best = torch.load(
                output / "best-development-candidate.pth",
                map_location="cpu",
                weights_only=False,
            )
            latest = torch.load(
                output / "latest.pth", map_location="cpu", weights_only=False
            )
            self.assertEqual(best["completed_episodes"], 80)
            self.assertFalse(best["resumable"])
            self.assertEqual(
                best["validation_summary"]["validation_lambda"],
                best["dual_state"]["lambda_value"],
            )
            self.assertEqual(best["pending_dual_batch"]["episode_count"], 10)
            self.assertFalse(latest["resumable"])
            self.assertEqual(latest["dual_state"]["lambda_value"], actual_lambda)
            self.assertEqual(
                latest["terminal_dual_proposal"]["lambda_after"], proposed_lambda
            )
            self.assertEqual(latest["pending_dual_batch"]["episode_count"], 0)

    def test_run_echo_mismatch_fails_closed(self):
        class BadRuntime(_FakeV21Runtime):
            def run_episode(self, **kwargs):
                run = super().run_episode(**kwargs)
                run["policy_mode"] = "map"
                return run

        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "bad-echo"
            args = _args(output)
            with self.assertRaisesRegex(RuntimeError, "authenticate policy_mode"):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_development_calibration(args, runtime=BadRuntime(args))


if __name__ == "__main__":
    unittest.main()
