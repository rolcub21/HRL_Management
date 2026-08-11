import contextlib
import io
import math
from pathlib import Path
from statistics import fmean, stdev
import tempfile
import unittest

import torch

import train_vcg_constrained_v2_2 as v22


def _validation_row(instance_index, rng_index, *, physical=0, mae=3.0):
    seed = v22.DEFAULT_VALIDATION_SEEDS[instance_index]
    return {
        "strict_method_success": True,
        "physical_rehandles": int(physical),
        "required_deliveries": 5,
        "delivery_count": 5,
        "dense_return": 100.0,
        "mean_absolute_error": float(mae),
        "method_failure_reason": None,
        "illegal_drops": 0,
        "fallbacks": 0,
        "witness_mismatches": 0,
        "all_selected_candidates_exact_safe": True,
        "instance_seed": seed,
        "instance_index": instance_index,
        "episode_instance_id": f"instance-{seed}",
        "schedule_id": f"schedule-{seed}",
        "episode_instance_sha256": f"{seed:064x}",
        "policy_rng_index": rng_index,
        "policy_rng_seed": v22.validation_policy_rng_seed(
            instance_index, rng_index
        ),
        "evaluation_learning": False,
        "training_agent_unchanged": True,
        "validation_batch_state_unchanged": True,
        "fresh_evaluation_clone": True,
        "stochastic_selection_only": True,
        "macro_decisions": 1,
        "selected_action_counts": {"deliver": 1},
        "hold_outcome_counts": {},
        "policy_diagnostic_decisions": 1,
        "mean_outer_policy_entropy": 0.1,
        "mean_outer_map_probability": 0.9,
        "mean_selected_within_policy_entropy": 0.05,
        "mean_selected_within_map_probability": 0.95,
    }


def _validation_grid():
    rows = []
    for instance_index in range(len(v22.DEFAULT_VALIDATION_SEEDS)):
        # Cluster rates repeat 0, 5, 10, and 15 per 100 required blocks.
        physical_count = instance_index % 4
        for rng_index in range(v22.VALIDATION_POLICY_RNG_COUNT):
            rows.append(
                _validation_row(
                    instance_index,
                    rng_index,
                    physical=int(rng_index < physical_count),
                )
            )
    return rows


class V22FrozenProtocolTests(unittest.TestCase):
    def test_defaults_schedule_and_seed_namespaces_are_exact(self):
        with tempfile.TemporaryDirectory() as parent:
            args = v22.build_parser().parse_args(
                ["--output-dir", str(Path(parent) / "contract")]
            )
            contract = v22.build_training_contract(args)

        self.assertEqual(args.model_seed, 10)
        self.assertEqual(args.episodes, 200)
        self.assertEqual(args.validation_every, 20)
        self.assertEqual(args.max_steps, 2_000)
        self.assertEqual(tuple(args.validation_seeds), tuple(range(85_000, 85_012)))
        self.assertEqual(contract["validation_rows_per_checkpoint"], 48)
        self.assertEqual(
            contract["validation_design"]["statistical_unit"],
            "EpisodeInstance_cluster",
        )
        self.assertEqual(
            tuple(contract["validation_design"]["candidate_looks"]),
            v22.CANDIDATE_EPISODES,
        )
        self.assertFalse(contract["validation_design"]["final_panel_opened"])

        expected_temperatures = {
            4: (0.06309573444801933, 0.5492802716530588),
            5: (0.03981071705534972, 0.30170881682725814),
            6: (0.0251188643150958, 0.16572270086699936),
            7: (0.01584893192461113, 0.09102821015130401),
        }
        for block, (within, group) in expected_temperatures.items():
            observed_within, observed_group = v22.temperatures_for_block(block)
            self.assertEqual(observed_within, (within,) * 4)
            self.assertEqual(observed_group, group)
        self.assertEqual(v22.temperatures_for_block(20), ((0.01,) * 4, 0.05))

        self.assertEqual(v22.training_policy_rng_seed(1), 610_000_000)
        self.assertEqual(v22.training_policy_rng_seed(200), 610_000_199)
        self.assertEqual(v22.validation_policy_rng_seed(0, 0), 620_000_000)
        self.assertEqual(v22.validation_policy_rng_seed(11, 3), 620_000_047)
        self.assertLess(620_000_047, v22.FINAL_POLICY_RNG_BASE)

    def test_contract_rejects_development_and_final_panel_leakage(self):
        with tempfile.TemporaryDirectory() as parent:
            parser = v22.build_parser()
            for seeds in ((83_000,) * 12, tuple(range(86_000, 86_012))):
                argv = ["--output-dir", str(Path(parent) / str(seeds[0]))]
                argv += ["--validation-seeds", *(str(seed) for seed in seeds)]
                with self.subTest(first_seed=seeds[0]):
                    with self.assertRaisesRegex(
                        v22.ConstrainedV22TrainerError, "frozen 85000"
                    ):
                        v22.build_training_contract(parser.parse_args(argv))

    def test_ucb_uses_twelve_episode_instance_cluster_means(self):
        rows = _validation_grid()
        summary = v22.summarize_validation(
            rows,
            v22.V22DualConfig(),
            checkpoint_episode=80,
            validation_lambda=0.0,
            schedule_state=v22.schedule_for_episode(80, validation=True).to_dict(),
        )
        cluster_rates = [0.0, 5.0, 10.0, 15.0] * 3
        expected_mean = fmean(cluster_rates)
        expected_se = stdev(cluster_rates) / math.sqrt(12)
        expected_ucb = expected_mean + v22.VALIDATION_COST_T_95 * expected_se
        self.assertEqual(
            tuple(summary["episode_instance_cluster_rehandle_rates"]),
            tuple(cluster_rates),
        )
        self.assertAlmostEqual(
            summary["expected_physical_rehandles_per_100_required_deliveries"],
            expected_mean,
        )
        self.assertAlmostEqual(
            summary["physical_rehandle_rate_cluster_standard_error"], expected_se
        )
        self.assertAlmostEqual(
            summary["physical_rehandle_rate_one_sided_95_ucb"], expected_ucb
        )
        self.assertTrue(summary["development_candidate_eligible"])

        # Historical saturation is diagnostic only. A recovered, currently
        # feasible stochastic policy remains eligible under the frozen rule.
        recovered = v22.summarize_validation(
            rows,
            v22.V22DualConfig(),
            checkpoint_episode=80,
            validation_lambda=1.0,
            schedule_state=v22.schedule_for_episode(80, validation=True).to_dict(),
            positive_residual_saturation_count=3,
        )
        self.assertEqual(
            recovered["positive_residual_dual_saturation_count_before_validation"], 3
        )
        self.assertFalse(recovered["positive_residual_dual_saturation"])
        self.assertTrue(recovered["development_candidate_eligible"])

    def test_candidate_looks_and_all_row_integrity_fail_closed(self):
        rows = _validation_grid()
        early = v22.summarize_validation(
            rows,
            v22.V22DualConfig(),
            checkpoint_episode=60,
            validation_lambda=0.0,
            schedule_state=v22.schedule_for_episode(60, validation=True).to_dict(),
        )
        self.assertFalse(early["candidate_look_gate"])
        self.assertFalse(early["development_candidate_eligible"])

        corrupted = [dict(row) for row in rows]
        corrupted[17]["training_agent_unchanged"] = False
        corrupted[18]["validation_batch_state_unchanged"] = False
        failed = v22.summarize_validation(
            corrupted,
            v22.V22DualConfig(),
            checkpoint_episode=80,
            validation_lambda=0.0,
            schedule_state=v22.schedule_for_episode(80, validation=True).to_dict(),
        )
        self.assertFalse(failed["strict_integrity_gate"])
        self.assertFalse(failed["development_candidate_eligible"])
        with self.assertRaisesRegex(v22.ConstrainedV22TrainerError, "exactly 48"):
            v22.summarize_validation(
                rows[:-1],
                v22.V22DualConfig(),
                checkpoint_episode=80,
                validation_lambda=0.0,
                schedule_state=v22.schedule_for_episode(80, validation=True).to_dict(),
            )


class _FakeRuntime:
    def __init__(self, args):
        self.args = args
        self.dual_lambda = 0.0
        self.schedule_state = {}
        self.events = []
        self.core_contract = {
            "checkpoint_family": v22.CHECKPOINT_FAMILY,
            "controller": v22.CONTROLLER_ARCHITECTURE,
            "backup_version": v22.BACKUP_VERSION,
            "policy_schedule_protocol": v22.POLICY_SCHEDULE_PROTOCOL,
            "training_policy": v22.POLICY_REALIZATION,
            "backup_policy": v22.POLICY_REALIZATION,
            "validation_policy": v22.POLICY_REALIZATION,
            "deployment_policy": v22.POLICY_REALIZATION,
            "map_policy_available": False,
            "same_policy_for_behavior_backup_validation_deployment": True,
            "single_shared_lagrangian_policy_for_both_target_heads": True,
            "entropy_or_kl_added_to_raw_component_targets": False,
            "validation_protocol": v22.VALIDATION_PROTOCOL,
            "exact_safe_frontier_authoritative": True,
            "baseline_teacher": False,
            "baseline_policy_query": False,
            "agent_config": v22.frozen_agent_config(args),
        }

    def set_dual_lambda(self, value):
        self.dual_lambda = float(value)

    def set_schedule_state(self, state):
        self.schedule_state = dict(state)

    def begin_validation_batch(self):
        self.events.append({"validation_batch": "begin"})
        return {
            "validation_batch_started": True,
            "full_training_state_digest": "fake-digest",
        }

    def end_validation_batch(self):
        self.events.append({"validation_batch": "end"})
        return {
            "validation_batch_ended": True,
            "training_agent_unchanged": True,
            "full_training_state_digest_before": "fake-digest",
            "full_training_state_digest_after": "fake-digest",
        }

    def run_episode(
        self,
        *,
        instance_seed,
        training,
        max_steps,
        policy_rng_index=None,
        policy_rng_seed=None,
    ):
        del max_steps
        self.events.append(
            {
                "training": bool(training),
                "instance_seed": int(instance_seed),
                "block": self.schedule_state["block_number"],
                "lambda": self.dual_lambda,
                "policy_rng_index": policy_rng_index,
                "policy_rng_seed": policy_rng_seed,
            }
        )
        result = {
            "strict_method_success": True,
            "physical_rehandles": 2 if training else 0,
            "required_deliveries": 8,
            "delivery_count": 8,
            "dense_return": 100.0,
            "mean_absolute_error": 3.0,
            "method_failure_reason": None,
            "illegal_drops": 0,
            "fallbacks": 0,
            "witness_mismatches": 0,
            "all_selected_candidates_exact_safe": True,
            "method_version": v22.METHOD_VERSION,
            "policy_schedule_protocol": v22.POLICY_SCHEDULE_PROTOCOL,
            "policy_mode": v22.POLICY_REALIZATION,
            "execution_context": self.schedule_state["execution_context"],
            "within_group_temperatures": tuple(
                self.schedule_state["within_group_temperatures"]
            ),
            "group_temperature": self.schedule_state["group_temperature"],
            "dual_lambda": self.dual_lambda,
            "map_selection_used": False,
            "macro_decisions": 1,
            "selected_action_counts": {"deliver": 1},
            "hold_outcome_counts": {},
            "policy_diagnostic_decisions": 1,
            "mean_outer_policy_entropy": 0.1,
            "mean_outer_map_probability": 0.9,
            "mean_selected_within_policy_entropy": 0.05,
            "mean_selected_within_map_probability": 0.95,
            "policy_rng_index": policy_rng_index,
            "policy_rng_seed": policy_rng_seed,
            "instance_seed": int(instance_seed),
            "episode_instance_id": f"instance-{instance_seed}",
            "schedule_id": f"schedule-{instance_seed}",
            "episode_instance_sha256": f"{int(instance_seed):064x}",
            "replay_rng_seed": v22.REPLAY_RNG_SEED,
            "behavior_policy_rng_reset": bool(training),
        }
        if not training:
            result.update(
                evaluation_learning=False,
                training_agent_unchanged=True,
                fresh_evaluation_clone=True,
                stochastic_selection_only=True,
                option_evaluation_mode=True,
            )
        return result

    def checkpoint_state(self, *, include_replay):
        return {
            "fake": True,
            "include_replay": bool(include_replay),
            "dual_lambda": self.dual_lambda,
            "schedule_state": dict(self.schedule_state),
        }


class V22CadenceAndCheckpointTests(unittest.TestCase):
    def test_transient_per_row_training_state_mutation_fails_closed(self):
        class TransientMutationRuntime(_FakeRuntime):
            emitted = False

            def run_episode(self, **kwargs):
                row = super().run_episode(**kwargs)
                if not kwargs["training"] and not self.emitted:
                    self.emitted = True
                    row["training_agent_unchanged"] = False
                return row

        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "transient-mutation"
            args = v22.build_parser().parse_args(["--output-dir", str(output)])
            with self.assertRaisesRegex(RuntimeError, "mutated and later restored"):
                with contextlib.redirect_stdout(io.StringIO()):
                    v22.run_development_calibration(
                        args, runtime=TransientMutationRuntime(args)
                    )

    def test_dual_updates_on_nonvalidation_blocks_and_only_best_is_candidate(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "fake-v22"
            args = v22.build_parser().parse_args(["--output-dir", str(output)])
            runtime = _FakeRuntime(args)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = v22.run_development_calibration(args, runtime=runtime)

            self.assertEqual(summary["completed_training_episodes"], 200)
            self.assertEqual(summary["dual_state"]["update_count"], 17)
            self.assertEqual(len(summary["dual_update_history"]), 18)
            self.assertFalse(summary["dual_update_history"][-1]["applied"])
            self.assertEqual(
                [item["checkpoint_episode"] for item in summary["validation_history"]],
                list(range(20, 201, 20)),
            )

            validation_events = [
                event
                for event in runtime.events
                if "training" in event and not event["training"]
            ]
            first_ep40 = validation_events[48]
            self.assertEqual(first_ep40["block"], 4)
            # Block 3 updates lambda at ep30 even though there is no validation.
            self.assertGreater(first_ep40["lambda"], 0.0)
            self.assertTrue(
                all(
                    event["instance_seed"] not in v22.FINAL_PANEL_SEEDS
                    for event in runtime.events
                    if "instance_seed" in event
                )
            )

            best = torch.load(
                output / "best-development-candidate.pth",
                map_location="cpu",
                weights_only=False,
            )
            latest = torch.load(
                output / "latest.pth", map_location="cpu", weights_only=False
            )
            self.assertEqual(best["checkpoint_role"], "best_development_candidate")
            self.assertEqual(best["completed_episodes"], 80)
            self.assertTrue(best["development_candidate_eligible"])
            self.assertTrue(best["validation_authenticates_checkpoint_lambda"])
            self.assertEqual(best["pending_dual_batch"]["episode_count"], 10)

            self.assertEqual(latest["checkpoint_role"], "latest_development_state")
            self.assertFalse(latest["development_candidate_eligible"])
            self.assertFalse(latest["deployment_checkpoint_eligible"])
            self.assertFalse(latest["terminal_dual_proposal"]["applied"])


if __name__ == "__main__":
    unittest.main()
