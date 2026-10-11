import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from example.small_rooms_env import SmallRoomsEnv
from PSLAP.viability_candidates import enumerate_viability_candidates
from PSLAP.viability_filter import ViabilitySearchConfig
from train_viability_graph_smdp import (
    CERTIFICATE_SCOPE,
    NO_FALLBACK_CONTRACT,
    SMDP_RETURN_CONTRACT,
    VCG_SMOKE_PROTOCOL,
    _validate_args,
    build_parser,
    epsilon_at,
    execute_certified_macro,
    main,
)


class ViabilityGraphMacroExecutionTests(unittest.TestCase):
    def test_exact_bound_option_produces_one_discounted_smdp_transition(self):
        env = SmallRoomsEnv(
            grid_rows=5,
            grid_cols=5,
            number_blocks=1,
            choose_storage=False,
            proc_mean=5,
        )
        env.reset(instance=env.sample_episode_instance(73500))
        snapshot = enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            search_config=ViabilitySearchConfig(max_nodes=20_000),
        )

        result = execute_certified_macro(
            env,
            snapshot.candidates[0],
            gamma=0.99,
            remaining_steps=80,
            evaluation=False,
        )

        self.assertTrue(result.option_success)
        self.assertTrue(result.option_terminated)
        self.assertFalse(result.env_terminal)
        self.assertGreater(result.duration, 1)
        self.assertNotEqual(result.discounted_return, result.raw_return)
        self.assertTrue(env.blocks[0].stored)
        self.assertEqual(result.illegal_drops, 0)

    def test_decision_based_epsilon_schedule_saturates(self):
        self.assertEqual(
            epsilon_at(0, start=0.9, end=0.1, decay_decisions=100),
            0.9,
        )
        self.assertAlmostEqual(
            epsilon_at(50, start=0.9, end=0.1, decay_decisions=100),
            0.5,
        )
        self.assertAlmostEqual(
            epsilon_at(1000, start=0.9, end=0.1, decay_decisions=100),
            0.1,
        )


class ViabilityGraphSmokeEntrypointTests(unittest.TestCase):
    def test_calibration_runner_refuses_the_declared_sealed_holdout(self):
        args = build_parser().parse_args(
            (
                "--output-dir",
                "unused",
                "--episodes",
                "1",
                "--eval-episodes",
                "1",
                "--eval-instance-seed-base",
                "69000",
            )
        )
        with self.assertRaisesRegex(ValueError, "sealed stress_v1"):
            _validate_args(args)

    def test_tiny_calibration_run_saves_audited_json_and_resumable_checkpoint(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "vcg"
            result = main(
                (
                    "--output-dir",
                    str(output),
                    "--episodes",
                    "1",
                    "--eval-episodes",
                    "1",
                    "--number-blocks",
                    "1",
                    "--proc-mean",
                    "5",
                    "--max-steps",
                    "80",
                    "--batch-size",
                    "2",
                    "--replay-size",
                    "50",
                    "--epsilon-start",
                    "0.2",
                    "--epsilon-end",
                    "0.0",
                    "--epsilon-decay-decisions",
                    "10",
                    "--graph-hidden-dim",
                    "8",
                    "--graph-embedding-dim",
                    "8",
                    "--action-embedding-dim",
                    "4",
                    "--head-hidden-dim",
                    "8",
                    "--message-passing-steps",
                    "1",
                    "--device",
                    "cpu",
                )
            )

            checkpoint_path = output / "checkpoint.pth"
            result_path = output / "results.json"
            self.assertTrue(checkpoint_path.is_file())
            self.assertTrue(result_path.is_file())
            with result_path.open(encoding="utf-8") as handle:
                serialized = json.load(handle)
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(result["protocol"], VCG_SMOKE_PROTOCOL)
        self.assertEqual(serialized["protocol"], VCG_SMOKE_PROTOCOL)
        self.assertEqual(result["smdp_return_contract"], SMDP_RETURN_CONTRACT)
        self.assertEqual(result["no_fallback_contract"], NO_FALLBACK_CONTRACT)
        self.assertFalse(result["baseline_teacher"])
        self.assertFalse(result["baseline_policy_query"])
        self.assertFalse(result["future_schedule_visible_to_policy"])
        self.assertTrue(result["exact_verifier_authoritative"])
        self.assertEqual(result["certificate_scope"], CERTIFICATE_SCOPE)
        self.assertFalse(result["complete_episode_certificate"])
        self.assertEqual(result["training"]["summary"]["episodes"], 1)
        self.assertEqual(result["evaluation"]["summary"]["episodes"], 1)
        self.assertEqual(result["evaluation"]["agent_audit"]["replay_size"], 0)
        decisions = result["training"]["runs"][0]["decisions"]
        self.assertGreater(len(decisions), 0)
        self.assertTrue(
            all(item["certificate_status"] == "SAFE" for item in decisions)
        )
        self.assertTrue(
            all(item["certificate_witness_macros"] >= 0 for item in decisions)
        )
        self.assertIn(
            "exact_analysis_seconds",
            result["training"]["runs"][0]["audit"],
        )
        self.assertFalse(checkpoint["baseline_teacher"])
        self.assertFalse(checkpoint["baseline_policy_query"])
        self.assertEqual(checkpoint["instance_regime"], "calibration_only")
        replay = checkpoint["agent_state"]["replay"]["memory"]
        self.assertGreater(len(replay), 0)
        self.assertTrue(all(item.duration >= 1 for item in replay))
        self.assertTrue(all(item.chosen.action_type != "" for item in replay))


if __name__ == "__main__":
    unittest.main()
