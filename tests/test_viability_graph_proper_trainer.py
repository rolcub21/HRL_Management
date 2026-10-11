import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from train_viability_graph_smdp import epsilon_at
from train_viability_graph_smdp_proper import (
    DEFAULT_VALIDATION_SEEDS,
    SEALED_IN_REGIME_TEST_SEEDS,
    TRAINING_PROTOCOL,
    _validate_args,
    build_parser,
    main,
)


class ProperViabilityGraphTrainerTests(unittest.TestCase):
    def test_default_protocol_has_disjoint_predeclared_namespaces(self):
        args = build_parser().parse_args(("--output-dir", "unused"))
        train_seed_base = _validate_args(args)

        self.assertEqual(train_seed_base, 30_000_000)
        self.assertEqual(tuple(args.validation_seeds), DEFAULT_VALIDATION_SEEDS)
        self.assertFalse(
            set(range(train_seed_base, train_seed_base + args.total_episodes))
            & set(args.validation_seeds)
        )
        self.assertFalse(set(args.validation_seeds) & SEALED_IN_REGIME_TEST_SEEDS)

    def test_training_runner_refuses_a_sealed_test_seed(self):
        args = build_parser().parse_args(
            (
                "--output-dir",
                "unused",
                "--validation-seeds",
                "77000",
            )
        )
        with self.assertRaisesRegex(ValueError, "sealed test seeds"):
            _validate_args(args)

    def test_epsilon_warmup_uses_the_decision_clock(self):
        self.assertEqual(
            epsilon_at(
                99,
                start=0.9,
                end=0.1,
                warmup_decisions=100,
                decay_decisions=100,
            ),
            0.9,
        )
        self.assertAlmostEqual(
            epsilon_at(
                150,
                start=0.9,
                end=0.1,
                warmup_decisions=100,
                decay_decisions=100,
            ),
            0.5,
        )

    def test_episode_boundary_pause_resume_and_validation_are_audited(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "proper"
            common = (
                "--output-dir",
                str(output),
                "--model-seed",
                "0",
                "--total-episodes",
                "2",
                "--train-instance-seed-base",
                "900000",
                "--validation-seeds",
                "910000",
                "--eval-every",
                "1",
                "--checkpoint-every",
                "1",
                "--log-every",
                "1",
                "--grid-rows",
                "5",
                "--grid-cols",
                "5",
                "--number-blocks",
                "1",
                "--arrival-rate",
                "1.5",
                "--proc-mean",
                "5",
                "--max-steps",
                "80",
                "--batch-size",
                "2",
                "--replay-size",
                "50",
                "--target-update-every",
                "2",
                "--epsilon-decay-decisions",
                "10",
                "--graph-hidden-dim",
                "8",
                "--graph-embedding-dim",
                "8",
                "--message-passing-steps",
                "1",
                "--action-embedding-dim",
                "4",
                "--head-hidden-dim",
                "8",
                "--device",
                "cpu",
            )
            paused = main(common + ("--stop-after-episode", "1"))
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["completed_training_episodes"], 1)

            latest_path = output / "latest.pth"
            best_path = output / "best.pth"
            self.assertTrue(latest_path.is_file())
            self.assertTrue(best_path.is_file())
            resumed = main(common + ("--resume", str(latest_path)))

            self.assertEqual(resumed["training_protocol"], TRAINING_PROTOCOL)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["completed_training_episodes"], 2)
            self.assertEqual(len(resumed["training"]["runs"]), 2)
            self.assertEqual(len(resumed["validation_history"]), 2)
            for record in resumed["validation_history"]:
                summary = record["summary"]
                self.assertEqual(summary["replay_size"], 0)
                self.assertFalse(summary["q_local_training_mode"])
                self.assertTrue(summary["fresh_certificate_cache_per_episode"])
                self.assertEqual(summary["evaluation_epsilon"], 0.0)

            latest = torch.load(
                latest_path,
                map_location="cpu",
                weights_only=False,
            )
            self.assertTrue(latest["trainer_resumable"])
            self.assertIn("replay", latest["agent_state"])
            self.assertEqual(latest["completed_training_episodes"], 2)
            self.assertFalse(latest["test_panels_opened"])
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            self.assertFalse(best["trainer_resumable"])
            self.assertNotIn("replay", best["agent_state"])
            self.assertEqual(best["environment"], best["resume_contract"]["environment"])
            self.assertEqual(
                best["viability_search"],
                best["resume_contract"]["search_config"],
            )
            self.assertEqual(
                best["liveness_rule"],
                best["resume_contract"]["liveness_rule"],
            )

            uninterrupted_output = Path(directory) / "uninterrupted"
            uninterrupted_args = (
                "--output-dir",
                str(uninterrupted_output),
            ) + common[2:]
            uninterrupted = main(uninterrupted_args)
            uninterrupted_latest = torch.load(
                uninterrupted_output / "latest.pth",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(uninterrupted["status"], "complete")
            for network_name in ("Q_local", "Q_target"):
                resumed_network = latest["agent_state"][network_name]
                direct_network = uninterrupted_latest["agent_state"][network_name]
                self.assertEqual(set(resumed_network), set(direct_network))
                self.assertTrue(
                    all(
                        torch.equal(resumed_network[name], direct_network[name])
                        for name in resumed_network
                    )
                )
            for clock in (
                "transition_count",
                "decision_count",
                "gradient_steps",
                "target_updates",
                "rng_state",
            ):
                self.assertEqual(
                    latest["agent_state"][clock],
                    uninterrupted_latest["agent_state"][clock],
                )

            with (output / "training-summary.json").open(encoding="utf-8") as handle:
                serialized = json.load(handle)
            self.assertEqual(serialized["status"], "complete")
            self.assertFalse(serialized["formal_test_authorized"])


if __name__ == "__main__":
    unittest.main()
