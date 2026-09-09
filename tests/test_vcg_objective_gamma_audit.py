from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
import io
import unittest

import torch

from PSLAP.viability import ViabilityStatus
from run_vcg_objective_gamma_audit import (
    ARM_TRAINING_PROTOCOL,
    DENSE_PIECEWISE,
    LEGACY_CLIPPED,
    _execute_audit_macro,
    _validate_args,
    build_parser,
    main,
)
from viability_graph_episodic_audit import (
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
    EpisodicViabilityGraphHierarchyAgent,
)
from viability_graph_hierarchy import ViabilityGraphHierarchyAgent


class _ThreeStepOption:
    def initiation(self, state):
        return state == 0

    def policy(self, state, test=False):
        del state, test
        return 0

    def termination(self, state):
        return state >= 3


class _RewardSequenceEnv:
    def __init__(self):
        self.state = 0
        self.rewards = (1.0, 2.0, 3.0)

    def get_current_state(self):
        return self.state

    def step(self, action):
        del action
        reward = self.rewards[self.state]
        self.state += 1
        return self.state, reward, False, {}


def _candidate():
    return SimpleNamespace(
        key="fake-safe",
        certificate=SimpleNamespace(status=ViabilityStatus.SAFE),
        option=_ThreeStepOption(),
        action_type=SimpleNamespace(value="deliver"),
        mode=SimpleNamespace(value="recover"),
    )


class VcgObjectiveGammaAuditTests(unittest.TestCase):
    def test_gamma_one_executor_is_exact_undiscounted_macro_sum(self):
        execution = _execute_audit_macro(
            _RewardSequenceEnv(),
            _candidate(),
            gamma=1.0,
            remaining_steps=10,
            evaluation=True,
        )
        self.assertTrue(execution.option_success)
        self.assertEqual(execution.duration, 3)
        self.assertEqual(execution.raw_return, 6.0)
        self.assertEqual(execution.discounted_return, 6.0)

        discounted = _execute_audit_macro(
            _RewardSequenceEnv(),
            _candidate(),
            gamma=0.99,
            remaining_steps=10,
            evaluation=True,
        )
        self.assertAlmostEqual(
            discounted.discounted_return,
            1.0 + 0.99 * 2.0 + (0.99**2) * 3.0,
        )

    def test_runner_refuses_every_declared_sealed_panel(self):
        for protected_seed in (69_000, 77_000):
            args = build_parser().parse_args(
                (
                    "--output-dir",
                    "unused",
                    "--validation-seeds",
                    str(protected_seed),
                )
            )
            with self.assertRaisesRegex(ValueError, "sealed test seeds"):
                _validate_args(args)

    def test_two_by_two_smoke_is_resumable_deterministic_and_isolated(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            resumed_output = root / "resumed"
            direct_output = root / "direct"
            common_tail = (
                "--model-seed",
                "0",
                "--episodes",
                "2",
                "--train-instance-seed-base",
                "940000",
                "--validation-seeds",
                "950000",
                "--eval-every",
                "2",
                "--checkpoint-every",
                "1",
                "--log-every",
                "2",
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
            with redirect_stdout(io.StringIO()):
                paused = main(
                    (
                        "--output-dir",
                        str(resumed_output),
                        "--stop-after-episode",
                        "1",
                    )
                    + common_tail
                )
            self.assertEqual(paused["status"], "paused")
            with redirect_stdout(io.StringIO()):
                resumed = main(
                    (
                        "--output-dir",
                        str(resumed_output),
                        "--resume-existing",
                    )
                    + common_tail
                )
                direct = main(
                    ("--output-dir", str(direct_output)) + common_tail
                )
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(direct["status"], "complete")
            self.assertEqual(len(resumed["arms"]), 4)
            self.assertEqual(len(resumed["paired_final_validation"]), 1)
            with redirect_stdout(io.StringIO()):
                completed_reread = main(
                    (
                        "--output-dir",
                        str(resumed_output),
                        "--resume-existing",
                    )
                    + common_tail
                )
            self.assertEqual(completed_reread["status"], "complete")

            expected = {
                "legacy-gamma0p99": (LEGACY_CLIPPED, 0.99),
                "legacy-gamma1p00": (LEGACY_CLIPPED, 1.0),
                "dense-gamma0p99": (DENSE_PIECEWISE, 0.99),
                "dense-gamma1p00": (DENSE_PIECEWISE, 1.0),
            }
            contracts = set()
            for arm_id, (objective, gamma) in expected.items():
                resumed_latest = torch.load(
                    resumed_output / arm_id / "latest.pth",
                    map_location="cpu",
                    weights_only=False,
                )
                direct_latest = torch.load(
                    direct_output / arm_id / "latest.pth",
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(
                    resumed_latest["training_protocol"],
                    ARM_TRAINING_PROTOCOL,
                )
                self.assertEqual(
                    resumed_latest["checkpoint_family"],
                    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
                )
                self.assertEqual(resumed_latest["timing_objective"], objective)
                self.assertEqual(resumed_latest["gamma"], gamma)
                self.assertFalse(resumed_latest["test_panels_opened"])
                self.assertTrue(resumed_latest["trainer_resumable"])
                contracts.add(
                    (
                        resumed_latest["factorial_arm_id"],
                        resumed_latest["timing_objective"],
                        resumed_latest["gamma"],
                    )
                )
                for network_name in ("Q_local", "Q_target"):
                    resumed_state = resumed_latest["agent_state"][network_name]
                    direct_state = direct_latest["agent_state"][network_name]
                    self.assertEqual(set(resumed_state), set(direct_state))
                    self.assertTrue(
                        all(
                            torch.equal(resumed_state[name], direct_state[name])
                            for name in resumed_state
                        )
                    )
                restored = (
                    EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
                        resumed_latest,
                        resumable=True,
                    )
                )
                self.assertEqual(restored.config.gamma, gamma)
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    ViabilityGraphHierarchyAgent.from_checkpoint(resumed_latest)

                if gamma == 1.0:
                    for run in resumed_latest["training_history"]:
                        self.assertAlmostEqual(
                            run["episode_start_discounted_return"],
                            run["return"],
                        )

            self.assertEqual(len(contracts), 4)


if __name__ == "__main__":
    unittest.main()
