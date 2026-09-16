from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from train_fully_learned_track_b import (
    FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION,
    FULLY_LEARNED_TRAINER_VERSION,
    PHASE_ORDER,
    RAW_MACRO_RETURN_CONTRACT,
    TRUNCATION_CONTRACT,
    FAILURE_CONTRACT,
    V3_TEMPORAL_WARM_START_METADATA,
    V4_SPATIAL_WARM_START_METADATA,
    _configure_phase_optimization,
    _select_action,
    _validate_v3_temporal_warm_start,
    _validate_v4_spatial_warm_start,
    _validate_resume_best_artifact,
    _validate_resume_payload,
    joint_gate_decision,
    linear_decision_schedule,
    parse_args,
    scheduled_bc_weight,
    scheduled_epsilon,
    scheduled_teacher_mixture,
    td_learning_enabled,
    validation_quality_gate,
)


def _schedule_args():
    return SimpleNamespace(
        temporal_epsilon_start=0.25,
        temporal_epsilon_end=0.05,
        temporal_epsilon_warmup_decisions=0,
        temporal_epsilon_decay_decisions=10_000,
        joint_epsilon_start=0.20,
        joint_epsilon_end=0.05,
        joint_epsilon_warmup_decisions=0,
        joint_epsilon_decay_decisions=20_000,
        temporal_teacher_mixture_start=0.50,
        temporal_teacher_mixture_end=0.0,
        temporal_teacher_mixture_warmup_decisions=0,
        temporal_teacher_mixture_decay_decisions=10_000,
        temporal_bc_start=0.50,
        temporal_bc_end=0.05,
        temporal_bc_warmup_decisions=0,
        temporal_bc_decay_decisions=10_000,
        joint_teacher_mixture_start=0.25,
        joint_teacher_mixture_end=0.0,
        joint_teacher_mixture_warmup_decisions=2_000,
        joint_teacher_mixture_decay_decisions=8_000,
        joint_bc_start=0.20,
        joint_bc_end=0.02,
        joint_bc_warmup_decisions=5_000,
        joint_bc_decay_decisions=20_000,
    )


def _parse_args(*extra):
    return parse_args(
        (
            "--selector-checkpoint",
            str(Path("selector.pth")),
            "--output-dir",
            str(Path("results/test")),
            *extra,
        )
    )


class PhaseScheduleTests(unittest.TestCase):
    def test_linear_schedule_boundaries_and_saturation(self):
        self.assertEqual(
            linear_decision_schedule(
                0, start=0.5, end=0.1, warmup=10, decay=100
            ),
            0.5,
        )
        self.assertEqual(
            linear_decision_schedule(
                10, start=0.5, end=0.1, warmup=10, decay=100
            ),
            0.5,
        )
        self.assertAlmostEqual(
            linear_decision_schedule(
                60, start=0.5, end=0.1, warmup=10, decay=100
            ),
            0.3,
        )
        self.assertEqual(
            linear_decision_schedule(
                10_000, start=0.5, end=0.1, warmup=10, decay=100
            ),
            0.1,
        )

    def test_temporal_and_joint_epsilon_use_independent_phase_clocks(self):
        args = _schedule_args()
        phase_decisions = {phase: 0 for phase in PHASE_ORDER}
        phase_decisions["imitation"] = 4_190

        self.assertEqual(
            scheduled_epsilon(phase_decisions["temporal"], args, "temporal"),
            0.25,
        )
        phase_decisions["temporal"] = 10_000
        self.assertEqual(
            scheduled_epsilon(phase_decisions["temporal"], args, "temporal"),
            0.05,
        )
        self.assertEqual(
            scheduled_epsilon(phase_decisions["joint"], args, "joint"),
            0.20,
        )

    def test_all_four_phase_clocks_are_independent(self):
        args = _schedule_args()
        phase_decisions = {
            "imitation": 4_190,
            "temporal": 10_000,
            "spatial": 3_000,
            "joint": 2_500,
        }

        self.assertEqual(tuple(phase_decisions), PHASE_ORDER)
        self.assertEqual(
            scheduled_epsilon(phase_decisions["imitation"], args, "imitation"),
            0.0,
        )
        self.assertEqual(
            scheduled_epsilon(phase_decisions["temporal"], args, "temporal"),
            0.05,
        )
        self.assertEqual(
            scheduled_epsilon(phase_decisions["spatial"], args, "spatial"),
            0.0,
        )
        self.assertAlmostEqual(
            scheduled_teacher_mixture(
                phase_decisions["joint"], args, "joint"
            ),
            0.234375,
        )
        self.assertEqual(
            scheduled_bc_weight(phase_decisions["spatial"], args, "spatial"),
            1.0,
        )

    def test_first_temporal_decision_uses_repaired_start_values(self):
        args = _schedule_args()

        self.assertEqual(scheduled_epsilon(0, args, "temporal"), 0.25)
        self.assertEqual(scheduled_teacher_mixture(0, args, "temporal"), 0.50)
        self.assertEqual(scheduled_bc_weight(0, args, "temporal"), 0.50)
        self.assertEqual(scheduled_teacher_mixture(0, args, "joint"), 0.25)
        self.assertEqual(scheduled_bc_weight(0, args, "joint"), 0.20)

    def test_temporal_stabilizers_anneal_to_their_configured_endpoints(self):
        args = _schedule_args()

        self.assertEqual(
            scheduled_teacher_mixture(10_000, args, "temporal"), 0.0
        )
        self.assertEqual(scheduled_bc_weight(10_000, args, "temporal"), 0.05)

    def test_joint_stabilizers_hold_then_decay_and_saturate(self):
        args = _schedule_args()

        self.assertEqual(scheduled_teacher_mixture(0, args, "joint"), 0.25)
        self.assertEqual(
            scheduled_teacher_mixture(2_000, args, "joint"), 0.25
        )
        self.assertAlmostEqual(
            scheduled_teacher_mixture(6_000, args, "joint"), 0.125
        )
        self.assertEqual(
            scheduled_teacher_mixture(10_000, args, "joint"), 0.0
        )
        self.assertEqual(scheduled_bc_weight(0, args, "joint"), 0.20)
        self.assertEqual(scheduled_bc_weight(5_000, args, "joint"), 0.20)
        self.assertAlmostEqual(
            scheduled_bc_weight(15_000, args, "joint"), 0.11
        )
        self.assertEqual(scheduled_bc_weight(25_000, args, "joint"), 0.02)

    def test_joint_td_waits_for_complete_replay_warmup(self):
        self.assertTrue(
            td_learning_enabled(
                "temporal", 0, joint_warmup_decisions=2_000
            )
        )
        self.assertFalse(
            td_learning_enabled(
                "joint", 2_000, joint_warmup_decisions=2_000
            )
        )
        self.assertTrue(
            td_learning_enabled(
                "joint", 2_001, joint_warmup_decisions=2_000
            )
        )


class QualityGateTests(unittest.TestCase):
    def test_quality_gate_enforces_return_and_mae_thresholds(self):
        reference = {"mean_return": 1_000.0, "mean_absolute_error": 5.0}

        passing = validation_quality_gate(
            {"mean_return": 900.0, "mean_absolute_error": 10.0},
            reference,
            min_return_ratio=0.90,
            max_mae_ratio=2.0,
        )
        self.assertTrue(passing["passed"])
        self.assertEqual(passing["return_threshold"], 900.0)
        self.assertEqual(passing["mae_threshold"], 10.0)

        low_return = validation_quality_gate(
            {"mean_return": 899.0, "mean_absolute_error": 5.0},
            reference,
            min_return_ratio=0.90,
            max_mae_ratio=2.0,
        )
        self.assertFalse(low_return["passed"])
        self.assertFalse(low_return["return_ok"])
        self.assertTrue(low_return["mae_ok"])

        high_mae = validation_quality_gate(
            {"mean_return": 1_000.0, "mean_absolute_error": 10.01},
            reference,
            min_return_ratio=0.90,
            max_mae_ratio=2.0,
        )
        self.assertFalse(high_mae["passed"])
        self.assertTrue(high_mae["return_ok"])
        self.assertFalse(high_mae["mae_ok"])

    def test_quality_gate_rejects_nonfinite_observations(self):
        reference = {"mean_return": 1_000.0, "mean_absolute_error": 0.0}
        gate = validation_quality_gate(
            {"mean_return": float("nan"), "mean_absolute_error": float("nan")},
            reference,
            min_return_ratio=0.90,
            max_mae_ratio=2.0,
        )

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["return_ok"])
        self.assertFalse(gate["mae_ok"])
        self.assertEqual(gate["mae_threshold"], 1.0e-6)

    def test_quality_gate_preserves_strict_and_reservation_rates(self):
        reference = {
            "mean_return": 1_000.0,
            "mean_absolute_error": 5.0,
            "strict_method_success_rate": 1.0,
            "reservation_integrity_rate": 1.0,
        }
        validation = {
            "mean_return": 1_100.0,
            "mean_absolute_error": 4.0,
            "strict_method_success_rate": 0.8,
            "reservation_integrity_rate": 1.0,
        }

        gate = validation_quality_gate(
            validation,
            reference,
            min_return_ratio=0.90,
            max_mae_ratio=2.0,
        )

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["strict_ok"])
        self.assertTrue(gate["reservation_ok"])

    def test_joint_performance_gate_is_diagnostic_during_warmup(self):
        state = joint_gate_decision(
            {
                "return_ok": False,
                "mae_ok": False,
                "strict_ok": True,
                "reservation_ok": True,
            },
            phase_decisions=2_999,
            warmup_decisions=3_000,
            prior_bad_validations=1,
            collapse_patience=2,
        )

        self.assertFalse(state["performance_enforced"])
        self.assertEqual(state["bad_validations"], 0)
        self.assertFalse(state["blocked"])

    def test_joint_performance_gate_uses_post_warmup_patience(self):
        gate = {
            "return_ok": False,
            "mae_ok": True,
            "strict_ok": True,
            "reservation_ok": True,
        }
        first = joint_gate_decision(
            gate,
            phase_decisions=3_000,
            warmup_decisions=3_000,
            prior_bad_validations=0,
            collapse_patience=2,
        )
        second = joint_gate_decision(
            gate,
            phase_decisions=3_500,
            warmup_decisions=3_000,
            prior_bad_validations=first["bad_validations"],
            collapse_patience=2,
        )

        self.assertFalse(first["blocked"])
        self.assertTrue(second["blocked"])
        self.assertEqual(second["block_reason"], "performance")

    def test_joint_safety_gate_is_immediate_during_warmup(self):
        state = joint_gate_decision(
            {
                "return_ok": True,
                "mae_ok": True,
                "strict_ok": False,
                "reservation_ok": True,
            },
            phase_decisions=1,
            warmup_decisions=3_000,
            prior_bad_validations=0,
            collapse_patience=2,
        )

        self.assertTrue(state["blocked"])
        self.assertEqual(state["block_reason"], "safety")


class PhaseOptimizationTests(unittest.TestCase):
    def test_joint_uses_reduced_learning_rates_only(self):
        class Agent:
            def __init__(self):
                self.calls = []

            def set_learning_rate_scales(self, *, main, spatial):
                self.calls.append((main, spatial))

        args = SimpleNamespace(
            joint_main_lr_scale=0.20,
            joint_spatial_lr_scale=0.10,
        )
        agent = Agent()

        for phase in ("imitation", "temporal", "spatial", "joint"):
            _configure_phase_optimization(agent, args, phase)

        self.assertEqual(
            agent.calls,
            [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.20, 0.10)],
        )


class TrainerInterfaceTests(unittest.TestCase):
    def test_select_action_forwards_mixture_and_retention_controls(self):
        class Agent:
            def __init__(self):
                self.received = None

            def select_action(
                self,
                state,
                eps,
                teacher_forcing,
                teacher_probability,
                retain_teacher_label,
            ):
                self.received = (
                    state,
                    eps,
                    teacher_forcing,
                    teacher_probability,
                    retain_teacher_label,
                )
                return "option"

        agent = Agent()
        selected = _select_action(
            agent,
            "state",
            epsilon=0.25,
            teacher_forcing=False,
            teacher_probability=0.5,
            retain_teacher_label=True,
        )

        self.assertEqual(selected, "option")
        self.assertEqual(agent.received, ("state", 0.25, False, 0.5, True))

    def test_repaired_curriculum_defaults_are_safe(self):
        args = _parse_args()

        self.assertEqual(args.imitation_episodes, 50)
        self.assertEqual(args.temporal_episodes, 150)
        self.assertEqual(args.spatial_episodes, 50)
        self.assertEqual(args.joint_episodes, 250)
        self.assertEqual(args.temporal_epsilon_start, 0.25)
        self.assertEqual(args.joint_epsilon_start, 0.05)
        self.assertEqual(args.joint_epsilon_end, 0.05)
        self.assertEqual(args.temporal_teacher_mixture_start, 0.50)
        self.assertEqual(args.joint_teacher_mixture_start, 0.25)
        self.assertEqual(args.joint_teacher_mixture_end, 0.0)
        self.assertEqual(args.joint_teacher_mixture_warmup_decisions, 2_000)
        self.assertEqual(args.joint_teacher_mixture_decay_decisions, 8_000)
        self.assertEqual(args.temporal_bc_start, 0.50)
        self.assertEqual(args.joint_bc_start, 0.20)
        self.assertEqual(args.joint_bc_end, 0.02)
        self.assertEqual(args.joint_bc_warmup_decisions, 5_000)
        self.assertEqual(args.joint_updates_per_macro, 1)
        self.assertEqual(args.joint_td_warmup_decisions, 2_000)
        self.assertEqual(args.joint_main_lr_scale, 0.10)
        self.assertEqual(args.joint_spatial_lr_scale, 0.10)
        self.assertEqual(args.joint_spatial_freeze_decisions, 5_000)
        self.assertEqual(args.spatial_min_return_ratio, 0.90)
        self.assertEqual(args.spatial_max_mae_ratio, 2.0)
        self.assertEqual(args.joint_min_return_ratio, 0.90)
        self.assertEqual(args.joint_max_mae_ratio, 2.0)
        self.assertEqual(args.joint_gate_warmup_decisions, 3_000)
        self.assertEqual(args.joint_collapse_patience, 2)
        self.assertEqual(args.teacher_coefficient, 0.0)
        self.assertEqual(args.replay_sampling, "mode_balanced")
        self.assertTrue(args.reset_optimizer_on_phase_transition)
        self.assertEqual(args.temporal_min_validation_strict, 1.0)
        self.assertEqual(args.temporal_collapse_patience, 2)

    def test_warm_start_accepts_only_spatial_then_joint_schedule(self):
        args = _parse_args(
            "--warm-start-temporal",
            "temporal-end.pth",
            "--imitation-episodes",
            "0",
            "--temporal-episodes",
            "0",
            "--spatial-episodes",
            "50",
            "--joint-episodes",
            "250",
        )

        self.assertEqual(args.warm_start_temporal, Path("temporal-end.pth"))
        self.assertEqual(args.imitation_episodes, 0)
        self.assertEqual(args.temporal_episodes, 0)

        with self.assertRaises(SystemExit):
            _parse_args(
                "--warm-start-temporal",
                "temporal-end.pth",
                "--temporal-episodes",
                "1",
            )

    def test_resume_and_warm_start_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            _parse_args(
                "--resume",
                "latest.pth",
                "--warm-start-temporal",
                "temporal-end.pth",
            )

    def test_spatial_warm_start_skips_completed_phases(self):
        args = _parse_args(
            "--warm-start-spatial",
            "spatial-end.pth",
            "--imitation-episodes",
            "0",
            "--temporal-episodes",
            "0",
            "--spatial-episodes",
            "0",
            "--joint-episodes",
            "250",
        )

        self.assertEqual(args.warm_start_spatial, Path("spatial-end.pth"))
        self.assertEqual(args.spatial_episodes, 0)

        with self.assertRaises(SystemExit):
            _parse_args(
                "--warm-start-spatial",
                "spatial-end.pth",
                "--imitation-episodes",
                "0",
                "--temporal-episodes",
                "0",
                "--spatial-episodes",
                "1",
            )

    def test_fresh_run_requires_temporal_phase_but_v4_resume_preserves_schedule(self):
        with self.assertRaises(SystemExit):
            _parse_args("--temporal-episodes", "0")

        resumed = _parse_args(
            "--resume",
            "latest.pth",
            "--imitation-episodes",
            "0",
            "--temporal-episodes",
            "0",
            "--spatial-episodes",
            "50",
            "--joint-episodes",
            "250",
        )
        self.assertEqual(resumed.temporal_episodes, 0)


class V3WarmStartValidationTests(unittest.TestCase):
    def test_synthetic_v3_temporal_checkpoint_is_authenticated(self):
        args = _parse_args(
            "--warm-start-temporal",
            "temporal-end.pth",
            "--imitation-episodes",
            "0",
            "--temporal-episodes",
            "0",
        )
        validation = {
            "mean_return": 1_491.24,
            "mean_absolute_error": 3.475,
            "strict_method_success_rate": 1.0,
            "reservation_integrity_rate": 1.0,
        }
        payload = {
            **V3_TEMPORAL_WARM_START_METADATA,
            "resumable": True,
            "completed_temporal_episodes": 150,
            "completed_joint_episodes": 0,
            "completed_training_episodes": 200,
            "training_seed": args.seed,
            "training_lambda": args.lam,
            "training_mu": args.mu,
            "macro_return_contract": RAW_MACRO_RETURN_CONTRACT,
            "truncation_contract": TRUNCATION_CONTRACT,
            "failure_contract": FAILURE_CONTRACT,
            "gamma": args.gamma,
            "reward_scale": args.reward_scale,
            "validation_seeds": tuple(args.validation_seeds),
            "validation_steps": args.validation_steps,
            "selector_deployment_digest": "synthetic-selector-digest",
            "geometry": {
                "grid_rows": args.grid_rows,
                "grid_cols": args.grid_cols,
                "requested_exit_width": args.exit_width,
            },
            "fully_learned_config": {},
            "validation": validation,
            "agent_checkpoint_state": {"Q_local": {"weight": "sentinel"}},
        }

        with TemporaryDirectory() as directory:
            source = Path(directory) / "temporal-end.pth"
            source.write_bytes(b"synthetic-v3-checkpoint")
            with unittest.mock.patch(
                "train_fully_learned_track_b.selector_deployment_digest",
                return_value="synthetic-selector-digest",
            ):
                reference, provenance = _validate_v3_temporal_warm_start(
                    payload, args, {}, source
                )

        self.assertEqual(reference["mean_return"], 1_491.24)
        self.assertEqual(reference["mean_absolute_error"], 3.475)
        self.assertEqual(reference["source"], "v3_temporal_end")
        self.assertEqual(provenance["source_trainer_version"], 2)
        self.assertEqual(len(provenance["source_sha256"]), 64)

    def test_synthetic_warm_start_rejects_wrong_phase(self):
        args = _parse_args(
            "--warm-start-temporal",
            "temporal-end.pth",
            "--imitation-episodes",
            "0",
            "--temporal-episodes",
            "0",
        )
        payload = {
            **V3_TEMPORAL_WARM_START_METADATA,
            "training_phase": "joint",
        }

        with TemporaryDirectory() as directory:
            source = Path(directory) / "temporal-end.pth"
            source.write_bytes(b"wrong-phase")
            with unittest.mock.patch(
                "train_fully_learned_track_b.selector_deployment_digest",
                return_value="synthetic-selector-digest",
            ):
                with self.assertRaisesRegex(ValueError, "training_phase"):
                    _validate_v3_temporal_warm_start(
                        payload, args, {}, source
                    )


class V4SpatialWarmStartValidationTests(unittest.TestCase):
    def test_synthetic_v4_spatial_checkpoint_is_authenticated(self):
        args = _parse_args(
            "--warm-start-spatial",
            "spatial-end.pth",
            "--imitation-episodes",
            "0",
            "--temporal-episodes",
            "0",
            "--spatial-episodes",
            "0",
            "--validation-seeds",
            "10000",
            "10001",
            "10002",
        )
        validation = {
            "episode": 50,
            "phase": "spatial",
            "mean_return": 1_484.77,
            "mean_absolute_error": 3.658,
            "strict_method_success_rate": 1.0,
            "reservation_integrity_rate": 1.0,
            "quality_gate": {"passed": True},
        }
        spatial_reference = {
            key: validation[key]
            for key in (
                "mean_return",
                "mean_absolute_error",
                "strict_method_success_rate",
                "reservation_integrity_rate",
            )
        }
        spatial_reference["source"] = "spatial_end"
        temporal_reference = {
            "mean_return": 1_491.24,
            "mean_absolute_error": 3.475,
            "strict_method_success_rate": 1.0,
            "reservation_integrity_rate": 1.0,
            "source": "v3_temporal_end",
        }
        payload = {
            **V4_SPATIAL_WARM_START_METADATA,
            "resumable": True,
            "quality_gate_blocked": False,
            "completed_spatial_episodes": 50,
            "completed_joint_episodes": 0,
            "completed_training_episodes": 50,
            "training_seed": args.seed,
            "training_lambda": args.lam,
            "training_mu": args.mu,
            "macro_return_contract": RAW_MACRO_RETURN_CONTRACT,
            "truncation_contract": TRUNCATION_CONTRACT,
            "failure_contract": FAILURE_CONTRACT,
            "max_steps": args.max_steps,
            "max_defer_steps": args.max_defer_steps,
            "gamma": args.gamma,
            "reward_scale": args.reward_scale,
            "failure_penalty": args.failure_penalty,
            "validation_seeds": tuple(args.validation_seeds),
            "validation_steps": args.validation_steps,
            "target_window": args.target_window,
            "selector_deployment_digest": "synthetic-selector-digest",
            "geometry": {
                "grid_rows": args.grid_rows,
                "grid_cols": args.grid_cols,
                "requested_exit_width": args.exit_width,
            },
            "fully_learned_config": {},
            "validation": validation,
            "temporal_reference_validation": temporal_reference,
            "spatial_reference_validation": spatial_reference,
            "agent_checkpoint_state": {
                "training_phase": "spatial",
                "Q_local": {"weight": "sentinel"},
            },
        }

        with TemporaryDirectory() as directory:
            source = Path(directory) / "spatial-end.pth"
            source.write_bytes(b"synthetic-v4-spatial-checkpoint")
            with unittest.mock.patch(
                "train_fully_learned_track_b.selector_deployment_digest",
                return_value="synthetic-selector-digest",
            ):
                temporal, spatial, provenance = (
                    _validate_v4_spatial_warm_start(
                        payload, args, {}, source
                    )
                )

        self.assertEqual(temporal, temporal_reference)
        self.assertEqual(spatial["mean_return"], 1_484.77)
        self.assertEqual(spatial["source"], "v4_spatial_end")
        self.assertEqual(provenance["source_trainer_version"], 3)
        self.assertEqual(len(provenance["source_sha256"]), 64)

    def test_spatial_warm_start_rejects_failed_quality_gate(self):
        args = _parse_args(
            "--warm-start-spatial",
            "spatial-end.pth",
            "--imitation-episodes",
            "0",
            "--temporal-episodes",
            "0",
            "--spatial-episodes",
            "0",
        )
        payload = {
            **V4_SPATIAL_WARM_START_METADATA,
            "validation": {"quality_gate": {"passed": False}},
        }

        with TemporaryDirectory() as directory:
            source = Path(directory) / "spatial-end.pth"
            source.write_bytes(b"failed-spatial-checkpoint")
            with unittest.mock.patch(
                "train_fully_learned_track_b.selector_deployment_digest",
                return_value="synthetic-selector-digest",
            ):
                with self.assertRaisesRegex(ValueError, "validation_quality_gate"):
                    _validate_v4_spatial_warm_start(
                        payload, args, {}, source
                    )


class ResumeGateContractTests(unittest.TestCase):
    def test_gate_failed_diagnostic_cannot_resume_past_blocker(self):
        args = _parse_args()
        payload = {
            "fully_learned_checkpoint_schema_version": (
                FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION
            ),
            "fully_learned_trainer_version": FULLY_LEARNED_TRAINER_VERSION,
            "checkpoint_kind": "resumable_latest",
            "resumable": True,
            "quality_gate_blocked": True,
        }

        with self.assertRaisesRegex(ValueError, "gate-failed diagnostic"):
            _validate_resume_payload(payload, args, {})

    def test_joint_resume_without_prior_passing_best_does_not_require_best_file(self):
        payload = {
            "completed_joint_episodes": 10,
            "best_score": (-1.0, -float("inf")),
        }
        with TemporaryDirectory() as directory:
            self.assertIsNone(
                _validate_resume_best_artifact(payload, Path(directory))
            )

    def test_joint_resume_with_recorded_passing_best_requires_artifact(self):
        payload = {
            "completed_joint_episodes": 10,
            "best_score": (1.0, 1_400.0),
        }
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "records a passing"):
                _validate_resume_best_artifact(payload, Path(directory))


if __name__ == "__main__":
    unittest.main()
