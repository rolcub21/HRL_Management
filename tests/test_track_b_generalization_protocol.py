from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from compare_track_b_generalization import (
    DEFAULT_METHODS,
    FULL_METHOD,
    STRESS_STAGE_SEEDS,
    ZERO_SHOT_METHOD,
    audit_stress_activation,
    parse_args,
    validate_mixed_checkpoint,
    validate_zero_shot_checkpoint,
    validate_suite_separation,
    zero_shot_shift_axes,
    summarize,
)
from example.track_b_regimes import (
    MIXED_OOD_V1,
    MIXED_TRAIN_V1,
    STRESS_V1,
    canonical_manifest,
)
from fully_learned_hierarchy import (
    FULLY_LEARNED_ACTION_INTERFACE,
    FULLY_LEARNED_BACKUP_VERSION,
    FULLY_LEARNED_CANDIDATE_FEATURE_VERSION,
    FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION,
    FULLY_LEARNED_CONTROLLER_ARCHITECTURE,
    FULLY_LEARNED_NETWORK_ARCHITECTURE,
    FULLY_LEARNED_REPLAY_VERSION,
)
from PSLAP.checkpoint_identity import selector_deployment_digest
from train_fully_learned_track_b_mixed import (
    BASE_PROVENANCE_FIELDS,
    MIXED_BASE_DEPLOYMENT_CONTRACT,
    MIXED_CHECKPOINT_SCHEMA_VERSION,
    MIXED_CURRICULUM,
    MIXED_INSTANCE_SEED_CONTRACT,
    MIXED_METHOD,
    MIXED_MODEL_SELECTION_CONTRACT,
    MIXED_REPLAY_CONTRACT,
    MIXED_SAMPLER_CONTRACT,
    MIXED_TARGET_SYNC_CONTRACT,
    MIXED_TRAINER_VERSION,
    MIXED_TRANSFER_CONTRACT,
    _json_digest,
    score_validation,
    synchronize_imported_target,
)
from train_fully_learned_track_b import (
    CURRICULUM_CONTRACT,
    FAILURE_CONTRACT,
    FAILURE_PENALTY_CONTRACT,
    FULLY_LEARNED_TRAINER_VERSION,
    MODEL_SELECTION_CONTRACT,
    RAW_MACRO_RETURN_CONTRACT,
    TRUNCATION_CONTRACT,
)


BASELINE = "nearest_free"


def _validation(mean_return, mae=5.0, *, regime_id="r"):
    return {
        "per_regime": {
            regime_id: {
                "mean_return": float(mean_return),
                "mean_absolute_error": float(mae),
                "strict_method_success_rate": 1.0,
                "reservation_integrity_rate": 1.0,
            }
        }
    }


def _selector_payload():
    return {
        "selector_architecture": "test",
        "selector_feature_version": 5,
        "config": {"test": True},
        "deployment_state_dict": {"weight": torch.tensor([1.0])},
    }


def _checkpoint_payload():
    selector = _selector_payload()
    selector_digest = selector_deployment_digest(selector)
    anchor = MIXED_TRAIN_V1[0]
    base_fields = {
        "fully_learned_checkpoint_schema_version": (
            FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION
        ),
        "fully_learned_trainer_version": FULLY_LEARNED_TRAINER_VERSION,
        "curriculum_contract": CURRICULUM_CONTRACT,
        "checkpoint_kind": "deployment_best",
        "resumable": False,
        "controller_architecture": FULLY_LEARNED_CONTROLLER_ARCHITECTURE,
        "controller_action_interface": FULLY_LEARNED_ACTION_INTERFACE,
        "network_architecture": FULLY_LEARNED_NETWORK_ARCHITECTURE,
        "replay_version": FULLY_LEARNED_REPLAY_VERSION,
        "backup_version": FULLY_LEARNED_BACKUP_VERSION,
        "candidate_feature_version": FULLY_LEARNED_CANDIDATE_FEATURE_VERSION,
        "macro_return_contract": RAW_MACRO_RETURN_CONTRACT,
        "truncation_contract": TRUNCATION_CONTRACT,
        "failure_contract": FAILURE_CONTRACT,
        "failure_penalty_contract": FAILURE_PENALTY_CONTRACT,
        "model_selection_contract": MODEL_SELECTION_CONTRACT,
        "common_continuation": True,
        "deployment_reg_policy_query": False,
        "selector_frozen_independent_replay_disabled": True,
        "gamma": 0.99,
        "reward_scale": 0.01,
        "failure_penalty": -50.0,
        "training_lambda": anchor.arrival_rate,
        "training_mu": anchor.proc_mean,
        "training_seed": 0,
        "selector_deployment_digest": selector_digest,
        "geometry": anchor.provenance()["geometry"],
    }
    base_fields = json.loads(json.dumps(base_fields))
    assert set(base_fields) == set(BASE_PROVENANCE_FIELDS)
    return {
        "mixed_checkpoint_schema_version": MIXED_CHECKPOINT_SCHEMA_VERSION,
        "mixed_trainer_version": MIXED_TRAINER_VERSION,
        "method": MIXED_METHOD,
        "checkpoint_kind": "deployment_best",
        "resumable": False,
        "curriculum_contract": MIXED_CURRICULUM,
        "replay_sampling_contract": MIXED_REPLAY_CONTRACT,
        "regime_sampler_contract": MIXED_SAMPLER_CONTRACT,
        "instance_seed_contract": MIXED_INSTANCE_SEED_CONTRACT,
        "transfer_contract": MIXED_TRANSFER_CONTRACT,
        "warm_start_target_sync": MIXED_TARGET_SYNC_CONTRACT,
        "model_selection_contract": MIXED_MODEL_SELECTION_CONTRACT,
        "macro_return_contract": RAW_MACRO_RETURN_CONTRACT,
        "truncation_contract": TRUNCATION_CONTRACT,
        "failure_contract": FAILURE_CONTRACT,
        "failure_penalty_contract": FAILURE_PENALTY_CONTRACT,
        "controller_architecture": FULLY_LEARNED_CONTROLLER_ARCHITECTURE,
        "controller_action_interface": FULLY_LEARNED_ACTION_INTERFACE,
        "network_architecture": FULLY_LEARNED_NETWORK_ARCHITECTURE,
        "replay_version": FULLY_LEARNED_REPLAY_VERSION,
        "backup_version": FULLY_LEARNED_BACKUP_VERSION,
        "candidate_feature_version": FULLY_LEARNED_CANDIDATE_FEATURE_VERSION,
        "training_phase": "joint",
        "common_continuation": True,
        "teacher_coefficient": 0.0,
        "accept_policy_lock": False,
        "spatial_trainable": True,
        "frozen_reg_runtime_policy_query": False,
        "deployment_reg_policy_query": False,
        "replay_sampling": "regime_mode_balanced",
        "selector_checkpoint": selector,
        "selector_deployment_digest": selector_digest,
        "base_deployment_provenance": {
            "contract": MIXED_BASE_DEPLOYMENT_CONTRACT,
            "fields": base_fields,
            "sha256": _json_digest(base_fields),
        },
        "base_checkpoint": "/authenticated/base-v4-1.pth",
        "base_checkpoint_sha256": "a" * 64,
        "base_training_seed": 0,
        "agent_checkpoint_state": {
            "training_phase": "joint",
            "Q_local": {},
            "Q_target": {},
        },
        "fully_learned_config": {},
        "training_manifest": canonical_manifest(MIXED_TRAIN_V1),
    }


def _zero_shot_payload(mixed_payload=None):
    mixed_payload = mixed_payload or _checkpoint_payload()
    fields = deepcopy(
        mixed_payload["base_deployment_provenance"]["fields"]
    )
    return {
        **fields,
        "selector_checkpoint": deepcopy(mixed_payload["selector_checkpoint"]),
        "agent_checkpoint_state": {
            "training_phase": "joint",
            "Q_local": {},
            "Q_target": {},
        },
        "fully_learned_config": {"failure_penalty": -50.0},
        "failure_boundary_penalty": -50.0,
        "completed_joint_episodes": 1,
        "training_phase": "joint",
        "teacher_action_mixture": 0.0,
        "teacher_coefficient": 0.0,
        "accept_policy_lock": False,
        "spatial_trainable": True,
        "frozen_reg_runtime_policy_query": False,
        "teacher_prior_application": "behavior_selection_only_v1",
        "td_teacher_prior": False,
        "assignment_source_independent_replay": False,
        "replay_sampling": "mode_balanced",
    }


def _run(
    regime_id,
    method,
    seed,
    *,
    return_=100.0,
    mae=5.0,
    manifest_block_count=40,
    delivery_count=None,
):
    if delivery_count is None:
        delivery_count = manifest_block_count
    steps = 100
    relocations = 1
    return {
        "regime_id": regime_id,
        "method": method,
        "eval_seed": int(seed),
        "instance_id": f"instance-{regime_id}-{seed}",
        "schedule_id": f"schedule-{regime_id}-{seed}",
        "return": float(return_),
        "strict_method_success": 1.0,
        "reservation_integrity": 1.0,
        "steps": steps,
        "delivery_count": delivery_count,
        "manifest_block_count": manifest_block_count,
        "storage_capacity_cells": 63,
        "completion_rate": delivery_count / manifest_block_count,
        "return_per_manifest_block": return_ / manifest_block_count,
        "return_per_delivery": return_ / delivery_count,
        "steps_per_delivery": steps / delivery_count,
        "mean_absolute_error": float(mae),
        "mean_tardiness": float(mae),
        "within_target_window_rate": 0.9,
        "relocations": relocations,
        "relocations_per_100_deliveries": (
            100.0 * relocations / delivery_count
        ),
        "occupancy_pressure_metric_contract": (
            "post_transition_physical_storage_and_decision_epoch_candidates_v1"
        ),
        "peak_active_stored_fraction": 0.5,
        "peak_physical_storage_occupancy_fraction": 0.5,
        "minimum_free_storage_cells": 31,
        "storage_steps_at_or_above_80pct_occupied": 0,
        "storage_step_fraction_at_or_above_80pct_occupied": 0.0,
        "maximum_inbound_queue_count": 3,
        "minimum_live_candidate_count": 20,
        "minimum_live_accept_candidate_count": 20,
        "invalid_assignments": 0,
        "fallbacks": 0,
        "method_failure_reason": None,
    }


def _regime(regime_id, *, number_blocks=40, storage_capacity_cells=63):
    return SimpleNamespace(
        regime_id=regime_id,
        number_blocks=number_blocks,
        provenance=lambda: {
            "geometry": {"storage_cell_count": storage_capacity_cells}
        },
    )


class MixedScoreSelectionTests(unittest.TestCase):
    def test_negative_reference_return_preserves_improvement_direction(self):
        reference = _validation(-100.0)
        baseline_score = score_validation(reference, reference)
        improved = score_validation(_validation(-80.0), reference)
        degraded = score_validation(_validation(-120.0), reference)

        self.assertGreater(improved, baseline_score)
        self.assertLess(degraded, baseline_score)
        self.assertEqual(improved[2], 1.0)
        self.assertEqual(degraded[2], 0.0)

    def test_positive_reference_retains_ratio_semantics(self):
        score = score_validation(_validation(900.0), _validation(1000.0))

        self.assertAlmostEqual(score[3], 0.9)
        self.assertAlmostEqual(score[5], 0.9)
        self.assertEqual(score[2], 1.0)

    def test_score_rejects_nonfinite_or_misaligned_regimes(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            score_validation(_validation(float("nan")), _validation(100.0))
        with self.assertRaisesRegex(ValueError, "regime mismatch"):
            score_validation(
                _validation(100.0, regime_id="a"),
                _validation(100.0, regime_id="b"),
            )


class MixedCheckpointValidationTests(unittest.TestCase):
    def test_accepts_exact_sealed_deployment_payload(self):
        self.assertIsNone(validate_mixed_checkpoint(_checkpoint_payload()))

    def test_rejects_selector_digest_or_resumable_state_tampering(self):
        payload = _checkpoint_payload()
        payload["selector_deployment_digest"] = "wrong"
        with self.assertRaisesRegex(ValueError, "selector_deployment_digest"):
            validate_mixed_checkpoint(payload)

    def test_rejects_target_sync_or_base_provenance_tampering(self):
        payload = _checkpoint_payload()
        payload["warm_start_target_sync"] = "not_synced"
        with self.assertRaisesRegex(ValueError, "warm_start_target_sync"):
            validate_mixed_checkpoint(payload)

        payload = _checkpoint_payload()
        payload["base_deployment_provenance"]["fields"]["source"] = (
            "modified"
        )
        with self.assertRaisesRegex(ValueError, "base_deployment_sha256"):
            validate_mixed_checkpoint(payload)

        payload = _checkpoint_payload()
        payload["agent_checkpoint_state"]["optimizer"] = {}
        with self.assertRaisesRegex(ValueError, "deployment_state"):
            validate_mixed_checkpoint(payload)

    def test_rejects_a_different_but_canonical_training_manifest(self):
        payload = _checkpoint_payload()
        payload["training_manifest"] = canonical_manifest(MIXED_OOD_V1)

        with self.assertRaisesRegex(ValueError, "sealed mixed_train_v1"):
            validate_mixed_checkpoint(payload)

    def test_rejects_missing_or_malformed_base_file_identity(self):
        payload = _checkpoint_payload()
        payload["base_checkpoint_sha256"] = "not-a-sha256"
        with self.assertRaisesRegex(ValueError, "base_checkpoint_sha256"):
            validate_mixed_checkpoint(payload)

        payload = _checkpoint_payload()
        payload["base_training_seed"] = 9
        with self.assertRaisesRegex(ValueError, "base_training_seed"):
            validate_mixed_checkpoint(payload)


class ZeroShotComparatorAuthenticationTests(unittest.TestCase):
    def test_accepts_exact_v4_1_semantics_and_file_digest(self):
        mixed = _checkpoint_payload()
        base = _zero_shot_payload(mixed)
        with patch(
            "compare_track_b_generalization._sha256",
            return_value=mixed["base_checkpoint_sha256"],
        ):
            self.assertIsNone(
                validate_zero_shot_checkpoint(base, mixed, "base.pth")
            )

    def test_rejects_compatible_but_different_v4_1_or_file(self):
        mixed = _checkpoint_payload()
        base = _zero_shot_payload(mixed)
        base["training_seed"] = 7
        with self.assertRaisesRegex(
            ValueError, "base_deployment_provenance"
        ):
            validate_zero_shot_checkpoint(base, mixed)

        base = _zero_shot_payload(mixed)
        with patch(
            "compare_track_b_generalization._sha256",
            return_value="b" * 64,
        ):
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                validate_zero_shot_checkpoint(base, mixed, "other.pth")

    def test_audits_each_supported_shift_axis(self):
        base = _zero_shot_payload()
        shifts = {
            regime.regime_id: zero_shot_shift_axes(base, regime)
            for regime in MIXED_TRAIN_V1
        }
        self.assertEqual(shifts["anchor_10x10_b40"], ())
        self.assertEqual(shifts["narrow_10x10_b40"], ("exit_width",))
        self.assertEqual(
            shifts["compact_9x9_b36"],
            ("load", "grid_size", "exit_width", "block_count"),
        )
        self.assertEqual(
            shifts["large_11x11_b48"],
            ("load", "grid_size", "exit_width", "block_count"),
        )

    def test_rejects_transfer_to_a_different_geometry_contract(self):
        base = _zero_shot_payload()
        anchor = MIXED_TRAIN_V1[0]
        target = SimpleNamespace(
            regime_id="different_topology",
            arrival_rate=anchor.arrival_rate,
            proc_mean=anchor.proc_mean,
            grid_rows=anchor.grid_rows,
            grid_cols=anchor.grid_cols,
            number_blocks=anchor.number_blocks,
            provenance=lambda: {
                "geometry": {
                    **anchor.provenance()["geometry"],
                    "geometry_contract": "arbitrary_walled_yard_v1",
                }
            },
        )
        with self.assertRaisesRegex(ValueError, "geometry contract mismatch"):
            zero_shot_shift_axes(base, target)


class OptionalComparatorCliTests(unittest.TestCase):
    def _argv(self, output):
        return [
            "--checkpoint",
            "mixed.pth",
            "--suite",
            "mixed_train_v1",
            "--seeds",
            "1",
            "--output-dir",
            str(output),
        ]

    def test_omission_preserves_original_default_method_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(self._argv(Path(directory) / "new"))
        self.assertEqual(args.methods, DEFAULT_METHODS)
        self.assertNotIn(ZERO_SHOT_METHOD, args.methods)

    def test_checkpoint_option_activates_comparator(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(
                [
                    *self._argv(Path(directory) / "new"),
                    "--zero-shot-checkpoint",
                    "base.pth",
                ]
            )
        self.assertEqual(args.methods[-1], ZERO_SHOT_METHOD)
        self.assertEqual(args.methods.count(ZERO_SHOT_METHOD), 1)

    def test_method_cannot_be_requested_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                parse_args(
                    [
                        *self._argv(Path(directory) / "new"),
                        "--methods",
                        FULL_METHOD,
                        ZERO_SHOT_METHOD,
                    ]
                )

    def test_stress_stage_supplies_predeclared_seed_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(
                [
                    "--checkpoint",
                    "mixed.pth",
                    "--suite",
                    "stress_v1",
                    "--stress-stage",
                    "calibration",
                    "--output-dir",
                    str(Path(directory) / "new"),
                ]
            )

        self.assertEqual(args.seeds, STRESS_STAGE_SEEDS["calibration"])

    def test_stress_stage_rejects_manual_or_nonstress_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            base = [
                "--checkpoint",
                "mixed.pth",
                "--output-dir",
                str(Path(directory) / "new"),
            ]
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                parse_args(
                    [
                        *base,
                        "--suite",
                        "stress_v1",
                        "--stress-stage",
                        "holdout",
                        "--seeds",
                        "1",
                    ]
                )
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                parse_args(
                    [
                        *base,
                        "--suite",
                        "mixed_train_v1",
                        "--stress-stage",
                        "calibration",
                    ]
                )

    def test_sealed_stress_stage_requires_activation_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                parse_args(
                    [
                        "--checkpoint",
                        "mixed.pth",
                        "--suite",
                        "stress_v1",
                        "--stress-stage",
                        "calibration",
                        "--methods",
                        FULL_METHOD,
                        "--output-dir",
                        str(Path(directory) / "new"),
                    ]
                )


class OODSeparationTests(unittest.TestCase):
    def test_sealed_ood_suite_is_disjoint(self):
        overlap = validate_suite_separation(
            "mixed_ood_v1", canonical_manifest(MIXED_TRAIN_V1), MIXED_OOD_V1
        )
        self.assertEqual(overlap, [])

    def test_sealed_ood_suite_rejects_training_profile_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlaps training"):
            validate_suite_separation(
                "mixed_ood_v1",
                canonical_manifest(MIXED_TRAIN_V1),
                MIXED_TRAIN_V1,
            )

    def test_sealed_stress_suite_is_disjoint(self):
        overlap = validate_suite_separation(
            "stress_v1", canonical_manifest(MIXED_TRAIN_V1), STRESS_V1
        )
        self.assertEqual(overlap, [])

    def test_sealed_stress_suite_rejects_training_profile_overlap(self):
        with self.assertRaisesRegex(ValueError, "sealed stress_v1"):
            validate_suite_separation(
                "stress_v1",
                canonical_manifest(MIXED_TRAIN_V1),
                MIXED_TRAIN_V1,
            )


class MixedTransferInitializationTests(unittest.TestCase):
    def test_target_is_hard_synced_and_optimizer_history_is_cleared(self):
        local = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        with torch.no_grad():
            target.weight.fill_(99.0)
            target.bias.fill_(-99.0)
        optimizer = torch.optim.Adam(local.parameters(), lr=1.0e-3)
        optimizer.zero_grad(set_to_none=True)
        local(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        self.assertTrue(optimizer.state)
        agent = SimpleNamespace(
            Q_local=local,
            Q_target=target,
            optimizer=optimizer,
        )

        synchronize_imported_target(agent)

        self.assertFalse(optimizer.state)
        self.assertFalse(target.training)
        self.assertTrue(all(not item.requires_grad for item in target.parameters()))
        for name, value in local.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[name])


class PairedComparatorSummaryTests(unittest.TestCase):
    def test_complete_matrix_is_summarized_and_paired(self):
        regimes = (_regime("a"), _regime("b"))
        runs = []
        for regime in regimes:
            for seed in (1, 2):
                runs.append(
                    _run(
                        regime.regime_id,
                        FULL_METHOD,
                        seed,
                        return_=110.0,
                        mae=4.0,
                    )
                )
                runs.append(
                    _run(
                        regime.regime_id,
                        BASELINE,
                        seed,
                        return_=100.0,
                        mae=5.0,
                    )
                )

        cells, macro, paired = summarize(
            runs, regimes, (FULL_METHOD, BASELINE)
        )

        self.assertEqual(len(cells), 4)
        self.assertEqual(len(macro), 2)
        self.assertEqual(paired[0]["n"], 4)
        self.assertEqual(paired[0]["mean_return_advantage"], 10.0)
        self.assertEqual(paired[0]["mean_absolute_error_reduction"], 1.0)

    def test_cross_regime_summary_uses_manifest_normalized_return(self):
        regimes = (
            _regime("b40", number_blocks=40),
            _regime("b52", number_blocks=52),
        )
        runs = [
            _run(
                "b40",
                FULL_METHOD,
                1,
                return_=400.0,
                manifest_block_count=40,
            ),
            _run(
                "b52",
                FULL_METHOD,
                1,
                return_=520.0,
                manifest_block_count=52,
            ),
        ]

        cells, macro, paired = summarize(runs, regimes, (FULL_METHOD,))

        self.assertEqual(paired, [])
        self.assertEqual(
            [item["mean_return_per_manifest_block"] for item in cells],
            [10.0, 10.0],
        )
        self.assertEqual(
            macro[0]["macro_mean_return_per_manifest_block"], 10.0
        )
        self.assertEqual(macro[0]["macro_mean_return"], 460.0)

    def test_zero_shot_is_required_in_every_cell_and_paired_to_mixed(self):
        regimes = (_regime("a"), _regime("b"))
        methods = (FULL_METHOD, ZERO_SHOT_METHOD, BASELINE)
        runs = []
        for regime in regimes:
            for seed in (1, 2):
                runs.extend(
                    [
                        _run(
                            regime.regime_id,
                            FULL_METHOD,
                            seed,
                            return_=110.0,
                            mae=4.0,
                        ),
                        _run(
                            regime.regime_id,
                            ZERO_SHOT_METHOD,
                            seed,
                            return_=105.0,
                            mae=4.5,
                        ),
                        _run(
                            regime.regime_id,
                            BASELINE,
                            seed,
                            return_=100.0,
                            mae=5.0,
                        ),
                    ]
                )

        cells, macro, paired = summarize(runs, regimes, methods)

        self.assertEqual(len(cells), 6)
        self.assertEqual(len(macro), 3)
        by_method = {item["comparison_method"]: item for item in paired}
        self.assertEqual(by_method[ZERO_SHOT_METHOD]["n"], 4)
        self.assertEqual(
            by_method[ZERO_SHOT_METHOD]["mean_return_advantage"], 5.0
        )
        self.assertEqual(
            by_method[ZERO_SHOT_METHOD]["mean_absolute_error_reduction"],
            0.5,
        )

        missing_zero_shot = [
            item for item in runs if item["method"] != ZERO_SHOT_METHOD
        ]
        with self.assertRaisesRegex(ValueError, "missing run cells"):
            summarize(missing_zero_shot, regimes, methods)

    def test_rejects_missing_duplicate_or_unpaired_runs(self):
        regimes = (_regime("a"),)
        learned = _run("a", FULL_METHOD, 1)
        baseline = _run("a", BASELINE, 1)

        with self.assertRaisesRegex(ValueError, "missing run cells"):
            summarize([learned], regimes, (FULL_METHOD, BASELINE))
        with self.assertRaisesRegex(ValueError, "duplicate run"):
            summarize(
                [learned, deepcopy(learned), baseline],
                regimes,
                (FULL_METHOD, BASELINE),
            )
        bad_instance = deepcopy(baseline)
        bad_instance["instance_id"] = "different"
        with self.assertRaisesRegex(ValueError, "instance mismatch"):
            summarize(
                [learned, bad_instance],
                regimes,
                (FULL_METHOD, BASELINE),
            )
        bad_schedule = deepcopy(baseline)
        bad_schedule["schedule_id"] = "different"
        with self.assertRaisesRegex(ValueError, "schedule mismatch"):
            summarize(
                [learned, bad_schedule],
                regimes,
                (FULL_METHOD, BASELINE),
            )


class StressActivationAuditTests(unittest.TestCase):
    def test_uses_only_integrity_and_pressure_gates(self):
        regimes = (_regime("stress"),)
        runs = []
        for seed in (1, 2, 3):
            item = _run(
                "stress",
                "dynamic_pslap",
                seed,
                return_=-999.0,
                mae=999.0,
            )
            item.update(
                {
                    "peak_physical_storage_occupancy_fraction": 0.625,
                    "minimum_live_accept_candidate_count": 3,
                    "relocations": 1,
                }
            )
            runs.append(item)

        audit = audit_stress_activation(runs, regimes)

        self.assertTrue(audit["passed"])
        self.assertEqual(
            audit["performance_metrics_used_for_profile_acceptance"], []
        )

        runs[0]["relocations"] = 0
        failed = audit_stress_activation(runs, regimes)
        self.assertFalse(failed["passed"])
        self.assertFalse(
            failed["per_regime"][0]["criteria"][
                "relocation_observed_every_episode"
            ]
        )


if __name__ == "__main__":
    unittest.main()
