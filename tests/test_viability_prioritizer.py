from dataclasses import replace
import math
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import torch

from PSLAP.dynamic_yard import BlockView
from PSLAP.viability import RecoveryState
from PSLAP.viability_critic import (
    CounterfactualViabilityCritic,
    ViabilityCriticEnsemble,
)
from PSLAP.viability_prioritizer import (
    StatePriorityBatch,
    StatePriorityEntry,
    VIABILITY_PRIORITY_PROTOCOL,
    ViabilityCriticPrioritizer,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CALIBRATED_CHECKPOINT = (
    REPOSITORY_ROOT / "results" / "viability-critic-v1-seed0" / "best.pth"
)
CALIBRATED_MANIFEST = (
    REPOSITORY_ROOT
    / "results"
    / "viability-critic-v1-seed0"
    / "data"
    / "dataset-manifest.json"
)


def direct_recovery_state() -> RecoveryState:
    return RecoveryState(
        rows=1,
        cols=3,
        traversable=frozenset({(0, 0), (0, 1), (0, 2)}),
        storage_cells=frozenset({(0, 1)}),
        exits=((0, 2),),
        blocks=(BlockView("A", (0, 1), 10.0),),
        agent_position=(0, 0),
    )


def empty_recovery_state() -> RecoveryState:
    return RecoveryState(
        rows=2,
        cols=2,
        traversable=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
        storage_cells=frozenset({(0, 1), (1, 0)}),
        exits=((1, 1),),
        blocks=(),
        agent_position=(0, 0),
    )


def tiny_prioritizer() -> ViabilityCriticPrioritizer:
    torch.manual_seed(9107)
    members = tuple(
        CounterfactualViabilityCritic(
            graph_hidden_dim=8,
            graph_embedding_dim=6,
            message_passing_steps=1,
            head_hidden_dim=5,
        )
        for _ in range(2)
    )
    return ViabilityCriticPrioritizer(
        ensemble=ViabilityCriticEnsemble(members),
        threshold=0.5,
        timing_scale=100.0,
        lcb_scale=2.0,
        checkpoint_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        split_manifest_sha256="c" * 64,
        device="cpu",
    )


def priority_entry(
    original_index: int,
    key: str,
    state: RecoveryState,
) -> StatePriorityEntry:
    return StatePriorityEntry(
        original_index=original_index,
        key=key,
        state=state,
        safety_probability_mean=0.8,
        safety_probability_std=0.1,
        safety_probability_lcb=0.6,
        recovery_rank_mean=2.0,
        recovery_rank_std=0.25,
        primitive_steps_mean=8.0,
        primitive_steps_std=1.0,
        priority_pass=True,
    )


class StatePriorityBatchContractTests(unittest.TestCase):
    def setUp(self):
        self.entries = (
            priority_entry(0, "first", direct_recovery_state()),
            priority_entry(1, "second", empty_recovery_state()),
        )

    def test_rejects_incomplete_duplicate_and_out_of_range_permutations(self):
        for ordered_indices in ((0,), (0, 0), (0, 2)):
            with self.subTest(ordered_indices=ordered_indices):
                with self.assertRaisesRegex(ValueError, "complete permutation"):
                    StatePriorityBatch(
                        entries=self.entries,
                        ordered_indices=ordered_indices,
                        inference_seconds=0.1,
                        threshold=0.5,
                        checkpoint_sha256="a" * 64,
                    )

    def test_rejects_entries_that_do_not_retain_original_coordinates(self):
        reversed_coordinates = (
            replace(self.entries[0], original_index=1),
            replace(self.entries[1], original_index=0),
        )

        with self.assertRaisesRegex(ValueError, "original coordinates"):
            StatePriorityBatch(
                entries=reversed_coordinates,
                ordered_indices=(0, 1),
                inference_seconds=0.1,
                threshold=0.5,
                checkpoint_sha256="a" * 64,
            )

    def test_rejects_any_attempt_to_change_certificate_authority(self):
        common = {
            "entries": self.entries,
            "ordered_indices": (0, 1),
            "inference_seconds": 0.1,
            "threshold": 0.5,
            "checkpoint_sha256": "a" * 64,
        }
        with self.assertRaisesRegex(ValueError, "critic cannot"):
            StatePriorityBatch(
                **common,
                critic_certificate_authority=True,
            )
        with self.assertRaisesRegex(ValueError, "exact verifier"):
            StatePriorityBatch(
                **common,
                exact_verifier_authoritative=False,
            )


class ViabilityPrioritizerTests(unittest.TestCase):
    def test_empty_input_is_a_valid_empty_permutation(self):
        prioritizer = tiny_prioritizer()

        result = prioritizer.prioritize(())

        self.assertEqual(result.entries, ())
        self.assertEqual(result.ordered_indices, ())
        self.assertEqual(result.ordered_entries, ())
        self.assertEqual(result.priority_pass_count, 0)
        self.assertEqual(result.protocol, VIABILITY_PRIORITY_PROTOCOL)
        self.assertFalse(result.critic_certificate_authority)
        self.assertTrue(result.exact_verifier_authoritative)

    def test_input_validation_fails_before_scoring(self):
        prioritizer = tiny_prioritizer()
        state = direct_recovery_state()

        with self.assertRaisesRegex(ValueError, "unique"):
            prioritizer.prioritize((("same", state), ("same", state)))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            prioritizer.prioritize((("", state),))
        with self.assertRaisesRegex(TypeError, "RecoveryState"):
            prioritizer.prioritize((("bad", object()),))

    def test_float32_overflowing_feature_fails_closed(self):
        prioritizer = tiny_prioritizer()
        state = direct_recovery_state()
        extreme = replace(
            state,
            blocks=(BlockView("A", (0, 1), 1.0e308),),
        )

        with self.assertRaisesRegex(ValueError, "non-finite float32 feature"):
            prioritizer.prioritize((("extreme", extreme),))

    def test_scores_are_batch_permutation_invariant_and_keep_coordinates(self):
        prioritizer = tiny_prioritizer()
        direct = direct_recovery_state()
        empty = empty_recovery_state()

        forward = prioritizer.prioritize((("direct", direct), ("empty", empty)))
        reverse = prioritizer.prioritize((("empty", empty), ("direct", direct)))

        self.assertEqual(
            tuple(entry.original_index for entry in forward.entries),
            (0, 1),
        )
        self.assertEqual(
            tuple(sorted(forward.ordered_indices)),
            (0, 1),
        )
        forward_by_key = {entry.key: entry for entry in forward.entries}
        reverse_by_key = {entry.key: entry for entry in reverse.entries}
        for key in forward_by_key:
            left = forward_by_key[key]
            right = reverse_by_key[key]
            for field in (
                "safety_probability_mean",
                "safety_probability_std",
                "safety_probability_lcb",
                "recovery_rank_mean",
                "recovery_rank_std",
                "primitive_steps_mean",
                "primitive_steps_std",
            ):
                self.assertAlmostEqual(
                    getattr(left, field),
                    getattr(right, field),
                    places=6,
                )
            self.assertEqual(left.priority_pass, right.priority_pass)

        expected_order = tuple(
            entry.key
            for entry in sorted(
                forward.entries,
                key=lambda entry: (
                    not entry.priority_pass,
                    -entry.safety_probability_lcb,
                    entry.key,
                ),
            )
        )
        self.assertEqual(
            tuple(entry.key for entry in forward.ordered_entries),
            expected_order,
        )


class CalibratedArtifactSmokeTests(unittest.TestCase):
    def require_artifact(self):
        if not CALIBRATED_CHECKPOINT.is_file() or not CALIBRATED_MANIFEST.is_file():
            self.skipTest("local calibrated viability-critic artifact is absent")

    def test_authenticated_artifact_loads_and_returns_finite_non_authoritative_scores(self):
        self.require_artifact()

        prioritizer = ViabilityCriticPrioritizer.from_checkpoint(
            CALIBRATED_CHECKPOINT,
            dataset_manifest=CALIBRATED_MANIFEST,
            device="cpu",
        )
        result = prioritizer.prioritize(
            (("direct", direct_recovery_state()), ("empty", empty_recovery_state()))
        )

        self.assertEqual(len(prioritizer.ensemble.members), 5)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", prioritizer.checkpoint_sha256))
        self.assertTrue(all(not parameter.requires_grad for parameter in prioritizer.ensemble.parameters()))
        self.assertFalse(prioritizer.ensemble.training)
        self.assertEqual(tuple(sorted(result.ordered_indices)), (0, 1))
        self.assertFalse(result.critic_certificate_authority)
        self.assertTrue(result.exact_verifier_authoritative)
        for entry in result.entries:
            self.assertTrue(
                all(
                    math.isfinite(value)
                    for value in (
                        entry.safety_probability_mean,
                        entry.safety_probability_std,
                        entry.safety_probability_lcb,
                        entry.recovery_rank_mean,
                        entry.recovery_rank_std,
                        entry.primitive_steps_mean,
                        entry.primitive_steps_std,
                    )
                )
            )
            self.assertGreaterEqual(entry.safety_probability_lcb, 0.0)
            self.assertLessEqual(entry.safety_probability_lcb, 1.0)

    def test_pinned_checkpoint_identity_rejects_wrong_artifact(self):
        self.require_artifact()

        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            ViabilityCriticPrioritizer.from_checkpoint(
                CALIBRATED_CHECKPOINT,
                dataset_manifest=CALIBRATED_MANIFEST,
                expected_checkpoint_sha256="0" * 64,
            )

    def test_loader_rejects_broken_deployment_and_calibration_contracts(self):
        self.require_artifact()
        payload = torch.load(
            CALIBRATED_CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
        broken_kind = {**payload, "checkpoint_kind": "training"}
        broken_features = {
            **payload,
            "node_feature_names": list(reversed(payload["node_feature_names"])),
        }
        broken_selection = {
            **payload,
            "calibration": {
                **payload["calibration"],
                "selection": {
                    **payload["calibration"]["selection"],
                    "ready": False,
                },
            },
        }
        broken_test_confirmation = {
            **payload,
            "calibration": {
                **payload["calibration"],
                "independent_test_confirmation": False,
            },
        }
        cases = (
            (broken_kind, "incompatible viability critic checkpoint"),
            (broken_features, "incompatible viability critic checkpoint"),
            (broken_selection, "calibration is not ready"),
            (broken_test_confirmation, "independent test confirmation"),
        )
        for broken, message in cases:
            with self.subTest(message=message), patch(
                "PSLAP.viability_prioritizer.torch.load",
                return_value=broken,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    ViabilityCriticPrioritizer.from_checkpoint(
                        CALIBRATED_CHECKPOINT,
                        dataset_manifest=CALIBRATED_MANIFEST,
                        device="cpu",
                    )


if __name__ == "__main__":
    unittest.main()
