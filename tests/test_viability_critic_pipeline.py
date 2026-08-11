import copy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from PSLAP.dynamic_yard import BlockView
from PSLAP.viability import (
    RecoveryState,
    ViabilityStatus,
    analyze_recoverability,
)
from PSLAP.viability_critic import DynamicsViabilityLabel
from PSLAP.viability_dataset import (
    ViabilityExample,
    canonical_sha256,
    read_dataset,
    recovery_state_from_dict,
    recovery_state_to_dict,
    split_examples_by_layout,
    stratified_bootstrap_indices,
    write_dataset,
)
from train_viability_critic import (
    EnsemblePredictions,
    choose_lcb_threshold,
    one_sided_binomial_upper_bound,
)


def rich_safe_state() -> RecoveryState:
    return RecoveryState(
        rows=2,
        cols=3,
        traversable=frozenset(
            {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)}
        ),
        storage_cells=frozenset({(0, 1), (0, 2)}),
        exits=((1, 2),),
        blocks=(BlockView("A", (0, 1), -7.25),),
        agent_position=(0, 0),
        fixed_obstacles=frozenset({(1, 0)}),
        reserved_cells=frozenset({(0, 2)}),
        pickup_cells=frozenset({(0, 0)}),
        wait_cells=frozenset({(1, 0)}),
    )


def compact_unsafe_state() -> RecoveryState:
    # At full occupancy, the agent is trapped in the left storage branch.
    # The same compact topology is used by the exhaustive-kernel tests.
    return RecoveryState(
        rows=4,
        cols=4,
        traversable=frozenset(
            {(0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2)}
        ),
        storage_cells=frozenset({(1, 1), (2, 1), (2, 2)}),
        exits=((3, 2),),
        blocks=(
            BlockView("A", (1, 1), -10.0),
            BlockView("B", (2, 1), 0.0),
            BlockView("C", (2, 2), 10.0),
        ),
        agent_position=(1, 1),
        fixed_obstacles=frozenset({(0, 2), (1, 2)}),
        pickup_cells=frozenset({(1, 2)}),
        wait_cells=frozenset({(0, 2)}),
    )


def compact_multistep_state() -> RecoveryState:
    # This full-yard state is recoverable, but requires multiple macros.  A
    # one-node computational search budget must therefore remain UNKNOWN.
    return replace(compact_unsafe_state(), agent_position=(2, 1))


def empty_state() -> RecoveryState:
    return RecoveryState(
        rows=1,
        cols=1,
        traversable=frozenset({(0, 0)}),
        storage_cells=frozenset(),
        exits=((0, 0),),
        blocks=(),
        agent_position=(0, 0),
    )


def example_from_certificate(
    layout_id: str,
    state: RecoveryState,
    certificate,
) -> ViabilityExample:
    state_payload = recovery_state_to_dict(state)
    state_digest = canonical_sha256(state_payload)[:20]
    return ViabilityExample(
        example_id=f"{layout_id}:{state_digest}",
        layout_id=layout_id,
        state=state,
        label=DynamicsViabilityLabel.from_certificate(certificate),
        occupancy_count=len(state.blocks),
        search_order=certificate.search_order,
        explored_nodes=certificate.explored_nodes,
        generated_states=certificate.generated_states,
        exhaustive=certificate.exhaustive,
        reason=certificate.reason,
    )


def calibration_examples(
    *,
    safe_count: int,
    unsafe_count: int,
    unknown_count: int = 0,
) -> tuple[ViabilityExample, ...]:
    safe_state = rich_safe_state()
    safe_certificate = analyze_recoverability(
        safe_state,
        max_nodes=None,
        search_order="breadth_first",
    )
    unsafe_state = compact_unsafe_state()
    unsafe_certificate = analyze_recoverability(
        unsafe_state,
        max_nodes=None,
        search_order="goal_directed",
    )
    unknown_state = compact_multistep_state()
    unknown_certificate = analyze_recoverability(
        unknown_state,
        max_nodes=1,
        search_order="goal_directed",
    )
    return tuple(
        [
            example_from_certificate(
                f"cal-safe-{index}", safe_state, safe_certificate
            )
            for index in range(safe_count)
        ]
        + [
            example_from_certificate(
                f"cal-unsafe-{index}", unsafe_state, unsafe_certificate
            )
            for index in range(unsafe_count)
        ]
        + [
            example_from_certificate(
                f"cal-unknown-{index}", unknown_state, unknown_certificate
            )
            for index in range(unknown_count)
        ]
    )


def predictions_from_lcbs(values) -> EnsemblePredictions:
    lcb = np.asarray(tuple(values), dtype=np.float64)
    zeros = np.zeros_like(lcb)
    return EnsemblePredictions(
        safety_mean=lcb.copy(),
        safety_std=zeros.copy(),
        safety_lcb=lcb,
        recovery_rank_mean=zeros.copy(),
        primitive_steps_mean=zeros.copy(),
    )


class ViabilityDatasetRoundTripTests(unittest.TestCase):
    def test_recovery_state_and_example_round_trip_losslessly(self):
        state = rich_safe_state()
        certificate = analyze_recoverability(
            state,
            max_nodes=None,
            search_order="breadth_first",
        )
        self.assertIs(certificate.status, ViabilityStatus.SAFE)
        example = example_from_certificate("layout-rich", state, certificate)

        state_payload = json.loads(json.dumps(recovery_state_to_dict(state)))
        restored_state = recovery_state_from_dict(state_payload)
        restored_example = ViabilityExample.from_dict(
            json.loads(json.dumps(example.to_dict()))
        )

        self.assertEqual(restored_state, state)
        self.assertEqual(restored_example, example)
        self.assertEqual(
            recovery_state_to_dict(restored_example.state), state_payload
        )

    def test_unknown_record_survives_jsonl_round_trip_without_targets(self):
        state = compact_multistep_state()
        certificate = analyze_recoverability(
            state,
            max_nodes=1,
            search_order="goal_directed",
        )
        self.assertIs(certificate.status, ViabilityStatus.UNKNOWN)
        example = example_from_certificate("layout-unknown", state, certificate)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.jsonl"
            write_dataset(path, (example,))
            restored = read_dataset(path)

        self.assertEqual(restored, (example,))
        self.assertIs(restored[0].label.status, ViabilityStatus.UNKNOWN)
        self.assertFalse(restored[0].label.has_safety_target)
        self.assertIsNone(restored[0].label.recovery_rank)
        self.assertIsNone(restored[0].label.primitive_steps)


class ViabilityDatasetSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.safe_state = rich_safe_state()
        cls.safe_certificate = analyze_recoverability(
            cls.safe_state,
            max_nodes=None,
            search_order="breadth_first",
        )
        cls.unsafe_state = compact_unsafe_state()
        cls.unsafe_certificate = analyze_recoverability(
            cls.unsafe_state,
            max_nodes=None,
            search_order="goal_directed",
        )
        if cls.unsafe_certificate.status is not ViabilityStatus.UNSAFE:
            raise AssertionError("compact unsafe fixture must be exhaustive UNSAFE")

    def test_complete_layout_groups_are_disjoint_across_splits(self):
        examples = tuple(
            example
            for layout_index in range(8)
            for example in (
                example_from_certificate(
                    f"layout-{layout_index}",
                    self.safe_state,
                    self.safe_certificate,
                ),
                example_from_certificate(
                    f"layout-{layout_index}",
                    self.unsafe_state,
                    self.unsafe_certificate,
                ),
            )
        )

        splits, manifest = split_examples_by_layout(examples, seed=718)
        repeated, repeated_manifest = split_examples_by_layout(
            examples, seed=718
        )

        self.assertEqual(splits, repeated)
        self.assertEqual(manifest, repeated_manifest)
        layout_sets = {
            name: {example.layout_id for example in values}
            for name, values in splits.items()
        }
        for left_name, left in layout_sets.items():
            for right_name, right in layout_sets.items():
                if left_name < right_name:
                    self.assertTrue(left.isdisjoint(right))
        self.assertEqual(
            set().union(*layout_sets.values()),
            {f"layout-{index}" for index in range(8)},
        )
        self.assertEqual(
            {example.example_id for values in splits.values() for example in values},
            {example.example_id for example in examples},
        )
        for values in splits.values():
            counts = {}
            for example in values:
                counts[example.layout_id] = counts.get(example.layout_id, 0) + 1
            self.assertTrue(all(count == 2 for count in counts.values()))

    def test_bootstrap_is_reproducible_but_distinct_across_member_seeds(self):
        examples = []
        for index in range(8):
            examples.append(
                example_from_certificate(
                    f"safe-{index}", self.safe_state, self.safe_certificate
                )
            )
            examples.append(
                example_from_certificate(
                    f"unsafe-{index}",
                    self.unsafe_state,
                    self.unsafe_certificate,
                )
            )
        unknown_state = compact_multistep_state()
        unknown_certificate = analyze_recoverability(
            unknown_state,
            max_nodes=1,
            search_order="goal_directed",
        )
        examples.append(
            example_from_certificate(
                "unknown-only", unknown_state, unknown_certificate
            )
        )
        examples = tuple(examples)

        first = stratified_bootstrap_indices(examples, seed=1001)
        repeated = stratified_bootstrap_indices(examples, seed=1001)
        second = stratified_bootstrap_indices(examples, seed=1002)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertEqual(len(first["safe"]), 8)
        self.assertEqual(len(first["unsafe"]), 8)
        self.assertTrue(
            all(
                examples[index].label.status is ViabilityStatus.SAFE
                for index in first["safe"]
            )
        )
        self.assertTrue(
            all(
                examples[index].label.status is ViabilityStatus.UNSAFE
                for index in first["unsafe"]
            )
        )
        self.assertNotIn(len(examples) - 1, first["safe"] + first["unsafe"])


class ViabilityLabelPayloadContractTests(unittest.TestCase):
    def test_goal_directed_trivial_empty_yard_rank_zero_is_valid(self):
        state = empty_state()
        certificate = analyze_recoverability(
            state,
            max_nodes=None,
            search_order="goal_directed",
        )
        self.assertTrue(certificate.recovery_rank_is_exact)
        self.assertEqual(certificate.exact_recovery_rank, 0)
        example = example_from_certificate("layout-empty", state, certificate)

        restored = ViabilityExample.from_dict(example.to_dict())

        self.assertEqual(restored, example)
        self.assertEqual(restored.label.recovery_rank, 0.0)

    def test_nontrivial_exact_rank_requires_breadth_first_search(self):
        state = rich_safe_state()
        certificate = analyze_recoverability(
            state,
            max_nodes=None,
            search_order="breadth_first",
        )
        example = example_from_certificate("layout-bfs", state, certificate)
        payload = example.to_dict()
        self.assertGreater(payload["label"]["recovery_rank"], 0)
        payload["search"]["order"] = "goal_directed"

        with self.assertRaisesRegex(ValueError, "exact recovery rank"):
            ViabilityExample.from_dict(payload)

    def test_horizon_conditioned_payload_is_rejected(self):
        state = rich_safe_state()
        certificate = analyze_recoverability(
            state,
            max_nodes=None,
            search_order="breadth_first",
        )
        example = example_from_certificate("layout-horizon", state, certificate)
        payload = copy.deepcopy(example.to_dict())
        payload["search"]["max_primitive_steps"] = 5

        with self.assertRaisesRegex(ValueError, "horizon-conditioned"):
            ViabilityExample.from_dict(payload)


class ViabilityCalibrationFunctionTests(unittest.TestCase):
    def test_zero_failures_still_has_nonzero_one_sided_upper_bound(self):
        confidence = 0.95
        trials = 20

        upper = one_sided_binomial_upper_bound(
            0,
            trials,
            confidence=confidence,
        )

        self.assertGreater(upper, 0.0)
        self.assertLess(upper, 1.0)
        self.assertAlmostEqual(
            upper,
            1.0 - (1.0 - confidence) ** (1.0 / trials),
            places=12,
        )

    def test_threshold_uses_only_known_calibration_record_scores(self):
        examples = calibration_examples(
            safe_count=3,
            unsafe_count=20,
            unknown_count=1,
        )
        known_scores = (0.90, 0.80, 0.70) + (0.20,) * 20
        # The UNKNOWN score is deliberately between two safe candidates.  It
        # is retained in the calibration data but cannot define a threshold.
        predictions = predictions_from_lcbs(known_scores + (0.75,))

        selection = choose_lcb_threshold(
            examples,
            predictions,
            confidence=0.95,
            max_false_safe_upper_bound=0.15,
            min_safe_accepted=2,
        )

        self.assertTrue(selection["ready"])
        self.assertEqual(selection["threshold"], 0.70)
        self.assertIn(selection["threshold"], set(known_scores))
        self.assertNotEqual(selection["threshold"], 0.75)
        self.assertEqual(selection["safe_accepted"], 3)
        self.assertEqual(selection["false_safe_count"], 0)

    def test_no_unsafe_calibration_evidence_fails_closed(self):
        examples = calibration_examples(safe_count=3, unsafe_count=0)
        predictions = predictions_from_lcbs((0.90, 0.80, 0.70))

        selection = choose_lcb_threshold(
            examples,
            predictions,
            confidence=0.95,
            max_false_safe_upper_bound=0.50,
            min_safe_accepted=1,
        )

        self.assertFalse(selection["ready"])
        self.assertGreater(selection["threshold"], 1.0)
        self.assertIn("no exact UNSAFE evidence", selection["reason"])
        self.assertTrue(selection["exact_verifier_authoritative"])
        self.assertFalse(selection["critic_certificate_authority"])

    def test_insufficient_statistical_evidence_or_safe_coverage_fails_closed(self):
        evidence_examples = calibration_examples(
            safe_count=2,
            unsafe_count=5,
        )
        evidence_predictions = predictions_from_lcbs(
            (0.90, 0.80) + (0.10,) * 5
        )

        insufficient_evidence = choose_lcb_threshold(
            evidence_examples,
            evidence_predictions,
            confidence=0.95,
            max_false_safe_upper_bound=0.10,
            min_safe_accepted=2,
        )

        self.assertFalse(insufficient_evidence["ready"])
        self.assertGreater(insufficient_evidence["threshold"], 1.0)
        self.assertIn(
            "no threshold satisfies", insufficient_evidence["reason"]
        )
        self.assertGreater(
            insufficient_evidence["best_possible_zero_error_upper_bound"],
            0.10,
        )

        coverage_examples = calibration_examples(
            safe_count=1,
            unsafe_count=20,
        )
        coverage_predictions = predictions_from_lcbs(
            (0.90,) + (0.10,) * 20
        )
        insufficient_coverage = choose_lcb_threshold(
            coverage_examples,
            coverage_predictions,
            confidence=0.95,
            max_false_safe_upper_bound=0.15,
            min_safe_accepted=2,
        )

        self.assertFalse(insufficient_coverage["ready"])
        self.assertGreater(insufficient_coverage["threshold"], 1.0)
        self.assertIn(
            "insufficient SAFE examples", insufficient_coverage["reason"]
        )


if __name__ == "__main__":
    unittest.main()
