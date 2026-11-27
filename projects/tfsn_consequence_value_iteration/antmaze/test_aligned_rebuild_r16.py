"""Focused tests for the AntMaze aligned-rebuild controller runtime."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .aligned_rebuild_r16 import _OnDemandOfflineReachability
from . import r52_e549_preparation as r52


class _FakeSolver:
    target_batch = 2

    def __init__(self):
        self.calls = []
        self.weight = None

    def set_bridge_weight(self, weight):
        self.weight = np.asarray(weight).copy()

    def solve(self, targets, remaining, current_node):
        targets = np.asarray(targets, np.float32)
        self.calls.append((targets.copy(), int(remaining), int(current_node)))
        return targets + 10.0 * int(remaining) + int(current_node)


class _FakeScorer:
    def __init__(self):
        self.batched_reachability = _FakeSolver()
        self.bridge_weight = np.asarray([[0.25, 0.75]], np.float32)


class OnDemandOfflineReachabilityTest(unittest.TestCase):
    def test_matches_numpy_table_indexing_and_caches_one_column(self):
        scorer = _FakeScorer()
        view = _OnDemandOfflineReachability(
            scorer, np.asarray([2, 5, 9, 11, 14], np.int32)
        )
        expected = np.asarray([35, 38, 42, 44, 47], np.float32)

        np.testing.assert_array_equal(view[:, 3, 3], expected)
        self.assertEqual(len(scorer.batched_reachability.calls), 3)
        self.assertEqual(float(view[2, 3, 3]), 42.0)
        self.assertEqual(len(scorer.batched_reachability.calls), 3)
        np.testing.assert_array_equal(
            scorer.batched_reachability.weight, scorer.bridge_weight
        )

        self.assertEqual(float(view[0, 2, 3]), 25.0)
        self.assertEqual(len(scorer.batched_reachability.calls), 6)

    def test_rejects_non_table_indexing(self):
        view = _OnDemandOfflineReachability(
            _FakeScorer(), np.asarray([1], np.int32)
        )
        with self.assertRaises(IndexError):
            _ = view[0]


class PreparationRuntimeIdentityTest(unittest.TestCase):
    def test_verified_rebuild_identity_is_loaded_from_manifest(self):
        with TemporaryDirectory() as directory:
            reference = Path(directory) / "METHOD.md"
            reference.write_text("frozen rebuild\n")
            summary = {
                "method_reference": str(reference),
                "method_reference_sha256": r52.sha256(reference),
                "method_revision": "independent_rebuild",
                "claim_level": "preparation_only",
            }
            self.assertEqual(
                r52._verified_runtime_identity(summary),
                (reference.resolve(), "independent_rebuild", "preparation_only"),
            )

    def test_historical_r52_cannot_be_relabeled(self):
        with TemporaryDirectory() as directory:
            reference = Path(directory) / "OTHER.md"
            reference.write_text("not R52\n")
            summary = {
                "method_reference": str(reference),
                "method_reference_sha256": r52.sha256(reference),
                "method_revision": r52.METHOD_REVISION,
                "claim_level": r52.CLAIM_LEVEL,
            }
            with self.assertRaises(ValueError):
                r52._verified_runtime_identity(summary)


if __name__ == "__main__":
    unittest.main()
