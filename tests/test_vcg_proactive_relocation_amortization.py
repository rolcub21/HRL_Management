from types import SimpleNamespace
import unittest

from PSLAP.viability_candidates import ViabilityActionType

import audit_vcg_proactive_relocation_amortization as audit


def candidate(key, action_type, label=None):
    return SimpleNamespace(
        key=key,
        action_type=action_type,
        target_label=label,
    )


def branch(
    branch_id,
    *,
    relocations,
    future_relocations,
    steps,
    dense,
    legacy,
    dense_discounted,
    legacy_discounted,
    errors,
    expected=("B1", "B2"),
    strict=True,
):
    return {
        "branch_id": branch_id,
        "total_physical_relocations": relocations,
        "future_only_physical_relocations": future_relocations,
        "total_steps": steps,
        "total_dense_return_raw": dense,
        "total_legacy_return_raw": legacy,
        "total_dense_discounted_return": dense_discounted,
        "total_legacy_discounted_return": legacy_discounted,
        "labeled_errors": errors,
        "expected_remaining_labels": expected,
        "strict_method_success": strict,
    }


class ProactiveRelocationAmortizationTests(unittest.TestCase):
    def test_highest_q_delivery_uses_stable_key_tie_break(self):
        snapshot = SimpleNamespace(
            candidates=(
                candidate("reconfigure:B1:2:2", ViabilityActionType.RECONFIGURE, "B1"),
                candidate("deliver:B2:4:3", ViabilityActionType.DELIVER, "B2"),
                candidate("deliver:B1:4:2", ViabilityActionType.DELIVER, "B1"),
                candidate("deliver:B1:4:1", ViabilityActionType.DELIVER, "B1"),
            )
        )
        prepared = SimpleNamespace(source_indices=(0, 1, 2, 3))
        q = (99.0, 7.0, 7.0, 7.0)
        best = audit._stable_best_index(
            snapshot, prepared, q, ViabilityActionType.DELIVER
        )
        self.assertEqual(
            snapshot.candidates[prepared.source_indices[best]].key,
            "deliver:B1:4:1",
        )

    def test_same_block_delivery_is_separate_secondary_estimand(self):
        snapshot = SimpleNamespace(
            candidates=(
                candidate("deliver:B2:4:1", ViabilityActionType.DELIVER, "B2"),
                candidate("deliver:B1:4:3", ViabilityActionType.DELIVER, "B1"),
                candidate("deliver:B1:4:1", ViabilityActionType.DELIVER, "B1"),
            )
        )
        prepared = SimpleNamespace(source_indices=(0, 1, 2))
        best = audit._stable_best_delivery_for_label(
            snapshot, prepared, (20.0, 3.0, 3.0), "B1"
        )
        self.assertEqual(
            snapshot.candidates[prepared.source_indices[best]].key,
            "deliver:B1:4:1",
        )
        self.assertIsNone(
            audit._stable_best_delivery_for_label(
                snapshot, prepared, (20.0, 3.0, 3.0), "B9"
            )
        )

    def test_dual_raw_and_discounted_returns_use_primitive_records(self):
        records = (
            {
                "step_reward_by_objective": {
                    "legacy_clipped_v1": 1.0,
                    "dense_piecewise_v1": 2.0,
                }
            },
            {
                "step_reward_by_objective": {
                    "legacy_clipped_v1": 3.0,
                    "dense_piecewise_v1": -4.0,
                }
            },
        )
        totals = audit._objective_totals(records, 0.5)
        self.assertEqual(totals["legacy_return_raw"], 4.0)
        self.assertEqual(totals["dense_return_raw"], -2.0)
        self.assertEqual(totals["legacy_discounted_return"], 2.5)
        self.assertEqual(totals["dense_discounted_return"], 0.0)

    def test_comparison_deltas_are_positive_when_R_is_better(self):
        R = branch(
            "R",
            relocations=2,
            future_relocations=1,
            steps=100,
            dense=50.0,
            legacy=40.0,
            dense_discounted=30.0,
            legacy_discounted=20.0,
            errors={"B1": 2.0, "B2": -4.0},
        )
        D = branch(
            "D_Q",
            relocations=4,
            future_relocations=4,
            steps=120,
            dense=10.0,
            legacy=5.0,
            dense_discounted=8.0,
            legacy_discounted=4.0,
            errors={"B1": 8.0, "B2": -10.0},
        )
        result = audit._branch_comparison(R, D)
        self.assertEqual(result["future_relocations_avoided_by_R"], 3)
        self.assertEqual(result["total_relocations_avoided_by_R"], 2)
        self.assertEqual(result["steps_avoided_by_R"], 20)
        self.assertEqual(result["dense_return_advantage_for_R"], 40.0)
        self.assertEqual(result["mean_absolute_error_improvement_for_R"], 6.0)
        self.assertTrue(result["timing_comparison_eligible"])

    def test_incomplete_branch_cannot_silently_use_survivor_timing(self):
        R = branch(
            "R",
            relocations=1,
            future_relocations=0,
            steps=10,
            dense=1.0,
            legacy=1.0,
            dense_discounted=1.0,
            legacy_discounted=1.0,
            errors={"B1": 1.0, "B2": 2.0},
        )
        failed = branch(
            "D_Q",
            relocations=0,
            future_relocations=0,
            steps=5,
            dense=-1.0,
            legacy=-1.0,
            dense_discounted=-1.0,
            legacy_discounted=-1.0,
            errors={"B1": 0.0},
            strict=False,
        )
        result = audit._branch_comparison(R, failed)
        self.assertFalse(result["timing_comparison_eligible"])
        self.assertIsNone(result["mean_absolute_error_improvement_for_R"])
        self.assertEqual(result["alternative_missing_labels"], ("B2",))

    def test_no_synthetic_wait_is_in_declared_branch_contract(self):
        self.assertIn("W", audit.BRANCH_IDS)
        self.assertNotIn("WAIT", audit.BRANCH_IDS)
        self.assertEqual(audit.EXPECTED_FULL_PRIMARY_EVENT_COUNT, 167)


if __name__ == "__main__":
    unittest.main()

