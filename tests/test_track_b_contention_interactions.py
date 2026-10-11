import unittest

from compare_track_b_contention import (
    EGRESS_CONSTRAINED,
    ORDINARY,
    geometry_interactions,
    oriented_delta,
)


REFERENCE = "reg_selector_v5"
COMPARISON = "dynamic_pslap"


def synthetic_run(
    condition,
    source,
    schedule_id,
    *,
    instance_id=None,
    return_value=0.0,
    relocations=0,
    replans=0,
):
    """Build the complete metric surface consumed by geometry_interactions."""

    return {
        "geometry_condition": condition,
        "assignment_source": source,
        "schedule_id": schedule_id,
        "instance_id": instance_id or f"{condition}-{schedule_id}",
        "return": float(return_value),
        "success": 1.0,
        "strict_method_success": 1.0,
        "steps": 100.0,
        "mean_absolute_error": 4.0,
        "mean_tardiness": 2.0,
        "within_target_window_rate": 1.0,
        "mean_storage_flow_time": 40.0,
        "p90_storage_flow_time": 50.0,
        "p95_storage_flow_time": 55.0,
        "max_storage_flow_time": 60.0,
        "obstructive_moves": int(relocations),
        "scheduler_audit": {
            "retrieve_relocations": int(relocations),
            "retrieve_replans": int(replans),
        },
    }


class OrientedDeltaTests(unittest.TestCase):
    def test_supports_both_orientations_for_top_level_metrics(self):
        reference = synthetic_run(
            ORDINARY, REFERENCE, "schedule-a", return_value=12.0
        )
        comparison = synthetic_run(
            ORDINARY, COMPARISON, "schedule-a", return_value=7.0
        )

        self.assertEqual(
            oriented_delta(
                reference,
                comparison,
                "return",
                "reference_minus_comparison",
            ),
            5.0,
        )
        self.assertEqual(
            oriented_delta(
                reference,
                comparison,
                "return",
                "comparison_minus_reference",
            ),
            -5.0,
        )

    def test_resolves_nested_relocation_metrics_and_filters_nonfinite_values(self):
        reference = synthetic_run(
            ORDINARY, REFERENCE, "schedule-a", relocations=2
        )
        comparison = synthetic_run(
            ORDINARY, COMPARISON, "schedule-a", relocations=5
        )

        self.assertEqual(
            oriented_delta(
                reference,
                comparison,
                "retrieve_relocations",
                "comparison_minus_reference",
            ),
            3.0,
        )
        reference["return"] = float("nan")
        self.assertIsNone(
            oriented_delta(
                reference,
                comparison,
                "return",
                "reference_minus_comparison",
            )
        )

    def test_rejects_an_unknown_orientation(self):
        reference = synthetic_run(ORDINARY, REFERENCE, "schedule-a")
        comparison = synthetic_run(ORDINARY, COMPARISON, "schedule-a")

        with self.assertRaisesRegex(ValueError, "unknown orientation"):
            oriented_delta(
                reference,
                comparison,
                "return",
                "smaller_is_prettier",
            )


class GeometryInteractionTests(unittest.TestCase):
    def _matched_runs(self):
        runs = []
        values = {
            "schedule-a": {
                ORDINARY: (10.0, 8.0, 1, 3),
                EGRESS_CONSTRAINED: (15.0, 10.0, 2, 7),
            },
            "schedule-b": {
                ORDINARY: (20.0, 17.0, 0, 1),
                EGRESS_CONSTRAINED: (21.0, 17.0, 2, 4),
            },
        }
        for schedule_id, conditions in values.items():
            for condition, (
                reference_return,
                comparison_return,
                reference_relocations,
                comparison_relocations,
            ) in conditions.items():
                instance_id = f"{condition}-{schedule_id}"
                runs.extend(
                    (
                        synthetic_run(
                            condition,
                            REFERENCE,
                            schedule_id,
                            instance_id=instance_id,
                            return_value=reference_return,
                            relocations=reference_relocations,
                        ),
                        synthetic_run(
                            condition,
                            COMPARISON,
                            schedule_id,
                            instance_id=instance_id,
                            return_value=comparison_return,
                            relocations=comparison_relocations,
                        ),
                    )
                )
        return runs

    def test_uses_matched_schedule_difference_in_differences(self):
        comparisons = geometry_interactions(
            self._matched_runs(),
            (REFERENCE, COMPARISON),
            REFERENCE,
            samples=100,
        )

        self.assertEqual(len(comparisons), 1)
        comparison = comparisons[0]
        self.assertEqual(comparison["n"], 2)
        self.assertEqual(
            comparison["schedule_ids"], ["schedule-a", "schedule-b"]
        )

        returns = comparison["metrics"]["return_advantage"]
        self.assertEqual(returns["n"], 2)
        self.assertEqual(returns["mean_interaction"], 2.0)
        self.assertEqual(
            [item["ordinary_reference_advantage"] for item in returns["per_schedule"]],
            [2.0, 3.0],
        )
        self.assertEqual(
            [
                item["constrained_reference_advantage"]
                for item in returns["per_schedule"]
            ],
            [5.0, 4.0],
        )
        self.assertEqual(
            [item["interaction"] for item in returns["per_schedule"]],
            [3.0, 1.0],
        )
        self.assertEqual(
            returns["interaction_wins_ties_losses"],
            {"wins": 2, "ties": 0, "losses": 0},
        )

        relocations = comparison["metrics"][
            "retrieval_relocation_reduction"
        ]
        self.assertEqual(relocations["n"], 2)
        self.assertEqual(relocations["mean_interaction"], 2.0)
        self.assertEqual(
            [item["interaction"] for item in relocations["per_schedule"]],
            [3.0, 1.0],
        )

    def test_excludes_incomplete_or_mismatched_schedules_from_pairing(self):
        runs = self._matched_runs()
        runs = [
            run
            for run in runs
            if not (
                run["schedule_id"] == "schedule-b"
                and run["geometry_condition"] == EGRESS_CONSTRAINED
                and run["assignment_source"] == COMPARISON
            )
        ]
        runs.append(
            synthetic_run(
                EGRESS_CONSTRAINED,
                COMPARISON,
                "mismatched-schedule",
                return_value=17.0,
            )
        )

        comparison = geometry_interactions(
            runs,
            (REFERENCE, COMPARISON),
            REFERENCE,
            samples=20,
        )[0]

        self.assertEqual(comparison["n"], 1)
        self.assertEqual(comparison["schedule_ids"], ["schedule-a"])
        self.assertEqual(
            comparison["metrics"]["return_advantage"]["n"], 1
        )


if __name__ == "__main__":
    unittest.main()
