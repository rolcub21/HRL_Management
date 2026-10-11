from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_vcg_v11_nested_lambda_frontier_confirmation_89k as subject


class Confirmation89Tests(unittest.TestCase):
    def test_frozen_grid_has_exactly_450_rows(self):
        self.assertEqual(subject.MODEL_SEEDS, (0, 1, 2))
        self.assertEqual(subject.INSTANCE_SEEDS, tuple(range(89_000, 89_030)))
        self.assertEqual(subject.LAMBDA_GRID, (0.0, 0.025, 0.05, 0.1, 0.2))
        self.assertEqual(subject.EXPECTED_ROWS, 450)

    def test_prepare_does_not_construct_or_sample_environment(self):
        contract = {
            "contract_sha256": "frozen",
            "status": "prepared_panel_unopened",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "confirmation"
            with (
                patch.object(subject, "_contract", return_value=contract),
                patch.object(
                    subject,
                    "_new_environment",
                    side_effect=AssertionError("prepare opened the panel"),
                ),
            ):
                result = subject.prepare(Path(directory), output)
        self.assertEqual(result["status"], "prepared_panel_unopened")
        self.assertFalse(result["panel_opened"])

    def test_nondominated_uses_mae_and_rehandles(self):
        points = {
            0.0: {"mean_absolute_error": 10.0, "physical_rehandles_per_100": 20.0},
            0.1: {"mean_absolute_error": 11.0, "physical_rehandles_per_100": 10.0},
            0.2: {"mean_absolute_error": 12.0, "physical_rehandles_per_100": 11.0},
        }
        self.assertEqual(subject._nondominated(points), [0.0, 0.1])

    def test_summary_uses_all_supplied_rows_not_development_panel_size(self):
        rows = [
            {
                "dense_return": 1.0,
                "mean_absolute_error": 2.0,
                "steps": 3,
                "physical_rehandles": 1,
            }
            for _ in range(90)
        ]
        result = subject._summary(rows)
        self.assertEqual(result["total_physical_rehandles"], 90)
        self.assertEqual(result["physical_rehandles_per_100"], 12.5)

    def test_lambda_zero_has_no_cost_checkpoint_in_ledger_identity(self):
        contract = {"contract_sha256": "contract"}
        manifest = {"manifest_sha256": "manifest"}
        record = {
            "instance_seed": 89_000,
            "episode_instance_id": "episode",
            "schedule_id": "schedule",
        }
        row = dict(record)
        ledger = subject._with_hash(
            {
                "protocol": subject.PROTOCOL,
                "contract_sha256": "contract",
                "manifest_sha256": "manifest",
                "handling_lambda": 0.0,
                "model_seed": 0,
                **record,
                "checkpoint_sha256": subject.SELECTED_CHECKPOINT_SHA256[0],
                "cost_head_sha256": None,
                "row": row,
            },
            "ledger_sha256",
        )
        self.assertEqual(
            subject._validate_ledger(
                ledger,
                contract=contract,
                manifest=manifest,
                value=0.0,
                model_seed=0,
                record=record,
            ),
            row,
        )

    def test_monotonicity_reports_only_increases(self):
        metrics = {
            value: {"physical_rehandles_per_100": count}
            for value, count in zip(subject.LAMBDA_GRID, (20, 15, 10, 11, 5))
        }
        result = subject._monotonicity(metrics)
        self.assertEqual(result["inversion_count"], 1)
        self.assertEqual(result["total_inversion_magnitude"], 1)


if __name__ == "__main__":
    unittest.main()
