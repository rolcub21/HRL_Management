from __future__ import annotations

import unittest
from unittest.mock import patch

import final86_v23_runtime as adapter
import run_vcg_final86_four_method as final86
import run_vcg_v11_nested_all_baselines_89k as subject


class Matched89ContractTests(unittest.TestCase):
    def test_exact_grid(self):
        self.assertEqual(subject.INSTANCE_SEEDS, tuple(range(89000, 89030)))
        self.assertEqual(subject.V23_ROWS, 360)
        self.assertEqual(subject.DYNAMIC_ROWS, 30)
        self.assertEqual(subject.GA_ROWS, 120)
        self.assertEqual(subject.KIM_ROWS, 450)
        self.assertEqual(subject.EXPECTED_NEW_ROWS, 960)

    def test_v23_context_rebinds_and_restores_all_seed_guards(self):
        old_final = final86.INSTANCE_SEEDS
        old_matched = final86.matched.PANEL_SEEDS
        old_adapter = adapter.FINAL_INSTANCE_SEEDS
        with subject._final89_context():
            self.assertEqual(final86.INSTANCE_SEEDS, subject.INSTANCE_SEEDS)
            self.assertEqual(final86.matched.PANEL_SEEDS, subject.INSTANCE_SEEDS)
            self.assertEqual(adapter.FINAL_INSTANCE_SEEDS, subject.INSTANCE_SEEDS)
        self.assertEqual(final86.INSTANCE_SEEDS, old_final)
        self.assertEqual(final86.matched.PANEL_SEEDS, old_matched)
        self.assertEqual(adapter.FINAL_INSTANCE_SEEDS, old_adapter)

    def test_config_reuses_only_requested_nested_endpoints_and_never_trains(self):
        config = subject._config({"manifest_sha256": "x" * 64})
        self.assertEqual(config["reused_lambda_values"], [0.0, 0.2])
        self.assertEqual(config["existing_nested_vcg_rows_reused"], 180)
        self.assertFalse(config["training_or_learning"])

    def test_prepare_configures_runtime_before_panel_authentication(self):
        calls = []

        def configured():
            calls.append("runtime")

        def panel():
            calls.append("panel")
            raise RuntimeError("stop")

        with (
            patch.object(subject.final86, "_configure_runtime", side_effect=configured),
            patch.object(subject, "_panel_inputs", side_effect=panel),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                subject.prepare()
        self.assertEqual(calls, ["runtime", "panel"])


if __name__ == "__main__":
    unittest.main()
