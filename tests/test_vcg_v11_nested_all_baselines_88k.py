from __future__ import annotations

import unittest

import final86_v23_runtime as adapter
import run_vcg_final86_four_method as final86
import run_vcg_v11_nested_all_baselines_88k as subject


class Matched88ContractTests(unittest.TestCase):
    def test_exact_grid(self):
        self.assertEqual(subject.INSTANCE_SEEDS, tuple(range(88000, 88030)))
        self.assertEqual(subject.V23_ROWS, 360)
        self.assertEqual(subject.DYNAMIC_ROWS, 30)
        self.assertEqual(subject.GA_ROWS, 120)
        self.assertEqual(subject.KIM_ROWS, 450)
        self.assertEqual(subject.EXPECTED_NEW_ROWS, 960)

    def test_v23_context_rebinds_and_restores_all_seed_guards(self):
        old_final = final86.INSTANCE_SEEDS
        old_matched = final86.matched.PANEL_SEEDS
        old_adapter = adapter.FINAL_INSTANCE_SEEDS
        with subject._final88_context():
            self.assertEqual(final86.INSTANCE_SEEDS, subject.INSTANCE_SEEDS)
            self.assertEqual(final86.matched.PANEL_SEEDS, subject.INSTANCE_SEEDS)
            self.assertEqual(adapter.FINAL_INSTANCE_SEEDS, subject.INSTANCE_SEEDS)
        self.assertEqual(final86.INSTANCE_SEEDS, old_final)
        self.assertEqual(final86.matched.PANEL_SEEDS, old_matched)
        self.assertEqual(adapter.FINAL_INSTANCE_SEEDS, old_adapter)

    def test_config_does_not_duplicate_v11_or_train(self):
        manifest = {"manifest_sha256": "x" * 64}
        config = subject._config(manifest)
        self.assertTrue(config["vcg_1_1_not_rerun_because_lambda0_is_exactly_vcg_1_1"])
        self.assertFalse(config["training_or_learning"])
        self.assertNotIn(final86.V11_METHOD, config["new_rows"])


if __name__ == "__main__":
    unittest.main()
