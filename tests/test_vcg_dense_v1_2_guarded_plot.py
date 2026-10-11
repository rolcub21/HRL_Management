from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from plot_vcg_dense_pareto import load_report as load_source_report
from plot_vcg_dense_v1_2_guarded import (
    build_guarded_plot_spec,
    load_guarded_report,
    render_guarded_plot,
)


ROOT = Path(__file__).resolve().parents[1]
GUARDED_REPORT = (
    ROOT
    / "results"
    / "vcg-dense-v1-2-posthoc-development-30seed"
    / "guarded-selection-report.json"
)
SOURCE_REPORT = (
    ROOT
    / "results"
    / "vcg-dense-v1-1-pareto-development-30seed"
    / "pareto-report.json"
)
EXPECTED_REPORT_SHA256 = (
    "48f4a145770344a6f4dad44bd92a24117511d92287a24a812e69a7e53e5ed5cc"
)


class VcgDenseV12GuardedPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guarded_report = load_guarded_report(GUARDED_REPORT)
        cls.source_report = load_source_report(SOURCE_REPORT)
        cls.spec = build_guarded_plot_spec(
            cls.guarded_report,
            cls.source_report,
            source_report_path=SOURCE_REPORT,
        )

    def test_frozen_report_hash_and_guarded_coordinates(self):
        self.assertEqual(
            hashlib.sha256(GUARDED_REPORT.read_bytes()).hexdigest(),
            EXPECTED_REPORT_SHA256,
        )
        spec = self.spec
        self.assertEqual(spec.instance_count, 30)
        self.assertTrue(spec.guarded.diagnostic_only)
        self.assertTrue(spec.episode500.diagnostic_only)
        self.assertFalse(spec.selected.diagnostic_only)
        self.assertTrue(spec.positive_relocation_gate)
        self.assertTrue(spec.mandatory_reference_fallback)
        self.assertEqual(
            tuple(shift.model_seed for shift in spec.seed_shifts),
            (0, 1, 2),
        )
        self.assertEqual(
            tuple(shift.chosen_episode for shift in spec.seed_shifts),
            (475, 500, 500),
        )
        self.assertTrue(
            all(shift.guard_mae_cost <= 2.0 for shift in spec.seed_shifts)
        )
        self.assertTrue(
            all(shift.guard_relocation_saving > 0.0 for shift in spec.seed_shifts)
        )
        self.assertAlmostEqual(spec.guarded.mae, 12.418055555555556)
        self.assertAlmostEqual(
            spec.guarded.relocations_per_100,
            18.194444444444443,
        )
        self.assertAlmostEqual(
            spec.guarded_mae_delta_vs_selected,
            1.0152777777777777,
        )
        self.assertAlmostEqual(
            spec.guarded_relocation_delta_vs_selected,
            -5.6944444444444455,
        )
        self.assertAlmostEqual(
            spec.guarded_mae_delta_vs_enhanced,
            -2.340277777777778,
        )
        self.assertAlmostEqual(
            spec.guarded_relocation_delta_vs_enhanced,
            0.6944444444444443,
        )

    def test_plot_fails_closed_if_new_guardrails_are_removed(self):
        for field in (
            "positive_panel_B_relocation_saving_required",
            "reference_is_mandatory_last_fallback",
        ):
            with self.subTest(field=field):
                altered = deepcopy(self.guarded_report)
                altered["guardrails"][field] = False
                with self.assertRaisesRegex(ValueError, field):
                    build_guarded_plot_spec(altered, self.source_report)

    def test_plot_recomputes_positive_relocation_gate(self):
        altered = deepcopy(self.guarded_report)
        accepted = altered["per_training_seed"][0]["sequential_guard"][
            "sequential_panel_B_trials"
        ][0]
        accepted["panel_B_relocation_saving_per_100_deliveries"] = 0.0
        with self.assertRaisesRegex(ValueError, "accepted flag"):
            build_guarded_plot_spec(altered, self.source_report)

    def test_render_writes_separate_png_and_pdf(self):
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "guarded-pareto"
            png, pdf = render_guarded_plot(self.spec, prefix, dpi=72)

            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(pdf.read_bytes()[:4], b"%PDF")
            self.assertGreater(png.stat().st_size, 20_000)
            self.assertGreater(pdf.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
