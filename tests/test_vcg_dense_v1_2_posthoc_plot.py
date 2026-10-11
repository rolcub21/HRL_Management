from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from plot_vcg_dense_pareto import load_report as load_source_report
from plot_vcg_dense_v1_2_posthoc import (
    build_posthoc_plot_spec,
    load_posthoc_report,
    render_posthoc_plot,
)


ROOT = Path(__file__).resolve().parents[1]
POSTHOC_REPORT = (
    ROOT
    / "results"
    / "vcg-dense-v1-2-posthoc-development-30seed"
    / "posthoc-report.json"
)
SOURCE_REPORT = (
    ROOT
    / "results"
    / "vcg-dense-v1-1-pareto-development-30seed"
    / "pareto-report.json"
)


class VcgDenseV12PosthocPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = build_posthoc_plot_spec(
            load_posthoc_report(POSTHOC_REPORT),
            load_source_report(SOURCE_REPORT),
            source_report_path=SOURCE_REPORT,
        )

    def test_real_reports_preserve_posthoc_status_and_point_relation(self):
        spec = self.spec

        self.assertEqual(spec.instance_count, 30)
        self.assertTrue(spec.candidate.diagnostic_only)
        self.assertTrue(spec.episode500.diagnostic_only)
        self.assertFalse(spec.selected.diagnostic_only)
        self.assertEqual(
            tuple(shift.model_seed for shift in spec.seed_shifts),
            (0, 1, 2),
        )
        self.assertEqual(
            tuple(shift.chosen_episode for shift in spec.seed_shifts),
            (475, 425, 500),
        )

        self.assertAlmostEqual(spec.candidate.mae, 12.9625)
        self.assertAlmostEqual(spec.candidate.relocations_per_100, 14.861111111111112)
        self.assertLess(spec.candidate.mae, spec.enhanced_ga.mae)
        self.assertLess(
            spec.candidate.relocations_per_100,
            spec.enhanced_ga.relocations_per_100,
        )
        self.assertAlmostEqual(
            spec.candidate_mae_cost_vs_enhanced,
            -1.7958333333333332,
        )
        self.assertAlmostEqual(
            spec.candidate_relocation_saving_vs_enhanced,
            2.638888888888889,
        )
        self.assertTrue(spec.mae_intervals_cross_zero)
        self.assertTrue(spec.relocation_intervals_cross_zero)

    def test_render_writes_separate_png_and_pdf(self):
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "posthoc-pareto"
            png, pdf = render_posthoc_plot(self.spec, prefix, dpi=72)

            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(pdf.read_bytes()[:4], b"%PDF")
            self.assertGreater(png.stat().st_size, 20_000)
            self.assertGreater(pdf.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
