from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from plot_vcg_dense_pareto import (
    FINAL_GROUP,
    SELECTED_GROUP,
    build_plot_spec,
    load_report,
    render_plot,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "results"
    / "vcg-dense-v1-1-pareto-development-30seed"
    / "pareto-report.json"
)


class VcgDenseParetoPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = build_plot_spec(load_report(REPORT))

    def test_real_report_preserves_gate_roles_and_all_seed_shifts(self):
        spec = self.spec
        numeric_ids = {point.point_id for point in spec.aggregate_points}

        self.assertEqual(spec.instance_count, 30)
        self.assertEqual(spec.mae_margin, 2.0)
        self.assertEqual(
            numeric_ids,
            {
                SELECTED_GROUP,
                FINAL_GROUP,
                "duration_aware_enhanced_complete_rolling_ga",
                "duration_aware_nearest_free",
            },
        )
        self.assertTrue(
            set(spec.excluded_baselines).isdisjoint(numeric_ids)
        )
        self.assertEqual(
            set(spec.excluded_baselines),
            {
                "duration_aware_dynamic_pslap",
                "duration_aware_pslap_ga_2009_rolling",
            },
        )
        self.assertFalse(spec.selected_aggregate.diagnostic_only)
        self.assertTrue(spec.final_aggregate.diagnostic_only)

        self.assertEqual(
            tuple(shift.model_seed for shift in spec.seed_shifts),
            (0, 1, 2),
        )
        self.assertEqual(
            tuple(shift.zero_length for shift in spec.seed_shifts),
            (True, False, False),
        )
        for shift in spec.seed_shifts:
            self.assertEqual(shift.selected.policy_group, SELECTED_GROUP)
            self.assertEqual(shift.final.policy_group, FINAL_GROUP)
            self.assertTrue(shift.final.diagnostic_only)

        self.assertAlmostEqual(
            spec.final_aggregate.mae - spec.selected_aggregate.mae,
            0.6430555555555575,
        )
        self.assertAlmostEqual(
            spec.selected_aggregate.relocations_per_100
            - spec.final_aggregate.relocations_per_100,
            4.861111111111111,
        )

    def test_render_writes_nonempty_png_and_pdf(self):
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "pareto"
            png, pdf = render_plot(self.spec, prefix, dpi=72)

            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(pdf.read_bytes()[:4], b"%PDF")
            self.assertGreater(png.stat().st_size, 20_000)
            self.assertGreater(pdf.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
