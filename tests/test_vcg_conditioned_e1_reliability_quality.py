from pathlib import Path

import pytest

from experiments.conditioned_vcg.E01_benchmark_90k import render_reliability_quality as rq


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/vcg-conditioned-final-comparison-90k-cpu-v3"


def test_closed_e1_incomplete_baseline_summaries() -> None:
    report, rows = rq.build_report(ROOT, OUTPUT)

    assert len(rows) == 900
    assert report["original_strict_gate_report_remains_immutable"] is True
    assert report["training_runs"] == 0
    assert report["evaluation_runs"] == 0

    dynamic = report["incomplete_method_summaries"][rq.final90.DYNAMIC_METHOD]
    assert dynamic["strict_safe_complete_rows"] == 28
    assert dynamic["expected_rows"] == 30
    assert dynamic["completion_rate"] == pytest.approx(28 / 30)
    assert dynamic["conditional_metrics"]["mean_absolute_error"] == pytest.approx(21.8705357143)
    assert dynamic["conditional_metrics"]["physical_rehandles_per_100_required_deliveries"] == pytest.approx(7.5892857143)

    kim = report["incomplete_method_summaries"][rq.final90.KIM_METHOD]
    assert kim["strict_safe_complete_rows"] == 448
    assert kim["expected_rows"] == 450
    assert kim["completion_rate"] == pytest.approx(448 / 450)
    assert kim["conditional_metrics"]["mean_absolute_error"] == pytest.approx(18.5072222222)
    assert kim["conditional_metrics"]["physical_rehandles_per_100_required_deliveries"] == pytest.approx(15.5555555556)
    kim_row = next(row for row in report["method_rows"] if row["method"] == rq.final90.KIM_METHOD)
    assert kim_row["display_name"] == "A3C adaptation"
