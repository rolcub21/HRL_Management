from pathlib import Path

import plot_vcg_conditioned_final_comparison_90k as plot90


def _record(*, display_name, safe, expected, eligible, metrics):
    return {
        "display_name": display_name,
        "strict_safe_complete_rows": safe,
        "expected_rows": expected,
        "whole_method_eligible": eligible,
        "metrics": metrics,
    }


def _metrics(value=1.0):
    return {
        metric: value for metric, _label in plot90.TABLE_METRICS
    }


def test_e1_table_retains_failures_and_suppresses_their_metrics(tmp_path: Path):
    report = {
        "methods": [
            _record(
                display_name="VCG (lambda=0)",
                safe=90,
                expected=90,
                eligible=True,
                metrics=_metrics(),
            ),
            _record(
                display_name="Dynamic PSLAP",
                safe=28,
                expected=30,
                eligible=False,
                metrics=None,
            ),
        ]
    }

    paths = plot90.render_benchmark_table(tmp_path, report)

    assert {Path(path).suffix for path in paths} == {".csv", ".md"}
    markdown = (tmp_path / "e1-benchmark-table.md").read_text(encoding="utf-8")
    assert "90/90" in markdown
    assert "28/30" in markdown
    assert "Dynamic PSLAP | 28/30 | — | — | — | — | —" in markdown


def test_e1_plot_labels_only_selected_lambda_coordinates():
    assert plot90.SELECTED_LAMBDA_LABELS == {0.0, 0.05, 0.1, 0.2}
