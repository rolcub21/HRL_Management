from pathlib import Path
from statistics import fmean

import plot_vcg_conditioned_final_comparison_90k as plot90
import run_vcg_conditioned_final_comparison_90k as final90


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


def test_seed_paths_reproduce_three_seed_aggregate():
    rows = []
    methods = []
    for value in final90.LAMBDA_GRID:
        seed_points = []
        for seed in final90.MODEL_SEEDS:
            rehandles = 20.0 - 50.0 * value + seed
            mae = 10.0 + 10.0 * value + 0.5 * seed
            seed_points.append((rehandles, mae))
            rows.extend(
                {
                    "model_seed": seed,
                    "preference_lambda": value,
                    "strict_safe_complete": True,
                    "physical_rehandles_per_100_required_deliveries": rehandles,
                    "mean_absolute_error": mae,
                }
                for _instance in final90.INSTANCE_SEEDS
            )
        methods.append(
            {
                "method": final90.CONDITIONED_METHOD,
                "preference_lambda": value,
                "metrics": {
                    "physical_rehandles_per_100_required_deliveries": fmean(
                        point[0] for point in seed_points
                    ),
                    "mean_absolute_error": fmean(
                        point[1] for point in seed_points
                    ),
                },
            }
        )

    result = plot90._seed_operating_paths(rows, {"methods": methods})

    assert result["model_seeds"] == [0, 1, 2]
    assert result["lambdas"] == list(final90.LAMBDA_GRID)
    assert len(result["paths"][0]) == len(final90.LAMBDA_GRID)
    assert result["paths"][2][-1]["mean_absolute_error"] == 13.0
