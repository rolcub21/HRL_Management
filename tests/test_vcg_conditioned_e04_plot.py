from pathlib import Path

from experiments.conditioned_vcg.E04_safe_frontier_ranking_92k import (
    render_results as plot,
)
from experiments.conditioned_vcg.E04_safe_frontier_ranking_92k import run as e04


def _row(method, instance_seed, *, model_seed=None, value=None, ranking_seed=None):
    base = float(instance_seed - 92_000)
    return {
        "ranking_signal": method,
        "model_seed": model_seed,
        "preference_lambda": value,
        "ranking_seed": ranking_seed,
        "instance_seed": instance_seed,
        "strict_safe_complete": True,
        "all_selected_candidates_exact_safe": True,
        "mean_absolute_error": 10.0 + base / 10.0,
        "physical_rehandles_per_100_required_deliveries": 20.0 - base / 10.0,
        "steps": 150.0 + base,
        "dense_return": 180.0 - base,
        "within_target_window_rate": 0.8,
    }


def _synthetic_rows():
    rows = []
    for instance_seed in e04.INSTANCE_SEEDS:
        rows.append(_row(e04.HEURISTIC_SAFE, instance_seed))
        for ranking_seed in e04.RANDOM_RANKING_SEEDS:
            rows.append(
                _row(
                    e04.RANDOM_SAFE,
                    instance_seed,
                    ranking_seed=ranking_seed,
                )
            )
        for seed in e04.MODEL_SEEDS:
            rows.append(
                _row(
                    e04.QOP_SAFE,
                    instance_seed,
                    model_seed=seed,
                    value=0.0,
                )
            )
            for value in e04.CONDITIONED_LAMBDAS:
                rows.append(
                    _row(
                        e04.CONDITIONED_SAFE,
                        instance_seed,
                        model_seed=seed,
                        value=value,
                    )
                )
    return rows


def test_e04_figure_data_retains_controls_and_three_seed_trajectories():
    data = plot.build_figure_data(
        _synthetic_rows(), {"report_sha256": "synthetic"}
    )
    assert len(data["learned_aggregate"]) == 4
    assert set(data["learned_per_model_seed"]) == {"0", "1", "2"}
    assert all(
        len(points) == 4 for points in data["learned_per_model_seed"].values()
    )
    assert len(data["random_safe_per_ranking_seed"]) == 5
    assert data["heuristic_safe"]["rows"] == 30


def test_e04_figure_and_table_render_from_authenticated_shape(tmp_path: Path):
    rows = _synthetic_rows()
    data = plot.build_figure_data(rows, {"report_sha256": "synthetic"})
    figures = plot.render_figure(tmp_path, data)
    assert {Path(path).suffix for path in figures} == {".pdf", ".png", ".svg"}
    assert all(Path(path).stat().st_size > 0 for path in figures)
