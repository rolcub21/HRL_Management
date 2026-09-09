from pathlib import Path

from experiments.conditioned_vcg.E05_handling_model_ablation_92k import (
    render_future_consequence as plot,
)
from experiments.conditioned_vcg.E05_handling_model_ablation_92k import (
    run_future_consequence as e05c,
)


def _row(seed, value, instance_seed, *, future):
    offset = float(instance_seed - e05c.INSTANCE_SEEDS[0]) / 100.0
    handling = 3.0 + seed if future else 8.0 + seed
    return {
        "model_seed": seed,
        "deployment_lambda": value,
        "instance_seed": instance_seed,
        "strict_safe_complete": True,
        "behavior_digest": f"{future}-{seed}-{value}-{instance_seed}",
        "mean_absolute_error": 12.0 + value + offset + (0.1 if future else 0.2),
        "physical_rehandles_per_100_required_deliveries": handling + offset,
        "steps": 150.0 + handling + offset,
        "dense_return": 180.0 - handling - offset,
        "within_target_window_rate": 0.8,
    }


def _pairs_and_report():
    immediate = []
    future = {}
    for seed in e05c.MODEL_SEEDS:
        for value in e05c.DEPLOYMENT_LAMBDAS:
            for instance_seed in e05c.INSTANCE_SEEDS:
                immediate.append(_row(seed, value, instance_seed, future=False))
                future[(seed, value, instance_seed)] = _row(
                    seed, value, instance_seed, future=True
                )
    pairs = plot.collect_pairs(
        immediate,
        lambda seed, value, instance_seed: future[(seed, value, instance_seed)],
    )
    report = {"report_sha256": "synthetic"}
    return pairs, report


def test_e05c_figure_data_keeps_three_paired_seed_means():
    pairs, report = _pairs_and_report()
    data = plot.build_figure_data(pairs, report)
    assert len(data["coordinates"]) == 3
    assert all(item["pairs"] == 90 for item in data["coordinates"])
    assert all(len(item["per_seed"]) == 3 for item in data["coordinates"])
    assert data["direction"] == "arrows_run_from_immediate_only_to_learned_future"


def test_e05c_figure_renders_vector_and_raster_outputs(tmp_path: Path):
    pairs, report = _pairs_and_report()
    data = plot.build_figure_data(pairs, report)
    figures = plot.render_figure(tmp_path, data)
    assert {Path(path).suffix for path in figures} == {".pdf", ".png", ".svg"}
    assert all(Path(path).stat().st_size > 0 for path in figures)
