from pathlib import Path

from experiments.conditioned_vcg.E03_certification_ablation_90k import (
    render_recovery_basin as basin,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/viability-kernel-5x5-all-goal.json"


def test_frozen_compact_recovery_basin_summary() -> None:
    payload = basin.load_source(SOURCE)
    report = basin.build_summary(payload, SOURCE)

    assert report["state_count"] == 1280
    assert report["status_counts"] == {"SAFE": 1261, "UNSAFE": 19}
    assert report["unknown_count"] == 0
    by_occupancy = {row["occupancy"]: row for row in report["occupancy_rows"]}
    assert by_occupancy[4]["safe_fraction"] == 1.0
    assert by_occupancy[5]["safe_count"] == 335
    assert by_occupancy[7]["safe_count"] == 56
    assert by_occupancy[8]["safe_count"] == 4
    assert by_occupancy[8]["state_count"] == 9

    distance_rows = {
        (row["occupancy"], row["static_exit_distance"]): row
        for row in report["exit_distance_rows"]
    }
    assert distance_rows[(8, 0)]["safe_fraction"] == 1.0
    assert distance_rows[(8, 2)]["safe_fraction"] == 1.0
    assert distance_rows[(8, 3)]["safe_fraction"] == 0.0
    assert distance_rows[(8, 5)]["safe_fraction"] == 0.0
    effort = {row["occupancy"]: row for row in report["witness_step_summaries"]}
    assert effort[0]["median"] == 0.0
    assert effort[8]["median"] == 64.0

    full = report["full_occupancy_boundary"]
    safe_positions = {
        tuple(row["agent_position"]) for row in full if row["status"] == "SAFE"
    }
    assert safe_positions == {(2, 3), (3, 2), (3, 3), (4, 3)}
    assert report["training_runs"] == 0
    assert report["evaluation_runs"] == 0


def test_recovery_basin_renderer_writes_paper_artifacts(tmp_path: Path) -> None:
    result = basin.render(SOURCE, tmp_path)

    assert result["status"] == "complete"
    for key in (
        "report",
        "figure_pdf",
        "figure_png",
        "preferred_aggregate_figure_pdf",
        "preferred_aggregate_figure_png",
        "caption",
    ):
        path = Path(result[key])
        assert path.is_file()
        assert path.stat().st_size > 100
