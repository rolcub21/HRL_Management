import run_vcg_v11_conditioned_handling_damped_evaluation_85k as evaluation


def _metric(mae, rehandles):
    return {
        "mean_absolute_error": float(mae),
        "physical_rehandles_per_100": float(rehandles),
    }


def test_distinct_frontier_collapses_duplicate_lambda_coordinates():
    points = {
        "lambda=.1": _metric(12, 6),
        "lambda=.175": _metric(14, 3),
        "lambda=.2": _metric(14, 3),
    }
    result = evaluation.distinct_nondominated_groups(points)
    assert len(result) == 2
    duplicate = [row for row in result if row["physical_rehandles_per_100"] == 3]
    assert duplicate[0]["members"] == ["lambda=.175", "lambda=.2"]


def test_distinct_frontier_removes_dominated_coordinates():
    points = {
        "best_timing": _metric(10, 8),
        "tradeoff": _metric(12, 4),
        "duplicate_tradeoff": _metric(12, 4),
        "dominated": _metric(13, 9),
    }
    result = evaluation.distinct_nondominated_groups(points)
    members = {member for row in result for member in row["members"]}
    assert members == {"best_timing", "tradeoff", "duplicate_tradeoff"}


def test_frontier_coordinate_rounding_avoids_float_noise_duplicates():
    points = {
        "a": _metric(12.0, 3.0),
        "b": _metric(12.0 + 1e-14, 3.0 - 1e-14),
    }
    result = evaluation.distinct_nondominated_groups(points)
    assert len(result) == 1
    assert result[0]["members"] == ["a", "b"]
