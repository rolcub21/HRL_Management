import pytest

import run_vcg_v11_nested_handling_seed_stability as stability


def _summary(return_, mae, steps, total):
    return {
        "whole_method_eligible": True,
        "metrics": {
            "mean_dense_return": float(return_),
            "mean_absolute_error": float(mae),
            "mean_steps": float(steps),
            "total_physical_rehandles": int(total),
            "physical_rehandles_per_100": 100.0 * int(total) / 96.0,
        },
    }


def test_protocol_freezes_only_selected_lambda_and_two_new_seed_grids():
    assert stability.SELECTED_LAMBDA == 0.025
    assert stability.MODEL_SEEDS == (0, 1, 2)
    assert stability.NEW_MODEL_SEEDS == (1, 2)
    assert stability.SELECTED_EPISODES == {0: 500, 1: 450, 2: 375}
    assert len(stability.INSTANCE_SEEDS) == 12
    assert len(stability.NEW_MODEL_SEEDS) * len(stability.INSTANCE_SEEDS) == 24


def test_individual_gate_requires_two_saved_rehandles_and_operational_bounds():
    base = _summary(170, 13, 180, 20)["metrics"]
    assert stability._metrics_gate(
        base, _summary(155, 15, 175, 18)["metrics"]
    )["passed"]
    assert not stability._metrics_gate(
        base, _summary(155, 15, 175, 19)["metrics"]
    )["passed"]
    assert not stability._metrics_gate(
        base, _summary(149.9, 13, 175, 18)["metrics"]
    )["passed"]
    assert not stability._metrics_gate(
        base, _summary(155, 15.01, 175, 18)["metrics"]
    )["passed"]


def test_equal_seed_aggregate_weights_model_seeds_equally():
    summaries = {
        0: _summary(100, 10, 150, 3),
        1: _summary(200, 20, 180, 6),
        2: _summary(300, 30, 210, 9),
    }
    result = stability._aggregate(summaries)
    assert result["mean_dense_return"] == 200.0
    assert result["mean_absolute_error"] == 20.0
    assert result["mean_steps"] == 180.0
    assert result["total_physical_rehandles"] == 18
    assert result["physical_rehandles_per_100"] == pytest.approx(6.25)


def test_aggregate_suppresses_if_any_seed_is_ineligible():
    summaries = {
        0: _summary(100, 10, 150, 3),
        1: _summary(200, 20, 180, 6),
        2: {"whole_method_eligible": False, "metrics": None},
    }
    assert stability._aggregate(summaries) is None
