import math

from experiments.conditioned_vcg.E19_service_tails_cost_sensitivity import (
    analyze as e19,
)


def _row(*, strict=True, deviations=(-2.0, 0.0, 4.0), rehandles=3):
    return {
        "method_key": "method",
        "display_name": "Method",
        "strict_safe_complete": strict,
        "instance_seed": 1,
        "model_seed": 0,
        "rng_index": 0,
        "required_deliveries": 3,
        "delivery_deviations": list(deviations),
        "delivery_count": len(deviations),
        "mean_earliness": 2.0 / 3.0,
        "mean_tardiness": 4.0 / 3.0,
        "mean_absolute_error": 2.0,
        "within_target_window_rate": 1.0,
        "physical_rehandles": rehandles,
        "steps": 30,
        "failure_reason": None if strict else "failed",
    }


def test_type7_quantile_interpolates():
    assert e19._quantile([0.0, 10.0], 0.95) == 9.5


def test_service_summary_separates_signed_tails_and_steps():
    summary = e19._service_summary([_row()])["method"]
    assert summary["service_tail_eligible"] is True
    assert math.isclose(summary["service"]["early_job_fraction"], 1.0 / 3.0)
    assert math.isclose(summary["service"]["late_job_fraction"], 1.0 / 3.0)
    assert summary["service"]["absolute_error"]["maximum"] == 4.0
    assert summary["service"]["worst_signed_deviation"] == {
        "earliest": -2.0,
        "latest": 4.0,
    }
    assert summary["episode_metrics"]["steps_per_required_delivery"] == 10.0


def test_method_level_failure_suppresses_success_aggregates():
    failed = _row(strict=False, deviations=())
    summary = e19._service_summary([_row(), failed])["method"]
    assert summary["strict_safe_complete"] == 1
    assert summary["recorded_unmet_jobs"] == 3
    assert summary["service"] is None
    assert summary["episode_metrics"] is None


def test_cost_uses_early_late_and_rehandles_without_steps():
    row = _row()
    observed = e19._cost_loss(row, tardiness_weight=2.0, rehandle_equivalent=10.0)
    expected = 2.0 / 3.0 + 2.0 * 4.0 / 3.0 + 10.0
    assert math.isclose(observed, expected)
    row["steps"] = 3000
    assert math.isclose(
        e19._cost_loss(row, tardiness_weight=2.0, rehandle_equivalent=10.0),
        expected,
    )
