from experiments.conditioned_vcg.development.D10_scalability_support_screen import run
import pytest


def test_scale_ladder_is_unique_and_factor_separated():
    assert len(run.SCALE_BY_ID) == len(run.SCALES) == 8
    reference = run.SCALE_BY_ID["reference_5x5_n8"]
    assert len(reference.make_env().storage_positions) == 8
    geometry = [scale for scale in run.SCALES if scale.axis == "geometry_only"]
    assert {scale.blocks for scale in geometry} == {8}
    workload = [scale for scale in run.SCALES if scale.axis == "workload_at_6x6"]
    assert {(scale.rows, scale.cols) for scale in workload} == {(6, 6)}
    assert all(len(scale.exits) == 3 for scale in run.SCALES)


def test_zero_shot_scale_instances_validate_and_are_deterministic():
    for scale in run.SCALES:
        env = scale.make_env()
        left = env.sample_episode_instance(run.INSTANCE_SEEDS[0])
        right = env.sample_episode_instance(run.INSTANCE_SEEDS[0])
        assert left == right
        left.validate_for(env)
        assert left.number_blocks == scale.blocks


def test_distribution_summary_has_declared_tail_statistics():
    summary = run._distribution([1.0, 2.0, 3.0, 4.0])
    assert summary["n"] == 4
    assert summary["median"] == 2.5
    assert summary["p90"] is not None
    assert summary["p95"] is not None
    assert summary["p99"] is not None
    assert summary["maximum"] == 4.0


def test_timing_summary_is_workload_size_aware():
    eight = run._timing(range(-4, 4), expected_deliveries=8)
    twelve = run._timing(range(-6, 6), expected_deliveries=12)
    assert eight["mean_absolute_error"] == 2.0
    assert twelve["mean_absolute_error"] == 3.0
    assert eight["within_target_window_rate"] == 1.0
    assert twelve["within_target_window_rate"] == 1.0


def test_timing_summary_rejects_wrong_or_nonfinite_delivery_count():
    with pytest.raises(run.D10Error, match="needs 12 finite timing deviations"):
        run._timing([0.0] * 8, expected_deliveries=12)
    with pytest.raises(run.D10Error, match="finite timing deviations"):
        run._timing([0.0] * 11 + [float("nan")], expected_deliveries=12)
