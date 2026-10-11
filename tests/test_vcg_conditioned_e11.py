from experiments.conditioned_vcg.E11_distribution_shift_93k import run as e11


def test_e11_regimes_preserve_nominal_problem_size_and_declared_capacity():
    assert len(e11.REGIMES) == 7
    for regime in e11.REGIMES:
        env = regime.make_env()
        assert (env.grid_rows, env.grid_cols, env.number_blocks) == (5, 5, 8)
        assert len(env.storage_positions) == 8
        assert len(env.exit_cells) == 3


def test_e11_one_factor_instances_hold_nonshifted_schedule_components_fixed():
    seed = e11.INSTANCE_SEEDS[0]
    reference = e11._instance(e11.REGIME_BY_ID["reference"], seed)
    spread = e11._instance(e11.REGIME_BY_ID["arrival_spread"], seed)
    short = e11._instance(e11.REGIME_BY_ID["dwell_short"], seed)
    mirrored = e11._instance(e11.REGIME_BY_ID["mirrored_entry"], seed)
    assert spread.storage_steps_needed == reference.storage_steps_needed
    assert short.arrival_steps == reference.arrival_steps
    assert mirrored.arrival_steps == reference.arrival_steps
    assert mirrored.storage_steps_needed == reference.storage_steps_needed
    assert mirrored.door_cell == (0, 1)
    assert mirrored.start_state == (1, 3)
    assert reference.door_cell == (0, 3)
    assert reference.start_state == (1, 1)


def test_e11_predeclares_thirteen_policy_rows_per_instance_and_suppresses_failures():
    assert len(e11._specs(None)) == 13
    complete = {
        "strict_safe_complete": True,
        "deadlock_or_empty_frontier": False,
        "episode_step_limit": False,
        "method_failure_reason": None,
        **{name: 1.0 for name in e11.METRICS},
        "nominal_peak_concurrency": 8,
        "actual_peak_active_jobs": 8,
        "actual_peak_storage_occupancy": 7,
        "actual_peak_arrived_unstored": 4,
        "jobs_ever_admitted": 8,
        "jobs_unadmitted_at_stop": 0,
        "jobs_delivered": 8,
    }
    failed = dict(complete)
    failed.update(
        {
            "strict_safe_complete": False,
            "method_failure_reason": "episode_step_limit",
            "episode_step_limit": True,
            **{name: None for name in e11.METRICS},
        }
    )
    result = e11._aggregate([complete, failed], expected=2)
    assert result["strict_safe_complete"] == 1
    assert result["strict_completion_rate"] == 0.5
    assert result["available_row_completion_rate"] == 0.5
    assert result["complete_case_metrics_suppressed"] is True
    assert result["mean_absolute_error"] is None
