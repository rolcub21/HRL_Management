import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r46_online_consequence_correction as r46,
)


def test_online_posterior_is_identity_without_support():
    prior = np.asarray([[0.2, 0.5, 0.8]], np.float32)
    actual = r46.bayesian_local_connection(
        prior, np.zeros(1, np.float32), np.zeros(1, np.float32)
    )
    np.testing.assert_allclose(actual, prior)


def test_one_identical_miss_halves_unit_prior():
    prior = np.asarray([[0.8, 0.6, 0.4]], np.float32)
    actual = r46.bayesian_local_connection(
        prior, np.ones(1, np.float32), np.zeros(1, np.float32)
    )
    np.testing.assert_allclose(actual, prior / 2.0)


def test_one_identical_success_updates_toward_one():
    prior = np.asarray([[0.8, 0.6, 0.4]], np.float32)
    actual = r46.bayesian_local_connection(
        prior, np.ones(1, np.float32), np.ones(1, np.float32)
    )
    np.testing.assert_allclose(actual, (prior + 1.0) / 2.0)


def test_protocol_keeps_episode_contract():
    protocol = r46.load_contract()
    assert protocol["gate_task_ids"] == [3, 5]
    assert protocol["episodes_per_task"] == 5
    assert protocol["early_episode_termination"] is False
    assert protocol["task_gradient_updates"] == 0


def test_single_task_audit_parser_is_isolated_and_defaults_to_task3():
    args = r46.parser().parse_args(["evaluate-single"])
    assert args.task_id == 3
    assert args.episodes == 5
    assert args.output_name == "audit_task003"
    assert r46.evaluation_directory(args.output_name).name == "audit_task003"
