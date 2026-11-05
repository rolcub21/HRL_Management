from collections import Counter

import pytest

from experiments.conditioned_vcg.E243_e195_expanded_prefilled_curve_302k_v1 import (
    protocol,
    run,
    scenario,
)


def test_e243_expands_each_fill_to_25_witnessed_days():
    assert scenario.INITIAL_FILLS == (4, 5, 6, 7)
    assert scenario.CASES_PER_FILL == 25
    assert len(scenario.SEEDS) == 100
    assert protocol.EXPECTED_ROWS == 400
    assert Counter(scenario.initial_fill(seed) for seed in scenario.SEEDS) == {
        4: 25,
        5: 25,
        6: 25,
        7: 25,
    }


def test_e243_pairs_duration_draws_across_fill_levels():
    for replicate in range(scenario.CASES_PER_FILL):
        durations = []
        for fill_index in range(len(scenario.INITIAL_FILLS)):
            seed = scenario.SEED_BASE + fill_index * 100 + replicate
            env = scenario.Prefilled5x5Env(scenario.SCENARIO, run.e195.SPEC)
            durations.append(env.sample_episode_instance(seed).storage_steps_needed)
        assert len(set(durations)) == 1


def test_e243_checkpoint_bundle_is_authenticated():
    if not protocol.SOURCE_CHECKPOINT.is_file():
        pytest.skip("E243 frozen source bundle is distributed as a release asset")
    identity = protocol.checkpoint_identity()
    assert identity["model_seed"] == 0
    assert identity["selected_episode"] == 300
    assert identity["source_checkpoint_sha256"] == (
        "03acf192e1162b8cc879df2831790e8d43f163aa8f3390ce3cb92bec5f1dfec4"
    )


def test_e243_every_initial_state_has_a_completion_witness():
    if not protocol.SOURCE_CHECKPOINT.is_file():
        pytest.skip("E243 frozen source bundle is distributed as a release asset")
    rows = run._validate_initial_states()
    assert len(rows) == len(scenario.SEEDS)
    assert all(row["status"] == "SAFE" for row in rows)
    assert all(row["witness_length"] > 0 for row in rows)


def test_e243_dynamic_pair_has_one_authenticated_ranker():
    assert protocol.DYNAMIC_ARMS == (
        "dynamic_certified",
        "dynamic_physical_unfiltered",
    )
    value = protocol.dynamic_ranker_sha256()
    assert len(value) == 64
    assert all(character in "0123456789abcdef" for character in value)
