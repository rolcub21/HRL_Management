import pytest

import run_vcg_v11_conditioned_handling_two_phase_seeds12 as two_phase


def test_recipe_has_four_full_then_four_damped_rounds():
    assert [two_phase.recipe_phase(index) for index in range(two_phase.TOTAL_ROUNDS)] == [
        "full_update",
        "full_update",
        "full_update",
        "full_update",
        "rho_0p25_damped",
        "rho_0p25_damped",
        "rho_0p25_damped",
        "rho_0p25_damped",
    ]


def test_recipe_schedules_exactly_match_finalized_seed0_streams():
    for index in range(two_phase.FULL_ROUNDS):
        assert two_phase.recipe_schedule(index) == tuple(
            two_phase.full.stratified_lambda_schedule(index)
        )
    for index in range(two_phase.DAMPED_ROUNDS):
        assert two_phase.recipe_schedule(index + two_phase.FULL_ROUNDS) == tuple(
            two_phase.damped.damped_lambda_schedule(index)
        )


def test_training_instance_stream_is_paired_across_model_seeds():
    expected = [
        two_phase.full.TRAIN_SEED_BASE + round_index * two_phase.full.ROUND_SEED_STRIDE + position
        for round_index in range(two_phase.TOTAL_ROUNDS)
        for position in range(two_phase.EPISODES_PER_ROUND)
    ]
    observed_seed1 = [
        two_phase.recipe_instance_seed(round_index, position)
        for round_index in range(two_phase.TOTAL_ROUNDS)
        for position in range(two_phase.EPISODES_PER_ROUND)
    ]
    observed_seed2 = list(observed_seed1)
    assert observed_seed1 == expected
    assert observed_seed2 == expected
    assert len(set(expected)) == two_phase.COLLECTION_EPISODES_PER_SEED


def test_recipe_rejects_invalid_coordinates():
    with pytest.raises(ValueError, match="round"):
        two_phase.recipe_schedule(two_phase.TOTAL_ROUNDS)
    with pytest.raises(ValueError, match="position"):
        two_phase.recipe_instance_seed(0, two_phase.EPISODES_PER_ROUND)
