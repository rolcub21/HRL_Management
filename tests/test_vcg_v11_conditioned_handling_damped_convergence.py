import pytest
import torch
from torch import nn

import train_vcg_v11_conditioned_handling_damped_convergence as damped


def test_polyak_blend_uses_fixed_quarter_step():
    network = nn.Linear(3, 2)
    with torch.no_grad():
        network.weight.zero_()
        network.bias.fill_(2.0)
    previous = damped._clone_state(network)
    with torch.no_grad():
        network.weight.fill_(4.0)
        network.bias.fill_(6.0)

    damped.polyak_blend_network(network, previous)

    torch.testing.assert_close(network.weight, torch.full_like(network.weight, 1.0))
    torch.testing.assert_close(network.bias, torch.full_like(network.bias, 3.0))
    assert all(not parameter.requires_grad for parameter in network.parameters())


def test_polyak_blend_rejects_misaligned_state():
    network = nn.Linear(2, 1)
    with pytest.raises(ValueError, match="do not align"):
        damped.polyak_blend_network(network, {"wrong": torch.zeros(1)})


def test_damped_schedule_is_fixed_stratified_and_pairs_first_two_rounds():
    for index in range(damped.DAMPED_ROUNDS):
        schedule = damped.damped_lambda_schedule(index)
        assert len(schedule) == damped.EPISODES_PER_ROUND
        values = [row["behavior_lambda"] for row in schedule]
        assert all(0.0 < value < 0.2 for value in values)
        assert sorted(
            int(value / 0.2 * damped.EPISODES_PER_ROUND) for value in values
        ) == list(range(damped.EPISODES_PER_ROUND))
        assert schedule[0]["global_round_number"] == 5 + index
    damped_first = damped.damped_lambda_schedule(0)
    full_update_first = damped.extension.extension_lambda_schedule(0)
    paired_fields = (
        "global_episode_number",
        "global_round_number",
        "position_in_round",
        "behavior_lambda",
        "schedule_rng_seed",
    )
    assert [
        tuple(row[field] for field in paired_fields) for row in damped_first
    ] == [
        tuple(row[field] for field in paired_fields) for row in full_update_first
    ]


def test_schedule_rejects_round_outside_fixed_experiment():
    with pytest.raises(ValueError, match="out of range"):
        damped.damped_lambda_schedule(damped.DAMPED_ROUNDS)
