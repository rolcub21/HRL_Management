import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r47_current_behavior_alignment as r47,
)


def test_protocol_is_reward_free_and_current_behavior():
    value = r47.load_protocol()
    assert value["base_method_revision"] == r47.r46.METHOD_REVISION
    assert value["hidden_goals_accessed"] is False
    assert value["task_reward_used"] is False
    assert value["environment_reward_consumed"] is False
    assert value["oracle_inputs"] == []


def test_continuation_target_labels_next_command_not_current_command():
    count = 12
    states = np.zeros((count, 101, 29), np.float32)
    states[1, -1, 0] = r47.r31.SIGMA
    command_index = np.arange(count) % 10
    target, valid = r47._continuation_targets(states, command_index)
    assert valid.tolist() == [1] * 9 + [0, 1, 0]
    assert np.isclose(target[0], 1.0 - np.exp(-0.5))
    assert target[9] == 0.0
