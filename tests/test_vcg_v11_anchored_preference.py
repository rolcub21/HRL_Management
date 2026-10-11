from types import SimpleNamespace
import random

import pytest
import torch

from vcg_v11_anchored_preference import (
    AnchoredPreferenceConfig,
    AnchoredReplayBuffer,
    AnchoredResidualNetwork,
    CachedAnchoredTransition,
)
from vcg_v11_nested_handling import DetachedHandlingCostNetwork


def _config(**overrides):
    values = {
        "feature_dim": 14,
        "hidden_dim": 8,
        "lambda_max": 0.2,
        "batch_size": 2,
        "replay_capacity": 10,
    }
    values.update(overrides)
    return AnchoredPreferenceConfig(**values)


def test_operational_residual_is_exactly_zero_at_initialization():
    network = AnchoredResidualNetwork(_config(), seed=7)
    features = torch.randn(5, 14)
    immediate = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
    for value in (0.0, 0.025, 0.2):
        output = network(features, value, immediate)
        assert torch.equal(output[:, 0], torch.zeros(5))


def test_handling_warm_start_is_lambda_invariant_and_exact():
    config = _config()
    source_config = SimpleNamespace(
        graph_embedding_dim=4,
        action_embedding_dim=2,
        head_hidden_dim=8,
    )
    source = DetachedHandlingCostNetwork(source_config, seed=13)
    target = AnchoredResidualNetwork(config, seed=99)
    target.initialize_handling_from(source)
    features = torch.randn(9, 14)
    immediate = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    expected = immediate + source(features)
    for value in (0.0, 0.025, 0.1, 0.2):
        observed = target(features, value, immediate)[:, 1]
        assert torch.equal(observed, expected)


def test_executor_gamma_alias_is_operational_gamma():
    config = _config(gamma_op=0.97)
    assert config.gamma == 0.97


def _transition(mode_id: int, *, done: bool = False):
    count = 0 if done else 2
    return CachedAnchoredTransition(
        chosen_feature=torch.arange(14, dtype=torch.float32),
        chosen_base_q=1.5,
        chosen_immediate=float(mode_id == 1),
        chosen_mode_id=mode_id,
        operational_return=2.0,
        raw_rehandles=1,
        duration=3,
        next_features=torch.ones((count, 14)),
        next_base_q=torch.ones(count),
        next_immediate=torch.zeros(count),
        next_mode_ids=tuple(0 for _ in range(count)),
        next_keys=tuple(f"k{i}" for i in range(count)),
        done=done,
        behavior_lambda=0.1,
    )


def test_replay_is_numeric_and_mode_balanced():
    replay = AnchoredReplayBuffer(10)
    for mode in (0, 0, 0, 1, 1, 2):
        replay.add(_transition(mode))
    batch = replay.sample_mode_balanced(6, random.Random(3))
    assert {item.chosen_mode_id for item in batch} == {0, 1, 2}
    assert all(item.chosen_feature.device.type == "cpu" for item in batch)


def test_terminal_transition_cannot_retain_a_successor_frontier():
    with pytest.raises(ValueError, match="terminal transition"):
        CachedAnchoredTransition(
            chosen_feature=torch.zeros(14),
            chosen_base_q=0.0,
            chosen_immediate=0.0,
            chosen_mode_id=0,
            operational_return=0.0,
            raw_rehandles=0,
            duration=1,
            next_features=torch.zeros((1, 14)),
            next_base_q=torch.zeros(1),
            next_immediate=torch.zeros(1),
            next_mode_ids=(0,),
            next_keys=("x",),
            done=True,
            behavior_lambda=0.0,
        )
