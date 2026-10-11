from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vcg_v11_nested_handling import (
    CachedCostDataset,
    DetachedHandlingCostNetwork,
    HandlingAugmentedV11Agent,
    cost_checkpoint,
    fit_cached_cost_network,
    load_cost_checkpoint,
)


def _config():
    return SimpleNamespace(
        graph_embedding_dim=2,
        action_embedding_dim=1,
        head_hidden_dim=4,
        timing_scale=10.0,
    )


class _Base:
    def __init__(self):
        self.config = _config()
        self.device = torch.device("cpu")
        self.Q_local = nn.Sequential(nn.Linear(2, 3), nn.SiLU())
        self.Q_target = nn.Sequential(nn.Linear(2, 3), nn.SiLU())
        self.calls = []

    def select(self, snapshot, *, training=False, epsilon=None):
        self.calls.append((snapshot, training, epsilon))
        return "original-v1-decision"

    def reset_episode_state(self):
        return None

    def observe_outcome(self, *args, **kwargs):
        return None


def test_lambda_zero_is_direct_delegation_and_never_scores_cost():
    base = _Base()
    cost = DetachedHandlingCostNetwork(base.config, seed=3)
    wrapper = HandlingAugmentedV11Agent(base, cost, handling_lambda=0.0)

    def forbidden(_features):
        raise AssertionError("cost head was called")

    wrapper.cost_network.forward = forbidden
    assert wrapper.select("snapshot", training=False, epsilon=0.0) == (
        "original-v1-decision"
    )
    assert base.calls == [("snapshot", False, 0.0)]
    assert not any(parameter.requires_grad for parameter in base.Q_local.parameters())
    assert not any(parameter.requires_grad for parameter in base.Q_target.parameters())


def test_augmented_wrapper_cannot_update_operational_controller():
    wrapper = HandlingAugmentedV11Agent(
        _Base(), DetachedHandlingCostNetwork(_config(), seed=4), handling_lambda=0.0
    )
    with pytest.raises(RuntimeError, match="evaluation-only"):
        wrapper.remember(None)
    with pytest.raises(RuntimeError, match="cannot update Qop"):
        wrapper.learn()
    with pytest.raises(ValueError, match="evaluation-only"):
        wrapper.select("snapshot", training=True, epsilon=0.0)


def test_cached_fit_changes_only_cost_parameters_and_selects_validation_state():
    config = _config()
    network = DetachedHandlingCostNetwork(config, seed=5)
    width = 3 * config.graph_embedding_dim + config.action_embedding_dim
    generator = torch.Generator().manual_seed(7)
    train_features = torch.randn(32, width, generator=generator)
    val_features = torch.randn(12, width, generator=generator)
    training = CachedCostDataset(
        train_features,
        torch.zeros(32),
        torch.relu(train_features[:, 0]) + 0.5,
    )
    validation = CachedCostDataset(
        val_features,
        torch.zeros(12),
        torch.relu(val_features[:, 0]) + 0.5,
    )
    before = {name: value.clone() for name, value in network.state_dict().items()}
    report = fit_cached_cost_network(
        network,
        training,
        validation,
        epochs=3,
        batch_size=8,
        learning_rate=1e-2,
        seed=11,
    )
    assert report["optimizer_steps"] == 12
    assert 1 <= report["selected_epoch"] <= 3
    assert any(
        not torch.equal(before[name], value)
        for name, value in network.state_dict().items()
    )


def test_cost_checkpoint_is_bound_to_exact_base_identity_and_config():
    config = _config()
    network = DetachedHandlingCostNetwork(config, seed=13)
    payload = cost_checkpoint(
        network,
        source_checkpoint_sha256="a" * 64,
        source_policy_digest="b" * 64,
        model_seed=0,
        training_record={"complete": True},
    )
    loaded = load_cost_checkpoint(
        payload,
        config=config,
        device="cpu",
        expected_source_checkpoint_sha256="a" * 64,
        expected_source_policy_digest="b" * 64,
        expected_model_seed=0,
    )
    assert isinstance(loaded, DetachedHandlingCostNetwork)
    with pytest.raises(ValueError, match="model_seed"):
        load_cost_checkpoint(
            payload,
            config=config,
            device="cpu",
            expected_source_checkpoint_sha256="a" * 64,
            expected_source_policy_digest="b" * 64,
            expected_model_seed=1,
        )
