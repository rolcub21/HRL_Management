from types import SimpleNamespace

import pytest
import torch

import train_vcg_v11_conditioned_handling_iterative as trainer
from vcg_v11_conditioned_handling import (
    ConditionedFutureHandlingNetwork,
    ConditionedHandlingConfig,
    PendingHandlingObservation,
    future_samples_from_episode,
)
from vcg_v11_nested_handling import DetachedHandlingCostNetwork


def _source_config():
    return SimpleNamespace(
        graph_embedding_dim=2,
        action_embedding_dim=3,
        head_hidden_dim=7,
    )


def _conditioned_config():
    source = _source_config()
    return ConditionedHandlingConfig(
        feature_dim=3 * source.graph_embedding_dim + source.action_embedding_dim,
        hidden_dim=source.head_hidden_dim,
        lambda_max=0.2,
    )


def test_nested_warm_start_ignores_lambda_and_preserves_future_prediction():
    source_config = _source_config()
    source = DetachedHandlingCostNetwork(source_config, seed=17)
    target = ConditionedFutureHandlingNetwork(_conditioned_config(), seed=23)
    target.initialize_from_nested_head(source)
    features = torch.randn(11, _conditioned_config().feature_dim)

    with torch.no_grad():
        expected = source(features)
        for value in (0.001, 0.025, 0.1, 0.2):
            observed = target(features, value)
            torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)

    assert torch.count_nonzero(target.future_head[0].weight[:, -1]).item() == 0


def test_future_mc_labels_exclude_the_current_known_rehandle():
    actions = (
        ("reconfigure", 1),
        ("deliver", 0),
        ("reconfigure", 1),
        ("deliver", 0),
    )
    observations = tuple(
        PendingHandlingObservation(
            feature=torch.full((_conditioned_config().feature_dim,), float(index)),
            preference_lambda=0.1,
            immediate_rehandles=immediate,
            realized_rehandles=immediate,
            action_type=action,
            mode_id=0,
            candidate_key=f"candidate-{index}",
        )
        for index, (action, immediate) in enumerate(actions)
    )

    samples = future_samples_from_episode(observations)

    assert [sample.immediate_rehandles for sample in samples] == [1, 0, 1, 0]
    assert [sample.future_rehandles for sample in samples] == [1, 1, 0, 0]


def test_future_mc_labels_reject_action_semantic_mismatch():
    observation = PendingHandlingObservation(
        feature=torch.zeros(_conditioned_config().feature_dim),
        preference_lambda=0.1,
        immediate_rehandles=1,
        realized_rehandles=0,
        action_type="reconfigure",
        mode_id=0,
        candidate_key="bad",
    )
    with pytest.raises(ValueError, match="action-semantic"):
        future_samples_from_episode((observation,))


def test_conditioned_future_prediction_is_nonnegative_and_lambda_is_bounded():
    config = _conditioned_config()
    network = ConditionedFutureHandlingNetwork(config, seed=31)
    features = torch.randn(5, config.feature_dim)
    assert bool((network(features, 0.13) >= 0).all())
    with pytest.raises(ValueError, match="lie in"):
        network(features, 0.21)


def test_continuous_round_schedule_is_stratified_reproducible_and_interior():
    first = trainer.stratified_lambda_schedule(0)
    second = trainer.stratified_lambda_schedule(0)
    assert first == second
    assert len(first) == trainer.EPISODES_PER_ROUND
    values = [item["behavior_lambda"] for item in first]
    assert all(0.0 < value < 0.2 for value in values)
    strata = sorted(int(value / 0.2 * trainer.EPISODES_PER_ROUND) for value in values)
    assert strata == list(range(trainer.EPISODES_PER_ROUND))
    assert len(set(values)) == len(values)


def test_config_rejects_invalid_lambda_interval():
    with pytest.raises(ValueError, match="lambda_max"):
        ConditionedHandlingConfig(feature_dim=4, lambda_max=0.0)
