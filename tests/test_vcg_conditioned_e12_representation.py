from types import SimpleNamespace

import torch

from methods.conditioned_vcg.representation_ablation import (
    FULL_RELATIONAL_SUCCESSOR,
    NONRELATIONAL_SUCCESSOR,
    RELATIONAL_CURRENT_CANDIDATE,
    REPRESENTATION_VARIANTS,
    EdgeBlindGraphEncoder,
    agent_class_for_variant,
)
from viability_graph_hierarchy import ViabilityGraphConfig


def test_e12_operational_arms_are_parameter_matched_and_checkpoint_bound():
    counts = []
    for variant in REPRESENTATION_VARIANTS:
        agent_class = agent_class_for_variant(variant)
        agent = agent_class(config=ViabilityGraphConfig(), seed=17)
        counts.append(sum(parameter.numel() for parameter in agent.Q_local.parameters()))
        payload = agent.checkpoint(include_replay=False)
        restored = agent_class.from_checkpoint(payload, seed=19)
        assert restored.checkpoint_metadata()["representation_variant"] == variant
        assert payload["exact_checker_changed"] is False
        assert payload["candidate_generator_changed"] is False
    assert len(set(counts)) == 1


def test_nonrelational_encoder_is_edge_blind_without_dropping_layers():
    agent = agent_class_for_variant(NONRELATIONAL_SUCCESSOR)(
        config=ViabilityGraphConfig(), seed=3
    )
    encoder = agent.Q_local.graph_encoder
    assert isinstance(encoder, EdgeBlindGraphEncoder)
    assert len(encoder.encoder.message_layers) == 3
    features = torch.randn(1, 4, encoder.encoder.input_dim)
    mask = torch.ones(1, 4, dtype=torch.bool)
    edges_a = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edges_b = torch.tensor([[3, 2, 1], [2, 1, 0]], dtype=torch.long)
    assert torch.equal(encoder(features, mask, edges_a), encoder(features, mask, edges_b))


def test_successor_free_adapter_repeats_current():
    current_a, current_b = object(), object()
    records = (
        SimpleNamespace(current_state=current_a, successor_state=object(), action_features=(1,)),
        SimpleNamespace(current_state=current_b, successor_state=object(), action_features=(2,)),
    )
    current, successor, action = agent_class_for_variant(
        RELATIONAL_CURRENT_CANDIDATE
    )._record_tensors(records)
    assert current == (current_a, current_b)
    assert successor == current
    assert action == ((1,), (2,))


def test_full_arm_retains_true_successor_records():
    current, successor = object(), object()
    records = (
        SimpleNamespace(current_state=current, successor_state=successor, action_features=(1,)),
    )
    observed_current, observed_successor, _ = agent_class_for_variant(
        FULL_RELATIONAL_SUCCESSOR
    )._record_tensors(records)
    assert observed_current == (current,)
    assert observed_successor == (successor,)
