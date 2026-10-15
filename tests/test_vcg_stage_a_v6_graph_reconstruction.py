import torch

from experiments.conditioned_vcg.target_size_dense_stage_a_v5 import (
    acceleration as v5,
)
from experiments.conditioned_vcg.target_size_dense_stage_a_v6 import (
    acceleration as v6,
)
from tests.test_viability_graph_hierarchy import make_snapshot
from viability_graph_hierarchy import (
    ViabilityGraphConfig,
    ViabilityGraphHierarchyAgent,
    ViabilityGraphTransition,
    prepare_viability_snapshot,
)


def _agent():
    return ViabilityGraphHierarchyAgent(
        config=ViabilityGraphConfig(
            graph_hidden_dim=8,
            graph_embedding_dim=8,
            message_passing_steps=1,
            action_embedding_dim=8,
            head_hidden_dim=16,
            batch_size=2,
            replay_capacity=16,
        ),
        seed=7,
        epsilon=0.0,
    )


def _clear(agent):
    for name in (
        "_graph_input_cache",
        "_graph_topology_cache",
        "_graph_static_template_cache",
        "_graph_input_cache_signature",
        "_graph_input_cache_audit",
        "_target_score_cache",
        "_target_score_cache_generation",
        "_target_score_cache_audit",
    ):
        if hasattr(agent, name):
            delattr(agent, name)


def test_direct_graph_inputs_exactly_match_v5_encoding():
    agent = _agent()
    records = prepare_viability_snapshot(make_snapshot()).records
    states = tuple(record.current_state for record in records) + tuple(
        record.successor_state for record in records
    )
    for state in states:
        expected = v5._encoded_input(agent, state, agent.Q_local)
        _clear(agent)
        observed = v6._encoded_input(agent, state, agent.Q_local)
        torch.testing.assert_close(observed[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(observed[1], expected[1], rtol=0, atol=0)
        _clear(agent)


def test_v6_scores_match_v5_scores():
    agent = _agent()
    records = prepare_viability_snapshot(make_snapshot()).records
    with torch.no_grad():
        expected = v5._deduplicated_cached_scores(
            agent, records, agent.Q_local
        )
        _clear(agent)
        observed = v6._deduplicated_cached_scores(
            agent, records, agent.Q_local
        )
    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)


def test_v6_reuses_graph_inputs_and_declares_larger_bound():
    agent = _agent()
    records = prepare_viability_snapshot(make_snapshot()).records
    with torch.no_grad():
        v6._deduplicated_cached_scores(agent, records, agent.Q_local)
    first = dict(v6.audit_payload(agent)["graph_input_tensor_cache"])
    with torch.no_grad():
        v6._deduplicated_cached_scores(agent, records, agent.Q_local)
    second = v6.audit_payload(agent)["graph_input_tensor_cache"]
    assert second["misses"] == first["misses"]
    assert second["hits"] - first["hits"] == first["misses"]
    assert second["maximum_cache_entries"] == 1_000_000
    assert second["static_template_cache_entries"] == 1
    assert second["resident_feature_tensor_bytes"] > 0


def test_v6_td_batch_matches_v5_td_batch():
    agent = _agent()
    records = prepare_viability_snapshot(make_snapshot()).records
    transitions = (
        ViabilityGraphTransition(
            chosen=records[0],
            reward=1.25,
            duration=3,
            next_candidates=records,
            done=False,
        ),
        ViabilityGraphTransition(
            chosen=records[-1],
            reward=-0.5,
            duration=2,
            next_candidates=records,
            done=False,
        ),
    )
    with torch.no_grad():
        expected_prediction, expected_target = (
            v5.accelerated_cached_tensor_td_batch(agent, transitions)
        )
        _clear(agent)
        observed_prediction, observed_target = v6.accelerated_cached_td_batch(
            agent, transitions
        )
    torch.testing.assert_close(
        observed_prediction, expected_prediction, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        observed_target, expected_target, rtol=1e-6, atol=1e-6
    )


def test_v6_exposes_v5_compatible_td_batch_name():
    assert (
        v6.accelerated_cached_tensor_td_batch
        is v6.accelerated_direct_tensor_td_batch
    )
