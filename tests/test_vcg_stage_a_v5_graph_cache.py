import torch

from experiments.conditioned_vcg.target_size_dense_stage_a_v3 import acceleration as v3
from experiments.conditioned_vcg.target_size_dense_stage_a_v5 import acceleration as v5
from tests.test_viability_graph_hierarchy import make_snapshot
from viability_graph_hierarchy import (
    ViabilityGraphConfig,
    ViabilityGraphHierarchyAgent,
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


def test_cached_graph_scores_match_existing_deduplicated_path():
    agent = _agent()
    records = prepare_viability_snapshot(make_snapshot()).records
    with torch.no_grad():
        expected = v3._deduplicated_no_grad_scores(records, agent.Q_local)
        observed = v5._deduplicated_cached_scores(
            agent, records, agent.Q_local
        )
    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)


def test_second_score_reuses_every_unique_graph_input():
    agent = _agent()
    records = prepare_viability_snapshot(make_snapshot()).records
    with torch.no_grad():
        v5._deduplicated_cached_scores(agent, records, agent.Q_local)
    first = dict(v5.audit_payload(agent)["graph_input_tensor_cache"])
    with torch.no_grad():
        v5._deduplicated_cached_scores(agent, records, agent.Q_local)
    second = v5.audit_payload(agent)["graph_input_tensor_cache"]
    assert second["misses"] == first["misses"]
    assert second["hits"] - first["hits"] == first["misses"]
    assert second["current_cache_entries"] == first["misses"]
    assert second["topology_cache_entries"] == 1
