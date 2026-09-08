from PSLAP.viability_filter import ViabilitySearchConfig
import PSLAP.viability as viability
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.development.D11_shared_search_opportunity_audit import audit
from experiments.conditioned_vcg.development.D11_shared_search_opportunity_audit import program
from vcg_robust_recovery_snapshot_5x5 import make_two_block_5x5_fixture


def test_deterministic_sample_retains_three_complete_anchors_and_all_frontiers():
    sizes = (2, 4, 6, 8, 10)
    queries = []
    index = 0
    roles = ("current_state", "accept", "deliver", "reconfigure")
    for frontier, size in enumerate(sizes):
        for offset in range(size):
            queries.append({
                "query_index": index,
                "frontier_index": frontier,
                "check_role": roles[offset % len(roles)],
                "result": {"status": "SAFE"},
            })
            index += 1
    trace = {
        "queries": queries,
        "completed_frontiers": [
            {"frontier_index": value} for value in range(len(sizes))
        ],
    }
    sample = program.select_queries(trace)
    selected = set(sample["query_indices"])
    anchors = {
        item["frontier_index"]
        for item in sample["anchor_frontiers"].values()
    }
    for query in queries:
        if query["frontier_index"] in anchors:
            assert query["query_index"] in selected
    assert sample["completed_frontier_count"] == len(sizes)
    assert {
        query["frontier_index"] for query in queries
        if query["query_index"] in selected
    } == set(range(len(sizes)))


def test_instrumentation_preserves_exact_certificate_and_counts_expansions():
    state = make_two_block_5x5_fixture()
    config = ViabilitySearchConfig(
        max_nodes=20_000, search_order="goal_directed"
    )
    with reuse.path_cleanup_active():
        expected = viability.analyze_recoverability(
            state, max_nodes=20_000, search_order="goal_directed"
        )
    observed, events, witness_ids, _wall = audit.instrumented_search(
        state, config
    )
    assert reuse.certificate_to_dict(observed) == reuse.certificate_to_dict(expected)
    assert len(events) == observed.explored_nodes
    assert witness_ids
    assert all(event["successor_calls"] <= event["action_count"] for event in events)


def _event(state_id, *, expansion=0, seconds=1.0):
    return {
        "expansion_index": expansion,
        "canonical_state_id": state_id,
        "action_enumeration_seconds": seconds,
        "successor_application_seconds": 0.0,
        "cache_key_construction_seconds": 0.0,
        "action_signature": f"signature-{state_id}",
        "action_payload_bytes": 10,
        "estimated_successor_identifier_bytes": 32,
        "on_returned_witness": False,
    }


def _root(query, frontier, events):
    return {
        "query_index": query,
        "frontier_index": frontier,
        "check_role": "reconfigure",
        "reference_path_search_seconds": 10.0,
        "instrumented_search_wall_seconds": 10.0,
        "certificate_and_witness_exact_match": True,
        "events": events,
    }


def test_overlap_scopes_are_mutually_attributed():
    records = [
        _root(0, 0, [_event("A"), _event("B", expansion=1)]),
        _root(1, 0, [_event("A"), _event("C", expansion=1)]),
        _root(
            2,
            1,
            [_event("B"), _event("D", expansion=1), _event("B", expansion=2)],
        ),
    ]
    result = audit.overlap_analysis(records)
    assert result["sample"]["expanded_state_occurrences"] == 7
    assert result["sample"]["unique_canonical_states"] == 4
    assert result["overlap_by_scope"]["within_one_root_search"]["occurrences"] == 1
    assert result["overlap_by_scope"]["across_candidates_within_one_frontier"]["occurrences"] == 1
    assert result["overlap_by_scope"]["across_decision_frontiers"]["occurrences"] == 1
