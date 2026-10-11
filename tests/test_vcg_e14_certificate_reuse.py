from dataclasses import replace

import PSLAP.viability as viability
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import ViabilityCertificateCache
from PSLAP.viability_filter import ViabilitySearchConfig
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E14_certification_scalability_95k import replay
from experiments.conditioned_vcg.E14_certification_scalability_95k import capture
from PSLAP.viability_dataset import recovery_state_to_dict
from vcg_robust_recovery_snapshot_5x5 import make_two_block_5x5_fixture


def _config(**kwargs):
    values = {"max_nodes": 20_000, "search_order": "goal_directed"}
    values.update(kwargs)
    return ViabilitySearchConfig(**values)


def _certificate(state, *, optimized=False):
    context = reuse.path_cleanup_active() if optimized else _null_context()
    with context:
        return viability.analyze_recoverability(
            state, max_nodes=20_000, search_order="goal_directed"
        )


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


def test_timing_projection_is_an_action_transition_homomorphism():
    state = make_two_block_5x5_fixture()
    aged = replace(
        state,
        blocks=tuple(
            replace(block, remaining_time=block.remaining_time - 123.0)
            for block in state.blocks
        ),
    )
    assert state != aged
    assert reuse.physical_state_id(state) == reuse.physical_state_id(aged)
    assert reuse.full_state_id(state) != reuse.full_state_id(aged)
    assert reuse.validate_timing_abstraction(state)["successors_equivariant"]
    assert reuse.validate_timing_abstraction(aged)["successors_equivariant"]


def test_path_cleanup_preserves_certificate_and_witness():
    state = make_two_block_5x5_fixture()
    original = _certificate(state)
    optimized = _certificate(state, optimized=True)
    assert reuse.legal_recovery_actions_optimized(state) == viability.legal_recovery_actions(state)
    assert reuse.certificate_to_dict(optimized) == reuse.certificate_to_dict(original)


def test_timing_invariant_cache_reuses_clock_variants_only():
    state = make_two_block_5x5_fixture()
    aged = replace(
        state,
        blocks=tuple(
            replace(block, remaining_time=block.remaining_time - 1.0)
            for block in state.blocks
        ),
    )
    certificate = _certificate(state)
    key = (state, None, 20_000, None, "goal_directed")
    aged_key = (aged, None, 20_000, None, "goal_directed")

    ordinary = ViabilityCertificateCache()
    ordinary[key] = certificate
    assert ordinary.get(aged_key) is None

    projected = reuse.TimingInvariantCertificateCache()
    projected[key] = certificate
    assert projected.get(aged_key) == certificate


def test_positive_store_retains_constructive_suffixes_and_respects_horizon():
    state = make_two_block_5x5_fixture()
    certificate = _certificate(state)
    store = reuse.PositiveProofStore()
    added = store.insert_certificate(state, certificate)
    assert added == len(certificate.witness) + 1

    first_successor = reuse.apply_enumerated_action(
        reuse.timing_erased_state(state), certificate.witness[0]
    )
    suffix = store.lookup(first_successor, _config())
    assert suffix is not None
    assert suffix.status is ViabilityStatus.SAFE
    assert suffix.witness == certificate.witness[1:]
    assert suffix.exact_recovery_rank is None
    assert not reuse.replay_witness(first_successor, suffix).blocks

    too_shallow = store.lookup(
        state, _config(max_depth=len(certificate.witness) - 1)
    )
    assert too_shallow is None
    assert store.lookup(
        state,
        ViabilitySearchConfig(max_nodes=20_000, search_order="breadth_first"),
    ) is None


def test_unknown_is_not_inserted_as_a_positive_proof():
    state = make_two_block_5x5_fixture()
    unknown = viability.analyze_recoverability(
        state, max_nodes=1, search_order="goal_directed"
    )
    assert unknown.status is ViabilityStatus.UNKNOWN
    store = reuse.PositiveProofStore()
    assert store.insert_certificate(state, unknown) == 0
    assert len(store) == 0


def test_certificate_serialization_round_trip():
    state = make_two_block_5x5_fixture()
    certificate = _certificate(state)
    restored = reuse.certificate_from_dict(reuse.certificate_to_dict(certificate))
    assert restored == certificate


def test_offline_stages_separate_clock_and_constructive_suffix_reuse():
    state = make_two_block_5x5_fixture()
    certificate = _certificate(state)
    successor = reuse.apply_enumerated_action(
        reuse.timing_erased_state(state), certificate.witness[0]
    )
    successor_certificate = _certificate(successor)
    aged = replace(
        state,
        blocks=tuple(
            replace(block, remaining_time=block.remaining_time - 7.0)
            for block in state.blocks
        ),
    )
    search = reuse.config_to_dict(_config())

    def query(index, value, result, seconds):
        return {
            "query_index": index,
            "full_state_id": reuse.full_state_id(value),
            "physical_state_id": reuse.physical_state_id(value),
            "search": search,
            "original_cache_hit": False,
            "original_search_seconds": seconds,
            "result": reuse.certificate_to_dict(result),
        }

    queries = [
        query(0, state, certificate, 1.0),
        query(1, aged, certificate, 1.1),
        query(2, successor, successor_certificate, 0.5),
    ]
    trace = {
        "queries": queries,
        "physical_states": {
            reuse.physical_state_id(state): recovery_state_to_dict(
                reuse.timing_erased_state(state)
            ),
            reuse.physical_state_id(successor): recovery_state_to_dict(successor),
        },
    }
    records = {
        replay._query_key(item): {
            "search_seconds": seconds,
            "timing_abstraction_validation_seconds": 0.01,
            "exact_certificate_and_witness_match": True,
            "positive_witness_replayed": True,
            "certificate": item["result"],
        }
        for item, seconds in ((queries[0], 0.4), (queries[2], 0.2))
    }

    timing = replay._stage_timing_key(trace)
    assert timing["accumulated"]["timing_key_hits"] == 1
    assert timing["accumulated"]["searches_completed"] == 2

    suffix = replay._stage_suffix(trace, records)
    assert suffix["accumulated"]["searches"] == 1
    assert suffix["accumulated"]["outcome_cache_hits"] == 1
    assert suffix["accumulated"]["constructive_proof_hits"] == 1
    assert suffix["proof_store"]["unknown_insertions"] == 0


def test_capture_records_recovery_miss_only_when_exact_check_is_requested():
    state = make_two_block_5x5_fixture()
    action = viability.legal_recovery_actions(state)[0]
    successor = viability.apply_recovery_action(state, action)
    tracer = capture.QueryTracer(reserve_queue_cells=True)
    tracer.frontier_index = 0
    tracer.decision_epoch = 0
    tracer.register_recovery_successor(successor, action.kind)
    key = (successor, None, 20_000, None, "goal_directed")

    tracer.cache_lookup(key, None)  # preliminary scan
    assert tracer.queries == []
    with tracer.role("deliver"):
        tracer.cache_lookup(key, None)  # actual ordered check
    assert len(tracer.queries) == 1
    assert tracer.queries[0]["check_role"] == "deliver"
