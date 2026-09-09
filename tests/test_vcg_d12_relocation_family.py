from dataclasses import replace

import pytest

from PSLAP import viability as v
from PSLAP import viability_candidates as candidate_module
from PSLAP.relocation_family_certification import (
    RELOCATION_FAMILY_CERTIFICATION,
    ValidatedRelocationFamilyAnchor,
    try_relocation_family,
)
from PSLAP.viability_candidates import (
    ViabilityActionType,
    ViabilityCertificateCache,
    enumerate_viability_candidates,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from example.small_rooms_env import SmallRoomsEnv
from experiments.conditioned_vcg.development.D12_relocation_family_certification import proofs
from experiments.conditioned_vcg.development.D12_relocation_family_certification.run import select_queries
from vcg_robust_recovery_snapshot_5x5 import make_two_block_5x5_fixture


@pytest.fixture
def family():
    state = make_two_block_5x5_fixture()
    certificate = v.analyze_recoverability(state, search_order="goal_directed")
    return state, proofs.ValidatedAnchor(state, certificate)


def relocations(state):
    return [a for a in v.legal_recovery_actions(state) if a.kind is v.RecoveryActionKind.RELOCATION]


def test_targeted_actions_equal_native_enumeration_across_successors(family):
    state, _ = family
    states = [state] + [v.apply_recovery_action(state, a) for a in relocations(state)[:4]]
    for current in states:
        native = v.legal_recovery_actions(current)
        for block in current.blocks:
            for kind, destinations in (
                (v.RecoveryActionKind.RELOCATION, current.traversable),
                (v.RecoveryActionKind.DELIVERY, current.traversable),
            ):
                for destination in destinations:
                    action = proofs.targeted_action(current, kind, block.label, destination)
                    matches = [a for a in native if (a.kind, a.block_label, a.destination)
                               == (kind, block.label, destination)]
                    assert action == (matches[0] if matches else None)


def test_constructed_family_witnesses_replay_with_native_verifier(family):
    state, anchor = family
    hits = 0
    transporter_mismatches = 0
    for action in relocations(state):
        successor = v.apply_recovery_action(state, action)
        attempt = proofs.try_relocation(anchor, action, successor, ViabilitySearchConfig())
        if attempt.certificate is None:
            continue
        hits += 1
        certificate = attempt.certificate
        restored = v.apply_recovery_action(successor, certificate.witness[0])
        transporter_mismatches += restored.agent_position != state.agent_position
        current = successor
        for step in certificate.witness:
            current = v.apply_recovery_action(current, step)
        assert not current.blocks
        assert certificate.exact_recovery_rank is None
        assert certificate.witness_primitive_steps == sum(a.steps for a in certificate.witness)
    assert hits > 0
    assert transporter_mismatches > 0


def test_limits_and_breadth_first_fall_back_to_exact_search(family):
    state, anchor = family
    action = relocations(state)[0]
    successor = v.apply_recovery_action(state, action)
    for config in (
        ViabilitySearchConfig(max_depth=0),
        ViabilitySearchConfig(max_primitive_steps=0),
        ViabilitySearchConfig(search_order="breadth_first"),
    ):
        assert proofs.try_relocation(anchor, action, successor, config).certificate is None
        actual, source = proofs.certify_with_fallback(anchor, action, successor, config)
        expected = v.analyze_recoverability(
            successor, max_depth=config.max_depth, max_nodes=config.max_nodes,
            max_primitive_steps=config.max_primitive_steps, search_order=config.search_order,
        )
        assert source == "exact_fallback"
        assert actual == expected


def test_changed_reservation_prevents_undo_and_rejects_family_match(family):
    state, _ = family
    action = relocations(state)[0]
    anchor = proofs.ValidatedAnchor(state, v.analyze_recoverability(state, search_order="goal_directed"))
    successor = v.apply_recovery_action(state, action)
    reserved = replace(successor, reserved_cells=successor.reserved_cells | {action.source})
    assert proofs.targeted_action(reserved, action.kind, action.block_label, action.source) is None
    attempt = proofs.try_relocation(anchor, action, reserved, ViabilitySearchConfig())
    assert attempt.certificate is None
    assert attempt.reason == "successor_mismatch"


def test_rejects_invalid_anchor_and_changed_successor(family):
    state, anchor = family
    unknown = v.analyze_recoverability(state, max_nodes=1, search_order="goal_directed")
    assert unknown.is_unknown
    with pytest.raises(ValueError):
        proofs.ValidatedAnchor(state, unknown)
    safe = v.analyze_recoverability(state, search_order="goal_directed")
    with pytest.raises(ValueError):
        proofs.ValidatedAnchor(state, replace(safe, witness=()))
    action = relocations(state)[0]
    successor = v.apply_recovery_action(state, action)
    changed = replace(successor, reserved_cells=successor.reserved_cells | {action.source})
    assert proofs.try_relocation(anchor, action, changed, ViabilitySearchConfig()).reason == "successor_mismatch"


def test_clock_variants_can_connect_but_invalid_paths_cannot(family):
    state, anchor = family
    for action in relocations(state):
        successor = v.apply_recovery_action(state, action)
        if proofs.try_relocation(anchor, action, successor, ViabilitySearchConfig()).certificate:
            break
    aged = replace(successor, blocks=tuple(replace(b, remaining_time=-100) for b in successor.blocks))
    assert proofs.try_relocation(anchor, action, aged, ViabilitySearchConfig(max_nodes=1)).certificate
    forged = replace(action, transport_path=(action.source, action.destination))
    if forged == action:
        forged = replace(action, approach_path=())
    assert proofs.try_relocation(anchor, forged, successor, ViabilitySearchConfig()).reason == "invalid_outgoing_action"


def test_sampling_is_deterministic_and_excludes_unfinished_frontiers():
    trace = {
        "completed_frontiers": [{"frontier_index": 0}],
        "queries": [{"frontier_index": i // 10, "query_index": i,
                     "check_role": "reconfigure", "result": {}} for i in range(20)],
    }
    assert [q["query_index"] for q in select_queries(trace, 3)] == [0, 4, 9]
    assert [q["query_index"] for q in select_queries(trace, 1)] == [5]


def test_undo_can_succeed_while_rejoining_first_macro_is_impossible(family):
    # The agent initially shares a block's cell. It can approach either side
    # from there, but after undo on the left it cannot cross that stored block.
    fixture, _ = family
    template = fixture.blocks[0]
    cells = frozenset((0, i) for i in range(7))
    state = v.RecoveryState(
        rows=1, cols=7, traversable=cells, storage_cells=cells - {(0, 6)},
        exits=((0, 6),), agent_position=(0, 3),
        blocks=tuple(replace(template, label=label, position=(0, col))
                     for label, col in (("A", 5), ("B", 1), ("Z", 3))),
    )
    certificate = v.analyze_recoverability(state, search_order="goal_directed")
    assert certificate.is_safe
    assert certificate.witness[0].block_label == "A"
    anchor = proofs.ValidatedAnchor(state, certificate)
    action = proofs.targeted_action(state, v.RecoveryActionKind.RELOCATION, "B", (0, 0))
    successor = v.apply_recovery_action(state, action)
    assert proofs.targeted_action(successor, action.kind, "B", (0, 1)) is not None
    attempt = proofs.try_relocation(anchor, action, successor, ViabilitySearchConfig())
    assert attempt.reason == "first_macro_unavailable"
    assert attempt.certificate is None


def test_family_hit_does_not_launch_exact_search(family, monkeypatch):
    state, anchor = family
    for action in relocations(state):
        successor = v.apply_recovery_action(state, action)
        if proofs.try_relocation(anchor, action, successor, ViabilitySearchConfig()).certificate:
            break
    def forbidden(*args, **kwargs):
        pytest.fail("family hit must not launch exact search")
    monkeypatch.setattr(v, "analyze_recoverability", forbidden)
    certificate, source = proofs.certify_with_fallback(anchor, action, successor, ViabilitySearchConfig())
    assert source == "family"
    assert certificate.is_safe


def test_online_adapter_preserves_native_current_state_cache_and_witness(family):
    from PSLAP import viability_candidates as candidates
    from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
    from experiments.conditioned_vcg.development.D12_relocation_family_certification.online import FamilyVerifier
    state, anchor = family
    config = ViabilitySearchConfig()
    cache = reuse.TimingInvariantCertificateCache()
    adapter = FamilyVerifier(candidates._analyze_cached)
    initial = adapter(state, config, cache)
    assert initial[0].witness == anchor.witness
    for action in relocations(state):
        successor = v.apply_recovery_action(state, action)
        if proofs.try_relocation(anchor, action, successor, config).certificate:
            break
    family_certificate, hit, seconds = adapter(successor, config, cache)
    assert family_certificate.reason.startswith("safe:validated_relocation")
    assert not hit
    assert cache.get(candidates._certificate_key(successor, config)) is None
    adapter.begin()
    current, _, _ = adapter(successor, config, cache)
    expected = v.analyze_recoverability(successor, search_order="goal_directed")
    assert current == expected
    assert cache.get(candidates._certificate_key(successor, config)) == expected


def test_online_comparison_uses_frontier_choices_and_decisions_not_proof_metadata():
    from experiments.conditioned_vcg.development.D12_relocation_family_certification.online import compare_prefix
    reference = {"frontiers": [{"candidate_keys": ["a", "b"], "decision_epoch": 1}],
                 "decisions": [{"selected_key": "a", "decision_epoch": 1}]}
    assert compare_prefix(reference, reference["frontiers"], reference["decisions"])["selected_action_or_epoch_mismatches"] == []
    changed = [{"candidate_keys": ["a"], "decision_epoch": 1}]
    assert compare_prefix(reference, changed, reference["decisions"])["candidate_key_or_epoch_mismatches"] == [0]


def test_full_selection_includes_every_completed_relocation_query():
    trace = {"completed_frontiers": [{"frontier_index": 0}],
             "queries": [{"frontier_index": 0, "query_index": i,
                          "check_role": "reconfigure", "result": {}} for i in range(10)]}
    assert len(select_queries(trace, None)) == 10


def test_integrated_core_matches_development_proof_rule(family):
    state, development_anchor = family
    core_anchor = ValidatedRelocationFamilyAnchor(
        state,
        v.analyze_recoverability(state, search_order="goal_directed"),
    )
    for action in relocations(state):
        successor = v.apply_recovery_action(state, action)
        development = proofs.try_relocation(
            development_anchor, action, successor, ViabilitySearchConfig()
        )
        integrated = try_relocation_family(
            core_anchor, action, successor, ViabilitySearchConfig()
        )
        assert integrated.certificate == development.certificate
        assert integrated.reason == development.reason


def test_integrated_frontier_uses_family_proofs_without_caching_them():
    env = SmallRoomsEnv(
        grid_rows=5,
        grid_cols=5,
        number_blocks=2,
        choose_storage=False,
    )
    env.reset(instance=env.sample_episode_instance(95_120))
    for block, cell in zip(env.blocks, ((2, 2), (2, 3))):
        block.position = cell
        block.storage_location = cell
        block.carrying = False
        block.stored = True
        block.delivered = False
        block.stored_time_step = 0
        block.storage_steps_elapsed = 0
    env.current_state = (1, 1)
    env.time_steps = max(block.storage_steps_needed for block in env.blocks)
    config = ViabilitySearchConfig(max_nodes=20_000)
    exact = enumerate_viability_candidates(
        env,
        consecutive_defer_decisions=0,
        search_config=config,
    )
    cache = ViabilityCertificateCache()
    integrated = enumerate_viability_candidates(
        env,
        consecutive_defer_decisions=0,
        search_config=config,
        cache=cache,
        recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION,
    )

    assert tuple(item.key for item in integrated.candidates) == tuple(
        item.key for item in exact.candidates
    )
    assert integrated.audit.relocation_family_anchor_available
    assert integrated.audit.relocation_family_proof_count > 0
    assert (
        integrated.audit.relocation_family_attempt_count
        == integrated.audit.relocation_family_proof_count
        + integrated.audit.relocation_family_miss_count
    )
    family_candidates = tuple(
        item for item in integrated.candidates
        if item.action_type is ViabilityActionType.RECONFIGURE
        and item.certificate.reason.startswith(
            "safe:validated_relocation_undo_and_rejoin"
        )
    )
    assert len(family_candidates) == integrated.audit.relocation_family_proof_count
    for candidate in family_candidates:
        current = candidate.successor_state
        for action in candidate.certificate.witness:
            current = v.apply_recovery_action(current, action)
        assert not current.blocks
        assert cache.get(
            candidate_module._certificate_key(
                candidate.successor_state, config
            )
        ) is None


def test_integrated_strategy_rejects_unknown_name():
    env = SmallRoomsEnv(
        grid_rows=5,
        grid_cols=5,
        number_blocks=1,
        choose_storage=False,
    )
    env.reset(instance=env.sample_episode_instance(95_121))
    with pytest.raises(ValueError, match="recovery_certification_strategy"):
        enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            recovery_certification_strategy="similarity_guess",
        )


def test_integrated_confirmation_distinguishes_complete_and_censored_references():
    from experiments.conditioned_vcg.development.D12_relocation_family_certification import confirmation

    frontier = {"candidate_keys": ["deliver:B1:1:1"], "decision_epoch": 3}
    decision = {"selected_key": "deliver:B1:1:1", "decision_epoch": 3}
    inputs = {
        "medium_reference": {
            "frontiers": [frontier],
            "decisions": [decision],
        },
        "high_reference": {
            "row": {"completed_frontiers": [frontier]},
        },
    }
    longer_frontiers = [frontier, {**frontier, "decision_epoch": 4}]

    high = confirmation._compare_reference(
        inputs,
        "size_10x10_occ_high",
        longer_frontiers,
        (),
        completed=True,
    )
    assert high["reference_is_complete"] is False
    assert high["reference_prefix_fully_covered"] is True
    assert high["complete_reference_length_match"] is None
    confirmation._validate_reference_comparison(
        "size_10x10_occ_high", high, strict=True
    )

    medium = confirmation._compare_reference(
        inputs,
        "size_10x10_occ_medium",
        longer_frontiers,
        [decision],
        completed=True,
    )
    assert medium["reference_is_complete"] is True
    assert medium["complete_reference_length_match"] is False
    with pytest.raises(confirmation.ConfirmationError, match="frontier length"):
        confirmation._validate_reference_comparison(
            "size_10x10_occ_medium", medium, strict=True
        )
