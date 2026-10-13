from experiments.conditioned_vcg.E83_legal_proposal_5x5_pilot_148k_v1 import legal, run
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    ViabilityActionType, ViabilityCertificateCache,
    enumerate_viability_candidates,
)
from types import SimpleNamespace
from unittest.mock import patch


def test_legal_frontier_contains_certified_candidates_with_matching_features(tmp_path):
    result = run.self_check(tmp_path)
    assert result["status"] == "passed"
    assert result["representation_parity"]
    assert all(legal >= certified for legal, certified in zip(
        result["physical_candidates"], result["certified_candidates"]
    ))


def test_unsafe_proposal_is_not_promoted_to_candidate():
    proposal = SimpleNamespace(
        action_type=ViabilityActionType.ACCEPT, successor_state=object(),
    )
    unsafe = SimpleNamespace(status=ViabilityStatus.UNSAFE)
    with patch.object(run.vc, "_analyze_cached", return_value=(unsafe, False, 0.1)):
        candidate, hit, elapsed = run._certify_one(
            proposal, None, None, None, None,
        )
    assert candidate is None
    assert not hit
    assert elapsed == 0.1


def test_backup_retains_whole_next_frontier(tmp_path):
    args = run.runtime_args(tmp_path)
    env = run.audit._new_env(args, run.dense.FROZEN_OBJECTIVE_SPEC)
    env.reset(instance=env.sample_episode_instance(run.TRAIN_SEED_BASE))
    snapshot = enumerate_viability_candidates(
        env, consecutive_defer_decisions=0,
        search_config=run.audit._search_config(args),
        liveness_rule=run.audit._liveness_rule(args),
        cache=ViabilityCertificateCache(),
    )
    records = tuple(legal.masked_certified_record(item, snapshot.recovery_state)
                    for item in snapshot.candidates)
    agent = run._agent(args)
    run._append_transition(agent, (records[0], 1.0, 1, False),
                           records, updates=0)
    assert len(agent.replay.memory[0].next_candidates) == len(records)
