from pathlib import Path
from unittest.mock import patch

from PSLAP.viability import ViabilityStatus
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E83_legal_proposal_5x5_pilot_148k_v1 import run as e83
from experiments.conditioned_vcg.E84_unsafe_legal_5x5_pilot_149k_v1 import run as e84


def test_exposure_is_a_real_unsafe_successor_not_unknown(tmp_path: Path):
    report = e84.exposure(tmp_path)
    assert report["status"] == "passed"
    assert report["current_status"] == "SAFE"
    assert report["legal_candidates"] == 5
    assert report["certified_candidates"] == 4
    assert report["unsafe_candidates"] == 1
    assert report["unknown_candidates"] == 0
    assert report["proposal_statuses"][e84.EXPECTED_UNSAFE] == "UNSAFE"
    assert report["representation_parity"]


def test_rank_then_certify_rejects_unsafe_first_proposal(tmp_path: Path):
    args = e84._args(tmp_path)
    env = e84._dense_env(args)
    env.reset(instance=env.sample_episode_instance(e84.TRAIN_SEED_BASE))
    agent = e83._agent(args)

    def unsafe_first(_agent, records, *, epsilon, training):
        del _agent, epsilon, training
        return next((i for i, record in enumerate(records)
                     if record.key == e84.EXPECTED_UNSAFE), 0)

    with reuse.path_cleanup_active(), patch.object(e83, "_rank_index", unsafe_first):
        prelude = e84._certified_prelude(
            env, args, e83.audit._search_config(args),
            e83.audit._liveness_rule(args),
            reuse.TimingInvariantCertificateCache(),
        )
        selected, _record, backup, diagnostics = e83._choose(
            agent, env, arm="rank_then_certify",
            search=e83.audit._search_config(args),
            liveness=e83.audit._liveness_rule(args),
            cache=reuse.TimingInvariantCertificateCache(),
            consecutive_defer=0, nonprogress=0, epsilon=0.0,
            training=False,
        )
    assert prelude["action_type"] == "defer"
    assert prelude["duration"] == 1
    assert diagnostics["attempts"] == 2
    assert diagnostics["legal_candidates"] == 5
    assert len(backup) == 5
    assert e84.EXPECTED_UNSAFE in {record.key for record in backup}
    assert selected.key != e84.EXPECTED_UNSAFE
    assert selected.certificate.status is ViabilityStatus.SAFE
