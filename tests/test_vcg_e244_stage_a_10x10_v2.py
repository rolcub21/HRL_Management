from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    run as e187_run,
)
from experiments.conditioned_vcg.E209_e191_method_10x10_longstay_268k_v1 import (
    run as e209_run,
    scenario,
    stage as e209_stage,
)
from experiments.conditioned_vcg.E244_e215_scale_matched_10x10_stage_a_303k_v2 import (
    protocol as p,
    roundoff,
    run,
)
from experiments.conditioned_vcg.E189_e188_roundoff_repair_248k_v1 import (
    roundoff as e189_roundoff,
)
from projects.temporal_consequence_realization.method_conformance import (
    validate_protocol,
)


def test_final_exploration_schedule_is_job_normalized():
    assert p.EPSILON_DECAY_DECISIONS == round(57_300 * 64 / 36) == 101_867
    assert p.EPSILON_START == 0.90
    assert p.EPSILON_END == 0.05
    assert p.EPSILON_TARGET_FLOOR_EPISODE == 220


def test_budget_and_scale_are_locked():
    assert p.MODEL_SEEDS == (0, 1, 2)
    assert p.TOTAL_EPISODES == 300
    assert p.VALIDATION_EVERY == 20
    assert p.CHECKPOINT_EVERY == 1
    assert scenario.TOTAL_JOBS == 64
    assert scenario.CAPACITY == 63
    assert scenario.MEAN_DURATION / scenario.CAPACITY == 20.0


def test_contract_is_current_and_development_only():
    contract = p.contract()
    declaration = validate_protocol(contract, source=p.PROTOCOL)
    assert declaration["method_revision"] == p.METHOD_REVISION
    assert declaration["claim_level"] == "expressibility"
    assert contract["architecture"]["identical_to_e215_e218"] is True
    assert contract["initialization"]["kind"] == "fresh_random_from_model_seed"
    assert contract["test_panel_opened"] is False
    assert contract["paper_evidence"] is False
    assert all((p.ROOT / relative).is_file() for relative in contract["source_sha256"])


def test_protocol_reaches_both_inherited_training_seams():
    with run.implementation():
        assert e209_run.p is p
        assert e209_stage.p is p
        assert e209_run.repaired_audit is nullcontext
        with e209_run.implementation():
            assert e187_run.p is p
            assert e187_run.BUDGET_RULE == run.BUDGET_RULE


def test_roundoff_audit_is_bound_to_the_e244_environment_once():
    assert run.repaired_audit is roundoff.repaired_audit
    with roundoff.repaired_audit():
        assert e189_roundoff.p.TargetDenseEnv is p.TargetDenseEnv


def test_roundoff_audit_receives_e244_episode_records(monkeypatch):
    proxies = []

    class RecordingProxy(e189_roundoff._MathProxy):
        def __init__(self):
            super().__init__()
            proxies.append(self)

    records = tuple(
        {"step_reward_by_objective": {"selected": -122.0}}
        for _index in range(630)
    )
    monkeypatch.setattr(e189_roundoff, "_MathProxy", RecordingProxy)
    monkeypatch.setattr(
        p.TargetDenseEnv,
        "objective_audit_summary",
        lambda _env: {"status": "synthetic"},
    )

    with roundoff.repaired_audit():
        summary = p.TargetDenseEnv.objective_audit_summary(
            SimpleNamespace(objective_audit_records=records)
        )

    assert summary == {"status": "synthetic"}
    assert len(proxies) == 1
    assert proxies[0].primitive_steps == 630
    assert proxies[0].bound > 4.71482053399e-9
    assert proxies[0].isclose(
        0.0, 4.71482053399e-9, rel_tol=0.0, abs_tol=1.0e-9
    )


def test_contract_authenticates_the_v2_roundoff_binding():
    contract = p.contract()
    assert contract["audit_repair"] == {
        "kind": "single E189 episode-derived floating-point bound",
        "environment": "E244 10x10 TargetDenseEnv",
        "nested_parent_wrapper_suppressed": True,
        "learning_semantics_changed": False,
    }
    assert any(path.endswith("/roundoff.py") for path in contract["source_sha256"])
