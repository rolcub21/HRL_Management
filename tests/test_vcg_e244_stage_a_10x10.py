from pathlib import Path

from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    run as e187_run,
)
from experiments.conditioned_vcg.E209_e191_method_10x10_longstay_268k_v1 import (
    run as e209_run,
    scenario,
    stage as e209_stage,
)
from experiments.conditioned_vcg.E244_e215_scale_matched_10x10_stage_a_303k_v1 import (
    protocol as p,
    run,
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
        with e209_run.implementation():
            assert e187_run.p is p
            assert e187_run.BUDGET_RULE == run.BUDGET_RULE

