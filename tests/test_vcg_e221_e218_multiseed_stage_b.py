import torch

from experiments.conditioned_vcg.E221_e218_multiseed_stage_b_280k_v1 import (
    protocol as p,
    run,
)
from projects.temporal_consequence_realization.method_conformance import (
    validate_protocol,
)


def test_e221_imports_both_finalized_e218_selections():
    contract = p.contract()
    frozen = {row["model_seed"]: row for row in contract["frozen_stage_a"]}
    assert frozen[1]["selected_episode"] == 280
    assert frozen[2]["selected_episode"] == 220
    assert frozen[1]["checkpoint_sha256"] == (
        "9b0f4950e49ad2480e4adcd365bc5f3985e3eaaa91b255e768a66f75ba0e3713"
    )
    assert frozen[2]["checkpoint_sha256"] == (
        "77e8ce5cd1959205559d427ace80e428b9be400a9fbd47ca3df8d9e21359a189"
    )


def test_e221_stage_b_procedure_and_schedule_match_e220():
    contract = p.contract()
    assert contract["source_stage_b_procedure"] == p.e220.STAGE_B_PROCEDURE
    assert p.training_schedule() == p.e220.training_schedule()
    assert len(contract["training_schedule"]) == 36
    assert contract["rounds"] == 4
    assert contract["episodes_per_round"] == 9
    assert p.head_seed(1) != p.head_seed(2)


def test_e221_contract_is_conforming_and_development_only():
    contract = p.contract()
    declaration = validate_protocol(contract, source=p.PROTOCOL)
    assert declaration["claim_level"] == "expressibility"
    assert contract["expected_complete_episode_runs"] == 72
    assert contract["test_panel_opened"] is False
    assert contract["paper_evidence"] is False


def test_e221_both_checkpoints_load_as_distinct_frozen_agents():
    contract = p.contract()
    digests = []
    for seed in p.MODEL_SEEDS:
        base, _search, _liveness, digest, frozen = run.load_base(
            contract, seed, torch.device("cpu")
        )
        with run._engine(contract, seed):
            head = run.e214.new_agent(base)
        assert frozen["selected_episode"] in (220, 280)
        assert head.config.lambda_center == 10.0
        assert head.config.lambda_max == 40.0
        digests.append(digest)
    assert len(set(digests)) == 2
