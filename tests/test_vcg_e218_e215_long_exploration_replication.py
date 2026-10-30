from projects.temporal_consequence_realization.method_conformance import (
    validate_protocol,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    run as e191_run,
    stage as e191_stage,
)
from experiments.conditioned_vcg.E218_e215_long_exploration_replication_277k_v1 import (
    protocol as p,
    run,
)


def test_e218_is_the_matched_seed_one_two_replication():
    contract = p.contract()
    assert p.MODEL_SEEDS == (1, 2)
    assert p.TOTAL_EPISODES == 300
    assert p.EPSILON_DECAY_DECISIONS == 57_300
    assert contract["replication"]["fresh_network_per_seed"] is True
    assert contract["replication"]["architecture_unchanged"] is True
    assert contract["replication"]["objective_unchanged"] is True
    assert contract["replication"]["certification_unchanged"] is True
    assert contract["replication"]["replay_capacity"] == 20_000
    validate_protocol(contract, source=p.PROTOCOL)


def test_e218_installs_both_e191_runtime_seams():
    old_run = e191_run.p
    old_stage = e191_stage.p
    with run.implementation():
        assert e191_run.p is p
        assert e191_stage.p is p
    assert e191_run.p is old_run
    assert e191_stage.p is old_stage


def test_e218_seed_outputs_are_disjoint(tmp_path):
    root = tmp_path / "training/operational" / p.SCENARIO
    first = root / "seed-1"
    second = root / "seed-2"
    assert first != second
    assert first.name == "seed-1"
    assert second.name == "seed-2"
