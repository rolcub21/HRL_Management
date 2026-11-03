import pytest

from experiments.conditioned_vcg.E218_e215_long_exploration_replication_277k_v1 import (
    protocol as e218,
)
from experiments.conditioned_vcg.E226_e215_seed0_guard_free_training_285k_v1 import (
    representation,
)
from experiments.conditioned_vcg.E227_e218_seeds12_guard_free_training_286k_v1 import (
    protocol as p,
    run,
)
from projects.temporal_consequence_realization.method_conformance import (
    validate_protocol,
)


def test_e227_matches_e218_except_guard_intervention():
    assert p.MODEL_SEEDS == e218.MODEL_SEEDS == (1, 2)
    assert p.TOTAL_EPISODES == e218.TOTAL_EPISODES == 300
    assert p.VALIDATION_EVERY == e218.VALIDATION_EVERY == 20
    assert p.CHECKPOINT_EVERY == e218.CHECKPOINT_EVERY == 1
    assert p.TRAIN_BASE == e218.TRAIN_BASE
    assert p.VALIDATION_SEEDS == e218.VALIDATION_SEEDS
    assert p.EPSILON_DECAY_DECISIONS == e218.EPSILON_DECAY_DECISIONS == 57_300
    contract = p.contract()
    assert contract["defer_contract"] == {
        "identical_to_e218": True,
        "max_option_steps": 10,
        "max_consecutive_defers": 16,
    }
    assert contract["guard_free_contract"]["reconfigure_mask"] is False
    assert contract["guard_free_contract"]["witness_fallback"] is False
    assert contract["guard_free_contract"]["selector_bypass"] is False


def test_e227_reuses_e226_controller_without_a_behavior_fork():
    with run.implementation():
        assert (
            run.e191_run.representation.FreshChoicePreservingQueueAgent
            is representation.GuardFreeTrainingQueueAgent
        )
        assert run.e191_run.instrumentation is run.instrumentation


def test_e227_reference_checkpoint_identities_are_frozen():
    assert p.REFERENCE_CHECKPOINTS == {
        1: {
            "selected_episode": 280,
            "sha256": "9b0f4950e49ad2480e4adcd365bc5f3985e3eaaa91b255e768a66f75ba0e3713",
        },
        2: {
            "selected_episode": 220,
            "sha256": "77e8ce5cd1959205559d427ace80e428b9be400a9fbd47ca3df8d9e21359a189",
        },
    }
    for seed in p.MODEL_SEEDS:
        path = p.reference_checkpoint_path(seed)
        assert path.name == "best.pth"
        assert path.parent.name == f"seed-{seed}"


def test_e227_contract_conforms_and_each_seed_requires_its_smoke(tmp_path):
    validate_protocol(p.contract(), source=p.PROTOCOL)
    for seed in p.MODEL_SEEDS:
        with pytest.raises(run.E227Error, match="smoke"):
            run._smoke_gate(tmp_path, seed)


@pytest.mark.parametrize("seed", (1, 2))
def test_e227_training_enters_through_e191_adapter(tmp_path, monkeypatch, seed):
    run.prepare(tmp_path)
    run.e187_run._write(run._smoke_path(tmp_path, seed), {
        "contract_sha256": p.contract()["contract_sha256"],
        "model_seed": seed,
        "strict_complete": True,
    })
    observed = {}

    def e191_train(output, model_seed, through_episode, device):
        observed.update({
            "output": output,
            "seed": model_seed,
            "through_episode": through_episode,
            "device": device,
        })
        return {
            "status": "paused",
            "e191_protocol": "patched",
            "e191_contract_sha256": "patched",
        }

    monkeypatch.setattr(run.e191_run, "train_seed", e191_train)
    result = run.train_seed(tmp_path, seed, 60, "cuda")
    assert observed == {
        "output": tmp_path,
        "seed": seed,
        "through_episode": 60,
        "device": "cuda",
    }
    assert result["status"] == "paused"
    assert result["model_seed"] == seed
    assert result["guard_intervention_removed"] is True
