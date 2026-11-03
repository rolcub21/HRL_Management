import pytest

from experiments.conditioned_vcg.E215_e191_long_exploration_drift_pilot_274k_v1 import (
    protocol as e215,
)
from experiments.conditioned_vcg.E226_e215_seed0_guard_free_training_285k_v1 import (
    instrumentation,
    protocol as p,
    representation,
    run,
)
from projects.temporal_consequence_realization.method_conformance import (
    validate_protocol,
)


def test_e226_matches_e215_schedule_and_changes_only_guard_behavior():
    assert p.MODEL_SEEDS == (0,)
    assert p.TOTAL_EPISODES == e215.TOTAL_EPISODES == 300
    assert p.VALIDATION_EVERY == e215.VALIDATION_EVERY == 20
    assert p.CHECKPOINT_EVERY == e215.CHECKPOINT_EVERY == 1
    assert p.TRAIN_BASE == e215.TRAIN_BASE
    assert p.VALIDATION_SEEDS == e215.VALIDATION_SEEDS
    assert p.EPSILON_DECAY_DECISIONS == e215.EPSILON_DECAY_DECISIONS == 57_300
    contract = p.contract()
    assert contract["defer_contract"]["max_option_steps"] == 10
    assert contract["defer_contract"]["max_consecutive_defers"] == 16
    assert contract["guard_free_contract"] == {
        "reconfigure_mask": False,
        "witness_fallback": False,
        "selector_bypass": False,
        "critic_scores_every_certified_action": True,
        "td_target_uses_every_certified_next_action": True,
        "passive_guard_context_input_retained": True,
    }


def test_e226_metadata_authenticates_no_guard_intervention():
    from viability_graph_episodic_audit import EpisodicViabilityGraphConfig

    agent = representation.GuardFreeTrainingQueueAgent(
        config=EpisodicViabilityGraphConfig(
            gamma=p.GAMMA, episode_horizon_steps=p.MAX_STEPS
        ),
        seed=0,
        device="cpu",
        epsilon=p.EPSILON_START,
    )
    metadata = agent.checkpoint_metadata()
    assert metadata["e226_full_certified_frontier_in_selection"] is True
    assert metadata["e226_full_certified_frontier_in_td_targets"] is True
    assert metadata["e226_guard_intervention_enabled"] is False
    assert metadata["e226_reconfigure_mask_enabled"] is False
    assert metadata["e226_witness_fallback_enabled"] is False
    assert metadata["e226_critic_bypass_enabled"] is False


def test_e226_episode_telemetry_keeps_defer_inside_nonprogress_run():
    records = (
        {
            "selected_action_type": "reconfigure",
            "safe_reconfigure_candidates": 2,
            "safe_nonreconfigure_candidates": 0,
        },
        {
            "selected_action_type": "defer",
            "safe_reconfigure_candidates": 1,
            "safe_nonreconfigure_candidates": 1,
        },
        {
            "selected_action_type": "reconfigure",
            "safe_reconfigure_candidates": 1,
            "safe_nonreconfigure_candidates": 1,
        },
        {
            "selected_action_type": "deliver",
            "safe_reconfigure_candidates": 0,
            "safe_nonreconfigure_candidates": 1,
        },
    )
    telemetry = instrumentation._episode_telemetry(
        records, {"horizon_cap_hit": False, "audit": {}}
    )
    assert telemetry["longest_reconfigure_run_without_accept_or_deliver"] == 2
    assert telemetry["reconfigure_only_safe_frontiers"] == 1
    assert telemetry["step_limit_hit"] is False


def test_e226_contract_conforms_and_training_requires_smoke(tmp_path):
    validate_protocol(p.contract(), source=p.PROTOCOL)
    with pytest.raises(run.E226Error, match="smoke"):
        run._smoke_gate(tmp_path)


def test_e226_training_enters_through_e191_adapter(tmp_path, monkeypatch):
    run.prepare(tmp_path)
    run.e187_run._write(run._smoke_path(tmp_path), {
        "contract_sha256": p.contract()["contract_sha256"],
        "strict_complete": True,
    })
    observed = {}

    def e191_train(output, seed, through_episode, device):
        observed.update({
            "output": output,
            "seed": seed,
            "through_episode": through_episode,
            "device": device,
        })
        return {
            "status": "paused",
            "e191_protocol": "patched",
            "e191_contract_sha256": "patched",
        }

    def wrong_entry(*_args, **_kwargs):
        raise AssertionError("E226 bypassed E191 and entered E187 directly")

    monkeypatch.setattr(run.e191_run, "train_seed", e191_train)
    monkeypatch.setattr(run.e187_run, "train_seed", wrong_entry)
    result = run.train_seed(tmp_path, 60, "cuda")
    assert observed == {
        "output": tmp_path,
        "seed": 0,
        "through_episode": 60,
        "device": "cuda",
    }
    assert result["status"] == "paused"
    assert result["guard_intervention_removed"] is True
