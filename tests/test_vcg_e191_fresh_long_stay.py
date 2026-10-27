from __future__ import annotations

import json

import torch

from viability_graph_episodic_audit import EpisodicViabilityGraphConfig
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    representation as e187_representation,
)
from experiments.conditioned_vcg.E188_mask_only_progress_guard_247k_v1 import (
    representation as e188_representation,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol,
    representation,
    run,
    scenario,
    stage,
)


def test_generator_levels_are_exact_deterministic_and_paired():
    env = protocol.TargetDenseEnv(protocol.SCENARIO, protocol.SPEC)
    cases = {}
    for seed in scenario.VALIDATION_SEEDS:
        first = env.sample_episode_instance(seed)
        second = env.sample_episode_instance(seed)
        assert first == second
        meta = scenario.workload_metadata(first)
        level = meta["target_nominal_occupancy"]
        gap = scenario.gap_for_level(level)
        assert first.arrival_steps == tuple(
            gap * index for index in range(scenario.TOTAL_JOBS)
        )
        assert meta["nominal_occupancy_from_declared_mean"] == level
        cases[(level, seed % 100)] = first

    for replicate in range(scenario.VALIDATION_CASES_PER_LEVEL):
        durations = {
            cases[(level, replicate)].storage_steps_needed
            for level in scenario.LEVELS
        }
        assert len(durations) == 1
        arrivals = {
            cases[(level, replicate)].arrival_steps for level in scenario.LEVELS
        }
        assert len(arrivals) == len(scenario.LEVELS)


def test_training_seed_stratification_is_balanced():
    levels = [
        scenario.target_level(scenario.TRAIN_BASE + index)
        for index in range(protocol.TOTAL_EPISODES)
    ]
    assert {level: levels.count(level) for level in scenario.LEVELS} == {
        level: 100 for level in scenario.LEVELS
    }


def test_fresh_agent_has_exact_e189_architecture_and_roundtrips():
    config = EpisodicViabilityGraphConfig(
        gamma=protocol.GAMMA,
        episode_horizon_steps=protocol.MAX_STEPS,
    )
    fresh = representation.FreshChoicePreservingQueueAgent(
        config=config, seed=0, device="cpu", epsilon=protocol.EPSILON_START
    )
    with e187_representation.warm_start_disabled():
        parent = e188_representation.ChoicePreservingQueueAgent(
            config=config, seed=0, device="cpu", epsilon=protocol.EPSILON_START
        )
    assert (
        representation.NETWORK_ARCHITECTURE
        == e188_representation.NETWORK_ARCHITECTURE
    )
    assert (
        representation.QUEUE_CONTEXT_NAMES
        == e188_representation.QUEUE_CONTEXT_NAMES
    )
    assert representation.architecture_signature(fresh) == (
        representation.architecture_signature(parent)
    )
    assert fresh.e187_warm_start_manifest is None
    assert not fresh.optimizer.state
    assert len(fresh.replay) == 0

    payload = fresh.checkpoint(
        include_replay=True,
        training_protocol=protocol.PROTOCOL,
        future_schedule_visible_to_policy=False,
    )
    restored = representation.FreshChoicePreservingQueueAgent.from_checkpoint(
        payload, device="cpu", resumable=True, seed=0
    )
    for name, value in fresh.Q_local.state_dict().items():
        assert torch.equal(value, restored.Q_local.state_dict()[name])
    assert restored.e187_warm_start_manifest is None


def test_contract_states_scenario_only_inputs_and_all_accelerations():
    value = protocol.contract()
    assert value["architecture"]["exactly_identical_to_e189"] is True
    assert value["architecture"]["arrival_history_features_added"] is False
    assert value["architecture"]["queue_context_dim"] == 8
    assert value["initialization"]["source_checkpoint"] is None
    assert value["workload"]["occupancy_is_initial_yard_fill"] is False
    assert all(value["acceleration_assertion"].values())
    assert value["defer_contract"]["unchanged_from_e189"] is True
    assert value["finite_horizon"]["steps"] == 9000


def test_self_check_is_non_episodic_and_idempotent(tmp_path):
    first = run.self_check(tmp_path)
    second = run.self_check(tmp_path)
    assert first["status"] == second["status"] == "passed"
    assert first["contract_sha256"] == second["contract_sha256"]
    assert first["episode_runs"] == second["episode_runs"] == 0
    observed = json.loads((tmp_path / "contract.json").read_text())
    assert observed == protocol.contract()


def test_training_adapter_uses_truthful_longstay_constants(tmp_path, monkeypatch):
    observed = {}

    def fake_main(argv):
        observed.update({
            "argv": tuple(argv),
            "episodes": stage.dense.FROZEN_TOTAL_EPISODES,
            "max_steps": stage.dense.FROZEN_MAX_STEPS,
            "validation": tuple(stage.dense.FROZEN_VALIDATION_SEEDS),
            "defaults": dict(stage.dense.FROZEN_AUDIT_DEFAULTS),
        })
        return {
            "status": "partial",
            "completed_training_episodes": 0,
            "latest_checkpoint": None,
            "best_checkpoint": None,
            "deployment_checkpoint_eligible": False,
        }

    monkeypatch.setattr(stage.dense, "main", fake_main)
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: True)
    result = run.train_seed(tmp_path, 0, 3, "cuda")
    assert result["fresh_random_initialization"] is True
    assert observed["episodes"] == 300
    assert observed["max_steps"] == 9000
    assert len(observed["validation"]) == 9
    assert observed["defaults"]["proc_mean"] == 700
    assert observed["defaults"]["number_blocks"] == 36
    assert observed["defaults"]["arrival_rate"] == 1.0 / 40.0
