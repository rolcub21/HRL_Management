from pathlib import Path

import torch

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    representation as e191_representation,
)
from experiments.conditioned_vcg.E209_e191_method_10x10_longstay_268k_v1 import (
    protocol as p,
    representation,
    scenario,
)
from projects.temporal_consequence_realization.method_conformance import (
    validate_protocol,
)
from viability_graph_episodic_audit import EpisodicViabilityGraphConfig


def test_10x10_scale_match_arithmetic_and_geometry():
    assert scenario.SCENARIO_SPEC.storage_capacity == 63
    assert scenario.TOTAL_JOBS == 64
    assert scenario.MEAN_DURATION / scenario.CAPACITY == 20.0
    assert {
        level: scenario.nominal_occupancy(scenario.gap_for_level(level))
        for level in scenario.LEVELS
    } == {0.20: 0.20, 0.50: 0.50, 0.80: 0.80}


def test_validation_cases_are_paired_and_balanced():
    env = p.TargetDenseEnv(p.SCENARIO, p.SPEC)
    by_replicate = {}
    for seed in scenario.VALIDATION_SEEDS:
        instance = env.sample_episode_instance(seed)
        level = scenario.target_level(seed)
        gap = scenario.gap_for_level(level)
        assert instance.arrival_steps == tuple(
            gap * index for index in range(scenario.TOTAL_JOBS)
        )
        assert scenario.nominal_occupancy(gap) == level
        previous = by_replicate.setdefault(
            seed % 100, tuple(instance.storage_steps_needed)
        )
        assert tuple(instance.storage_steps_needed) == previous
    assert len(by_replicate) == 3


def test_architecture_is_exactly_e191_and_initialization_is_fresh():
    config = EpisodicViabilityGraphConfig(
        gamma=p.GAMMA, episode_horizon_steps=p.MAX_STEPS
    )
    candidate = representation.FreshScaleMatched10x10Agent(
        config=config, seed=3, device="cpu", epsilon=p.EPSILON_START
    )
    reference = e191_representation.FreshChoicePreservingQueueAgent(
        config=config, seed=3, device="cpu", epsilon=p.EPSILON_START
    )
    assert representation.architecture_signature(candidate) == (
        e191_representation.architecture_signature(reference)
    )
    assert len(candidate.replay) == 0
    assert candidate.decision_count == 0
    assert candidate.gradient_steps == 0
    assert not candidate.optimizer.state


def test_exploration_is_normalized_by_fixed_job_ratio():
    assert p.EPSILON_DECAY_DECISIONS == round(
        e191.EPSILON_DECAY_DECISIONS * 64 / 36
    )
    assert p.EPSILON_START == e191.EPSILON_START
    assert p.EPSILON_END == e191.EPSILON_END


def test_contract_is_current_conforming_and_development_only():
    contract = p.contract()
    declaration = validate_protocol(contract, source=p.PROTOCOL)
    assert declaration["claim_level"] == "expressibility"
    assert contract["initialization"]["kind"] == "fresh_random_from_model_seed"
    assert contract["architecture"]["exactly_identical_to_e191"]
    assert contract["initial_remote_target_episode"] == 120
    assert contract["paper_evidence"] is False
    assert all(Path(p.ROOT / relative).is_file() for relative in contract["source_sha256"])


def test_horizon_has_declared_light_day_margin():
    last_release = 100 * (scenario.TOTAL_JOBS - 1)
    bound = last_release + 2 * scenario.MEAN_DURATION + 100 * scenario.TOTAL_JOBS
    assert bound == 15_220
    assert bound <= scenario.MAX_STEPS
