"""Frozen E28-style workloads and split identities for target-size learning.

This module lives in a staging bundle until copied into HRL_Management. The
HRL_PROJECT_ROOT override is for read-only integration checks from /tmp only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("HRL_PROJECT_ROOT", Path(__file__).resolve().parents[3])).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.small_rooms_env import SmallRoomsEnv
from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import run as e28
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
)

PROTOCOL = "vcg_target_size_training_empty_start_development_v1"
SCENARIOS = {
    name: e28.SCENARIO_BY_ID[name]
    for name in ("empty_8x8_n36", "empty_10x10_n64")
}
MODEL_SEEDS = (0, 1)
PREFERENCES = (0.0, 0.05, 0.10, 0.20)
OPERATIONAL_EPISODES = 500
VALIDATION_EVERY = 50
HANDLING_ROUNDS = 8
HANDLING_EPISODES_PER_ROUND = 50
HANDLING_LATE_UPDATE_FRACTION = 0.25
TRAIN_BASE = {
    "empty_8x8_n36": 112_800_000,
    "empty_10x10_n64": 112_100_000,
}
HANDLING_BASE = {
    "empty_8x8_n36": 114_800_000,
    "empty_10x10_n64": 114_100_000,
}
VALIDATION_SEEDS = {
    "empty_8x8_n36": tuple(range(112_800, 112_805)),
    "empty_10x10_n64": tuple(range(112_100, 112_105)),
}
PILOT_SEEDS = {
    "empty_8x8_n36": (112_820,),
    "empty_10x10_n64": (112_120,),
}
TEST_SEEDS = {
    "empty_8x8_n36": tuple(range(112_850, 112_853)),
    "empty_10x10_n64": tuple(range(112_150, 112_153)),
}
DEFAULT_OUTPUT = ROOT / "results/vcg-target-size-training-development-v1"


def digest(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenario_for(name: str):
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(f"unknown scenario: {name}") from error


def instance_for(name: str, seed: int):
    """Reuse the exact E28 geometry, Poisson durations, and 10-step releases."""
    scenario = scenario_for(name)
    instance = e28._instance(scenario, int(seed))
    expected = tuple(occupancy.EXTRA_ARRIVAL_GAP * i for i in range(scenario.total_jobs))
    if scenario.initial_occupied_slots != 0 or instance.arrival_steps != expected:
        raise RuntimeError("E28 generator no longer gives the declared empty-start workload")
    instance.validate_for(scenario.make_base_env())
    return instance


class TargetSizeEnv(SmallRoomsEnv):
    """Training environment with E28's geometry and instance sampler."""

    def __init__(self, name: str):
        scenario = scenario_for(name)
        super().__init__(
            grid_rows=scenario.rows,
            grid_cols=scenario.cols,
            start_state=(1, 1),
            door_cell=scenario.door,
            pickup_cells=(1, scenario.cols - 2),
            exit_cells=list(scenario.exits),
            number_blocks=scenario.total_jobs,
            choose_storage=False,
            arrival_rate=1.0 / occupancy.EXTRA_ARRIVAL_GAP,
            proc_mean=occupancy.PROC_MEAN,
        )
        self.target_scenario_name = name

    def sample_episode_instance(self, seed=None):
        if seed is None:
            raise ValueError("target-size training requires an explicit instance seed")
        instance = instance_for(self.target_scenario_name, int(seed))
        instance.validate_for(self)
        return instance


def training_seed(name: str, model_seed: int) -> int:
    if model_seed not in MODEL_SEEDS:
        raise ValueError("unsupported model seed")
    return TRAIN_BASE[name] + model_seed * 1_000_000


def handling_seed(name: str, model_seed: int, round_index: int, offset: int) -> int:
    if model_seed not in MODEL_SEEDS or not 0 <= round_index < HANDLING_ROUNDS:
        raise ValueError("unsupported handling coordinate")
    if not 0 <= offset < HANDLING_EPISODES_PER_ROUND:
        raise ValueError("unsupported handling episode offset")
    return HANDLING_BASE[name] + model_seed * 1_000_000 + round_index * 10_000 + offset


def check_splits() -> None:
    occupied: set[int] = set()
    for name in SCENARIOS:
        groups = [
            set(VALIDATION_SEEDS[name]), set(PILOT_SEEDS[name]),
            set(TEST_SEEDS[name]),
        ]
        for model_seed in MODEL_SEEDS:
            base = training_seed(name, model_seed)
            groups.append(set(range(base, base + OPERATIONAL_EPISODES)))
            groups.append({
                handling_seed(name, model_seed, r, i)
                for r in range(HANDLING_ROUNDS)
                for i in range(HANDLING_EPISODES_PER_ROUND)
            })
        for group in groups:
            if occupied.intersection(group):
                raise RuntimeError("target-size instance seed splits overlap")
            occupied.update(group)


def contract_payload() -> dict:
    check_splits()
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("run.py").resolve(),
        Path(__file__).with_name("evaluate.py").resolve(),
        Path(__file__).with_name("run.sh").resolve(),
        ROOT / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.py",
        ROOT / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/adapter.py",
        ROOT / "experiments/conditioned_vcg/E25_scale_baseline_transfer_95k/run.py",
        ROOT / "experiments/conditioned_vcg/E25_v4_reconfiguration_baselines_95k/controller.py",
        ROOT / "experiments/conditioned_vcg/E27_certified_dynamic_sustained_99k/run.py",
        ROOT / "experiments/conditioned_vcg/E14_certification_scalability_95k/reuse.py",
        ROOT / "experiments/conditioned_vcg/development/D10_scalability_support_screen/occupancy_extension.py",
        ROOT / "benchmark_viability_critic_priority.py",
        ROOT / "PSLAP/viability_candidates.py",
        ROOT / "PSLAP/viability_filter.py",
        ROOT / "PSLAP/relocation_family_certification.py",
        ROOT / "example/small_rooms_env.py",
        ROOT / "example/urgency_scheduler.py",
        ROOT / "train_viability_graph_smdp_proper.py",
        ROOT / "train_viability_graph_smdp.py",
        ROOT / "train_vcg_preference_conditioned.py",
        ROOT / "vcg_v11_conditioned_handling.py",
        ROOT / "viability_graph_hierarchy.py",
        ROOT / "run_vcg_v11_conditioned_handling_seed0_85k.py",
        ROOT / "run_vcg_v11_nested_handling_pilot.py",
    ]
    return {
        "protocol": PROTOCOL,
        "scope": "development first; held-out panel sealed until both target checkpoints exist",
        "scenarios": {name: item.public_dict() for name, item in SCENARIOS.items()},
        "model_seeds": list(MODEL_SEEDS),
        "preferences": list(PREFERENCES),
        "operational_episodes_per_model": OPERATIONAL_EPISODES,
        "operational_validation_every": VALIDATION_EVERY,
        "handling_rounds": HANDLING_ROUNDS,
        "handling_episodes_per_round": HANDLING_EPISODES_PER_ROUND,
        "handling_update_fraction_rounds_1_to_4": 1.0,
        "handling_update_fraction_rounds_5_to_8": HANDLING_LATE_UPDATE_FRACTION,
        "validation_seeds": {k: list(v) for k, v in VALIDATION_SEEDS.items()},
        "pilot_seeds": {k: list(v) for k, v in PILOT_SEEDS.items()},
        "test_seeds": {k: list(v) for k, v in TEST_SEEDS.items()},
        "train_seed_base": TRAIN_BASE,
        "handling_seed_base": HANDLING_BASE,
        "instance_rule": "E28._instance: zero initial storage; arrivals 10*j; Poisson(80) durations",
        "comparison_caveat": "E28 A3C controls are frozen 5x5-trained models, not target-size retraining",
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }


def contract() -> dict:
    payload = contract_payload()
    return {**payload, "contract_sha256": digest(payload)}
