"""Frozen identities and dense target-size environments for Stage A v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(
    os.environ.get("HRL_PROJECT_ROOT", Path(__file__).resolve().parents[3])
).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conditioned_vcg.target_size_vcg_training import protocol as v1
from vcg_objective_audit import ObjectiveAuditSmallRoomsEnv, TimingObjectiveSpec


PROTOCOL = "vcg_target_size_dense_stage_a_development_v2"
METHOD_VERSION = "vcg_dense_v1_1_target_size_stage_a_v2"
SCENARIOS = v1.SCENARIOS
MODEL_SEEDS = (0, 1)
TOTAL_EPISODES = 500
VALIDATION_EVERY = 50
CHECKPOINT_EVERY = 1
VALIDATION_SEEDS = v1.VALIDATION_SEEDS
TRAIN_BASE = v1.TRAIN_BASE
DEFAULT_OUTPUT = ROOT / "results/vcg-target-size-dense-stage-a-development-v2"


def scenario_for(name: str):
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(f"unknown target-size scenario: {name}") from error


def instance_for(name: str, seed: int):
    return v1.instance_for(name, int(seed))


class TargetDenseEnv(ObjectiveAuditSmallRoomsEnv):
    """VCG-Dense objective with the exact E28 empty-start workload."""

    def __init__(self, name: str, objective: TimingObjectiveSpec):
        scenario = scenario_for(name)
        super().__init__(
            timing_objective=objective,
            grid_rows=scenario.rows,
            grid_cols=scenario.cols,
            start_state=(1, 1),
            door_cell=scenario.door,
            pickup_cells=(1, scenario.cols - 2),
            exit_cells=list(scenario.exits),
            number_blocks=scenario.total_jobs,
            choose_storage=False,
            arrival_rate=0.1,
            proc_mean=80,
        )
        self.target_scenario_name = name

    def sample_episode_instance(self, seed=None):
        if seed is None:
            raise ValueError("target-size training requires an explicit instance seed")
        instance = instance_for(self.target_scenario_name, int(seed))
        instance.validate_for(self)
        return instance


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def contract_payload() -> dict:
    sources = (
        Path(__file__).resolve(),
        Path(__file__).with_name("run.py").resolve(),
        Path(__file__).with_name("run.sh").resolve(),
        ROOT / "experiments/conditioned_vcg/target_size_vcg_training/protocol.py",
        ROOT / "experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.py",
        ROOT / "experiments/conditioned_vcg/E14_certification_scalability_95k/reuse.py",
        ROOT / "train_vcg_dense_proper.py",
        ROOT / "run_vcg_objective_gamma_audit.py",
        ROOT / "vcg_objective_audit.py",
        ROOT / "viability_graph_episodic_audit.py",
        ROOT / "PSLAP/relocation_family_certification.py",
    )
    return {
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": "target-size Stage A development; no test panel opened",
        "scenarios": {name: value.public_dict() for name, value in SCENARIOS.items()},
        "model_seeds": list(MODEL_SEEDS),
        "total_episodes_per_model": TOTAL_EPISODES,
        "validation_every": VALIDATION_EVERY,
        "validation_seeds": {name: list(values) for name, values in VALIDATION_SEEDS.items()},
        "checkpoint_every": CHECKPOINT_EVERY,
        "objective": "40-1.5|e|-0.5(|e|-20)+",
        "gamma": 0.99,
        "epsilon": {"start": 0.90, "end": 0.05, "decisions": 3000},
        "agent_family": "episodic_viability_graph_hierarchy",
        "instance_rule": "E28._instance: empty start, arrivals at 10*j, Poisson(80) durations",
        "acceleration": "D12 relocation family + E14 timing cache/path cleanup",
        "source_sha256": {str(path): _sha256(path) for path in sources},
    }


def contract() -> dict:
    payload = contract_payload()
    return {**payload, "contract_sha256": _digest(payload)}
