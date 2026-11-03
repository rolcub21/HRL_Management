"""Locked E218-matched seeds-1/2 guard-free training protocol."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)
from experiments.conditioned_vcg.E218_e215_long_exploration_replication_277k_v1 import (
    protocol as e218,
)
from experiments.conditioned_vcg.E226_e215_seed0_guard_free_training_285k_v1 import (
    protocol as e226,
    representation as guard_free_representation,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e227_e218_seeds12_guard_free_training_286k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e227_e218_seeds12_guard_free_training"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
SCENARIO = e218.SCENARIO
SCENARIOS = e218.SCENARIOS
MODEL_SEEDS = (1, 2)
TOTAL_EPISODES = e218.TOTAL_EPISODES
EPISODES_BY_SCENARIO = {SCENARIO: TOTAL_EPISODES}
VALIDATION_EVERY = e218.VALIDATION_EVERY
CHECKPOINT_EVERY = e218.CHECKPOINT_EVERY
VALIDATION_SEEDS = e218.VALIDATION_SEEDS
TRAIN_BASE = e218.TRAIN_BASE
TRAINING_JOB_BUDGET = TOTAL_EPISODES * e191.workload.TOTAL_JOBS
MAX_STEPS = e218.MAX_STEPS
GAMMA = e218.GAMMA
EPSILON_START = e218.EPSILON_START
EPSILON_END = e218.EPSILON_END
EPSILON_WARMUP_DECISIONS = e218.EPSILON_WARMUP_DECISIONS
EPSILON_DECAY_DECISIONS = e218.EPSILON_DECAY_DECISIONS
EPSILON_TARGET_FLOOR_EPISODE = e218.EPSILON_TARGET_FLOOR_EPISODE
REPORT_CHECKPOINTS = tuple(range(20, TOTAL_EPISODES + 1, 20))
EARLY_READ_CHECKPOINTS = (20, 40, 60)
LATE_CHECKPOINTS = REPORT_CHECKPOINTS
TD_BACKUP_IMPLEMENTATION = e218.TD_BACKUP_IMPLEMENTATION
SPEC = e218.SPEC
TargetDenseEnv = e218.TargetDenseEnv
scenario_for = e218.scenario_for
workload = e218.workload
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e227-e218-seeds12-guard-free-training-286k-v1"
)
REFERENCE_OUTPUT = e218.DEFAULT_OUTPUT
MAX_ALLOWED_NONPROGRESS_REHANDLE_RUN = e226.MAX_ALLOWED_NONPROGRESS_REHANDLE_RUN
MAX_J_REGRESSION_FRACTION = e226.MAX_J_REGRESSION_FRACTION
MAX_REHANDLE_PER_100_REGRESSION = e226.MAX_REHANDLE_PER_100_REGRESSION

REFERENCE_CHECKPOINTS = {
    1: {
        "selected_episode": 280,
        "sha256": "9b0f4950e49ad2480e4adcd365bc5f3985e3eaaa91b255e768a66f75ba0e3713",
    },
    2: {
        "selected_episode": 220,
        "sha256": "77e8ce5cd1959205559d427ace80e428b9be400a9fbd47ca3df8d9e21359a189",
    },
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def method_revision() -> str:
    match = re.search(
        r"^\*\*Revision:\*\* `([^`]+)`",
        (ROOT / METHOD_REFERENCE).read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError("E227 cannot parse the current method revision")
    return match.group(1)


def method_declaration() -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E227 is a development-only historical conditioned-VCG guard "
                "ablation, not a canonical law-condition claim."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    for identifier, evidence in {
        "reproducibility.method_revision": (
            "E227 pins the current method-reference revision at preparation."
        ),
        "reproducibility.content_hashes": (
            "The protocol, implementation, inherited E226 controller, E218 "
            "coordinates, and selected smoke checkpoints are hashed."
        ),
        "reproducibility.oracle_declarations": (
            "No oracle input is used; guard removal is declared explicitly."
        ),
        "reproducibility.failed_gates_retained": (
            "Incomplete episodes, cap hits, validation rows, and failed gates "
            "remain in each seed's training history."
        ),
    }.items():
        audit[identifier] = {"status": "pass", "evidence": evidence}
    audit["reproducibility.preprotected_integration_gate"] = {
        "status": "not_applicable",
        "evidence": (
            "Only the already-open E191 development coordinates are used. Each "
            "seed must pass a complete local smoke episode before training."
        ),
    }
    value = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": method_revision(),
        "claim_level": "expressibility",
        "declared_deviations": [
            "E227 removes all E188/E218 guard intervention while retaining the "
            "four passive guard-context values as critic inputs."
        ],
        "oracle_inputs": [],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(value, source=PROTOCOL)
    return value


def reference_checkpoint_path(seed: int) -> Path:
    if int(seed) not in MODEL_SEEDS:
        raise ValueError("E227 reference seed must be 1 or 2")
    return (
        REFERENCE_OUTPUT / "training/operational" / SCENARIO
        / f"seed-{int(seed)}/best.pth"
    )


def _source_paths() -> tuple[Path, ...]:
    directory = Path(__file__).resolve().parent
    return (
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
        ROOT / "experiments/conditioned_vcg/E218_e215_long_exploration_replication_277k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E218_e215_long_exploration_replication_277k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E226_e215_seed0_guard_free_training_285k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E226_e215_seed0_guard_free_training_285k_v1/instrumentation.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/scenario.py",
        ROOT / "experiments/conditioned_vcg/E100_accelerated_release_clock_163k_v1/fast_inputs.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/acceleration.py",
        ROOT / "run_vcg_objective_gamma_audit.py",
        ROOT / "train_vcg_dense_proper.py",
        ROOT / "viability_graph_hierarchy.py",
    )


def contract() -> dict:
    parent = e191.contract()
    frontier = deepcopy(parent["frontier"])
    frontier.update({
        "reconfigure_mask_after_nonprogress_bound": False,
        "one_step_witness_fallback": False,
        "critic_bypass": False,
        "selection_frontier": "complete exact-SAFE frontier",
        "bellman_frontier": "complete exact-SAFE frontier",
    })
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        **deepcopy(method_declaration()),
        "scope": (
            "fresh seeds-1/2 E218-matched guard-free training replication on "
            "the already-open development workload family"
        ),
        "development_only": True,
        "test_panel_opened": False,
        "paper_evidence": False,
        "reference_protocol": e218.PROTOCOL,
        "guard_free_controller_protocol": e226.PROTOCOL,
        "single_behavior_delta": (
            "remove Reconfigure masking, retained-witness fallback, and all "
            "selector bypass; critic and Bellman target use every certified action"
        ),
        "objective": deepcopy(parent["objective"]),
        "audit_scaffold_objective": deepcopy(parent["audit_scaffold_objective"]),
        "gamma": GAMMA,
        "frontier": frontier,
        "acceleration": deepcopy(parent["acceleration"]),
        "scenario": SCENARIO,
        "model_seeds": list(MODEL_SEEDS),
        "episodes_per_seed": TOTAL_EPISODES,
        "train_instance_seed_base": TRAIN_BASE[SCENARIO],
        "validation_every": VALIDATION_EVERY,
        "validation_seeds": list(VALIDATION_SEEDS[SCENARIO]),
        "checkpoint_every": CHECKPOINT_EVERY,
        "checkpoint_note": (
            "validation/selection every 20 episodes; resumable checkpoint every "
            "episode, exactly as E218"
        ),
        "report_checkpoints": list(REPORT_CHECKPOINTS),
        "early_read_checkpoints": list(EARLY_READ_CHECKPOINTS),
        "finite_horizon": {
            "steps": MAX_STEPS,
            "role": "error cap only; a hit is a failed episode",
        },
        "epsilon": {
            "start": EPSILON_START,
            "end": EPSILON_END,
            "warmup_decisions": EPSILON_WARMUP_DECISIONS,
            "decay_decisions": EPSILON_DECAY_DECISIONS,
            "target_floor_episode_approximate": EPSILON_TARGET_FLOOR_EPISODE,
            "identical_to_e218": True,
        },
        "architecture": {
            "controller_architecture": (
                guard_free_representation.CONTROLLER_ARCHITECTURE
            ),
            "network_architecture": guard_free_representation.NETWORK_ARCHITECTURE,
            "queue_context_names": list(
                guard_free_representation.QUEUE_CONTEXT_NAMES
            ),
            "queue_context_dim": guard_free_representation.QUEUE_CONTEXT_DIM,
            "neural_inputs_identical_to_e218": True,
            "passive_reconfigure_streak_context_retained": True,
        },
        "initialization": {
            "kind": "fresh_random_from_model_seed",
            "model_seeds": list(MODEL_SEEDS),
            "same_training_days_as_e218": True,
            "optimizer": "fresh",
            "replay": "empty",
            "counters": "zero",
        },
        "warm_start": {
            "enabled": False,
            "kind": "fresh_random_from_model_seed",
            "transferred_modules": [],
            "source_checkpoint": None,
            "optimizer": "fresh",
            "replay": "empty",
            "counters": "zero",
            "guard_memory": "passive and empty",
        },
        "defer_contract": {
            "identical_to_e218": True,
            "max_option_steps": 10,
            "max_consecutive_defers": 16,
        },
        "guard_free_contract": deepcopy(e226.contract()["guard_free_contract"]),
        "telemetry": deepcopy(e226.contract()["telemetry"]),
        "decision_rule": deepcopy(e226.contract()["decision_rule"]),
        "accelerations_retained": deepcopy(e226.contract()["accelerations_retained"]),
        "reference_smoke_checkpoints": {
            str(seed): {
                **deepcopy(REFERENCE_CHECKPOINTS[seed]),
                "path": str(reference_checkpoint_path(seed).relative_to(ROOT)),
                "role": "execution-path smoke only; never a training warm start",
            }
            for seed in MODEL_SEEDS
        },
        "smoke_gate": (
            "prepare, self-check, and one complete local smoke episode per model "
            "seed must pass before that seed's official training command"
        ),
        "checkpoint_selection": (
            "same fixed nine-case development-panel average and every-20-episode "
            "selection opportunities as E218; sealed test panel remains closed"
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()
        },
    }
    return {**payload, "contract_sha256": digest(payload)}


def trainer_protocol() -> SimpleNamespace:
    value = contract()
    return SimpleNamespace(
        PROTOCOL=PROTOCOL,
        METHOD_VERSION=METHOD_VERSION,
        SCENARIO=SCENARIO,
        SCENARIOS=SCENARIOS,
        MODEL_SEEDS=MODEL_SEEDS,
        EPISODES_BY_SCENARIO=EPISODES_BY_SCENARIO,
        TOTAL_EPISODES=TOTAL_EPISODES,
        VALIDATION_EVERY=VALIDATION_EVERY,
        CHECKPOINT_EVERY=CHECKPOINT_EVERY,
        VALIDATION_SEEDS=VALIDATION_SEEDS,
        TRAIN_BASE=TRAIN_BASE,
        TRAINING_JOB_BUDGET=TRAINING_JOB_BUDGET,
        TD_BACKUP_IMPLEMENTATION=TD_BACKUP_IMPLEMENTATION,
        LATE_CHECKPOINTS=LATE_CHECKPOINTS,
        DEFAULT_OUTPUT=DEFAULT_OUTPUT,
        TargetDenseEnv=TargetDenseEnv,
        scenario_for=scenario_for,
        contract=lambda: value,
    )


__all__ = [
    "CHECKPOINT_EVERY", "DEFAULT_OUTPUT", "EARLY_READ_CHECKPOINTS",
    "EPISODES_BY_SCENARIO", "EPSILON_DECAY_DECISIONS", "EPSILON_END",
    "EPSILON_START", "EPSILON_TARGET_FLOOR_EPISODE",
    "EPSILON_WARMUP_DECISIONS", "GAMMA", "LATE_CHECKPOINTS", "MAX_STEPS",
    "METHOD_VERSION", "MODEL_SEEDS", "PROTOCOL", "REFERENCE_CHECKPOINTS",
    "REFERENCE_OUTPUT", "REPORT_CHECKPOINTS", "SCENARIO", "SCENARIOS",
    "SPEC", "TD_BACKUP_IMPLEMENTATION", "TOTAL_EPISODES", "TRAIN_BASE",
    "TRAINING_JOB_BUDGET", "TargetDenseEnv", "VALIDATION_EVERY",
    "VALIDATION_SEEDS", "contract", "digest", "method_declaration",
    "method_revision", "reference_checkpoint_path", "scenario_for", "sha256",
    "trainer_protocol",
]

