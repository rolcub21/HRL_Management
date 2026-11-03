"""Locked E215-matched seed-0 protocol with guard intervention removed."""

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
from experiments.conditioned_vcg.E215_e191_long_exploration_drift_pilot_274k_v1 import (
    protocol as e215,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)

from . import representation


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e226_e215_seed0_guard_free_training_285k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e226_e215_seed0_guard_free_training"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
SCENARIO = e215.SCENARIO
SCENARIOS = e215.SCENARIOS
MODEL_SEEDS = (0,)
TOTAL_EPISODES = e215.TOTAL_EPISODES
EPISODES_BY_SCENARIO = {SCENARIO: TOTAL_EPISODES}
VALIDATION_EVERY = e215.VALIDATION_EVERY
# E215 evaluates every 20 episodes but saves a resumable checkpoint every episode.
CHECKPOINT_EVERY = e215.CHECKPOINT_EVERY
VALIDATION_SEEDS = e215.VALIDATION_SEEDS
TRAIN_BASE = e215.TRAIN_BASE
TRAINING_JOB_BUDGET = TOTAL_EPISODES * e191.workload.TOTAL_JOBS
MAX_STEPS = e215.MAX_STEPS
GAMMA = e215.GAMMA
EPSILON_START = e215.EPSILON_START
EPSILON_END = e215.EPSILON_END
EPSILON_WARMUP_DECISIONS = e215.EPSILON_WARMUP_DECISIONS
EPSILON_DECAY_DECISIONS = e215.EPSILON_DECAY_DECISIONS
EPSILON_TARGET_FLOOR_EPISODE = e215.EPSILON_TARGET_FLOOR_EPISODE
REPORT_CHECKPOINTS = tuple(range(20, TOTAL_EPISODES + 1, 20))
EARLY_READ_CHECKPOINTS = (20, 40, 60)
LATE_CHECKPOINTS = REPORT_CHECKPOINTS
TD_BACKUP_IMPLEMENTATION = e215.TD_BACKUP_IMPLEMENTATION
SPEC = e215.SPEC
TargetDenseEnv = e215.TargetDenseEnv
scenario_for = e215.scenario_for
e189 = e215.e189
workload = e215.workload
UNCHANGED_METHOD_FIELDS = e215.UNCHANGED_METHOD_FIELDS
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e226-e215-seed0-guard-free-training-285k-v1"
)
REFERENCE_OUTPUT = e215.DEFAULT_OUTPUT
MAX_ALLOWED_NONPROGRESS_REHANDLE_RUN = 10
MAX_J_REGRESSION_FRACTION = 0.10
MAX_REHANDLE_PER_100_REGRESSION = 10.0


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
        raise RuntimeError("E226 cannot parse the current method revision")
    return match.group(1)


def method_declaration() -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E226 is a development-only historical conditioned-VCG "
                "guard ablation, not a canonical law-condition claim."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    entries = {
        "reproducibility.method_revision": (
            "E226 pins the current method-reference revision at preparation."
        ),
        "reproducibility.content_hashes": (
            "The protocol, implementation, inherited behavioral sources, and "
            "matched E215 coordinates are content-hashed."
        ),
        "reproducibility.oracle_declarations": (
            "No oracle input is used; removal of guard intervention is explicit."
        ),
        "reproducibility.failed_gates_retained": (
            "All incomplete episodes, cap hits, validation rows, and fixed-gate "
            "failures remain in the training history."
        ),
    }
    for identifier, evidence in entries.items():
        audit[identifier] = {"status": "pass", "evidence": evidence}
    audit["reproducibility.preprotected_integration_gate"] = {
        "status": "not_applicable",
        "evidence": (
            "E226 uses only the already-open E191 development coordinates. A "
            "complete smoke episode is required before training is launched."
        ),
    }
    declaration = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": method_revision(),
        "claim_level": "expressibility",
        "declared_deviations": [
            "E226 removes all E188/E215 guard intervention while retaining its "
            "four observed context values as passive critic inputs."
        ],
        "oracle_inputs": [],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(declaration, source=PROTOCOL)
    return declaration


def _source_paths() -> tuple[Path, ...]:
    directory = Path(__file__).resolve().parent
    return (
        directory / "protocol.py",
        directory / "representation.py",
        directory / "instrumentation.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT / METHOD_REFERENCE,
        ROOT / "projects/temporal_consequence_realization/method_conformance.py",
        ROOT / "experiments/conditioned_vcg/E215_e191_long_exploration_drift_pilot_274k_v1/protocol.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/scenario.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E225_e215_seed0_guard_free_development_284k_v1/guard_free.py",
        ROOT / "experiments/conditioned_vcg/E188_mask_only_progress_guard_247k_v1/guard.py",
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
            "fresh seed-0 E215-matched guard-free training ablation on the "
            "already-open development workload family"
        ),
        "development_only": True,
        "test_panel_opened": False,
        "paper_evidence": False,
        "reference_protocol": e215.PROTOCOL,
        "reference_output": str(REFERENCE_OUTPUT.relative_to(ROOT)),
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
        "model_seeds": [0],
        "episodes_per_seed": TOTAL_EPISODES,
        "train_instance_seed_base": TRAIN_BASE[SCENARIO],
        "validation_every": VALIDATION_EVERY,
        "validation_seeds": list(VALIDATION_SEEDS[SCENARIO]),
        "checkpoint_every": CHECKPOINT_EVERY,
        "checkpoint_note": (
            "validation/selection every 20 episodes; resumable checkpoint every episode, "
            "exactly as E215"
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
            "identical_to_e215": True,
        },
        "architecture": {
            "controller_architecture": representation.CONTROLLER_ARCHITECTURE,
            "network_architecture": representation.NETWORK_ARCHITECTURE,
            "queue_context_names": list(representation.QUEUE_CONTEXT_NAMES),
            "queue_context_dim": representation.QUEUE_CONTEXT_DIM,
            "neural_inputs_identical_to_e215": True,
            "passive_reconfigure_streak_context_retained": True,
        },
        "initialization": {
            "kind": "fresh_random_from_model_seed",
            "model_seed": 0,
            "same_initialization_and_training_days_as_e215_seed0": True,
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
            "identical_to_e215": True,
            "max_option_steps": 10,
            "max_consecutive_defers": 16,
        },
        "guard_free_contract": {
            "reconfigure_mask": False,
            "witness_fallback": False,
            "selector_bypass": False,
            "critic_scores_every_certified_action": True,
            "td_target_uses_every_certified_next_action": True,
            "passive_guard_context_input_retained": True,
        },
        "telemetry": {
            "every_training_and_validation_episode": [
                "longest_reconfigure_run_without_accept_or_deliver",
                "reconfigure_only_safe_frontiers",
                "reconfigure_only_safe_frontier_rate",
                "step_limit_hit",
            ],
        },
        "decision_rule": {
            "guard_not_needed_requires": [
                "every validation case completes at every checkpoint",
                "no validation step-limit hits",
                f"maximum nonprogress rehandle run <= {MAX_ALLOWED_NONPROGRESS_REHANDLE_RUN}",
                f"mean checkpoint J no more than {100 * MAX_J_REGRESSION_FRACTION:.0f}% worse than E215",
                f"rehandles per 100 no more than {MAX_REHANDLE_PER_100_REGRESSION:.0f} above E215",
            ],
            "guard_needed_if": [
                "a validation case fails or hits the step limit",
                "nonprogress rehandle runs above the bound appear and persist",
            ],
            "early_reads": list(EARLY_READ_CHECKPOINTS),
        },
        "accelerations_retained": [
            "relocation-family certification",
            "timing-invariant certificate cache",
            "path cleanup",
            "fast successor inputs",
            "identity-deduplicated batched scoring",
            "cached batched TD backup",
            "roundoff-repaired objective audit",
        ],
        "checkpoint_selection": (
            "same fixed nine-case development-panel average and every-20-episode "
            "selection opportunities as E215; sealed test panel remains closed"
        ),
        "smoke_gate": (
            "prepare, self-check, and one complete development smoke episode must "
            "pass before the official training command is run"
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
    "METHOD_VERSION", "MODEL_SEEDS", "PROTOCOL", "REFERENCE_OUTPUT",
    "REPORT_CHECKPOINTS", "SCENARIO", "SCENARIOS", "SPEC",
    "TD_BACKUP_IMPLEMENTATION", "TOTAL_EPISODES", "TRAIN_BASE",
    "TRAINING_JOB_BUDGET", "TargetDenseEnv", "VALIDATION_EVERY",
    "VALIDATION_SEEDS", "contract", "digest", "method_declaration",
    "method_revision", "scenario_for", "sha256", "trainer_protocol",
]

