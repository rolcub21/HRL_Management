"""Frozen one-factor E188 protocol paired exactly with E187."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    protocol as e187,
)


ROOT = e187.ROOT
PROTOCOL = "vcg_conditioned_e188_mask_only_progress_guard_247k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e188_wait_corrected_choice_preserving_guard"
SCENARIO = e187.SCENARIO
SCENARIOS = e187.SCENARIOS
MODEL_SEEDS = e187.MODEL_SEEDS
TOTAL_EPISODES = e187.TOTAL_EPISODES
EPISODES_BY_SCENARIO = dict(e187.EPISODES_BY_SCENARIO)
VALIDATION_EVERY = e187.VALIDATION_EVERY
CHECKPOINT_EVERY = e187.CHECKPOINT_EVERY
VALIDATION_SEEDS = deepcopy(e187.VALIDATION_SEEDS)
# Deliberate common-random-number pairing with E187.
TRAIN_BASE = dict(e187.TRAIN_BASE)
TRAINING_JOB_BUDGET = e187.TRAINING_JOB_BUDGET
MAX_STEPS = e187.MAX_STEPS
GAMMA = e187.GAMMA
EPSILON_START = e187.EPSILON_START
EPSILON_END = e187.EPSILON_END
EPSILON_WARMUP_DECISIONS = e187.EPSILON_WARMUP_DECISIONS
EPSILON_DECAY_DECISIONS = e187.EPSILON_DECAY_DECISIONS
REPORT_CHECKPOINTS = e187.REPORT_CHECKPOINTS
LATE_CHECKPOINTS = REPORT_CHECKPOINTS
TD_BACKUP_IMPLEMENTATION = e187.TD_BACKUP_IMPLEMENTATION
SPEC = e187.SPEC
TargetDenseEnv = e187.TargetDenseEnv
scenario_for = e187.scenario_for
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e188-mask-only-progress-guard-247k-v1"
PARENT_OUTPUT = e187.DEFAULT_OUTPUT


MATCHED_PARENT_FIELDS = (
    "scenario",
    "scenario_public",
    "model_seeds",
    "episodes_per_seed",
    "training_jobs_per_seed",
    "train_instance_seed_base",
    "validation_every",
    "validation_seeds",
    "checkpoint_every",
    "report_checkpoints",
    "objective",
    "audit_scaffold_objective",
    "gamma",
    "finite_horizon",
    "epsilon",
    "workload",
    "architecture",
    "critic_information",
    "warm_start",
    "acceleration",
    "checkpoint_selection",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _matched_parent() -> tuple[dict, dict]:
    parent = e187.contract()
    expected_constants = {
        "model_seeds": list(MODEL_SEEDS),
        "episodes_per_seed": TOTAL_EPISODES,
        "train_instance_seed_base": TRAIN_BASE[SCENARIO],
        "validation_every": VALIDATION_EVERY,
        "validation_seeds": list(VALIDATION_SEEDS[SCENARIO]),
        "checkpoint_every": CHECKPOINT_EVERY,
        "gamma": GAMMA,
    }
    mismatches = {
        name: (parent.get(name), expected)
        for name, expected in expected_constants.items()
        if parent.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"E188 no longer matches E187: {mismatches!r}")
    selected = {name: deepcopy(parent[name]) for name in MATCHED_PARENT_FIELDS}
    return parent, selected


def contract() -> dict:
    parent, matched = _matched_parent()
    directory = Path(__file__).resolve().parent
    local_sources = (
        directory / "protocol.py",
        directory / "guard.py",
        directory / "representation.py",
        directory / "instrumentation.py",
        directory / "run.py",
        directory / "run.sh",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": "paired one-factor E187 guard intervention; development only",
        **matched,
        "parent_protocol": e187.PROTOCOL,
        "parent_contract_sha256": parent["contract_sha256"],
        "matched_parent_fields": list(MATCHED_PARENT_FIELDS),
        "matched_parent_fields_sha256": digest(matched),
        "paired_training_instances": True,
        "paired_validation_instances": True,
        "fresh_initialization_from_same_authenticated_e78_sources": True,
        "frontier": {
            **deepcopy(parent["frontier"]),
            "progress_guard": (
                "at the E187 nonprogress threshold remove Reconfigure only; "
                "the critic chooses among certified Accept, Deliver, and "
                "Defer; use one retained witness step only when none remains"
            ),
            "reconfigure_mask_at_threshold": True,
            "critic_retains_progress_choice": True,
            "one_witness_step_last_resort": True,
            "full_witness_commitment": False,
        },
        "guard_intervention": {
            "only_intended_method_delta_from_e187": True,
            "threshold": 2,
            "removed_action_at_threshold": "reconfigure",
            "preserved_actions": ["accept", "deliver", "defer"],
            "fallback": "one retained exact completion-witness step then recheck",
            "fallback_released_when_progress_choice_exists": True,
            "forced_transitions_retained_in_replay": True,
            "bellman_frontier_matches_deployment_guard": True,
        },
        "comparison": {
            "primary_checkpoints": [20, 40, 60],
            "all_shared_validation_checkpoints": list(REPORT_CHECKPOINTS),
            "pairing_key": "model_seed, checkpoint_episode, instance_seed",
            "lower_is_better": [
                "J", "admission_wait", "storage_clock_mae",
                "release_clock_mae", "physical_rehandles",
                "guard_forced_delivery_share", "steps",
            ],
            "higher_is_better": [
                "storage_within_20_rate", "release_within_20_rate",
            ],
            "behavior_counts": [
                "accept_decisions", "deliver_decisions",
                "reconfigure_decisions", "liveness_forced_decisions",
                "liveness_guard_activations", "peak_stored",
            ],
        },
        "validation_panel_role": (
            "same fresh E187-only development panel for paired guard comparison"
        ),
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            **deepcopy(parent["source_sha256"]),
            **{
                str(path.relative_to(ROOT)): sha256(path)
                for path in local_sources
            },
        },
    }
    return {**payload, "contract_sha256": digest(payload)}


def trainer_protocol():
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
    "CHECKPOINT_EVERY", "DEFAULT_OUTPUT", "EPISODES_BY_SCENARIO",
    "EPSILON_DECAY_DECISIONS", "EPSILON_END", "EPSILON_START",
    "EPSILON_WARMUP_DECISIONS", "GAMMA", "LATE_CHECKPOINTS", "MAX_STEPS",
    "METHOD_VERSION", "MODEL_SEEDS", "PARENT_OUTPUT", "PROTOCOL",
    "REPORT_CHECKPOINTS", "SCENARIO", "SCENARIOS", "SPEC",
    "TD_BACKUP_IMPLEMENTATION", "TOTAL_EPISODES", "TRAIN_BASE",
    "TRAINING_JOB_BUDGET", "VALIDATION_EVERY", "VALIDATION_SEEDS",
    "TargetDenseEnv", "contract", "scenario_for", "trainer_protocol",
]

