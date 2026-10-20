"""Frozen protocol for E115 Release-Clock ET Cost VCG."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    protocol as e107,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1.representation import (
    QUEUE_CONTEXT_NAMES,
)
from .model import (
    ANCHOR_CONFIDENCE,
    ANCHOR_LOSS_WEIGHT,
    ANCHOR_MARGIN,
    COMPONENT_LOSS_WEIGHTS,
    CONSERVATIVE_UPDATE,
    LR_PHASES,
    MAX_ANCHOR_FRONTIERS_PER_UPDATE,
)
from .rewards import (
    ARRIVAL_GAP,
    COMPONENT_NAMES,
    EARLINESS_WEIGHT,
    EQUIPMENT_WEIGHT,
    REHANDLE_WEIGHT,
    ReleaseETCostEnv,
    SCALAR_WEIGHTS,
    TARDINESS_WEIGHT,
    TOLERANCE,
)


ROOT = e107.ROOT
PROTOCOL = "vcg_conditioned_e115_release_et_cost_vcg_8x8_179k_v1_repair2"
METHOD_VERSION = "release_clock_et_cost_vcg_e115_v1_repair2"
ARCHITECTURE_NAME = "Release-Clock ET Cost VCG"
SCENARIO = e107.SCENARIO
MODEL_SEEDS = (0,)
MAX_EPISODE = 240
DECISION_CHECKPOINTS = (36, 60, 84, 112, 144, 180, 216, 240)
EPSILON_START = 0.90
EPSILON_END = 0.05
EPSILON_WARMUP_DECISIONS = 0
EPSILON_DECAY_DECISIONS = 24_000
TRAIN_BASE = {SCENARIO: 179_000_000}
TRAINING_JOB_BUDGET = MAX_EPISODE * e107.e105.e78.SCENARIOS[SCENARIO].total_jobs
TD_BACKUP_IMPLEMENTATION = (
    "e115_batched_exact_vector_double_q_frozen_target_component_cache_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/vcg-conditioned-e115-release-et-cost-vcg-8x8-179k-v1-repair2"
)
HORIZON_RULE = (
    "fresh one-seed development training through at most 240 episodes; "
    "the first mechanism decision is episode 36 and no single reused-panel "
    "optimum is a paper-selection rule"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    scenario = e107.e105.e78.SCENARIOS[SCENARIO]
    sources = (
        directory / "protocol.py",
        directory / "model.py",
        directory / "rewards.py",
        directory / "instrumentation.py",
        directory / "run.py",
        directory / "run.sh",
        ROOT / "experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E100_accelerated_release_clock_163k_v1/fast_inputs.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.py",
        ROOT / "train_vcg_dense_proper.py",
        ROOT / "run_vcg_objective_gamma_audit.py",
        ROOT / "viability_graph_hierarchy.py",
        ROOT / "viability_graph_episodic_audit.py",
        ROOT / "viability_graph_preference_conditioned.py",
    )
    learning_rate_schedule = [
        {"before_decision": int(boundary), "learning_rate": value}
        for boundary, value in LR_PHASES if boundary != float("inf")
    ] + [{"from_decision": 24_000, "learning_rate": LR_PHASES[-1][1]}]
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "architecture_name": ARCHITECTURE_NAME,
        "scope": "development-only one-seed objective/representation pilot",
        "scenario": SCENARIO,
        "scenario_public": scenario.public_dict(),
        "model_seeds": list(MODEL_SEEDS),
        "fresh_initialization": True,
        "source_checkpoint": None,
        "maximum_episode": MAX_EPISODE,
        "first_training_stop": 36,
        "decision_checkpoints": list(DECISION_CHECKPOINTS),
        "training_job_budget": TRAINING_JOB_BUDGET,
        "train_instance_seed_base": TRAIN_BASE[SCENARIO],
        "validation_every": e107.e105.e78.VALIDATION_EVERY,
        "validation_seeds": list(e107.e105.e78.VALIDATION_SEEDS[SCENARIO]),
        "validation_panel_role": "reused development panel; no test claim",
        "workload": {
            "empty_start": True,
            "arrival_gap_steps": ARRIVAL_GAP,
            "duration_distribution": "E28 frozen Poisson mean 80",
            "last_release_step": ARRIVAL_GAP * (scenario.total_jobs - 1),
            "capacity_rationale": (
                "30-step spacing exceeds the 22.3-step-per-delivery reference "
                "observed for the strongest E111 baseline"
            ),
        },
        "objective": {
            "sense": "minimize",
            "formula": "sum_i(2*T_i + 1*E_i) + 0.1*L + 10*R",
            "due_date": "d_i = arrival_step_i + storage_steps_needed_i",
            "tolerance_steps": TOLERANCE,
            "tardiness_weight": TARDINESS_WEIGHT,
            "earliness_weight": EARLINESS_WEIGHT,
            "equipment_work_weight": EQUIPMENT_WEIGHT,
            "rehandle_weight_beta": REHANDLE_WEIGHT,
            "equipment_work_definition": "all non-WAIT primitive actions",
            "completion": "hard certified/guarded feasibility requirement",
            "delivery_base": 0.0,
            "dense_tardiness": (
                "one unit per overdue undelivered job per primitive interval"
            ),
            "earliness": "charged once at delivery beyond the 20-step tolerance",
            "primitive_components": list(COMPONENT_NAMES),
            "scalar_weights_in_q_units": list(SCALAR_WEIGHTS),
            "legacy_audit_compatibility": (
                "both inherited dual-rescore slots carry the E115 operator "
                "return; exactness is checked from primitive heads and the "
                "completed-episode terminal ET formula"
            ),
        },
        "critic": {
            "shared_graph_action_queue_trunk": True,
            "heads": list(COMPONENT_NAMES),
            "nonnegative_cost_heads": list(COMPONENT_NAMES),
            "component_loss_weights": list(COMPONENT_LOSS_WEIGHTS),
            "public_queue_context": list(QUEUE_CONTEXT_NAMES),
            "released_job_due_slack_in_graph": True,
            "due_slack_formula": "arrival + duration - current_time",
            "future_arrival_schedule_visible": False,
            "unreleased_duration_visible": False,
        },
        "conservative_update": {
            "version": CONSERVATIVE_UPDATE,
            "confidence": ANCHOR_CONFIDENCE,
            "margin": ANCHOR_MARGIN,
            "loss_weight": ANCHOR_LOSS_WEIGHT,
            "maximum_frontiers_per_update": MAX_ANCHOR_FRONTIERS_PER_UPDATE,
            "scope": "local frozen-target anti-drift trust region; not an oracle label",
        },
        "epsilon": {
            "start": EPSILON_START,
            "end": EPSILON_END,
            "warmup_decisions": EPSILON_WARMUP_DECISIONS,
            "decay_decisions": EPSILON_DECAY_DECISIONS,
        },
        "learning_rate_schedule": learning_rate_schedule,
        "horizon_rule": HORIZON_RULE,
        "gamma": 1.0,
        "reward_scale": 0.01,
        "certification_and_guard": "unchanged exact VCG-Dense v1.1",
        "td_backup": (
            "online-selected deterministic hierarchical candidate supplies "
            "all four frozen-target continuation cost heads"
        ),
        "td_backup_implementation": TD_BACKUP_IMPLEMENTATION,
        "checkpoint_policy": (
            "report the prespecified trajectory and rolling windows; do not "
            "select a lucky development checkpoint"
        ),
        "episode_36_mechanism_gate": {
            "all_validation_episodes_strict_complete": True,
            "component_accounting_exact": True,
            "finite_operator_cost": True,
            "purpose": (
                "verify learnability and accounting before authorizing longer "
                "training; no superiority threshold before matched baselines"
            ),
        },
        "stage_b_authorized": False,
        "independent_test_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    return {**payload, "contract_sha256": digest(payload)}


def trainer_protocol():
    value = contract()
    return SimpleNamespace(
        PROTOCOL=PROTOCOL,
        METHOD_VERSION=METHOD_VERSION,
        SCENARIO=SCENARIO,
        SCENARIOS=e107.e105.e78.SCENARIOS,
        MODEL_SEEDS=MODEL_SEEDS,
        EPISODES_BY_SCENARIO={SCENARIO: MAX_EPISODE},
        TOTAL_EPISODES=MAX_EPISODE,
        VALIDATION_EVERY=e107.e105.e78.VALIDATION_EVERY,
        CHECKPOINT_EVERY=e107.e105.e78.CHECKPOINT_EVERY,
        VALIDATION_SEEDS=e107.e105.e78.VALIDATION_SEEDS,
        TRAIN_BASE=TRAIN_BASE,
        TRAINING_JOB_BUDGET=TRAINING_JOB_BUDGET,
        TD_BACKUP_IMPLEMENTATION=TD_BACKUP_IMPLEMENTATION,
        LATE_CHECKPOINTS=DECISION_CHECKPOINTS,
        DEFAULT_OUTPUT=DEFAULT_OUTPUT,
        TargetDenseEnv=ReleaseETCostEnv,
        scenario_for=e107.e105.e78.scenario_for,
        contract=lambda: value,
    )


__all__ = [
    "ARCHITECTURE_NAME",
    "DECISION_CHECKPOINTS",
    "DEFAULT_OUTPUT",
    "EPSILON_DECAY_DECISIONS",
    "EPSILON_END",
    "EPSILON_START",
    "EPSILON_WARMUP_DECISIONS",
    "HORIZON_RULE",
    "MAX_EPISODE",
    "METHOD_VERSION",
    "MODEL_SEEDS",
    "PROTOCOL",
    "ROOT",
    "SCENARIO",
    "TD_BACKUP_IMPLEMENTATION",
    "TRAIN_BASE",
    "contract",
    "trainer_protocol",
]
