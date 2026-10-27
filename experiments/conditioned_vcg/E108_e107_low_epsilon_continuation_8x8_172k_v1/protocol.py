"""Frozen protocol for the exact E107 episode-224 continuation."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    protocol as e107,
)


ROOT = e107.ROOT
PROTOCOL = "vcg_conditioned_e108_e107_low_epsilon_continuation_8x8_172k_v1"
METHOD_VERSION = "vcg_dense_v1_1_e108_e107_low_epsilon_continuation"
SCENARIO = e107.SCENARIO
MODEL_SEEDS = (0,)
SOURCE_EPISODE = 224
MAX_EPISODE = 320
# These are exact validation boundaries: 240 and 288 are divisible by the
# six-episode cadence, and the trainer always validates at the final horizon.
DECISION_CHECKPOINTS = (240, 288, 320)
STABILITY_WINDOW = 3
DEFAULT_OUTPUT = (
    ROOT
    / "results/vcg-conditioned-e108-e107-low-epsilon-continuation-8x8-172k-v1"
)
SOURCE_OUTPUT = e107.DEFAULT_OUTPUT
SOURCE_LATEST = (
    SOURCE_OUTPUT
    / "training/operational"
    / SCENARIO
    / "seed-0/latest.pth"
)
SOURCE_REPORT = SOURCE_OUTPUT / "e107-report.json"
SOURCE_CONTRACT = SOURCE_OUTPUT / "contract.json"
HORIZON_RULE = (
    "exact development continuation from E107 episode 224; retain the 0.05 "
    "epsilon floor and all learning state; inspect persistence at episodes "
    "240 and 288 before the episode-320 maximum"
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


@lru_cache(maxsize=1)
def _source_identity() -> dict:
    for path in (SOURCE_LATEST, SOURCE_REPORT, SOURCE_CONTRACT):
        if not path.is_file():
            raise RuntimeError(f"E108 missing E107 source artifact: {path}")
    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    source_contract = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
    if source_contract != e107.contract():
        raise RuntimeError("E107 source contract or authenticated code changed")
    if (
        report.get("status") != "complete"
        or int(report.get("completed_training_episodes", -1)) != SOURCE_EPISODE
    ):
        raise RuntimeError("E108 requires the completed E107 episode-224 report")
    checkpoint = torch.load(SOURCE_LATEST, map_location="cpu", weights_only=False)
    state = checkpoint.get("agent_state", {})
    if (
        int(checkpoint.get("completed_training_episodes", -1)) != SOURCE_EPISODE
        or checkpoint.get("training_protocol") != e107.PROTOCOL
        or abs(float(state.get("epsilon", -1.0)) - e107.EPSILON_END) > 1.0e-12
        or int(state.get("decision_count", -1)) < e107.EPSILON_DECAY_DECISIONS
        or len(checkpoint.get("training_history", ())) != SOURCE_EPISODE
        or checkpoint.get("queue_context_stored_in_replay") is not True
        or checkpoint.get("future_schedule_visible_to_policy") is not False
    ):
        raise RuntimeError("E107 source checkpoint is not the expected resumable endpoint")
    return {
        "checkpoint": str(SOURCE_LATEST.relative_to(ROOT)),
        "checkpoint_sha256": sha256(SOURCE_LATEST),
        "report_sha256": sha256(SOURCE_REPORT),
        "contract_sha256": source_contract["contract_sha256"],
        "completed_episode": SOURCE_EPISODE,
        "decision_count": int(state["decision_count"]),
        "gradient_steps": int(state["gradient_steps"]),
        "target_updates": int(state["target_updates"]),
        "replay_size": len(state["replay"]["memory"]),
        "epsilon": float(state["epsilon"]),
        "selected_checkpoint_episode": checkpoint.get(
            "selection_finalized_at_training_episode"
        ),
    }


@lru_cache(maxsize=1)
def contract() -> dict:
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        ROOT
        / "experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/representation.py",
        ROOT
        / "experiments/conditioned_vcg/E107_queue_context_stage_a_8x8_171k_v1/instrumentation.py",
        ROOT
        / "experiments/conditioned_vcg/E105_wait_aware_stage_a_8x8_169k_v1/rewards.py",
        ROOT
        / "experiments/conditioned_vcg/E100_accelerated_release_clock_163k_v1/fast_inputs.py",
        ROOT
        / "experiments/conditioned_vcg/target_size_dense_stage_a_v4/acceleration.py",
        ROOT / "experiments/conditioned_vcg/target_size_dense_stage_a_v2/run.py",
        ROOT / "train_vcg_dense_proper.py",
        ROOT / "run_vcg_objective_gamma_audit.py",
        ROOT / "viability_graph_hierarchy.py",
        ROOT / "viability_graph_episodic_audit.py",
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "scope": "development-only one-seed exact low-epsilon continuation",
        "source": _source_identity(),
        "scenario": SCENARIO,
        "model_seed": 0,
        "source_episode": SOURCE_EPISODE,
        "maximum_episode": MAX_EPISODE,
        "decision_checkpoints": list(DECISION_CHECKPOINTS),
        "validation_every": e107.e105.e78.VALIDATION_EVERY,
        "validation_seeds": list(e107.e105.e78.VALIDATION_SEEDS[SCENARIO]),
        "validation_panel_role": "reused development panel; no test claim",
        "epsilon_schedule_unchanged": {
            "start": e107.EPSILON_START,
            "end": e107.EPSILON_END,
            "warmup_decisions": e107.EPSILON_WARMUP_DECISIONS,
            "decay_decisions": e107.EPSILON_DECAY_DECISIONS,
            "source_at_floor": True,
        },
        "single_change_from_e107": "maximum training horizon 224 -> 320",
        "held_constant": (
            "E107 queue-observed architecture, reward, certification, guard, "
            "network, replay, optimizer, RNG, target cadence, epsilon schedule, "
            "workload, validation panel, and exact accelerations"
        ),
        "horizon_rule": HORIZON_RULE,
        "checkpoint_policy": (
            "report every validation; require three consecutive development "
            "validations for stability; no single best row is evidence"
        ),
        "stability_window_validations": STABILITY_WINDOW,
        "joint_mechanism_thresholds": deepcopy(
            e107.contract()["joint_mechanism_thresholds"]
        ),
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
        TRAIN_BASE=e107.TRAIN_BASE,
        TRAINING_JOB_BUDGET=(
            MAX_EPISODE * e107.e105.e78.SCENARIOS[SCENARIO].total_jobs
        ),
        TD_BACKUP_IMPLEMENTATION=e107.TD_BACKUP_IMPLEMENTATION,
        LATE_CHECKPOINTS=DECISION_CHECKPOINTS,
        DEFAULT_OUTPUT=DEFAULT_OUTPUT,
        TargetDenseEnv=e107.e105.WaitAwareDenseEnv,
        scenario_for=e107.e105.e78.scenario_for,
        contract=lambda: value,
    )


def continuation_resume_contract(source: dict, target_contract: dict) -> dict:
    value = deepcopy(source["resume_contract"])
    value.pop("e107_contract_sha256", None)
    value.update(
        {
            "training_protocol": PROTOCOL,
            "method_version": METHOD_VERSION,
            "total_episodes": MAX_EPISODE,
            "target_contract_sha256": target_contract["contract_sha256"],
            "e108_contract_sha256": target_contract["contract_sha256"],
            "e108_source_checkpoint_sha256": target_contract["source"][
                "checkpoint_sha256"
            ],
            "training_horizon_rule": HORIZON_RULE,
            "test_panel_opened": False,
        }
    )
    return value

