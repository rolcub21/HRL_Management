"""Protocol for the E215 episode-300 seed-0 Stage-B expansion."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    scenario as e191_scenario,
)
from experiments.conditioned_vcg.E215_e191_long_exploration_drift_pilot_274k_v1 import (
    protocol as e215,
)
from experiments.conditioned_vcg.E216_e214_stage_b_procedure_freeze_275k_v1 import (
    protocol as e216,
)
from projects.temporal_consequence_realization.method_conformance import (
    CHECKLIST,
    validate_protocol,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e220_e215_ep300_stage_b_279k_v1"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
METHOD_REVISION = "TCR-METHOD-2026-10-31.10"
MODEL_SEED = 0
SCENARIO = e191.SCENARIO
ROUNDS = 4
EPISODES_PER_ROUND = 9
TRAIN_EPISODES = ROUNDS * EPISODES_PER_ROUND
ALPHA_MIN = 10.0
ALPHA_MAX = 40.0
ALPHA_BANDS = ((10.0, 20.0), (20.0, 30.0), (30.0, 40.0))
EVAL_LAMBDAS = (10.0, 20.0, 40.0)
IMMEDIATE_ONLY_LAMBDAS = (20.0, 40.0)
TRACE_ETA = 0.99
TRAIN_BASE = 279_220_101  # divisible by three; modulo selects 20/50/80%
EVAL_BASE = 279_220_500   # divisible by three
EVAL_INSTANCE_SEEDS = tuple(EVAL_BASE + offset for offset in range(3))
SCHEDULE_SEED = 279_220_700
HEAD_SEED = 279_220_800
OPTIMIZER_LR = 2.0e-4
FIT_EPOCHS_PER_ROUND = 40
FIT_BATCH_SIZE = 128
MAX_RELATIVE_J_REGRESSION = 0.25
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e220-e215-ep300-stage-b-279k-v1"
)
E215_OUTPUT = e215.DEFAULT_OUTPUT
E215_DIRECTORY = E215_OUTPUT / "training/operational" / SCENARIO / "seed-0"
E215_CONTRACT = E215_OUTPUT / "contract.json"
E215_SUMMARY = E215_DIRECTORY / "training-summary.json"
E215_CHECKPOINT = E215_DIRECTORY / "best.pth"
STAGE_B_PROCEDURE = deepcopy(e216.STAGE_B_PROCEDURE)


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


def _read(path: Path) -> dict:
    if not Path(path).is_file():
        raise RuntimeError(f"missing E220 source artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E220 source artifact: {path}")
    return value


def frozen_stage_a_identity() -> dict:
    contract = _read(E215_CONTRACT)
    summary = _read(E215_SUMMARY)
    selected = summary.get("best_validation_record", {})
    if (
        contract.get("protocol") != e215.PROTOCOL
        or summary.get("status") != "complete"
        or int(summary.get("completed_training_episodes", -1)) != 300
        or int(selected.get("checkpoint_episode", -1)) != 300
        or Path(summary.get("best_checkpoint", "")).resolve()
        != E215_CHECKPOINT.resolve()
        or summary.get("best_checkpoint_sha256") != sha256(E215_CHECKPOINT)
    ):
        raise RuntimeError("E220 requires finalized E215 episode-300 seed 0")
    return {
        "source_protocol": e215.PROTOCOL,
        "source_contract_sha256": contract["contract_sha256"],
        "model_seed": MODEL_SEED,
        "selected_episode": 300,
        "checkpoint_role": "best_deployment_finalized",
        "checkpoint": str(E215_CHECKPOINT),
        "checkpoint_sha256": sha256(E215_CHECKPOINT),
        "summary": str(E215_SUMMARY),
        "summary_sha256": sha256(E215_SUMMARY),
        "selection_score": list(selected["selection_score"]),
        "selection_score_fields": list(selected["selection_score_fields"]),
    }


def round_schedule(round_index: int) -> tuple[dict, ...]:
    if not 0 <= int(round_index) < ROUNDS:
        raise ValueError("invalid E220 round")
    rng = random.Random(SCHEDULE_SEED + 100 * int(round_index))
    origin = TRAIN_BASE + 300 * int(round_index)
    rows = []
    for band_index, (lower, upper) in enumerate(ALPHA_BANDS):
        for workload_index, workload in enumerate(e191_scenario.LEVELS):
            seed = origin + band_index * 3 + workload_index
            row = {
                "round": int(round_index) + 1,
                "instance_seed": int(seed),
                "workload": float(workload),
                "alpha": float(lower + (upper - lower) * rng.random()),
                "alpha_band": [float(lower), float(upper)],
            }
            if float(e191_scenario.target_level(seed)) != float(workload):
                raise RuntimeError("E220 schedule workload identity differs")
            rows.append(row)
    rng.shuffle(rows)
    for offset, row in enumerate(rows):
        row["round_episode"] = offset + 1
        row["episode"] = int(round_index) * EPISODES_PER_ROUND + offset + 1
    return tuple(rows)


def training_schedule() -> tuple[dict, ...]:
    return tuple(
        row for round_index in range(ROUNDS) for row in round_schedule(round_index)
    )


def workload_for_seed(seed: int) -> float:
    if int(seed) not in EVAL_INSTANCE_SEEDS:
        raise ValueError("seed is outside the E220 development panel")
    value = float(e191_scenario.target_level(int(seed)))
    expected = float(e191_scenario.LEVELS[int(seed) - EVAL_BASE])
    if value != expected:
        raise RuntimeError("E220 evaluation workload identity differs")
    return value


def method_declaration() -> dict:
    audit = {
        identifier: {
            "status": "not_applicable",
            "evidence": (
                "E220 expands a historical conditioned-VCG controller head and "
                "does not train or evaluate the canonical law-condition representation."
            ),
        }
        for identifier, _description in CHECKLIST
    }
    evidence = {
        "reproducibility.method_revision": (
            "E220 pins the current method-reference revision at preparation."
        ),
        "reproducibility.content_hashes": (
            "The E215 checkpoint and summary, frozen E216 procedure, schedules, "
            "development instances, and behavioral implementations are hashed."
        ),
        "reproducibility.oracle_declarations": (
            "No oracle input or forced-prefix intervention is used."
        ),
        "reproducibility.failed_gates_retained": (
            "Calibration, completion, immediate-only comparisons, and every "
            "failed endpoint gate remain in the report."
        ),
        "reproducibility.preprotected_integration_gate": (
            "No fresh or protected panel is opened; E220 is development-only."
        ),
    }
    for identifier, text in evidence.items():
        audit[identifier] = {
            "status": "pass" if identifier != "reproducibility.preprotected_integration_gate" else "not_applicable",
            "evidence": text,
        }
    value = {
        "method_reference": METHOD_REFERENCE,
        "method_revision": METHOD_REVISION,
        "claim_level": "expressibility",
        "declared_deviations": [
            "E220 is a historical conditioned-VCG optimization experiment, not "
            "evidence for the canonical law-condition method."
        ],
        "oracle_inputs": [],
        "nonconforming_shortcuts": [],
        "conformance_audit": audit,
    }
    validate_protocol(value, source=PROTOCOL)
    return value


def _source_paths() -> tuple[Path, ...]:
    directory = Path(__file__).resolve().parent
    return (
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT / "experiments/conditioned_vcg/E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1/run.py",
        ROOT / "experiments/conditioned_vcg/E214_on_policy_trace_rehandle_stage_b_8x8_273k_v1/returns.py",
        ROOT / "experiments/conditioned_vcg/E204_e191_centered_rehandle_stage_b_263k_v1/centered.py",
        ROOT / "experiments/conditioned_vcg/E207_e204_immediate_only_ablation_266k_v1/immediate.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/representation.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py",
        ROOT / "train_vcg_preference_conditioned.py",
    )


def contract() -> dict:
    frozen = frozen_stage_a_identity()
    changed = {
        "frozen_stage_a": "E215 seed 0 episode-300 finalized development selection",
        "collection_instances": "new E220 development-only seeds",
        "evaluation_instances": "new paired E220 development-only seeds",
        "immediate_only_comparator": "rerun on the same E215 Stage A and instances",
        "causal_replays": "omitted because E213's forced prefix is specific to old E191",
        "method_revision": METHOD_REVISION,
    }
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        **deepcopy(method_declaration()),
        "scope": (
            "seed-0 development-only Stage-B expansion on finalized E215 episode 300; "
            "no sealed test panel and no multiseed claim"
        ),
        "source_stage_a_protocol": e215.PROTOCOL,
        "source_stage_b_freeze_protocol": e216.PROTOCOL,
        "source_stage_b_procedure": deepcopy(STAGE_B_PROCEDURE),
        "changes_from_frozen_e216_procedure": changed,
        "frozen_stage_a": [frozen],
        "scenario": SCENARIO,
        "scenario_public": deepcopy(e191_scenario.SCENARIO_SPEC.public_dict()),
        "model_seed": MODEL_SEED,
        "objective_J10": deepcopy(e191.contract()["objective"]),
        "quantity": STAGE_B_PROCEDURE["predicted_quantity"],
        "trace_target": STAGE_B_PROCEDURE["target"],
        "trace_eta": TRACE_ETA,
        "deployment_merit": STAGE_B_PROCEDURE["deployment_merit"],
        "preference_semantics": STAGE_B_PROCEDURE["preference_semantics"],
        "supported_alpha_range": [ALPHA_MIN, ALPHA_MAX],
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES_PER_ROUND,
        "round_data_retention": STAGE_B_PROCEDURE["data_retention"],
        "alpha_schedule": STAGE_B_PROCEDURE["alpha_sampling"],
        "alpha_bands": [list(value) for value in ALPHA_BANDS],
        "training_schedule": list(training_schedule()),
        "fit": deepcopy(STAGE_B_PROCEDURE["fit"]),
        "endpoint_evaluation": {
            "instance_seeds": list(EVAL_INSTANCE_SEEDS),
            "workload_levels": [workload_for_seed(seed) for seed in EVAL_INSTANCE_SEEDS],
            "learned_lambdas": list(EVAL_LAMBDAS),
            "immediate_only_lambdas": list(IMMEDIATE_ONLY_LAMBDAS),
            "pairing": "same deterministic EpisodeInstance for both arms at each alpha",
            "learned_episode_runs": len(EVAL_INSTANCE_SEEDS) * len(EVAL_LAMBDAS),
            "immediate_only_episode_runs": len(EVAL_INSTANCE_SEEDS) * len(IMMEDIATE_ONLY_LAMBDAS),
        },
        "predeclared_gates": {
            "calibration": "sample-weighted held-out MC MAE is below exact zero-head MC MAE",
            "frontier_value": "learned future head has lower J_alpha than immediate-only in at least one paired case",
            "no_catastrophic_regression": f"every paired J_alpha is at most {1.0 + MAX_RELATIVE_J_REGRESSION:.2f} times immediate-only",
            "strict_completion": "every learned and immediate-only development episode completes strictly",
        },
        "unchanged_frontier_certification_guard": True,
        "unchanged_acceleration": deepcopy(e191.contract()["acceleration_assertion"]),
        "training_runs": 1,
        "expected_complete_episode_runs": TRAIN_EPISODES + 15,
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()
        },
    }
    validate_protocol(payload, source=PROTOCOL)
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [name for name in globals() if name.isupper()] + [
    "contract", "digest", "frozen_stage_a_identity", "method_declaration",
    "round_schedule", "sha256", "training_schedule", "workload_for_seed",
]
