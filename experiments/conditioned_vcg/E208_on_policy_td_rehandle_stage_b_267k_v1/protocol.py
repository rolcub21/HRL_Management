"""Locked protocol for E208's on-policy TD Stage-B pilot."""

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
from experiments.conditioned_vcg.E202_e191_multiseed_development_freeze_261k_v1 import run as e202
from experiments.conditioned_vcg.E205_e204_paired_tradeoff_development_264k_v1 import protocol as e205
from projects.temporal_consequence_realization.method_conformance import CHECKLIST, validate_protocol


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e208_on_policy_td_rehandle_stage_b_267k_v1"
METHOD_REFERENCE = "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
METHOD_REVISION = "TCR-METHOD-2026-10-29.8"
MODEL_SEED = 0
SCENARIO = e191.SCENARIO
TRAIN_EPISODES = 36
CHECKPOINT_EPISODES = (12, 24, 36)
ALPHA_MIN = 10.0
ALPHA_MAX = 40.0
ALPHA_BANDS = ((10.0, 20.0), (20.0, 30.0), (30.0, 40.0))
EVAL_LAMBDAS = (10.0, 20.0, 40.0)
TRAIN_BASE = 267_208_101  # divisible by three
SCHEDULE_SEED = 267_208_700
HEAD_SEED = 267_208_800
OPTIMIZER_LR = 2.0e-4
TD_EPOCHS_PER_EPISODE = 4
TD_BATCH_SIZE = 128
TARGET_TAU = 0.05
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e208-on-policy-td-rehandle-stage-b-267k-v1"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def _read(path: Path) -> dict:
    if not Path(path).is_file():
        raise RuntimeError(f"missing E208 source artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E208 source artifact: {path}")
    return value


def freeze_identity() -> dict:
    manifest_path = e202.DEFAULT_OUTPUT / f"seed-{MODEL_SEED}" / e202.MANIFEST_NAME
    manifest = _read(manifest_path)
    checkpoint = Path(manifest["frozen_checkpoint"])
    expected = {
        "status": "complete", "protocol": e202.PROTOCOL,
        "source_protocol": e191.PROTOCOL, "model_seed": MODEL_SEED,
        "test_panels_opened": False, "development_only": True,
    }
    mismatch = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatch:
        raise RuntimeError(f"E208 frozen E191 identity differs: {mismatch}")
    if not checkpoint.is_file() or sha256(checkpoint) != manifest["frozen_checkpoint_sha256"]:
        raise RuntimeError("E208 frozen E191 checkpoint differs")
    return {
        "model_seed": MODEL_SEED, "selected_episode": int(manifest["selected_episode"]),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "seed_manifest_sha256": sha256(manifest_path),
        "cohort_manifest_sha256": sha256(e202.DEFAULT_OUTPUT / e202.MANIFEST_NAME),
    }


def training_schedule() -> tuple[dict, ...]:
    rng = random.Random(SCHEDULE_SEED)
    values = []
    for cycle in range(4):
        cycle_values = []
        for band_index, (lower, upper) in enumerate(ALPHA_BANDS):
            for workload_index, workload in enumerate(e191_scenario.LEVELS):
                seed = TRAIN_BASE + cycle * 300 + band_index * 3 + workload_index
                alpha = lower + (upper - lower) * rng.random()
                cycle_values.append({
                    "cycle": cycle + 1, "instance_seed": seed,
                    "workload": float(workload), "alpha": float(alpha),
                    "alpha_band": [lower, upper],
                })
        rng.shuffle(cycle_values)
        values.extend(cycle_values)
    if len(values) != TRAIN_EPISODES:
        raise RuntimeError("E208 schedule size differs")
    for row in values:
        observed = float(e191_scenario.target_level(row["instance_seed"]))
        if observed != row["workload"] or not ALPHA_MIN <= row["alpha"] < ALPHA_MAX:
            raise RuntimeError("E208 schedule identity differs")
    return tuple({**row, "episode": index + 1} for index, row in enumerate(values))


def method_declaration() -> dict:
    audit = {
        identifier: {"status": "not_applicable", "evidence": (
            "E208 is a VCG controller-head learning diagnostic and does not train "
            "or evaluate the canonical law-condition representation."
        )}
        for identifier, _description in CHECKLIST
    }
    for identifier, evidence in {
        "reproducibility.method_revision": "The E208 contract pins the current method revision.",
        "reproducibility.content_hashes": "Sources, frozen E191, schedule, instances, and checkpoints are hashed.",
        "reproducibility.oracle_declarations": "No oracle input is used; controller-diagnostic scope is explicit.",
        "reproducibility.failed_gates_retained": "Bellman residuals, MC errors, collapses, and price-regret failures are retained.",
    }.items():
        audit[identifier] = {"status": "pass", "evidence": evidence}
    value = {
        "method_reference": METHOD_REFERENCE, "method_revision": METHOD_REVISION,
        "claim_level": "expressibility",
        "declared_deviations": [
            "E208 diagnoses preference-conditioned VCG control, not the canonical law-condition method."
        ],
        "oracle_inputs": [], "nonconforming_shortcuts": [], "conformance_audit": audit,
    }
    validate_protocol(value, source=PROTOCOL)
    return value


def contract() -> dict:
    directory = Path(__file__).resolve().parent
    sources = tuple(directory / name for name in (
        "td.py", "protocol.py", "run.py", "plot.py", "run.sh", "README.md"
    ))
    frozen = freeze_identity()
    instances_path = e205.DEFAULT_OUTPUT / "instances.json"
    payload = {
        "schema_version": 1, "protocol": PROTOCOL, **deepcopy(method_declaration()),
        "scope": "seed-0 development pilot; fresh multiseed panel required",
        "source_protocol": e191.PROTOCOL,
        "source_contract_sha256": e191.contract()["contract_sha256"],
        "frozen_stage_a": [frozen], "scenario": SCENARIO,
        "scenario_public": deepcopy(e191_scenario.SCENARIO_SPEC.public_dict()),
        "model_seed": MODEL_SEED, "base_selected_episode": frozen["selected_episode"],
        "objective_J10": deepcopy(e191.contract()["objective"]),
        "quantity": "N_alpha(s,c)=E[sum_{k>0} I(reconfigure_k) | s,c,then follow pi_alpha]",
        "bellman_target": "I(c_next)+N_target_alpha(s_next,c_next); terminal target 0",
        "behavior_target_relation": "c_next is the action actually selected by the same fixed-alpha episode policy (SARSA)",
        "deployment_merit": "Q_E191 - 0.01*(alpha-10)*(I_reconfigure+N_alpha)",
        "alpha_schedule": "continuous stratified uniform per episode; fixed within episode",
        "alpha_range": [ALPHA_MIN, ALPHA_MAX], "alpha_bands": [list(value) for value in ALPHA_BANDS],
        "training_schedule": list(training_schedule()), "training_episodes": TRAIN_EPISODES,
        "checkpoint_episodes": list(CHECKPOINT_EPISODES),
        "td": {
            "algorithm": "episodic on-policy semi-gradient SARSA evaluation",
            "replay": "current complete episode only; no stale labeled replay",
            "epochs_per_episode": TD_EPOCHS_PER_EPISODE,
            "batch_size": TD_BATCH_SIZE, "learning_rate": OPTIMIZER_LR,
            "target_network": "Polyak delayed copy", "target_tau": TARGET_TAU,
            "gamma_over_macros": 1.0, "gradient_clip": 5.0,
            "initialization": "fresh near-zero future-count head; immediate-only initial policy",
        },
        "endpoint_evaluation": {
            "lambdas": list(EVAL_LAMBDAS), "instances": "exact E205 three-day panel",
            "instances_sha256": sha256(instances_path), "episode_runs": 9,
            "tests": ["paired frontier versus E207 immediate-only", "J_w cross-price regret on 10..40", "held-out MC and Bellman residuals"],
        },
        "unchanged_frontier_certification_guard": True,
        "unchanged_acceleration": deepcopy(e191.contract()["acceleration_assertion"]),
        "training_runs": 1, "test_panel_opened": False, "paper_evidence": False,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [name for name in globals() if name.isupper()] + [
    "contract", "digest", "freeze_identity", "method_declaration", "sha256", "training_schedule"
]
