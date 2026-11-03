"""Locked development protocol for E191's centered rehandle Stage B."""

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
from experiments.conditioned_vcg.E202_e191_multiseed_development_freeze_261k_v1 import (
    run as e202,
)


ROOT = e191.ROOT
PROTOCOL = "vcg_conditioned_e204_e191_centered_rehandle_stage_b_263k_v1"
SCENARIO = e191.SCENARIO
MODEL_SEEDS = e191.MODEL_SEEDS
LAMBDA_CENTER = 10.0
LAMBDAS = (0.0, 5.0, 10.0, 20.0, 40.0)
WORKLOAD_LEVELS = tuple(float(value) for value in e191_scenario.LEVELS)
ROUNDS = 4
EPISODES_PER_ROUND = len(LAMBDAS) * len(WORKLOAD_LEVELS)
VALIDATION_EPISODES_PER_ROUND = len(LAMBDAS)
LATE_UPDATE_FRACTION = 0.25
COLLECTION_BASE = 263_204_100  # divisible by three; seed modulo selects workload
COLLECTION_MODEL_STRIDE = 100_002
COLLECTION_ROUND_STRIDE = 3_000
FREEZE_OUTPUT = e202.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    ROOT / "results/vcg-conditioned-e204-e191-centered-rehandle-stage-b-263k-v1"
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read(path: Path) -> dict:
    if not Path(path).is_file():
        raise RuntimeError(f"missing E204 source artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid E204 source artifact: {path}")
    return value


def freeze_identity(model_seed: int) -> dict:
    if model_seed not in MODEL_SEEDS:
        raise ValueError("model seed must be 0, 1, or 2")
    root = _read(FREEZE_OUTPUT / e202.MANIFEST_NAME)
    manifest_path = FREEZE_OUTPUT / f"seed-{model_seed}" / e202.MANIFEST_NAME
    manifest = _read(manifest_path)
    checkpoint = Path(manifest["frozen_checkpoint"])
    expected = {
        "status": "complete",
        "protocol": e202.PROTOCOL,
        "source_protocol": e191.PROTOCOL,
        "source_contract_sha256": e191.contract()["contract_sha256"],
        "model_seed": model_seed,
        "test_panels_opened": False,
        "development_only": True,
    }
    mismatch = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"E204 E202 seed manifest differs: {mismatch}")
    if (
        root.get("status") != "complete"
        or root.get("protocol") != e202.PROTOCOL
        or tuple(root.get("model_seeds", ())) != MODEL_SEEDS
        or root.get("test_panels_opened") is not False
    ):
        raise RuntimeError("E204 E202 cohort manifest differs")
    if not checkpoint.is_file():
        raise RuntimeError(f"E204 frozen checkpoint is missing: {checkpoint}")
    observed = sha256(checkpoint)
    if observed != manifest["frozen_checkpoint_sha256"]:
        raise RuntimeError("E204 frozen checkpoint SHA-256 differs")
    return {
        "model_seed": model_seed,
        "selected_episode": int(manifest["selected_episode"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": observed,
        "seed_manifest_sha256": sha256(manifest_path),
        "cohort_manifest_sha256": sha256(FREEZE_OUTPUT / e202.MANIFEST_NAME),
    }


def collection_coordinates(model_seed: int, round_index: int) -> tuple[dict, ...]:
    if model_seed not in MODEL_SEEDS or not 0 <= round_index < ROUNDS:
        raise ValueError("invalid E204 collection coordinate")
    values = []
    for lambda_index, value in enumerate(LAMBDAS):
        for level_index, level in enumerate(WORKLOAD_LEVELS):
            seed = (
                COLLECTION_BASE
                + model_seed * COLLECTION_MODEL_STRIDE
                + round_index * COLLECTION_ROUND_STRIDE
                + lambda_index * 3
                + level_index
            )
            if seed % 3 != level_index:
                raise RuntimeError("E204 collection seed no longer selects its workload")
            values.append({
                "instance_seed": seed,
                "lambda": value,
                "lambda_index": lambda_index,
                "workload": level,
                "workload_index": level_index,
                "validation": level_index == (lambda_index + round_index) % 3,
            })
    random.Random(
        COLLECTION_BASE + model_seed * 10_000 + round_index
    ).shuffle(values)
    if sum(bool(value["validation"]) for value in values) != VALIDATION_EPISODES_PER_ROUND:
        raise RuntimeError("E204 validation split no longer covers every lambda")
    return tuple(values)


def contract() -> dict:
    parent = e191.contract()
    directory = Path(__file__).resolve().parent
    sources = (
        directory / "centered.py",
        directory / "protocol.py",
        directory / "run.py",
        directory / "run.sh",
        directory / "README.md",
        ROOT / "vcg_v11_conditioned_handling.py",
        ROOT / "train_vcg_preference_conditioned.py",
        ROOT / "experiments/conditioned_vcg/E191_fresh_long_stay_crowding_250k_v1/stage.py",
        ROOT / "experiments/conditioned_vcg/E202_e191_multiseed_development_freeze_261k_v1/run.py",
    )
    freezes = [freeze_identity(seed) for seed in MODEL_SEEDS]
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "development-only centered Stage B; no sealed quality panel",
        "source_protocol": e191.PROTOCOL,
        "source_contract_sha256": parent["contract_sha256"],
        "freeze_protocol": e202.PROTOCOL,
        "freeze_output": str(FREEZE_OUTPUT),
        "frozen_stage_a": freezes,
        "scenario": SCENARIO,
        "scenario_public": deepcopy(e191_scenario.SCENARIO_SPEC.public_dict()),
        "objective_J10": deepcopy(parent["objective"]),
        "gamma": float(parent["gamma"]),
        "reward_scale": 0.01,
        "model_seeds": list(MODEL_SEEDS),
        "lambda_values": list(LAMBDAS),
        "lambda_center": LAMBDA_CENTER,
        "deployment_merit": (
            "Q_E191 - 0.01*(lambda-10)*(immediate_rehandle + "
            "predicted_future_rehandles)"
        ),
        "lambda_center_behavior": "exact frozen E191 action delegation",
        "rounds": ROUNDS,
        "episodes_per_round": EPISODES_PER_ROUND,
        "episodes_per_seed": ROUNDS * EPISODES_PER_ROUND,
        "workload_levels_per_round": list(WORKLOAD_LEVELS),
        "full_factorial_per_round": True,
        "validation_episodes_per_round": VALIDATION_EPISODES_PER_ROUND,
        "validation_split": (
            "one episode per lambda; workload rotates deterministically by round"
        ),
        "collection_base": COLLECTION_BASE,
        "handling_head": {
            "input": (
                "E191 full 256-dimensional pre-Q-head representation including "
                "the public queue/progress context embedding"
            ),
            "initialization": "deterministic near-zero future-count prior",
            "initial_future_rehandles": 0.01,
            "epochs_per_round": 8,
            "batch_size": 256,
            "learning_rate": 0.0005,
            "update_fraction_rounds_1_to_3": 1.0,
            "update_fraction_round_4": LATE_UPDATE_FRACTION,
            "future_label": "complete_episode_mc_future_physical_rehandles",
            "condition": "deployment rehandle weight lambda",
        },
        "unchanged_e191_frontier_certification_guard": True,
        "unchanged_e191_acceleration": deepcopy(parent["acceleration_assertion"]),
        "checkpoint_policy": "terminal_round_4_only; fresh later panel required",
        "e203_not_used_for_design_or_selection": True,
        "test_panel_opened": False,
        "paper_evidence": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    return {**payload, "contract_sha256": digest(payload)}


__all__ = [
    "COLLECTION_BASE",
    "DEFAULT_OUTPUT",
    "EPISODES_PER_ROUND",
    "FREEZE_OUTPUT",
    "LAMBDA_CENTER",
    "LAMBDAS",
    "LATE_UPDATE_FRACTION",
    "MODEL_SEEDS",
    "PROTOCOL",
    "ROUNDS",
    "SCENARIO",
    "VALIDATION_EPISODES_PER_ROUND",
    "WORKLOAD_LEVELS",
    "collection_coordinates",
    "contract",
    "digest",
    "freeze_identity",
    "sha256",
]
