"""Shared contract helpers for the frozen R74/R15 multiseed rebuild."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = ROOT / (
    "projects/tfsn_consequence_value_iteration/aligned_rebuild_protocol.json"
)
SPECIFICATION = ROOT / (
    "projects/tfsn_consequence_value_iteration/FROZEN_R74_METHOD_SPECIFICATION.md"
)
REFERENCE = ROOT / (
    "projects/tfsn_consequence_value_iteration/ALIGNED_REBUILD_REFERENCE.md"
)
METHOD_REVISION = "tfsn_frozen_r74_aligned_multiseed_rebuild_r16"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "claim_level": (
            "reward_free_cross_domain_multiseed_preparation_rebuild_only"
        ),
        "declared_deviations": [],
        "oracle_inputs": [],
        "preparation_seeds": [0, 1, 2],
        "fixed_command_duration": 100,
        "episode_primitive_steps": 2000,
        "commands_per_physical_episode": 20,
        "commands_per_preparation_task": 75,
        "total_commands_per_seed": 36_000,
        "graph_includes_all_collection_phases": True,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "environment_reward_consumed": False,
        "task_gradient_updates": 0,
        "success_radius_read_by_planner_or_preparation": False,
        "confirmation_panel_accessed": False,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"aligned rebuild protocol has another {name}")
    if (ROOT / value["method_reference"]).resolve() != SPECIFICATION:
        raise ValueError("aligned rebuild method reference changed")
    if value.get("method_reference_sha256") != sha256(SPECIFICATION):
        raise ValueError("aligned rebuild method-reference hash changed")
    if (ROOT / value["rebuild_reference"]).resolve() != REFERENCE:
        raise ValueError("aligned rebuild reference changed")
    if value.get("rebuild_reference_sha256") != sha256(REFERENCE):
        raise ValueError("aligned rebuild rebuild-reference hash changed")
    commands = int(value["commands_per_preparation_task"])
    bootstrap = value["bootstrap"]
    alignment = value["alignment"]
    if int(bootstrap["tasks"]) * commands != int(bootstrap["commands"]):
        raise ValueError("bootstrap allocation changed")
    if (
        int(alignment["tasks_per_phase"]) * commands
        != int(alignment["commands_per_phase"])
    ):
        raise ValueError("alignment allocation changed")
    if (
        int(bootstrap["commands"])
        + len(alignment["phases"]) * int(alignment["commands_per_phase"])
        != int(value["total_commands_per_seed"])
    ):
        raise ValueError("preparation allocation does not sum to 36,000")
    return value


PROTOCOL_VALUE = load_protocol()


def seed_values(preparation_seed: int) -> dict:
    seed = int(preparation_seed)
    if seed not in PROTOCOL_VALUE["preparation_seeds"]:
        raise ValueError("preparation seed must be 0, 1, or 2")
    return {
        "preparation_seed": seed,
        "collection_seed_base": 7_400_000 + 100_000 * seed,
        "split_seed": 74_000 + seed,
        "training_seed": 75_000 + seed,
    }


def task_seed(preparation_seed: int, phase_index: int, task_id: int) -> int:
    base = seed_values(preparation_seed)["collection_seed_base"]
    return int(base + 1_000_003 * int(phase_index) + 10_000_019 * int(task_id))


def result_root(domain: str, preparation_seed: int, profile: str) -> Path:
    seed_values(preparation_seed)
    if domain not in ("antmaze", "pointmaze"):
        raise ValueError("unknown rebuild domain")
    if profile not in ("smoke", "full"):
        raise ValueError("profile must be smoke or full")
    return ROOT / (
        "projects/tfsn_consequence_value_iteration/results/"
        f"aligned_rebuild_r16/{domain}/seed{int(preparation_seed)}/{profile}"
    )


def dimensions(profile: str) -> tuple[int, int, int]:
    if profile == "smoke":
        return 2, 2, 3
    return (
        int(PROTOCOL_VALUE["bootstrap"]["tasks"]),
        int(PROTOCOL_VALUE["alignment"]["tasks_per_phase"]),
        int(PROTOCOL_VALUE["commands_per_preparation_task"]),
    )


def output_paths(root: Path) -> dict:
    root = Path(root).resolve()
    return {
        "root": root,
        "collection": root / "collection",
        "bootstrap": root / "collection/bootstrap",
        "acquisition": root / "collection/alignment/acquisition",
        "return": root / "collection/alignment/return",
        "models": root / "models",
        "roadmap": root / "roadmap",
        "planner": root / "planner",
        "evaluation": root / "evaluation",
        "audit": root / "audit",
    }
