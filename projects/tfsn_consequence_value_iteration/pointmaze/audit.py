"""Confirm Part A without opening the current method's confirmation outcomes."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path

from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv

from .core import atomic_json, load_json, protocol, resolve_repository, sha256
from projects.tfsn_consequence_value_iteration.shared.settings import (
    COMMAND_STEPS,
    MANIFEST_PATH,
    finite_horizon,
)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = protocol()
    evaluator = contract["evaluator"]
    derived_horizon = finite_horizon(evaluator["episode_primitive_steps"])

    executor_rows = []
    for seed_text, item in contract["executor"]["seeds"].items():
        checkpoint = Path(item["checkpoint"])
        manifest = Path(item["manifest"])
        if sha256(checkpoint) != item["checkpoint_sha256"]:
            raise ValueError(f"E105 seed {seed_text} checkpoint changed")
        if sha256(manifest) != item["manifest_sha256"]:
            raise ValueError(f"E105 seed {seed_text} manifest changed")
        metadata = load_json(manifest)
        executor_rows.append({
            "seed": int(seed_text),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": item["checkpoint_sha256"],
            "shared_reward_free_environment_transitions": metadata[
                "shared_reward_free_environment_transitions"
            ],
            "gradient_updates": metadata["gradient_updates"],
            "additional_environment_transitions": metadata[
                "additional_environment_transitions"
            ],
            "goal_conditioning": metadata["goal_conditioning"],
        })

    for source in contract["initial_reward_free_sources_seed0"]:
        summary = resolve_repository(source["path"]) / "summary.json"
        if sha256(summary) != source["summary_sha256"]:
            raise ValueError(f"reward-free source changed: {source['name']}")
        metadata = load_json(summary)
        if metadata.get("task_reward_used") is not False:
            raise ValueError(f"source is not reward-free: {source['name']}")
        if metadata.get("hidden_goals_used") is not False:
            raise ValueError(f"source exposed hidden goals: {source['name']}")
        if int(metadata["rows"]) != int(source["rows"]):
            raise ValueError(f"source row count changed: {source['name']}")

    development = contract["development_panel"]
    development_path = resolve_repository(development["protocol"])
    if sha256(development_path) != development["protocol_sha256"]:
        raise ValueError("development panel protocol changed")
    confirmation = contract["confirmation_panel"]
    confirmation_path = resolve_repository(confirmation["paper_protocol"])
    if sha256(confirmation_path) != confirmation["paper_protocol_sha256"]:
        raise ValueError("E373 panel protocol changed")
    paper = load_json(confirmation_path)
    panel = paper["panel"]
    if panel["goals_per_seed"] != confirmation["goals"]:
        raise ValueError("E373 goal count changed")
    if panel["environment_seeds"] != confirmation["environment_seeds"]:
        raise ValueError("E373 environment seeds changed")

    signature = inspect.signature(PointMazeEnv.__init__)
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name in ("continuing_task", "reset_target")
    }
    if defaults != {"continuing_task": True, "reset_target": False}:
        raise ValueError("installed PointMaze defaults changed")
    report = {
        "schema_version": 1,
        "method_revision": contract["method_revision"],
        "claim_level": contract["claim_level"],
        "method_reference": str(resolve_repository(contract["method_reference"])),
        "method_reference_sha256": sha256(
            resolve_repository(contract["method_reference"])
        ),
        "environment": contract["environment"],
        "gymnasium_version": package_version("gymnasium"),
        "gymnasium_robotics_version": package_version("gymnasium-robotics"),
        "installed_PointMazeEnv_defaults": defaults,
        "episode_primitive_steps": evaluator["episode_primitive_steps"],
        "command_primitive_steps": COMMAND_STEPS,
        "derived_commands_per_episode": derived_horizon,
        "shared_method_manifest": str(MANIFEST_PATH.resolve()),
        "shared_method_manifest_sha256": sha256(MANIFEST_PATH),
        "success_radius_m_evaluator_only": evaluator[
            "success_radius_m_evaluator_only"
        ],
        "executor_seeds": executor_rows,
        "initial_reward_free_rows_seed0": sum(
            int(source["rows"])
            for source in contract["initial_reward_free_sources_seed0"]
        ),
        "development_panel": development,
        "confirmation_panel": confirmation,
        "confirmation_panel_accessed_for_current_method": False,
        "passed": True,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
