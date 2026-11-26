"""Create the method-neutral prospective PointMaze confirmation panel."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from scripts.pointmaze.supplied_goal.make_native_bank import (
    build_bank,
    load_native_geometry,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "panels" / "fresh_confirmation_r1"
GOALS = OUTPUT / "goals.csv"
STARTS = OUTPUT / "starts.csv"
FREEZE = OUTPUT / "freeze.json"

ENVIRONMENT = "PointMaze_Medium-v3"
SPLIT_SEED = 202611240026
START_BLOCK_SEED = 61
EPISODES = 8
EXPECTED_CELLS = 26

PRIOR_GOAL_FILES = {
    "E373": (
        HERE.parents[1]
        / "tfsn_supplied_goal_pointmaze"
        / "baseline_panel"
        / "goals.csv"
    ),
    "E441": Path(
        "/home/ai_diagnosis/SG/tfsn-phase2-e373-isolated/"
        "experiments/441_pointmaze_direct_consequence_confirmation/"
        "split/goals.csv"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate(goals: pd.DataFrame, starts: pd.DataFrame) -> None:
    if len(goals) != EXPECTED_CELLS or goals.goal_name.nunique() != EXPECTED_CELLS:
        raise ValueError("fresh panel does not contain one goal per free cell")
    if len(starts) != EXPECTED_CELLS * EPISODES:
        raise ValueError("fresh panel start matrix has the wrong size")
    if starts.duplicated(["goal_name", "seed", "episode"]).any():
        raise ValueError("fresh panel contains duplicate scenario keys")
    if set(starts.goal_name) != set(goals.goal_name):
        raise ValueError("fresh panel goals and starts do not match")
    if set(starts.seed.astype(int)) != {START_BLOCK_SEED}:
        raise ValueError("fresh panel start block seed changed")
    if not starts.groupby("goal_name").size().eq(EPISODES).all():
        raise ValueError("fresh panel episode blocks are incomplete")


def verify_existing() -> dict | None:
    present = [path.exists() for path in (GOALS, STARTS, FREEZE)]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("fresh panel is partially present; repair is forbidden")
    payload = json.loads(FREEZE.read_text())
    if sha256(GOALS) != payload.get("goals_sha256"):
        raise ValueError("fresh panel goal hash changed")
    if sha256(STARTS) != payload.get("starts_sha256"):
        raise ValueError("fresh panel start hash changed")
    validate(pd.read_csv(GOALS), pd.read_csv(STARTS))
    return payload


def main() -> None:
    existing = verify_existing()
    if existing is not None:
        print(json.dumps(existing, indent=2, sort_keys=True))
        return

    geometry = load_native_geometry(ENVIRONMENT)
    goals, starts = build_bank(
        geometry,
        split_seed=SPLIT_SEED,
        scenario_seeds=[START_BLOCK_SEED],
        episodes=EPISODES,
        min_start_goal_distance=2.0,
        max_start_goal_distance=8.0,
        success_radius=0.4,
        expected_free_cells=EXPECTED_CELLS,
    )
    goal_frame = pd.DataFrame(goals).sort_values("goal_name").reset_index(drop=True)
    start_frame = (
        pd.DataFrame(starts)
        .sort_values(["goal_name", "seed", "episode"])
        .reset_index(drop=True)
    )
    validate(goal_frame, start_frame)

    fresh_xy = set(zip(goal_frame.goal_x.astype(float), goal_frame.goal_y.astype(float)))
    prior_matches = {}
    for name, path in PRIOR_GOAL_FILES.items():
        prior = pd.read_csv(path)
        prior_xy = set(zip(prior.goal_x.astype(float), prior.goal_y.astype(float)))
        prior_matches[name] = len(fresh_xy & prior_xy)
    if any(prior_matches.values()):
        raise ValueError(f"fresh goal coordinates overlap a prior panel: {prior_matches}")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    atomic_csv(GOALS, goal_frame)
    atomic_csv(STARTS, start_frame)
    payload = {
        "schema_version": 1,
        "experiment_id": "pointmaze_fresh_confirmation_r1",
        "status": "frozen_before_any_current_method_or_baseline_evaluation",
        "environment": ENVIRONMENT,
        "panel_generation": (
            "one_independently_jittered_coordinate_per_native_free_cell"
        ),
        "split_seed": SPLIT_SEED,
        "start_block_seed": START_BLOCK_SEED,
        "native_goal_cells": EXPECTED_CELLS,
        "episodes_per_goal": EPISODES,
        "hidden_goal_episode_rows_per_preparation_seed": EXPECTED_CELLS * EPISODES,
        "supplied_goal_episodes_per_goal": 2,
        "supplied_goal_episode_rows_per_preparation_seed": EXPECTED_CELLS * 2,
        "preparation_seeds_planned": [0, 1, 2],
        "same_goal_and_start_rows_for_every_method_and_preparation_seed": True,
        "goal_or_controller_outcomes_used": False,
        "task_reward_used": False,
        "exact_prior_goal_coordinate_matches": prior_matches,
        "goals": str(GOALS.relative_to(HERE.parents[2])),
        "starts": str(STARTS.relative_to(HERE.parents[2])),
        "goals_sha256": sha256(GOALS),
        "starts_sha256": sha256(STARTS),
    }
    atomic_json(FREEZE, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
