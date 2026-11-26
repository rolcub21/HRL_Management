"""Locked, exposed PointMaze development panel construction."""

from __future__ import annotations

import pandas as pd
import numpy as np

from .core import load_json, protocol, resolve_repository, sha256


def load_development_panel() -> pd.DataFrame:
    """Recreate the inherited nine-goal panel without reading task outcomes."""

    contract = protocol()["development_panel"]
    evaluation_path = resolve_repository(contract["protocol"])
    if sha256(evaluation_path) != contract["protocol_sha256"]:
        raise ValueError("development panel protocol changed")
    evaluation = load_json(evaluation_path)
    goals_path = (evaluation_path.parent / evaluation["source_goal_file"]).resolve()
    if sha256(goals_path) != evaluation["source_goal_file_sha256"]:
        raise ValueError("development source goals changed")

    goals = pd.read_csv(goals_path).sort_values("goal_name").reset_index(drop=True)
    coordinates = goals[["cell_center_x", "cell_center_y"]].to_numpy(dtype=float)
    selected = [0]
    while len(selected) < int(evaluation["tasks"]):
        squared = np.sum(
            np.square(coordinates[:, None, :] - coordinates[selected][None, :, :]),
            axis=2,
        )
        nearest = squared.min(axis=1)
        nearest[selected] = -1.0
        selected.append(int(np.argmax(nearest)))
    anchors = goals.iloc[selected].reset_index(drop=True)
    if anchors.base_goal_name.astype(str).tolist() != evaluation[
        "expected_base_goal_order"
    ]:
        raise ValueError("development spatial panel selection changed")

    rows = []
    for target_index, target in anchors.iterrows():
        for episode, offset in enumerate(range(1, len(anchors))):
            start = anchors.iloc[(target_index + offset) % len(anchors)]
            rows.append({
                "goal_name": f"spatial_diagnostic_{target.base_goal_name}",
                "seed": int(evaluation["panel_seed"]),
                "episode": int(episode),
                "goal_x": float(target.goal_x),
                "goal_y": float(target.goal_y),
                "start_x": float(start.goal_x),
                "start_y": float(start.goal_y),
            })
    panel = pd.DataFrame(rows)
    expected_rows = int(evaluation["tasks"]) * int(evaluation["episodes_per_task"])
    if len(panel) != expected_rows:
        raise ValueError("development panel row count changed")
    if panel.groupby(["goal_name", "seed"]).ngroups != int(evaluation["tasks"]):
        raise ValueError("development panel task count changed")
    return panel


def load_frozen_hidden_goal_panel(panel_protocol) -> pd.DataFrame:
    """Load a declared hidden-goal subset from frozen goal/start files."""

    contract = load_json(panel_protocol)
    if contract.get("oracle_inputs") != []:
        raise ValueError("hidden-goal panel declares oracle inputs")
    if contract.get("hidden_goal_visible_to_planner_executor_or_models") is not False:
        raise ValueError("panel does not declare the goal hidden")
    task_set = contract.get("evaluation_set", contract.get("development_subset"))
    if task_set is None:
        raise ValueError("hidden-goal panel has no declared task set")
    if task_set.get("outcomes_consulted_for_selection") is not False:
        raise ValueError("hidden-goal panel selection consulted outcomes")

    source = contract["panel"]
    goals_path = resolve_repository(source["goals"])
    starts_path = resolve_repository(source["starts"])
    if sha256(goals_path) != source["goals_sha256"]:
        raise ValueError("hidden-goal source goals changed")
    if sha256(starts_path) != source["starts_sha256"]:
        raise ValueError("hidden-goal source starts changed")

    subset = task_set
    names = [str(value) for value in subset["goal_names"]]
    if len(names) != len(set(names)) or len(names) != int(subset["tasks"]):
        raise ValueError("hidden-goal panel names are not a unique task set")
    goals = pd.read_csv(goals_path)
    goals = goals[goals.goal_name.isin(names)]
    if set(goals.goal_name.astype(str)) != set(names):
        raise ValueError("hidden-goal panel references unknown goals")

    episodes = int(subset["episodes_per_goal"])
    starts = pd.read_csv(starts_path)
    starts = starts[
        starts.seed.astype(int).eq(int(source["environment_seed"]))
        & starts.goal_name.isin(names)
        & starts.episode.astype(int).lt(episodes)
    ]
    panel = starts.merge(goals, on="goal_name", how="left", validate="many_to_one")
    panel = panel.sort_values(["goal_name", "episode"]).reset_index(drop=True)
    if len(panel) != int(subset["rows"]):
        raise ValueError("hidden-goal panel row count changed")
    if not panel.groupby("goal_name").size().eq(episodes).all():
        raise ValueError("hidden-goal panel episode blocks are incomplete")
    expected_episodes = list(range(episodes))
    for _, frame in panel.groupby("goal_name"):
        if frame.episode.astype(int).tolist() != expected_episodes:
            raise ValueError("hidden-goal panel episode identities changed")
    return panel

