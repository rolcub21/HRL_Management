"""Reward-free preflight for the R14 observation-memory comparison."""

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.shared.latent_detection import (
    LatentRadiusPosterior,
    SigmaKernelPosterior,
)

from .core import atomic_json, load_json, sha256
from .panel import load_development_panel
from .roadmap import RecordedRoadmap


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "sigma_latent5_goal4_protocol_r14.json"
REFERENCE = HERE / "R14_SIGMA_LATENT5_GOAL4_REFERENCE.md"
MANIFEST = HERE / "method_manifest_factorial_r12.json"
EXPECTED_ARMS = (
    "sigma_kernel__nearest_recorded",
    "latent_radius__nearest_recorded",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--roadmap", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    screen = load_json(PROTOCOL)
    if tuple(screen["factorial_arms"]) != EXPECTED_ARMS:
        raise ValueError("R14 arm panel changed")
    if not all(
        row.get("status") == "pass" for row in screen["conformance_audit"]
    ):
        raise ValueError("R14 protocol contains a failed conformance item")

    road = load_json(args.roadmap / "roadmap.json")
    graph_path = Path(road["graph"])
    if sha256(graph_path) != road["graph_sha256"]:
        raise ValueError("R14 recorded graph changed")
    bridges = load_json(args.planner / "bridges.json")
    if bridges["recorded_graph_sha256"] != road["graph_sha256"]:
        raise ValueError("R14 bridge cache uses another graph")
    bridge_path = Path(bridges["artifact"])
    if sha256(bridge_path) != bridges["artifact_sha256"]:
        raise ValueError("R14 bridge artifact changed")
    models = load_json(args.models / "factorized_consequence_metrics.json")
    if bridges["factorized_consequence_sha256"] != models["artifact_sha256"]:
        raise ValueError("R14 bridge cache uses another consequence model")

    graph = RecordedRoadmap.load(graph_path)
    sigma = float(road["landing_sigma_m"])
    fixed = SigmaKernelPosterior(graph.nodes[:, :2], sigma)
    latent = LatentRadiusPosterior(graph.nodes[:, :2], sigma)
    if not np.array_equal(fixed.goal_xy, latent.goal_xy):
        raise ValueError("R14 arms do not share exact candidate support")

    panel = load_development_panel()
    goals = panel.groupby("goal_name", sort=True)[
        ["goal_x", "goal_y"]
    ].first().reset_index()
    corner = tuple(screen["corner_goal_names"])
    remaining = goals[~goals.goal_name.isin(corner)].copy()
    support_mean = fixed.goal_xy.mean(axis=0)
    remaining["support_mean_distance"] = np.linalg.norm(
        remaining[["goal_x", "goal_y"]].to_numpy(dtype=np.float64)
        - support_mean[None],
        axis=1,
    )
    selected_ordinary = tuple(
        remaining.sort_values(
            ["support_mean_distance", "goal_name"], kind="stable"
        ).head(2).goal_name.astype(str)
    )
    if selected_ordinary != tuple(screen["ordinary_goal_names"]):
        raise ValueError("R14 outcome-free ordinary-goal selection changed")
    if set(screen["goal_names"]) != set(corner + selected_ordinary):
        raise ValueError("R14 four-goal panel changed")

    report = {
        "schema_version": 1,
        "conforms": True,
        "status": "pass",
        "method_revision": screen["method_revision"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_manifest": str(MANIFEST),
        "method_manifest_sha256": sha256(MANIFEST),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "claim_level": screen["claim_level"],
        "declared_deviations": screen["declared_deviations"],
        "conformance_audit": screen["conformance_audit"],
        "oracle_inputs": [],
        "reward_used": False,
        "hidden_goal_used": False,
        "arms": list(EXPECTED_ARMS),
        "goals": screen["goal_names"],
        "corner_goals": list(corner),
        "ordinary_goals": list(selected_ordinary),
        "ordinary_goal_selection_rule": screen[
            "ordinary_goal_selection_rule"
        ],
        "roadmap_support_mean_xy": support_mean.tolist(),
        "candidate_goal_count": int(len(fixed.goal_xy)),
        "landing_sigma_m": sigma,
        "latent_rho_max_m": latent.rho_max,
        "episodes_per_goal": int(screen["episodes_per_goal"]),
        "evaluation_command_cap": int(screen["evaluation_command_cap"]),
        "benchmark_horizon_commands": int(
            screen["benchmark_horizon_commands"]
        ),
        "recorded_graph_sha256": road["graph_sha256"],
        "bridge_artifact_sha256": bridges["artifact_sha256"],
        "factorized_consequence_sha256": models["artifact_sha256"],
        "command_regions_loaded": False,
        "learned_passage_loaded": False,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
