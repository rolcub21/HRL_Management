"""Reward-free preflight for the PointMaze command-region screen."""

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.shared.command_regions import (
    CommandRegionCover,
    collapsed_candidate_roadmap_links,
    movement_weighted_command_reach,
)

from .core import atomic_json, load_json, sha256
from .roadmap import RecordedRoadmap


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "command_region_protocol_r13.json"
REFERENCE = HERE / "R13_COMMAND_REGION_REFERENCE.md"
MANIFEST = HERE.parent / "shared" / "method_manifest_command_regions_r13.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roadmap", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = load_json(args.roadmap / "roadmap.json")
    graph_path = Path(metadata["graph"])
    if sha256(graph_path) != metadata["graph_sha256"]:
        raise ValueError("PointMaze recorded graph changed")
    reach_path = Path(metadata["reach_candidates"])
    if sha256(reach_path) != metadata["reach_candidates_sha256"]:
        raise ValueError("PointMaze reach candidates changed")
    bridge_metadata = load_json(args.planner / "bridges.json")
    if bridge_metadata["recorded_graph_sha256"] != metadata["graph_sha256"]:
        raise ValueError("PointMaze bridges use another recorded graph")
    bridge_path = Path(bridge_metadata["artifact"])
    if sha256(bridge_path) != bridge_metadata["artifact_sha256"]:
        raise ValueError("PointMaze bridge artifact changed")
    graph = RecordedRoadmap.load(graph_path)
    with np.load(reach_path, allow_pickle=False) as stored:
        targets = stored["target_index"].astype(np.int32)
    with np.load(bridge_path, allow_pickle=False) as stored:
        weight = (
            stored["weight"].astype(np.float32)
            if "weight" in stored.files
            else stored["member_weight"].min(axis=2).astype(np.float32)
        )
    start = graph.nodes[graph.source_index, :2]
    endpoint = graph.nodes[graph.endpoint_index, :2]
    reach = movement_weighted_command_reach(
        start, graph.commands, endpoint, float(metadata["landing_sigma_m"])
    )
    candidate, source, target, reliability = collapsed_candidate_roadmap_links(
        graph.nodes[:, :2],
        graph.source_index,
        graph.endpoint_index,
        graph.edge_reliability,
        targets,
        weight,
    )
    protocol = load_json(PROTOCOL)
    if not all(
        row.get("status") == "pass"
        for row in protocol["conformance_audit"]
    ):
        raise ValueError("command-region protocol has a failed conformance item")
    common = {
        "method_revision": "pointmaze_command_region_acquisition_r13",
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_manifest": str(MANIFEST),
        "method_manifest_sha256": sha256(MANIFEST),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256(PROTOCOL),
        "claim_level": protocol["claim_level"],
        "declared_deviations": protocol["declared_deviations"],
        "conformance_audit": protocol["conformance_audit"],
        "reward_used": False,
        "hidden_goal_used": False,
        "oracle_inputs": [],
        "recorded_graph": str(graph_path),
        "recorded_graph_sha256": metadata["graph_sha256"],
        "bridge_artifact": str(bridge_path),
        "bridge_artifact_sha256": bridge_metadata["artifact_sha256"],
        "command_region_candidate_count": int(len(candidate)),
        "command_region_cover_radius_m": reach,
    }
    regions = CommandRegionCover.build_roadmap(
        candidate,
        reach,
        source,
        target,
        reliability,
        alias_radius=float(metadata["landing_sigma_m"]),
        allow_unassigned=True,
    )
    region_diagnostic = regions.diagnostic()
    unassigned = int(
        region_diagnostic["command_region_unassigned_candidate_count"]
    )
    report = {
        **common,
        "conforms": True,
        "status": (
            "pass" if unassigned == 0
            else "pass_with_reported_roadmap_coverage_gap"
        ),
        "initial_uniform_unassigned_posterior_mass": float(
            unassigned / len(candidate)
        ),
        "euclidean_fallback_used": False,
        **region_diagnostic,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
