"""Build the immutable recorded road map and empirical passage cache."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .core import atomic_json, atomic_npz, load_json, protocol, sha256
from .dataset import load_current_aligned_dataset, load_initial_dataset
from .passages import prepare_passages
from .roadmap import graph_from_dataset, reach_targets
from projects.tfsn_consequence_value_iteration.shared.settings import (
    ARTIFACT_NODE_SHARD,
    BRIDGE_CANDIDATES,
)


METHOD_REVISION = "pointmaze_initial_recorded_roadmap_r1"
ALIGNED_METHOD_REVISION = "pointmaze_current_aligned_recorded_roadmap_r4"
MOVEMENT_REACH_METHOD_REVISION = "pointmaze_movement_conditioned_reach_roadmap_r7"
REFERENCE_METHOD_REVISION = "pointmaze_continuous_reference_roadmap_r11"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--alignment", type=Path, default=None,
        help="Include the completed current-planner alignment round.",
    )
    parser.add_argument(
        "--continuous-reach-weighting",
        action="store_true",
        help=(
            "Build the empirical reach measure with continuous requested and "
            "realized advance weights; no transition is thresholded."
        ),
    )
    parser.add_argument("--movement-conditioned-reach", action="store_true")
    args = parser.parse_args()
    if args.movement_conditioned_reach:
        raise ValueError(
            "the fixed sigma-multiple reach mask was removed; use "
            "--continuous-reach-weighting"
        )
    started = time.time()
    contract = protocol()
    metrics = load_json(args.models / "factorized_consequence_metrics.json")
    checkpoint = Path(metrics["artifact"])
    if sha256(checkpoint) != metrics["artifact_sha256"]:
        raise ValueError("factorized consequence checkpoint changed")
    if args.schedule == "full" and metrics["schedule"] != "full":
        raise ValueError("full roadmap preparation requires a full consequence model")
    sigma = float(metrics["kernel_sigma_m"])
    max_groups = 3 if args.schedule == "smoke" else None
    dataset = (
        load_current_aligned_dataset(
            args.alignment, max_groups_per_source=max_groups
        )
        if args.alignment is not None
        else load_initial_dataset(max_groups_per_source=max_groups)
    )
    graph = graph_from_dataset(dataset, sigma)
    bridge_count = int(BRIDGE_CANDIDATES)
    landmarks, targets, distance, offsets = reach_targets(
        graph,
        bridge_count,
        sigma=sigma,
        continuous_weighting=args.continuous_reach_weighting,
    )
    output = args.output.resolve()
    graph_path = output / "recorded_graph.npz"
    reach_path = output / "reach_candidates.npz"
    graph.save(graph_path)
    atomic_npz(
        reach_path,
        landmark_index=landmarks,
        target_index=targets,
        target_distance=distance,
        reach_offsets=offsets,
    )
    passage = prepare_passages(
        graph, targets, sigma, output / "passages",
        shard_nodes=ARTIFACT_NODE_SHARD,
    )
    displacement = (
        graph.nodes[graph.endpoint_index, :2]
        - graph.nodes[graph.source_index, :2]
    )
    report = {
        "schema_version": 1,
        "artifact_type": "pointmaze_recorded_roadmap",
        "method_reference": contract["method_reference"],
        "method_revision": (
            REFERENCE_METHOD_REVISION
            if args.continuous_reach_weighting
            else (
                ALIGNED_METHOD_REVISION
                if args.alignment is not None else METHOD_REVISION
            )
        ),
        "claim_level": (
            "smoke_only" if args.schedule == "smoke"
            else (
                "reward_free_current_aligned_roadmap_preparation_only"
                if args.alignment is not None
                else "reward_free_initial_roadmap_preparation_only"
            )
        ),
        "declared_deviations": [],
        "oracle_inputs": [],
        "schedule": args.schedule,
        "factorized_consequence_schedule": metrics["schedule"],
        "dataset_rows": dataset.rows,
        "dataset_sources": list(dataset.source_names),
        "alignment_input": (
            str(args.alignment.resolve())
            if args.alignment is not None else None
        ),
        "graph": str(graph_path),
        "graph_sha256": sha256(graph_path),
        "reach_candidates": str(reach_path),
        "reach_candidates_sha256": sha256(reach_path),
        "factorized_consequence_sha256": metrics["artifact_sha256"],
        "landing_sigma_m": sigma,
        "recorded_nodes": graph.node_count,
        "recorded_edges": graph.edge_count,
        "recorded_nodes_deleted": False,
        "recorded_edges_deleted": False,
        "landmarks": int(len(landmarks)),
        "landmarks_are_exact_recorded_endpoints": True,
        "bridge_candidates_per_node": int(targets.shape[1]),
        "continuous_reach_weighting": bool(
            args.continuous_reach_weighting
        ),
        "movement_request_threshold": None,
        "reach_offset_minimum_m": float(
            np.linalg.norm(offsets, axis=1).min()
        ),
        "bridge_candidate_rule": (
            "continuous_requested_advance_times_realized_advance_weighted_"
            "empirical_displacement_measure"
            if args.continuous_reach_weighting
            else "uniform_empirical_displacement_measure"
        ),
        "recorded_reliability_estimator": (
            "shared_kernel_posterior_of_realized_delta_repeatability"
        ),
        "recorded_edge_reliability_mean": float(
            graph.edge_reliability.mean()
        ),
        "empirical_displacement_median_m": float(
            np.median(np.linalg.norm(displacement, axis=1))
        ),
        "empirical_displacement_p90_m": float(
            np.quantile(np.linalg.norm(displacement, axis=1), 0.9)
        ),
        "passage_geometry": passage,
        "task_reward_used": False,
        "hidden_goals_used": False,
        "wall_clock_seconds": time.time() - started,
    }
    atomic_json(output / "roadmap.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
