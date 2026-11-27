"""Compare rebuilt seed 0 with the frozen results and audit task-6 shortcut.

No numerical acceptance tolerance is invented here.  The report records exact
deltas and a conformance verdict; scientific equivalence remains a reported
comparison rather than a hidden pass/fail rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r74_reach_scale_route_correction as r74,
)
from projects.tfsn_consequence_value_iteration.shared.aligned_rebuild import (
    METHOD_REVISION,
    PROTOCOL_VALUE,
    SPECIFICATION,
    atomic_json,
    sha256,
)


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "projects/tfsn_consequence_value_iteration/results"


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _headlines(summary: dict) -> dict:
    names = (
        "tasks", "acquired_tasks", "acquisition_at_1", "acquisition_at_5",
        "acquisition_at_10", "acquisition_at_20",
        "pooled_conditional_return", "mean_e2e",
        "acquisition_new_1m_cells_per_command",
        "online_destination_arrival_rate",
    )
    return {name: summary.get(name) for name in names if name in summary}


def _delta(current: dict, rebuilt: dict) -> dict:
    output = {}
    for name in set(current) & set(rebuilt):
        left, right = current[name], rebuilt[name]
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            output[name] = float(right) - float(left)
    return output


def _pointmaze(args) -> dict:
    current_paths = [
        RESULTS / "pointmaze/frozen_r74_port_r15/original9/summary.json",
        RESULTS / "pointmaze/frozen_r74_port_r15/disjoint15/summary.json",
    ]
    rebuilt_paths = [
        args.rebuilt / "evaluation/original9/summary.json",
        args.rebuilt / "evaluation/disjoint15/summary.json",
    ]
    comparisons = []
    for panel, current_path, rebuilt_path in zip(
        ("original9", "disjoint15"), current_paths, rebuilt_paths
    ):
        current = _headlines(_json(current_path))
        rebuilt = _headlines(_json(rebuilt_path))
        comparisons.append({
            "panel": panel,
            "current_summary": str(current_path),
            "rebuilt_summary": str(rebuilt_path),
            "current": current,
            "rebuilt": rebuilt,
            "rebuilt_minus_current": _delta(current, rebuilt),
        })
    return {
        "domain": "pointmaze",
        "comparison": comparisons,
        "automatic_performance_tolerance": None,
        "automatic_performance_tolerance_reason": (
            "the frozen specification declares no numerical rebuild tolerance"
        ),
    }


def _bridge_family(preparation: Path, source_xy, target_xy) -> dict:
    runtime = preparation / "runtime"
    graph_path = runtime / "recorded_transition_graph.npz"
    bridge_path = runtime / "planner/bridges.npz"
    with np.load(graph_path, allow_pickle=False) as stored:
        nodes = stored["nodes"].astype(np.float64)
    with np.load(bridge_path, allow_pickle=False) as stored:
        targets = stored["target_index"].astype(np.int32)
        weight = stored["weight"].astype(np.float64)
    source_xy = np.asarray(source_xy, np.float64)
    target_xy = np.asarray(target_xy, np.float64)
    source_node = int(np.argmin(np.linalg.norm(nodes[:, :2] - source_xy, axis=1)))
    candidate_nodes = targets[source_node]
    candidate_distance = np.linalg.norm(nodes[candidate_nodes, :2] - target_xy, axis=1)
    column = int(np.argmin(candidate_distance))
    exact = {
        "source_node": source_node,
        "source_node_xy": nodes[source_node, :2].tolist(),
        "source_distance_m": float(np.linalg.norm(nodes[source_node, :2] - source_xy)),
        "target_node": int(candidate_nodes[column]),
        "target_node_xy": nodes[candidate_nodes[column], :2].tolist(),
        "target_distance_m": float(candidate_distance[column]),
        "bridge_column": column,
        "bridge_weight": float(weight[source_node, column]),
    }
    reach = r74.empirical_reach_covariance(preparation)
    inverse = np.asarray(reach["inverse_covariance_per_m2"], np.float64)
    source_delta = nodes[:, :2] - source_xy
    source_distance = np.einsum(
        "ni,ij,nj->n", source_delta, inverse, source_delta
    )
    kernel_mass = 0.0
    weighted_weight = 0.0
    maximum = (-1.0, None, None)
    for begin in range(0, len(nodes), 1024):
        stop = min(begin + 1024, len(nodes))
        target_delta = nodes[targets[begin:stop], :2] - target_xy
        target_distance = np.einsum(
            "nki,ij,nkj->nk", target_delta, inverse, target_delta
        )
        kernel = np.exp(-0.5 * (
            source_distance[begin:stop, None] + target_distance
        ))
        local_weight = weight[begin:stop]
        kernel_mass += float(kernel.sum())
        weighted_weight += float((kernel * local_weight).sum())
        contribution = kernel * local_weight
        flat = int(np.argmax(contribution))
        value = float(contribution.reshape(-1)[flat])
        if value > maximum[0]:
            row, column_index = np.unravel_index(flat, contribution.shape)
            maximum = (value, begin + int(row), int(column_index))
    max_value, max_source, max_column = maximum
    return {
        "preparation": str(preparation),
        "graph_sha256": sha256(graph_path),
        "bridges_sha256": sha256(bridge_path),
        "nearest_declared_shortcut": exact,
        "reach_covariance_m2": np.asarray(reach["covariance_m2"]).tolist(),
        "continuous_reach_kernel_mass": kernel_mass,
        "continuous_reach_kernel_weighted_bridge_mean": (
            weighted_weight / kernel_mass if kernel_mass else None
        ),
        "maximum_kernel_times_weight": {
            "value": max_value,
            "source_node": max_source,
            "target_node": int(targets[max_source, max_column]),
            "bridge_weight": float(weight[max_source, max_column]),
        },
        "similarity_cutoff": None,
    }


def _antmaze(args) -> dict:
    current_summary_path = (
        RESULTS / "antmaze/e549_5m_r74_twenty_episode_r75/original9/summary.json"
    )
    rebuilt_summary_path = args.rebuilt / "evaluation/original9/summary.json"
    current_summary = _headlines(_json(current_summary_path))
    rebuilt_summary = _headlines(_json(rebuilt_summary_path))
    shortcut = PROTOCOL_VALUE["antmaze"]["task6_false_shortcut"]
    current_preparation = (
        ROOT / PROTOCOL_VALUE["antmaze"]["current_preparation"]
    ).resolve()
    current = _bridge_family(
        current_preparation, shortcut["source_xy"], shortcut["target_xy"]
    )
    rebuilt = _bridge_family(
        args.rebuilt, shortcut["source_xy"], shortcut["target_xy"]
    )
    declared = float(shortcut["current_bridge_weight"])
    measured = float(current["nearest_declared_shortcut"]["bridge_weight"])
    if not np.isclose(declared, measured, rtol=0.0, atol=1e-7):
        raise ValueError("current task-6 shortcut no longer matches protocol")
    rebuilt_weight = float(rebuilt["nearest_declared_shortcut"]["bridge_weight"])
    return {
        "domain": "antmaze",
        "comparison": {
            "current_summary": str(current_summary_path),
            "rebuilt_summary": str(rebuilt_summary_path),
            "current": current_summary,
            "rebuilt": rebuilt_summary,
            "rebuilt_minus_current": _delta(current_summary, rebuilt_summary),
        },
        "task6_false_shortcut": {
            "declared_source_xy": shortcut["source_xy"],
            "declared_target_xy": shortcut["target_xy"],
            "current": current,
            "rebuilt": rebuilt,
            "nearest_bridge_weight_change": rebuilt_weight - measured,
            "nearest_bridge_weight_ratio": (
                rebuilt_weight / measured if measured else None
            ),
            "rebuilt_nearest_weight_is_lower": rebuilt_weight < measured,
            "automatic_high_rating_threshold": None,
            "interpretation": (
                "Exact weight, relative change, and continuous reach-family "
                "statistics are reported without inventing a cutoff."
            ),
        },
        "automatic_performance_tolerance": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("pointmaze", "antmaze"), required=True)
    parser.add_argument("--rebuilt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.rebuilt = args.rebuilt.resolve()
    domain = _pointmaze(args) if args.domain == "pointmaze" else _antmaze(args)
    result = {
        "schema_version": 1,
        "method_reference": str(SPECIFICATION),
        "method_reference_sha256": sha256(SPECIFICATION),
        "method_revision": METHOD_REVISION,
        "claim_level": "rebuild_seed0_comparison_audit",
        "declared_deviations": [],
        "oracle_inputs": [],
        "preparation_seed": 0,
        "method_contract_matches_frozen_specification": True,
        **domain,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
