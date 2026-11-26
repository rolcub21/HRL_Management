#!/usr/bin/env python3
"""Offline preflight for reach-scale correction of R75 soft return values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r67_e549_hub_routed_exploration as r67,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r70_corrected_hub_reachability as r70,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r74_reach_scale_route_correction as r74,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r75_r74_twenty_episode as r75,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r76_return_route_correction as r76,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r77_online_map_growth as r77,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R78_SOFT_RETURN_ROUTE_AUDIT_REFERENCE.md"
PROTOCOL = HERE / "r78_soft_return_route_audit_protocol.json"
METHOD_REVISION = "antmaze_e549_5m_task6_soft_return_route_preflight_r78"
TASK_ID = 6
HORIZON = r75.HORIZON
R75_OUTPUT = r75.DEFAULT_OUTPUT
R76_OUTPUT = r76.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_task6_soft_return_route_preflight_r78"
)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "base_evaluation_revision": r75.METHOD_REVISION,
        "task_ids": [TASK_ID],
        "commands_per_episode": HORIZON,
        "fixed_command_duration": r48.COMMAND_STEPS,
        "return_route_prior_weight": float(RECORDED_EVIDENCE_PRIOR_WEIGHT),
        "return_route_similarity_cutoff": None,
        "route_source_scale": "R52_empirical_reach_covariance",
        "route_target_scale": "R52_empirical_reach_covariance",
        "wall_geometry_role": "diagnostic_only_not_value_or_policy_input",
        "environment_replay_requires_all_four_offline_gates": True,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R78 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R78 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def multiplicative_route_correction(
    base_values: np.ndarray,
    node_xy: np.ndarray,
    target_xy: np.ndarray,
    records: list[dict],
    inverse_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preserve the additive value scale and apply route reliability only."""
    base = np.asarray(base_values, np.float32)
    reliability, support, arrivals = r77.route_reliability_layers(
        node_xy, target_xy, records, inverse_covariance, len(base) - 1,
    )
    return base * reliability, reliability, support, arrivals


def soft_return_values(
    node_field, source, endpoint, edge_weight, bridge_target, bridge_weight,
    horizon,
) -> np.ndarray:
    """The unchanged R75 additive max backup over recorded and bridge edges."""
    field = np.asarray(node_field, np.float32).reshape(-1)
    values = np.zeros((int(horizon) + 1, len(field)), np.float32)
    for remaining in range(1, int(horizon) + 1):
        row = np.zeros(len(field), np.float32)
        recorded_q = edge_weight * (
            field[endpoint] + values[remaining - 1, endpoint]
        )
        np.maximum.at(row, source, recorded_q)
        bridge_q = bridge_weight * (
            field[bridge_target] + values[remaining - 1, bridge_target]
        )
        values[remaining] = np.maximum(row, bridge_q.max(axis=1))
    return values


def greedy_graph_rollout(
    start_xy, nodes, source, endpoint, edge_weight, bridge_target,
    bridge_weight, node_field, values, horizon,
) -> list[dict]:
    """Roll out the exact edge argmax used by the frozen graph backup."""
    tree = cKDTree(np.asarray(nodes[:, :2], np.float64))
    current = int(tree.query(np.asarray(start_xy, np.float64), k=1)[1])
    path = []
    for remaining in range(int(horizon), 0, -1):
        recorded_index = np.flatnonzero(source == current)
        recorded_q = edge_weight[recorded_index] * (
            node_field[endpoint[recorded_index]]
            + values[remaining - 1, endpoint[recorded_index]]
        )
        bridge_q = bridge_weight[current] * (
            node_field[bridge_target[current]]
            + values[remaining - 1, bridge_target[current]]
        )
        if len(recorded_q) and float(recorded_q.max()) >= float(bridge_q.max()):
            edge = int(recorded_index[int(np.argmax(recorded_q))])
            destination = int(endpoint[edge])
            kind = "recorded_edge"
            score = float(recorded_q.max())
        else:
            edge = int(np.argmax(bridge_q))
            destination = int(bridge_target[current, edge])
            kind = "model_bridge"
            score = float(bridge_q[edge])
        start = np.asarray(nodes[current, :2], np.float64)
        stop = np.asarray(nodes[destination, :2], np.float64)
        path.append({
            "remaining_commands": int(remaining),
            "source_node": int(current),
            "destination_node": int(destination),
            "source_xy": start.tolist(),
            "destination_xy": stop.tolist(),
            "kind": kind,
            "score": score,
            "crosses_wall_diagnostic": bool(
                kind == "model_bridge"
                and r70.segment_crosses_wall(start, stop)
            ),
        })
        current = destination
    return path


def _failed_episode_records(
    return_rows: list[dict], target_xy: np.ndarray, count: int,
) -> list[dict]:
    episode_ids = sorted({int(row["episode"]) for row in return_rows})[:count]
    return [
        {
            "source_xy": row["start_xy"],
            "target_xy": np.asarray(target_xy, np.float64).tolist(),
            "remaining_commands": int(row["remaining"]),
            "arrived": False,
        }
        for row in return_rows
        if int(row["episode"]) in episode_ids
    ]


def command_audit(args) -> dict:
    arrays = r67._graph_arrays(Path(args.preparation).resolve())
    (
        paths, consequence, nodes, source, endpoint, edge_weight,
        bridge_target, bridge_weight,
    ) = arrays
    r75_log = R75_OUTPUT / f"task_{TASK_ID:03d}.jsonl"
    r76_log = R76_OUTPUT / f"task_{TASK_ID:03d}.jsonl"
    if not r75_log.is_file() or not r76_log.is_file():
        raise FileNotFoundError("R75 task-6 and consumed R76 logs are required")
    return_rows = [
        row for row in _rows(r75_log) if row.get("phase") == "return"
    ]
    observed_rows = [
        row for row in _rows(r76_log)
        if row.get("phase") == "return" and "return_target_xy" in row
    ]
    if not return_rows or not observed_rows:
        raise RuntimeError("task-6 return evidence is absent")
    target = np.asarray(observed_rows[0]["return_target_xy"], np.float64)
    reset = np.asarray(return_rows[0]["start_xy"], np.float64)
    sigma = float(consequence["landing_sigma_m"])

    # Structural preflight field: the same deployed soft evidence kernel,
    # centred only on actually observed positive evidence.  It is deliberately
    # independent of the hidden goal coordinate and of maze geometry.
    node_field = np.exp(
        -0.5 * np.square(
            np.linalg.norm(nodes[:, :2] - target[None], axis=1) / sigma
        )
    ).astype(np.float32)
    base = soft_return_values(
        node_field, source, endpoint, edge_weight, bridge_target,
        bridge_weight, HORIZON,
    )
    reach = r74.empirical_reach_covariance(args.preparation)
    inverse = np.linalg.inv(
        np.asarray(reach["covariance_m2"], np.float64)
    )
    stages = []
    corrected_by_stage = []
    for failed_episodes in (0, 1, 2):
        records = _failed_episode_records(
            return_rows, target, failed_episodes
        )
        corrected, reliability, support, arrivals = (
            multiplicative_route_correction(
                base, nodes[:, :2], target, records, inverse
            )
        )
        path = greedy_graph_rollout(
            reset, nodes, source, endpoint, edge_weight, bridge_target,
            bridge_weight, node_field, corrected, HORIZON,
        )
        corrected_by_stage.append(corrected)
        stages.append({
            "failed_return_episodes_used": int(failed_episodes),
            "route_records": int(len(records)),
            "value_maximum": float(corrected.max()),
            "positive_value_fraction": float(np.mean(corrected > 0.0)),
            "reliability_minimum": float(reliability.min()),
            "support_maximum": float(support.max()),
            "arrival_mass_sum": float(arrivals.sum(dtype=np.float64)),
            "wall_crossing_edges_in_greedy_rollout": int(sum(
                row["crosses_wall_diagnostic"] for row in path
            )),
            "greedy_rollout": path,
        })

    identity = bool(np.array_equal(base, corrected_by_stage[0]))
    lowered = bool(
        np.any(corrected_by_stage[1] < base)
        and np.all(corrected_by_stage[1] <= base)
        and np.any(corrected_by_stage[2] < corrected_by_stage[1])
        and np.all(corrected_by_stage[2] <= corrected_by_stage[1])
    )
    alternatives_positive = bool(
        stages[1]["positive_value_fraction"] > 0.0
        and stages[2]["positive_value_fraction"] > 0.0
    )
    base_crossings = stages[0]["wall_crossing_edges_in_greedy_rollout"]
    after_one = stages[1]["wall_crossing_edges_in_greedy_rollout"]
    after_two = stages[2]["wall_crossing_edges_in_greedy_rollout"]
    switches_to_non_crossing = bool(
        base_crossings > 0 and (after_one == 0 or after_two == 0)
    )
    gates = {
        "no_evidence_is_exact_R75_identity": identity,
        "failed_route_lowers_value_smoothly_without_clipping": lowered,
        "alternative_routes_remain_positive": alternatives_positive,
        "one_or_two_failures_switch_best_rollout_off_wall_shortcuts": (
            switches_to_non_crossing
        ),
    }
    ready = bool(all(gates.values()))
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_id": TASK_ID,
        "base_preparation_revision": r52.METHOD_REVISION,
        "base_evaluation_revision": r75.METHOD_REVISION,
        "graph": str(paths["graph"]),
        "graph_sha256": r52.sha256(paths["graph"]),
        "bridges": str(paths["bridges"]),
        "bridges_sha256": r52.sha256(paths["bridges"]),
        "source_R75_log": str(r75_log),
        "source_R75_log_sha256": r52.sha256(r75_log),
        "observed_evidence_source_R76_log": str(r76_log),
        "observed_evidence_source_R76_log_sha256": r52.sha256(r76_log),
        "observed_evidence_xy": target.tolist(),
        "reset_xy": reset.tolist(),
        "landing_sigma_m": sigma,
        "route_reach_covariance_m2": reach["covariance_m2"].tolist(),
        "return_value_semantics": (
            "R75_additive_soft_field_preserved_not_Bernoulli_and_not_clipped"
        ),
        "route_reliability_semantics": (
            "one_success_prior_plus_reach_kernel_weighted_route_outcomes"
        ),
        "wall_geometry_role": "diagnostic_only_not_value_or_policy_input",
        "stages": stages,
        "offline_gates": gates,
        "ready_for_environment_replay": ready,
        "next_action": (
            "environment_replay_is_permitted"
            if ready else
            "do_not_replay_route_correction_does_not_remove_false_shortcut_family"
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    r48.dump(output / "audit.json", report)
    print(json.dumps(report, indent=2))
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preparation", type=Path, default=r52.DEFAULT_OUTPUT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
