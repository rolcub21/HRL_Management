"""R24: generalized policy improvement over directed successor maps.

R23 described one uniform random behavior and failed on detours.  R24 keeps
the frozen R22 graph, edges, consequence model, field, live candidate set,
argmax, and alias partition.  It changes only the behavior-policy family used
for the offline future:

* 128 exact reward-free experience anchors reuse the existing global
  candidate-computation budget.
* Each anchor defines one fixed, task-independent directed behavior.
* Each behavior chooses the edge with the largest model-weighted reduction in
  squared distance to its anchor.
* At task time every successor map is read against the supplied-goal field and
  generalized policy improvement takes their pointwise maximum.

As in the isolated R22/R23 comparison, edge readout is the frozen implemented
term ``w * (F(destination) + V(destination))``.  No failure self-loop is added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    averaged_successor_r23 as r23,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    reach_based_r14 as r22,
)
from projects.tfsn_consequence_value_iteration.antmaze.directed_successor_gpi import (
    directed_policy_edges,
    gpi_successor_values,
    spread_anchor_indices,
)
from scripts.antmaze.current_tfsn.collect import (
    SuppliedGoalField,
    evaluate_supplied_goal_task,
)
from scripts.antmaze.current_tfsn.constants import EPISODE_STEPS
from scripts.antmaze.current_tfsn.executor import (
    FaramaSACExecutor,
    full_state,
    make_env,
)
from scripts.antmaze.current_tfsn.main import dump, paths


METHOD_REVISION = "antmaze_directed_successor_gpi_r24"
CLAIM_LEVEL = "expanded_consumed_development_directed_successor_gpi_test"
ALL_TASK_IDS = tuple(range(15))
HORIZON = 10
# Reuse an existing global computation budget; do not add an Ant-specific knob.
DIRECTED_BEHAVIORS = int(r22.BRIDGE_CANDIDATES)
IMPLEMENTED_EDGE_TERM = r23.IMPLEMENTED_EDGE_TERM


def policy_artifact_paths(root):
    directory = Path(root) / "model_planning" / "directed_successor_gpi_r24"
    return {
        "directory": directory,
        "policies": directory / "directed_policies.npz",
        "metadata": directory / "directed_policies.json",
    }


def output_directory(root):
    return (
        r22.expanded_output_paths(root)["directory"]
        / "directed_successor_gpi_r24"
    )


def contrast_path(root):
    return output_directory(root) / "reset_value_contrast.json"


def _r22_edge_arrays(root, graph):
    artifact = r22.artifact_paths(root)
    for name in ("landmarks", "bridges"):
        if not artifact[name].exists():
            raise FileNotFoundError(
                f"missing frozen R22 {name}; run reach-based-r14-prepare"
            )
    with np.load(artifact["landmarks"]) as data:
        landmark_index = data["landmark_index"].astype(np.int32)
        expected_targets = data["target_index"].astype(np.int32)
    with np.load(artifact["bridges"]) as data:
        saved_landmarks = data["landmark_index"].astype(np.int32)
        targets = data["target_index"].astype(np.int32)
        weight = data["weight"].astype(np.float32)
    if not np.array_equal(landmark_index, saved_landmarks):
        raise ValueError("R22 bridge and landmark indices differ")
    if not np.array_equal(expected_targets, targets):
        raise ValueError("R22 bridge and candidate targets differ")
    if targets.shape != (graph.node_count, r22.BRIDGE_CANDIDATES):
        raise ValueError("R22 bridge array has another shape")
    return targets, weight


def _policy_contract(args, graph):
    source = r22.artifact_paths(args.root)
    return {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "base_graph_sha256": r22.sha256_file(args.graph),
        "r22_landmarks_sha256": r22.sha256_file(source["landmarks"]),
        "r22_bridges_sha256": r22.sha256_file(source["bridges"]),
        "nodes": int(graph.node_count),
        "directed_behaviors": int(DIRECTED_BEHAVIORS),
        "anchor_rule": (
            "deterministic_farthest_first_exact_reward_free_positions"
        ),
        "behavior_rule": (
            "argmax_model_weighted_reduction_in_squared_distance_to_anchor"
        ),
        "behavior_task_independent": True,
        "directed_behavior_budget_source": (
            "existing_global_bridge_candidate_budget"
        ),
        "distance_thresholds": [],
        "task_reward_used": False,
        "hidden_goals_accessed": False,
        "oracle_inputs": [],
        "implemented_edge_term": IMPLEMENTED_EDGE_TERM,
        "failure_self_loop_added": False,
    }


def command_prepare(args):
    graph = r22._verify_graph(args)
    targets, weight = _r22_edge_arrays(args.root, graph)
    artifact = policy_artifact_paths(args.root)
    artifact["directory"].mkdir(parents=True, exist_ok=True)
    contract = _policy_contract(args, graph)
    if artifact["policies"].exists() and artifact["metadata"].exists():
        metadata = json.loads(artifact["metadata"].read_text())
        for key, expected in contract.items():
            if metadata.get(key) != expected:
                raise ValueError(f"saved R24 policies have another {key}")
        if metadata.get("artifact_sha256") != r22.sha256_file(
            artifact["policies"]
        ):
            raise ValueError("R24 policy artifact hash mismatch")
        print(json.dumps(metadata, indent=2))
        return

    anchor_index = spread_anchor_indices(
        graph.nodes[:, :2], DIRECTED_BEHAVIORS
    )
    policy_target, policy_weight, policy_kind, policy_score = (
        directed_policy_edges(
            graph, targets, weight, anchor_index, node_chunk=256
        )
    )
    r22.mobility._save_npz_atomic(
        artifact["policies"],
        anchor_index=anchor_index,
        policy_target=policy_target,
        policy_weight=policy_weight,
        policy_kind=policy_kind,
    )
    metadata = {
        **contract,
        "artifact": str(artifact["policies"].resolve()),
        "artifact_sha256": r22.sha256_file(artifact["policies"]),
        "anchor_xy_minimum": graph.nodes[anchor_index, :2].min(axis=0).tolist(),
        "anchor_xy_maximum": graph.nodes[anchor_index, :2].max(axis=0).tolist(),
        "recorded_edge_choice_fraction": float(np.mean(policy_kind == 1)),
        "selected_edge_weight_mean": float(policy_weight.mean()),
        "selected_edge_weight_median": float(np.median(policy_weight)),
        "selected_directed_progress_mean": float(policy_score.mean()),
        "selected_directed_progress_median": float(np.median(policy_score)),
    }
    dump(artifact["metadata"], metadata)
    print(json.dumps(metadata, indent=2))


def _load_policy_artifact(args, graph):
    artifact = policy_artifact_paths(args.root)
    if not artifact["policies"].exists() or not artifact["metadata"].exists():
        raise FileNotFoundError("run gpi-successor-r24-prepare first")
    metadata = json.loads(artifact["metadata"].read_text())
    for key, expected in _policy_contract(args, graph).items():
        if metadata.get(key) != expected:
            raise ValueError(f"saved R24 policies have another {key}")
    if metadata.get("artifact_sha256") != r22.sha256_file(
        artifact["policies"]
    ):
        raise ValueError("R24 policy artifact hash mismatch")
    with np.load(artifact["policies"]) as data:
        anchor_index = data["anchor_index"].astype(np.int32)
        policy_target = data["policy_target"].astype(np.int32)
        policy_weight = data["policy_weight"].astype(np.float32)
        policy_kind = data["policy_kind"].astype(np.uint8)
    expected_shape = (DIRECTED_BEHAVIORS, graph.node_count)
    for name, value in (
        ("policy_target", policy_target),
        ("policy_weight", policy_weight),
        ("policy_kind", policy_kind),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} has shape {value.shape}, expected {expected_shape}")
    return artifact, metadata, anchor_index, policy_target, policy_weight


class DirectedGpiSuccessorPlanner(r22.ReachBasedR14Planner):
    method_revision = METHOD_REVISION
    planner_name = "directed_successor_gpi"

    def attach_policy_atlas(
        self, anchor_index, policy_target, policy_weight
    ):
        self.policy_anchor_index = np.asarray(anchor_index, np.int32)
        self.policy_target = np.asarray(policy_target, np.int32)
        self.policy_weight = np.asarray(policy_weight, np.float32)

    def _set_goal(self, fields):
        key = tuple(np.asarray(fields.goal, np.float32))
        if key == self.goal_key:
            return
        if not hasattr(self, "policy_target"):
            raise RuntimeError("directed successor policy atlas is not attached")
        goal = np.asarray(fields.goal, np.float32)
        self.node_field = np.exp(
            -np.square(self.graph.nodes[:, :2] - goal[None]).sum(axis=1)
            / (2.0 * r22.SIGMA**2)
        ).astype(np.float32)
        (
            self.values,
            self.winning_policy,
            self.coverage,
        ) = gpi_successor_values(
            self.policy_target,
            self.policy_weight,
            self.node_field,
            self.horizon,
        )
        self.goal_key = key


def _load_planner(args):
    graph = r22._verify_graph(args)
    artifact, metadata, anchors, policy_target, policy_weight = (
        _load_policy_artifact(args, graph)
    )
    planner, hashes = r22.lazy._load_planner(
        args, DirectedGpiSuccessorPlanner
    )
    planner.attach_policy_atlas(anchors, policy_target, policy_weight)
    planner.artifact_hashes = {
        **hashes,
        "reach_landmarks_sha256": r22.sha256_file(
            r22.artifact_paths(args.root)["landmarks"]
        ),
        "directed_policy_atlas_sha256": r22.sha256_file(
            artifact["policies"]
        ),
    }
    planner.policy_metadata = metadata
    return planner


def _load_runtime(args):
    planner = _load_planner(args)
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000)
    return planner, executor, env


def _evaluate_task(args, planner, executor, env, task_id):
    task = r22._task(args.root, task_id)
    output = output_directory(args.root)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task_id:03d}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION, r22.SIGMA
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("field_sigma_m"),
        )
        if actual != expected:
            raise ValueError(f"existing task has another contract: {result_path}")
        print(json.dumps(saved))
        return saved

    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    row = evaluate_supplied_goal_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        r22.SIGMA,
        max_steps=EPISODE_STEPS,
        log_path=output / f"task_{task_id:03d}.jsonl",
    )
    row.update(task)
    row.update({
        "evaluation_arm": "directed_successor_gpi",
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "field_sigma_m": r22.SIGMA,
        "future_policy_family": "reward_free_landmark_directed",
        "directed_behaviors": DIRECTED_BEHAVIORS,
        "future_policy_reduction": "gpi_max_over_fixed_successor_maps",
        "live_decision_reduction": "argmax_unchanged",
        "alias_advance_reduction": "argmax_unchanged",
        "implemented_edge_term": IMPLEMENTED_EDGE_TERM,
        "failure_self_loop_added": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "value_coverage": planner.coverage,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def command_task(args):
    planner, executor, env = _load_runtime(args)
    try:
        _evaluate_task(args, planner, executor, env, args.task_id)
    finally:
        env.close()


def command_reset_contrast(args):
    planner = _load_planner(args)
    env = make_env(max_episode_steps=10_000)
    rows = []
    try:
        for task_id in ALL_TASK_IDS:
            task = r22._task(args.root, task_id)
            task_seed = int(args.seed) + 1_000_003 * int(task_id)
            planner.rng = np.random.default_rng(task_seed)
            planner.seed = task_seed
            planner.goal_key = None
            observation, _ = env.reset(seed=int(task["reset_seed"]))
            state = full_state(observation)
            fields = SuppliedGoalField(r22.SIGMA, task["goal"])
            command, diagnostic = planner.choose(state, 0, fields)
            rows.append({
                "task_id": int(task_id),
                "arm": "r24_directed_successor_gpi",
                "candidate_score_best": diagnostic["candidate_score_best"],
                "candidate_score_median": diagnostic["candidate_score_median"],
                "best_to_median_ratio": diagnostic[
                    "candidate_score_best_to_median_ratio"
                ],
                "median_is_zero": diagnostic[
                    "candidate_score_median_is_zero"
                ],
                "selected_command": np.asarray(command).tolist(),
            })
    finally:
        env.close()
    prior = json.loads(r23.contrast_path(args.root).read_text())
    result = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "contrast_point": "physical_reset_before_first_command",
        "candidate_set_identical": True,
        "live_argmax_identical": True,
        "r22": prior["r22"],
        "r23": prior["r23"],
        "r24": rows,
    }
    output_directory(args.root).mkdir(parents=True, exist_ok=True)
    dump(contrast_path(args.root), result)
    print(json.dumps({
        "artifact": str(contrast_path(args.root).resolve()),
        "r24_tasks": len(rows),
    }, indent=2))


def command_summarize(args):
    output = output_directory(args.root)
    rows = []
    for task_id in ALL_TASK_IDS:
        path = output / f"task_{task_id:03d}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text()))
    if not contrast_path(args.root).exists():
        command_reset_contrast(args)
    contrast = json.loads(contrast_path(args.root).read_text())
    direct, oracle = r23._reference_rows(args.root)
    r22_rows = json.loads((
        r22.output_directory(args.root) / "summary.json"
    ).read_text())["rows"]
    r23_rows = json.loads((
        r23.output_directory(args.root) / "summary.json"
    ).read_text())["rows"]
    detour_ids = [
        task_id for task_id in ALL_TASK_IDS
        if not direct[task_id]["success"] and oracle[task_id]["success"]
    ]
    result = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "consumed_development_goals",
            "fixed_reward_free_directed_behavior_family",
            "gpi_across_precomputed_successor_readouts",
            "failure_self_loop_omitted_to_match_frozen_r22_and_r23",
        ],
        "oracle_inputs": [],
        "controlled_change_vs_r23": (
            "uniform_behavior_to_reward_free_directed_behavior_family_gpi"
        ),
        "candidate_set_identical": True,
        "live_argmax_identical": True,
        "alias_advance_argmax_identical": True,
        "directed_behaviors": DIRECTED_BEHAVIORS,
        "implemented_edge_term": IMPLEMENTED_EDGE_TERM,
        "failure_self_loop_added": False,
        "all_15": {
            "tasks": 15,
            "r24_successes": int(sum(row["success"] for row in rows)),
            "r23_successes": int(sum(row["success"] for row in r23_rows)),
            "r22_successes": int(sum(row["success"] for row in r22_rows)),
            "direct_successes": int(sum(row["success"] for row in direct)),
            "oracle_successes": int(sum(row["success"] for row in oracle)),
        },
        "detour": {
            "definition": "direct_executor_failure_and_oracle_success",
            "task_ids": list(detour_ids),
            "tasks": len(detour_ids),
            "r24_successes": int(sum(rows[i]["success"] for i in detour_ids)),
            "r23_successes": int(sum(r23_rows[i]["success"] for i in detour_ids)),
            "r22_successes": int(sum(r22_rows[i]["success"] for i in detour_ids)),
            "direct_successes": 0,
            "oracle_successes": len(detour_ids),
        },
        "reset_value_contrast": {
            "r24_all15": r23._contrast_summary(contrast["r24"], ALL_TASK_IDS),
            "r23_all15": r23._contrast_summary(contrast["r23"], ALL_TASK_IDS),
            "r22_all15": r23._contrast_summary(contrast["r22"], ALL_TASK_IDS),
            "r24_detour": r23._contrast_summary(contrast["r24"], detour_ids),
            "r23_detour": r23._contrast_summary(contrast["r23"], detour_ids),
            "r22_detour": r23._contrast_summary(contrast["r22"], detour_ids),
        },
        "per_goal": [
            {
                "task_id": int(task_id),
                "detour_goal": bool(task_id in detour_ids),
                "r24_success": bool(rows[task_id]["success"]),
                "r23_success": bool(r23_rows[task_id]["success"]),
                "r22_success": bool(r22_rows[task_id]["success"]),
                "direct_success": bool(direct[task_id]["success"]),
                "oracle_success": bool(oracle[task_id]["success"]),
            }
            for task_id in ALL_TASK_IDS
        ],
        "task_reward_used": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        "all_15": result["all_15"],
        "detour": result["detour"],
        "reset_value_contrast": result["reset_value_contrast"],
    }, indent=2))


def command_evaluate_all(args):
    if not contrast_path(args.root).exists():
        command_reset_contrast(args)
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in ALL_TASK_IDS:
            _evaluate_task(args, planner, executor, env, task_id)
    finally:
        env.close()
    command_summarize(args)


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=605)
    subparsers = value.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare").set_defaults(func=command_prepare)
    task = subparsers.add_parser("task")
    task.add_argument("--task-id", type=int, choices=ALL_TASK_IDS, required=True)
    task.set_defaults(func=command_task)
    subparsers.add_parser("reset-contrast").set_defaults(
        func=command_reset_contrast
    )
    subparsers.add_parser("evaluate-all15").set_defaults(
        func=command_evaluate_all
    )
    subparsers.add_parser("summarize").set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = r22.graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
