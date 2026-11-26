"""R28: consumption-aware hidden-goal acquisition on frozen R22.

R27 supplies reward-free empirical passages for recorded and model bridge
edges.  R28 changes only the acquisition backup.  Passage line mass is
converted to a fraction of the residual reachable search mass with the
existing Gaussian kernel's exact transverse mass, sqrt(2 pi) sigma.  The
finite-horizon recursion then backs up first contact probability rather than
adding reusable search mass.  Values therefore remain in [0, 1].

Return is R27: frozen R22 plus exact positive online evidence candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    hidden_goal_empirical_passage_r27 as r27,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    hidden_goal_r22 as baseline,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    reach_based_r14 as r22,
)
from projects.tfsn_consequence_value_iteration.antmaze.mobility_stitched_planning import (
    MobilityStitchedPlanner,
)
from projects.tfsn_consequence_value_iteration.antmaze.successor_reductions import (
    value_coverage,
)
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths


METHOD_REVISION = "antmaze_hidden_goal_bounded_consumption_r28"
BASE_METHOD_REVISION = r22.METHOD_REVISION
CLAIM_LEVEL = "consumed_development_hidden_goal_bounded_search_test"
ALL_TASK_IDS = r27.ALL_TASK_IDS
GATE_TASK_IDS = r27.GATE_TASK_IDS
DEFAULT_EPISODES = r27.DEFAULT_EPISODES


def bounded_passage_values(
    graph,
    targets,
    weight,
    recorded_gain,
    bridge_gain,
    horizon=10,
):
    """Back up first-contact probability with failure staying at source."""
    recorded_gain = np.clip(
        np.asarray(recorded_gain, np.float32), 0.0, 1.0
    )
    bridge_gain = np.clip(
        np.asarray(bridge_gain, np.float32), 0.0, 1.0
    )
    values = np.zeros((int(horizon) + 1, graph.node_count), np.float32)
    for remaining in range(1, int(horizon) + 1):
        previous = values[remaining - 1]
        success = recorded_gain + (
            1.0 - recorded_gain
        ) * previous[graph.endpoint_index]
        reliability = np.asarray(graph.edge_reliability, np.float32)
        recorded_q = (
            reliability * success
            + (1.0 - reliability) * previous[graph.source_index]
        )
        row = previous.copy()
        np.maximum.at(row, graph.source_index, recorded_q)

        bridge_success = bridge_gain + (
            1.0 - bridge_gain
        ) * previous[targets]
        bridge_q = (
            weight * bridge_success
            + (1.0 - weight) * previous[:, None]
        )
        row = np.maximum(row, bridge_q.max(axis=1))
        values[remaining] = np.clip(row, 0.0, 1.0)
    return values, value_coverage(values)


class BoundedPassageGeometry(r27.EmpiricalPassageGeometry):
    """Normalize empirical line gain by reachable residual spatial mass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Primitive trajectories define the reachable spatial support.  Each
        # lattice cell contributes once to the denominator, regardless of
        # reward-free visitation density.
        self.reachable_cell_index = np.unique(self.recorded_cell_index)
        self.transverse_kernel_mass_m = float(
            np.sqrt(2.0 * np.pi) * float(r22.SIGMA)
        )
        self.residual_reachable_mass_m2 = None
        self.recorded_gain_maximum = None
        self.bridge_gain_maximum = None

    def _residual_mass(self, fields):
        residual = fields.search(
            self.grid_xy[self.reachable_cell_index]
        ).astype(np.float32)
        return float(
            residual.sum(dtype=np.float64) * float(r22.SIGMA) ** 2
        )

    def _normalize(self, line_gain):
        line_gain = np.asarray(line_gain, np.float32)
        denominator = float(self.residual_reachable_mass_m2 or 0.0)
        if denominator <= 0.0:
            return np.zeros_like(line_gain)
        swept_mass = self.transverse_kernel_mass_m * line_gain
        return np.clip(swept_mass / denominator, 0.0, 1.0).astype(
            np.float32
        )

    def graph_gains(self, fields):
        recorded, bridge = super().graph_gains(fields)
        self.residual_reachable_mass_m2 = self._residual_mass(fields)
        recorded = self._normalize(recorded)
        bridge = self._normalize(bridge)
        self.recorded_gain_maximum = float(recorded.max(initial=0.0))
        self.bridge_gain_maximum = float(bridge.max(initial=0.0))
        return recorded, bridge

    def live_segment_gain(self, start_xy, endpoint_xy, fields):
        return self._normalize(
            super().live_segment_gain(start_xy, endpoint_xy, fields)
        )


class BoundedConsumptionPlanner(r27.EmpiricalPassagePlanner):
    method_revision = METHOD_REVISION
    planner_name = "hidden_goal_bounded_consumption_r28"

    def _set_goal(self, fields):
        if fields.contacted:
            self.acquisition_active = False
            return baseline.HiddenGoalR22Planner._set_goal(self, fields)
        self.acquisition_active = True
        self.node_field = fields.search(self.graph.nodes[:, :2]).astype(
            np.float32
        )
        self.recorded_gain, self.bridge_gain = (
            self.passage_geometry.graph_gains(fields)
        )
        self.values, self.coverage = bounded_passage_values(
            self.graph,
            self.targets,
            self.weight,
            self.recorded_gain,
            self.bridge_gain,
            horizon=self.horizon,
        )
        self.goal_key = (
            "dynamic_acquisition_bounded_consumption",
            len(fields.coords),
        )

    def _candidate_value(
        self,
        state,
        commands,
        target_index,
        remaining,
        reliability,
        seed,
        return_members=False,
    ):
        if not self.acquisition_active:
            return MobilityStitchedPlanner._candidate_value(
                self,
                state,
                commands,
                target_index,
                remaining,
                reliability,
                seed,
                return_members=return_members,
            )
        (_unused, connection, _unused_members, member_connection) = (
            MobilityStitchedPlanner._candidate_value(
                self,
                state,
                commands,
                target_index,
                remaining,
                reliability,
                seed,
                return_members=True,
            )
        )
        gain = self.passage_geometry.live_segment_gain(
            state[:2], self.graph.nodes[target_index, :2], self.current_fields
        )
        previous = self.values[remaining - 1]
        success = gain + (1.0 - gain) * previous[target_index]
        source_index = int(self.graph.project(state[None])[0][0])
        failure = float(previous[source_index])
        reliability = np.asarray(reliability, np.float32).reshape(-1)
        member_probability = member_connection * reliability[:, None]
        member_q = (
            member_probability * success[:, None]
            + (1.0 - member_probability) * failure
        )
        q = member_q.min(axis=1)
        if return_members:
            return q, connection, member_q, member_connection
        return q, connection

    def _greedy_outgoing(self, node_index, remaining):
        if not self.acquisition_active:
            return super()._greedy_outgoing(node_index, remaining)
        node_index = np.asarray(node_index, np.int32).reshape(-1)
        previous = self.values[remaining - 1]
        model_target = self.targets[node_index]
        model_success = self.bridge_gain[node_index] + (
            1.0 - self.bridge_gain[node_index]
        ) * previous[model_target]
        model_q = (
            self.weight[node_index] * model_success
            + (1.0 - self.weight[node_index]) * previous[node_index, None]
        )
        model_choice = np.argmax(model_q, axis=1)
        row = np.arange(len(node_index))
        target = model_target[row, model_choice].astype(np.int32)
        command = self.graph.nodes[target, :2].copy()
        reliability = np.ones(len(node_index), np.float32)
        offline_score = model_q[row, model_choice].astype(np.float32)
        kind = np.full(
            len(node_index), "alias_next_model_bridge", dtype=object
        )
        for output_index, node in enumerate(node_index):
            begin = self.outgoing_offsets[node]
            stop = self.outgoing_offsets[node + 1]
            if begin == stop:
                continue
            edges = self.outgoing_order[begin:stop]
            endpoints = self.graph.endpoint_index[edges]
            edge_gain = self.recorded_gain[edges]
            success = edge_gain + (
                1.0 - edge_gain
            ) * previous[endpoints]
            edge_reliability = self.graph.edge_reliability[edges]
            recorded_q = (
                edge_reliability * success
                + (1.0 - edge_reliability) * previous[node]
            )
            local = int(np.argmax(recorded_q))
            if recorded_q[local] <= offline_score[output_index]:
                continue
            edge = int(edges[local])
            command[output_index] = self.graph.commands[edge]
            target[output_index] = self.graph.endpoint_index[edge]
            reliability[output_index] = self.graph.edge_reliability[edge]
            offline_score[output_index] = recorded_q[local]
            kind[output_index] = "alias_next_recorded_edge"
        return command, target, reliability, offline_score, kind

    def choose(self, state, command_index, fields, *args, **kwargs):
        command, diagnostic = super().choose(
            state, command_index, fields, *args, **kwargs
        )
        if not fields.contacted:
            diagnostic.update({
                "acquisition_backup": "bounded_consumption_aware_first_contact",
                "search_value_minimum": float(self.values.min()),
                "search_value_maximum": float(self.values.max()),
                "residual_reachable_mass_m2": float(
                    self.passage_geometry.residual_reachable_mass_m2
                ),
                "transverse_kernel_mass_m": float(
                    self.passage_geometry.transverse_kernel_mass_m
                ),
                "recorded_normalized_gain_maximum": float(
                    self.passage_geometry.recorded_gain_maximum
                ),
                "bridge_normalized_gain_maximum": float(
                    self.passage_geometry.bridge_gain_maximum
                ),
            })
        return command, diagnostic


def _load_runtime(args):
    graph = r22._verify_graph(args)
    planner, hashes = r22.lazy._load_planner(
        args, BoundedConsumptionPlanner
    )
    geometry = BoundedPassageGeometry(
        args.root, graph, planner.targets, planner.weight
    )
    planner.configure_hidden_fields(float(r22.SIGMA))
    planner.configure_empirical_passage(geometry)
    planner.artifact_hashes = {
        **hashes,
        "reach_landmarks_sha256": r27.sha256_file(
            r22.artifact_paths(args.root)["landmarks"]
        ),
        "empirical_passage_geometry_sha256": r27.sha256_file(
            r27.artifact_paths(args.root)["metadata"]
        ),
    }
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000_000)
    return planner, executor, env


def _task_ids(name):
    return GATE_TASK_IDS if name == "gate2" else ALL_TASK_IDS


def output_directory(root, name):
    return paths(root)["eval"] / name


def _evaluate_task(args, planner, executor, env, task_id):
    task = r22._task(args.root, task_id)
    output = output_directory(args.root, args.output_name)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task_id:03d}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION,
            int(args.episodes), float(r22.SIGMA), float(r22.SIGMA),
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("episodes"),
            saved.get("acquisition_search_sigma_m"),
            saved.get("return_field_sigma_m"),
        )
        if actual != expected:
            raise ValueError(
                f"existing task has another contract: {result_path}"
            )
        print(json.dumps(saved))
        return saved
    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    row = baseline.evaluate_hidden_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        int(args.episodes),
        r22.SIGMA,
        output / f"task_{task_id:03d}.jsonl",
    )
    row.update(task)
    row.update({
        "evaluation_arm": "hidden_goal_bounded_consumption_aware_search",
        "method_revision": METHOD_REVISION,
        "base_method_revision": BASE_METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "episodes": int(args.episodes),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "evaluator_uses_hidden_goal_for_contact_only": True,
        "task_memory_persisted_across_physical_resets": True,
        "acquisition_search_sigma_m": float(r22.SIGMA),
        "return_field_sigma_m": float(r22.SIGMA),
        "return_field_sigma_estimator": r22.ESTIMATOR,
        "acquisition_immediate_readout": "empirical_passage_integral",
        "acquisition_backup": "bounded_consumption_aware_first_contact",
        "return_planner": "R22_plus_exact_online_positive_evidence_candidates",
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def command_evaluate_set(args):
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in _task_ids(args.set_name):
            _evaluate_task(args, planner, executor, env, task_id)
    finally:
        env.close()
    command_summarize(args)


def command_summarize(args):
    task_ids = _task_ids(args.set_name)
    output = output_directory(args.root, args.output_name)
    rows = [
        json.loads((output / f"task_{task_id:03d}.json").read_text())
        for task_id in task_ids
    ]
    contact = [row["first_contact_episode"] for row in rows]
    successes = [value for row in rows for value in row["successes"]]
    returns = []
    for row in rows:
        first = row["first_contact_episode"]
        if first is not None:
            returns.extend(row["successes"][int(first) + 1:])
    result = {
        "schema_version": 1,
        "method_revision": METHOD_REVISION,
        "base_method_revision": BASE_METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "consumed_development_goals",
            "acquisition_uses_empirical_passage_integral",
            "acquisition_uses_bounded_consumption_aware_backup",
            "return_adds_exact_online_positive_evidence_candidates",
            "one_meter_coverage_cells_are_diagnostic_only",
        ],
        "oracle_inputs": [],
        "set_name": args.set_name,
        "task_ids": list(task_ids),
        "tasks": len(rows),
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(sum(value is not None for value in contact)),
        "acquisition_at_1": float(np.mean([value == 0 for value in contact])),
        "acquisition_at_5": float(np.mean([
            value is not None and value < 5 for value in contact
        ])),
        "pooled_conditional_return": (
            float(np.mean(returns)) if returns else None
        ),
        "mean_e2e": float(np.mean(successes)),
        "acquisition_new_1m_cells_per_command": float(np.mean([
            row["acquisition_new_1m_cells_per_command"] for row in rows
        ])),
        "acquisition_zero_new_1m_cell_fraction": float(np.mean([
            row["acquisition_zero_new_1m_cell_fraction"] for row in rows
        ])),
        "acquisition_revisited_endpoint_fraction": float(np.mean([
            row["acquisition_revisited_endpoint_fraction"] for row in rows
        ])),
        "acquisition_two_cycle_count": int(sum(
            row["acquisition_two_cycle_count"] for row in rows
        )),
        "acquisition_search_sigma_m": float(r22.SIGMA),
        "return_field_sigma_m": float(r22.SIGMA),
        "return_field_sigma_estimator": r22.ESTIMATOR,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "fresh_confirmation_panel_accessed": False,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        key: result[key]
        for key in (
            "tasks", "acquired_tasks", "acquisition_at_1",
            "acquisition_at_5", "pooled_conditional_return", "mean_e2e",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "acquisition_two_cycle_count",
        )
    }, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=605)
    commands = value.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate-set")
    evaluate.add_argument(
        "--set-name", choices=("gate2", "all15"), required=True
    )
    evaluate.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    evaluate.add_argument("--output-name", required=True)
    evaluate.set_defaults(func=command_evaluate_set)
    summarize = commands.add_parser("summarize")
    summarize.add_argument(
        "--set-name", choices=("gate2", "all15"), required=True
    )
    summarize.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    summarize.add_argument("--output-name", required=True)
    summarize.set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = r22.graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
