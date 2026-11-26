"""R32: consume the first imagined passage before evaluating the VI tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from projects.tfsn_consequence_value_iteration.shared.settings import (
    CONSUMPTIVE_LOOKAHEAD_BEAM_WIDTH,
    CONSUMPTIVE_LOOKAHEAD_DEPTH,
)

from projects.tfsn_consequence_value_iteration.antmaze import (
    r31_aligned_hidden_goal as r31,
)
from scripts.antmaze.current_tfsn.constants import CANDIDATES, STATE_DIM
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths


HERE = Path(__file__).resolve().parent
METHOD_REFERENCE = HERE / "R32_CONSUMPTIVE_LOOKAHEAD_REFERENCE.md"
PROTOCOL = HERE / "r32_consumptive_lookahead_protocol.json"
METHOD_REVISION = "antmaze_consumptive_lookahead_r32"
LOOKAHEAD_DEPTH = CONSUMPTIVE_LOOKAHEAD_DEPTH
LOOKAHEAD_BEAM_WIDTH = CONSUMPTIVE_LOOKAHEAD_BEAM_WIDTH
RESULT_DIRECTORY = (
    r31.MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "consumptive_lookahead_r32"
)


def load_contract() -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_method_revision": r31.METHOD_REVISION,
        "oracle_inputs": [],
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "gate_task_ids": [3, 5],
        "episodes_per_task": 5,
        "lookahead_depth_global": LOOKAHEAD_DEPTH,
        "lookahead_beam_width_global": LOOKAHEAD_BEAM_WIDTH,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"R32 protocol has another {key}")
    reference = Path(protocol["method_reference"])
    if not reference.is_absolute():
        reference = r31.MANAGEMENT_ROOT / reference
    if reference.resolve() != METHOD_REFERENCE:
        raise ValueError("R32 method reference changed")
    return protocol


PROTOCOL_VALUE = load_contract()


class ConsumedSearchField:
    """Multiplicatively remove kernel mass around imagined passages."""

    def __init__(self, base, passages, sigma):
        self.base = base
        self.sigma = float(sigma)
        points = [np.asarray(row, np.float32).reshape(-1, 2) for row in passages]
        self.points = np.concatenate(points, axis=0)
        self.tree = cKDTree(self.points)

    def search(self, xy):
        xy = np.asarray(xy, np.float32).reshape(-1, 2)
        residual = np.asarray(self.base.search(xy), np.float32)
        distance = self.tree.query(xy, k=1, workers=-1)[0]
        kernel = np.exp(
            -np.square(distance) / (2.0 * self.sigma**2)
        ).astype(np.float32)
        return residual * (1.0 - kernel)


def matched_passage_samples(geometry, start_xy, endpoint_xy):
    endpoint = np.asarray(endpoint_xy, np.float32).reshape(-1, 2)
    start = np.broadcast_to(
        np.asarray(start_xy, np.float32).reshape(1, 2), endpoint.shape
    )
    displacement = endpoint - start
    query = np.concatenate((start, displacement), axis=1) / float(r31.SIGMA)
    _distance, edge = geometry.tree.query(query, k=1, workers=-1)
    edge = np.asarray(edge, np.int32).reshape(-1)
    return start[:, None] + geometry.relative[edge]


class ConsumptiveLookaheadPlanner(r31.AlignedBoundedConsumptionPlanner):
    method_revision = METHOD_REVISION
    planner_name = "hidden_goal_consumptive_lookahead_r32"

    def _consumed_tail(self, fields, passage, remaining, target_index):
        if remaining <= 1:
            return 0.0
        consumed = ConsumedSearchField(fields, [passage], r31.SIGMA)
        recorded_gain, bridge_gain = self.passage_geometry.graph_gains(consumed)
        values, _coverage = r31.r28.bounded_passage_values(
            self.graph,
            self.targets,
            self.weight,
            recorded_gain,
            bridge_gain,
            horizon=remaining - 1,
        )
        return float(values[remaining - 1, int(target_index)])

    def choose(
        self,
        state,
        command_index,
        fields,
        online_states=(),
        online_trajectories=(),
    ):
        if fields.contacted:
            return super().choose(
                state,
                command_index,
                fields,
                online_states,
                online_trajectories,
            )
        del online_states, online_trajectories
        self.current_fields = fields
        self._set_goal(fields)
        remaining = self.horizon - int(command_index)
        if remaining <= 0:
            raise ValueError("command index exceeds planning horizon")
        state = np.asarray(state, np.float32).reshape(STATE_DIM)

        distance, local = self.landmark_tree.query(
            state[:2],
            k=min(r31.lazy.LANDMARK_NEIGHBORS, len(self.landmark_index)),
        )
        distance = np.atleast_1d(distance).astype(np.float32)
        bridge_target = self.landmark_index[np.atleast_1d(local)]
        bridge_command = self.graph.nodes[bridge_target, :2]
        association = r31.lazy.alias_kernel(distance)
        bridge_reliability, advance_association = self._association_weights(
            association
        )
        (
            bridge_q,
            bridge_weight,
            _bridge_member_q,
            bridge_member_connection,
        ) = self._candidate_value(
            state,
            bridge_command,
            bridge_target,
            remaining,
            bridge_reliability,
            self.seed + command_index * 10_007,
            return_members=True,
        )

        recorded_command, recorded_edge, proposal = self.graph.sample_commands(
            self.rng, state, CANDIDATES
        )
        recorded_target = self.graph.endpoint_index[recorded_edge]
        recorded_reliability = self.graph.edge_reliability[recorded_edge]
        (
            recorded_q,
            recorded_weight,
            _recorded_member_q,
            recorded_member_connection,
        ) = self._candidate_value(
            state,
            recorded_command,
            recorded_target,
            remaining,
            recorded_reliability,
            self.seed + command_index * 10_007 + 97,
            return_members=True,
        )

        (
            advance_command,
            advance_target,
            advance_reliability,
            advance_offline_score,
            advance_kind,
        ) = self._greedy_outgoing(bridge_target, remaining)
        advance_partition = advance_reliability * advance_association
        (
            advance_q,
            advance_weight,
            _advance_member_q,
            advance_member_connection,
        ) = self._candidate_value(
            state,
            advance_command,
            advance_target,
            remaining,
            advance_partition,
            self.seed + command_index * 10_007 + 193,
            return_members=True,
        )

        command = np.concatenate(
            (bridge_command, recorded_command, advance_command), axis=0
        )
        target = np.concatenate(
            (bridge_target, recorded_target, advance_target), axis=0
        ).astype(np.int32)
        baseline_score = np.concatenate(
            (bridge_q, recorded_q, advance_q), axis=0
        )
        connection = np.concatenate(
            (bridge_weight, recorded_weight, advance_weight), axis=0
        )
        member_probability = np.concatenate((
            bridge_member_connection * bridge_reliability[:, None],
            recorded_member_connection * recorded_reliability[:, None],
            advance_member_connection * advance_partition[:, None],
        ), axis=0)
        kind = np.concatenate((
            np.full(len(bridge_command), "model_bridge", dtype=object),
            np.full(len(recorded_command), "recorded_edge", dtype=object),
            advance_kind,
        ))
        association_all = np.concatenate((
            np.zeros(len(bridge_command) + len(recorded_command), np.float32),
            association,
        ))
        partition_all = np.concatenate((
            bridge_reliability,
            np.ones(len(recorded_command), np.float32),
            advance_association,
        ))
        offline_all = np.concatenate((
            np.zeros(len(bridge_command) + len(recorded_command), np.float32),
            advance_offline_score,
        ))

        beam_count = min(LOOKAHEAD_BEAM_WIDTH, len(baseline_score))
        beam = np.argpartition(baseline_score, -beam_count)[-beam_count:]
        beam = beam[np.argsort(baseline_score[beam])[::-1]]
        first_gain = self.passage_geometry.live_segment_gain(
            state[:2], self.graph.nodes[target[beam], :2], fields
        )
        passages = matched_passage_samples(
            self.passage_geometry,
            state[:2],
            self.graph.nodes[target[beam], :2],
        )
        source_index = int(self.graph.project(state[None])[0][0])
        failure = float(self.values[remaining - 1, source_index])
        original_geometry = (
            self.passage_geometry.residual_reachable_mass_m2,
            self.passage_geometry.recorded_gain_maximum,
            self.passage_geometry.bridge_gain_maximum,
        )
        lookahead_score = np.full(len(baseline_score), -np.inf, np.float32)
        consumed_tail = np.zeros(beam_count, np.float32)
        try:
            for local_index, candidate_index in enumerate(beam):
                tail = self._consumed_tail(
                    fields,
                    passages[local_index],
                    remaining,
                    int(target[candidate_index]),
                )
                consumed_tail[local_index] = tail
                success = float(first_gain[local_index]) + (
                    1.0 - float(first_gain[local_index])
                ) * tail
                member_q = (
                    member_probability[candidate_index] * success
                    + (1.0 - member_probability[candidate_index]) * failure
                )
                lookahead_score[candidate_index] = float(member_q.min())
        finally:
            (
                self.passage_geometry.residual_reachable_mass_m2,
                self.passage_geometry.recorded_gain_maximum,
                self.passage_geometry.bridge_gain_maximum,
            ) = original_geometry

        winner = int(np.argmax(lookahead_score))
        baseline_winner = int(np.argmax(baseline_score))
        finite_scores = lookahead_score[beam]
        diagnostic = {
            "planner": self.planner_name,
            "method_revision": self.method_revision,
            "phase": "acquisition",
            "remaining": int(remaining),
            "selected_score": float(lookahead_score[winner]),
            "selected_baseline_score": float(baseline_score[winner]),
            "baseline_selected_kind": str(kind[baseline_winner]),
            "baseline_selected_command_distance": float(
                np.linalg.norm(command[baseline_winner] - state[:2])
            ),
            "selected_kind": str(kind[winner]),
            "selected_connection_weight": float(connection[winner]),
            "selected_command_distance": float(
                np.linalg.norm(command[winner] - state[:2])
            ),
            "selected_alias_association": float(association_all[winner]),
            "selected_execution_partition_weight": float(partition_all[winner]),
            "selected_alias_offline_score": float(offline_all[winner]),
            "selected_first_passage_gain": float(
                first_gain[int(np.flatnonzero(beam == winner)[0])]
            ),
            "selected_consumed_tail_value": float(
                consumed_tail[int(np.flatnonzero(beam == winner)[0])]
            ),
            "consumptive_lookahead_depth_global": LOOKAHEAD_DEPTH,
            "consumptive_lookahead_beam_width_global": LOOKAHEAD_BEAM_WIDTH,
            "lookahead_changed_baseline_choice": bool(winner != baseline_winner),
            "lookahead_beam_indices": beam.tolist(),
            "lookahead_beam_baseline_scores": baseline_score[beam].tolist(),
            "lookahead_beam_consumed_scores": finite_scores.tolist(),
            "alias_advance_candidates": int(len(advance_command)),
            "maximum_alias_association": float(association.max()),
            "positive_value_fraction": self.coverage[-1]["positive_fraction"],
            "meaningful_value_fraction": self.coverage[-1]["meaningful_fraction"],
            "top_scores": np.sort(finite_scores)[-5:][::-1].tolist(),
            "candidate_score_best": float(lookahead_score[winner]),
            "candidate_score_median": float(np.median(finite_scores)),
            "candidate_score_best_to_median_ratio": (
                float(lookahead_score[winner] / np.median(finite_scores))
                if np.median(finite_scores) > 0 else None
            ),
            "candidate_score_median_is_zero": bool(np.median(finite_scores) == 0),
            "search_enabled": True,
            "acquisition_search_mode": "fine",
            "fine_search_sigma": float(fields.sigma),
            "coarse_search_sigma": float(fields.sigma),
            "coarse_search_weight": 0.0,
            "fine_unsearched_mass": None,
            "coarse_unsearched_mass": None,
            "acquisition_immediate_readout": "empirical_reward_free_passage_integral",
            "recorded_passage_source": "exact_reward_free_primitive_trajectory",
            "bridge_passage_source": "locally_matched_reward_free_realized_trajectory",
            "bridge_passage_match_features": "start_xy_and_realized_displacement",
            "bridge_passage_bins_computational_only_global": r31.r27.PASSAGE_BINS,
            "return_online_positive_evidence_candidates": 0,
            "acquisition_backup": "two_command_consumptive_lookahead_plus_bounded_tail",
            "search_value_minimum": float(self.values.min()),
            "search_value_maximum": float(self.values.max()),
            "residual_reachable_mass_m2": float(original_geometry[0]),
            "transverse_kernel_mass_m": float(
                self.passage_geometry.transverse_kernel_mass_m
            ),
            "recorded_normalized_gain_maximum": float(original_geometry[1]),
            "bridge_normalized_gain_maximum": float(original_geometry[2]),
        }
        diagnostic.update(proposal)
        return command[winner].copy(), diagnostic


def _load_runtime(args):
    graph = r31.verify_graph(args.graph)
    planner, hashes = r31.lazy._load_planner(
        args, ConsumptiveLookaheadPlanner
    )
    geometry = r31.r28.BoundedPassageGeometry(
        args.root, graph, planner.targets, planner.weight
    )
    planner.configure_hidden_fields(r31.SIGMA)
    planner.configure_empirical_passage(geometry)
    planner.artifact_hashes = {
        **hashes,
        "reach_landmarks_sha256": r31.recorded.sha256_file(
            r31.artifact_paths()["landmarks"]
        ),
        "empirical_passage_geometry_sha256": r31.recorded.sha256_file(
            r31.passage_paths()["metadata"]
        ),
        "r30_summary_sha256": r31.sha256(r31.R30_SUMMARY),
    }
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000_000)
    return planner, executor, env


def evaluation_directory():
    return RESULT_DIRECTORY / "evaluation_gate2"


def _evaluate_task(args, planner, executor, env, task_id):
    task = r31.r22._task(args.root, task_id)
    output = evaluation_directory()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"task_{task_id:03d}.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION,
            int(args.episodes), LOOKAHEAD_DEPTH, LOOKAHEAD_BEAM_WIDTH,
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("episodes"),
            saved.get("lookahead_depth_global"),
            saved.get("lookahead_beam_width_global"),
        )
        if actual != expected:
            raise ValueError(f"existing R32 task has another contract: {result_path}")
        print(json.dumps(saved))
        return saved

    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.goal_key = None
    row = r31.r28.baseline.evaluate_hidden_task(
        env,
        executor,
        planner,
        np.asarray(task["goal"], np.float32),
        task["reset_seed"],
        int(args.episodes),
        r31.SIGMA,
        output / f"task_{task_id:03d}.jsonl",
    )
    row.update(task)
    row.update({
        "evaluation_arm": "R31_plus_two_command_consumptive_lookahead",
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r31.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "episodes": int(args.episodes),
        "lookahead_depth_global": LOOKAHEAD_DEPTH,
        "lookahead_beam_width_global": LOOKAHEAD_BEAM_WIDTH,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "evaluator_uses_hidden_goal_for_contact_only": True,
        "task_memory_persisted_across_physical_resets": True,
        "landing_sigma_m": float(r31.SIGMA),
        "acquisition_search_sigma_m": float(r31.SIGMA),
        "return_field_sigma_m": float(r31.SIGMA),
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def command_evaluate(args):
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in PROTOCOL_VALUE["gate_task_ids"]:
            _evaluate_task(args, planner, executor, env, int(task_id))
    finally:
        env.close()
    command_summarize(args)


def command_summarize(args):
    output = evaluation_directory()
    rows = [
        json.loads((output / f"task_{int(task_id):03d}.json").read_text())
        for task_id in PROTOCOL_VALUE["gate_task_ids"]
    ]
    contact = [row["first_contact_episode"] for row in rows]
    successes = [bool(value) for row in rows for value in row["successes"]]
    returns = []
    for row in rows:
        first = row["first_contact_episode"]
        if first is not None:
            returns.extend(row["successes"][int(first) + 1:])
    result = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r31.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_ids": PROTOCOL_VALUE["gate_task_ids"],
        "tasks": len(rows),
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(sum(value is not None for value in contact)),
        "acquisition_at_5": float(np.mean([
            value is not None and value < 5 for value in contact
        ])),
        "pooled_conditional_return": float(np.mean(returns)) if returns else None,
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
        "lookahead_depth_global": LOOKAHEAD_DEPTH,
        "lookahead_beam_width_global": LOOKAHEAD_BEAM_WIDTH,
        "landing_sigma_m": float(r31.SIGMA),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        key: result[key] for key in (
            "tasks", "acquired_tasks", "acquisition_at_5",
            "pooled_conditional_return", "mean_e2e",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "acquisition_two_cycle_count",
        )
    }, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377"))
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=605)
    commands = value.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate-gate2")
    evaluate.add_argument("--episodes", type=int, default=5)
    evaluate.set_defaults(func=command_evaluate)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--episodes", type=int, default=5)
    summarize.set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = r31.graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
