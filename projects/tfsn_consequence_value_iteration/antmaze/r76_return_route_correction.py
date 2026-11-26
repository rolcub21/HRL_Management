#!/usr/bin/env python3
"""Focused R75 task-6 replay with route-level return correction."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r67_e549_hub_routed_exploration as r67,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r70_corrected_hub_reachability as r70,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r71_sigma_generalized_correction as r71,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r72_hub_route_correction as r72,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r74_reach_scale_route_correction as r74,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r75_r74_twenty_episode as r75,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R76_RETURN_ROUTE_CORRECTION_REFERENCE.md"
PROTOCOL = HERE / "r76_return_route_correction_protocol.json"
METHOD_REVISION = "antmaze_e549_5m_task6_return_route_correction_r76"
TASK_ID = 6
HORIZON = r75.HORIZON
EPISODES = r75.EPISODES
R75_OUTPUT = r75.DEFAULT_OUTPUT
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_task6_return_route_correction_r76"
)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "base_evaluation_revision": r75.METHOD_REVISION,
        "task_ids": [TASK_ID],
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": HORIZON * r48.COMMAND_STEPS,
        "fixed_command_duration": r48.COMMAND_STEPS,
        "evaluation_seed": 607,
        "return_target": "first_observed_positive_evidence_coordinate",
        "return_route_similarity_cutoff": None,
        "return_route_prior_weight": float(RECORDED_EVIDENCE_PRIOR_WEIGHT),
        "route_source_scale": "R52_empirical_reach_covariance",
        "route_target_scale": "R52_empirical_reach_covariance",
        "acquisition_change_from_R75": "none",
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R76 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R76 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def corrected_route_layers(
    base_layers: np.ndarray,
    node_xy: np.ndarray,
    target_xy: np.ndarray,
    records: list[dict],
    inverse_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correct each horizon-conditioned state-to-evidence probability."""
    base = np.asarray(base_layers, np.float32)
    nodes = np.asarray(node_xy, np.float64).reshape(-1, 2)
    target = np.asarray(target_xy, np.float64).reshape(2)
    inverse = np.asarray(inverse_covariance, np.float64).reshape(2, 2)
    corrected = base.copy()
    support = np.zeros_like(base)
    arrival_mass = np.zeros_like(base)
    for remaining in range(len(base)):
        matching = [
            record for record in records
            if int(record["remaining_commands"]) == int(remaining)
        ]
        if not matching:
            continue
        observed_source = np.asarray(
            [record["source_xy"] for record in matching], np.float64
        )
        observed_target = np.asarray(
            [record["target_xy"] for record in matching], np.float64
        )
        outcome = np.asarray(
            [record["arrived"] for record in matching], np.float64
        )
        source_delta = nodes[None] - observed_source[:, None]
        target_delta = target[None] - observed_target
        source_distance = np.einsum(
            "rni,ij,rnj->rn", source_delta, inverse, source_delta
        )
        target_distance = np.einsum(
            "ri,ij,rj->r", target_delta, inverse, target_delta
        )
        similarity = np.exp(
            -0.5 * (source_distance + target_distance[:, None])
        )
        support[remaining] = similarity.sum(axis=0).astype(np.float32)
        arrival_mass[remaining] = (
            similarity * outcome[:, None]
        ).sum(axis=0).astype(np.float32)
        corrected[remaining] = r72.correct_hub_route_probability(
            base[remaining], support[remaining], arrival_mass[remaining]
        ).astype(np.float32)
    return corrected, support, arrival_mass


def has_observed_positive_evidence(fields) -> bool:
    """Distinguish task contact from a supplied routing target.

    Acquisition routes to hubs through ``R56SuppliedGoalField``.  That field
    intentionally exposes the supplied-goal interface (including
    ``contacted``) while keeping ``rewards`` empty because no task contact was
    observed.  Return-route correction must therefore be gated by recorded
    positive evidence, not by the routing interface's phase flag.
    """
    rewards = np.asarray(getattr(fields, "rewards", ()), np.float32).reshape(-1)
    return bool(rewards.size and np.any(rewards > 0.0))


class ReturnRouteCorrectedPlanner(
    r71.SigmaGeneralizedDestinationArrivalPlanner
):
    """R74 acquisition plus empirical-reach correction of return routes."""

    method_revision = METHOD_REVISION
    planner_name = "E549_R74_with_empirical_reach_return_route_correction"

    def configure_return_route(self, arrays, reach_covariance) -> None:
        (
            _paths, _consequence, self.return_nodes, self.return_source,
            self.return_endpoint, self.return_edge_weight,
            self.return_bridge_target, self.return_bridge_weight,
        ) = arrays
        self.return_node_tree = cKDTree(self.return_nodes[:, :2])
        self.return_inverse_covariance = np.linalg.inv(
            np.asarray(reach_covariance, np.float64).reshape(2, 2)
        )
        self.return_route_records = []
        self.return_route_attempts = 0
        self.return_route_arrivals = 0
        self.return_current_prefixes = []
        self.return_episode_active = False
        self.return_episode_positive_count = 0
        self.return_target_xy = None
        self.return_route_values = None
        self.return_route_base = None
        self.return_route_support = None
        self.return_route_arrival_mass = None
        self.return_route_decision_active = False

    def reset_online_correction_task(self):
        super().reset_online_correction_task()
        if hasattr(self, "return_route_records"):
            self.return_route_records.clear()
            self.return_route_attempts = 0
            self.return_route_arrivals = 0
            self.return_current_prefixes = []
            self.return_episode_active = False
            self.return_episode_positive_count = 0
            self.return_target_xy = None
            self.return_route_decision_active = False

    @staticmethod
    def _first_positive_xy(fields) -> np.ndarray:
        rewards = np.asarray(getattr(fields, "rewards", ()), np.float32)
        positive = np.flatnonzero(rewards > 0.0)
        if not len(positive):
            raise RuntimeError("return phase has no observed positive evidence")
        coords = np.asarray(getattr(fields, "coords", ()), np.float32).reshape(-1, 2)
        if len(coords) != len(rewards):
            raise RuntimeError("return evidence coordinates and rewards disagree")
        return np.asarray(coords[int(positive[0])], np.float32).reshape(2)

    def _finish_return_episode(self, arrived: bool) -> None:
        if not self.return_episode_active:
            return
        for source_xy, remaining in self.return_current_prefixes:
            self.return_route_records.append({
                "source_xy": np.asarray(source_xy, np.float32).tolist(),
                "target_xy": np.asarray(
                    self.return_target_xy, np.float32
                ).tolist(),
                "remaining_commands": int(remaining),
                "arrived": bool(arrived),
            })
        self.return_route_attempts += 1
        self.return_route_arrivals += int(bool(arrived))
        self.return_current_prefixes = []
        self.return_episode_active = False

    def finalize_return_episode(self, arrived: bool) -> None:
        self._finish_return_episode(bool(arrived))

    def _begin_return_decision(self, command_index, fields, online_states):
        if int(command_index) == 0:
            if self.return_episode_active:
                self._finish_return_episode(
                    int(np.count_nonzero(np.asarray(fields.rewards) > 0.0))
                    > int(self.return_episode_positive_count)
                )
            self.return_episode_active = True
            self.return_current_prefixes = []
            self.return_episode_positive_count = int(
                np.count_nonzero(np.asarray(fields.rewards) > 0.0)
            )
        target = self._first_positive_xy(fields)
        if self.return_target_xy is None:
            self.return_target_xy = target.copy()
        elif not np.array_equal(target, self.return_target_xy):
            raise RuntimeError("first observed return evidence changed")

        scorer = getattr(self, "hub_score_override", None)
        if scorer is None or not hasattr(scorer, "_synchronize"):
            raise RuntimeError("R76 requires R74 corrected link evidence")
        scorer._synchronize(online_states)
        corrected_bridge = r70.corrected_bridge_weights(
            self.return_bridge_weight, scorer.support, scorer.outcome_mass
        )
        target_node = int(self.return_node_tree.query(target, k=1)[1])
        base = r67._one_target_layers(
            target_node,
            int(self.horizon),
            len(self.return_nodes),
            self.return_source,
            self.return_endpoint,
            self.return_edge_weight,
            self.return_bridge_target,
            corrected_bridge,
        )
        corrected, support, arrival_mass = corrected_route_layers(
            base,
            self.return_nodes[:, :2],
            target,
            self.return_route_records,
            self.return_inverse_covariance,
        )
        self.return_route_base = base
        self.return_route_values = corrected
        self.return_route_support = support
        self.return_route_arrival_mass = arrival_mass
        self.return_route_decision_active = True

    def _candidate_value(
        self, state, commands, target_index, remaining, reliability, seed,
        return_members=False,
    ):
        if not self.return_route_decision_active:
            return super()._candidate_value(
                state, commands, target_index, remaining, reliability, seed,
                return_members=return_members,
            )
        (
            _old_q, connection, _old_member_q, corrected_connection,
        ) = super()._candidate_value(
            state, commands, target_index, remaining, reliability, seed,
            return_members=True,
        )
        target = np.asarray(target_index, np.int32).reshape(-1)
        reliability = np.asarray(reliability, np.float32).reshape(-1)
        continuation = self.return_route_values[
            max(int(remaining) - 1, 0), target
        ]
        member_q = (
            corrected_connection
            * reliability[:, None]
            * continuation[:, None]
        )
        q = member_q.min(axis=1)
        if return_members:
            return q, connection, member_q, corrected_connection
        return q, connection

    def _positive_evidence_value(
        self, state, commands, fields, remaining, seed,
    ):
        if not self.return_route_decision_active:
            return super()._positive_evidence_value(
                state, commands, fields, remaining, seed
            )
        del seed
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        source = np.repeat(
            np.asarray(state, np.float32)[None], len(commands), axis=0
        )
        member_connection = r48.exact_member_connection(
            self.consequence, self.mobility, source, commands, commands
        )
        support, outcome = self._online_link_mass(
            state, commands, commands
        )
        corrected_connection = r48.bayesian_local_connection(
            member_connection, support, outcome
        )
        nearest = self.return_node_tree.query(commands, k=1)[1]
        immediate = np.clip(fields.reward_rate(commands), 0.0, 1.0)
        future = self.return_route_values[
            max(int(remaining) - 1, 0), np.asarray(nearest, np.int32)
        ]
        continuation = immediate + (1.0 - immediate) * future
        return (
            (corrected_connection * continuation[:, None]).min(axis=1),
            corrected_connection.min(axis=1),
        )

    def choose(
        self, state, command_index, fields, online_states=(),
        online_trajectories=(),
    ):
        observed_contact = has_observed_positive_evidence(fields)
        if observed_contact:
            self._begin_return_decision(
                command_index, fields, online_states
            )
        else:
            self.return_route_decision_active = False
        command, diagnostic = super().choose(
            state, command_index, fields, online_states, online_trajectories
        )
        if observed_contact:
            remaining = int(self.horizon) - int(command_index)
            self.return_current_prefixes.append((
                np.asarray(state, np.float32)[:2].copy(), remaining
            ))
            current_node = int(
                self.return_node_tree.query(np.asarray(state)[:2], k=1)[1]
            )
            diagnostic.update({
                "return_route_correction": "empirical_reach_online_posterior",
                "return_target_source": "first_observed_positive_evidence",
                "return_target_xy": self.return_target_xy.astype(float).tolist(),
                "return_route_prior_at_current_state": float(
                    self.return_route_base[remaining, current_node]
                ),
                "return_route_corrected_at_current_state": float(
                    self.return_route_values[remaining, current_node]
                ),
                "return_route_support_at_current_state": float(
                    self.return_route_support[remaining, current_node]
                ),
                "return_route_records_available": int(
                    len(self.return_route_records)
                ),
                "return_route_similarity_cutoff": None,
            })
        return command, diagnostic

    def return_route_summary(self) -> dict:
        return {
            "return_route_attempts": int(self.return_route_attempts),
            "return_route_arrivals": int(self.return_route_arrivals),
            "return_route_failures": int(
                self.return_route_attempts - self.return_route_arrivals
            ),
            "return_route_arrival_rate": (
                self.return_route_arrivals / self.return_route_attempts
                if self.return_route_attempts else None
            ),
            "return_route_prefix_outcomes": int(len(self.return_route_records)),
        }


def _bind_reporting_contract() -> None:
    r67.REFERENCE = REFERENCE
    r67.PROTOCOL = PROTOCOL
    r67.METHOD_REVISION = METHOD_REVISION
    r67.PROTOCOL_VALUE = PROTOCOL_VALUE


def _load_runtime(args):
    r74.configure_implementation()
    loaded = list(r74._load_runtime(args))
    planner = loaded[0]
    planner.__class__ = ReturnRouteCorrectedPlanner
    planner.method_revision = METHOD_REVISION
    planner.planner_name = (
        "E549_R74_with_empirical_reach_return_route_correction"
    )
    reach = r74.empirical_reach_covariance(args.preparation)
    planner.configure_return_route(
        r67._graph_arrays(args.preparation), reach["covariance_m2"]
    )
    loaded[0] = planner
    _bind_reporting_contract()
    return tuple(loaded)


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def acquisition_signature(path: Path) -> dict:
    rows = [row for row in _rows(path) if row.get("phase") == "acquisition"]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(np.asarray([
            int(row["episode"]), int(row["command_index"])
        ], np.int32).tobytes())
        digest.update(np.asarray(row["command"], np.float32).tobytes())
        digest.update(
            np.asarray(row["actual_endpoint"], np.float32).tobytes()
        )
        digest.update(np.asarray([bool(row["reward"])], np.uint8).tobytes())
    return {"commands": len(rows), "sha256": digest.hexdigest()}


def command_audit(args) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        base = r74.command_audit(args)
    _bind_reporting_contract()
    baseline_task = R75_OUTPUT / f"task_{TASK_ID:03d}.json"
    baseline_log = R75_OUTPUT / f"task_{TASK_ID:03d}.jsonl"
    baseline = json.loads(baseline_task.read_text()) if baseline_task.is_file() else {}
    reach = r74.empirical_reach_covariance(args.preparation)
    ready = bool(
        base.get("ready_for_evaluation") is True
        and baseline_task.is_file()
        and baseline_log.is_file()
        and baseline.get("method_revision") == r75.METHOD_REVISION
        and baseline.get("first_contact_episode") == 5
        and baseline.get("conditional_return") == 0.0
    )
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_ids": [TASK_ID],
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "base_R75_task": str(baseline_task),
        "base_R75_task_sha256": (
            r52.sha256(baseline_task) if baseline_task.is_file() else None
        ),
        "base_R75_first_contact_episode": baseline.get(
            "first_contact_episode"
        ),
        "base_R75_conditional_return": baseline.get("conditional_return"),
        "return_target": PROTOCOL_VALUE["return_target"],
        "return_route_outcome": PROTOCOL_VALUE["return_route_outcome"],
        "return_route_similarity": PROTOCOL_VALUE[
            "return_route_similarity"
        ],
        "return_route_similarity_cutoff": None,
        "return_route_prior_weight": float(
            RECORDED_EVIDENCE_PRIOR_WEIGHT
        ),
        "route_reach_covariance_m2": reach["covariance_m2"].tolist(),
        "acquisition_change_from_R75": "none",
        "ready_for_evaluation": ready,
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


def command_evaluate(args) -> dict:
    if int(args.episodes) != EPISODES:
        raise ValueError("R76 requires twenty episodes")
    if int(args.seed) != int(PROTOCOL_VALUE["evaluation_seed"]):
        raise ValueError("R76 requires evaluation seed 607")
    audit = command_audit(args)
    if not audit["ready_for_evaluation"]:
        raise FileNotFoundError("R75 task 6 or R74 artifacts are incomplete")
    output = Path(args.output).resolve()
    summary_path = output / "summary.json"
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text())
        if saved.get("method_revision") != METHOD_REVISION:
            raise ValueError("saved R76 summary has another method")
        print(json.dumps(saved, indent=2))
        return saved

    (
        planner, executor, env, hub_nodes, hub_xy, nodes, reachability,
    ) = _load_runtime(args)
    try:
        row = r67._evaluate_task(
            args, planner, executor, env, TASK_ID, output, hub_nodes, hub_xy,
            reachability, nodes, float(r48.SIGMA),
        )
    finally:
        env.close()
    planner.finalize_return_episode(bool(row["successes"][-1]))
    log_path = output / f"task_{TASK_ID:03d}.jsonl"
    row.update(planner.return_route_summary())
    row["return_route_correction"] = (
        "R51_corrected_exact_reachability_plus_empirical_reach_route_posterior"
    )
    r48.dump(output / f"task_{TASK_ID:03d}.json", row)

    baseline_log = R75_OUTPUT / f"task_{TASK_ID:03d}.jsonl"
    baseline_signature = acquisition_signature(baseline_log)
    replay_signature = acquisition_signature(log_path)
    acquisition_exact = baseline_signature == replay_signature
    baseline = json.loads(
        (R75_OUTPUT / f"task_{TASK_ID:03d}.json").read_text()
    )
    result = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_id": TASK_ID,
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "R75_first_contact_episode": baseline["first_contact_episode"],
        "R76_first_contact_episode": row["first_contact_episode"],
        "R75_conditional_return": baseline["conditional_return"],
        "R76_conditional_return": row["conditional_return"],
        "R75_successes": baseline["successes"],
        "R76_successes": row["successes"],
        "acquisition_replay_R75": baseline_signature,
        "acquisition_replay_R76": replay_signature,
        "acquisition_replay_exact": acquisition_exact,
        **planner.return_route_summary(),
        "return_route_similarity_cutoff": None,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    r48.dump(summary_path, result)
    print(json.dumps(result, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    value.add_argument("--episodes", type=int, default=EPISODES)
    value.add_argument("--preparation", type=Path, default=r52.DEFAULT_OUTPUT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--graph", type=Path)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("evaluate").set_defaults(func=command_evaluate)
    return value


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
