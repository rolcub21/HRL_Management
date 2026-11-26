"""R33: posterior-sampled hidden-goal acquisition with frozen R31/R22.

The actual hidden goal remains evaluator-only.  At each physical reset, a
hypothetical goal is drawn from the residual spatial search belief and held
for the ten-command physical episode.  The unchanged supplied-goal R22
planner routes to that hypothesis.  Observed reward switches to R31 return.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r31_aligned_hidden_goal as r31,
)
from scripts.antmaze.current_tfsn.collect import SuppliedGoalField
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, paths


HERE = Path(__file__).resolve().parent
METHOD_REFERENCE = HERE / "R33_POSTERIOR_SAMPLING_REFERENCE.md"
PROTOCOL = HERE / "r33_posterior_sampling_protocol.json"
METHOD_REVISION = "antmaze_posterior_sampled_acquisition_r33"
RESULT_DIRECTORY = (
    r31.MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "posterior_sampled_acquisition_r33"
)


def load_contract() -> dict:
    protocol = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_method_revision": r31.METHOD_REVISION,
        "oracle_inputs": [],
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_reward_used_for_preparation": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "posterior_prior": (
            "uniform_over_unique_reward_free_reachable_"
            "landing_precision_cells"
        ),
        "posterior_likelihood": (
            "residual_search_field_from_actual_visited_states"
        ),
        "hypothesis_resampling_schedule": "once_per_physical_episode",
        "gate_task_ids": [3, 5],
        "development_task_ids": list(range(15)),
        "episodes_per_task": 5,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"R33 protocol has another {key}")
    reference = Path(protocol["method_reference"])
    if not reference.is_absolute():
        reference = r31.MANAGEMENT_ROOT / reference
    if reference.resolve() != METHOD_REFERENCE:
        raise ValueError("R33 method reference changed")
    return protocol


PROTOCOL_VALUE = load_contract()


def posterior_mass(support_xy, fields):
    """Return an exact discrete posterior over unique reachable cells."""
    support = np.asarray(support_xy, np.float32).reshape(-1, 2)
    if not len(support):
        raise ValueError("posterior support must not be empty")
    raw = np.asarray(fields.search(support), np.float64).reshape(-1)
    if not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise ValueError("residual search belief must be finite and nonnegative")
    total = float(raw.sum())
    exhausted = total == 0.0
    if exhausted:
        raw = np.ones(len(support), np.float64)
        total = float(len(support))
    return raw, total, exhausted


class PosteriorGoalField(SuppliedGoalField):
    """Fixed hypothetical goal field, distinct from observed task evidence."""

    def __init__(self, sigma, goal):
        super().__init__(sigma, goal)
        # R31's return layer inspects these only when positive online evidence
        # exists.  A hypothesis is deliberately not evidence.
        self.coords = []
        self.rewards = []


class PosteriorSampledPlanner(r31.AlignedBoundedConsumptionPlanner):
    method_revision = METHOD_REVISION
    planner_name = "posterior_sampled_hidden_goal_r33"

    def configure_posterior_support(self, support_xy):
        support = np.asarray(support_xy, np.float32).reshape(-1, 2)
        if not len(support) or not np.isfinite(support).all():
            raise ValueError("posterior support must be finite and nonempty")
        if len(np.unique(support, axis=0)) != len(support):
            raise ValueError("posterior support must contain unique spatial cells")
        self.posterior_support_xy = support
        self.reset_posterior_task()

    def reset_posterior_task(self):
        self.posterior_goal = None
        self.posterior_episode = -1
        self.posterior_probability = None
        self.posterior_residual_mass = None
        self.posterior_positive_cells = None
        self.posterior_exhausted = False
        self.goal_key = None

    def _sample_posterior_goal(self, fields):
        if not hasattr(self, "posterior_support_xy"):
            raise RuntimeError("posterior support was not configured")
        mass, total, exhausted = posterior_mass(
            self.posterior_support_xy, fields
        )
        cdf = np.cumsum(mass)
        cdf[-1] = total
        draw = float(self.rng.random()) * total
        index = min(int(np.searchsorted(cdf, draw, side="right")), len(mass) - 1)
        self.posterior_goal = self.posterior_support_xy[index].copy()
        self.posterior_probability = float(mass[index] / total)
        self.posterior_residual_mass = total
        self.posterior_positive_cells = int(np.sum(mass > 0.0))
        self.posterior_exhausted = bool(exhausted)
        self.posterior_episode += 1
        self.goal_key = None

    def _set_goal(self, fields):
        if isinstance(fields, PosteriorGoalField):
            self.acquisition_active = False
            # This is the frozen supplied-goal R22 recursion, cached for the
            # episode because the hypothetical goal is fixed for ten commands.
            return r31.lazy.LazyAliasAdvancePlanner._set_goal(self, fields)
        return super()._set_goal(fields)

    def choose(
        self,
        state,
        command_index,
        fields,
        online_states=(),
        online_trajectories=(),
    ):
        if fields.contacted:
            command, diagnostic = super().choose(
                state,
                command_index,
                fields,
                online_states,
                online_trajectories,
            )
            diagnostic.update({
                "planner": self.planner_name,
                "method_revision": self.method_revision,
                "phase": "return",
                "posterior_hypothesis_goal": None,
            })
            return command, diagnostic

        if int(command_index) == 0 or self.posterior_goal is None:
            self._sample_posterior_goal(fields)
        hypothesis = PosteriorGoalField(r31.SIGMA, self.posterior_goal)
        command, diagnostic = super().choose(
            state,
            command_index,
            hypothesis,
            online_states,
            online_trajectories,
        )
        diagnostic.update({
            "planner": self.planner_name,
            "method_revision": self.method_revision,
            "phase": "acquisition",
            "search_enabled": True,
            "acquisition_search_mode": "posterior_sampled_hypothesis",
            "fine_search_sigma": float(fields.sigma),
            "coarse_search_sigma": float(fields.coarse_sigma),
            "coarse_search_weight": float(fields.coarse_weight),
            "posterior_hypothesis_goal": self.posterior_goal.tolist(),
            "posterior_hypothesis_episode": int(self.posterior_episode),
            "posterior_hypothesis_probability": self.posterior_probability,
            "posterior_residual_mass_cells": self.posterior_residual_mass,
            "posterior_positive_support_cells": self.posterior_positive_cells,
            "posterior_support_cells": int(len(self.posterior_support_xy)),
            "posterior_exhausted_uniform_fallback": self.posterior_exhausted,
            "posterior_prior": (
                "uniform_over_unique_reward_free_reachable_"
                "landing_precision_cells"
            ),
            "hypothesis_resampling_schedule": "once_per_physical_episode",
            "acquisition_backup": "frozen_supplied_goal_R22",
            "acquisition_immediate_readout": (
                "Gaussian_field_at_posterior_sampled_hypothesis"
            ),
            "return_online_positive_evidence_candidates": 0,
        })
        return command, diagnostic


def _load_runtime(args):
    graph = r31.verify_graph(args.graph)
    planner, hashes = r31.lazy._load_planner(
        args, PosteriorSampledPlanner
    )
    geometry = r31.r28.BoundedPassageGeometry(
        args.root, graph, planner.targets, planner.weight
    )
    planner.configure_hidden_fields(r31.SIGMA)
    planner.configure_empirical_passage(geometry)
    support = geometry.grid_xy[geometry.reachable_cell_index]
    planner.configure_posterior_support(support)
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


def task_ids(set_name):
    if set_name == "gate2":
        return tuple(int(x) for x in PROTOCOL_VALUE["gate_task_ids"])
    if set_name == "all15":
        return tuple(int(x) for x in PROTOCOL_VALUE["development_task_ids"])
    raise ValueError(set_name)


def evaluation_directory(set_name):
    return RESULT_DIRECTORY / f"evaluation_{set_name}"


def task_result_path(set_name, task_id):
    gate = evaluation_directory("gate2") / f"task_{task_id:03d}.json"
    if set_name == "all15" and task_id in task_ids("gate2") and gate.exists():
        return gate
    return evaluation_directory(set_name) / f"task_{task_id:03d}.json"


def _evaluate_task(args, planner, executor, env, task_id):
    task = r31.r22._task(args.root, task_id)
    output = evaluation_directory(args.set_name)
    output.mkdir(parents=True, exist_ok=True)
    result_path = task_result_path(args.set_name, task_id)
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = (
            task["goal"], task["reset_seed"], METHOD_REVISION,
            int(args.episodes), float(r31.SIGMA),
            r31.sha256(r31.R30_CHECKPOINT),
        )
        actual = (
            saved.get("goal"), saved.get("reset_seed"),
            saved.get("method_revision"), saved.get("episodes"),
            saved.get("landing_sigma_m"),
            saved.get("factorized_consequence_sha256"),
        )
        if actual != expected:
            raise ValueError(f"existing R33 task has another contract: {result_path}")
        print(json.dumps(saved))
        return saved

    task_seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(task_seed)
    planner.seed = task_seed
    planner.reset_posterior_task()
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
        "evaluation_arm": "posterior_sampled_acquisition_with_frozen_R31",
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r31.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "episodes": int(args.episodes),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "evaluator_uses_hidden_goal_for_contact_only": True,
        "task_memory_persisted_across_physical_resets": True,
        "posterior_prior": PROTOCOL_VALUE["posterior_prior"],
        "posterior_likelihood": PROTOCOL_VALUE["posterior_likelihood"],
        "hypothesis_resampling_schedule": (
            PROTOCOL_VALUE["hypothesis_resampling_schedule"]
        ),
        "posterior_support_cells": int(len(planner.posterior_support_xy)),
        "landing_sigma_m": float(r31.SIGMA),
        "landing_sigma_estimator": r31.SIGMA_ESTIMATOR,
        "acquisition_search_sigma_m": float(r31.SIGMA),
        "return_field_sigma_m": float(r31.SIGMA),
        "return_planner": "unchanged_R31_return",
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def command_evaluate_set(args):
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in task_ids(args.set_name):
            _evaluate_task(args, planner, executor, env, task_id)
    finally:
        env.close()
    command_summarize(args)


def command_summarize(args):
    selected = task_ids(args.set_name)
    rows = [
        json.loads(task_result_path(args.set_name, task_id).read_text())
        for task_id in selected
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
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(METHOD_REFERENCE),
        "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
        "method_revision": METHOD_REVISION,
        "base_method_revision": r31.METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "set_name": args.set_name,
        "task_ids": list(selected),
        "tasks": len(rows),
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(sum(x is not None for x in contact)),
        "acquisition_at_1": float(np.mean([x == 0 for x in contact])),
        "acquisition_at_5": float(np.mean([
            x is not None and x < 5 for x in contact
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
        "posterior_support_cells": int(rows[0]["posterior_support_cells"]),
        "landing_sigma_m": float(r31.SIGMA),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
        "rows": rows,
    }
    output = evaluation_directory(args.set_name)
    dump(output / "summary.json", result)
    print(json.dumps({
        key: result[key]
        for key in (
            "tasks", "acquired_tasks", "acquisition_at_1",
            "acquisition_at_5", "pooled_conditional_return", "mean_e2e",
            "acquisition_new_1m_cells_per_command",
            "acquisition_zero_new_1m_cell_fraction",
            "acquisition_revisited_endpoint_fraction",
            "acquisition_two_cycle_count", "posterior_support_cells",
        )
    }, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(
        func=lambda _args: print(json.dumps({
            "method_revision": METHOD_REVISION,
            "method_reference": str(METHOD_REFERENCE),
            "method_reference_sha256": r31.sha256(METHOD_REFERENCE),
            "base_method_revision": r31.METHOD_REVISION,
            "landing_sigma_m": r31.SIGMA,
            "posterior_prior": PROTOCOL_VALUE["posterior_prior"],
            "hypothesis_resampling_schedule": (
                PROTOCOL_VALUE["hypothesis_resampling_schedule"]
            ),
            "hidden_goal_visible_to_planner_executor_or_models": False,
            "confirmation_panel_accessed": False,
        }, indent=2))
    )
    evaluate = commands.add_parser("evaluate-set")
    evaluate.add_argument("--set-name", choices=("gate2", "all15"), required=True)
    evaluate.add_argument(
        "--episodes", type=int, default=PROTOCOL_VALUE["episodes_per_task"]
    )
    evaluate.set_defaults(func=command_evaluate_set)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--set-name", choices=("gate2", "all15"), required=True)
    summarize.add_argument(
        "--episodes", type=int, default=PROTOCOL_VALUE["episodes_per_task"]
    )
    summarize.set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = r31.graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
