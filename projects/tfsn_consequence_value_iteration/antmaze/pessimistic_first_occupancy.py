"""R20: cached first-occupancy planning with ensemble pessimism.

R19 integrates every spatial component deterministically but averages the
three independently trained consequence members before selecting a command.
R20 changes only that aggregation: each live command is valued by its least
optimistic member.  The frozen R10 graph is unchanged; its learned bridge
weights were already constructed with a member-wise minimum.

Per-command logs retain all three member values and additionally report each
member's expected endpoint mobility.  On a failed gate this exposes whether
the transition that entered a sustained stall was warned against by any
member, without adding a posture rule to the planner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze.cached_first_occupancy import (
    CLAIM_LEVEL,
    CachedFirstOccupancyPlanner,
    command_evaluate_set_for,
    command_summarize_for,
    command_task_for,
)


METHOD_REVISION = "antmaze_pessimistic_cached_first_occupancy_r20"


class PessimisticFirstOccupancyPlanner(CachedFirstOccupancyPlanner):
    method_revision = METHOD_REVISION
    planner_name = "pessimistic_cached_first_occupancy"
    output_name = "pessimistic_cached_first_occupancy_r20"
    evaluation_arm = "pessimistic_cached_first_occupancy"
    claim_level = CLAIM_LEVEL
    ensemble_aggregation = "member_minimum"

    def _aggregate_member_values(self, member_q):
        """Use the lower envelope of the independently integrated members."""
        values = np.asarray(member_q, np.float32)
        if values.ndim != 2:
            raise ValueError("member values must be candidate by ensemble")
        return values.min(axis=1)


def command_task(args):
    command_task_for(args, PessimisticFirstOccupancyPlanner)


def command_evaluate_set(args):
    command_evaluate_set_for(args, PessimisticFirstOccupancyPlanner)


def command_summarize(args):
    command_summarize_for(args, PessimisticFirstOccupancyPlanner)


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=605)
    subparsers = value.add_subparsers(dest="command", required=True)
    task = subparsers.add_parser("task")
    task.add_argument("--task-id", type=int, required=True)
    task.set_defaults(func=command_task)
    evaluate = subparsers.add_parser("evaluate-set")
    evaluate.add_argument(
        "--set-name", choices=("gate", "all15"), required=True
    )
    evaluate.set_defaults(func=command_evaluate_set)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument(
        "--set-name", choices=("gate", "all15"), required=True
    )
    summarize.set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = (
            Path(args.root)
            / "model_planning"
            / "round2_recorded_transition_graph_r6.npz"
        )
    args.func(args)


if __name__ == "__main__":
    main()
