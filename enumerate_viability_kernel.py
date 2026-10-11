"""Enumerate compact-yard strict-macro recoverability labels.

This utility builds supervision from the transition model itself.  It never
runs or queries a storage assignment baseline.  The default post-transfer
agent set covers the two atomic decision-boundary locations produced by the
new action model: an occupied storage destination or a delivery exit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

from example.yard_geometry import make_shipyard_env
from PSLAP.dynamic_yard import BlockView
from PSLAP.viability import RecoveryState, analyze_recoverability
from PSLAP.viability_filter import ROBUST_QUEUE_OBSTACLE_CONTRACT


PROTOCOL = "exhaustive_compact_recovery_kernel_post_transfer_states_v1"


def _optional_positive(value: str):
    number = int(value)
    if number == 0:
        return None
    if number < 0:
        raise argparse.ArgumentTypeError("value must be zero or positive")
    return number


def _state_payload(state, certificate, mask: int) -> dict:
    return {
        "occupancy_mask": int(mask),
        "occupancy_count": len(state.blocks),
        "agent_position": tuple(state.agent_position),
        "status": certificate.status.value,
        "witness_macro_count": (
            len(certificate.witness) if certificate.is_safe else None
        ),
        "exact_recovery_rank": certificate.exact_recovery_rank,
        "recovery_rank_exact": certificate.recovery_rank_is_exact,
        "witness_primitive_steps": certificate.witness_primitive_steps,
        "explored_nodes": certificate.explored_nodes,
        "generated_states": certificate.generated_states,
        "max_depth_reached": certificate.max_depth_reached,
        "frontier_states": certificate.frontier_states,
        "exhaustive": certificate.exhaustive,
        "reason": certificate.reason,
    }


def enumerate_kernel(args) -> dict:
    env = make_shipyard_env(
        arrival_rate=args.lam,
        proc_mean=args.mu,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        exit_width=args.exit_width,
        number_blocks=max(1, args.number_blocks),
    )
    storage = tuple(sorted(tuple(cell) for cell in env.storage_positions))
    traversable = frozenset(
        (row, col)
        for row in range(env.grid_rows)
        for col in range(env.grid_cols)
        if env.rooms[row, col] != "#"
    )
    fixed = frozenset((tuple(env.pickup_cell), tuple(env.waiting_cell)))
    maximum = min(
        len(storage),
        len(storage) if args.max_occupancy is None else args.max_occupancy,
    )
    minimum = min(maximum, int(args.min_occupancy))

    rows = []
    status_counts = Counter()
    by_occupancy = defaultdict(Counter)
    explored_total = 0
    generated_total = 0
    for occupied_count in range(minimum, maximum + 1):
        for occupied_indices in combinations(range(len(storage)), occupied_count):
            occupied = tuple(storage[index] for index in occupied_indices)
            mask = sum(1 << index for index in occupied_indices)
            blocks = tuple(
                BlockView(f"B{index:03d}", cell, 0.0)
                for index, cell in enumerate(occupied)
            )
            # Delivery leaves the agent at an exit; storage/reconfiguration
            # leaves it sharing the just-placed block's cell.
            agent_positions = tuple(sorted(set(env.exit_cells) | set(occupied)))
            if not agent_positions:
                agent_positions = (tuple(env.start_state),)
            for agent_position in agent_positions:
                state = RecoveryState(
                    rows=env.grid_rows,
                    cols=env.grid_cols,
                    traversable=traversable,
                    storage_cells=frozenset(storage),
                    exits=tuple(tuple(cell) for cell in env.exit_cells),
                    blocks=blocks,
                    agent_position=agent_position,
                    fixed_obstacles=fixed - {agent_position},
                    pickup_cells=frozenset({tuple(env.pickup_cell)}),
                    wait_cells=frozenset({tuple(env.waiting_cell)}),
                )
                certificate = analyze_recoverability(
                    state,
                    max_depth=args.max_depth,
                    max_nodes=args.max_nodes,
                    max_primitive_steps=args.max_primitive_steps,
                    search_order=args.search_order,
                )
                status = certificate.status.value
                status_counts[status] += 1
                by_occupancy[occupied_count][status] += 1
                explored_total += certificate.explored_nodes
                generated_total += certificate.generated_states
                if args.include_states:
                    rows.append(_state_payload(state, certificate, mask))

    total = sum(status_counts.values())
    return {
        "protocol": PROTOCOL,
        "baseline_viability_teacher": False,
        "information_regime": "online_no_future_schedule",
        "fixed_obstacle_contract": ROBUST_QUEUE_OBSTACLE_CONTRACT,
        "geometry": {
            "grid_rows": env.grid_rows,
            "grid_cols": env.grid_cols,
            "storage_cells": storage,
            "storage_cell_count": len(storage),
            "exit_cells": tuple(tuple(cell) for cell in env.exit_cells),
            "pickup_cell": tuple(env.pickup_cell),
            "waiting_cell": tuple(env.waiting_cell),
        },
        "search": {
            "min_occupancy": minimum,
            "max_occupancy": maximum,
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
            "max_primitive_steps": args.max_primitive_steps,
            "search_order": args.search_order,
        },
        "state_count": total,
        "status_counts": dict(status_counts),
        "safe_fraction": status_counts["SAFE"] / max(1, total),
        "unknown_fraction": status_counts["UNKNOWN"] / max(1, total),
        "mean_explored_nodes": explored_total / max(1, total),
        "mean_generated_states": generated_total / max(1, total),
        "by_occupancy": {
            str(count): dict(values)
            for count, values in sorted(by_occupancy.items())
        },
        "states": rows if args.include_states else None,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--exit-width", type=int, default=1)
    parser.add_argument("--number-blocks", type=int, default=24)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.8)
    parser.add_argument("--mu", type=float, default=100.0)
    parser.add_argument("--max-occupancy", type=int)
    parser.add_argument("--min-occupancy", type=int, default=0)
    parser.add_argument("--max-depth", type=_optional_positive, default=None)
    parser.add_argument("--max-nodes", type=_optional_positive, default=100_000)
    parser.add_argument(
        "--max-primitive-steps", type=_optional_positive, default=None
    )
    parser.add_argument(
        "--search-order",
        choices=("breadth_first", "goal_directed"),
        default="goal_directed",
    )
    parser.add_argument(
        "--include-states",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.grid_rows < 4 or args.grid_cols < 4:
        parser.error("grid dimensions must be at least 4")
    if args.max_occupancy is not None and args.max_occupancy < 0:
        parser.error("max-occupancy must be nonnegative")
    if args.min_occupancy < 0:
        parser.error("min-occupancy must be nonnegative")
    if (
        args.max_occupancy is not None
        and args.min_occupancy > args.max_occupancy
    ):
        parser.error("min-occupancy cannot exceed max-occupancy")
    return args


def main(argv=None):
    args = parse_args(argv)
    payload = enumerate_kernel(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {key: value for key, value in payload.items() if key != "states"}
    print(json.dumps(summary, indent=2))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
