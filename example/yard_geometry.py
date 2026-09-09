"""Versioned geometry construction for matched shipyard experiments."""

from __future__ import annotations

import hashlib
import json

from example.small_rooms_env import SmallRoomsEnv


GEOMETRY_CONTRACT = "open_yard_right_aligned_bottom_gate_v1"


def right_aligned_exit_cells(
    grid_rows: int,
    grid_cols: int,
    exit_width: int,
) -> tuple[tuple[int, int], ...]:
    """Return a contiguous bottom gate ending at the right interior column."""

    grid_rows = int(grid_rows)
    grid_cols = int(grid_cols)
    exit_width = int(exit_width)
    if grid_rows < 4 or grid_cols < 4:
        raise ValueError("grid dimensions must both be at least 4")
    interior_width = grid_cols - 2
    if not 1 <= exit_width <= interior_width:
        raise ValueError(
            f"exit_width must be in [1, {interior_width}] for this grid"
        )
    right = grid_cols - 2
    left = right - exit_width + 1
    return tuple((grid_rows - 1, col) for col in range(left, right + 1))


def make_shipyard_env(
    *,
    arrival_rate: float,
    proc_mean: float,
    grid_rows: int = 10,
    grid_cols: int = 10,
    exit_width: int | None = None,
    number_blocks: int = 40,
) -> SmallRoomsEnv:
    """Construct one environment under an explicit geometry contract."""

    grid_rows = int(grid_rows)
    grid_cols = int(grid_cols)
    kwargs = {
        "choose_storage": False,
        "arrival_rate": float(arrival_rate),
        "proc_mean": float(proc_mean),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "number_blocks": int(number_blocks),
    }
    if exit_width is not None:
        kwargs["exit_cells"] = list(
            right_aligned_exit_cells(grid_rows, grid_cols, exit_width)
        )
    return SmallRoomsEnv(**kwargs)


def geometry_metadata(env, *, requested_exit_width: int | None) -> dict:
    geometry_payload = {
        "geometry_contract": GEOMETRY_CONTRACT,
        "grid_rows": int(env.grid_rows),
        "grid_cols": int(env.grid_cols),
        "start_state": tuple(env.start_state),
        "door_cell": tuple(env.door_cell),
        "pickup_cell": tuple(env.pickup_cell),
        "waiting_cell": tuple(env.waiting_cell),
        "exit_cells": tuple(tuple(cell) for cell in env.exit_cells),
        "storage_positions": tuple(
            tuple(cell) for cell in env.storage_positions
        ),
        "room_rows": tuple("".join(row) for row in env.rooms),
    }
    canonical = json.dumps(
        geometry_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        **geometry_payload,
        "geometry_signature": hashlib.sha256(canonical).hexdigest()[:16],
        "geometry_regime": (
            "ordinary_default_gate"
            if requested_exit_width is None
            else f"right_aligned_gate_width_{int(requested_exit_width)}"
        ),
        "requested_exit_width": requested_exit_width,
        "actual_exit_width": len(env.exit_cells),
        "storage_cell_count": len(env.storage_positions),
        "block_count": len(env.blocks),
        "nominal_storage_density": (
            len(env.blocks) / max(1, len(env.storage_positions))
        ),
    }


__all__ = [
    "GEOMETRY_CONTRACT",
    "geometry_metadata",
    "make_shipyard_env",
    "right_aligned_exit_cells",
]
