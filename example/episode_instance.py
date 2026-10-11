"""Immutable, serializable shipyard episode definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Optional


Cell = tuple[int, int]


def _cell(value) -> Cell:
    row, col = value
    return int(row), int(col)


def _cells(values) -> tuple[Cell, ...]:
    return tuple(_cell(value) for value in values)


@dataclass(frozen=True)
class EpisodeInstance:
    """All stochastic inputs and geometry required to replay one episode."""

    schema_version: int
    seed: Optional[int]
    arrival_rate: float
    proc_mean: float
    arrival_steps: tuple[int, ...]
    storage_steps_needed: tuple[int, ...]
    grid_rows: int
    grid_cols: int
    start_state: Cell
    door_cell: Cell
    pickup_cell: Cell
    waiting_cell: Cell
    exit_cells: tuple[Cell, ...]
    storage_positions: tuple[Cell, ...]
    room_rows: tuple[str, ...]

    SCHEMA_VERSION = 1

    def __post_init__(self):
        # ``frozen=True`` prevents field reassignment; canonicalizing every
        # sequence also prevents callers from retaining mutable list aliases.
        object.__setattr__(
            self,
            "arrival_steps",
            tuple(int(value) for value in self.arrival_steps),
        )
        object.__setattr__(
            self,
            "storage_steps_needed",
            tuple(int(value) for value in self.storage_steps_needed),
        )
        object.__setattr__(self, "start_state", _cell(self.start_state))
        object.__setattr__(self, "door_cell", _cell(self.door_cell))
        object.__setattr__(self, "pickup_cell", _cell(self.pickup_cell))
        object.__setattr__(self, "waiting_cell", _cell(self.waiting_cell))
        object.__setattr__(self, "exit_cells", _cells(self.exit_cells))
        object.__setattr__(
            self, "storage_positions", _cells(self.storage_positions)
        )
        object.__setattr__(
            self, "room_rows", tuple(str(row) for row in self.room_rows)
        )
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported EpisodeInstance schema {self.schema_version}; "
                f"expected {self.SCHEMA_VERSION}"
            )
        if len(self.arrival_steps) != len(self.storage_steps_needed):
            raise ValueError("arrival and storage-duration counts must match")
        if not self.arrival_steps:
            raise ValueError("an episode instance must contain at least one block")
        if any(value < 0 for value in self.arrival_steps):
            raise ValueError("arrival steps must be nonnegative")
        if tuple(sorted(self.arrival_steps)) != self.arrival_steps:
            raise ValueError("arrival steps must be nondecreasing")
        if self.arrival_steps[0] != 0:
            raise ValueError("the first block must arrive at step zero")
        if any(value < 1 for value in self.storage_steps_needed):
            raise ValueError("storage durations must be positive")
        if len(self.room_rows) != self.grid_rows or any(
            len(row) != self.grid_cols for row in self.room_rows
        ):
            raise ValueError("room geometry does not match grid dimensions")

    @property
    def number_blocks(self) -> int:
        return len(self.arrival_steps)

    @property
    def instance_id(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:16]

    @property
    def schedule_id(self) -> str:
        """Hash stochastic work content while deliberately excluding geometry."""

        canonical = json.dumps(
            {
                "arrival_rate": self.arrival_rate,
                "proc_mean": self.proc_mean,
                "arrival_steps": self.arrival_steps,
                "storage_steps_needed": self.storage_steps_needed,
                "number_blocks": self.number_blocks,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EpisodeInstance":
        return cls(
            schema_version=int(payload["schema_version"]),
            seed=(
                None if payload.get("seed") is None else int(payload["seed"])
            ),
            arrival_rate=float(payload["arrival_rate"]),
            proc_mean=float(payload["proc_mean"]),
            arrival_steps=tuple(int(value) for value in payload["arrival_steps"]),
            storage_steps_needed=tuple(
                int(value) for value in payload["storage_steps_needed"]
            ),
            grid_rows=int(payload["grid_rows"]),
            grid_cols=int(payload["grid_cols"]),
            start_state=_cell(payload["start_state"]),
            door_cell=_cell(payload["door_cell"]),
            pickup_cell=_cell(payload["pickup_cell"]),
            waiting_cell=_cell(payload["waiting_cell"]),
            exit_cells=_cells(payload["exit_cells"]),
            storage_positions=_cells(payload["storage_positions"]),
            room_rows=tuple(str(row) for row in payload["room_rows"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, value: str) -> "EpisodeInstance":
        return cls.from_dict(json.loads(value))

    def validate_for(self, env) -> None:
        expected = {
            "arrival_rate": float(env.arrival_rate),
            "proc_mean": float(env.proc_mean),
            "grid_rows": int(env.grid_rows),
            "grid_cols": int(env.grid_cols),
            "start_state": tuple(env.start_state),
            "door_cell": tuple(env.door_cell),
            "pickup_cell": tuple(env.pickup_cell),
            "waiting_cell": tuple(env.waiting_cell),
            "exit_cells": tuple(env.exit_cells),
            "storage_positions": tuple(env.storage_positions),
            "room_rows": tuple("".join(row) for row in env.rooms),
            "number_blocks": len(env.blocks),
        }
        actual = {
            "arrival_rate": self.arrival_rate,
            "proc_mean": self.proc_mean,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "start_state": self.start_state,
            "door_cell": self.door_cell,
            "pickup_cell": self.pickup_cell,
            "waiting_cell": self.waiting_cell,
            "exit_cells": self.exit_cells,
            "storage_positions": self.storage_positions,
            "room_rows": self.room_rows,
            "number_blocks": self.number_blocks,
        }
        mismatches = [
            key for key in expected if expected[key] != actual[key]
        ]
        if mismatches:
            raise ValueError(
                "EpisodeInstance is incompatible with environment: "
                + ", ".join(mismatches)
            )


__all__ = ["EpisodeInstance"]
