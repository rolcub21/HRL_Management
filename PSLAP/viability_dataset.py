"""Reproducible dynamics-only data for the recoverability critic.

The data in this module are generated from :mod:`PSLAP.viability` search,
never from a storage selector or a behavior policy.  Every record contains a
lossless :class:`~PSLAP.viability.RecoveryState`, so a saved label can be
rechecked without reconstructing hidden environment state.

Splits are made by static layout, not by row.  Counterfactual siblings from
one topology therefore cannot leak between training, model selection,
calibration, and test partitions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random
from typing import Callable, Iterable, Mapping, Optional, Sequence

from PSLAP.dynamic_yard import BlockView, Cell
from PSLAP.viability import (
    CLOSED_ADMISSION_CONTRACT,
    STRICT_MACRO_ACTION_MODEL,
    RecoveryState,
    ViabilityStatus,
    analyze_recoverability,
)
from PSLAP.viability_critic import (
    DYNAMICS_LABEL_CONTRACT,
    DynamicsViabilityLabel,
)
from PSLAP.yard_graph import (
    NODE_FEATURE_DIM,
    NODE_FEATURE_NAMES,
    encode_recovery_state,
)
from example.yard_geometry import make_shipyard_env


VIABILITY_DATASET_PROTOCOL = "dynamics_viability_multi_layout_v1"
VIABILITY_SPLIT_PROTOCOL = "layout_group_disjoint_stratified_v1"
DEFAULT_SPLIT_RATIOS = {
    "train": 0.65,
    "validation": 0.15,
    "calibration": 0.10,
    "test": 0.10,
}


def canonical_sha256(value) -> str:
    """Return a stable digest for JSON-compatible protocol data."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cells_payload(cells: Iterable[Cell]) -> list[list[int]]:
    return [[int(row), int(col)] for row, col in sorted(set(cells))]


def _cells_from_payload(values, *, name: str) -> frozenset[Cell]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    cells = []
    for value in values:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise ValueError(f"{name} must contain integer [row, column] cells")
        cells.append((int(value[0]), int(value[1])))
    if len(cells) != len(set(cells)):
        raise ValueError(f"{name} contains duplicate cells")
    return frozenset(cells)


def recovery_state_to_dict(state: RecoveryState) -> dict:
    """Serialize every field needed to re-run dynamics certification."""

    if not isinstance(state, RecoveryState):
        raise TypeError("state must be a RecoveryState")
    blocks = []
    for block in state.blocks:
        remaining = float(block.remaining_time)
        if not math.isfinite(remaining):
            raise ValueError("block remaining_time must be finite")
        blocks.append(
            {
                "label": str(block.label),
                "position": [int(block.position[0]), int(block.position[1])],
                "remaining_time": remaining,
            }
        )
    return {
        "rows": int(state.rows),
        "cols": int(state.cols),
        "traversable": _cells_payload(state.traversable),
        "storage_cells": _cells_payload(state.storage_cells),
        "exits": _cells_payload(state.exits),
        "blocks": blocks,
        "agent_position": [
            int(state.agent_position[0]),
            int(state.agent_position[1]),
        ],
        "fixed_obstacles": _cells_payload(state.fixed_obstacles),
        "reserved_cells": _cells_payload(state.reserved_cells),
        "pickup_cells": _cells_payload(state.pickup_cells),
        "wait_cells": _cells_payload(state.wait_cells),
    }


def recovery_state_from_dict(value: Mapping) -> RecoveryState:
    """Load and validate a lossless state payload."""

    expected = {
        "rows",
        "cols",
        "traversable",
        "storage_cells",
        "exits",
        "blocks",
        "agent_position",
        "fixed_obstacles",
        "reserved_cells",
        "pickup_cells",
        "wait_cells",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        missing = sorted(expected - set(value)) if isinstance(value, Mapping) else sorted(expected)
        extra = sorted(set(value) - expected) if isinstance(value, Mapping) else []
        raise ValueError(
            f"noncanonical RecoveryState payload: missing={missing}, extra={extra}"
        )
    rows = value["rows"]
    cols = value["cols"]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (rows, cols)):
        raise ValueError("rows and cols must be integers")
    agent_values = value["agent_position"]
    if (
        not isinstance(agent_values, list)
        or len(agent_values) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in agent_values)
    ):
        raise ValueError("agent_position must be an integer [row, column]")
    raw_blocks = value["blocks"]
    if not isinstance(raw_blocks, list):
        raise ValueError("blocks must be a list")
    blocks = []
    for raw in raw_blocks:
        if not isinstance(raw, Mapping) or set(raw) != {
            "label",
            "position",
            "remaining_time",
        }:
            raise ValueError("block payload is noncanonical")
        position = raw["position"]
        if (
            not isinstance(raw["label"], str)
            or not raw["label"]
            or not isinstance(position, list)
            or len(position) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in position)
            or isinstance(raw["remaining_time"], bool)
            or not isinstance(raw["remaining_time"], (int, float))
            or not math.isfinite(float(raw["remaining_time"]))
        ):
            raise ValueError("block payload contains invalid values")
        blocks.append(
            BlockView(
                str(raw["label"]),
                (int(position[0]), int(position[1])),
                float(raw["remaining_time"]),
            )
        )
    state = RecoveryState(
        rows=int(rows),
        cols=int(cols),
        traversable=_cells_from_payload(value["traversable"], name="traversable"),
        storage_cells=_cells_from_payload(value["storage_cells"], name="storage_cells"),
        exits=tuple(_cells_from_payload(value["exits"], name="exits")),
        blocks=tuple(blocks),
        agent_position=(int(agent_values[0]), int(agent_values[1])),
        fixed_obstacles=_cells_from_payload(value["fixed_obstacles"], name="fixed_obstacles"),
        reserved_cells=_cells_from_payload(value["reserved_cells"], name="reserved_cells"),
        pickup_cells=_cells_from_payload(value["pickup_cells"], name="pickup_cells"),
        wait_cells=_cells_from_payload(value["wait_cells"], name="wait_cells"),
    )
    # Graph encoding adds strict bounds and finite-feature validation that the
    # compact RecoveryState constructor intentionally leaves to its consumers.
    graph = encode_recovery_state(state)
    if any(
        not math.isfinite(feature)
        for node in graph.node_features
        for feature in node
    ):
        raise ValueError("RecoveryState produced non-finite graph features")
    return state


@dataclass(frozen=True)
class ViabilityLayout:
    """One static topology used as a split group."""

    layout_id: str
    generator_seed: int
    rows: int
    cols: int
    traversable: frozenset[Cell]
    storage_cells: frozenset[Cell]
    exits: tuple[Cell, ...]
    fixed_obstacles: frozenset[Cell]
    pickup_cells: frozenset[Cell]
    wait_cells: frozenset[Cell]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout_id", str(self.layout_id))
        object.__setattr__(self, "generator_seed", int(self.generator_seed))
        object.__setattr__(self, "traversable", frozenset(self.traversable))
        object.__setattr__(self, "storage_cells", frozenset(self.storage_cells))
        object.__setattr__(self, "exits", tuple(sorted(set(self.exits))))
        object.__setattr__(self, "fixed_obstacles", frozenset(self.fixed_obstacles))
        object.__setattr__(self, "pickup_cells", frozenset(self.pickup_cells))
        object.__setattr__(self, "wait_cells", frozenset(self.wait_cells))
        if not self.layout_id:
            raise ValueError("layout_id cannot be empty")
        if not self.exits or not self.storage_cells:
            raise ValueError("layout requires exits and storage cells")
        probe = RecoveryState(
            rows=self.rows,
            cols=self.cols,
            traversable=self.traversable,
            storage_cells=self.storage_cells,
            exits=self.exits,
            blocks=(),
            agent_position=self.exits[0],
            fixed_obstacles=self.fixed_obstacles,
            pickup_cells=self.pickup_cells,
            wait_cells=self.wait_cells,
        )
        encode_recovery_state(probe)

    def state(
        self,
        *,
        blocks: Sequence[BlockView],
        agent_position: Cell,
        reserved_cells: Iterable[Cell] = (),
    ) -> RecoveryState:
        return RecoveryState(
            rows=self.rows,
            cols=self.cols,
            traversable=self.traversable,
            storage_cells=self.storage_cells,
            exits=self.exits,
            blocks=tuple(blocks),
            agent_position=agent_position,
            fixed_obstacles=self.fixed_obstacles,
            reserved_cells=frozenset(reserved_cells),
            pickup_cells=self.pickup_cells,
            wait_cells=self.wait_cells,
        )

    def to_dict(self) -> dict:
        return {
            "layout_id": self.layout_id,
            "generator_seed": self.generator_seed,
            "rows": self.rows,
            "cols": self.cols,
            "traversable": _cells_payload(self.traversable),
            "storage_cells": _cells_payload(self.storage_cells),
            "exits": _cells_payload(self.exits),
            "fixed_obstacles": _cells_payload(self.fixed_obstacles),
            "pickup_cells": _cells_payload(self.pickup_cells),
            "wait_cells": _cells_payload(self.wait_cells),
        }


@dataclass(frozen=True)
class ViabilityExample:
    """One self-contained search label and its auditable provenance."""

    example_id: str
    layout_id: str
    state: RecoveryState
    label: DynamicsViabilityLabel
    occupancy_count: int
    search_order: str
    explored_nodes: int
    generated_states: int
    exhaustive: bool
    reason: str
    certificate_contract: str = CLOSED_ADMISSION_CONTRACT
    action_model: str = STRICT_MACRO_ACTION_MODEL

    @property
    def split_group(self) -> str:
        return self.layout_id

    def to_dict(self) -> dict:
        return {
            "protocol": VIABILITY_DATASET_PROTOCOL,
            "example_id": self.example_id,
            "layout_id": self.layout_id,
            "state": recovery_state_to_dict(self.state),
            "label": {
                "status": self.label.status.value,
                "recovery_rank": self.label.recovery_rank,
                "primitive_steps": self.label.primitive_steps,
                "contract": self.label.contract,
            },
            "occupancy_count": int(self.occupancy_count),
            "search": {
                "order": self.search_order,
                "explored_nodes": int(self.explored_nodes),
                "generated_states": int(self.generated_states),
                "exhaustive": bool(self.exhaustive),
                "reason": self.reason,
                "max_primitive_steps": None,
            },
            "certificate_contract": self.certificate_contract,
            "action_model": self.action_model,
            "baseline_viability_teacher": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "ViabilityExample":
        if value.get("protocol") != VIABILITY_DATASET_PROTOCOL:
            raise ValueError("unsupported viability dataset protocol")
        if value.get("baseline_viability_teacher") is not False:
            raise ValueError("dataset must explicitly declare no baseline teacher")
        if value.get("certificate_contract") != CLOSED_ADMISSION_CONTRACT:
            raise ValueError("mixed or unsupported certificate contract")
        if value.get("action_model") != STRICT_MACRO_ACTION_MODEL:
            raise ValueError("mixed or unsupported recovery action model")
        search = value.get("search")
        if not isinstance(search, Mapping) or search.get("max_primitive_steps", "missing") is not None:
            raise ValueError("horizon-conditioned labels are not supported")
        raw_label = value.get("label")
        if not isinstance(raw_label, Mapping) or raw_label.get("contract") != DYNAMICS_LABEL_CONTRACT:
            raise ValueError("mixed or unsupported dynamics label contract")
        try:
            status = ViabilityStatus(raw_label["status"])
        except (KeyError, ValueError) as error:
            raise ValueError("invalid viability label status") from error
        label = DynamicsViabilityLabel(
            status=status,
            recovery_rank=raw_label.get("recovery_rank"),
            primitive_steps=raw_label.get("primitive_steps"),
            contract=raw_label["contract"],
        )
        if status is ViabilityStatus.UNSAFE and search.get("exhaustive") is not True:
            raise ValueError("UNSAFE label must come from exhaustive search")
        state = recovery_state_from_dict(value.get("state"))
        trivial_complete_state = (
            not state.blocks
            and label.recovery_rank == 0.0
            and label.primitive_steps == 0.0
        )
        if (
            label.recovery_rank is not None
            and search.get("order") != "breadth_first"
            and not trivial_complete_state
        ):
            raise ValueError("exact recovery rank requires breadth-first search")
        occupancy_count = value.get("occupancy_count")
        if occupancy_count != len(state.blocks):
            raise ValueError("occupancy_count does not match serialized state")
        example = cls(
            example_id=str(value.get("example_id", "")),
            layout_id=str(value.get("layout_id", "")),
            state=state,
            label=label,
            occupancy_count=int(occupancy_count),
            search_order=str(search.get("order")),
            explored_nodes=int(search.get("explored_nodes")),
            generated_states=int(search.get("generated_states")),
            exhaustive=bool(search.get("exhaustive")),
            reason=str(search.get("reason")),
        )
        if not example.example_id or not example.layout_id:
            raise ValueError("example_id and layout_id must be nonempty")
        canonical = example.to_dict()
        expected_id = (
            f"{example.layout_id}:"
            f"{canonical_sha256(canonical['state'])[:20]}"
        )
        if example.example_id != expected_id:
            raise ValueError("example_id does not authenticate its state")
        return example


def _reachable(start: Cell, traversable: frozenset[Cell]) -> frozenset[Cell]:
    frontier = [start]
    seen = {start}
    while frontier:
        row, col = frontier.pop()
        for neighbor in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if neighbor in traversable and neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return frozenset(seen)


def generate_layouts(
    *,
    grid_sizes: Sequence[tuple[int, int]],
    layouts_per_size: int,
    seed: int,
    max_wall_fraction: float = 0.20,
) -> tuple[ViabilityLayout, ...]:
    """Create deterministic open-yard variants with connected wall changes."""

    if not grid_sizes:
        raise ValueError("at least one grid size is required")
    if layouts_per_size <= 0:
        raise ValueError("layouts_per_size must be positive")
    if not 0.0 <= max_wall_fraction < 1.0:
        raise ValueError("max_wall_fraction must be in [0, 1)")
    layouts = []
    for size_index, (rows, cols) in enumerate(grid_sizes):
        if rows < 4 or cols < 4:
            raise ValueError("grid dimensions must be at least 4")
        for variant in range(layouts_per_size):
            layout_seed = int(seed) + size_index * 1_000_003 + variant * 10_007
            rng = random.Random(layout_seed)
            maximum_exit_width = max(1, min(3, cols - 2))
            exit_width = 1 + (variant % maximum_exit_width)
            env = make_shipyard_env(
                arrival_rate=0.8,
                proc_mean=100.0,
                grid_rows=rows,
                grid_cols=cols,
                exit_width=exit_width,
                number_blocks=1,
            )
            traversable = frozenset(
                (row, col)
                for row in range(rows)
                for col in range(cols)
                if env.rooms[row, col] != "#"
            )
            storage = set(tuple(cell) for cell in env.storage_positions)
            exits = tuple(tuple(cell) for cell in env.exit_cells)
            pickup = frozenset({tuple(env.pickup_cell)})
            wait = frozenset({tuple(env.waiting_cell)})
            fixed = pickup | wait
            protected = set(exits) | set(fixed) | {tuple(env.start_state)}
            candidates = sorted(storage - protected)
            rng.shuffle(candidates)
            maximum_walls = int(math.floor(max_wall_fraction * len(storage)))
            requested_walls = 0 if variant == 0 else rng.randint(0, maximum_walls)
            removed = set()
            for cell in candidates:
                if len(removed) >= requested_walls or len(storage - removed) <= 3:
                    break
                proposed = traversable - removed - {cell} - fixed
                reachable = _reachable(exits[0], frozenset(proposed))
                remaining_storage = storage - removed - {cell}
                if remaining_storage <= reachable and set(exits) <= reachable:
                    removed.add(cell)
            traversable = traversable - removed
            storage = storage - removed
            static_payload = {
                "rows": rows,
                "cols": cols,
                "traversable": _cells_payload(traversable),
                "storage_cells": _cells_payload(storage),
                "exits": _cells_payload(exits),
                "fixed_obstacles": _cells_payload(fixed),
                "pickup_cells": _cells_payload(pickup),
                "wait_cells": _cells_payload(wait),
            }
            signature = canonical_sha256(static_payload)[:16]
            layout = ViabilityLayout(
                layout_id=f"L-{rows}x{cols}-{signature}",
                generator_seed=layout_seed,
                rows=rows,
                cols=cols,
                traversable=frozenset(traversable),
                storage_cells=frozenset(storage),
                exits=exits,
                fixed_obstacles=fixed,
                pickup_cells=pickup,
                wait_cells=wait,
            )
            if any(existing.layout_id == layout.layout_id for existing in layouts):
                # Two requested variants may legitimately yield the same wall
                # pattern.  Keep split groups topologically unique.
                continue
            layouts.append(layout)
    if len(layouts) < 4:
        raise ValueError(
            "generation produced fewer than four unique layouts; increase "
            "layouts_per_size or grid-size diversity"
        )
    return tuple(layouts)


def _configuration_keys(
    layout: ViabilityLayout,
    *,
    state_count: int,
    rng: random.Random,
    reservation_probability: float,
    max_reservations: int,
) -> tuple[tuple[int, Cell, tuple[int, ...]], ...]:
    storage = tuple(sorted(layout.storage_cells))
    count = len(storage)
    exits = tuple(layout.exits)
    keys: set[tuple[int, Cell, tuple[int, ...]]] = set()

    def add(mask: int, agent: Cell) -> None:
        empty = [index for index in range(count) if not mask & (1 << index)]
        reserved: tuple[int, ...] = ()
        if empty and max_reservations > 0 and rng.random() < reservation_probability:
            number = rng.randint(1, min(max_reservations, len(empty)))
            reserved = tuple(sorted(rng.sample(empty, number)))
        keys.add((mask, agent, reserved))

    # Deterministic boundary anchors ensure every layout contains low- and
    # high-load states before random sampling fills the interior strata.
    add(0, exits[0])
    full_mask = (1 << count) - 1
    for agent in tuple(exits) + storage:
        if agent in exits or full_mask & (1 << storage.index(agent)):
            add(full_mask, agent)
    for missing in range(count):
        mask = full_mask ^ (1 << missing)
        agents = tuple(exits) + tuple(
            storage[index] for index in range(count) if mask & (1 << index)
        )
        add(mask, agents[missing % len(agents)])

    if count <= 9:
        all_keys = []
        for occupied_count in range(count + 1):
            for occupied_indices in combinations(range(count), occupied_count):
                mask = sum(1 << index for index in occupied_indices)
                agents = tuple(exits) + tuple(storage[index] for index in occupied_indices)
                for agent in sorted(set(agents)):
                    all_keys.append((mask, agent))
        rng.shuffle(all_keys)
        for mask, agent in all_keys:
            if len(keys) >= state_count:
                break
            add(mask, agent)

    attempts = 0
    maximum_attempts = max(10_000, state_count * 100)
    high_load_floor = max(0, count - 4)
    while len(keys) < state_count and attempts < maximum_attempts:
        attempts += 1
        if rng.random() < 0.75:
            occupied_count = rng.randint(high_load_floor, count)
        else:
            occupied_count = rng.randint(0, count)
        occupied = tuple(sorted(rng.sample(range(count), occupied_count)))
        mask = sum(1 << index for index in occupied)
        agents = tuple(exits) + tuple(storage[index] for index in occupied)
        add(mask, rng.choice(agents))
    return tuple(
        sorted(keys, key=lambda item: (item[0].bit_count(), item[0], item[1], item[2]))
    )[:state_count]


def generate_examples(
    layouts: Sequence[ViabilityLayout],
    *,
    states_per_layout: int,
    seed: int,
    search_max_nodes: Optional[int] = 100_000,
    rank_label_fraction: float = 0.10,
    rank_search_max_nodes: Optional[int] = 100_000,
    reservation_probability: float = 0.20,
    max_reservations: int = 2,
    timing_values: Sequence[float] = (-100.0, -20.0, 0.0, 20.0, 100.0),
    progress: Optional[Callable[[int, int, ViabilityLayout, Counter], None]] = None,
) -> tuple[ViabilityExample, ...]:
    """Generate exact-status labels and selective BFS rank supervision.

    Safety is first checked with goal-directed dynamics search.  A deterministic
    subset of SAFE examples is re-run with BFS; only a successful BFS witness
    supplies the exact recovery-rank target.  Primitive horizons are
    intentionally unsupported so every exhaustive UNSAFE has one semantics.
    """

    layouts = tuple(layouts)
    if not layouts:
        raise ValueError("at least one layout is required")
    if states_per_layout <= 0:
        raise ValueError("states_per_layout must be positive")
    if search_max_nodes is not None and search_max_nodes <= 0:
        raise ValueError("search_max_nodes must be positive or None")
    if rank_search_max_nodes is not None and rank_search_max_nodes <= 0:
        raise ValueError("rank_search_max_nodes must be positive or None")
    if not 0.0 <= rank_label_fraction <= 1.0:
        raise ValueError("rank_label_fraction must be in [0, 1]")
    if not 0.0 <= reservation_probability <= 1.0:
        raise ValueError("reservation_probability must be in [0, 1]")
    if max_reservations < 0:
        raise ValueError("max_reservations must be non-negative")
    timing_values = tuple(float(value) for value in timing_values)
    if not timing_values or any(not math.isfinite(value) for value in timing_values):
        raise ValueError("timing_values must be nonempty and finite")

    examples = []
    total_layouts = len(layouts)
    for layout_index, layout in enumerate(layouts):
        rng = random.Random(int(seed) + layout.generator_seed * 97 + 31)
        keys = _configuration_keys(
            layout,
            state_count=states_per_layout,
            rng=rng,
            reservation_probability=reservation_probability,
            max_reservations=max_reservations,
        )
        storage = tuple(sorted(layout.storage_cells))
        layout_status = Counter()
        for mask, agent, reserved_indices in keys:
            blocks = tuple(
                BlockView(
                    label=f"B{index:03d}",
                    position=cell,
                    remaining_time=rng.choice(timing_values),
                )
                for index, cell in enumerate(storage)
                if mask & (1 << index)
            )
            reserved = tuple(storage[index] for index in reserved_indices)
            state = layout.state(
                blocks=blocks,
                agent_position=agent,
                reserved_cells=reserved,
            )
            state_payload = recovery_state_to_dict(state)
            state_digest = canonical_sha256(state_payload)[:20]
            example_id = f"{layout.layout_id}:{state_digest}"
            goal = analyze_recoverability(
                state,
                max_nodes=search_max_nodes,
                max_primitive_steps=None,
                search_order="goal_directed",
            )
            selected = goal
            rank_draw = int(
                hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:16], 16
            ) / float(0xFFFFFFFFFFFFFFFF)
            if goal.is_safe and rank_draw < rank_label_fraction:
                breadth_first = analyze_recoverability(
                    state,
                    max_nodes=rank_search_max_nodes,
                    max_primitive_steps=None,
                    search_order="breadth_first",
                )
                if breadth_first.is_safe:
                    selected = breadth_first
            label = DynamicsViabilityLabel.from_certificate(selected)
            layout_status[label.status.value] += 1
            examples.append(
                ViabilityExample(
                    example_id=example_id,
                    layout_id=layout.layout_id,
                    state=state,
                    label=label,
                    occupancy_count=len(blocks),
                    search_order=selected.search_order,
                    explored_nodes=selected.explored_nodes,
                    generated_states=selected.generated_states,
                    exhaustive=selected.exhaustive,
                    reason=selected.reason,
                )
            )
        if progress is not None:
            progress(layout_index + 1, total_layouts, layout, layout_status)
    identifiers = [example.example_id for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("dataset generation produced duplicate example ids")
    return tuple(examples)


def _split_capacities(
    group_count: int,
    ratios: Mapping[str, float],
) -> dict[str, int]:
    if group_count < len(ratios):
        raise ValueError("not enough layout groups for all requested splits")
    normalized = {}
    total_ratio = float(sum(ratios.values()))
    if total_ratio <= 0.0:
        raise ValueError("split ratios must have positive sum")
    for name, ratio in ratios.items():
        if not name or not math.isfinite(float(ratio)) or ratio <= 0.0:
            raise ValueError("split names and ratios must be positive")
        normalized[name] = float(ratio) / total_ratio
    capacities = {name: 1 for name in normalized}
    remaining = group_count - len(capacities)
    raw = {name: normalized[name] * remaining for name in normalized}
    for name in capacities:
        capacities[name] += int(math.floor(raw[name]))
    leftover = group_count - sum(capacities.values())
    order = sorted(
        capacities,
        key=lambda name: (-(raw[name] - math.floor(raw[name])), name),
    )
    for name in order[:leftover]:
        capacities[name] += 1
    return capacities


def split_examples_by_layout(
    examples: Sequence[ViabilityExample],
    *,
    ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    seed: int,
) -> tuple[dict[str, tuple[ViabilityExample, ...]], dict]:
    """Assign complete layouts using deterministic status-aware balancing."""

    examples = tuple(examples)
    if not examples:
        raise ValueError("cannot split an empty dataset")
    grouped: dict[str, list[ViabilityExample]] = defaultdict(list)
    for example in examples:
        grouped[example.layout_id].append(example)
    capacities = _split_capacities(len(grouped), ratios)
    total_unsafe = sum(
        example.label.status is ViabilityStatus.UNSAFE for example in examples
    )
    total_known = sum(example.label.has_safety_target for example in examples)
    ratio_total = float(sum(ratios.values()))
    target_unsafe = {
        name: total_unsafe * float(ratios[name]) / ratio_total for name in ratios
    }
    target_known = {
        name: total_known * float(ratios[name]) / ratio_total for name in ratios
    }
    group_stats = {}
    for layout_id, values in grouped.items():
        group_stats[layout_id] = {
            "unsafe": sum(
                value.label.status is ViabilityStatus.UNSAFE for value in values
            ),
            "known": sum(value.label.has_safety_target for value in values),
            "tie": canonical_sha256({"seed": int(seed), "layout_id": layout_id}),
        }
    ordered = sorted(
        grouped,
        key=lambda layout_id: (
            -group_stats[layout_id]["unsafe"],
            -group_stats[layout_id]["known"],
            group_stats[layout_id]["tie"],
        ),
    )
    assigned = {name: [] for name in ratios}
    running_unsafe = Counter()
    running_known = Counter()
    for layout_id in ordered:
        eligible = [
            name for name in ratios if len(assigned[name]) < capacities[name]
        ]
        if not eligible:  # pragma: no cover - capacity arithmetic guards this
            raise RuntimeError("split capacities were exhausted early")

        def priority(name: str):
            unsafe_deficit = (
                target_unsafe[name] - running_unsafe[name]
            ) / max(target_unsafe[name], 1.0)
            known_deficit = (
                target_known[name] - running_known[name]
            ) / max(target_known[name], 1.0)
            capacity_fraction = (
                capacities[name] - len(assigned[name])
            ) / capacities[name]
            return (unsafe_deficit, known_deficit, capacity_fraction, name)

        chosen = max(eligible, key=priority)
        assigned[chosen].append(layout_id)
        running_unsafe[chosen] += group_stats[layout_id]["unsafe"]
        running_known[chosen] += group_stats[layout_id]["known"]

    splits = {}
    split_payload = {}
    all_ids = set()
    for name in ratios:
        layout_ids = tuple(sorted(assigned[name]))
        values = tuple(
            example
            for layout_id in layout_ids
            for example in sorted(
                grouped[layout_id], key=lambda item: item.example_id
            )
        )
        current_ids = {value.example_id for value in values}
        if all_ids.intersection(current_ids):
            raise RuntimeError("split assignment leaked an example")
        all_ids.update(current_ids)
        statuses = Counter(value.label.status.value for value in values)
        splits[name] = values
        split_payload[name] = {
            "layout_ids": list(layout_ids),
            "layout_count": len(layout_ids),
            "example_count": len(values),
            "status_counts": dict(sorted(statuses.items())),
            "exact_rank_count": sum(
                value.label.recovery_rank is not None for value in values
            ),
        }
    payload = {
        "protocol": VIABILITY_SPLIT_PROTOCOL,
        "seed": int(seed),
        "ratios": {name: float(ratios[name]) for name in ratios},
        "splits": split_payload,
    }
    payload["split_sha256"] = canonical_sha256(payload)
    return splits, payload


def stratified_bootstrap_indices(
    examples: Sequence[ViabilityExample],
    *,
    seed: int,
) -> dict[str, tuple[int, ...]]:
    """Return independent with-replacement SAFE and UNSAFE bootstrap pools."""

    safe = [
        index
        for index, example in enumerate(examples)
        if example.label.status is ViabilityStatus.SAFE
    ]
    unsafe = [
        index
        for index, example in enumerate(examples)
        if example.label.status is ViabilityStatus.UNSAFE
    ]
    if not safe or not unsafe:
        raise ValueError("training split requires both SAFE and UNSAFE labels")
    rng = random.Random(int(seed))
    return {
        "safe": tuple(rng.choice(safe) for _ in range(len(safe))),
        "unsafe": tuple(rng.choice(unsafe) for _ in range(len(unsafe))),
    }


def dataset_manifest(
    *,
    layouts: Sequence[ViabilityLayout],
    examples: Sequence[ViabilityExample],
    split_manifest: Mapping,
    generation_config: Mapping,
    timing_scale: float,
) -> dict:
    """Build the protocol record bound into every critic checkpoint."""

    statuses = Counter(example.label.status.value for example in examples)
    layout_payloads = [layout.to_dict() for layout in sorted(layouts, key=lambda item: item.layout_id)]
    manifest = {
        "protocol": VIABILITY_DATASET_PROTOCOL,
        "baseline_viability_teacher": False,
        "exact_verifier_authoritative": True,
        "critic_certificate_authority": False,
        "unknown_supervision": "masked",
        "label_contract": DYNAMICS_LABEL_CONTRACT,
        "certificate_contract": CLOSED_ADMISSION_CONTRACT,
        "action_model": STRICT_MACRO_ACTION_MODEL,
        "max_primitive_steps": None,
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "node_feature_dim": NODE_FEATURE_DIM,
        "timing_scale": float(timing_scale),
        "generation_config": dict(generation_config),
        "layouts": layout_payloads,
        "layout_count": len(layout_payloads),
        "example_count": len(examples),
        "status_counts": dict(sorted(statuses.items())),
        "exact_rank_count": sum(
            example.label.recovery_rank is not None for example in examples
        ),
        "split_manifest": dict(split_manifest),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def write_dataset(
    path: Path,
    examples: Sequence[ViabilityExample],
) -> str:
    """Write canonical JSONL and return its byte-level SHA-256."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for example in sorted(examples, key=lambda item: item.example_id):
            line = (
                json.dumps(
                    example.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            handle.write(line)
            digest.update(line)
    return digest.hexdigest()


def read_dataset(path: Path) -> tuple[ViabilityExample, ...]:
    path = Path(path)
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                examples.append(ViabilityExample.from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid viability dataset record at line {line_number}: {error}"
                ) from error
    if not examples:
        raise ValueError("viability dataset is empty")
    identifiers = [example.example_id for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("viability dataset contains duplicate example ids")
    return tuple(examples)


__all__ = [
    "DEFAULT_SPLIT_RATIOS",
    "VIABILITY_DATASET_PROTOCOL",
    "VIABILITY_SPLIT_PROTOCOL",
    "ViabilityExample",
    "ViabilityLayout",
    "canonical_sha256",
    "dataset_manifest",
    "generate_examples",
    "generate_layouts",
    "read_dataset",
    "recovery_state_from_dict",
    "recovery_state_to_dict",
    "split_examples_by_layout",
    "stratified_bootstrap_indices",
    "write_dataset",
]
