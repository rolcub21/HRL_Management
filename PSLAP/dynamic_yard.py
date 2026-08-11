"""Read-only yard planning for the repaired dynamic PSLAP baseline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from typing import Iterable, Optional


Cell = tuple[int, int]


@dataclass(frozen=True)
class BlockView:
    """Planning-only block data detached from the live environment object."""

    label: str
    position: Cell
    remaining_time: float


@dataclass(frozen=True)
class YardSnapshot:
    """Immutable description of traversable cells, exits, and stored blocks."""

    rows: int
    cols: int
    traversable: frozenset[Cell]
    storage_cells: frozenset[Cell]
    exits: tuple[Cell, ...]
    blocks: tuple[BlockView, ...]

    @classmethod
    def from_env(cls, env) -> "YardSnapshot":
        traversable = frozenset(
            (row, col)
            for row in range(env.grid_rows)
            for col in range(env.grid_cols)
            if env.rooms[row, col] != "#"
        )
        blocks = tuple(
            BlockView(
                label=block.label,
                position=block.position,
                remaining_time=float(
                    env.signed_remaining_storage_time(block)
                    if hasattr(env, "signed_remaining_storage_time")
                    else block.get_remaining_storage_time()
                ),
            )
            for block in env.blocks
            if (
                block.stored
                and not block.delivered
                and not block.carrying
                and block.position in env.storage_positions
            )
        )
        return cls(
            rows=env.grid_rows,
            cols=env.grid_cols,
            traversable=traversable,
            storage_cells=frozenset(env.storage_positions),
            exits=tuple(env.exit_cells),
            blocks=blocks,
        )

    def occupancy(self) -> dict[Cell, BlockView]:
        return {block.position: block for block in self.blocks}

    def block(self, label: str) -> Optional[BlockView]:
        return next((block for block in self.blocks if block.label == label), None)

    def without(self, label: str) -> "YardSnapshot":
        return replace(
            self,
            blocks=tuple(block for block in self.blocks if block.label != label),
        )

    def with_block(self, block: BlockView, position: Cell) -> "YardSnapshot":
        base = self.without(block.label)
        return replace(base, blocks=base.blocks + (replace(block, position=position),))


@dataclass(frozen=True)
class ObstructionPath:
    cells: tuple[Cell, ...]
    exit_cell: Cell
    obstruction_labels: tuple[str, ...]

    @property
    def obstruction_count(self) -> int:
        return len(self.obstruction_labels)


def find_min_obstruction_path(
    yard: YardSnapshot, block_label: str
) -> Optional[ObstructionPath]:
    """Return a real-exit path minimizing blockers, then travel distance."""

    target = yard.block(block_label)
    return (
        None
        if target is None
        else find_min_obstruction_route(
            yard,
            target.position,
            yard.exits,
            ignore_labels=(block_label,),
        )
    )


def find_min_obstruction_route(
    yard: YardSnapshot,
    start: Cell,
    goals: Iterable[Cell],
    *,
    ignore_labels: Iterable[str] = (),
) -> Optional[ObstructionPath]:
    """Find a route minimizing distinct blockers and then travel distance."""

    goal_set = frozenset(goals)
    if start not in yard.traversable or not goal_set:
        return None
    if start in goal_set:
        return ObstructionPath((start,), start, ())

    occupied = yard.occupancy()
    ignored = frozenset(ignore_labels)
    queue = []
    serial = 0
    heappush(queue, (0, 0, serial, start, (start,), ()))
    best: dict[Cell, tuple[int, int]] = {start: (0, 0)}

    while queue:
        count, distance, _, cell, path, labels = heappop(queue)
        if best.get(cell) != (count, distance):
            continue
        if cell in goal_set:
            return ObstructionPath(path, cell, labels)

        row, col = cell
        for neighbor in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if neighbor not in yard.traversable:
                continue
            next_labels = labels
            occupant = occupied.get(neighbor)
            if (
                occupant is not None
                and occupant.label not in ignored
                and occupant.label not in labels
            ):
                next_labels = labels + (occupant.label,)
            next_cost = (len(next_labels), distance + 1)
            if next_cost >= best.get(neighbor, (10**9, 10**9)):
                continue
            best[neighbor] = next_cost
            serial += 1
            heappush(
                queue,
                (
                    next_cost[0],
                    next_cost[1],
                    serial,
                    neighbor,
                    path + (neighbor,),
                    next_labels,
                ),
            )

    return None


def shortest_clear_path(
    yard: YardSnapshot,
    start: Cell,
    goal: Cell,
    *,
    ignore_labels: Iterable[str] = (),
) -> Optional[tuple[Cell, ...]]:
    """Return a shortest path that does not cross walls or stored blocks."""

    if start not in yard.traversable or goal not in yard.traversable:
        return None
    ignored = frozenset(ignore_labels)
    occupied = {
        cell
        for cell, block in yard.occupancy().items()
        if block.label not in ignored and cell not in (start, goal)
    }
    queue = deque([start])
    parent: dict[Cell, Optional[Cell]] = {start: None}

    while queue:
        cell = queue.popleft()
        if cell == goal:
            path = []
            cursor: Optional[Cell] = goal
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            return tuple(reversed(path))
        row, col = cell
        for neighbor in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if (
                neighbor in yard.traversable
                and neighbor not in occupied
                and neighbor not in parent
            ):
                parent[neighbor] = cell
                queue.append(neighbor)
    return None


def _has_clear_exit_path(yard: YardSnapshot, block: BlockView) -> bool:
    return any(
        shortest_clear_path(
            yard,
            block.position,
            exit_cell,
            ignore_labels=(block.label,),
        )
        is not None
        for exit_cell in yard.exits
    )


def _preserves_earlier_access(
    yard: YardSnapshot, moving: BlockView, candidate: Cell
) -> bool:
    """Reject placements that newly trap a block with an earlier deadline."""

    base = yard.without(moving.label)
    placed = base.with_block(moving, candidate)
    for other in base.blocks:
        if other.remaining_time >= moving.remaining_time:
            continue
        if _has_clear_exit_path(base, other) and not _has_clear_exit_path(
            placed, other
        ):
            return False
    return True


def select_storage_location(
    yard: YardSnapshot,
    block: BlockView,
    *,
    source: Cell,
    reserved_cells: Iterable[Cell] = (),
) -> Optional[Cell]:
    """Choose a reachable free cell without sacrificing earlier-block access."""

    occupied = yard.occupancy()
    reserved = frozenset(reserved_cells)
    is_relocation = yard.block(block.label) is not None
    candidates = []
    for candidate in sorted(yard.storage_cells):
        if not is_storage_location_feasible(
            yard,
            block,
            candidate,
            source=source,
            reserved_cells=reserved,
        ):
            continue
        path = shortest_clear_path(
            yard, source, candidate, ignore_labels=(block.label,)
        )
        exit_distance = min(
            abs(candidate[0] - exit_cell[0])
            + abs(candidate[1] - exit_cell[1])
            for exit_cell in yard.exits
        )
        travel_distance = len(path) - 1
        if is_relocation:
            score = (travel_distance, exit_distance, candidate)
        else:
            score = (exit_distance, travel_distance, candidate)
        candidates.append((score, candidate))

    return min(candidates)[1] if candidates else None


def is_storage_location_feasible(
    yard: YardSnapshot,
    block: BlockView,
    candidate: Cell,
    *,
    source: Cell,
    reserved_cells: Iterable[Cell] = (),
) -> bool:
    """Validate a proposed assignment under the shared yard contracts."""

    if not is_storage_location_admissible(
        yard,
        block,
        candidate,
        source=source,
        reserved_cells=reserved_cells,
    ):
        return False
    return _preserves_earlier_access(yard, block, candidate)


def is_storage_location_admissible(
    yard: YardSnapshot,
    block: BlockView,
    candidate: Cell,
    *,
    source: Cell,
    reserved_cells: Iterable[Cell] = (),
) -> bool:
    """Check physical validity without applying a strategy-specific screen."""

    if candidate not in yard.storage_cells:
        return False
    if candidate in yard.occupancy() or candidate in frozenset(reserved_cells):
        return False
    if shortest_clear_path(
        yard, source, candidate, ignore_labels=(block.label,)
    ) is None:
        return False
    return True


def select_due_plan(
    yard: YardSnapshot, due_labels: Iterable[str]
) -> Optional[tuple[BlockView, ObstructionPath]]:
    """Choose among all due blocks using obstruction count as primary cost."""

    candidates = []
    for label in due_labels:
        block = yard.block(label)
        plan = find_min_obstruction_path(yard, label)
        if block is None or plan is None:
            continue
        score = (
            plan.obstruction_count,
            block.remaining_time,
            len(plan.cells),
            block.label,
        )
        candidates.append((score, block, plan))
    if not candidates:
        return None
    _, block, plan = min(candidates, key=lambda item: item[0])
    return block, plan


__all__ = [
    "BlockView",
    "Cell",
    "ObstructionPath",
    "YardSnapshot",
    "find_min_obstruction_path",
    "find_min_obstruction_route",
    "is_storage_location_admissible",
    "is_storage_location_feasible",
    "select_due_plan",
    "select_storage_location",
    "shortest_clear_path",
]
