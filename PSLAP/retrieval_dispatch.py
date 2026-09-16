"""Shared travel-time-aware retrieval scheduling for PSLAP baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PSLAP.dynamic_yard import (
    Cell,
    YardSnapshot,
    find_min_obstruction_path,
    find_min_obstruction_route,
    select_storage_location,
    shortest_clear_path,
)


RelocationSelector = Callable[..., Optional[Cell]]


@dataclass(frozen=True)
class RelocationPlan:
    """Move one obstructing block to a temporary storage cell."""

    block_label: str
    target: Cell


@dataclass(frozen=True)
class RetrievalPlan:
    """A complete placement-independent plan for one stored block."""

    block_label: str
    exit_cell: Cell
    relocations: tuple[RelocationPlan, ...]
    estimated_steps: int
    remaining_time: float

    @property
    def slack(self) -> float:
        """Steps left after accounting for the complete retrieval plan."""

        return self.remaining_time - self.estimated_steps


def _movement_steps(path: Optional[tuple[Cell, ...]]) -> Optional[int]:
    if path is None:
        return None
    return len(path) - 1


def plan_retrieval(
    yard: YardSnapshot,
    agent_position: Cell,
    block_label: str,
    *,
    relocation_selector: Optional[RelocationSelector] = None,
) -> Optional[RetrievalPlan]:
    """Build a feasible plan and count every movement and handling action.

    The returned duration includes travel to each block, PICKUP and PUTDOWN for
    relocations, and PICKUP and PUTDOWN for final delivery. Planning mutates only
    immutable snapshots, never live environment blocks.
    """

    if relocation_selector is None:
        relocation_selector = select_storage_location

    target = yard.block(block_label)
    if target is None:
        return None

    working = yard
    cursor = agent_position
    estimated_steps = 0
    relocations = []

    # Iteratively peel a reachable blocker from either the agent's approach or
    # the target's egress. This handles narrow gates where the agent itself is
    # separated from a due block by stored inventory.
    for _ in range(len(yard.blocks)):
        target = working.block(block_label)
        if target is None:
            return None
        approach = find_min_obstruction_route(
            working,
            cursor,
            (target.position,),
            ignore_labels=(target.label,),
        )
        egress = find_min_obstruction_path(working, block_label)
        if approach is None or egress is None:
            return None
        if not approach.obstruction_labels and not egress.obstruction_labels:
            break

        blocker_labels = tuple(
            dict.fromkeys(
                approach.obstruction_labels + egress.obstruction_labels
            )
        )
        reserved_cells = frozenset(approach.cells + egress.cells)
        choices = []
        for blocker_label in blocker_labels:
            blocker = working.block(blocker_label)
            if blocker is None:
                continue
            to_blocker = shortest_clear_path(
                working,
                cursor,
                blocker.position,
                ignore_labels=(blocker.label,),
            )
            if to_blocker is None:
                continue
            relocation_target = relocation_selector(
                working,
                blocker,
                source=blocker.position,
                reserved_cells=reserved_cells,
            )
            if relocation_target is None:
                continue
            with_block = shortest_clear_path(
                working,
                blocker.position,
                relocation_target,
                ignore_labels=(blocker.label,),
            )
            if with_block is None:
                continue
            score = (
                len(to_blocker),
                len(with_block),
                blocker.remaining_time,
                blocker.label,
            )
            choices.append(
                (score, blocker, relocation_target, to_blocker, with_block)
            )
        if not choices:
            return None

        _, blocker, relocation_target, to_blocker, with_block = min(
            choices, key=lambda item: item[0]
        )
        estimated_steps += (
            len(to_blocker) - 1 + 1 + len(with_block) - 1 + 1
        )
        relocations.append(RelocationPlan(blocker.label, relocation_target))
        working = working.with_block(blocker, relocation_target)
        cursor = relocation_target
    else:
        return None

    refreshed = find_min_obstruction_path(working, block_label)
    target = working.block(block_label)
    if target is None or refreshed is None or refreshed.obstruction_labels:
        return None

    to_target = shortest_clear_path(
        working,
        cursor,
        target.position,
        ignore_labels=(target.label,),
    )
    to_exit = shortest_clear_path(
        working,
        target.position,
        refreshed.exit_cell,
        ignore_labels=(target.label,),
    )
    approach_steps = _movement_steps(to_target)
    delivery_steps = _movement_steps(to_exit)
    if approach_steps is None or delivery_steps is None:
        return None

    estimated_steps += approach_steps + 1 + delivery_steps + 1
    return RetrievalPlan(
        block_label=block_label,
        exit_cell=refreshed.exit_cell,
        relocations=tuple(relocations),
        estimated_steps=estimated_steps,
        remaining_time=target.remaining_time,
    )


def select_dispatchable_retrieval(
    yard: YardSnapshot,
    agent_position: Cell,
    *,
    relocation_selector: Optional[RelocationSelector] = None,
) -> Optional[RetrievalPlan]:
    """Return the most urgent plan whose predicted delivery is due now."""

    candidates = []
    for block in yard.blocks:
        plan = plan_retrieval(
            yard,
            agent_position,
            block.label,
            relocation_selector=relocation_selector,
        )
        if plan is None or plan.slack > 0:
            continue
        score = (
            plan.slack,
            len(plan.relocations),
            plan.estimated_steps,
            plan.block_label,
        )
        candidates.append((score, plan))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


__all__ = [
    "RelocationSelector",
    "RelocationPlan",
    "RetrievalPlan",
    "plan_retrieval",
    "select_dispatchable_retrieval",
]
