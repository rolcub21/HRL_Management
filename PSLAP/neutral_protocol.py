"""Strategy-neutral physical candidate and relocation rules."""

from __future__ import annotations

from typing import Optional

from PSLAP.dynamic_yard import (
    BlockView,
    Cell,
    YardSnapshot,
    is_storage_location_admissible,
    shortest_clear_path,
)


NEUTRAL_RELOCATION_SELECTOR_VERSION = "nearest_feasible_path_then_cell_v1"


def shared_candidate_mask(
    yard: YardSnapshot,
    block: BlockView,
    source: Cell,
    *,
    reserved_cells=(),
) -> tuple[Cell, ...]:
    """Return the common physical storage mask used by every method."""

    return tuple(
        candidate
        for candidate in sorted(yard.storage_cells)
        if is_storage_location_admissible(
            yard,
            block,
            candidate,
            source=source,
            reserved_cells=reserved_cells,
        )
    )


def nearest_from_candidates(
    yard: YardSnapshot,
    block: BlockView,
    source: Cell,
    candidates: tuple[Cell, ...],
) -> Optional[Cell]:
    choices = []
    for candidate in candidates:
        path = shortest_clear_path(
            yard, source, candidate, ignore_labels=(block.label,)
        )
        if path is not None:
            choices.append(((len(path) - 1, candidate), candidate))
    return min(choices)[1] if choices else None


def neutral_relocation_selector(
    yard: YardSnapshot,
    block: BlockView,
    *,
    source: Cell,
    reserved_cells=(),
) -> Optional[Cell]:
    """Fixed downstream relocation rule: shortest path, then cell order."""

    candidates = shared_candidate_mask(
        yard,
        block,
        source,
        reserved_cells=reserved_cells,
    )
    return nearest_from_candidates(yard, block, source, candidates)


__all__ = [
    "NEUTRAL_RELOCATION_SELECTOR_VERSION",
    "nearest_from_candidates",
    "neutral_relocation_selector",
    "shared_candidate_mask",
]
