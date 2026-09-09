"""Utilities for executing verifier-certified cell paths."""

from __future__ import annotations


def actions_from_cell_path(env, path, *, start, goal):
    """Convert one exact adjacent-cell path into primitive movement actions."""

    cells = tuple(tuple(cell) for cell in path)
    start = tuple(start)
    goal = tuple(goal)
    if not cells or cells[0] != start or cells[-1] != goal:
        raise ValueError("certified path endpoints do not match the bound macro")
    actions = []
    action_for_delta = {
        (-1, 0): env.ACTION_IDS["UP"],
        (1, 0): env.ACTION_IDS["DOWN"],
        (0, -1): env.ACTION_IDS["LEFT"],
        (0, 1): env.ACTION_IDS["RIGHT"],
    }
    for left, right in zip(cells, cells[1:]):
        delta = (right[0] - left[0], right[1] - left[1])
        if delta not in action_for_delta:
            raise ValueError("certified path contains a non-adjacent transition")
        if env.rooms[right[0], right[1]] == "#":
            raise ValueError("certified path crosses a live wall")
        actions.append(action_for_delta[delta])
    return actions


__all__ = ["actions_from_cell_path"]
