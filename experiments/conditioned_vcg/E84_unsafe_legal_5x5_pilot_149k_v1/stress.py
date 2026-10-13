"""Deterministic dense 5x5 starts for the legal-proposal stress pilot."""

from __future__ import annotations

from dataclasses import replace

from example.small_rooms_env import SmallRoomsEnv
from vcg_objective_audit import ObjectiveAuditSmallRoomsEnv


def stress_instance(env: ObjectiveAuditSmallRoomsEnv, seed: int, occupied: int):
    if not 1 <= occupied < env.number_blocks:
        raise ValueError("occupied must leave at least one arriving job")
    base = SmallRoomsEnv.sample_episode_instance(env, seed)
    return replace(
        base,
        arrival_steps=(0,) * occupied + (1,) * (env.number_blocks - occupied),
    )


class DenseStartEnv(ObjectiveAuditSmallRoomsEnv):
    """The first K jobs are already stored at time zero in declared cells."""

    def __init__(self, *, placement_cells, agent_cell, **kwargs):
        self.placement_cells = tuple(tuple(cell) for cell in placement_cells)
        self.agent_cell = tuple(agent_cell)
        super().__init__(**kwargs)
        if len(self.placement_cells) >= self.number_blocks:
            raise ValueError("dense start must leave an arriving job")
        if len(self.placement_cells) != len(set(self.placement_cells)):
            raise ValueError("dense placement contains duplicate cells")
        if not set(self.placement_cells) <= set(self.storage_positions):
            raise ValueError("dense placement is outside storage")
        row, col = self.agent_cell
        if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols
                and self.rooms[row][col] != '#'):
            raise ValueError("dense transporter cell is not traversable")

    def sample_episode_instance(self, seed=None):
        if seed is None:
            raise ValueError("E84 requires a deterministic instance seed")
        return stress_instance(self, int(seed), len(self.placement_cells))

    def reset(self, instance=None):
        if instance is None:
            raise ValueError("E84 requires an explicit instance")
        if tuple(instance.arrival_steps[:len(self.placement_cells)]) != (0,) * len(self.placement_cells):
            raise ValueError("dense-start jobs must arrive at time zero")
        state = super().reset(instance=instance)
        for index, cell in enumerate(self.placement_cells):
            block = self.blocks[index]
            block.position = cell
            block.storage_location = cell
            block.carrying = False
            block.stored = True
            block.delivered = False
            block.picked = True
            block.stored_time_step = 0
            block.storage_steps_elapsed = 0
            self.storage_counts[cell] = 1
        self.current_state = self.agent_cell
        self.update_pickup_queue()
        return state
