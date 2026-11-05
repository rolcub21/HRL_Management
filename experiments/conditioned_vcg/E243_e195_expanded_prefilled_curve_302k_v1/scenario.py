"""Certified-retrievable prefilled 5x5 continuation states for E243."""

from __future__ import annotations

from dataclasses import replace
import random
from statistics import fmean

from example.small_rooms_env import SmallRoomsEnv
from experiments.conditioned_vcg.E195_e191_method_5x5_longstay_254k_v1 import (
    scenario as e195,
)


SCENARIO = "prefilled_5x5_n8_longstay_expanded_certification_curve"
CAPACITY = e195.CAPACITY
TOTAL_JOBS = e195.TOTAL_JOBS
MEAN_DURATION = e195.MEAN_DURATION
MAX_STEPS = e195.MAX_STEPS
INITIAL_FILLS = (4, 5, 6, 7)
CASES_PER_FILL = 25
FUTURE_ARRIVAL_GAP = 40
SEED_BASE = 302_243_000
SEEDS = tuple(
    SEED_BASE + fill_index * 100 + replicate
    for fill_index in range(len(INITIAL_FILLS))
    for replicate in range(CASES_PER_FILL)
)

# E200 established an exact completion witness with these seven occupied cells.
# Every E243 layout is a deterministic shuffled subset, so the denser endpoint
# is the audited E200 geometry and lower fills only remove stored blocks.
SAFE_PLACEMENT_POOL = (
    (1, 1), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3),
)
TRANSPORTER_CELLS = ((1, 1), (2, 2), (2, 3))


def initial_fill(seed: int) -> int:
    value = int(seed)
    if value not in SEEDS:
        raise ValueError(f"seed is outside the E243 panel: {seed}")
    return INITIAL_FILLS[(value - SEED_BASE) // 100]


def replicate(seed: int) -> int:
    initial_fill(seed)
    return int(seed) % 100


def placement_cells(seed: int) -> tuple[tuple[int, int], ...]:
    """Shuffle labels over an E200-witnessed geometry, independently of stays."""

    count = initial_fill(seed)
    cells = list(SAFE_PLACEMENT_POOL)
    random.Random(int(seed) + 302_243_911).shuffle(cells)
    return tuple(cells[:count])


def transporter_cell(seed: int) -> tuple[int, int]:
    return TRANSPORTER_CELLS[replicate(seed) % len(TRANSPORTER_CELLS)]


def duration_seed(seed: int) -> int:
    # Match the duration replicate across all four starting-fill levels.
    return 302_243_500 + replicate(seed)


def workload_metadata(instance) -> dict:
    fill = initial_fill(int(instance.seed))
    future = TOTAL_JOBS - fill
    arrivals = tuple(int(value) for value in instance.arrival_steps)
    return {
        "panel": "development_prefilled_certification_curve",
        "target_nominal_occupancy": fill / CAPACITY,
        "target_label": f"{fill}/8-preloaded",
        "arrival_gap_steps": FUTURE_ARRIVAL_GAP,
        "arrival_rate": float(instance.arrival_rate),
        "initial_stored": fill,
        "initial_occupancy_fraction": fill / CAPACITY,
        "initial_occupancy_label": f"{100.0 * fill / CAPACITY:.1f}%",
        "initial_placement_cells": [list(cell) for cell in placement_cells(instance.seed)],
        "transporter_cell": list(transporter_cell(instance.seed)),
        "jobs": TOTAL_JOBS,
        "future_arrivals": future,
        "future_arrival_gap_steps": FUTURE_ARRIVAL_GAP,
        "capacity": CAPACITY,
        "mean_duration_parameter": MEAN_DURATION,
        "realized_duration_mean": float(fmean(instance.storage_steps_needed)),
        "nominal_occupancy_from_declared_mean": fill / CAPACITY,
        "last_release_step": int(arrivals[-1]),
        "initial_state_selection": (
            "deterministic shuffled subset of E200 witnessed geometry; exact "
            "completion witness required before evaluation"
        ),
    }


class Prefilled5x5Env(e195.LongStay5x5Env):
    """E195 dynamics/objective initialized from a reproducible crowded state."""

    def __init__(self, name, objective):
        if name != SCENARIO:
            raise ValueError(f"unknown E243 scenario: {name}")
        super().__init__(e195.SCENARIO, objective)
        self.target_scenario_name = name

    def sample_episode_instance(self, seed=None):
        if seed is None:
            raise ValueError("E243 requires an explicit instance seed")
        seed = int(seed)
        fill = initial_fill(seed)
        self.arrival_rate = 1.0 / float(FUTURE_ARRIVAL_GAP)
        self.proc_mean = float(MEAN_DURATION)
        native = SmallRoomsEnv.sample_episode_instance(self, seed=duration_seed(seed))
        future = TOTAL_JOBS - fill
        arrivals = (0,) * fill + tuple(
            1 + FUTURE_ARRIVAL_GAP * index for index in range(future)
        )
        instance = replace(
            native,
            seed=seed,
            arrival_rate=float(self.arrival_rate),
            proc_mean=float(MEAN_DURATION),
            arrival_steps=arrivals,
        )
        instance.validate_for(self)
        return instance

    def reset(self, instance=None):
        if instance is None:
            raise ValueError("E243 requires an explicit panel instance")
        state = super().reset(instance=instance)
        fill = initial_fill(int(instance.seed))
        cells = placement_cells(int(instance.seed))
        for index, cell in enumerate(cells):
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

        # This is a continuation boundary. Objective accounting begins here;
        # no unreported primitive rewards are charged for constructing it.
        self.time_steps = 1
        self.current_state = transporter_cell(int(instance.seed))
        self.update_pickup_queue()
        self.e195_workload = workload_metadata(instance)
        self.e195_stored_cell_steps = 0
        self.e195_queue_job_steps = 0
        self.e195_equipment_steps = 0
        self.e195_arrival_window_stored_steps = 0
        self.e195_arrival_window_steps = 0
        self.e195_steady_window_stored_steps = 0
        self.e195_steady_window_steps = 0
        self.e195_peak_queue = 0
        if sum(bool(block.stored) for block in self.blocks) != fill:
            raise RuntimeError("E243 reset did not materialize the declared fill")
        return self.get_current_state()


__all__ = [
    "CAPACITY", "CASES_PER_FILL", "FUTURE_ARRIVAL_GAP", "INITIAL_FILLS",
    "MAX_STEPS", "MEAN_DURATION", "Prefilled5x5Env", "SCENARIO", "SEEDS",
    "TOTAL_JOBS", "duration_seed", "initial_fill", "placement_cells",
    "replicate", "transporter_cell", "workload_metadata",
]
