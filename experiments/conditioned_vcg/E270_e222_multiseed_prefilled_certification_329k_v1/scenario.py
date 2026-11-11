"""Witness-screened prefilled 8x8 continuation states for E270."""

from __future__ import annotations

from dataclasses import replace
import random
from statistics import fmean

from example.episode_instance import EpisodeInstance
from example.small_rooms_env import SmallRoomsEnv
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    scenario as e191,
)


SCENARIO = "prefilled_8x8_n36_longstay_multiseed_certification_curve"
CAPACITY = e191.CAPACITY
TOTAL_JOBS = e191.TOTAL_JOBS
MEAN_DURATION = e191.MEAN_DURATION
MAX_STEPS = e191.MAX_STEPS
INITIAL_FILLS = (18, 25, 32)
CASES_PER_FILL = 10
FUTURE_ARRIVAL_GAP = 40
SEED_BASE = 329_270_000
SEEDS = tuple(
    SEED_BASE + fill_index * 100 + replicate_index
    for fill_index in range(len(INITIAL_FILLS))
    for replicate_index in range(CASES_PER_FILL)
)
GATE_SEED = 329_270_900
GATE_FILL = 18
TRANSPORTER_CELL = (1, 1)
MAX_CANDIDATE_ATTEMPTS = 1_000


def initial_fill(seed: int) -> int:
    value = int(seed)
    if value not in SEEDS:
        raise ValueError(f"seed is outside the E270 panel: {seed}")
    return INITIAL_FILLS[(value - SEED_BASE) // 100]


def replicate(seed: int) -> int:
    initial_fill(seed)
    return int(seed) % 100


def duration_seed(seed: int) -> int:
    """Pair storage-duration streams across the three fill levels."""

    if int(seed) == GATE_SEED:
        return 329_270_950
    return 329_270_500 + replicate(seed)


def candidate_layout(seed: int, fill: int, attempt: int) -> tuple[tuple[int, int], ...]:
    """Draw one deterministic random labeling/layout, independent of stays."""

    base = e191.SCENARIO_SPEC.make_base_env()
    cells = list(base.storage_positions)
    random.Random(int(seed) * 1_000_003 + int(attempt) * 97_409 + 270).shuffle(cells)
    return tuple(tuple(cell) for cell in cells[: int(fill)])


def candidate_record(seed: int, fill: int, attempt: int) -> dict:
    return {
        "seed": int(seed),
        "initial_fill": int(fill),
        "replicate": None if int(seed) == GATE_SEED else replicate(seed),
        "candidate_attempt": int(attempt),
        "duration_seed": duration_seed(seed),
        "initial_layout": [list(cell) for cell in candidate_layout(seed, fill, attempt)],
        "transporter_cell": list(TRANSPORTER_CELL),
        "future_arrival_gap_steps": FUTURE_ARRIVAL_GAP,
    }


def workload_metadata(instance: EpisodeInstance, record: dict) -> dict:
    fill = int(record["initial_fill"])
    return {
        "panel": "witness_screened_prefilled_certification_curve",
        "target_nominal_occupancy": fill / CAPACITY,
        "target_label": f"{fill}/35-preloaded",
        "arrival_gap_steps": FUTURE_ARRIVAL_GAP,
        "arrival_rate": float(instance.arrival_rate),
        "initial_stored": fill,
        "initial_occupancy_fraction": fill / CAPACITY,
        "initial_occupancy_label": f"{100.0 * fill / CAPACITY:.1f}%",
        "initial_placement_cells": list(record["initial_layout"]),
        "transporter_cell": list(record["transporter_cell"]),
        "jobs": TOTAL_JOBS,
        "future_arrivals": TOTAL_JOBS - fill,
        "future_arrival_gap_steps": FUTURE_ARRIVAL_GAP,
        "capacity": CAPACITY,
        "mean_duration_parameter": MEAN_DURATION,
        "realized_duration_mean": float(fmean(instance.storage_steps_needed)),
        "last_release_step": int(instance.arrival_steps[-1]),
        "initial_clock_contract": (
            "storage requirements drawn by the E191 main-panel sampler; "
            "prefilled blocks begin at continuation clock zero"
        ),
        "initial_state_selection": (
            "random placement/stay pairing retained only after an exact SAFE "
            "current-state witness under B=20000"
        ),
    }


class Prefilled8x8Env(e191.LongStayCrowdingEnv):
    """E191 dynamics/objective initialized from one frozen crowded state."""

    def __init__(self, name, objective, record: dict):
        if name != SCENARIO:
            raise ValueError(f"unknown E270 scenario: {name}")
        super().__init__(e191.SCENARIO, objective)
        self.target_scenario_name = name
        self.e270_record = dict(record)

    def sample_episode_instance(self, seed=None):
        if seed is None or int(seed) != int(self.e270_record["seed"]):
            raise ValueError("E270 requires its frozen instance seed")
        if "instance_json" in self.e270_record:
            instance = EpisodeInstance.from_json(self.e270_record["instance_json"])
            instance.validate_for(self)
            return instance
        fill = int(self.e270_record["initial_fill"])
        self.arrival_rate = 1.0 / float(FUTURE_ARRIVAL_GAP)
        self.proc_mean = float(MEAN_DURATION)
        native = SmallRoomsEnv.sample_episode_instance(
            self, seed=int(self.e270_record["duration_seed"])
        )
        future = TOTAL_JOBS - fill
        arrivals = (0,) * fill + tuple(
            1 + FUTURE_ARRIVAL_GAP * index for index in range(future)
        )
        instance = replace(
            native,
            seed=int(seed),
            arrival_rate=float(self.arrival_rate),
            proc_mean=float(MEAN_DURATION),
            arrival_steps=arrivals,
        )
        instance.validate_for(self)
        return instance

    def reset(self, instance=None):
        if instance is None:
            raise ValueError("E270 requires an explicit frozen instance")
        state = super().reset(instance=instance)
        fill = int(self.e270_record["initial_fill"])
        cells = tuple(tuple(cell) for cell in self.e270_record["initial_layout"])
        if len(cells) != fill or len(set(cells)) != fill:
            raise RuntimeError("E270 initial layout is malformed")
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

        self.time_steps = 1
        self.current_state = tuple(self.e270_record["transporter_cell"])
        self.update_pickup_queue()
        self.e191_workload = workload_metadata(instance, self.e270_record)
        self.e191_stored_cell_steps = 0
        self.e191_queue_job_steps = 0
        self.e191_equipment_steps = 0
        self.e191_arrival_window_stored_steps = 0
        self.e191_arrival_window_steps = 0
        self.e191_steady_window_stored_steps = 0
        self.e191_steady_window_steps = 0
        self.e191_peak_queue = 0
        if sum(bool(block.stored) for block in self.blocks) != fill:
            raise RuntimeError("E270 reset did not materialize the declared fill")
        return self.get_current_state()


__all__ = [
    "CAPACITY", "CASES_PER_FILL", "FUTURE_ARRIVAL_GAP", "GATE_FILL",
    "GATE_SEED", "INITIAL_FILLS", "MAX_CANDIDATE_ATTEMPTS", "MAX_STEPS",
    "MEAN_DURATION", "Prefilled8x8Env", "SCENARIO", "SEEDS", "TOTAL_JOBS",
    "TRANSPORTER_CELL", "candidate_layout", "candidate_record",
    "duration_seed", "initial_fill", "replicate", "workload_metadata",
]
