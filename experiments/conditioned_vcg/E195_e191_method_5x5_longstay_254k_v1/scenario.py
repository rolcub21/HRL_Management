"""Scale-matched 5x5 long-stay workload for E195.

E191 has 35 storage cells and a mean stay of 700 primitive steps, or 20
expected stay steps per storage cell.  E195 has eight storage cells and uses
a mean stay of 160, preserving that normalized duration.  The common release
gaps 100, 40, and 25 therefore induce the same nominal 20%, 50%, and 80%
steady-state occupancy targets at both scales.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import fmean

from example.small_rooms_env import SmallRoomsEnv
from vcg_objective_audit import ObjectiveAuditSmallRoomsEnv, TimingObjectiveSpec
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    objective as e187_objective,
)


SCENARIO = "empty_5x5_n8_longstay_mixedrho"
TOTAL_JOBS = 8
CAPACITY = 8
MEAN_DURATION = 160
LEVELS = (0.20, 0.50, 0.80)
GAPS = {0.20: 100, 0.50: 40, 0.80: 25}
MAX_STEPS = 2_000
CONTEXT_NORMALIZATION_STEPS = MAX_STEPS

TRAIN_BASE = 195_500_000
VALIDATION_BASE = 250_195_000
VALIDATION_CASES_PER_LEVEL = 3
PILOT_BASE = 250_195_900


def _level_key(value: float) -> str:
    return f"{100.0 * float(value):.0f}%"


VALIDATION_LEVEL_BY_SEED = {
    VALIDATION_BASE + level_index * 100 + replicate: level
    for level_index, level in enumerate(LEVELS)
    for replicate in range(VALIDATION_CASES_PER_LEVEL)
}
VALIDATION_SEEDS = tuple(VALIDATION_LEVEL_BY_SEED)
PILOT_LEVEL_BY_SEED = {
    PILOT_BASE + index: level for index, level in enumerate(LEVELS)
}
PILOT_SEEDS = tuple(PILOT_LEVEL_BY_SEED)


def target_level(seed: int) -> float:
    """Return the balanced workload level assigned to an instance seed."""

    value = int(seed)
    if value in VALIDATION_LEVEL_BY_SEED:
        return float(VALIDATION_LEVEL_BY_SEED[value])
    if value in PILOT_LEVEL_BY_SEED:
        return float(PILOT_LEVEL_BY_SEED[value])
    return float(LEVELS[value % len(LEVELS)])


def duration_seed(seed: int) -> int:
    """Pair duration streams across validation workload levels."""

    value = int(seed)
    if value in VALIDATION_LEVEL_BY_SEED:
        replicate = value % 100
        return 250_195_500 + replicate
    if value in PILOT_LEVEL_BY_SEED:
        return 250_195_950
    return value + 1_950_007


def gap_for_level(level: float) -> int:
    try:
        return int(GAPS[float(level)])
    except KeyError as error:
        raise ValueError(f"unsupported E195 workload level: {level}") from error


def nominal_occupancy(gap: float) -> float:
    return float(MEAN_DURATION / (float(gap) * CAPACITY))


@dataclass(frozen=True)
class LongStay5x5Scenario:
    scenario_id: str = SCENARIO
    rows: int = 5
    cols: int = 5
    initial_occupied_slots: int = 0
    total_jobs: int = TOTAL_JOBS
    max_steps: int = MAX_STEPS

    @property
    def door(self) -> tuple[int, int]:
        return (0, self.cols - 2)

    @property
    def exits(self) -> tuple[tuple[int, int], ...]:
        right = self.cols - 2
        return tuple((self.rows - 1, col) for col in range(right - 2, right + 1))

    def make_base_env(self) -> SmallRoomsEnv:
        return SmallRoomsEnv(
            grid_rows=self.rows,
            grid_cols=self.cols,
            start_state=(1, 1),
            door_cell=self.door,
            pickup_cells=(1, self.cols - 2),
            exit_cells=list(self.exits),
            number_blocks=self.total_jobs,
            choose_storage=False,
            arrival_rate=1.0 / GAPS[0.50],
            proc_mean=MEAN_DURATION,
        )

    @property
    def storage_capacity(self) -> int:
        return len(self.make_base_env().storage_positions)

    def public_dict(self) -> dict:
        env = self.make_base_env()
        return {
            "scenario_id": self.scenario_id,
            "rows": self.rows,
            "cols": self.cols,
            "initial_occupied_slots": self.initial_occupied_slots,
            "total_jobs": self.total_jobs,
            "storage_capacity": len(env.storage_positions),
            "max_steps": self.max_steps,
            "door_cell": list(env.door_cell),
            "pickup_cell": list(env.pickup_cell),
            "exit_cells": [list(cell) for cell in env.exit_cells],
            "mean_storage_duration": MEAN_DURATION,
            "nominal_expected_occupancy_levels": list(LEVELS),
            "arrival_gaps": {
                _level_key(level): gap_for_level(level) for level in LEVELS
            },
            "occupancy_definition": (
                "arrival_rate * mean_storage_duration / storage_capacity"
            ),
            "occupancy_is_initial_yard_fill": False,
        }


SCENARIO_SPEC = LongStay5x5Scenario()
if SCENARIO_SPEC.storage_capacity != CAPACITY:
    raise RuntimeError("E195 5x5 geometry does not have eight storage cells")


def workload_metadata(instance) -> dict:
    arrivals = tuple(int(value) for value in instance.arrival_steps)
    gaps = tuple(
        arrivals[index + 1] - arrivals[index]
        for index in range(len(arrivals) - 1)
    )
    mean_gap = float(fmean(gaps)) if gaps else float("inf")
    target = target_level(int(instance.seed))
    return {
        "target_nominal_occupancy": target,
        "target_label": _level_key(target),
        "arrival_gap_steps": int(round(mean_gap)),
        "arrival_rate": float(instance.arrival_rate),
        "mean_duration_parameter": MEAN_DURATION,
        "realized_duration_mean": float(fmean(instance.storage_steps_needed)),
        "nominal_occupancy_from_declared_mean": nominal_occupancy(mean_gap),
        "last_release_step": int(arrivals[-1]),
        "jobs": len(arrivals),
        "capacity": CAPACITY,
        "finite_episode_caveat": (
            "nominal target is steady-state; achieved whole-episode mean includes "
            "empty-start ramp and terminal drain"
        ),
    }


class LongStay5x5Env(e187_objective.WaitCorrectedE78Env):
    """E191 objective semantics on the scale-matched 5x5 scenario."""

    def __init__(self, name: str, objective: TimingObjectiveSpec):
        if name != SCENARIO:
            raise ValueError(f"unknown E195 scenario: {name}")
        scenario = SCENARIO_SPEC
        ObjectiveAuditSmallRoomsEnv.__init__(
            self,
            timing_objective=objective,
            grid_rows=scenario.rows,
            grid_cols=scenario.cols,
            start_state=(1, 1),
            door_cell=scenario.door,
            pickup_cells=(1, scenario.cols - 2),
            exit_cells=list(scenario.exits),
            number_blocks=scenario.total_jobs,
            choose_storage=False,
            arrival_rate=1.0 / GAPS[0.50],
            proc_mean=MEAN_DURATION,
        )
        self.target_scenario_name = name

    def sample_episode_instance(self, seed=None):
        if seed is None:
            raise ValueError("E195 requires an explicit instance seed")
        seed = int(seed)
        level = target_level(seed)
        gap = gap_for_level(level)
        self.arrival_rate = 1.0 / float(gap)
        self.proc_mean = float(MEAN_DURATION)
        native = SmallRoomsEnv.sample_episode_instance(
            self, seed=duration_seed(seed)
        )
        arrivals = tuple(gap * index for index in range(TOTAL_JOBS))
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
        state = super().reset(instance=instance)
        self.e195_workload = workload_metadata(self.current_episode_instance)
        self.e195_stored_cell_steps = 0
        self.e195_queue_job_steps = 0
        self.e195_equipment_steps = 0
        self.e195_arrival_window_stored_steps = 0
        self.e195_arrival_window_steps = 0
        self.e195_steady_window_stored_steps = 0
        self.e195_steady_window_steps = 0
        self.e195_peak_queue = 0
        return state

    def step(self, action):
        before = int(self.time_steps)
        stored = sum(
            getattr(block, "stored_time_step", None) is not None
            and not bool(block.delivered)
            for block in self.blocks
        )
        queued = len(e187_objective.waiting_jobs(self, before))
        last_release = int(self.e195_workload["last_release_step"])
        self.e195_stored_cell_steps += int(stored)
        self.e195_queue_job_steps += int(queued)
        self.e195_equipment_steps += int(
            int(action) != int(self.ACTION_IDS["WAIT"])
        )
        self.e195_peak_queue = max(self.e195_peak_queue, int(queued))
        if before <= last_release:
            self.e195_arrival_window_stored_steps += int(stored)
            self.e195_arrival_window_steps += 1
        if MEAN_DURATION <= before <= last_release:
            self.e195_steady_window_stored_steps += int(stored)
            self.e195_steady_window_steps += 1
        return super().step(action)

    def e187_metrics(self) -> dict:
        result = super().e187_metrics()
        steps = max(int(self.time_steps), 1)
        arrival_steps = max(int(self.e195_arrival_window_steps), 1)
        steady_steps = int(self.e195_steady_window_steps)
        result.update({
            "mean_stored": self.e195_stored_cell_steps / steps,
            "mean_stored_fraction": (
                self.e195_stored_cell_steps / (steps * CAPACITY)
            ),
            "arrival_window_mean_stored": (
                self.e195_arrival_window_stored_steps / arrival_steps
            ),
            "nominal_steady_window_mean_stored": (
                None
                if steady_steps == 0
                else self.e195_steady_window_stored_steps / steady_steps
            ),
            "mean_external_queue": self.e195_queue_job_steps / steps,
            "peak_external_queue": int(self.e195_peak_queue),
            "transporter_utilization": self.e195_equipment_steps / steps,
            "e195_workload": dict(self.e195_workload),
        })
        return result


__all__ = [
    "CAPACITY", "CONTEXT_NORMALIZATION_STEPS", "GAPS", "LEVELS",
    "LongStay5x5Env", "LongStay5x5Scenario", "MAX_STEPS",
    "MEAN_DURATION", "PILOT_LEVEL_BY_SEED", "PILOT_SEEDS", "SCENARIO",
    "SCENARIO_SPEC", "TOTAL_JOBS", "TRAIN_BASE", "VALIDATION_CASES_PER_LEVEL",
    "VALIDATION_LEVEL_BY_SEED", "VALIDATION_SEEDS", "duration_seed",
    "gap_for_level", "nominal_occupancy", "target_level",
    "workload_metadata",
]
