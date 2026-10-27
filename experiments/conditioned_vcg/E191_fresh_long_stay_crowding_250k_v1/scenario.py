"""Deterministic long-stay workload family for E191.

The workload target is nominal steady-state expected occupancy, not initial
yard occupancy.  Every episode starts empty and contains the same 36 jobs.
Only the release gap changes across the three workload levels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import fmean

from example.small_rooms_env import SmallRoomsEnv
from vcg_objective_audit import ObjectiveAuditSmallRoomsEnv, TimingObjectiveSpec
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    objective as e187_objective,
    protocol as e187_protocol,
)


SCENARIO = "empty_8x8_n36_longstay_mixedrho"
TOTAL_JOBS = 36
CAPACITY = 35
MEAN_DURATION = 700
LEVELS = (0.20, 0.50, 0.80)
GAPS = {0.20: 100, 0.50: 40, 0.80: 25}
MAX_STEPS = 9_000
CONTEXT_NORMALIZATION_STEPS = MAX_STEPS

TRAIN_BASE = 191_800_000
VALIDATION_BASE = 250_191_000
VALIDATION_CASES_PER_LEVEL = 3
PILOT_BASE = 250_191_900


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
    """Return a deterministic, balanced workload level for an instance seed."""

    value = int(seed)
    if value in VALIDATION_LEVEL_BY_SEED:
        return float(VALIDATION_LEVEL_BY_SEED[value])
    if value in PILOT_LEVEL_BY_SEED:
        return float(PILOT_LEVEL_BY_SEED[value])
    return float(LEVELS[value % len(LEVELS)])


def duration_seed(seed: int) -> int:
    """Pair validation duration streams across workload levels."""

    value = int(seed)
    if value in VALIDATION_LEVEL_BY_SEED:
        replicate = value % 100
        return 250_191_500 + replicate
    if value in PILOT_LEVEL_BY_SEED:
        return 250_191_950
    return value + 1_910_007


def gap_for_level(level: float) -> int:
    try:
        return int(GAPS[float(level)])
    except KeyError as error:
        raise ValueError(f"unsupported E191 workload level: {level}") from error


def nominal_occupancy(gap: float) -> float:
    return float(MEAN_DURATION / (float(gap) * CAPACITY))


@dataclass(frozen=True)
class LongStayScenario:
    scenario_id: str = SCENARIO
    rows: int = 8
    cols: int = 8
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


SCENARIO_SPEC = LongStayScenario()


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


class LongStayCrowdingEnv(e187_objective.WaitCorrectedE78Env):
    """E189 objective/environment semantics with a new instance generator."""

    def __init__(self, name: str, objective: TimingObjectiveSpec):
        if name != SCENARIO:
            raise ValueError(f"unknown E191 scenario: {name}")
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
            raise ValueError("E191 requires an explicit instance seed")
        seed = int(seed)
        level = target_level(seed)
        gap = gap_for_level(level)
        self.arrival_rate = 1.0 / float(gap)
        self.proc_mean = float(MEAN_DURATION)

        # Call the physical environment sampler directly, bypassing E78's
        # fixed Poisson-80/E28 workload.  Geometry and block count are still
        # exactly those of the current E189 architecture.
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
        self.e191_workload = workload_metadata(self.current_episode_instance)
        self.e191_stored_cell_steps = 0
        self.e191_queue_job_steps = 0
        self.e191_equipment_steps = 0
        self.e191_arrival_window_stored_steps = 0
        self.e191_arrival_window_steps = 0
        self.e191_steady_window_stored_steps = 0
        self.e191_steady_window_steps = 0
        self.e191_peak_queue = 0
        return state

    def step(self, action):
        before = int(self.time_steps)
        stored = sum(
            getattr(block, "stored_time_step", None) is not None
            and not bool(block.delivered)
            for block in self.blocks
        )
        queued = len(e187_objective.waiting_jobs(self, before))
        last_release = int(self.e191_workload["last_release_step"])
        self.e191_stored_cell_steps += int(stored)
        self.e191_queue_job_steps += int(queued)
        self.e191_equipment_steps += int(
            int(action) != int(self.ACTION_IDS["WAIT"])
        )
        self.e191_peak_queue = max(self.e191_peak_queue, int(queued))
        if before <= last_release:
            self.e191_arrival_window_stored_steps += int(stored)
            self.e191_arrival_window_steps += 1
        if MEAN_DURATION <= before <= last_release:
            self.e191_steady_window_stored_steps += int(stored)
            self.e191_steady_window_steps += 1
        return super().step(action)

    def e187_metrics(self) -> dict:
        result = super().e187_metrics()
        steps = max(int(self.time_steps), 1)
        arrival_steps = max(int(self.e191_arrival_window_steps), 1)
        steady_steps = int(self.e191_steady_window_steps)
        result.update({
            "mean_stored": self.e191_stored_cell_steps / steps,
            "mean_stored_fraction": (
                self.e191_stored_cell_steps / (steps * CAPACITY)
            ),
            "arrival_window_mean_stored": (
                self.e191_arrival_window_stored_steps / arrival_steps
            ),
            "nominal_steady_window_mean_stored": (
                None
                if steady_steps == 0
                else self.e191_steady_window_stored_steps / steady_steps
            ),
            "mean_external_queue": self.e191_queue_job_steps / steps,
            "peak_external_queue": int(self.e191_peak_queue),
            "transporter_utilization": self.e191_equipment_steps / steps,
            "e191_workload": dict(self.e191_workload),
        })
        return result


# The E189 environment has the same physical geometry.  This assertion catches
# accidental layout drift without importing any learned state.
_BASE = e187_protocol.SCENARIOS[e187_protocol.SCENARIO]
if (
    (_BASE.rows, _BASE.cols, _BASE.total_jobs)
    != (SCENARIO_SPEC.rows, SCENARIO_SPEC.cols, SCENARIO_SPEC.total_jobs)
    or _BASE.storage_capacity != SCENARIO_SPEC.storage_capacity
):
    raise RuntimeError("E191 physical scenario no longer matches E189")


__all__ = [
    "CAPACITY", "CONTEXT_NORMALIZATION_STEPS", "GAPS", "LEVELS",
    "LongStayCrowdingEnv", "LongStayScenario", "MAX_STEPS",
    "MEAN_DURATION", "PILOT_LEVEL_BY_SEED", "PILOT_SEEDS", "SCENARIO",
    "SCENARIO_SPEC", "TOTAL_JOBS", "TRAIN_BASE", "VALIDATION_CASES_PER_LEVEL",
    "VALIDATION_LEVEL_BY_SEED", "VALIDATION_SEEDS", "duration_seed",
    "gap_for_level", "nominal_occupancy", "target_level",
    "workload_metadata",
]
