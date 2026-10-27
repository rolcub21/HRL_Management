"""Exact storage-clock operator cost used by E187.

The primitive components telescope to

    J = 2 T + E + 0.5 W + 0.1 L + 10 R,

where timing uses a +/-20 storage-clock tolerance and admission waiting starts
when a job is released.  No grace period is applied to W.
"""

from __future__ import annotations

from copy import deepcopy
import math
from statistics import fmean

from vcg_objective_audit import TIMING_OBJECTIVE_INFO_KEY
from experiments.conditioned_vcg.E78_original_vcg_base80_full_8x8_143k_v1 import (
    protocol as e78,
)


COMPONENT_NAMES = (
    "storage_tardiness_accrual",
    "storage_earliness_at_delivery",
    "admission_wait_accrual",
    "equipment_work",
    "physical_rehandles",
)
OBJECTIVE_WEIGHTS = (2.0, 1.0, 0.5, 0.1, 10.0)
TOLERANCE = 20


def scalar_cost(values) -> float:
    values = tuple(float(value) for value in values)
    if len(values) != len(COMPONENT_NAMES):
        raise ValueError("E187 component vector has the wrong width")
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("E187 components must be finite and nonnegative")
    return float(sum(weight * value for weight, value in zip(OBJECTIVE_WEIGHTS, values)))


def storage_due(block) -> int:
    stored = getattr(block, "stored_time_step", None)
    if stored is None:
        raise ValueError("storage-clock due date is undefined before admission")
    return int(stored) + int(block.storage_steps_needed)


def storage_error(block) -> float:
    delivered = getattr(block, "delivered_time_step", None)
    if delivered is None:
        raise ValueError("storage error requires a completed delivery")
    return float(int(delivered) - storage_due(block))


def split_timing_error(error: float) -> tuple[float, float]:
    """Return tardiness and earliness beyond the fixed tolerance window."""

    value = float(error)
    if not math.isfinite(value):
        raise ValueError("timing error must be finite")
    return (
        max(value - float(TOLERANCE), 0.0),
        max(-value - float(TOLERANCE), 0.0),
    )


def waiting_jobs(env, at_time: int | None = None):
    """Released jobs whose first storage admission has not completed."""

    now = int(env.time_steps if at_time is None else at_time)
    return tuple(
        block
        for block in env.blocks
        if int(block.arrival_step) <= now
        and getattr(block, "stored_time_step", None) is None
        and not bool(block.delivered)
    )


def _mean(values) -> float | None:
    values = tuple(float(value) for value in values)
    return float(fmean(values)) if values else None


def _quantile(values, fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class WaitCorrectedE78Env(e78.TargetDenseEnv):
    """E78 workload with the exact five-component operational objective."""

    def reset(self, instance=None):
        state = super().reset(instance=instance)
        self.e187_component_records: list[tuple[float, ...]] = []
        self.e187_peak_stored = 0
        return state

    def _overdue_stored_count(self, at_time: int) -> int:
        return sum(
            getattr(block, "stored_time_step", None) is not None
            and not bool(block.delivered)
            and int(at_time) >= storage_due(block) + TOLERANCE
            for block in self.blocks
        )

    def step(self, action):
        before_time = int(self.time_steps)
        tardiness = float(self._overdue_stored_count(before_time))
        wait = float(len(waiting_jobs(self, before_time)))
        equipment = float(int(action) != int(self.ACTION_IDS["WAIT"]))

        state, _old_reward, done, info = super().step(action)
        rehandle = float(bool(info.get("relocated_block")))
        earliness = 0.0
        if info.get("delivered_block"):
            delivered = next(
                block
                for block in self.blocks
                if str(block.label) == str(info["delivered_block"])
            )
            _late, earliness = split_timing_error(storage_error(delivered))

        components = (
            tardiness,
            float(earliness),
            wait,
            equipment,
            rehandle,
        )
        reward = -scalar_cost(components)
        self.e187_component_records.append(components)

        record = deepcopy(info[TIMING_OBJECTIVE_INFO_KEY])
        record["step_reward_by_objective"] = {
            key: float(reward)
            for key in record["step_reward_by_objective"]
        }
        record["selected_step_reward"] = float(reward)
        record["e187_components"] = dict(zip(COMPONENT_NAMES, components))
        record["e187_objective"] = "2T_storage+E_storage+0.5W+0.1L+10R"
        record["e187_wait_definition"] = "release_to_first_completed_storage"
        record["e187_wait_grace_steps"] = 0
        info = dict(info)
        info[TIMING_OBJECTIVE_INFO_KEY] = record
        self._objective_audit_records[-1] = record

        stored = sum(
            getattr(block, "stored_time_step", None) is not None
            and not bool(block.delivered)
            for block in self.blocks
        )
        self.e187_peak_stored = max(self.e187_peak_stored, stored)
        return state, float(reward), done, info

    def e187_metrics(self) -> dict:
        blocks = tuple(self.blocks)
        stored = tuple(
            block
            for block in blocks
            if getattr(block, "stored_time_step", None) is not None
        )
        delivered = tuple(
            block
            for block in blocks
            if getattr(block, "delivered_time_step", None) is not None
        )
        waits = tuple(
            int(block.stored_time_step) - int(block.arrival_step)
            for block in stored
        )
        storage_errors = tuple(storage_error(block) for block in delivered)
        release_errors = tuple(
            int(block.delivered_time_step)
            - int(block.arrival_step)
            - int(block.storage_steps_needed)
            for block in delivered
        )
        totals = tuple(
            float(sum(row[index] for row in self.e187_component_records))
            for index in range(len(COMPONENT_NAMES))
        )
        complete = len(delivered) == len(blocks)
        all_admitted = len(stored) == len(blocks)
        terminal_tardiness = sum(
            split_timing_error(error)[0] for error in storage_errors
        )
        terminal_earliness = sum(
            split_timing_error(error)[1] for error in storage_errors
        )
        terminal_wait = float(sum(waits))
        tardiness_exact = complete and math.isclose(
            totals[0], terminal_tardiness, abs_tol=1.0e-9
        )
        earliness_exact = complete and math.isclose(
            totals[1], terminal_earliness, abs_tol=1.0e-9
        )
        wait_exact = all_admitted and math.isclose(
            totals[2], terminal_wait, abs_tol=1.0e-9
        )
        dense_j = scalar_cost(totals)
        return {
            "primary_objective": "J_storage_clock_plus_full_admission_wait",
            "strict_complete": complete,
            "required_jobs": len(blocks),
            "stored_jobs": len(stored),
            "delivered_jobs": len(delivered),
            "J": dense_j,
            "components": dict(zip(COMPONENT_NAMES, totals)),
            "objective_weights": dict(zip(COMPONENT_NAMES, OBJECTIVE_WEIGHTS)),
            "mean_admission_wait": _mean(waits),
            "median_admission_wait": _quantile(waits, 0.5),
            "p90_admission_wait": _quantile(waits, 0.9),
            "max_admission_wait": max(waits) if waits else None,
            "total_admission_wait": terminal_wait,
            "storage_clock_mae": _mean(abs(value) for value in storage_errors),
            "storage_clock_signed_error": _mean(storage_errors),
            "storage_within_20_rate": (
                sum(abs(value) <= TOLERANCE for value in storage_errors)
                / len(storage_errors)
                if storage_errors else None
            ),
            "release_clock_mae": _mean(abs(value) for value in release_errors),
            "release_clock_signed_error": _mean(release_errors),
            "release_within_20_rate": (
                sum(abs(value) <= TOLERANCE for value in release_errors)
                / len(release_errors)
                if release_errors else None
            ),
            "peak_stored": int(self.e187_peak_stored),
            "storage_capacity": len(self.storage_positions),
            "peak_stored_fraction": (
                self.e187_peak_stored / len(self.storage_positions)
                if self.storage_positions else None
            ),
            "tardiness_accounting_exact": tardiness_exact,
            "earliness_accounting_exact": earliness_exact,
            "wait_accounting_exact": wait_exact,
            "objective_accounting_exact": bool(
                tardiness_exact and earliness_exact and wait_exact
            ),
            "reward_sum_equals_negative_J": math.isclose(
                sum(
                    float(record["selected_step_reward"])
                    for record in self.objective_audit_records
                ),
                -dense_j,
                abs_tol=1.0e-8,
            ),
        }


__all__ = [
    "COMPONENT_NAMES",
    "OBJECTIVE_WEIGHTS",
    "TOLERANCE",
    "WaitCorrectedE78Env",
    "scalar_cost",
    "split_timing_error",
    "storage_due",
    "storage_error",
    "waiting_jobs",
]
