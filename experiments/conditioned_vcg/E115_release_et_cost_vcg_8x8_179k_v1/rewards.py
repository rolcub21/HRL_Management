"""Release-clock earliness--tardiness cost components for E115.

The environment exposes the due slack that is known for released jobs and
charges the operator objective directly. There is deliberately no delivery
bonus. Completion remains a certified/guarded feasibility requirement.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from statistics import fmean

from experiments.conditioned_vcg.E78_original_vcg_base80_full_8x8_143k_v1 import (
    protocol as e78,
)
from vcg_objective_audit import (
    DENSE_PIECEWISE,
    LEGACY_CLIPPED,
    TIMING_OBJECTIVE_INFO_KEY,
)


COMPONENT_NAMES = (
    "release_tardiness_accrual",
    "release_earliness_at_delivery",
    "equipment_work",
    "physical_rehandles",
)
TARDINESS_WEIGHT = 2.0
EARLINESS_WEIGHT = 1.0
EQUIPMENT_WEIGHT = 0.1
REHANDLE_WEIGHT = 10.0
TOLERANCE = 20
ARRIVAL_GAP = 30
# The inherited trainer divides raw rewards by 100 before replay. Negative
# weights turn predicted costs into merits, so the selector still maximizes.
SCALAR_WEIGHTS = (
    -0.01 * TARDINESS_WEIGHT,
    -0.01 * EARLINESS_WEIGHT,
    -0.01 * EQUIPMENT_WEIGHT,
    -0.01 * REHANDLE_WEIGHT,
)


def release_error(block, delivery_time: int | None = None) -> float:
    """Delivery time minus the release-fixed due date a_i + p_i."""

    observed = (
        getattr(block, "delivered_time_step", None)
        if delivery_time is None else delivery_time
    )
    if observed is None:
        raise ValueError("release error requires a delivery time")
    return float(
        int(observed)
        - int(block.arrival_step)
        - int(block.storage_steps_needed)
    )


def earliness_tardiness(
    error: float, tolerance: int = TOLERANCE
) -> tuple[float, float]:
    if tolerance < 0 or not math.isfinite(float(error)):
        raise ValueError("invalid earliness--tardiness input")
    value = float(error)
    return (
        max(value - float(tolerance), 0.0),
        max(-value - float(tolerance), 0.0),
    )


def scalarized_components(values) -> float:
    values = tuple(float(value) for value in values)
    if len(values) != len(COMPONENT_NAMES):
        raise ValueError("E115 cost vector has the wrong width")
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("E115 cost components must be finite and nonnegative")
    return float(sum(weight * value for weight, value in zip(SCALAR_WEIGHTS, values)))


def objective_cost(values) -> float:
    """Return J for a raw four-component vector."""

    return -scalarized_components(values) / 0.01


@dataclass(frozen=True)
class PrimitiveComponents:
    release_tardiness_accrual: float
    release_earliness_at_delivery: float
    equipment_work: float
    physical_rehandles: float

    def vector(self) -> tuple[float, ...]:
        result = tuple(float(getattr(self, name)) for name in COMPONENT_NAMES)
        scalarized_components(result)
        return result


class ReleaseETCostEnv(e78.TargetDenseEnv):
    """Feasible 8x8 workload with a pure release-clock operator cost."""

    def sample_episode_instance(self, seed=None):
        if seed is None:
            raise ValueError("E115 requires an explicit instance seed")
        sampled = super().sample_episode_instance(seed=seed)
        instance = replace(
            sampled,
            arrival_steps=tuple(
                ARRIVAL_GAP * index for index in range(len(sampled.arrival_steps))
            ),
        )
        instance.validate_for(self)
        return instance

    def reset(self, instance=None):
        state = super().reset(instance=instance)
        self.factorized_component_records: list[tuple[float, ...]] = []
        self.equipment_work_total = 0.0
        self.physical_rehandles_total = 0.0
        return state

    def signed_remaining_storage_time(self, block):
        """Public release-relative due slack for an arrived, unfinished job."""

        if int(block.arrival_step) > int(self.time_steps) or bool(block.delivered):
            return None
        return float(
            int(block.arrival_step)
            + int(block.storage_steps_needed)
            - int(self.time_steps)
        )

    def compute_block_features(self, block, agent_pos, goal_state):
        values = list(super().compute_block_features(block, agent_pos, goal_state))
        remaining = self.signed_remaining_storage_time(block)
        if remaining is not None:
            values[5] = float(remaining)
        return tuple(values)

    def _overdue_undelivered_count(self, time_step: int) -> int:
        return sum(
            int(block.arrival_step) <= int(time_step)
            and not bool(block.delivered)
            and int(time_step) >= (
                int(block.arrival_step)
                + int(block.storage_steps_needed)
                + TOLERANCE
            )
            for block in self.blocks
        )

    def step(self, action):
        before_time = int(self.time_steps)
        tardiness = float(self._overdue_undelivered_count(before_time))
        equipment = float(int(action) != int(self.ACTION_IDS["WAIT"]))
        state, _base_reward, done, info = super().step(action)
        rehandle = float(bool(info.get("relocated_block")))
        early = 0.0
        if info.get("delivered_block"):
            block = next(
                item for item in self.blocks
                if str(item.label) == str(info["delivered_block"])
            )
            error = release_error(block)
            _late, early = earliness_tardiness(error)
            info["storage_delivery_error_time"] = info.get("delivery_error_time")
            info["delivery_error_time"] = error
            block.delivery_error_time = error
            self.delivery_error_times[-1] = (
                block.label, error, self.current_episode
            )

        components = PrimitiveComponents(
            tardiness, early, equipment, rehandle
        ).vector()
        selected = -objective_cost(components)
        record = deepcopy(info[TIMING_OBJECTIVE_INFO_KEY])
        streams = record["step_reward_by_objective"]
        for objective in (LEGACY_CLIPPED, DENSE_PIECEWISE):
            streams[objective] = float(selected)
        record["selected_step_reward"] = float(selected)
        record["e115_release_et_components"] = {
            name: value for name, value in zip(COMPONENT_NAMES, components)
        }
        record["e115_release_clock_due_rule"] = (
            "arrival_step + storage_steps_needed"
        )
        if not math.isclose(
            float(streams[record["selected_objective"]]),
            float(selected),
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise RuntimeError("E115 selected audit stream differs from reward")

        info = dict(info)
        info[TIMING_OBJECTIVE_INFO_KEY] = record
        self._objective_audit_records[-1] = record
        self.factorized_component_records.append(components)
        self.equipment_work_total += equipment
        self.physical_rehandles_total += rehandle
        return state, float(selected), done, info


def episode_service_metrics(env) -> dict:
    blocks = tuple(env.blocks)
    stored = tuple(
        block for block in blocks
        if getattr(block, "stored_time_step", None) is not None
    )
    delivered = tuple(
        block for block in blocks
        if getattr(block, "delivered_time_step", None) is not None
    )
    waits = tuple(
        int(block.stored_time_step) - int(block.arrival_step)
        for block in stored
    )
    release_errors = tuple(release_error(block) for block in delivered)
    storage_errors = tuple(
        int(block.delivered_time_step)
        - int(block.stored_time_step)
        - int(block.storage_steps_needed)
        for block in delivered
    )
    tardiness = tuple(earliness_tardiness(value)[0] for value in release_errors)
    earliness = tuple(earliness_tardiness(value)[1] for value in release_errors)

    def average(values):
        values = tuple(float(value) for value in values)
        return float(fmean(values)) if values else None

    component_totals = tuple(
        sum(record[index] for record in env.factorized_component_records)
        for index in range(len(COMPONENT_NAMES))
    )
    complete = len(delivered) == len(blocks)
    terminal_costs_match_dense = (
        complete
        and math.isclose(component_totals[0], sum(tardiness), abs_tol=1.0e-9)
        and math.isclose(component_totals[1], sum(earliness), abs_tol=1.0e-9)
    )
    return {
        "required_jobs": len(blocks),
        "stored_jobs": len(stored),
        "delivered_jobs": len(delivered),
        "strict_complete": complete,
        "mean_wait_before_storage": average(waits),
        "max_wait_before_storage": max(waits) if waits else None,
        "release_clock_mae": average(abs(value) for value in release_errors),
        "release_clock_mean_signed_error": average(release_errors),
        "mean_release_tardiness_beyond_tolerance": average(tardiness),
        "mean_release_earliness_beyond_tolerance": average(earliness),
        "within_tolerance_rate": (
            sum(abs(value) <= TOLERANCE for value in release_errors)
            / len(release_errors) if release_errors else None
        ),
        "storage_clock_mae": average(abs(value) for value in storage_errors),
        "equipment_work": float(env.equipment_work_total),
        "equipment_work_per_delivery": (
            float(env.equipment_work_total) / len(delivered) if delivered else None
        ),
        "physical_rehandles": float(env.physical_rehandles_total),
        "objective_cost": objective_cost(component_totals),
        "terminal_costs_match_dense_accrual": terminal_costs_match_dense,
        "component_record_count": len(env.factorized_component_records),
    }


__all__ = [
    "ARRIVAL_GAP",
    "COMPONENT_NAMES",
    "EARLINESS_WEIGHT",
    "EQUIPMENT_WEIGHT",
    "PrimitiveComponents",
    "REHANDLE_WEIGHT",
    "ReleaseETCostEnv",
    "SCALAR_WEIGHTS",
    "TARDINESS_WEIGHT",
    "TOLERANCE",
    "earliness_tardiness",
    "episode_service_metrics",
    "objective_cost",
    "release_error",
    "scalarized_components",
]
