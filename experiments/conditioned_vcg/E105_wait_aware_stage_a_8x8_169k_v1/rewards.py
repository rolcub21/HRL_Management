"""Wait-aware reward and two-clock episode metrics for E105."""

from __future__ import annotations

from copy import deepcopy
from statistics import fmean

from experiments.conditioned_vcg.E78_original_vcg_base80_full_8x8_143k_v1 import (
    protocol as e78,
)
from vcg_objective_audit import (
    DENSE_PIECEWISE,
    LEGACY_CLIPPED,
    TIMING_OBJECTIVE_INFO_KEY,
)


WAIT_CAP = 80
WAIT_PENALTY_PER_JOB_STEP = 0.5


def released_unstored_over_cap(blocks, time_step: int, cap: int = WAIT_CAP):
    """Return jobs charged on the transition beginning at ``time_step``.

    Charging a job when its age is at least ``cap`` makes the accumulated
    count before storage equal ``max(0, stored_time-arrival-cap)``.  A job
    stored at age exactly ``cap`` therefore receives no charge.
    """

    return tuple(
        block
        for block in blocks
        if getattr(block, "stored_time_step", None) is None
        and not bool(getattr(block, "delivered", False))
        and hasattr(block, "arrival_step")
        and int(block.arrival_step) <= int(time_step)
        and int(time_step) - int(block.arrival_step) >= int(cap)
    )


def wait_penalty(blocks, time_step: int, *, cap: int = WAIT_CAP,
                 coefficient: float = WAIT_PENALTY_PER_JOB_STEP) -> tuple[float, int]:
    count = len(released_unstored_over_cap(blocks, time_step, cap))
    return -float(coefficient) * count, count


def completed_wait_excess(blocks, cap: int = WAIT_CAP) -> int:
    return sum(
        max(0, int(block.stored_time_step) - int(block.arrival_step) - int(cap))
        for block in blocks
        if getattr(block, "stored_time_step", None) is not None
    )


class WaitAwareDenseEnv(e78.TargetDenseEnv):
    """E78 environment plus an accrued outside-wait-cap penalty.

    The original base-80 storage-clock delivery reward is unchanged.  Each
    primitive step charges 0.5 for each released job that is still unstored
    after its 80-step grace period.  The charge is common to the legacy and
    dense audit streams; only their delivery terms differ.
    """

    def reset(self, instance=None):
        state = super().reset(instance=instance)
        self.outside_wait_overcap_job_steps = 0
        self.outside_wait_penalty_total = 0.0
        return state

    def step(self, action):
        before_time = int(self.time_steps)
        adjustment, charged_jobs = wait_penalty(self.blocks, before_time)
        state, reward, done, info = super().step(action)
        selected = float(reward) + adjustment

        record = deepcopy(info[TIMING_OBJECTIVE_INFO_KEY])
        streams = record["step_reward_by_objective"]
        for objective in (LEGACY_CLIPPED, DENSE_PIECEWISE):
            streams[objective] = float(streams[objective]) + adjustment
        record["selected_step_reward"] = selected
        record["e105_outside_wait"] = {
            "contract": "released_unstored_over_cap_job_steps_v1",
            "before_time": before_time,
            "wait_cap": WAIT_CAP,
            "penalty_per_overcap_job_step": WAIT_PENALTY_PER_JOB_STEP,
            "charged_jobs": charged_jobs,
            "reward_adjustment": adjustment,
        }

        selected_stream = float(streams[record["selected_objective"]])
        if abs(selected_stream - selected) > 1.0e-10:
            raise RuntimeError("E105 selected reward and audit stream differ")

        info = dict(info)
        info[TIMING_OBJECTIVE_INFO_KEY] = record
        self._objective_audit_records[-1] = record
        self.outside_wait_overcap_job_steps += charged_jobs
        self.outside_wait_penalty_total += adjustment
        return state, selected, done, info


def _mean(values) -> float | None:
    values = tuple(float(value) for value in values)
    return float(fmean(values)) if values else None


def episode_service_metrics(env) -> dict:
    """Measure admission and delivery clocks from a completed or partial run."""

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
    release_errors = tuple(
        int(block.delivered_time_step) - int(block.arrival_step)
        - int(block.storage_steps_needed)
        for block in delivered
    )
    storage_errors = tuple(
        int(block.delivered_time_step) - int(block.stored_time_step)
        - int(block.storage_steps_needed)
        for block in delivered
    )
    excesses = tuple(max(0, wait - WAIT_CAP) for wait in waits)
    accrued_job_steps = int(getattr(env, "outside_wait_overcap_job_steps", 0))
    terminal_completed_excess = completed_wait_excess(blocks)
    return {
        "required_jobs": len(blocks),
        "stored_jobs": len(stored),
        "delivered_jobs": len(delivered),
        "mean_wait_before_storage": _mean(waits),
        "max_wait_before_storage": max(waits) if waits else None,
        "jobs_exceeding_wait_cap": sum(wait > WAIT_CAP for wait in waits),
        "wait_cap_compliance_rate": (
            sum(wait <= WAIT_CAP for wait in waits) / len(waits) if waits else None
        ),
        "wait_cap_excess_job_steps": sum(excesses),
        "mean_wait_cap_excess_per_job": (
            sum(excesses) / len(waits) if waits else None
        ),
        "release_clock_mae": _mean(abs(value) for value in release_errors),
        "release_clock_mean_signed_error": _mean(release_errors),
        "storage_clock_mae": _mean(abs(value) for value in storage_errors),
        "storage_clock_mean_signed_error": _mean(storage_errors),
        "storage_error_early_count": sum(value < 0 for value in storage_errors),
        "storage_error_late_count": sum(value > 0 for value in storage_errors),
        "outside_wait_overcap_job_steps_accrued": accrued_job_steps,
        "outside_wait_penalty_total": float(
            getattr(env, "outside_wait_penalty_total", 0.0)
        ),
        "completed_wait_excess_matches_accrual": bool(
            len(stored) == len(blocks)
            and terminal_completed_excess == accrued_job_steps
        ),
    }
