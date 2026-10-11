"""Explicit common scheduler completion for empty-start placement baselines.

This is an adaptation of the cited placement methods, not an implementation
claimed to be specified in their original papers.
"""

from __future__ import annotations

from options_agent import option_identifier
from example.urgency_scheduler import SchedulerInfeasibleError
from experiments.conditioned_vcg.E25_v4_reconfiguration_baselines_95k.controller import (
    ReconfigurationScheduler,
)
from experiments.conditioned_vcg.development.D10_scalability_support_screen import (
    occupancy_extension as occupancy,
)


class AuditedEmptyStartEnv(occupancy.OccupancyTrackingEnv):
    """Require all jobs to arrive after reset, with no pre-positioned inventory."""

    def __init__(self, scenario):
        if scenario.initial_occupied_slots != 0:
            raise ValueError("empty-start scenario must have zero initial blocks")
        super().__init__(scenario)
        self.warm_start_audit = None

    def reset(self, instance=None):
        if instance is None:
            raise ValueError("empty-start comparison requires a frozen instance")
        state = super().reset(instance=instance)
        if (self.time_steps != 0
            or self.occupancy_trace[0]["occupied_storage_slots"] != 0
            or any(block.stored or block.delivered or block.carrying
                   or block.stored_time_step is not None
                   for block in self.blocks)
            or self.blocks[0].arrival_step != 0
            or self.blocks[0].position != self.pickup_cell
            or any(block.position is not None or block.arrival_step <= 0
                   for block in self.blocks[1:])):
            raise RuntimeError("empty-start reset contained physical inventory")
        self.warm_start_audit = {
            "passed": True,
            "episode_instance_id": instance.instance_id,
            "time_step": 0,
            "initially_stored_jobs": 0,
            "occupied_storage_slots": 0,
            "future_arrivals": len(self.blocks),
        }
        return state


class WaitEnabledReconfigurationScheduler(ReconfigurationScheduler):
    """Use the existing retrieval/reconfiguration adapter, then bounded defer.

    The parent first tries admission, strict retrieval (which clears blockers),
    and a legal standalone reconfiguration. Only when those offer no selected
    action does the scheduler wait. Waiting is never mistaken for completion:
    the outer episode step limit remains authoritative.
    """

    VERSION = "empty_start_common_scheduler_with_bounded_wait_v1"
    POLICY = "strict_retrieval_then_reconfigure_then_defer_v1"

    def select_action(self, state):
        try:
            return super().select_action(state)
        except SchedulerInfeasibleError as error:
            if not self.defer_option.initiation(state):
                raise
            identifier = option_identifier(self.defer_option)
            self.decision_count += 1
            self.reason_counts["blocked_inbound_defer"] += 1
            self.control_decisions[identifier] += 1
            self.decisions.append({
                "decision_index": self.decision_count - 1,
                "time_step": int(self.env.time_steps),
                "reason": "blocked_inbound_defer",
                "option_id": identifier,
                "inbound_label": (
                    None if self._inbound_block() is None
                    else self._inbound_block().label
                ),
                "parent_scheduler_failure": str(error),
            })
            return self.defer_option


def empty_start_factory(scenario, *, arrival_rate, proc_mean, grid_rows,
                        grid_cols, exit_width, number_blocks):
    if (grid_rows != scenario.rows or grid_cols != scenario.cols
        or number_blocks != scenario.total_jobs
        or arrival_rate != 1.0 / occupancy.EXTRA_ARRIVAL_GAP
        or proc_mean != occupancy.PROC_MEAN or exit_width != 3):
        raise ValueError("baseline environment request differs from frozen scenario")
    return AuditedEmptyStartEnv(scenario)
