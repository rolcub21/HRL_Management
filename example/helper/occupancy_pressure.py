"""Environment-pressure diagnostics shared by Track-B controllers.

The tracker is deliberately policy-neutral.  It reads only public yard/block
state after an environment transition; controller candidate-set sizes are an
optional, separately recorded decision-epoch diagnostic.
"""

from __future__ import annotations

from math import fsum


OCCUPANCY_PRESSURE_METRIC_CONTRACT = (
    "post_transition_physical_storage_and_decision_epoch_candidates_v1"
)
HIGH_OCCUPANCY_THRESHOLD = 0.80


class OccupancyPressureTracker:
    """Accumulate storage, queue, and optional candidate pressure for one run.

    Peak/minimum state diagnostics include the reset state.  The time-fraction
    denominator, however, contains exactly one observation per completed
    environment transition.  Physical occupancy excludes a stored block while
    it is being carried; active stored inventory retains that block until
    delivery.
    """

    def __init__(self, env):
        self.env = env
        self.storage_cells = frozenset(
            tuple(cell) for cell in getattr(env, "storage_positions", ())
        )
        self.storage_capacity_cells = len(self.storage_cells)
        self.peak_active_stored_count = 0
        self.peak_physical_storage_occupancy_count = 0
        self.minimum_free_storage_cells = self.storage_capacity_cells
        self.storage_pressure_observed_steps = 0
        self.storage_steps_at_or_above_80pct_occupied = 0
        self.maximum_inbound_waiting_count = 0
        self.maximum_inbound_queue_count = 0
        self.minimum_live_candidate_count = None
        self.minimum_live_accept_candidate_count = None
        self._observe_state(count_transition=False)

    @staticmethod
    def _is_open_inbound(block) -> bool:
        return bool(
            not getattr(block, "carrying", False)
            and not getattr(block, "stored", False)
            and not getattr(block, "delivered", False)
        )

    def _observe_state(self, *, count_transition: bool) -> None:
        blocks = tuple(getattr(self.env, "blocks", ()))
        active_stored = sum(
            bool(
                getattr(block, "stored", False)
                and not getattr(block, "delivered", False)
            )
            for block in blocks
        )
        physically_occupied = {
            tuple(block.position)
            for block in blocks
            if (
                getattr(block, "stored", False)
                and not getattr(block, "delivered", False)
                and not getattr(block, "carrying", False)
                and getattr(block, "position", None) is not None
                and tuple(block.position) in self.storage_cells
            )
        }
        physical_count = len(physically_occupied)
        free_count = max(0, self.storage_capacity_cells - physical_count)
        waiting_cell = tuple(getattr(self.env, "waiting_cell", ()))
        pickup_cell = tuple(getattr(self.env, "pickup_cell", ()))
        waiting_count = sum(
            bool(
                self._is_open_inbound(block)
                and getattr(block, "position", None) is not None
                and tuple(block.position) == waiting_cell
            )
            for block in blocks
        )
        queue_cells = {waiting_cell, pickup_cell}
        queue_count = sum(
            bool(
                self._is_open_inbound(block)
                and getattr(block, "position", None) is not None
                and tuple(block.position) in queue_cells
            )
            for block in blocks
        )

        self.peak_active_stored_count = max(
            self.peak_active_stored_count, active_stored
        )
        self.peak_physical_storage_occupancy_count = max(
            self.peak_physical_storage_occupancy_count, physical_count
        )
        self.minimum_free_storage_cells = min(
            self.minimum_free_storage_cells, free_count
        )
        self.maximum_inbound_waiting_count = max(
            self.maximum_inbound_waiting_count, waiting_count
        )
        self.maximum_inbound_queue_count = max(
            self.maximum_inbound_queue_count, queue_count
        )
        if count_transition:
            self.storage_pressure_observed_steps += 1
            fraction = (
                physical_count / self.storage_capacity_cells
                if self.storage_capacity_cells
                else 0.0
            )
            self.storage_steps_at_or_above_80pct_occupied += int(
                fraction >= HIGH_OCCUPANCY_THRESHOLD
            )

    def observe_transition(self) -> None:
        """Record the state after one completed environment transition."""

        self._observe_state(count_transition=True)

    def observe_candidate_set(
        self, total_count: int | None, *, accept_count: int | None = None
    ) -> None:
        """Record one controller decision epoch's live candidate counts."""

        if total_count is not None:
            total_count = int(total_count)
            if total_count < 0:
                raise ValueError("total_count must be nonnegative")
            self.minimum_live_candidate_count = (
                total_count
                if self.minimum_live_candidate_count is None
                else min(self.minimum_live_candidate_count, total_count)
            )
        if accept_count is None:
            return
        accept_count = int(accept_count)
        if accept_count < 0 or (
            total_count is not None and accept_count > total_count
        ):
            raise ValueError(
                "accept_count must be nonnegative and no greater than "
                "total_count when total_count is observed"
            )
        # Zero means that accept mode was not live at this epoch, rather than
        # that its feasible candidate set was exhausted.  The latter is a
        # strict infeasibility and is reported elsewhere by the controller.
        if accept_count:
            self.minimum_live_accept_candidate_count = (
                accept_count
                if self.minimum_live_accept_candidate_count is None
                else min(
                    self.minimum_live_accept_candidate_count, accept_count
                )
            )

    def observe_decision(self, decision: dict) -> None:
        """Record the standard fully learned decision-audit representation."""

        mode_counts = decision.get("mode_counts", {})
        self.observe_candidate_set(
            decision["candidate_count"],
            accept_count=mode_counts.get("accept"),
        )

    def metrics(self) -> dict:
        capacity = self.storage_capacity_cells
        observed = self.storage_pressure_observed_steps
        return {
            "occupancy_pressure_metric_contract": (
                OCCUPANCY_PRESSURE_METRIC_CONTRACT
            ),
            "storage_capacity_cells": capacity,
            "peak_active_stored_count": self.peak_active_stored_count,
            "peak_active_stored_fraction": (
                self.peak_active_stored_count / capacity if capacity else 0.0
            ),
            "peak_physical_storage_occupancy_count": (
                self.peak_physical_storage_occupancy_count
            ),
            "peak_physical_storage_occupancy_fraction": (
                self.peak_physical_storage_occupancy_count / capacity
                if capacity
                else 0.0
            ),
            "minimum_free_storage_cells": self.minimum_free_storage_cells,
            "storage_pressure_observed_steps": observed,
            "storage_steps_at_or_above_80pct_occupied": (
                self.storage_steps_at_or_above_80pct_occupied
            ),
            "storage_step_fraction_at_or_above_80pct_occupied": (
                self.storage_steps_at_or_above_80pct_occupied / observed
                if observed
                else 0.0
            ),
            "maximum_inbound_waiting_count": (
                self.maximum_inbound_waiting_count
            ),
            "maximum_inbound_queue_count": self.maximum_inbound_queue_count,
            "minimum_live_candidate_count": self.minimum_live_candidate_count,
            "minimum_live_accept_candidate_count": (
                self.minimum_live_accept_candidate_count
            ),
        }


def summarize_occupancy_pressure_runs(runs) -> dict:
    """Aggregate episode diagnostics without treating missing data as zero."""

    runs = tuple(runs)
    pressure_runs = [
        item
        for item in runs
        if item.get("occupancy_pressure_metric_contract") is not None
    ]
    if not pressure_runs:
        return {}
    if len(pressure_runs) != len(runs):
        raise ValueError(
            "cannot mix runs with and without occupancy-pressure diagnostics"
        )
    contracts = {
        item["occupancy_pressure_metric_contract"] for item in pressure_runs
    }
    if contracts != {OCCUPANCY_PRESSURE_METRIC_CONTRACT}:
        raise ValueError("incompatible occupancy-pressure metric contracts")

    def values(key):
        return [float(item[key]) for item in pressure_runs]

    def optional_values(key):
        return [
            float(item[key])
            for item in pressure_runs
            if item.get(key) is not None
        ]

    observed_steps = int(
        sum(item["storage_pressure_observed_steps"] for item in pressure_runs)
    )
    high_steps = int(
        sum(
            item["storage_steps_at_or_above_80pct_occupied"]
            for item in pressure_runs
        )
    )
    candidates = optional_values("minimum_live_candidate_count")
    accept_candidates = optional_values("minimum_live_accept_candidate_count")
    return {
        "occupancy_pressure_metric_contract": (
            OCCUPANCY_PRESSURE_METRIC_CONTRACT
        ),
        "occupancy_pressure_episode_count": len(pressure_runs),
        "mean_storage_capacity_cells": (
            fsum(values("storage_capacity_cells")) / len(pressure_runs)
        ),
        "mean_peak_active_stored_count": (
            fsum(values("peak_active_stored_count")) / len(pressure_runs)
        ),
        "mean_peak_active_stored_fraction": (
            fsum(values("peak_active_stored_fraction")) / len(pressure_runs)
        ),
        "mean_peak_physical_storage_occupancy_count": (
            fsum(values("peak_physical_storage_occupancy_count"))
            / len(pressure_runs)
        ),
        "mean_peak_physical_storage_occupancy_fraction": (
            fsum(values("peak_physical_storage_occupancy_fraction"))
            / len(pressure_runs)
        ),
        "minimum_free_storage_cells": int(
            min(values("minimum_free_storage_cells"))
        ),
        "mean_minimum_free_storage_cells": (
            fsum(values("minimum_free_storage_cells")) / len(pressure_runs)
        ),
        "storage_pressure_observed_steps": observed_steps,
        "storage_steps_at_or_above_80pct_occupied": high_steps,
        "storage_step_fraction_at_or_above_80pct_occupied": (
            high_steps / observed_steps if observed_steps else 0.0
        ),
        "maximum_inbound_waiting_count": int(
            max(values("maximum_inbound_waiting_count"))
        ),
        "mean_maximum_inbound_waiting_count": (
            fsum(values("maximum_inbound_waiting_count"))
            / len(pressure_runs)
        ),
        "maximum_inbound_queue_count": int(
            max(values("maximum_inbound_queue_count"))
        ),
        "mean_maximum_inbound_queue_count": (
            fsum(values("maximum_inbound_queue_count")) / len(pressure_runs)
        ),
        "minimum_live_candidate_count": (
            int(min(candidates)) if candidates else None
        ),
        "mean_minimum_live_candidate_count": (
            fsum(candidates) / len(candidates) if candidates else None
        ),
        "minimum_live_accept_candidate_count": (
            int(min(accept_candidates)) if accept_candidates else None
        ),
        "mean_minimum_live_accept_candidate_count": (
            fsum(accept_candidates) / len(accept_candidates)
            if accept_candidates
            else None
        ),
    }


__all__ = [
    "HIGH_OCCUPANCY_THRESHOLD",
    "OCCUPANCY_PRESSURE_METRIC_CONTRACT",
    "OccupancyPressureTracker",
    "summarize_occupancy_pressure_runs",
]
