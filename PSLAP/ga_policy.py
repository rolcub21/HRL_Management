"""Offline and rolling GA assigners using the common online dispatcher."""

from __future__ import annotations

from dataclasses import replace

from PSLAP.dynamic_yard import (
    BlockView,
    is_storage_location_admissible,
    select_storage_location,
)
from PSLAP.ga_optimizer import (
    GAConfig,
    optimize_rolling_assignment,
    optimize_duration_aware_rolling_assignment,
    optimize_operational_rolling_assignment,
    optimize_schedule_assignment,
)
from PSLAP.online_policy import OnlinePSLAPPolicy


DEFAULT_ROLLING_GA_CONFIG = GAConfig(
    population_size=16,
    generations=10,
    elite_count=2,
    tournament_size=3,
)


class OfflineGAStorageAssigner:
    def __init__(self, env, assignment):
        self.env = env
        self.assignment = {
            block.label: cell
            for block, cell in zip(env.blocks, assignment)
        }
        self.fallback_count = 0

    def propose(self, yard, block, source, valid_candidates=None):
        return self.assignment.get(block.label)

    def __call__(self, yard, block, source):
        preferred = self.propose(yard, block, source)
        if preferred is not None and is_storage_location_admissible(
            yard, block, preferred, source=source
        ):
            return preferred
        self.fallback_count += 1
        return select_storage_location(yard, block, source=source)


class OfflinePSLAPGAPolicy(OnlinePSLAPPolicy):
    """Optimize the complete future schedule before online execution."""

    def __init__(self, env, config: GAConfig = GAConfig()):
        self.ga_result = optimize_schedule_assignment(env, config)
        self.ga_assigner = OfflineGAStorageAssigner(
            env, self.ga_result.best_assignment
        )
        super().__init__(env, self.ga_assigner)


class RollingGAStorageAssigner:
    """Reoptimize over arrived pending blocks at every inbound epoch."""

    VERSION = "pslap_2009_rolling_ga_arrived_only_v2"

    def __init__(self, env, config: GAConfig = DEFAULT_ROLLING_GA_CONFIG):
        self.env = env
        self.config = config
        self.decision_count = 0
        self.fallback_count = 0
        self.last_result = None
        self.last_known_labels = ()

    def _known_pending_blocks(self, current):
        known = [
            env_block
            for env_block in self.env.blocks
            if (
                env_block.position in (self.env.pickup_cell, self.env.waiting_cell)
                and not env_block.stored
                and not env_block.delivered
            )
        ]
        # Python's stable sort preserves the environment's block order when
        # several arrivals share the same discrete timestamp.
        known.sort(key=lambda item: item.arrival_step)
        views = [
            BlockView(
                label=env_block.label,
                position=self.env.pickup_cell,
                remaining_time=float(env_block.get_remaining_storage_time()),
            )
            for env_block in known
        ]
        if current.label not in {item.label for item in views}:
            views.insert(0, current)
        return tuple(views)

    def _optimize(self, yard, block, source, valid_candidates=None):
        pending = self._known_pending_blocks(block)
        decision_config = replace(
            self.config,
            seed=self.config.seed + self.decision_count,
        )
        result = optimize_rolling_assignment(
            yard,
            pending,
            source=source,
            config=decision_config,
            storage_cells=valid_candidates,
        )
        known_labels = tuple(item.label for item in pending)
        index = known_labels.index(block.label)
        return result, known_labels, result.best_assignment[index]

    def preview(self, yard, block, source, valid_candidates=None):
        """Return the next GA proposal without advancing its decision seed."""

        _, _, chosen = self._optimize(
            yard, block, source, valid_candidates
        )
        return chosen

    def propose(self, yard, block, source, valid_candidates=None):
        result, known_labels, chosen = self._optimize(
            yard, block, source, valid_candidates
        )
        self.last_result = result
        self.last_known_labels = known_labels
        self.decision_count += 1
        return chosen

    def on_assignment_committed(self, *, block_label, chosen_cell, time_step):
        """Advance the deterministic GA seed after a bound preview is committed."""

        self.decision_count += 1

    def __call__(self, yard, block, source):
        preferred = self.propose(yard, block, source)
        if is_storage_location_admissible(
            yard, block, preferred, source=source
        ):
            return preferred
        self.fallback_count += 1
        return select_storage_location(yard, block, source=source)


class DurationAwareRollingGAStorageAssigner(RollingGAStorageAssigner):
    """Online GA with urgency-ranked obstruction and egress-duration costs."""

    VERSION = "duration_aware_rolling_ga_arrived_only_v1"

    def __init__(
        self,
        env,
        config: GAConfig = DEFAULT_ROLLING_GA_CONFIG,
        *,
        egress_weight: int = 4,
    ):
        super().__init__(env, config)
        if not isinstance(egress_weight, int) or egress_weight < 1:
            raise ValueError("egress_weight must be a positive integer")
        self.egress_weight = egress_weight

    def _optimize(self, yard, block, source, valid_candidates=None):
        pending = self._known_pending_blocks(block)
        decision_config = replace(
            self.config,
            seed=self.config.seed + self.decision_count,
        )
        result = optimize_duration_aware_rolling_assignment(
            yard,
            pending,
            source=source,
            config=decision_config,
            storage_cells=valid_candidates,
            egress_weight=self.egress_weight,
        )
        known_labels = tuple(item.label for item in pending)
        index = known_labels.index(block.label)
        return result, known_labels, result.best_assignment[index]


class OperationalRollingGAStorageAssigner(DurationAwareRollingGAStorageAssigner):
    """V2 extension emphasizing the assigned blocks' egress durations."""

    VERSION = "operational_rolling_ga_arrived_only_v2"

    def _optimize(self, yard, block, source, valid_candidates=None):
        pending = self._known_pending_blocks(block)
        decision_config = replace(
            self.config,
            seed=self.config.seed + self.decision_count,
        )
        result = optimize_operational_rolling_assignment(
            yard,
            pending,
            source=source,
            config=decision_config,
            storage_cells=valid_candidates,
            egress_weight=self.egress_weight,
        )
        known_labels = tuple(item.label for item in pending)
        index = known_labels.index(block.label)
        return result, known_labels, result.best_assignment[index]


class CompleteSingletonDurationAwareRollingGAAssigner(
    DurationAwareRollingGAStorageAssigner
):
    """V3 hybrid: complete one-cell search, GA for multi-block epochs."""

    VERSION = "duration_aware_rolling_ga_singleton_complete_v3"

    def _optimize(self, yard, block, source, valid_candidates=None):
        pending = self._known_pending_blocks(block)
        decision_config = replace(
            self.config,
            seed=self.config.seed + self.decision_count,
        )
        result = optimize_duration_aware_rolling_assignment(
            yard,
            pending,
            source=source,
            config=decision_config,
            storage_cells=valid_candidates,
            egress_weight=self.egress_weight,
            exhaustive_singleton=True,
        )
        known_labels = tuple(item.label for item in pending)
        index = known_labels.index(block.label)
        return result, known_labels, result.best_assignment[index]


class RollingPSLAPGAPolicy(OnlinePSLAPPolicy):
    """Online GA assignment using arrived information only."""

    def __init__(self, env, config: GAConfig = DEFAULT_ROLLING_GA_CONFIG):
        self.ga_assigner = RollingGAStorageAssigner(env, config)
        super().__init__(env, self.ga_assigner)


# Compatibility names for code written before the offline/rolling split.
GAStorageAssigner = OfflineGAStorageAssigner
PSLAPGAPolicy = OfflinePSLAPGAPolicy


__all__ = [
    "DEFAULT_ROLLING_GA_CONFIG",
    "CompleteSingletonDurationAwareRollingGAAssigner",
    "DurationAwareRollingGAStorageAssigner",
    "GAStorageAssigner",
    "OfflineGAStorageAssigner",
    "OperationalRollingGAStorageAssigner",
    "OfflinePSLAPGAPolicy",
    "PSLAPGAPolicy",
    "RollingGAStorageAssigner",
    "RollingPSLAPGAPolicy",
]
