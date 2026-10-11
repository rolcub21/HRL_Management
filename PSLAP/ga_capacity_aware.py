"""Capacity-aware arrived-only rolling-GA assignment sources.

The historical rolling adapters optimize one unique cell for every arrived
pending block.  That formulation is stricter than Track B's atomic inbound
interface: one block must be placed now, while other arrived blocks may remain
queued.  Under contention the historical adapter can therefore reject a
feasible current placement merely because the number of queued blocks exceeds
the current shared candidate-mask cardinality.

This module defines new, versioned sources.  The historical classes and IDs are
left untouched.  At each decision the new sources optimize the mandatory
current block followed by the earliest arrived pending prefix that fits the
live candidate capacity.  Every excess arrived block is represented explicitly
as deferred in the committed source audit; it is never treated as assigned and
never silently discarded.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from typing import Iterable

from PSLAP.dynamic_yard import BlockView, Cell, YardSnapshot
from PSLAP.ga_optimizer import (
    GAConfig,
    optimize_duration_aware_rolling_assignment,
    optimize_operational_rolling_assignment,
    optimize_rolling_assignment,
)
from PSLAP.ga_policy import DEFAULT_ROLLING_GA_CONFIG, RollingGAStorageAssigner
from PSLAP.neutral_protocol import shared_candidate_mask


CAPACITY_AWARE_QUEUE_CONTRACT = (
    "mandatory_current_plus_earliest_arrived_capacity_limited_prefix_"
    "with_explicit_deferred_remainder_v1"
)
CAPACITY_AWARE_AUDIT_SCHEMA_VERSION = 1
ONLINE_INFORMATION_REGIME = "online_arrived_only"


class _CapacityAwareQueuedRollingGABase(RollingGAStorageAssigner):
    """Shared capacity/queue semantics for the four repaired GA variants."""

    VERSION = "abstract_capacity_aware_queued_rolling_ga"
    OBJECTIVE_VARIANT = "abstract"
    OBJECTIVE_LEXICOGRAPHIC_FIELDS: tuple[str, ...] = ()
    EXHAUSTIVE_SINGLETON = False

    def __init__(
        self,
        env,
        config: GAConfig = DEFAULT_ROLLING_GA_CONFIG,
        *,
        egress_weight: int = 4,
    ):
        super().__init__(env, config)
        if not isinstance(egress_weight, int) or isinstance(egress_weight, bool):
            raise TypeError("egress_weight must be an integer")
        if egress_weight < 1:
            raise ValueError("egress_weight must be positive")
        self.egress_weight = egress_weight
        self.last_planned_labels = ()
        self.last_deferred_labels = ()
        self._unbound_preview_plan: dict | None = None
        self._minted_preview_plans: dict[str, dict] = {}
        self._reserved_preview_plans: dict[str, dict] = {}
        self._committed_plans: list[dict] = []

    def on_episode_start(self) -> None:
        """Reset every behavioral and audit clock for a fresh episode."""

        self.decision_count = 0
        self.fallback_count = 0
        self.last_result = None
        self.last_known_labels = ()
        self.last_planned_labels = ()
        self.last_deferred_labels = ()
        self._unbound_preview_plan = None
        self._minted_preview_plans.clear()
        self._reserved_preview_plans.clear()
        self._committed_plans.clear()

    def _known_pending_blocks(self, current):
        """Read only blocks observable as arrived in pickup/waiting cells.

        The environment contains objects for future blocks, so merely iterating
        over ``env.blocks`` is not an information grant.  Future schedule data
        are never read: physical visibility/status are checked first, and only
        physically pending blocks have their already-realized arrival time and
        current remaining processing time inspected.
        """

        now = int(self.env.time_steps)
        observed = []
        for environment_index, env_block in enumerate(self.env.blocks):
            if env_block.position not in (
                self.env.pickup_cell,
                self.env.waiting_cell,
            ):
                continue
            if (
                env_block.stored
                or env_block.delivered
                or env_block.carrying
                or env_block.storage_location is not None
            ):
                continue
            arrival_step = int(env_block.arrival_step)
            if arrival_step > now:
                # Defensive exclusion for a malformed state that exposes a
                # future block in a queue cell.
                continue
            observed.append((arrival_step, environment_index, env_block))
        observed.sort(key=lambda item: (item[0], item[1]))
        views = [
            BlockView(
                label=env_block.label,
                position=env_block.position,
                remaining_time=float(env_block.get_remaining_storage_time()),
            )
            for _, _, env_block in observed
        ]
        return tuple(views)

    @staticmethod
    def _candidate_domain(
        yard: YardSnapshot,
        valid_candidates: Iterable[Cell] | None,
    ) -> tuple[Cell, ...]:
        occupied = frozenset(yard.occupancy())
        free_cells = yard.storage_cells - occupied
        if valid_candidates is None:
            raise ValueError(
                "capacity-aware rolling GA requires the current shared "
                "candidate mask"
            )
        else:
            domain = tuple(sorted(set(valid_candidates)))
            if any(cell not in free_cells for cell in domain):
                raise ValueError(
                    "capacity-aware rolling-GA domain contains an occupied "
                    "or non-storage cell"
                )
        if not domain:
            raise ValueError(
                "capacity-aware rolling GA requires a current admissible cell"
            )
        return domain

    def _capacity_limited_plan(
        self,
        yard: YardSnapshot,
        block: BlockView,
        valid_candidates: Iterable[Cell] | None,
    ) -> tuple[tuple[BlockView, ...], tuple[BlockView, ...], tuple[BlockView, ...], tuple[Cell, ...]]:
        observed = self._known_pending_blocks(block)
        current_matches = [item for item in observed if item.label == block.label]
        if len(current_matches) != 1:
            raise ValueError(
                "capacity-aware rolling GA requires one observed current block"
            )
        if len({item.label for item in observed}) != len(observed):
            raise ValueError("arrived pending block labels must be unique")
        live_pickup = [
            env_block
            for env_block in self.env.blocks
            if (
                env_block.position == self.env.pickup_cell
                and not env_block.stored
                and not env_block.delivered
                and not env_block.carrying
                and env_block.storage_location is None
                and int(env_block.arrival_step) <= int(self.env.time_steps)
            )
        ]
        if (
            len(live_pickup) != 1
            or live_pickup[0].label != block.label
            or tuple(block.position) != tuple(self.env.pickup_cell)
        ):
            raise ValueError(
                "capacity-aware rolling GA requires the unique observed pickup head"
            )
        # Track B exposes the actual FIFO queue head as the current inbound
        # block.  Reject an inconsistent caller rather than silently reorder
        # the observed queue and thereby change the method.
        if observed[0].label != block.label:
            raise ValueError(
                "capacity-aware rolling GA current block is not the observed "
                "FIFO pending head"
            )
        candidates = self._candidate_domain(yard, valid_candidates)
        capacity = min(len(observed), len(candidates))
        planned = observed[:capacity]
        deferred = observed[capacity:]
        if not planned or planned[0].label != block.label:
            raise RuntimeError("capacity-aware plan omitted its mandatory current block")
        if len(planned) > len(candidates):
            raise RuntimeError("capacity-aware plan exceeded live candidate capacity")
        if tuple(item.label for item in observed) != tuple(
            item.label for item in (*planned, *deferred)
        ):
            raise RuntimeError("capacity-aware pending partition is not exact")
        return observed, planned, deferred, candidates

    def _run_planned_optimizer(
        self,
        yard: YardSnapshot,
        planned: tuple[BlockView, ...],
        *,
        source: Cell,
        candidates: tuple[Cell, ...],
        config: GAConfig,
    ):
        raise NotImplementedError

    def _optimize_capacity_aware(
        self,
        yard: YardSnapshot,
        block: BlockView,
        source: Cell,
        valid_candidates: Iterable[Cell] | None,
    ):
        expected_shared_mask = tuple(
            shared_candidate_mask(yard, block, source)
        )
        if valid_candidates is None:
            valid_candidates = expected_shared_mask
        else:
            received_mask = tuple(valid_candidates)
            if received_mask != expected_shared_mask:
                raise ValueError(
                    "capacity-aware rolling GA requires the exact ordered "
                    "current shared candidate mask"
                )
        observed, planned, deferred, candidates = self._capacity_limited_plan(
            yard, block, valid_candidates
        )
        decision_config = replace(
            self.config,
            seed=self.config.seed + self.decision_count,
        )
        result = self._run_planned_optimizer(
            yard,
            planned,
            source=source,
            candidates=candidates,
            config=decision_config,
        )
        planned_labels = tuple(item.label for item in planned)
        current_index = planned_labels.index(block.label)
        chosen = tuple(result.best_assignment[current_index])
        observed_labels = tuple(item.label for item in observed)
        deferred_labels = tuple(item.label for item in deferred)
        candidate_mask_sha256 = hashlib.sha256(
            json.dumps(candidates, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        queue_partition_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "observed": observed_labels,
                    "planned": planned_labels,
                    "deferred": deferred_labels,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        record = {
            "audit_schema_version": CAPACITY_AWARE_AUDIT_SCHEMA_VERSION,
            "queue_contract": CAPACITY_AWARE_QUEUE_CONTRACT,
            "source_version": self.VERSION,
            "objective_variant": self.OBJECTIVE_VARIANT,
            "information_regime": ONLINE_INFORMATION_REGIME,
            "future_schedule_accessed": False,
            "selection_time_step": int(self.env.time_steps),
            "source_cell": tuple(source),
            "current_block_label": block.label,
            "observed_arrived_pending_labels": observed_labels,
            "planned_labels": planned_labels,
            "deferred_labels": deferred_labels,
            "observed_arrived_pending_count": len(observed),
            "planned_count": len(planned),
            "deferred_count": len(deferred),
            "admissible_candidate_count": len(candidates),
            "admissible_candidate_cells": candidates,
            "candidate_mask_sha256": candidate_mask_sha256,
            "queue_partition_sha256": queue_partition_sha256,
            "current_shared_mask_capacity": len(candidates),
            "optimizer_seed": int(decision_config.seed),
            "optimizer_config": asdict(decision_config),
            "optimizer_seed_offset_contract": (
                "base_seed_plus_committed_assignment_index"
            ),
            "egress_weight": (
                self.egress_weight
                if "duration" in self.OBJECTIVE_VARIANT
                or "operational" in self.OBJECTIVE_VARIANT
                else None
            ),
            "optimizer_mode": (
                "exhaustive_singleton"
                if self.EXHAUSTIVE_SINGLETON and len(planned) == 1
                else "genetic_search"
            ),
            "objective_lexicographic_fields": (
                self.OBJECTIVE_LEXICOGRAPHIC_FIELDS
            ),
            "best_assignment": tuple(tuple(cell) for cell in result.best_assignment),
            "best_cost_components": asdict(result.best_cost),
            "best_cost_scalar": int(result.best_cost.scalar),
            "chosen_cell": chosen,
            "current_block_planned": block.label in planned_labels,
            "current_block_plan_index": current_index,
            "current_block_is_gene_zero": current_index == 0,
            "current_block_is_unique_observed_pickup_head": True,
            "current_block_is_fifo_head": observed[0].label == block.label,
            "only_gene_zero_committed": True,
            "committed_gene_indices": (0,),
            "noncurrent_planned_genes_nonbinding": True,
            "capacity_respected": len(planned) <= len(candidates),
            "pending_partition_exact": tuple(item.label for item in observed)
            == tuple((*planned_labels, *(item.label for item in deferred))),
            "deferred_remainder_explicit": len(deferred)
            == len(observed) - len(planned),
            "deferred_semantics": (
                "nonbinding_rolling_horizon_overflow_not_an_executed_defer"
            ),
            "queue_overflow_labels": deferred_labels,
            "queue_overflow_count": len(deferred),
            "scheduler_defer_action_introduced": False,
            "queue_overflow_virtual_penalty": 0.0,
        }
        return result, record, chosen

    def preview(self, yard, block, source, valid_candidates=None):
        """Return one deterministic proposal without advancing the GA seed."""

        result, record, chosen = self._optimize_capacity_aware(
            yard, block, source, valid_candidates
        )
        # The selector binds this plan to its immutable proposal ID via
        # ``on_preview_bound``.  Until then it is not a committed/audited row.
        self._unbound_preview_plan = deepcopy(record)
        self.last_result = result
        self.last_known_labels = record["observed_arrived_pending_labels"]
        self.last_planned_labels = record["planned_labels"]
        self.last_deferred_labels = record["deferred_labels"]
        return chosen

    def on_preview_token_minted(
        self,
        *,
        proposal_id,
        block_label,
        chosen_cell,
        selection_time_step,
        candidate_count,
        candidate_mask_id,
    ) -> None:
        record = self._unbound_preview_plan
        self._unbound_preview_plan = None
        if record is None:
            raise RuntimeError("capacity-aware preview binding has no plan")
        for label, observed, expected in (
            ("block", block_label, record["current_block_label"]),
            ("cell", tuple(chosen_cell), tuple(record["chosen_cell"])),
            ("selection time", int(selection_time_step), record["selection_time_step"]),
            ("candidate count", int(candidate_count), record["admissible_candidate_count"]),
        ):
            if observed != expected:
                raise RuntimeError(
                    f"capacity-aware preview {label} differs from optimized plan"
                )
        if str(candidate_mask_id) != record["candidate_mask_sha256"][:16]:
            raise RuntimeError(
                "capacity-aware selector mask ID differs from optimized mask"
            )
        bound = deepcopy(record)
        bound["proposal_id"] = str(proposal_id)
        bound["candidate_mask_id"] = str(candidate_mask_id)
        incumbent = self._minted_preview_plans.get(str(proposal_id))
        if incumbent is not None and incumbent != bound:
            raise RuntimeError(
                "capacity-aware repeated proposal ID changed plan provenance"
            )
        self._minted_preview_plans[str(proposal_id)] = bound

    def on_preview_reserved(
        self,
        *,
        proposal_id,
        block_label,
        chosen_cell,
        selection_time_step,
        candidate_count,
        candidate_mask_id,
    ) -> None:
        proposal_id = str(proposal_id)
        record = self._minted_preview_plans.get(proposal_id)
        if record is None:
            raise RuntimeError("capacity-aware reservation has no minted preview")
        for label, observed, expected in (
            ("block", block_label, record["current_block_label"]),
            ("cell", tuple(chosen_cell), tuple(record["chosen_cell"])),
            ("selection time", int(selection_time_step), record["selection_time_step"]),
            ("candidate count", int(candidate_count), record["admissible_candidate_count"]),
            ("candidate mask", str(candidate_mask_id), record["candidate_mask_id"]),
        ):
            if observed != expected:
                raise RuntimeError(
                    f"capacity-aware reservation {label} differs from minted preview"
                )
        incumbent = self._reserved_preview_plans.get(proposal_id)
        if incumbent is not None and incumbent != record:
            raise RuntimeError("capacity-aware proposal reservation changed")
        self._reserved_preview_plans[proposal_id] = deepcopy(record)

    def propose(self, yard, block, source, valid_candidates=None):
        """Direct-call compatibility with an immediate audited commitment."""

        result, record, chosen = self._optimize_capacity_aware(
            yard, block, source, valid_candidates
        )
        self.last_result = result
        self.last_known_labels = record["observed_arrived_pending_labels"]
        self.last_planned_labels = record["planned_labels"]
        self.last_deferred_labels = record["deferred_labels"]
        self._commit_record(record, chosen_cell=chosen, commit_time_step=self.env.time_steps)
        self.decision_count += 1
        return chosen

    def _commit_record(self, record: dict, *, chosen_cell, commit_time_step: int) -> None:
        if tuple(chosen_cell) != tuple(record["chosen_cell"]):
            raise RuntimeError("capacity-aware committed cell differs from preview")
        committed = deepcopy(record)
        committed.update(
            {
                "decision_index": len(self._committed_plans),
                "commit_time_step": int(commit_time_step),
                "reservation_integrity": True,
            }
        )
        if committed["commit_time_step"] < committed["selection_time_step"]:
            raise RuntimeError("capacity-aware commit precedes selection")
        self._committed_plans.append(committed)

    def _validate_bound_commit(
        self,
        *,
        proposal_id,
        block_label,
        chosen_cell,
        selection_time_step,
        commit_time_step,
        candidate_count,
        candidate_mask_id,
    ) -> dict:
        record = self._reserved_preview_plans.get(str(proposal_id))
        if record is None:
            raise RuntimeError(
                "capacity-aware assignment commit has no bound preview plan"
            )
        for label, observed, expected in (
            ("proposal ID", str(proposal_id), record["proposal_id"]),
            ("block", block_label, record["current_block_label"]),
            ("selection time", int(selection_time_step), record["selection_time_step"]),
            ("candidate count", int(candidate_count), record["admissible_candidate_count"]),
            ("candidate mask", str(candidate_mask_id), record["candidate_mask_id"]),
        ):
            if observed != expected:
                raise RuntimeError(
                    f"capacity-aware commit {label} differs from bound preview"
                )
        if tuple(chosen_cell) != tuple(record["chosen_cell"]):
            raise RuntimeError(
                "capacity-aware commit cell differs from bound preview"
            )
        if int(commit_time_step) < int(selection_time_step):
            raise RuntimeError("capacity-aware commit precedes selection")
        return record

    def validate_bound_assignment_commit(self, **metadata) -> None:
        """Authenticate a reservation before the selector mutates live state."""

        self._validate_bound_commit(**metadata)

    def on_bound_assignment_committed(self, **metadata):
        record = self._validate_bound_commit(**metadata)
        proposal_id = str(metadata["proposal_id"])
        self._reserved_preview_plans.pop(proposal_id)
        self._minted_preview_plans.pop(proposal_id, None)
        self._commit_record(
            record,
            chosen_cell=metadata["chosen_cell"],
            commit_time_step=int(metadata["commit_time_step"]),
        )
        self.decision_count += 1

    def audit(self) -> dict:
        decisions = deepcopy(self._committed_plans)
        minted_unreserved_ids = tuple(
            sorted(
                set(self._minted_preview_plans)
                - set(self._reserved_preview_plans)
            )
        )
        reserved_uncommitted_ids = tuple(
            sorted(self._reserved_preview_plans)
        )
        return {
            "audit_schema_version": CAPACITY_AWARE_AUDIT_SCHEMA_VERSION,
            "queue_contract": CAPACITY_AWARE_QUEUE_CONTRACT,
            "assignment_source_version": self.VERSION,
            "objective_variant": self.OBJECTIVE_VARIANT,
            "information_regime": ONLINE_INFORMATION_REGIME,
            "future_schedule_accessed": False,
            "future_schedule_fields_read": (),
            "observable_fields_used": (
                "position",
                "stored",
                "delivered",
                "realized_arrival_step",
                "current_remaining_storage_time",
            ),
            "pending_order": "mandatory_current_then_earliest_arrived_stable",
            "preview_contract": (
                "environment_seed_and_decision_counter_invariant_with_"
                "replaceable_diagnostic_cache_v1"
            ),
            "optimizer_base_config": asdict(self.config),
            "optimizer_seed_offset_contract": (
                "base_seed_plus_committed_assignment_index"
            ),
            "egress_weight": (
                self.egress_weight
                if "duration" in self.OBJECTIVE_VARIANT
                or "operational" in self.OBJECTIVE_VARIANT
                else None
            ),
            "objective_lexicographic_fields": (
                self.OBJECTIVE_LEXICOGRAPHIC_FIELDS
            ),
            "deferred_semantics": (
                "nonbinding_rolling_horizon_overflow_not_an_executed_defer"
            ),
            "scheduler_defer_action_introduced": False,
            "queue_overflow_virtual_penalty": 0.0,
            "decision_count": self.decision_count,
            "half_minted_unbound_preview_present": (
                self._unbound_preview_plan is not None
            ),
            "committed_plan_count": len(decisions),
            "minted_unreserved_preview_count": len(minted_unreserved_ids),
            "minted_unreserved_proposal_ids": minted_unreserved_ids,
            "reserved_uncommitted_preview_count": len(
                reserved_uncommitted_ids
            ),
            "reserved_uncommitted_proposal_ids": reserved_uncommitted_ids,
            "decisions_with_deferred_remainder": sum(
                int(item["deferred_count"] > 0) for item in decisions
            ),
            "observed_pending_membership_count_total": sum(
                item["observed_arrived_pending_count"] for item in decisions
            ),
            "planned_membership_count_total": sum(
                item["planned_count"] for item in decisions
            ),
            "deferred_membership_count_total": sum(
                item["deferred_count"] for item in decisions
            ),
            "max_observed_arrived_pending_count": max(
                (item["observed_arrived_pending_count"] for item in decisions),
                default=0,
            ),
            "max_planned_count": max(
                (item["planned_count"] for item in decisions), default=0
            ),
            "max_deferred_count": max(
                (item["deferred_count"] for item in decisions), default=0
            ),
            "all_current_blocks_planned": all(
                item["current_block_planned"] for item in decisions
            ),
            "all_current_blocks_are_gene_zero": all(
                item["current_block_is_gene_zero"] for item in decisions
            ),
            "all_current_blocks_are_unique_observed_pickup_heads": all(
                item["current_block_is_unique_observed_pickup_head"]
                for item in decisions
            ),
            "all_current_blocks_are_fifo_heads": all(
                item["current_block_is_fifo_head"] for item in decisions
            ),
            "only_gene_zero_ever_committed": all(
                item["only_gene_zero_committed"] for item in decisions
            ),
            "committed_gene_indices": tuple(
                sorted(
                    {
                        int(index)
                        for item in decisions
                        for index in item["committed_gene_indices"]
                    }
                )
            ),
            "all_noncurrent_planned_genes_nonbinding": all(
                item["noncurrent_planned_genes_nonbinding"]
                for item in decisions
            ),
            "all_capacity_limits_respected": all(
                item["capacity_respected"] for item in decisions
            ),
            "all_pending_partitions_exact": all(
                item["pending_partition_exact"] for item in decisions
            ),
            "all_deferred_remainders_explicit": all(
                item["deferred_remainder_explicit"] for item in decisions
            ),
            "all_reservations_integral": all(
                item["reservation_integrity"] for item in decisions
            ),
            "decisions": decisions,
        }


class CapacityAwareRollingGAStorageAssigner(_CapacityAwareQueuedRollingGABase):
    VERSION = "pslap_2009_rolling_ga_arrived_only_capacity_aware_partial_v3"
    OBJECTIVE_VARIANT = "pslap_2009_rolling"
    OBJECTIVE_LEXICOGRAPHIC_FIELDS = (
        "infeasible_events",
        "obstructive_moves",
        "route_steps",
    )

    def _run_planned_optimizer(
        self, yard, planned, *, source, candidates, config
    ):
        return optimize_rolling_assignment(
            yard,
            planned,
            source=source,
            config=config,
            storage_cells=candidates,
        )


class CapacityAwareDurationAwareRollingGAAssigner(
    _CapacityAwareQueuedRollingGABase
):
    VERSION = "duration_aware_rolling_ga_arrived_only_capacity_aware_partial_v2"
    OBJECTIVE_VARIANT = "duration_aware_rolling"
    OBJECTIVE_LEXICOGRAPHIC_FIELDS = (
        "infeasible_events",
        "priority_obstruction_cost",
        "weighted_route_steps",
        "raw_obstructive_moves",
        "raw_route_steps",
    )

    def _run_planned_optimizer(
        self, yard, planned, *, source, candidates, config
    ):
        return optimize_duration_aware_rolling_assignment(
            yard,
            planned,
            source=source,
            config=config,
            storage_cells=candidates,
            egress_weight=self.egress_weight,
        )


class CapacityAwareOperationalRollingGAAssigner(
    _CapacityAwareQueuedRollingGABase
):
    VERSION = "operational_rolling_ga_arrived_only_capacity_aware_partial_v3"
    OBJECTIVE_VARIANT = "operational_rolling"
    OBJECTIVE_LEXICOGRAPHIC_FIELDS = (
        "infeasible_events",
        "priority_obstruction_cost",
        "weighted_route_steps",
        "raw_obstructive_moves",
        "raw_route_steps",
    )

    def _run_planned_optimizer(
        self, yard, planned, *, source, candidates, config
    ):
        return optimize_operational_rolling_assignment(
            yard,
            planned,
            source=source,
            config=config,
            storage_cells=candidates,
            egress_weight=self.egress_weight,
        )


class CapacityAwareCompleteRollingGAAssigner(
    CapacityAwareDurationAwareRollingGAAssigner
):
    VERSION = (
        "duration_aware_rolling_ga_singleton_complete_"
        "capacity_aware_partial_v4"
    )
    OBJECTIVE_VARIANT = "duration_aware_singleton_complete_rolling"
    EXHAUSTIVE_SINGLETON = True

    def _run_planned_optimizer(
        self, yard, planned, *, source, candidates, config
    ):
        return optimize_duration_aware_rolling_assignment(
            yard,
            planned,
            source=source,
            config=config,
            storage_cells=candidates,
            egress_weight=self.egress_weight,
            exhaustive_singleton=True,
        )


CAPACITY_AWARE_ASSIGNER_CLASSES = (
    CapacityAwareRollingGAStorageAssigner,
    CapacityAwareDurationAwareRollingGAAssigner,
    CapacityAwareOperationalRollingGAAssigner,
    CapacityAwareCompleteRollingGAAssigner,
)


__all__ = [
    "CAPACITY_AWARE_ASSIGNER_CLASSES",
    "CAPACITY_AWARE_AUDIT_SCHEMA_VERSION",
    "CAPACITY_AWARE_QUEUE_CONTRACT",
    "CapacityAwareCompleteRollingGAAssigner",
    "CapacityAwareDurationAwareRollingGAAssigner",
    "CapacityAwareOperationalRollingGAAssigner",
    "CapacityAwareRollingGAStorageAssigner",
    "ONLINE_INFORMATION_REGIME",
]
