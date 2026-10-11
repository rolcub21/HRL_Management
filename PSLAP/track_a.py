"""Neutral Track A assignment interface and strict validation protocol."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral
from time import perf_counter
from typing import Optional, Protocol

import numpy as np
import torch

from example.helper.timing_metrics import summarize_block_storage_flow
from PSLAP.dynamic_yard import (
    BlockView,
    Cell,
    YardSnapshot,
    select_storage_location,
    shortest_clear_path,
)
from PSLAP.ga_optimizer import GAConfig, optimize_schedule_assignment
from PSLAP.ga_capacity_aware import (
    CapacityAwareCompleteRollingGAAssigner,
    CapacityAwareDurationAwareRollingGAAssigner,
    CapacityAwareOperationalRollingGAAssigner,
    CapacityAwareRollingGAStorageAssigner,
)
from PSLAP.ga_policy import (
    DEFAULT_ROLLING_GA_CONFIG,
    CompleteSingletonDurationAwareRollingGAAssigner,
    DurationAwareRollingGAStorageAssigner,
    OperationalRollingGAStorageAssigner,
    OfflineGAStorageAssigner,
    RollingGAStorageAssigner,
)
from PSLAP.online_policy import OnlinePSLAPPolicy
from PSLAP.neutral_protocol import (
    nearest_from_candidates as _nearest_from_candidates,
    neutral_relocation_selector,
    shared_candidate_mask,
)


TRACK_A_DYNAMIC = "dynamic_pslap"
TRACK_A_NEAREST_FREE = "nearest_free"
TRACK_A_GA_ROLLING = "pslap_ga_2009_rolling"
TRACK_A_GA_ROLLING_DURATION_AWARE = "pslap_ga_duration_aware_rolling"
TRACK_A_GA_ROLLING_OPERATIONAL = "pslap_ga_operational_rolling"
TRACK_A_GA_ROLLING_COMPLETE = "pslap_ga_duration_aware_complete_rolling"
TRACK_A_GA_ROLLING_CAPACITY_AWARE = (
    "pslap_ga_2009_rolling_capacity_aware_partial"
)
TRACK_A_GA_ROLLING_DURATION_AWARE_CAPACITY_AWARE = (
    "pslap_ga_duration_aware_rolling_capacity_aware_partial"
)
TRACK_A_GA_ROLLING_OPERATIONAL_CAPACITY_AWARE = (
    "pslap_ga_operational_rolling_capacity_aware_partial"
)
TRACK_A_GA_ROLLING_COMPLETE_CAPACITY_AWARE = (
    "pslap_ga_duration_aware_complete_rolling_capacity_aware_partial"
)
TRACK_A_GA_OFFLINE = "pslap_ga_2009_offline"
TRACK_A_REG_SELECTOR = "reg_selector"
TRACK_A_REG_SELECTOR_V4 = "reg_selector_v4"
TRACK_A_REG_SELECTOR_V5 = "reg_selector_v5"
TRACK_A_KIM2020_A3C_SPATIAL = "kim2020_a3c_spatial_adapted"
TRACK_A_METHODS = (
    TRACK_A_KIM2020_A3C_SPATIAL,
    TRACK_A_REG_SELECTOR,
    TRACK_A_REG_SELECTOR_V4,
    TRACK_A_REG_SELECTOR_V5,
    TRACK_A_DYNAMIC,
    TRACK_A_NEAREST_FREE,
    TRACK_A_GA_ROLLING,
    TRACK_A_GA_ROLLING_DURATION_AWARE,
    TRACK_A_GA_ROLLING_OPERATIONAL,
    TRACK_A_GA_ROLLING_COMPLETE,
    TRACK_A_GA_ROLLING_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_DURATION_AWARE_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_OPERATIONAL_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_COMPLETE_CAPACITY_AWARE,
    TRACK_A_GA_OFFLINE,
)

TRACK_A_INFORMATION = {
    TRACK_A_KIM2020_A3C_SPATIAL: "online_arrived_only",
    TRACK_A_REG_SELECTOR: "online",
    TRACK_A_REG_SELECTOR_V4: "online",
    TRACK_A_REG_SELECTOR_V5: "online",
    TRACK_A_DYNAMIC: "online",
    TRACK_A_NEAREST_FREE: "online",
    TRACK_A_GA_ROLLING: "online_arrived_only",
    TRACK_A_GA_ROLLING_DURATION_AWARE: "online_arrived_only",
    TRACK_A_GA_ROLLING_OPERATIONAL: "online_arrived_only",
    TRACK_A_GA_ROLLING_COMPLETE: "online_arrived_only",
    TRACK_A_GA_ROLLING_CAPACITY_AWARE: "online_arrived_only",
    TRACK_A_GA_ROLLING_DURATION_AWARE_CAPACITY_AWARE: "online_arrived_only",
    TRACK_A_GA_ROLLING_OPERATIONAL_CAPACITY_AWARE: "online_arrived_only",
    TRACK_A_GA_ROLLING_COMPLETE_CAPACITY_AWARE: "online_arrived_only",
    TRACK_A_GA_OFFLINE: "offline_full_schedule",
}

NEAREST_FREE_ASSIGNMENT_SOURCE_VERSION = (
    "nearest_free_shared_mask_shortest_path_v1"
)
DYNAMIC_ASSIGNMENT_SOURCE_VERSION = (
    "dynamic_pslap_earlier_access_assignment_v1"
)


class AssignmentSource(Protocol):
    """A method may propose one cell but may not replace the shared mask."""

    def propose(
        self,
        yard: YardSnapshot,
        block: BlockView,
        source: Cell,
        valid_candidates: tuple[Cell, ...],
    ):
        ...


class NearestFreeAssignmentSource:
    VERSION = NEAREST_FREE_ASSIGNMENT_SOURCE_VERSION

    def propose(self, yard, block, source, valid_candidates):
        return _nearest_from_candidates(
            yard, block, source, valid_candidates
        )

    def preview(self, yard, block, source, valid_candidates):
        """Return the same deterministic proposal without side effects."""

        return self.propose(yard, block, source, valid_candidates)


class DynamicAssignmentSource:
    VERSION = DYNAMIC_ASSIGNMENT_SOURCE_VERSION

    def propose(self, yard, block, source, valid_candidates):
        # The method may apply its own preference screen. Returning None while
        # the shared mask is nonempty is recorded as a strict method failure.
        return select_storage_location(yard, block, source=source)

    def preview(self, yard, block, source, valid_candidates):
        """Return the same deterministic proposal without side effects."""

        return self.propose(yard, block, source, valid_candidates)


class REGSelectorAssignmentSource:
    """Use a frozen v3 selector Q-network under the exact shared mask."""

    def __init__(self, env, selector):
        if getattr(selector, "FEATURE_VERSION", None) != 3:
            raise ValueError(
                "Track A requires information-safe selector feature version 3"
            )
        self.env = env
        self.selector = selector
        self.selector.set_learning_enabled(False)
        self.selector.q.eval()

    def propose(self, yard, block, source, valid_candidates):
        env_block = next(
            item for item in self.env.blocks if item.label == block.label
        )
        selector = self.selector
        usage_counts = np.asarray(
            [self.env.storage_counts[cell] for cell in selector.cells],
            dtype=np.float32,
        )
        max_usage = max(float(usage_counts.max()), 1.0)
        usage = 1.0 - usage_counts / max_usage
        candidate_set = frozenset(valid_candidates)
        mask = np.asarray(
            [1.0 if cell in candidate_set else 0.0 for cell in selector.cells],
            dtype=np.float32,
        )

        saved_state = self.env.current_state
        saved_position = env_block.position
        saved_carrying = env_block.carrying
        try:
            # The selector was trained at the post-pickup decision state. Build
            # that read-only counterfactual without executing or committing it.
            self.env.current_state = source
            env_block.position = source
            env_block.carrying = True
            occupied = {
                item.position
                for item in self.env.blocks
                if not item.delivered and not item.carrying
            }
            congestion = np.asarray(
                [1.0 if cell in occupied else 0.0 for cell in selector.cells],
                dtype=np.float32,
            )
            selector.f_cong = congestion
            features = selector._φ(mask, usage, congestion, source)
            with torch.no_grad():
                values = selector.q(
                    torch.from_numpy(features)
                    .unsqueeze(0)
                    .to(selector.device)
                )[0]
            candidate_indices = [
                index
                for index, cell in enumerate(selector.cells)
                if cell in candidate_set
            ]
            if not candidate_indices:
                return None
            best = max(
                candidate_indices,
                key=lambda index: (float(values[index].item()), -index),
            )
            return selector.cells[best]
        finally:
            env_block.position = saved_position
            env_block.carrying = saved_carrying
            self.env.current_state = saved_state


def _normalise_cell(value) -> Optional[Cell]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or not isinstance(value[0], Integral)
        or not isinstance(value[1], Integral)
    ):
        return None
    return int(value[0]), int(value[1])


def _invalid_reason(
    proposal,
    normalised: Optional[Cell],
    yard: YardSnapshot,
    block: BlockView,
    source: Cell,
    valid_candidates: tuple[Cell, ...],
) -> Optional[str]:
    if proposal is None:
        return "no_proposal"
    if normalised is None:
        return "malformed_cell"
    if normalised not in yard.storage_cells:
        return "non_storage_cell"
    if normalised in yard.occupancy():
        return "occupied_cell"
    if shortest_clear_path(
        yard, source, normalised, ignore_labels=(block.label,)
    ) is None:
        return "unreachable_cell"
    if normalised not in valid_candidates:
        return "outside_shared_candidate_mask"
    return None


@dataclass(frozen=True)
class AssignmentDecision:
    decision_index: int
    time_step: int
    block_label: str
    valid_candidates: tuple[Cell, ...]
    proposal_repr: str
    proposed_cell: Optional[Cell]
    valid: bool
    invalid_reason: Optional[str]
    fallback_used: bool
    executed_cell: Optional[Cell]
    delivered_before_decision: int
    planning_seconds: float

    def to_dict(self):
        return asdict(self)

    def to_compact_dict(self):
        encoded = json.dumps(
            self.valid_candidates, separators=(",", ":")
        ).encode("utf-8")
        return {
            "decision_index": self.decision_index,
            "time_step": self.time_step,
            "block_label": self.block_label,
            "candidate_count": len(self.valid_candidates),
            "candidate_mask_id": hashlib.sha256(encoded).hexdigest()[:16],
            "proposal_repr": self.proposal_repr,
            "proposed_cell": self.proposed_cell,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "fallback_used": self.fallback_used,
            "executed_cell": self.executed_cell,
            "delivered_before_decision": self.delivered_before_decision,
            "planning_seconds": self.planning_seconds,
        }


class StrictAssignmentAdapter:
    """Validate raw proposals and isolate emergency continuation fallback."""

    def __init__(self, env, method_name: str, source: AssignmentSource):
        self.env = env
        self.method_name = method_name
        self.source = source
        self.decisions: list[AssignmentDecision] = []
        self.invalid_assignment_count = 0
        self.infeasible_epoch_count = 0
        self.infeasible_block_counts = Counter()
        self.fallback_count = 0
        self.contaminated = False
        self.first_invalid_step = None
        self.delivered_before_first_invalid = None
        self.planning_seconds = 0.0

    def __call__(self, yard, block, source):
        candidates = shared_candidate_mask(yard, block, source)
        delivered = sum(item.delivered for item in self.env.blocks)
        if not candidates:
            self.infeasible_epoch_count += 1
            self.infeasible_block_counts[block.label] += 1
            return None

        started = perf_counter()
        proposal = self.source.propose(
            yard, block, source, candidates
        )
        elapsed = perf_counter() - started
        self.planning_seconds += elapsed
        normalised = _normalise_cell(proposal)
        reason = _invalid_reason(
            proposal,
            normalised,
            yard,
            block,
            source,
            candidates,
        )
        valid = reason is None
        fallback_used = not valid
        executed = normalised if valid else _nearest_from_candidates(
            yard, block, source, candidates
        )

        if not valid:
            self.invalid_assignment_count += 1
            self.fallback_count += int(executed is not None)
            if not self.contaminated:
                self.contaminated = True
                self.first_invalid_step = int(self.env.time_steps)
                self.delivered_before_first_invalid = delivered

        self.decisions.append(
            AssignmentDecision(
                decision_index=len(self.decisions),
                time_step=int(self.env.time_steps),
                block_label=block.label,
                valid_candidates=candidates,
                proposal_repr=repr(proposal),
                proposed_cell=normalised,
                valid=valid,
                invalid_reason=reason,
                fallback_used=fallback_used,
                executed_cell=executed,
                delivered_before_decision=delivered,
                planning_seconds=elapsed,
            )
        )
        return executed

    @property
    def invalid_reasons(self):
        return dict(
            Counter(
                decision.invalid_reason
                for decision in self.decisions
                if decision.invalid_reason
                and decision.invalid_reason != "no_feasible_candidates"
            )
        )


class TrackAOnlinePolicy(OnlinePSLAPPolicy):
    """One assignment source under the fixed neutral downstream stack."""

    def __init__(self, env, method_name: str, source: AssignmentSource):
        self.assignment_audit = StrictAssignmentAdapter(
            env, method_name, source
        )
        super().__init__(
            env,
            self.assignment_audit,
            relocation_selector=neutral_relocation_selector,
        )


def make_assignment_source(
    method_name: str,
    env,
    *,
    offline_config: GAConfig = GAConfig(),
    rolling_config: GAConfig = DEFAULT_ROLLING_GA_CONFIG,
):
    """Construct one audited Track A source after the episode has been reset."""

    if method_name == TRACK_A_DYNAMIC:
        return DynamicAssignmentSource()
    if method_name == TRACK_A_NEAREST_FREE:
        return NearestFreeAssignmentSource()
    if method_name == TRACK_A_GA_ROLLING:
        return RollingGAStorageAssigner(env, rolling_config)
    if method_name == TRACK_A_GA_ROLLING_DURATION_AWARE:
        return DurationAwareRollingGAStorageAssigner(env, rolling_config)
    if method_name == TRACK_A_GA_ROLLING_OPERATIONAL:
        return OperationalRollingGAStorageAssigner(env, rolling_config)
    if method_name == TRACK_A_GA_ROLLING_COMPLETE:
        return CompleteSingletonDurationAwareRollingGAAssigner(
            env, rolling_config
        )
    if method_name == TRACK_A_GA_ROLLING_CAPACITY_AWARE:
        return CapacityAwareRollingGAStorageAssigner(env, rolling_config)
    if method_name == TRACK_A_GA_ROLLING_DURATION_AWARE_CAPACITY_AWARE:
        return CapacityAwareDurationAwareRollingGAAssigner(env, rolling_config)
    if method_name == TRACK_A_GA_ROLLING_OPERATIONAL_CAPACITY_AWARE:
        return CapacityAwareOperationalRollingGAAssigner(env, rolling_config)
    if method_name == TRACK_A_GA_ROLLING_COMPLETE_CAPACITY_AWARE:
        return CapacityAwareCompleteRollingGAAssigner(env, rolling_config)
    if method_name == TRACK_A_GA_OFFLINE:
        result = optimize_schedule_assignment(env, offline_config)
        source = OfflineGAStorageAssigner(env, result.best_assignment)
        source.ga_result = result
        return source
    if method_name in (
        TRACK_A_KIM2020_A3C_SPATIAL,
        TRACK_A_REG_SELECTOR,
        TRACK_A_REG_SELECTOR_V4,
        TRACK_A_REG_SELECTOR_V5,
    ):
        raise ValueError(
            f"{method_name} requires a loaded assignment_source override"
        )
    raise ValueError(
        f"Unknown Track A assignment method {method_name!r}; "
        f"choose one of {TRACK_A_METHODS}"
    )


def run_track_a_episode(
    env,
    method_name: str,
    *,
    max_steps: int,
    episode_instance=None,
    offline_config: GAConfig = GAConfig(),
    rolling_config: GAConfig = DEFAULT_ROLLING_GA_CONFIG,
    assignment_source: Optional[AssignmentSource] = None,
):
    """Run strict assignment auditing with operational fallback continuation."""

    env.reset(instance=episode_instance)
    setup_started = perf_counter()
    source = assignment_source
    if source is None:
        source = make_assignment_source(
            method_name,
            env,
            offline_config=offline_config,
            rolling_config=rolling_config,
        )
    source_setup_seconds = perf_counter() - setup_started
    on_episode_start = getattr(source, "on_episode_start", None)
    if callable(on_episode_start):
        on_episode_start()
    policy = TrackAOnlinePolicy(env, method_name, source)
    generator = policy.step()
    audit = policy.assignment_audit

    total_reward = 0.0
    strict_return = 0.0
    errors = []
    strict_errors = []
    obstructive_moves = 0
    strict_obstructive_moves = 0
    illegal_drops = 0
    empty_travel_steps = 0
    loaded_travel_steps = 0
    wait_actions = 0
    pickup_actions = 0
    putdown_actions = 0
    contaminated_steps = 0
    contaminated_deliveries = 0
    steps = 0

    execution_started = perf_counter()
    while steps < max_steps and not env.is_state_terminal(env.current_state):
        try:
            action = next(generator)
        except StopIteration:
            generator = policy.step()
            action = next(generator)

        contaminated = audit.contaminated
        carrying = any(block.carrying for block in env.blocks)
        if action in (
            env.ACTION_IDS["UP"],
            env.ACTION_IDS["DOWN"],
            env.ACTION_IDS["LEFT"],
            env.ACTION_IDS["RIGHT"],
        ):
            if carrying:
                loaded_travel_steps += 1
            else:
                empty_travel_steps += 1
        elif action == env.ACTION_IDS["WAIT"]:
            wait_actions += 1
        elif action == env.ACTION_IDS["PICKUP"]:
            pickup_actions += 1
        elif action == env.ACTION_IDS["PUTDOWN"]:
            putdown_actions += 1

        _, reward, done, info = env.step(action)
        reward = float(reward)
        on_step = getattr(source, "on_step", None)
        if callable(on_step):
            on_step(reward, info)
        total_reward += reward
        steps += 1
        if contaminated:
            contaminated_steps += 1
        else:
            strict_return += reward
        if "delivery_error_time" in info:
            error = float(info["delivery_error_time"])
            errors.append(error)
            if contaminated:
                contaminated_deliveries += 1
            else:
                strict_errors.append(error)
        if info.get("relocated_block"):
            obstructive_moves += 1
            if not contaminated:
                strict_obstructive_moves += 1
        if info.get("illegal_drop"):
            illegal_drops += 1
        if done:
            break
    episode_loop_seconds = perf_counter() - execution_started

    delivered = sum(block.delivered for block in env.blocks)
    block_count = len(env.blocks)
    on_episode_end = getattr(source, "on_episode_end", None)
    if callable(on_episode_end):
        on_episode_end(
            success=delivered == block_count,
            truncated=(
                steps >= max_steps
                and not env.is_state_terminal(env.current_state)
            ),
        )
    invalid = audit.invalid_assignment_count
    strict_delivered = (
        delivered
        if invalid == 0
        else int(audit.delivered_before_first_invalid or 0)
    )
    proposal_count = sum(
        bool(decision.valid_candidates) for decision in audit.decisions
    )
    ga_result = getattr(source, "ga_result", None)
    ga_cost = getattr(ga_result, "best_cost", None)
    return {
        "method": method_name,
        "information_regime": TRACK_A_INFORMATION[method_name],
        "instance_id": env.current_episode_instance.instance_id,
        "return": total_reward,
        "strict_return_before_failure": strict_return,
        "steps": steps,
        "success": float(delivered == block_count),
        "completion_fraction": delivered / block_count,
        "strict_method_success": float(invalid == 0 and delivered == block_count),
        "strict_completion_fraction": strict_delivered / block_count,
        "delivery_errors": errors,
        "strict_delivery_errors": strict_errors,
        "delivery_count": delivered,
        **summarize_block_storage_flow(env.blocks, env.time_steps),
        "obstructive_moves": obstructive_moves,
        "strict_obstructive_moves": strict_obstructive_moves,
        "obstructive_moves_per_delivered_block": (
            obstructive_moves / max(1, delivered)
        ),
        "illegal_drops": illegal_drops,
        "assignment_decision_count": proposal_count,
        "valid_assignment_count": proposal_count - invalid,
        "invalid_assignment_count": invalid,
        "invalid_assignment_rate": invalid / max(1, proposal_count),
        "invalid_reasons": audit.invalid_reasons,
        "infeasible_epoch_count": audit.infeasible_epoch_count,
        "infeasible_block_counts": dict(audit.infeasible_block_counts),
        "retrieval_live_plan_failure_count": int(
            policy.retrieval_live_plan_failure_count
        ),
        "inbound_approach_defer_count": int(
            policy.inbound_approach_defer_count
        ),
        "exact_recovery_search_count": int(
            policy.exact_recovery_search_count
        ),
        "exact_recovery_fallback_count": int(
            policy.exact_recovery_fallback_count
        ),
        "exact_recovery_failure_count": int(
            policy.exact_recovery_failure_count
        ),
        "exact_recovery_explored_nodes": int(
            policy.exact_recovery_explored_nodes
        ),
        "fallback_count": audit.fallback_count,
        "fallback_contaminated": int(audit.contaminated),
        "first_invalid_step": audit.first_invalid_step,
        "contaminated_steps": contaminated_steps,
        "contaminated_deliveries": contaminated_deliveries,
        "empty_travel_steps": empty_travel_steps,
        "loaded_travel_steps": loaded_travel_steps,
        "wait_actions": wait_actions,
        "pickup_actions": pickup_actions,
        "putdown_actions": putdown_actions,
        "source_setup_seconds": source_setup_seconds,
        "assignment_planning_seconds": audit.planning_seconds,
        "episode_loop_seconds": episode_loop_seconds,
        "assignment_decisions": [
            decision.to_dict() for decision in audit.decisions
        ],
        "assignment_decisions_compact": [
            decision.to_compact_dict() for decision in audit.decisions
        ],
        "ga_predicted_infeasible_events": getattr(
            ga_cost, "infeasible_events", None
        ),
        "ga_predicted_obstructive_moves": getattr(
            ga_cost, "obstructive_moves", None
        ),
        "ga_predicted_route_steps": getattr(ga_cost, "route_steps", None),
    }


__all__ = [
    "AssignmentDecision",
    "AssignmentSource",
    "DYNAMIC_ASSIGNMENT_SOURCE_VERSION",
    "DynamicAssignmentSource",
    "NearestFreeAssignmentSource",
    "NEAREST_FREE_ASSIGNMENT_SOURCE_VERSION",
    "REGSelectorAssignmentSource",
    "StrictAssignmentAdapter",
    "TRACK_A_DYNAMIC",
    "TRACK_A_GA_OFFLINE",
    "TRACK_A_GA_ROLLING",
    "TRACK_A_GA_ROLLING_DURATION_AWARE",
    "TRACK_A_GA_ROLLING_OPERATIONAL",
    "TRACK_A_GA_ROLLING_COMPLETE",
    "TRACK_A_GA_ROLLING_CAPACITY_AWARE",
    "TRACK_A_GA_ROLLING_DURATION_AWARE_CAPACITY_AWARE",
    "TRACK_A_GA_ROLLING_OPERATIONAL_CAPACITY_AWARE",
    "TRACK_A_GA_ROLLING_COMPLETE_CAPACITY_AWARE",
    "TRACK_A_INFORMATION",
    "TRACK_A_KIM2020_A3C_SPATIAL",
    "TRACK_A_METHODS",
    "TRACK_A_NEAREST_FREE",
    "TRACK_A_REG_SELECTOR",
    "TRACK_A_REG_SELECTOR_V4",
    "TRACK_A_REG_SELECTOR_V5",
    "TrackAOnlinePolicy",
    "make_assignment_source",
    "neutral_relocation_selector",
    "run_track_a_episode",
    "shared_candidate_mask",
]
