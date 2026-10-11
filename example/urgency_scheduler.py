"""Deterministic urgency-first scheduling over the atomic Track-B macros."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import math
from statistics import mean

from example.Options.AcceptStoreOption import (
    AcceptStoreOption,
    ReservedAcceptStoreOption,
)
from example.Options.RetrieveDeliverOption import StrictRetrieveDeliverOption
from example.Options.StrategicDeferOption import StrategicDeferOption
from example.helper.timing_metrics import summarize_block_storage_flow
from options_agent import option_identifier
from PSLAP.dynamic_yard import BlockView
from PSLAP.neutral_protocol import NEUTRAL_RELOCATION_SELECTOR_VERSION
from PSLAP.retrieval_context import retrieval_planning_context
from PSLAP.retrieval_dispatch import plan_retrieval
from PSLAP.retrieval_executor import CompleteLiveRetrievalExecutor


URGENCY_FIRST_METHOD = "reg_v5_urgency_first_atomic"
URGENCY_FIRST_ARCHITECTURE = "deterministic_urgency_first_atomic_v1"
URGENCY_FIRST_POLICY = "deterministic_eta_slack_due_first_v1"
URGENCY_FIRST_ACTION_INTERFACE = "interleaved_atomic_scheduler_v3"
DURATION_AWARE_METHOD = "reg_v5_duration_aware_atomic"
DURATION_AWARE_ARCHITECTURE = "deterministic_duration_aware_atomic_v2"
DURATION_AWARE_POLICY = "deterministic_reg_preview_duration_lookahead_v2"
ASSIGNMENT_ABLATION_METHOD = "duration_aware_assignment_source_ablation"
ASSIGNMENT_ABLATION_POLICY = (
    "deterministic_assignment_preview_duration_lookahead_v2"
)
RESERVED_CELL_ACTION_INTERFACE = (
    "interleaved_atomic_scheduler_reserved_cell_v4"
)
RESERVED_CELL_METHOD = "reg_v5_reserved_cell_hierarchical"
RESERVED_CELL_ARCHITECTURE = (
    "deterministic_duration_aware_reserved_cell_hierarchy_v3"
)
RESERVED_CELL_POLICY = "deterministic_bound_cell_duration_lookahead_v3"
RESERVED_ASSIGNMENT_ABLATION_METHOD = (
    "reserved_cell_assignment_source_ablation"
)
RESERVED_ASSIGNMENT_ABLATION_POLICY = (
    "deterministic_bound_assignment_duration_lookahead_v3"
)
RESERVED_DUE_ONLY_METHOD = "reserved_cell_urgency_first_atomic"
RESERVED_DUE_ONLY_ARCHITECTURE = (
    "deterministic_urgency_first_reserved_cell_v2"
)
RESERVED_DUE_ONLY_POLICY = "deterministic_due_first_bound_cell_v2"


class SchedulerInfeasibleError(RuntimeError):
    """Raised when the strict method has no executable macro."""


class UrgencyFirstAtomicScheduler:
    """Prioritize executable due retrievals, then accept inbound inventory.

    This policy is a no-training repaired candidate for Track B.  It uses the
    same frozen assignment model and atomic macro structure as the learned
    controller, but deliberately upgrades retrieval admission from v1 to the
    complete current-live multi-leg v3 contract.  Its macro selection is a transparent
    ETA/slack rule.  It must therefore not be described as a one-factor
    scheduler-only ablation against a v1 learned checkpoint.
    """

    METHOD = URGENCY_FIRST_METHOD
    VERSION = URGENCY_FIRST_ARCHITECTURE
    POLICY = URGENCY_FIRST_POLICY
    ACTION_INTERFACE = URGENCY_FIRST_ACTION_INTERFACE
    DECISION_EPOCH_CONTRACT = (
        "due_retrieval_then_inbound_then_capacity_then_event_v1"
    )

    def __init__(self, env):
        self.env = env
        managers = [option for option in env.options if not option.is_primitive]
        accepts = [
            option for option in managers if isinstance(option, AcceptStoreOption)
        ]
        defers = [
            option for option in managers if isinstance(option, StrategicDeferOption)
        ]
        retrievals = [
            option
            for option in managers
            if isinstance(option, StrictRetrieveDeliverOption)
        ]
        if len(accepts) != 1 or len(defers) != 1:
            raise ValueError(
                "Urgency scheduler requires exactly one AcceptStore and Defer"
            )
        if len(retrievals) != len(env.blocks):
            raise ValueError(
                "Urgency scheduler requires one strict retrieval per manifest block"
            )
        self.accept_option = accepts[0]
        self.defer_option = defers[0]
        self.retrieval_options = tuple(
            sorted(retrievals, key=lambda option: option.block_index)
        )
        if tuple(option.block_index for option in self.retrieval_options) != tuple(
            range(len(env.blocks))
        ):
            raise ValueError("Strict retrieval options do not match manifest slots")
        relocation_versions = {
            getattr(option.relocation_selector, "VERSION", None)
            or getattr(option.relocation_selector, "__name__", None)
            for option in self.retrieval_options
        }
        if len(relocation_versions) != 1:
            raise ValueError("Retrieval options must share one relocation rule")
        self.relocation_selector = self.retrieval_options[0].relocation_selector
        self.reset_audit()

    def reset_audit(self):
        self.decision_count = 0
        self.reason_counts = Counter()
        self.control_decisions = Counter()
        self.decisions = []
        self.accepts_before_first_retrieval = 0
        self.accepts_while_due = 0
        self.max_accept_streak = 0
        self.min_slack_at_accept = None
        self._accept_streak = 0
        self._seen_retrieval = False

    def _inbound_block(self):
        return next(
            (
                block
                for block in self.env.blocks
                if block.position == self.env.pickup_cell
                and not block.carrying
                and not block.stored
                and not block.delivered
            ),
            None,
        )

    @staticmethod
    def _score(option, plan):
        return (
            float(plan.slack),
            len(plan.relocations),
            int(plan.estimated_steps),
            int(option.block_index),
        )

    def _retrieval_candidates(self, state):
        context = retrieval_planning_context(
            self.env,
            relocation_selector=self.relocation_selector,
        )
        raw_plans = []
        executable = []
        for option in self.retrieval_options:
            plan = context.plan(option.target_label)
            if plan is None:
                continue
            raw_plans.append((option, plan))
            if option.initiation(state):
                executable.append((option, plan))
        raw_plans.sort(key=lambda item: self._score(*item))
        executable.sort(key=lambda item: self._score(*item))
        return raw_plans, executable

    def _choose_feasible_inbound(
        self,
        state,
        inbound,
        raw_plans,
        executable,
    ):
        """Return choice, selected retrieval plan, reason, and audit details."""

        return self.accept_option, None, "accept_inbound", {}

    def select_action(self, state):
        if any(block.carrying for block in self.env.blocks):
            raise SchedulerInfeasibleError(
                "Scheduler decision requested while inventory is carried"
            )
        inbound = self._inbound_block()
        accept_feasible = bool(self.accept_option.initiation(state))
        raw_plans, executable = self._retrieval_candidates(state)
        raw_due = [item for item in raw_plans if item[1].slack <= 0]
        due = [item for item in executable if item[1].slack <= 0]

        selected_plan = None
        decision_details = {}
        if due:
            choice, selected_plan = due[0]
            reason = "due_retrieval"
        elif inbound is not None and accept_feasible:
            (
                choice,
                selected_plan,
                reason,
                decision_details,
            ) = self._choose_feasible_inbound(
                state,
                inbound,
                raw_plans,
                executable,
            )
        elif inbound is not None:
            if not executable:
                self.reason_counts["no_action_infeasible"] += 1
                raise SchedulerInfeasibleError(
                    "Inbound placement is infeasible and no strict retrieval "
                    "can release capacity"
                )
            choice, selected_plan = executable[0]
            reason = "capacity_release_retrieval"
        elif raw_due:
            self.reason_counts["no_action_infeasible"] += 1
            self.reason_counts["blocked_due_infeasible"] += 1
            raise SchedulerInfeasibleError(
                "A due retrieval exists but no due retrieval has an "
                "executable live first leg"
            )
        elif self.defer_option.initiation(state):
            choice = self.defer_option
            reason = "defer_to_event"
        else:
            self.reason_counts["no_action_infeasible"] += 1
            raise SchedulerInfeasibleError(
                "No executable urgency-scheduler macro is available"
            )

        minimum_raw_slack = (
            float(raw_plans[0][1].slack) if raw_plans else None
        )
        if choice is self.accept_option:
            self._accept_streak += 1
            self.max_accept_streak = max(
                self.max_accept_streak, self._accept_streak
            )
            if not self._seen_retrieval:
                self.accepts_before_first_retrieval += 1
            if raw_due:
                self.accepts_while_due += 1
            if minimum_raw_slack is not None:
                self.min_slack_at_accept = (
                    minimum_raw_slack
                    if self.min_slack_at_accept is None
                    else min(self.min_slack_at_accept, minimum_raw_slack)
                )
        elif selected_plan is not None:
            self._accept_streak = 0
            self._seen_retrieval = True
        else:
            self._accept_streak = 0

        identifier = option_identifier(choice)
        decision = {
            "decision_index": self.decision_count,
            "time_step": int(self.env.time_steps),
            "reason": reason,
            "option_id": identifier,
            "inbound_label": inbound.label if inbound is not None else None,
            "accept_feasible": accept_feasible,
            "raw_retrieval_count": len(raw_plans),
            "executable_retrieval_count": len(executable),
            "raw_due_count": len(raw_due),
            "executable_due_count": len(due),
            "minimum_raw_slack": minimum_raw_slack,
            "target_label": (
                choice.target_label if selected_plan is not None else None
            ),
            "target_slack": (
                float(selected_plan.slack)
                if selected_plan is not None
                else None
            ),
            "target_eta": (
                int(selected_plan.estimated_steps)
                if selected_plan is not None
                else None
            ),
            "target_relocations": (
                len(selected_plan.relocations)
                if selected_plan is not None
                else None
            ),
        }
        decision.update(decision_details)
        self.decision_count += 1
        self.reason_counts[reason] += 1
        self.control_decisions[identifier] += 1
        self.decisions.append(decision)
        return choice

    def audit(self):
        return {
            "method": self.METHOD,
            "scheduler_architecture": self.VERSION,
            "policy_realization": self.POLICY,
            "controller_action_interface": self.ACTION_INTERFACE,
            "decision_epoch_contract": self.DECISION_EPOCH_CONTRACT,
            "retrieve_deliver_option_version": (
                StrictRetrieveDeliverOption.VERSION
            ),
            "retrieval_executor_version": (
                CompleteLiveRetrievalExecutor.VERSION
            ),
            "retrieval_start_contract": (
                "canonical_plan_plus_complete_current_live_multileg_v3"
            ),
            "ranking_eta_contract": (
                "stored_yard_canonical_complete_plan_v1"
            ),
            "relocation_selector_version": (
                NEUTRAL_RELOCATION_SELECTOR_VERSION
            ),
            "accept_store_option_version": type(self.accept_option).VERSION,
            "assignment_commitment_contract": getattr(
                self.accept_option,
                "COMMITMENT_CONTRACT",
                "post_pickup_recompute_v1",
            ),
            "reservation_contract": getattr(
                self.accept_option, "RESERVATION_CONTRACT", None
            ),
            "reservation_bound_count": int(
                getattr(
                    self.accept_option,
                    "episode_reservation_bound_count",
                    0,
                )
            ),
            "reservation_commit_count": int(
                getattr(
                    self.accept_option,
                    "episode_reservation_commit_count",
                    0,
                )
            ),
            "reservation_invalidation_count": int(
                getattr(
                    self.accept_option,
                    "episode_reservation_invalidation_count",
                    0,
                )
            ),
            "reservation_execution_match_count": int(
                getattr(
                    self.accept_option,
                    "episode_reservation_execution_match_count",
                    0,
                )
            ),
            "bound_proposal_ids": list(
                getattr(
                    self.accept_option, "episode_bound_proposal_ids", ()
                )
            ),
            "committed_proposal_ids": list(
                getattr(
                    self.accept_option, "episode_committed_proposal_ids", ()
                )
            ),
            "defer_option_version": StrategicDeferOption.VERSION,
            "retrieval_termination_contract": (
                "complete_delivery_or_safe_explicit_failure_v1"
            ),
            "decision_count": self.decision_count,
            "reason_counts": dict(self.reason_counts),
            "control_decisions": dict(self.control_decisions),
            "accepts_before_first_retrieval": self.accepts_before_first_retrieval,
            "accepts_while_due": self.accepts_while_due,
            "max_accept_streak": self.max_accept_streak,
            "min_slack_at_accept": self.min_slack_at_accept,
            "decisions": list(self.decisions),
        }


class DurationAwareAtomicScheduler(UrgencyFirstAtomicScheduler):
    """Compare retrieving now with one projected post-placement retrieval.

    The scheduler previews frozen REG-v5 at the current observable epoch and
    uses that cell to construct a nominal AcceptStore duration and hypothetical
    post-placement yard.  AcceptStore v1 still makes its real REG decision
    after pickup, so every preview and duration prediction is audited rather
    than represented as exact.
    """

    METHOD = DURATION_AWARE_METHOD
    VERSION = DURATION_AWARE_ARCHITECTURE
    POLICY = DURATION_AWARE_POLICY
    DECISION_EPOCH_CONTRACT = (
        "due_then_absolute_timing_lookahead_then_capacity_then_event_v2"
    )
    PREVIEW_WHEN_NO_RETRIEVAL = False
    ACCEPT_DURATION_ESTIMATE_CONTRACT = (
        "nominal_current_epoch_assignment_preview_path_with_handling_v2"
    )

    def __init__(self, env, *, margin_steps=0.0):
        margin_steps = float(margin_steps)
        if not math.isfinite(margin_steps) or margin_steps < 0.0:
            raise ValueError("lookahead margin must be finite and nonnegative")
        self.margin_steps = margin_steps
        super().__init__(env)

    def reset_audit(self):
        super().reset_audit()
        self.preview_call_count = 0
        self.preview_seconds = 0.0
        self.preview_failure_count = 0
        self.lookahead_tie_count = 0
        self._inbound_retrieval_streak = 0
        self.max_inbound_retrieval_streak = 0

    def _projected_plan(self, inbound, estimate, target_label):
        current = retrieval_planning_context(
            self.env,
            relocation_selector=self.relocation_selector,
        ).yard
        aged = tuple(
            replace(
                block,
                remaining_time=(
                    float(block.remaining_time) - estimate.total_steps
                ),
            )
            for block in current.blocks
        )
        projected = replace(
            current,
            blocks=aged
            + (
                BlockView(
                    label=inbound.label,
                    position=estimate.preview_storage_cell,
                    remaining_time=float(inbound.storage_steps_needed),
                ),
            ),
        )
        return plan_retrieval(
            projected,
            estimate.preview_storage_cell,
            target_label,
            relocation_selector=self.relocation_selector,
        )

    def _choose_feasible_inbound(
        self,
        state,
        inbound,
        raw_plans,
        executable,
    ):
        if not executable and not self.PREVIEW_WHEN_NO_RETRIEVAL:
            return (
                self.accept_option,
                None,
                "lookahead_accept_no_retrieval",
                {"lookahead_active": False},
            )

        self.preview_call_count += 1
        try:
            estimate = self.accept_option.estimate_duration()
        except (RuntimeError, ValueError) as exc:
            self.preview_failure_count += 1
            self.reason_counts["no_action_infeasible"] += 1
            self.reason_counts["lookahead_preview_infeasible"] += 1
            raise SchedulerInfeasibleError(
                "AcceptStore is feasible but its strict duration preview "
                f"raised {type(exc).__name__}: {exc}"
            ) from exc
        if estimate is not None:
            self.preview_seconds += float(estimate.preview_seconds)
        if estimate is None:
            self.preview_failure_count += 1
            self.reason_counts["no_action_infeasible"] += 1
            self.reason_counts["lookahead_preview_infeasible"] += 1
            raise SchedulerInfeasibleError(
                "AcceptStore is feasible but its strict duration preview failed"
            )

        if not executable:
            self._bind_accepted_estimate(estimate)
            return (
                self.accept_option,
                None,
                "lookahead_accept_no_retrieval",
                {
                    "lookahead_active": False,
                    "preview_contract": estimate.contract,
                    "preview_storage_cell": estimate.preview_storage_cell,
                    "preview_candidate_count": estimate.preview_candidate_count,
                    "preview_candidate_mask_id": getattr(
                        estimate, "preview_candidate_mask_id", None
                    ),
                    "preview_proposal_id": getattr(
                        getattr(estimate, "preview_proposal", None),
                        "proposal_id",
                        None,
                    ),
                    "estimated_accept_pickup_steps": estimate.pickup_steps,
                    "estimated_accept_storage_steps": estimate.storage_steps,
                    "estimated_accept_total_steps": estimate.total_steps,
                    "preview_seconds": estimate.preview_seconds,
                },
            )

        option, current_plan = executable[0]
        projected_plan = self._projected_plan(
            inbound,
            estimate,
            option.target_label,
        )
        error_now = -float(current_plan.slack)
        cost_now = abs(error_now)
        error_after = (
            None if projected_plan is None else -float(projected_plan.slack)
        )
        cost_after = (
            math.inf if error_after is None else abs(error_after)
        )
        retrieve_now = cost_now <= cost_after + self.margin_steps
        tie = bool(
            projected_plan is not None
            and math.isclose(
                cost_now,
                cost_after + self.margin_steps,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        self.lookahead_tie_count += int(tie)
        details = {
            "lookahead_active": True,
            "lookahead_contract": (
                "current_head_now_vs_projected_post_accept_absolute_error_v1"
            ),
            "lookahead_margin_steps": self.margin_steps,
            "preview_contract": estimate.contract,
            "preview_storage_cell": estimate.preview_storage_cell,
            "preview_candidate_count": estimate.preview_candidate_count,
            "preview_candidate_mask_id": getattr(
                estimate, "preview_candidate_mask_id", None
            ),
            "preview_proposal_id": getattr(
                getattr(estimate, "preview_proposal", None),
                "proposal_id",
                None,
            ),
            "preview_seconds": estimate.preview_seconds,
            "estimated_accept_pickup_steps": estimate.pickup_steps,
            "estimated_accept_storage_steps": estimate.storage_steps,
            "estimated_accept_total_steps": estimate.total_steps,
            "lookahead_head_label": option.target_label,
            "lookahead_error_if_retrieve_now": error_now,
            "lookahead_cost_if_retrieve_now": cost_now,
            "lookahead_error_if_accept_first": error_after,
            "lookahead_cost_if_accept_first": (
                None if projected_plan is None else cost_after
            ),
            "projected_retrieval_eta": (
                None
                if projected_plan is None
                else int(projected_plan.estimated_steps)
            ),
            "projected_retrieval_slack": (
                None
                if projected_plan is None
                else float(projected_plan.slack)
            ),
            "projected_retrieval_relocations": (
                None
                if projected_plan is None
                else len(projected_plan.relocations)
            ),
            "lookahead_tie": tie,
        }
        if retrieve_now:
            return option, current_plan, "lookahead_retrieval", details
        self._bind_accepted_estimate(estimate)
        return self.accept_option, None, "lookahead_accept", details

    def _bind_accepted_estimate(self, estimate):
        """Legacy v1 deliberately leaves the preview unbound."""

        return None

    def select_action(self, state):
        choice = super().select_action(state)
        decision = self.decisions[-1]
        if (
            decision["inbound_label"] is not None
            and decision["target_label"] is not None
        ):
            self._inbound_retrieval_streak += 1
            self.max_inbound_retrieval_streak = max(
                self.max_inbound_retrieval_streak,
                self._inbound_retrieval_streak,
            )
        elif choice is self.accept_option:
            self._inbound_retrieval_streak = 0
        elif decision["inbound_label"] is None:
            self._inbound_retrieval_streak = 0
        return choice

    @staticmethod
    def _p90(values):
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]

    def audit(self):
        result = super().audit()
        selector = self.accept_option.selector
        selector_decisions = {
            decision["block_label"]: decision
            for decision in selector.decisions
            if decision.get("valid")
        }
        inbound_outcomes = {
            outcome["block_label"]: outcome
            for outcome in self.accept_option.episode_outcomes
            if outcome.get("success")
        }
        preview_accepts = [
            decision
            for decision in self.decisions
            if decision["reason"]
            in ("lookahead_accept", "lookahead_accept_no_retrieval")
            and decision.get("preview_storage_cell") is not None
        ]
        cell_matches = []
        duration_errors = []
        for decision in preview_accepts:
            label = decision["inbound_label"]
            actual_assignment = selector_decisions.get(label)
            if actual_assignment is not None:
                cell_matches.append(
                    tuple(decision["preview_storage_cell"])
                    == tuple(actual_assignment["chosen_cell"])
                )
            outcome = inbound_outcomes.get(label)
            if outcome is not None:
                duration_errors.append(
                    int(outcome["actual_steps"])
                    - int(decision["estimated_accept_total_steps"])
                )
        storage_flow = summarize_block_storage_flow(
            self.env.blocks, self.env.time_steps
        )
        inbound_waits = storage_flow["completed_storage_flow_times"]
        result.update(
            {
                **storage_flow,
                "assignment_source": getattr(
                    selector, "assignment_source", "unknown"
                ),
                "assignment_source_family": getattr(
                    selector, "assignment_source_family", "unknown"
                ),
                "assignment_source_version": getattr(
                    selector, "assignment_source_version", "unknown"
                ),
                "assignment_preview_contract": getattr(
                    selector, "preview_contract", "unknown"
                ),
                "accept_duration_estimate_contract": (
                    self.ACCEPT_DURATION_ESTIMATE_CONTRACT
                ),
                "accept_assignment_contract": getattr(
                    selector, "deployment_assignment_contract", "unknown"
                ),
                "lookahead_cost_contract": (
                    "absolute_delivery_error_head_job_one_step_projection_v1"
                ),
                "lookahead_margin_steps": self.margin_steps,
                "preview_call_count": self.preview_call_count,
                "preview_seconds": self.preview_seconds,
                "preview_failure_count": self.preview_failure_count,
                "preview_accept_count": len(preview_accepts),
                "preview_cell_match_count": sum(cell_matches),
                "preview_cell_comparison_count": len(cell_matches),
                "preview_cell_match_rate": (
                    mean(cell_matches) if cell_matches else None
                ),
                "accept_duration_error_steps": duration_errors,
                "accept_duration_error_contract": (
                    "actual_steps_minus_estimated_steps"
                ),
                "accept_duration_actual_minus_estimated_steps": (
                    duration_errors
                ),
                "mean_accept_duration_error_steps": (
                    mean(duration_errors) if duration_errors else None
                ),
                "mean_accept_duration_actual_minus_estimated_steps": (
                    mean(duration_errors) if duration_errors else None
                ),
                "lookahead_tie_count": self.lookahead_tie_count,
                "lookahead_tie_contract": (
                    "cost_now_equals_cost_after_plus_margin"
                ),
                "lookahead_margin_boundary_tie_count": (
                    self.lookahead_tie_count
                ),
                "preview_timing_contract": (
                    "assignment_source_preview_only_excludes_path_and_"
                    "retrieval_projection"
                ),
                "max_consecutive_retrievals_while_inbound": (
                    self.max_inbound_retrieval_streak
                ),
                "arrival_to_storage_wait_alias_contract": (
                    "deprecated_alias_of_storage_flow_metrics_v1;"
                    "values_are_flow_time_not_queue_wait"
                ),
                "arrival_to_storage_wait_steps": inbound_waits,
                "mean_arrival_to_storage_wait_steps": storage_flow[
                    "mean_storage_flow_time"
                ],
                "p90_arrival_to_storage_wait_steps": storage_flow[
                    "p90_storage_flow_time"
                ],
                "max_arrival_to_storage_wait_steps": storage_flow[
                    "max_storage_flow_time"
                ],
            }
        )
        return result


class AssignmentSourceDurationAwareAtomicScheduler(
    DurationAwareAtomicScheduler
):
    """Source-neutral identity for the fixed assignment-source ablation."""

    METHOD = ASSIGNMENT_ABLATION_METHOD
    VERSION = DURATION_AWARE_ARCHITECTURE
    POLICY = ASSIGNMENT_ABLATION_POLICY


class ReservedCellUrgencyFirstAtomicScheduler(UrgencyFirstAtomicScheduler):
    """Due-first scheduler whose accepted inbound macro has a bound cell."""

    METHOD = RESERVED_DUE_ONLY_METHOD
    VERSION = RESERVED_DUE_ONLY_ARCHITECTURE
    POLICY = RESERVED_DUE_ONLY_POLICY
    ACTION_INTERFACE = RESERVED_CELL_ACTION_INTERFACE
    DECISION_EPOCH_CONTRACT = (
        "due_retrieval_then_bound_inbound_then_capacity_then_event_v2"
    )

    def _choose_feasible_inbound(
        self,
        state,
        inbound,
        raw_plans,
        executable,
    ):
        try:
            estimate = self.accept_option.estimate_duration()
            if estimate is None:
                raise ValueError("no valid reserved assignment proposal")
            preview = self.accept_option.bind_estimate(estimate)
        except (RuntimeError, ValueError) as exc:
            self.reason_counts["no_action_infeasible"] += 1
            self.reason_counts["reserved_assignment_infeasible"] += 1
            raise SchedulerInfeasibleError(
                f"Unable to bind inbound assignment: {exc}"
            ) from exc
        return (
            self.accept_option,
            None,
            "accept_bound_inbound",
            {
                "preview_contract": estimate.contract,
                "preview_storage_cell": estimate.preview_storage_cell,
                "preview_candidate_count": estimate.preview_candidate_count,
                "preview_candidate_mask_id": estimate.preview_candidate_mask_id,
                "preview_proposal_id": preview.proposal_id,
                "estimated_accept_pickup_steps": estimate.pickup_steps,
                "estimated_accept_storage_steps": estimate.storage_steps,
                "estimated_accept_total_steps": estimate.total_steps,
                "preview_seconds": estimate.preview_seconds,
            },
        )


class ReservedCellDurationAwareAtomicScheduler(DurationAwareAtomicScheduler):
    """Hierarchical manager that executes its exact proposed spatial action."""

    METHOD = RESERVED_CELL_METHOD
    VERSION = RESERVED_CELL_ARCHITECTURE
    POLICY = RESERVED_CELL_POLICY
    ACTION_INTERFACE = RESERVED_CELL_ACTION_INTERFACE
    DECISION_EPOCH_CONTRACT = (
        "due_then_exact_bound_cell_lookahead_then_capacity_then_event_v3"
    )
    PREVIEW_WHEN_NO_RETRIEVAL = True
    ACCEPT_DURATION_ESTIMATE_CONTRACT = (
        "exact_bound_cell_current_epoch_path_with_handling_v3"
    )

    def _bind_accepted_estimate(self, estimate):
        try:
            self.accept_option.bind_estimate(estimate)
        except (RuntimeError, ValueError) as exc:
            self.preview_failure_count += 1
            self.reason_counts["no_action_infeasible"] += 1
            self.reason_counts["reserved_assignment_infeasible"] += 1
            raise SchedulerInfeasibleError(
                f"Unable to bind accepted assignment proposal: {exc}"
            ) from exc


class AssignmentSourceReservedCellDurationAwareAtomicScheduler(
    ReservedCellDurationAwareAtomicScheduler
):
    """Source-neutral identity for the reserved-cell source ablation."""

    METHOD = RESERVED_ASSIGNMENT_ABLATION_METHOD
    POLICY = RESERVED_ASSIGNMENT_ABLATION_POLICY


__all__ = [
    "ASSIGNMENT_ABLATION_METHOD",
    "ASSIGNMENT_ABLATION_POLICY",
    "AssignmentSourceReservedCellDurationAwareAtomicScheduler",
    "AssignmentSourceDurationAwareAtomicScheduler",
    "DURATION_AWARE_ARCHITECTURE",
    "DURATION_AWARE_METHOD",
    "DURATION_AWARE_POLICY",
    "DurationAwareAtomicScheduler",
    "SchedulerInfeasibleError",
    "RESERVED_ASSIGNMENT_ABLATION_METHOD",
    "RESERVED_ASSIGNMENT_ABLATION_POLICY",
    "RESERVED_CELL_ACTION_INTERFACE",
    "RESERVED_CELL_ARCHITECTURE",
    "RESERVED_CELL_METHOD",
    "RESERVED_CELL_POLICY",
    "RESERVED_DUE_ONLY_ARCHITECTURE",
    "RESERVED_DUE_ONLY_METHOD",
    "RESERVED_DUE_ONLY_POLICY",
    "ReservedCellDurationAwareAtomicScheduler",
    "ReservedCellUrgencyFirstAtomicScheduler",
    "URGENCY_FIRST_ACTION_INTERFACE",
    "URGENCY_FIRST_ARCHITECTURE",
    "URGENCY_FIRST_METHOD",
    "URGENCY_FIRST_POLICY",
    "UrgencyFirstAtomicScheduler",
]
