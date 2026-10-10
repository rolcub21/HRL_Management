"""Matched Track-B scheduler with a physical standalone relocation escape.

The assignment source, duration-aware scheduling rule, and reserved-cell
admission contract are unchanged from E25-v3.  Only a deterministic, unlearned
macro is added when neither admission nor strict retrieval is executable.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from example.Options.ReconfigureOption import ReconfigureOption
from example.Options.RetrieveDeliverOption import RetrieveDeliverOption
from example.controller_options import build_controller_options, scheduler_episode_audit
from example.helper.timing_metrics import summarize_delivery_timing
from example.urgency_scheduler import (
    AssignmentSourceReservedCellDurationAwareAtomicScheduler,
    SchedulerInfeasibleError,
)
from options_agent import option_identifier
from PSLAP.dynamic_yard import YardSnapshot
from PSLAP.ga_policy import DEFAULT_ROLLING_GA_CONFIG
from PSLAP.kim2020_a3c_spatial import kim2020_deployment_digest
from PSLAP.viability import RecoveryActionKind, RecoveryState, legal_recovery_actions
from PSLAP.viability_filter import online_fixed_obstacles
from PSLAP.checkpoint_identity import selector_deployment_digest

from experiments.conditioned_vcg.E25_v3_matched_warm_start_95k import run as v3
from experiments.conditioned_vcg.E25_scale_baseline_transfer_95k import run as e25


class ReconfigurationScheduler(AssignmentSourceReservedCellDurationAwareAtomicScheduler):
    """Use a legal relocation only at an otherwise blocked decision epoch.

    The baseline's own least-slack retrieval plan nominates an obstructing
    block, when available.  Among legal relocations of that block, choose the
    shortest complete macro, with label and cell ties deterministic.  If its
    nominated move is not legal, choose the shortest legal relocation overall.
    Previously visited physical states are excluded to avoid two-move loops.
    No recovery certificates, learned scores, or test outcomes enter the rule.
    """

    METHOD = "duration_aware_assignment_source_with_reconfigure"
    VERSION = "duration_aware_reserved_cell_reconfigure_v1"
    POLICY = "blocked_epoch_plan_blocker_then_shortest_legal_relocation_v1"
    ACTION_INTERFACE = "reserved_cell_plus_standalone_reconfigure_v1"

    def __init__(self, env, *, margin_steps=0.0):
        super().__init__(env, margin_steps=margin_steps)
        self._seen_physical = set()
        self.reconfigure_options = []

    @staticmethod
    def _physical_key(recovery_state):
        return (
            recovery_state.agent_position,
            tuple((block.label, block.position) for block in recovery_state.blocks),
        )

    @staticmethod
    def _successor_key(recovery_state, action):
        return (
            action.destination,
            tuple(
                (block.label, action.destination if block.label == action.block_label else block.position)
                for block in recovery_state.blocks
            ),
        )

    def _physical_state(self):
        env = self.env
        agent = tuple(env.current_state)
        return RecoveryState.from_yard_snapshot(
            YardSnapshot.from_env(env),
            agent,
            fixed_obstacles=online_fixed_obstacles(env, reserve_queue_cells=True) - {agent},
            pickup_cells=(tuple(env.pickup_cell),),
            wait_cells=(tuple(env.waiting_cell),),
        )

    def select_action(self, state):
        if any(block.carrying for block in self.env.blocks):
            return super().select_action(state)
        inbound_feasible = bool(self.accept_option.initiation(state))
        _raw, executable = self._retrieval_candidates(state)
        if inbound_feasible or executable:
            return super().select_action(state)

        recovery_state = self._physical_state()
        self._seen_physical.add(self._physical_key(recovery_state))
        relocations = tuple(
            action for action in legal_recovery_actions(recovery_state)
            if action.kind is RecoveryActionKind.RELOCATION
            and self._successor_key(recovery_state, action) not in self._seen_physical
        )
        if not relocations:
            # Retain the original scheduler's defer/infeasible behavior when
            # the new action interface offers no fresh physical move.
            return super().select_action(state)

        nominated = None
        if _raw and _raw[0][1].relocations:
            nominated = _raw[0][1].relocations[0].block_label
        action = min(
            relocations,
            key=lambda item: (
                item.block_label != nominated,
                item.steps,
                item.block_label,
                item.destination,
            ),
        )
        option = ReconfigureOption.from_recovery_action(
            self.env, action, fixed_obstacles=recovery_state.fixed_obstacles
        )
        if not option.initiation(state):
            raise SchedulerInfeasibleError("selected physical relocation failed option initiation")
        identifier = option_identifier(option)
        self.decision_count += 1
        self.reason_counts["blocked_epoch_reconfigure"] += 1
        self.control_decisions[identifier] += 1
        self.decisions.append({
            "decision_index": self.decision_count - 1,
            "time_step": int(self.env.time_steps),
            "reason": "blocked_epoch_reconfigure",
            "option_id": identifier,
            "inbound_label": None if self._inbound_block() is None else self._inbound_block().label,
            "accept_feasible": False,
            "raw_retrieval_count": len(_raw),
            "executable_retrieval_count": 0,
            "target_label": action.block_label,
            "destination": list(action.destination),
            "primitive_steps_in_plan": action.steps,
            "physical_relocation_candidates": len(relocations),
        })
        self.reconfigure_options.append(option)
        return option

    def audit(self):
        result = super().audit()
        result["standalone_reconfiguration_attempts"] = len(self.reconfigure_options)
        result["standalone_reconfiguration_successes"] = sum(
            option.episode_success_count for option in self.reconfigure_options
        )
        result["standalone_reconfiguration_failures"] = sum(
            option.episode_failure_count for option in self.reconfigure_options
        )
        return result


def evaluate_one(args, seed, payload, *, instance, scenario, assignment_source,
                 policy_seed, deployment_mode):
    """The E25-v3 execution protocol with explicit standalone-event accounting."""
    b = e25.baseline
    if assignment_source == b.TRACK_A_REG_SELECTOR_V5:
        b.validate_selector_payload(payload, lam=args.lam, mu=args.mu,
                                    allow_regime_shift=args.allow_selector_regime_shift)
    elif assignment_source == b.TRACK_A_KIM2020_A3C_SPATIAL:
        b.validate_kim2020_payload(payload, lam=args.lam, mu=args.mu,
                                  allow_regime_shift=args.allow_selector_regime_shift)
    elif payload is not None:
        raise ValueError("deterministic assignment source received learned payload")
    b.seed_everything(seed)
    env = v3._warm_start_factory(
        scenario, arrival_rate=args.lam, proc_mean=args.mu,
        grid_rows=args.grid_rows, grid_cols=args.grid_cols,
        exit_width=args.exit_width, number_blocks=args.number_blocks,
    )
    instance.validate_for(env)
    setup_started = perf_counter()
    selector = b.make_track_b_assignment_option(
        env, assignment_source, selector_payload=payload,
        device=b.resolve_device(args.device), seed=seed,
        policy_seed=seed if policy_seed is None else policy_seed,
        deployment_mode=deployment_mode,
        allow_geometry_shift=bool(args.allow_assignment_geometry_shift),
        rolling_ga_config=replace(
            DEFAULT_ROLLING_GA_CONFIG,
            population_size=int(args.rolling_population),
            generations=int(args.rolling_generations),
            elite_count=min(2, int(args.rolling_population)-1),
            tournament_size=min(3, int(args.rolling_population)),
            seed=int(args.ga_seed_base)+seed,
        ),
        rolling_ga_egress_weight=int(args.rolling_ga_egress_weight),
    )
    build_controller_options(
        env, selector, controller_action_interface=b.RESERVED_CELL_ACTION_INTERFACE,
        max_defer_steps=args.max_defer_steps,
    )
    scheduler = ReconfigurationScheduler(env, margin_steps=float(args.lookahead_margin_steps))
    setup_seconds = perf_counter() - setup_started
    state = env.reset(instance=instance)
    b.seed_everything(seed)
    current_option = None
    errors = []
    total_return = 0.0
    physical_relocations = 0
    target_bound_relocations = 0
    standalone_relocations = 0
    illegal_drops = 0
    steps = 0
    done = False
    failure = None
    loop_started = perf_counter()
    while steps < args.max_steps and not done and failure is None:
        if current_option is None:
            try:
                current_option = scheduler.select_action(state)
            except SchedulerInfeasibleError as exc:
                failure = f"scheduler_infeasible:{exc}"
                break
        try:
            action = current_option.policy(state)
            next_state, reward, done, info = env.step(action)
            steps += 1
            selector.on_step(reward, info)
            total_return += float(reward)
            if "delivery_error_time" in info:
                errors.append(float(info["delivery_error_time"]))
            if info.get("relocated_block"):
                physical_relocations += 1
                if isinstance(current_option, ReconfigureOption):
                    standalone_relocations += 1
                elif isinstance(current_option, RetrieveDeliverOption):
                    target_bound_relocations += 1
                else:
                    raise RuntimeError("relocation outside a named macro")
            illegal_drops += int(bool(info.get("illegal_drop")))
            terminated = current_option.termination(next_state)
        except RuntimeError as exc:
            failure = f"macro_runtime:{exc}"
            break
        state = next_state
        if terminated:
            outcome = b._failed_macro_outcome(current_option)
            if outcome is not None:
                failure = f"macro_failure:{type(current_option).__name__}:{outcome.get('reason','unknown')}"
            current_option = None
    loop_seconds = perf_counter() - loop_started
    truncated = steps >= args.max_steps and not done
    selector.on_episode_end(success=done, truncated=truncated)
    selector_audit = selector.audit()
    option_audit = scheduler_episode_audit(env)
    scheduler_audit = scheduler.audit()
    if done and failure is None and target_bound_relocations != int(option_audit["retrieve_relocations"]):
        raise RuntimeError("target-bound retrieval relocation accounting mismatch")
    if done and failure is None and standalone_relocations != scheduler_audit["standalone_reconfiguration_successes"]:
        raise RuntimeError("standalone reconfiguration accounting mismatch")
    bound_ids = option_audit["bound_proposal_ids"]
    committed_ids = option_audit["committed_proposal_ids"]
    selector_committed_ids = [
        decision.get("proposal_id") for decision in selector_audit["decisions"]
        if decision.get("valid") and decision.get("commitment_contract")
        == "decision_epoch_proposal_bound_once_v2"
    ]
    reservation_integrity = bool(
        option_audit["reservation_invalidation_count"] == 0
        and option_audit["reservation_bound_count"] == option_audit["inbound_successes"]
        and option_audit["reservation_commit_count"] == option_audit["inbound_successes"]
        and option_audit["reservation_execution_match_count"] == option_audit["inbound_successes"]
        and len(bound_ids) == len(set(bound_ids))
        and bound_ids == committed_ids == selector_committed_ids
    )
    strict = bool(
        done and failure is None and selector_audit["invalid_assignment_count"] == 0
        and selector_audit.get("fallback_count", 0) == 0
        and option_audit["inbound_failures"] == 0
        and option_audit["retrieve_failures"] == 0
        and scheduler_audit["standalone_reconfiguration_failures"] == 0
        and illegal_drops == 0 and reservation_integrity
    )
    return {
        "eval_seed": seed, "instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id, "return": total_return,
        "success": float(done), "strict_method_success": float(strict),
        "method_failure_reason": failure, "truncated": float(truncated),
        "steps": steps, "delivery_count": len(errors),
        "physical_storage_relocations": physical_relocations,
        "target_bound_obstruction_clearances": target_bound_relocations,
        "standalone_reconfigurations": standalone_relocations,
        "illegal_drops": illegal_drops,
        **summarize_delivery_timing(errors, args.target_window),
        "geometry": b.geometry_metadata(env, requested_exit_width=args.exit_width),
        "assignment_source": assignment_source,
        "assignment_source_version": selector_audit["assignment_source_version"],
        "assignment_policy_realization": (
            deployment_mode if assignment_source == b.TRACK_A_KIM2020_A3C_SPATIAL
            else "deterministic_map"
        ),
        "selector_deployment_digest": (
            kim2020_deployment_digest(payload)
            if assignment_source == b.TRACK_A_KIM2020_A3C_SPATIAL
            else selector_deployment_digest(payload) if payload is not None else None
        ),
        "selector_audit": selector_audit, "scheduler_audit": option_audit,
        "urgency_scheduler_audit": scheduler_audit,
        "reservation_integrity": reservation_integrity,
        "initial_state_audit": env.warm_start_audit,
        "setup_seconds": setup_seconds, "decision_seconds": loop_seconds,
    }
