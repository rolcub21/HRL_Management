"""Frozen Dynamic PSLAP rule projected onto VCG's certified frontier.

This is a versioned *certified adaptation*, not an exact replay of the
uncertified Track-B option scheduler.  It retains Dynamic PSLAP's placement
ordering and duration-aware due/accept/retrieval order wherever those choices
remain in the SAFE frontier, and records every required projection.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import torch

from PSLAP.dynamic_yard import (
    BlockView, is_storage_location_feasible, select_storage_location,
    shortest_clear_path,
)
from PSLAP.neutral_protocol import neutral_relocation_selector
from PSLAP.retrieval_dispatch import plan_retrieval
from PSLAP.viability_candidates import ViabilityActionType, ViabilityMode
from viability_graph_hierarchy import ID_TO_MODE, ViabilityGraphDecision
from viability_graph_preference_conditioned import select_hierarchical_index


class CertifiedDynamicAgent:
    """Score one deterministic rule choice, then use the frozen selector shell."""

    POLICY = "certified_dynamic_placement_duration_aware_dispatch_v1"
    LOOKAHEAD_MARGIN_STEPS = 2.0
    ONE_HOT_MERIT = 1000.0

    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.config = base_agent.config
        self.choice_audit = []
        self.first_frontier_sha256 = None

    def reset_episode_state(self):
        self.choice_audit.clear()
        self.first_frontier_sha256 = None
        return self.base_agent.reset_episode_state()

    on_episode_reset = reset_episode_state

    @staticmethod
    def _arrival_view(snapshot, accepts):
        if not accepts:
            return None
        item = accepts[0][1]
        block = item.successor_state.block(item.target_label)
        if block is None or item.source is None:
            raise ValueError("SAFE Accept lacks its bound inbound block")
        return BlockView(item.target_label, item.source, block.remaining_time)

    @staticmethod
    def _accept_steps(snapshot, candidate):
        approach = shortest_clear_path(snapshot.yard, snapshot.agent_position, candidate.source)
        transport = shortest_clear_path(snapshot.yard, candidate.source, candidate.destination)
        if approach is None or transport is None:
            return None
        return (len(approach) - 1) + 1 + (len(transport) - 1) + 1

    @staticmethod
    def _placement_key(snapshot, inbound, candidate):
        cell = candidate.destination
        if not is_storage_location_feasible(snapshot.yard, inbound, cell, source=inbound.position):
            return None
        route = shortest_clear_path(snapshot.yard, inbound.position, cell)
        if route is None:
            return None
        return (
            min(abs(cell[0]-exit_cell[0])+abs(cell[1]-exit_cell[1])
                for exit_cell in snapshot.yard.exits),
            len(route)-1,
            cell,
        )

    def _choice(self, snapshot, prepared):
        items = [(i, snapshot.candidates[source_index])
                 for i, source_index in enumerate(prepared.source_indices)]
        typed = {kind: [(i, c) for i, c in items if c.action_type is kind]
                 for kind in ViabilityActionType}
        accepts = typed[ViabilityActionType.ACCEPT]
        delivers = typed[ViabilityActionType.DELIVER]
        reconfigs = typed[ViabilityActionType.RECONFIGURE]
        defers = typed[ViabilityActionType.DEFER]
        raw_plans = []
        for block in snapshot.yard.blocks:
            plan = plan_retrieval(
                snapshot.yard, snapshot.agent_position, block.label,
                relocation_selector=neutral_relocation_selector,
            )
            if plan is not None:
                raw_plans.append(plan)
        plan_by_label = {plan.block_label: plan for plan in raw_plans}
        def retrieval_key(item):
            candidate = item[1]
            plan = plan_by_label.get(candidate.target_label)
            if plan is None:
                block = snapshot.recovery_state.block(candidate.target_label)
                return (math.inf if block is None else block.remaining_time - candidate.recovery_action.steps,
                        0, candidate.recovery_action.steps, candidate.target_label)
            return (plan.slack, len(plan.relocations), plan.estimated_steps, candidate.target_label)
        delivers.sort(key=retrieval_key)
        due = [item for item in delivers if retrieval_key(item)[0] <= 0]
        preferred_cell = None
        projected = False
        valid_accepts = []
        if accepts:
            inbound = self._arrival_view(snapshot, accepts)
            preferred_cell = select_storage_location(
                snapshot.yard, inbound, source=inbound.position
            )
            valid_accepts = [(self._placement_key(snapshot, inbound, c), i, c)
                             for i, c in accepts]
            valid_accepts = [item for item in valid_accepts if item[0] is not None]
            valid_accepts.sort(key=lambda item: item[0])
            projected = bool(valid_accepts and valid_accepts[0][2].destination != preferred_cell)
        choice = None
        reason = None
        if due:
            choice, reason = due[0], "due_retrieval"
        elif valid_accepts:
            accept = (valid_accepts[0][1], valid_accepts[0][2])
            if delivers:
                delivery = delivers[0]
                plan = plan_by_label.get(delivery[1].target_label)
                accept_steps = self._accept_steps(snapshot, accept[1])
                projected_plan = None
                if plan is not None and accept_steps is not None:
                    aged = tuple(replace(block, remaining_time=block.remaining_time-accept_steps)
                                 for block in snapshot.yard.blocks)
                    projected_yard = replace(
                        snapshot.yard,
                        blocks=aged + (BlockView(inbound.label, accept[1].destination,
                                                inbound.remaining_time),),
                    )
                    projected_plan = plan_retrieval(
                        projected_yard, accept[1].destination, delivery[1].target_label,
                        relocation_selector=neutral_relocation_selector,
                    )
                if (plan is not None and
                    (projected_plan is None or abs(plan.slack) <=
                     abs(projected_plan.slack) + self.LOOKAHEAD_MARGIN_STEPS)):
                    choice, reason = delivery, "lookahead_retrieval"
                else:
                    choice, reason = accept, "lookahead_accept"
            else:
                choice, reason = accept, "accept_no_retrieval"
        elif snapshot.inbound_label is not None and delivers:
            choice, reason = delivers[0], "capacity_release_retrieval"
        elif defers and not any(plan.slack <= 0 for plan in raw_plans):
            choice, reason = min(defers, key=lambda item: item[1].key), "defer_to_event"
        elif reconfigs:
            raw_plans.sort(key=lambda plan: (
                plan.slack, len(plan.relocations), plan.estimated_steps, plan.block_label
            ))
            nominated = (raw_plans[0].relocations[0].block_label
                         if raw_plans and raw_plans[0].relocations else None)
            choice = min(reconfigs, key=lambda item: (
                item[1].target_label != nominated,
                item[1].recovery_action.steps,
                item[1].target_label,
                item[1].destination,
            ))
            reason = "blocked_reconfigure"
        elif delivers:
            choice, reason = delivers[0], "fallback_deliver"
        elif defers:
            choice, reason = min(defers, key=lambda item: item[1].key), "fallback_defer"
        elif accepts:
            # A certified Accept can fail Dynamic's earlier-access preference
            # screen.  Use the certified choice only as an explicitly logged
            # fallback, never call this faithful Dynamic placement.
            choice, reason = min(accepts, key=lambda item: item[1].key), "screened_accept_fallback"
            projected = True
        else:
            raise RuntimeError("certified Dynamic selector has no SAFE candidate")
        audit = {
            "decision_epoch": int(snapshot.decision_epoch),
            "reason": reason,
            "selected_key": choice[1].key,
            "dynamic_unfiltered_placement_cell": preferred_cell,
            "dynamic_placement_projected": projected,
            "safe_accept_count": len(accepts),
            "safe_deliver_count": len(delivers),
            "safe_reconfigure_count": len(reconfigs),
            "guard_forced": bool(prepared.liveness_restricted),
        }
        return choice[0], audit

    def select(self, snapshot, *, training=False, epsilon=0.0):
        if training or float(epsilon) != 0.0:
            raise ValueError("certified Dynamic deployment is deterministic")
        prepared, liveness_forced = self.base_agent._admissible_prepared(snapshot)
        if not prepared.records:
            raise RuntimeError("certified Dynamic received an empty SAFE frontier")
        if self.first_frontier_sha256 is None:
            keys = sorted(candidate.key for candidate in snapshot.candidates)
            self.first_frontier_sha256 = hashlib.sha256(
                json.dumps(keys, separators=(",", ":")).encode()
            ).hexdigest()
        preferred_index, audit = self._choice(snapshot, prepared)
        merits = torch.zeros(len(prepared.records), dtype=torch.float32,
                             device=self.base_agent.device)
        merits[preferred_index] = self.ONE_HOT_MERIT
        hierarchy = select_hierarchical_index(
            merits, prepared.mode_ids, self.base_agent.within_temperatures,
            candidate_keys=tuple(record.key for record in prepared.records),
        )
        if hierarchy.selected_index != preferred_index:
            raise RuntimeError("one-hot merit did not preserve Dynamic's preferred candidate")
        selected_index = 0 if liveness_forced else preferred_index
        source_index = prepared.source_indices[selected_index]
        candidate = snapshot.candidates[source_index]
        audit["actual_selected_key"] = candidate.key
        audit["guard_forced"] = liveness_forced
        self.choice_audit.append(audit)
        exact_rank_progress = bool(
            snapshot.audit.recovery_rank_exact and candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None and candidate.rank_delta > 0
        )
        decision = ViabilityGraphDecision(
            candidate=candidate, record=prepared.records[selected_index],
            prepared_snapshot=prepared,
            q_values=tuple(float(value) for value in merits.detach().cpu()),
            mode_values=tuple((ID_TO_MODE[mode], value)
                              for mode, value in hierarchy.mode_values),
            explored=False,
            selection_source=("exact_recovery_witness_guard" if liveness_forced
                              else self.POLICY),
            liveness_forced=liveness_forced,
            exact_rank_progress=exact_rank_progress,
        )
        base = self.base_agent
        base.decision_count += 1
        base.mode_decisions[candidate.mode.value] += 1
        base.action_decisions[candidate.action_type.value] += 1
        base.selection_sources[decision.selection_source] += 1
        base.safe_candidates_scored += len(prepared.records)
        base.interface_rejections += prepared.interface_rejection_count
        base.exact_rejections_seen += int(snapshot.audit.fail_closed_rejection_count)
        return decision

    def observe_outcome(self, decision, *, next_snapshot, done):
        return self.base_agent.observe_outcome(
            decision, next_snapshot=next_snapshot, done=done
        )
