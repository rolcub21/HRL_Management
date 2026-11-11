"""Matched whole-controller certification adapter for frozen E247 A3C.

The E247 network ranks placement cells only.  All other macro ordering follows
the same fixed Dynamic-style rule used by E270.  A stateless Gumbel ranking is
drawn once over the complete physical placement set; a certified arm takes the
first SAFE cell in that ranking while an unfiltered arm takes its first cell.
Thus the paired arms differ only in candidate eligibility, not in their random
policy draw.  The VCG recovery-witness guard is deliberately not consulted.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import torch

from PSLAP.dynamic_yard import BlockView
from PSLAP.kim2020_a3c_spatial import build_spatial_observation
from PSLAP.neutral_protocol import neutral_relocation_selector
from PSLAP.retrieval_dispatch import plan_retrieval
from PSLAP.viability_candidates import ViabilityActionType, ViabilityMode
from experiments.conditioned_vcg.E27_certified_dynamic_sustained_99k.selector import (
    CertifiedDynamicAgent,
)
from viability_graph_hierarchy import (
    ID_TO_MODE,
    NoCertifiedViableAction,
    ViabilityGraphDecision,
)
from viability_graph_preference_conditioned import select_hierarchical_index


class MatchedA3CPlacementAgent:
    """Frozen A3C placement ranker with one matched macro scheduler."""

    POLICY = "e270_matched_a3c_placement_dynamic_scheduler_v1"
    RANKING_CONTRACT = "stateless_gumbel_full_physical_placement_order_v1"
    ONE_HOT_MERIT = 1000.0
    LOOKAHEAD_MARGIN_STEPS = CertifiedDynamicAgent.LOOKAHEAD_MARGIN_STEPS

    def __init__(
        self,
        base_agent,
        source,
        tracker,
        *,
        policy_seed: int,
        checkpoint_sha256: str,
    ) -> None:
        self.base_agent = base_agent
        self.source = source
        self.tracker = tracker
        self.policy_seed = int(policy_seed)
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.choice_audit = []

    def __getattr__(self, name):
        return getattr(self.base_agent, name)

    def reset_episode_state(self) -> None:
        self.base_agent.reset_episode_state()
        self.source.on_episode_start()
        self.choice_audit.clear()

    on_episode_reset = reset_episode_state

    @staticmethod
    def _arrival_view(snapshot, accepts):
        return CertifiedDynamicAgent._arrival_view(snapshot, accepts)

    @staticmethod
    def _accept_steps(snapshot, candidate):
        return CertifiedDynamicAgent._accept_steps(snapshot, candidate)

    def _cell_noise(
        self, *, instance_id: str, time_step: int, block_label: str,
        source_cell, cell, observation_digest: str,
    ) -> float:
        key = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "policy_seed": self.policy_seed,
            "instance_id": str(instance_id),
            "time_step": int(time_step),
            "block_label": str(block_label),
            "source": tuple(source_cell),
            "cell": tuple(cell),
            "observation": observation_digest,
        }
        encoded = json.dumps(
            key, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        integer = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
        uniform = (integer + 0.5) / 2**64
        return -math.log(-math.log(uniform))

    def _placement_order(self, snapshot, physical_accepts):
        if not physical_accepts:
            return (), None
        inbound = self._arrival_view(snapshot, physical_accepts)
        cells = tuple(sorted({tuple(item.destination) for _, item in physical_accepts}))
        observation = build_spatial_observation(snapshot.yard, inbound, cells)
        logits = self.source.score_candidates(
            snapshot.yard, inbound, inbound.position, cells
        )
        observation_digest = hashlib.sha256(
            observation.spatial_features.tobytes()
        ).hexdigest()
        instance_id = getattr(
            getattr(self.source.env, "current_episode_instance", None),
            "instance_id", None,
        ) or getattr(snapshot, "episode_instance_id", "unknown-instance")
        scored = []
        for cell, logit in zip(cells, logits):
            score = float(logit) + self._cell_noise(
                instance_id=instance_id,
                time_step=int(getattr(self.source.env, "time_steps", 0)),
                block_label=inbound.label,
                source_cell=inbound.position,
                cell=cell,
                observation_digest=observation_digest,
            )
            scored.append((score, tuple(cell), float(logit)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(scored), inbound

    def _choice(self, snapshot, prepared):
        items = [
            (index, snapshot.candidates[source_index])
            for index, source_index in enumerate(prepared.source_indices)
        ]
        typed = {
            kind: [(index, candidate) for index, candidate in items
                   if candidate.action_type is kind]
            for kind in ViabilityActionType
        }
        accepts = typed[ViabilityActionType.ACCEPT]
        delivers = typed[ViabilityActionType.DELIVER]
        reconfigs = typed[ViabilityActionType.RECONFIGURE]
        defers = typed[ViabilityActionType.DEFER]

        latest = self.tracker.latest
        if latest is None or latest[1] is None:
            raise RuntimeError("E270 A3C ranker lacks the matched physical frontier")
        physical = latest[1]
        physical_accepts = [
            (index, candidate)
            for index, candidate in enumerate(physical.candidates)
            if candidate.action_type is ViabilityActionType.ACCEPT
        ]
        placement_order, inbound = self._placement_order(
            snapshot, physical_accepts
        )
        placement_rank = {
            cell: rank for rank, (_score, cell, _logit)
            in enumerate(placement_order)
        }
        eligible_accepts = [
            item for item in accepts
            if tuple(item[1].destination) in placement_rank
        ]
        eligible_accepts.sort(
            key=lambda item: (
                placement_rank[tuple(item[1].destination)], item[1].key
            )
        )

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
                return (
                    math.inf if block is None else
                    block.remaining_time - candidate.recovery_action.steps,
                    0, candidate.recovery_action.steps, candidate.target_label,
                )
            return (
                plan.slack, len(plan.relocations), plan.estimated_steps,
                candidate.target_label,
            )

        delivers.sort(key=retrieval_key)
        due = [item for item in delivers if retrieval_key(item)[0] <= 0]
        choice = None
        reason = None
        if due:
            choice, reason = due[0], "due_retrieval"
        elif eligible_accepts:
            accept = eligible_accepts[0]
            if delivers:
                delivery = delivers[0]
                plan = plan_by_label.get(delivery[1].target_label)
                accept_steps = self._accept_steps(snapshot, accept[1])
                projected_plan = None
                if plan is not None and accept_steps is not None:
                    aged = tuple(
                        replace(
                            block,
                            remaining_time=block.remaining_time - accept_steps,
                        )
                        for block in snapshot.yard.blocks
                    )
                    projected_yard = replace(
                        snapshot.yard,
                        blocks=aged + (
                            BlockView(
                                inbound.label, accept[1].destination,
                                inbound.remaining_time,
                            ),
                        ),
                    )
                    projected_plan = plan_retrieval(
                        projected_yard, accept[1].destination,
                        delivery[1].target_label,
                        relocation_selector=neutral_relocation_selector,
                    )
                if (
                    plan is not None
                    and (
                        projected_plan is None
                        or abs(plan.slack) <= abs(projected_plan.slack)
                        + self.LOOKAHEAD_MARGIN_STEPS
                    )
                ):
                    choice, reason = delivery, "lookahead_retrieval"
                else:
                    choice, reason = accept, "lookahead_accept"
            else:
                choice, reason = accept, "accept_no_retrieval"
        elif snapshot.inbound_label is not None and delivers:
            choice, reason = delivers[0], "capacity_release_retrieval"
        elif defers and not any(plan.slack <= 0 for plan in raw_plans):
            choice = min(defers, key=lambda item: item[1].key)
            reason = "defer_to_event"
        elif reconfigs:
            raw_plans.sort(key=lambda plan: (
                plan.slack, len(plan.relocations), plan.estimated_steps,
                plan.block_label,
            ))
            nominated = (
                raw_plans[0].relocations[0].block_label
                if raw_plans and raw_plans[0].relocations else None
            )
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
            choice = min(defers, key=lambda item: item[1].key)
            reason = "fallback_defer"
        elif accepts:
            choice = min(accepts, key=lambda item: item[1].key)
            reason = "screened_accept_fallback"
        else:
            raise RuntimeError("matched A3C scheduler has no eligible candidate")

        full_cell = None if not placement_order else placement_order[0][1]
        selected_cell = (
            tuple(choice[1].destination)
            if choice[1].action_type is ViabilityActionType.ACCEPT else None
        )
        return choice[0], {
            "decision_epoch": int(snapshot.decision_epoch),
            "reason": reason,
            "selected_key": choice[1].key,
            "a3c_unfiltered_first_cell": full_cell,
            "a3c_selected_cell": selected_cell,
            "a3c_placement_projected": bool(
                selected_cell is not None and full_cell != selected_cell
            ),
            "physical_accept_count": len(physical_accepts),
            "eligible_accept_count": len(accepts),
            "safe_deliver_count": len(delivers),
            "safe_reconfigure_count": len(reconfigs),
            "ranking_contract": self.RANKING_CONTRACT,
            "vcg_guard_used": False,
        }

    def select(self, snapshot, *, training=False, epsilon=0.0):
        if training or float(epsilon) != 0.0:
            raise ValueError("matched A3C evaluation is frozen")
        prepared = self.base_agent.prepare_snapshot(snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction("A3C received an empty frontier")
        preferred_index, audit = self._choice(snapshot, prepared)
        merits = torch.zeros(
            len(prepared.records), dtype=torch.float32,
            device=self.base_agent.device,
        )
        merits[preferred_index] = self.ONE_HOT_MERIT
        hierarchy = select_hierarchical_index(
            merits, prepared.mode_ids, self.base_agent.within_temperatures,
            candidate_keys=tuple(record.key for record in prepared.records),
        )
        if hierarchy.selected_index != preferred_index:
            raise RuntimeError("one-hot merit changed the A3C preferred candidate")
        source_index = prepared.source_indices[preferred_index]
        candidate = snapshot.candidates[source_index]
        exact_rank_progress = bool(
            snapshot.audit.recovery_rank_exact
            and candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None
            and candidate.rank_delta > 0
        )
        decision = ViabilityGraphDecision(
            candidate=candidate,
            record=prepared.records[preferred_index],
            prepared_snapshot=prepared,
            q_values=tuple(float(value) for value in merits.detach().cpu()),
            mode_values=tuple(
                (ID_TO_MODE[mode], value)
                for mode, value in hierarchy.mode_values
            ),
            explored=False,
            selection_source=self.POLICY,
            liveness_forced=False,
            exact_rank_progress=exact_rank_progress,
        )
        base = self.base_agent
        base.decision_count += 1
        base.mode_decisions[candidate.mode.value] += 1
        base.action_decisions[candidate.action_type.value] += 1
        base.selection_sources[self.POLICY] += 1
        base.safe_candidates_scored += len(prepared.records)
        base.interface_rejections += prepared.interface_rejection_count
        base.exact_rejections_seen += int(
            snapshot.audit.fail_closed_rejection_count
        )
        self.tracker.record_selection(decision)
        audit["actual_selected_key"] = candidate.key
        self.choice_audit.append(audit)
        return decision

    def observe_outcome(self, decision, *, next_snapshot, done):
        """Validate the next frontier without activating the VCG guard."""

        if done:
            return None
        if next_snapshot is None:
            raise ValueError("nonterminal A3C outcome requires next frontier")
        prepared = self.base_agent.prepare_snapshot(next_snapshot)
        if not prepared.records:
            raise NoCertifiedViableAction(
                "nonterminal A3C state has no eligible macro"
            )
        return prepared

    def audit(self) -> dict:
        enabled = self.tracker.arm == "a3c_certified"
        return {
            "policy": self.POLICY,
            "ranking_contract": self.RANKING_CONTRACT,
            "checkpoint_sha256": self.checkpoint_sha256,
            "policy_seed": self.policy_seed,
            "certification_filter_enabled": enabled,
            "whole_controller_filter_contract": True,
            "whole_controller_certification": enabled,
            "certification_scope": ["Accept", "Reconfigure"],
            "deliver_filter_required": False,
            "vcg_guard_used": False,
            "choice_audit": list(self.choice_audit),
        }


__all__ = ["MatchedA3CPlacementAgent"]
