"""Online execution shared by storage-location assignment baselines."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

from PSLAP.dynamic_yard import BlockView, Cell, YardSnapshot
from PSLAP.retrieval_dispatch import (
    plan_retrieval,
    select_dispatchable_retrieval,
)
from PSLAP.viability import (
    RecoveryActionKind,
    RecoveryState,
    analyze_recoverability,
)


StorageAssigner = Callable[[YardSnapshot, BlockView, Cell], Optional[Cell]]


EXACT_RECOVERY_FALLBACK_CONTRACT = (
    "overdue_greedy_stall_goal_directed_strict_macro_witness_v1"
)
EXACT_RECOVERY_MAX_NODES = 100_000


def plan_move_and_putdown(env, block, target) -> Optional[list[int]]:
    """Build the deterministic movement/handling sequence shared by methods."""

    source = block.position
    if source is None or target is None:
        return None
    to_block = env.plan_path_heuristic(
        env.current_state, source, ignore_block=block
    )
    if env.current_state != source and not to_block:
        return None
    with_block = env.plan_path_heuristic(
        source, target, ignore_block=block
    )
    if source != target and not with_block:
        return None
    return (
        list(to_block)
        + [env.ACTION_IDS["PICKUP"]]
        + list(with_block)
        + [env.ACTION_IDS["PUTDOWN"]]
    )


class OnlinePSLAPPolicy:
    """Combine one storage assigner with the common retrieval dispatcher."""

    def __init__(
        self,
        env,
        storage_assigner: StorageAssigner,
        *,
        relocation_selector=None,
    ):
        self.env = env
        self.storage_assigner = storage_assigner
        self.relocation_selector = relocation_selector
        self.retrieval_live_plan_failure_count = 0
        self.inbound_approach_defer_count = 0
        self.exact_recovery_search_count = 0
        self.exact_recovery_fallback_count = 0
        self.exact_recovery_failure_count = 0
        self.exact_recovery_explored_nodes = 0
        self._failed_recovery_signature = None

    def _env_block(self, label: str):
        return next(
            (block for block in self.env.blocks if block.label == label), None
        )

    def _plan_move_and_putdown(self, block, target) -> Optional[list[int]]:
        return plan_move_and_putdown(self.env, block, target)

    def _relocate(self, block, target) -> Optional[list[int]]:
        actions = self._plan_move_and_putdown(block, target)
        if actions is None:
            return None
        block.storage_location = target
        return actions

    def _retrieve(self, block, exit_cell) -> Optional[list[int]]:
        return self._plan_move_and_putdown(block, exit_cell)

    def _can_approach(self, block) -> bool:
        """Return whether the live agent can reach ``block`` before assignment.

        Storage sources may update learning state when called.  An inbound
        placement is therefore not a decision epoch until the executor can
        reach the pickup block and admit the storage macro.
        """

        source = block.position
        if source is None:
            return False
        if self.env.current_state == source:
            return True
        return bool(
            self.env.plan_path_heuristic(
                self.env.current_state, source, ignore_block=block
            )
        )

    def _retrieval_snapshot(self) -> YardSnapshot:
        """Build a retrieval view consistent with the live path executor.

        ``YardSnapshot.from_env`` deliberately models only stored inventory as
        relocatable blocks.  Arrived inbound/waiting blocks are nevertheless
        hard obstacles to the live A* executor.  Removing their cells from the
        retrieval graph prevents the detached dispatcher from selecting a
        route that execution must reject.  The agent's current cell remains a
        valid path origin if it happens to overlap an uncarried block.
        """

        yard = YardSnapshot.from_env(self.env)
        transient_obstacles = frozenset(
            block.position
            for block in self.env.blocks
            if (
                block.position in yard.traversable
                and block.position != self.env.current_state
                and not block.stored
                and not block.delivered
                and not block.carrying
            )
        )
        if not transient_obstacles:
            return yard
        return replace(
            yard,
            traversable=yard.traversable - transient_obstacles,
        )

    def _inbound_block(self):
        return next(
            (
                block
                for block in self.env.blocks
                if (
                    block.position == self.env.pickup_cell
                    and not block.stored
                    and not block.delivered
                    and not block.carrying
                )
            ),
            None,
        )

    def _recovery_signature(self, yard: YardSnapshot):
        return (
            tuple(self.env.current_state),
            tuple(
                sorted(
                    (block.label, block.position) for block in yard.blocks
                )
            ),
            tuple(sorted(yard.traversable)),
        )

    def _exact_recovery_actions(
        self, yard: YardSnapshot
    ) -> Optional[list[int]]:
        """Return one certified liveness macro after a greedy retrieval stall.

        The normal dispatcher remains authoritative whenever it can produce a
        due retrieval.  Exact search is activated only when at least one stored
        block is already overdue, no inbound block can first be cleared, and
        the normal dispatcher returned no plan.  Its first witness transition
        is a common physical recovery action, not a storage-policy decision.
        """

        if not yard.blocks or not any(
            block.remaining_time <= 0.0 for block in yard.blocks
        ):
            return None

        signature = self._recovery_signature(yard)
        if signature == self._failed_recovery_signature:
            return None

        state = RecoveryState.from_yard_snapshot(
            yard,
            self.env.current_state,
        )
        certificate = analyze_recoverability(
            state,
            max_nodes=EXACT_RECOVERY_MAX_NODES,
            search_order="goal_directed",
        )
        self.exact_recovery_search_count += 1
        self.exact_recovery_explored_nodes += int(
            certificate.explored_nodes
        )
        if not certificate.is_safe or not certificate.witness:
            self.exact_recovery_failure_count += 1
            self._failed_recovery_signature = signature
            return None

        action = certificate.witness[0]
        block = self._env_block(action.block_label)
        if block is None:
            self.exact_recovery_failure_count += 1
            self._failed_recovery_signature = signature
            return None
        if action.kind is RecoveryActionKind.RELOCATION:
            actions = self._relocate(block, action.destination)
        elif action.kind is RecoveryActionKind.DELIVERY:
            actions = self._retrieve(block, action.destination)
        else:  # pragma: no cover - the verifier has a closed action enum.
            actions = None
        if not actions:
            self.exact_recovery_failure_count += 1
            self._failed_recovery_signature = signature
            return None

        self._failed_recovery_signature = None
        self.exact_recovery_fallback_count += 1
        return actions

    def step(self):
        """Yield one coherent storage or travel-time-aware retrieval sequence."""

        yard = self._retrieval_snapshot()
        retrieval = select_dispatchable_retrieval(
            yard,
            self.env.current_state,
            relocation_selector=self.relocation_selector,
        )
        if retrieval is not None:
            for relocation in retrieval.relocations:
                blocker = self._env_block(relocation.block_label)
                if blocker is None:
                    self.retrieval_live_plan_failure_count += 1
                    yield self.env.ACTION_IDS["WAIT"]
                    return
                actions = self._relocate(blocker, relocation.target)
                if not actions:
                    self.retrieval_live_plan_failure_count += 1
                    yield self.env.ACTION_IDS["WAIT"]
                    return
                yield from actions

            # Validate again against live state after all relocations.
            yard = self._retrieval_snapshot()
            refreshed = plan_retrieval(
                yard,
                self.env.current_state,
                retrieval.block_label,
                relocation_selector=self.relocation_selector,
            )
            due = self._env_block(retrieval.block_label)
            if due is None or refreshed is None or refreshed.relocations:
                self.retrieval_live_plan_failure_count += 1
                yield self.env.ACTION_IDS["WAIT"]
                return
            actions = self._retrieve(due, refreshed.exit_cell)
            if not actions:
                self.retrieval_live_plan_failure_count += 1
                yield self.env.ACTION_IDS["WAIT"]
                return
            yield from actions
            return

        inbound = self._inbound_block()
        if inbound is not None:
            # Do not ask a stateful assignment source to select or record an
            # action before the live agent-to-pickup leg is executable.  Dense
            # yards can temporarily seal that leg even while pickup-to-storage
            # candidates remain feasible.
            if not self._can_approach(inbound):
                self.inbound_approach_defer_count += 1
                recovery_actions = self._exact_recovery_actions(yard)
                if recovery_actions:
                    yield from recovery_actions
                    return
                yield self.env.ACTION_IDS["WAIT"]
                return
            yard = YardSnapshot.from_env(self.env)
            inbound_view = BlockView(
                label=inbound.label,
                position=inbound.position,
                remaining_time=float(inbound.get_remaining_storage_time()),
            )
            target = self.storage_assigner(
                yard, inbound_view, inbound.position
            )
            if target is None:
                # A reachable inbound block can still have no physically
                # admissible storage cell.  Waiting alone cannot change that
                # topology when the greedy due-retrieval dispatcher is also
                # stalled, so use the same common exact recovery fallback to
                # clear capacity before asking the assignment source again.
                recovery_actions = self._exact_recovery_actions(
                    self._retrieval_snapshot()
                )
                if recovery_actions:
                    yield from recovery_actions
                    return
                yield self.env.ACTION_IDS["WAIT"]
                return
            actions = self._plan_move_and_putdown(inbound, target)
            if not actions:
                yield self.env.ACTION_IDS["WAIT"]
                return
            inbound.storage_location = target
            yield from actions
            return

        recovery_actions = self._exact_recovery_actions(yard)
        if recovery_actions:
            yield from recovery_actions
            return

        yield self.env.ACTION_IDS["WAIT"]


__all__ = [
    "EXACT_RECOVERY_FALLBACK_CONTRACT",
    "EXACT_RECOVERY_MAX_NODES",
    "OnlinePSLAPPolicy",
    "StorageAssigner",
    "plan_move_and_putdown",
]
