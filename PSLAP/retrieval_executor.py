"""Deterministic execution of one explicitly named retrieval job."""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from PSLAP.dynamic_yard import BlockView, YardSnapshot, shortest_clear_path
from PSLAP.online_policy import plan_move_and_putdown
from PSLAP.retrieval_context import retrieval_planning_context


class RetrievalExecutor:
    """Relocate blockers as needed, then retrieve and deliver one target.

    Planning is read-only.  The only planning-related live mutation is changing
    a block's relocation target immediately before its relocation PUTDOWN,
    matching the environment's existing relocation contract.
    """

    VERSION = "named_atomic_retrieval_executor_v1"

    def __init__(self, env, target_label, *, relocation_selector=None):
        self.env = env
        self.target_label = str(target_label)
        self.relocation_selector = relocation_selector
        self.reset()

    def reset(self):
        self.started = False
        self.done = False
        self.failed = False
        self.failure_reason = None
        self.initial_estimated_steps = None
        self.actual_steps = 0
        self.relocation_count = 0
        self.replan_count = 0
        self.midleg_replan_count = 0
        self._pending_plan = None
        self._actions = deque()
        self._leg_kind = None
        self._leg_block_label = None
        self._leg_target = None
        self._seen_yards = set()

    def _block(self, label):
        return next(
            (block for block in self.env.blocks if block.label == label), None
        )

    def _target(self):
        return self._block(self.target_label)

    def _yard_signature(self):
        return tuple(
            sorted(
                (block.label, tuple(block.position))
                for block in self.env.blocks
                if block.stored
                and not block.delivered
                and not block.carrying
                and block.position is not None
            )
        )

    def _plan(self):
        return retrieval_planning_context(
            self.env,
            relocation_selector=self.relocation_selector,
        ).plan(self.target_label)

    def _plan_carried_delivery(self, target):
        candidates = []
        for exit_cell in self.env.exit_cells:
            actions = self.env.plan_path_heuristic(
                self.env.current_state,
                exit_cell,
                ignore_block=target,
            )
            if self.env.current_state == exit_cell or actions:
                candidates.append((len(actions), tuple(exit_cell), list(actions)))
        if not candidates:
            return None
        _, exit_cell, actions = min(candidates, key=lambda item: (item[0], item[1]))
        return actions + [self.env.ACTION_IDS["PUTDOWN"]], exit_cell

    def _first_live_leg_actions(self, plan):
        """Build the first canonical leg against the complete live occupancy.

        ``RetrievalPlan`` deliberately contains only stored-yard inventory.
        The live route planner additionally sees inbound blocks at the waiting
        and pickup cells.  Strict admission uses this read-only check to ensure
        that the first complete pickup-and-putdown leg is executable now.
        """

        if plan.relocations:
            relocation = plan.relocations[0]
            blocker = self._block(relocation.block_label)
            return (
                None
                if blocker is None
                else plan_move_and_putdown(
                    self.env,
                    blocker,
                    relocation.target,
                )
            )
        target = self._target()
        return (
            None
            if target is None
            else plan_move_and_putdown(self.env, target, plan.exit_cell)
        )

    def _accept_unheld_plan(self, plan):
        """Versioned admission hook; v1 preserves canonical-plan semantics."""

        return True

    def _unheld_plan_failure_reason(self):
        return "unheld_plan_unavailable"

    def _continuation_after_relocation(self, plan):
        """v1/v2 replan after each relocation, preserving old execution."""

        return None

    def can_start(self):
        target = self._target()
        if target is None or not target.stored or target.delivered:
            return False
        carried = next(
            (block for block in self.env.blocks if block.carrying), None
        )
        if carried is not None:
            return carried is target and self._plan_carried_delivery(target) is not None
        plan = self._plan()
        return plan is not None and self._accept_unheld_plan(plan)

    def start(self):
        self.reset()
        target = self._target()
        if target is None or not target.stored or target.delivered:
            self._fail("target_not_retrievable")
            return False
        carried = next(
            (block for block in self.env.blocks if block.carrying), None
        )
        if carried is not None:
            if carried is not target:
                self._fail("different_block_already_carried")
                return False
            carried_plan = self._plan_carried_delivery(target)
            if carried_plan is None:
                self._fail("carried_target_has_no_exit_path")
                return False
            actions, exit_cell = carried_plan
            self._begin_leg("delivery", target.label, exit_cell, actions)
            self.initial_estimated_steps = len(actions)
        else:
            self._pending_plan = self._plan()
            if self._pending_plan is None:
                self._fail("no_canonical_retrieval_plan")
                return False
            if not self._accept_unheld_plan(self._pending_plan):
                self._fail(self._unheld_plan_failure_reason())
                return False
            self.initial_estimated_steps = self._pending_plan.estimated_steps
        self.started = True
        return True

    def _begin_leg(self, kind, block_label, target, actions):
        if not actions:
            self._fail(f"empty_{kind}_leg")
            return False
        self._leg_kind = kind
        self._leg_block_label = block_label
        self._leg_target = tuple(target)
        self._actions = deque(actions)
        return True

    def _fail(self, reason):
        self.failed = True
        self.failure_reason = str(reason)
        self._actions.clear()

    def _occupant(self, position, *, ignore_label=None):
        return next(
            (
                block
                for block in self.env.blocks
                if block.label != ignore_label
                and block.position == position
                and not block.delivered
                and not block.carrying
            ),
            None,
        )

    def _action_is_valid(self, action):
        """Validate the next cached action against the current live yard."""

        block = self._block(self._leg_block_label)
        if block is None:
            return False
        name = self.env.ACTION_NAMES[action]
        if name in ("UP", "DOWN", "LEFT", "RIGHT"):
            intended = self.env._get_intended_cell(
                self.env.current_state, action
            )
            if intended == self.env.current_state:
                return False
            # The environment permits walking through inventory while hands
            # are free, but a canonical retrieval path does not.  Only the
            # block being approached may occupy the next cell.
            occupant = self._occupant(intended, ignore_label=block.label)
            return occupant is None
        if name == "PICKUP":
            return bool(
                block.position == self.env.current_state
                and not block.carrying
                and not block.delivered
                and not any(item.carrying for item in self.env.blocks)
            )
        if name == "PUTDOWN":
            if not block.carrying or block.position != self.env.current_state:
                return False
            if self._leg_kind == "delivery":
                return bool(
                    block.label == self.target_label
                    and self.env.current_state in self.env.exit_cells
                )
            return bool(
                self._leg_kind == "relocation"
                and self.env.current_state == self._leg_target
                and self._occupant(
                    self._leg_target, ignore_label=block.label
                )
                is None
            )
        return False

    def _carried_leg(self, block):
        if self._leg_kind == "delivery":
            carried_plan = self._plan_carried_delivery(block)
            if carried_plan is None:
                return None
            actions, destination = carried_plan
            return "delivery", destination, actions

        destinations = [self._leg_target, block.storage_location]
        for destination in dict.fromkeys(destinations):
            if destination is None:
                continue
            if self._occupant(destination, ignore_label=block.label) is not None:
                continue
            actions = self.env.plan_path_heuristic(
                self.env.current_state,
                destination,
                ignore_block=block,
            )
            if self.env.current_state == destination or actions:
                return (
                    "relocation",
                    destination,
                    list(actions) + [self.env.ACTION_IDS["PUTDOWN"]],
                )
        return None

    def _rebuild_current_leg(self):
        """Replan after a cached action becomes stale without retargeting."""

        self.midleg_replan_count += 1
        self.replan_count += 1
        if self.midleg_replan_count > max(8, 4 * len(self.env.blocks)):
            if any(item.carrying for item in self.env.blocks):
                raise RuntimeError(
                    "Retrieval executor exceeded its replan bound while carrying"
                )
            self._fail("midleg_replan_bound_exceeded")
            return
        block = self._block(self._leg_block_label)
        carried = next(
            (item for item in self.env.blocks if item.carrying), None
        )
        if carried is not None:
            if block is None or carried is not block:
                raise RuntimeError(
                    "Retrieval executor observed an unrelated carried block"
                )
            recovery = self._carried_leg(block)
            if recovery is None:
                raise RuntimeError(
                    "Retrieval executor cannot safely recover its carried block"
                )
            kind, destination, actions = recovery
            self._begin_leg(kind, block.label, destination, actions)
            return

        # No inventory is in hand, so it is safe to discard the stale leg and
        # rebuild the complete named-target plan from the live yard.
        self._actions.clear()
        self._leg_kind = None
        self._leg_block_label = None
        self._leg_target = None
        self._pending_plan = self._plan()
        if self._pending_plan is None:
            self._fail("stale_leg_canonical_replan_failed")
            return
        self._seen_yards.discard(self._yard_signature())
        self._prepare_next_leg()

    def _finish_previous_leg(self):
        if self._leg_kind is None:
            return True
        block = self._block(self._leg_block_label)
        if self._leg_kind == "relocation":
            valid = bool(
                block is not None
                and block.stored
                and not block.delivered
                and not block.carrying
                and block.position == self._leg_target
                and block.storage_location == self._leg_target
            )
            if not valid:
                self._fail("relocation_postcondition_failed")
                return False
            self.relocation_count += 1
        elif self._leg_kind == "delivery":
            if block is None or not block.delivered:
                self._fail("delivery_postcondition_failed")
                return False
            self.done = True
        self._leg_kind = None
        self._leg_block_label = None
        self._leg_target = None
        return not self.failed

    def _prepare_next_leg(self):
        if not self._finish_previous_leg() or self.done:
            return
        signature = self._yard_signature()
        if signature in self._seen_yards:
            self._fail("relocation_cycle_detected")
            return
        self._seen_yards.add(signature)
        if self.relocation_count > len(self.env.blocks):
            self._fail("relocation_bound_exceeded")
            return

        plan = self._pending_plan
        self._pending_plan = None
        if plan is None:
            self.replan_count += 1
            plan = self._plan()
        if plan is None:
            self._fail("canonical_replan_failed")
            return
        if plan.relocations:
            relocation = plan.relocations[0]
            blocker = self._block(relocation.block_label)
            actions = (
                None
                if blocker is None
                else plan_move_and_putdown(
                    self.env, blocker, relocation.target
                )
            )
            if actions is None:
                self._fail("relocation_route_failed")
                return
            self._begin_leg(
                "relocation",
                relocation.block_label,
                relocation.target,
                actions,
            )
            self._pending_plan = self._continuation_after_relocation(plan)
            return

        target = self._target()
        if target is None:
            self._fail("target_disappeared")
            return
        actions = plan_move_and_putdown(self.env, target, plan.exit_cell)
        if actions is None:
            self._fail("delivery_route_failed")
            return
        self._begin_leg("delivery", target.label, plan.exit_cell, actions)

    def next_action(self):
        if self.failed:
            if any(block.carrying for block in self.env.blocks):
                raise RuntimeError(
                    "Failed retrieval executor may not wait while carrying"
                )
            return self.env.ACTION_IDS["WAIT"]
        if self.done:
            return self.env.ACTION_IDS["WAIT"]
        if not self.started:
            raise RuntimeError("retrieval executor has not been started")
        if not self._actions:
            self._prepare_next_leg()
        if self.done or self.failed or not self._actions:
            return self.env.ACTION_IDS["WAIT"]

        action = self._actions[0]
        if not self._action_is_valid(action):
            self._rebuild_current_leg()
            if self.done or self.failed or not self._actions:
                return self.env.ACTION_IDS["WAIT"]
            action = self._actions[0]
            if not self._action_is_valid(action):
                raise RuntimeError(
                    "Retrieval executor produced an invalid action after replanning"
                )
        action = self._actions.popleft()
        if (
            self._leg_kind == "relocation"
            and not self._actions
            and action == self.env.ACTION_IDS["PUTDOWN"]
        ):
            blocker = self._block(self._leg_block_label)
            if (
                blocker is None
                or not blocker.carrying
                or self.env.current_state != self._leg_target
                or self._occupant(
                    self._leg_target, ignore_label=self._leg_block_label
                )
                is not None
            ):
                self._fail("relocation_putdown_without_carried_block")
                return self.env.ACTION_IDS["WAIT"]
            blocker.storage_location = self._leg_target
        self.actual_steps += 1
        return action


class StrictStartRetrievalExecutor(RetrievalExecutor):
    """v2 admission: canonical planning plus an executable first live leg.

    This is separate from v1 so checkpoints retained as internal ablations
    continue to reproduce their original canonical-plan-only initiation rule.
    """

    VERSION = "named_atomic_retrieval_executor_v2"

    def _accept_unheld_plan(self, plan):
        return self._first_live_leg_actions(plan) is not None

    def _unheld_plan_failure_reason(self):
        return "first_live_leg_unavailable"


class CompleteLiveRetrievalExecutor(StrictStartRetrievalExecutor):
    """v3 admission: validate every canonical leg against live occupancy.

    Canonical retrieval planning intentionally models stored inventory only.
    Before admitting an atomic retrieval, v3 augments that immutable yard with
    every other currently visible block and replays the complete sequence of
    relocations plus the target delivery.  No live block is moved and no
    environment state is mutated by this validation.

    The check certifies the current live yard.  Arrivals that occur after the
    macro starts are still handled by the executor's bounded live replanning.
    """

    VERSION = "named_atomic_retrieval_executor_v3"
    LIVE_PLAN_CONTRACT = "complete_current_live_multileg_validation_v1"
    _FIXED_LABEL_PREFIX = "__retrieval_v3_live_obstacle__"

    def _live_validation_yard(self):
        """Return stored inventory plus non-stored live blocks as obstacles."""

        yard = YardSnapshot.from_env(self.env)
        stored_labels = frozenset(block.label for block in yard.blocks)
        fixed_positions = sorted(
            {
                tuple(block.position)
                for block in self.env.blocks
                if (
                    block.label not in stored_labels
                    and block.position is not None
                    and tuple(block.position) in yard.traversable
                    and not block.delivered
                    and not block.carrying
                )
            }
        )
        fixed_blocks = tuple(
            BlockView(
                label=f"{self._FIXED_LABEL_PREFIX}{index:04d}",
                position=position,
                remaining_time=float("inf"),
            )
            for index, position in enumerate(fixed_positions)
        )
        return replace(yard, blocks=yard.blocks + fixed_blocks)

    @staticmethod
    def _destination_is_free(yard, destination, moving_label):
        occupant = yard.occupancy().get(tuple(destination))
        return occupant is None or occupant.label == moving_label

    def _complete_live_plan_is_executable(self, plan):
        """Replay ``plan`` read-only against all currently visible blocks."""

        yard = self._live_validation_yard()
        cursor = tuple(self.env.current_state)

        for relocation in plan.relocations:
            blocker = yard.block(relocation.block_label)
            destination = tuple(relocation.target)
            if blocker is None or not self._destination_is_free(
                yard, destination, blocker.label
            ):
                return False
            if shortest_clear_path(
                yard,
                cursor,
                blocker.position,
                ignore_labels=(blocker.label,),
            ) is None:
                return False
            if shortest_clear_path(
                yard,
                blocker.position,
                destination,
                ignore_labels=(blocker.label,),
            ) is None:
                return False
            yard = yard.with_block(blocker, destination)
            cursor = destination

        target = yard.block(plan.block_label)
        destination = tuple(plan.exit_cell)
        if target is None or not self._destination_is_free(
            yard, destination, target.label
        ):
            return False
        if shortest_clear_path(
            yard,
            cursor,
            target.position,
            ignore_labels=(target.label,),
        ) is None:
            return False
        return (
            shortest_clear_path(
                yard,
                target.position,
                destination,
                ignore_labels=(target.label,),
            )
            is not None
        )

    def _accept_unheld_plan(self, plan):
        return self._complete_live_plan_is_executable(plan)

    def _unheld_plan_failure_reason(self):
        return "complete_live_plan_unavailable"

    def _continuation_after_relocation(self, plan):
        """Execute the validated tail instead of choosing a new live route."""

        return replace(plan, relocations=plan.relocations[1:])


__all__ = [
    "RetrievalExecutor",
    "StrictStartRetrievalExecutor",
    "CompleteLiveRetrievalExecutor",
]
