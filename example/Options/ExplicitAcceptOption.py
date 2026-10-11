"""One policy-independent, explicitly bound inbound-placement macro."""

from __future__ import annotations

from collections import deque

from option import BaseOption


class ExplicitAcceptOption(BaseOption):
    """Place one named inbound block in one explicitly supplied cell.

    The block, destination, episode instance, and decision epoch are bound at
    construction.  No selector, registry, or baseline is consulted: callers
    choose the parameterized action ``(block_label, destination)`` before they
    instantiate this option.

    Both the unladen approach and laden transport legs use the environment's
    canonical A* planner.  The target block alone is ignored; every other live
    block remains an obstacle.  ``storage_location`` stays ``None`` until the
    initial-placement PUTDOWN is emitted, and the environment owns the storage
    event and starts its clock while consuming that action.
    """

    VERSION = "bound_explicit_accept_option_v1"
    BINDING_CONTRACT = "inbound_block_cell_episode_epoch_bound_once_v1"
    PATH_CONTRACT = "canonical_other_blocks_obstruct_target_ignored_v1"
    CLOCK_CONTRACT = "initial_storage_clock_starts_on_env_putdown_v1"

    def __init__(
        self,
        env,
        block_label: str,
        destination,
        *,
        max_replans: int = 8,
    ):
        super().__init__(is_primitive=False)
        self.env = env
        self.target_label = str(block_label)
        try:
            row, column = destination
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "destination must be a two-dimensional cell"
            ) from exc
        self.destination = (int(row), int(column))
        if self.destination != tuple(destination):
            raise ValueError("destination coordinates must be integral")
        self.max_replans = int(max_replans)
        if self.max_replans <= 0:
            raise ValueError("max_replans must be positive")

        matching = [
            block
            for block in env.blocks
            if block.label == self.target_label
        ]
        if not matching:
            raise ValueError(f"unknown block label: {self.target_label}")
        if len(matching) != 1:
            raise ValueError(f"block label is not unique: {self.target_label}")
        self._bound_block = matching[0]
        self.source_cell = (
            None
            if self._bound_block.position is None
            else tuple(self._bound_block.position)
        )
        self.bound_storage_location = (
            None
            if self._bound_block.storage_location is None
            else tuple(self._bound_block.storage_location)
        )
        self.bound_stored_time_step = self._bound_block.stored_time_step
        self.bound_storage_steps_elapsed = int(
            self._bound_block.storage_steps_elapsed
        )
        self.bound_time_step = int(env.time_steps)
        instance = getattr(env, "current_episode_instance", None)
        self.bound_instance_id = (
            None if instance is None else instance.instance_id
        )

        cell_id = f"{self.destination[0]:02d}-{self.destination[1]:02d}"
        self.controller_identifier = (
            f"option:ExplicitAccept:{self.target_label}:cell-{cell_id}"
        )

        self.success_count = 0
        self.failure_count = 0
        self.episode_success_count = 0
        self.episode_failure_count = 0
        self.episode_replans = 0
        self.episode_outcomes = []
        self.last_outcome = None
        self._consumed = False
        self._reset_active()

    def _reset_active(self):
        self.started = False
        self.failed = False
        self.failure_reason = None
        self.actual_steps = 0
        self.replan_count = 0
        self._actions = deque()
        self._completion_recorded = False
        self.putdown_emitted_time_step = None
        self.expected_stored_time_step = None

    def _block(self):
        if self._bound_block not in self.env.blocks:
            return None
        matching = [
            block
            for block in self.env.blocks
            if block.label == self.target_label
        ]
        if len(matching) != 1 or matching[0] is not self._bound_block:
            return None
        return self._bound_block

    def _physical_inbound_blocks(self):
        return [
            block
            for block in self.env.blocks
            if block.position == self.env.pickup_cell
            and not block.carrying
            and not block.stored
            and not block.delivered
        ]

    def _occupant(self, cell, *, ignore_label=None):
        return next(
            (
                block
                for block in self.env.blocks
                if block.label != ignore_label
                and block.position == cell
                and not block.delivered
                and not block.carrying
            ),
            None,
        )

    def _binding_failure(self):
        if self._consumed:
            return "bound_action_already_consumed"
        if int(self.env.time_steps) != self.bound_time_step:
            return "bound_action_not_started_at_decision_epoch"
        instance = getattr(self.env, "current_episode_instance", None)
        instance_id = None if instance is None else instance.instance_id
        if instance_id != self.bound_instance_id:
            return "bound_action_episode_changed"
        block = self._block()
        if block is None:
            return "bound_block_disappeared_or_identity_changed"
        if any(item.carrying for item in self.env.blocks):
            return "inventory_already_carried"
        if self.source_cell != tuple(self.env.pickup_cell):
            return "bound_block_was_not_at_pickup"
        if block.position != self.source_cell:
            return "bound_block_source_changed"
        if block.stored or block.delivered or block.carrying:
            return "bound_block_not_unstored_inbound"
        inbound = self._physical_inbound_blocks()
        if len(inbound) != 1 or inbound[0] is not block:
            return "inbound_block_not_unique"
        if self.bound_storage_location is not None:
            return "bound_block_was_already_assigned"
        if block.storage_location != self.bound_storage_location:
            return "bound_block_assignment_changed"
        if self.bound_stored_time_step is not None:
            return "bound_block_storage_clock_already_started"
        if block.stored_time_step != self.bound_stored_time_step:
            return "bound_block_storage_clock_changed"
        if (
            int(block.storage_steps_elapsed)
            != self.bound_storage_steps_elapsed
        ):
            return "bound_block_elapsed_clock_changed"
        if self.destination not in self.env.storage_positions:
            return "destination_not_storage_cell"
        if self._occupant(
            self.destination, ignore_label=self.target_label
        ) is not None:
            return "destination_occupied"
        return None

    def _strict_actions(self, block):
        """Plan against complete live occupancy, ignoring only ``block``."""

        if block.delivered or block.stored or block.position is None:
            return None
        if block.storage_location is not None:
            return None
        if self._occupant(
            self.destination, ignore_label=block.label
        ) is not None:
            return None

        if block.carrying:
            transport = self.env.plan_path_heuristic(
                self.env.current_state,
                self.destination,
                ignore_block=block,
            )
            if self.env.current_state != self.destination and not transport:
                return None
            return list(transport) + [self.env.ACTION_IDS["PUTDOWN"]]

        if block.position != self.source_cell:
            return None
        inbound = self._physical_inbound_blocks()
        if len(inbound) != 1 or inbound[0] is not block:
            return None
        approach = self.env.plan_path_heuristic(
            self.env.current_state,
            block.position,
            ignore_block=block,
        )
        if self.env.current_state != block.position and not approach:
            return None
        transport = self.env.plan_path_heuristic(
            block.position,
            self.destination,
            ignore_block=block,
        )
        if block.position != self.destination and not transport:
            return None
        return (
            list(approach)
            + [self.env.ACTION_IDS["PICKUP"]]
            + list(transport)
            + [self.env.ACTION_IDS["PUTDOWN"]]
        )

    def initiation(self, state):
        if self.started or self.failed:
            return False
        if self._binding_failure() is not None:
            return False
        actions = self._strict_actions(self._block())
        return bool(actions)

    def _fail(self, reason):
        if any(block.carrying for block in self.env.blocks):
            raise RuntimeError(
                "ExplicitAcceptOption may not fail while carrying inventory"
            )
        self.failed = True
        self.failure_reason = str(reason)
        self._actions.clear()

    def _start(self):
        self._reset_active()
        self.started = True
        reason = self._binding_failure()
        if reason is not None:
            self._fail(reason)
            return False
        actions = self._strict_actions(self._block())
        if not actions:
            self._fail("bound_inbound_route_unavailable")
            return False
        self._actions = deque(actions)
        return True

    def _action_is_valid(self, action):
        block = self._block()
        if block is None:
            return False
        name = self.env.ACTION_NAMES[action]
        if name in ("UP", "DOWN", "LEFT", "RIGHT"):
            intended = self.env._get_intended_cell(
                self.env.current_state, action
            )
            if intended == self.env.current_state:
                return False
            return self._occupant(
                intended, ignore_label=block.label
            ) is None
        if name == "PICKUP":
            inbound = self._physical_inbound_blocks()
            return bool(
                len(inbound) == 1
                and inbound[0] is block
                and block.position == self.env.current_state
                and block.storage_location is None
                and not any(item.carrying for item in self.env.blocks)
            )
        if name == "PUTDOWN":
            return bool(
                not block.stored
                and not block.delivered
                and block.carrying
                and block.position == self.env.current_state
                and self.env.current_state == self.destination
                and block.storage_location is None
                and self._occupant(
                    self.destination, ignore_label=block.label
                ) is None
            )
        return False

    def _replan(self):
        self.replan_count += 1
        block = self._block()
        if self.replan_count > self.max_replans:
            if block is not None and block.carrying:
                raise RuntimeError(
                    "ExplicitAcceptOption exceeded its replan bound while "
                    "carrying"
                )
            self._fail("explicit_accept_replan_bound_exceeded")
            return
        actions = None if block is None else self._strict_actions(block)
        if not actions:
            if block is not None and block.carrying:
                raise RuntimeError(
                    "ExplicitAcceptOption cannot reach its bound destination "
                    "while carrying inventory"
                )
            self._fail("explicit_accept_live_replan_failed")
            return
        self._actions = deque(actions)

    def policy(self, state, test=False):
        if not self.started and not self.failed:
            self._start()
        if self.failed:
            return self.env.ACTION_IDS["WAIT"]
        block = self._block()
        if block is None:
            self._fail("bound_block_disappeared_during_execution")
            return self.env.ACTION_IDS["WAIT"]
        if (
            block.stored
            and not block.carrying
            and block.position == self.destination
        ):
            return self.env.ACTION_IDS["WAIT"]
        if self.putdown_emitted_time_step is not None:
            raise RuntimeError(
                "ExplicitAcceptOption PUTDOWN must be consumed by env.step "
                "before policy is called again"
            )
        if not self._actions:
            self._replan()
        if self.failed or not self._actions:
            return self.env.ACTION_IDS["WAIT"]
        action = self._actions[0]
        if not self._action_is_valid(action):
            self._replan()
            if self.failed or not self._actions:
                return self.env.ACTION_IDS["WAIT"]
            action = self._actions[0]
            if not self._action_is_valid(action):
                raise RuntimeError(
                    "ExplicitAcceptOption produced an invalid action after "
                    "replanning"
                )
        self._actions.popleft()
        if action == self.env.ACTION_IDS["PUTDOWN"]:
            if block.stored_time_step is not None:
                raise RuntimeError(
                    "Inbound storage clock started before PUTDOWN"
                )
            self.putdown_emitted_time_step = int(self.env.time_steps)
            self.expected_stored_time_step = (
                self.putdown_emitted_time_step + 1
            )
            # The environment consumes this assignment in the immediately
            # following step and owns the initial storage event and clock.
            block.storage_location = self.destination
        self.actual_steps += 1
        return action

    def _outcome(self, *, success, reason, block):
        elapsed_after = (
            None if block is None else int(block.storage_steps_elapsed)
        )
        return {
            "success": bool(success),
            "reason": str(reason),
            "block_label": self.target_label,
            "source_cell": self.source_cell,
            "destination": self.destination,
            "bound_instance_id": self.bound_instance_id,
            "bound_time_step": self.bound_time_step,
            "actual_steps": self.actual_steps,
            "replans": self.replan_count,
            "bound_storage_location": self.bound_storage_location,
            "storage_location_after": (
                None if block is None else block.storage_location
            ),
            "stored_time_step_before": self.bound_stored_time_step,
            "stored_time_step_after": (
                None if block is None else block.stored_time_step
            ),
            "storage_steps_elapsed_before": (
                self.bound_storage_steps_elapsed
            ),
            "storage_steps_elapsed_after": elapsed_after,
            "storage_steps_elapsed_delta": (
                None
                if elapsed_after is None
                else elapsed_after - self.bound_storage_steps_elapsed
            ),
            "putdown_emitted_time_step": self.putdown_emitted_time_step,
            "expected_stored_time_step": self.expected_stored_time_step,
        }

    def _record(self, *, success, reason, block):
        if self._completion_recorded:
            return
        if success:
            self.success_count += 1
            self.episode_success_count += 1
        else:
            self.failure_count += 1
            self.episode_failure_count += 1
        self.episode_replans += self.replan_count
        self.last_outcome = self._outcome(
            success=success, reason=reason, block=block
        )
        self.episode_outcomes.append(dict(self.last_outcome))
        self._completion_recorded = True
        self._consumed = True
        self.started = False

    def termination(self, state):
        block = self._block()
        if (
            block is not None
            and block.stored
            and not block.delivered
            and not block.carrying
            and block.position == self.destination
            and block.storage_location == self.destination
        ):
            if self.expected_stored_time_step is None:
                self._fail("storage_completed_without_option_putdown")
            elif block.stored_time_step != self.expected_stored_time_step:
                self._fail("storage_clock_did_not_start_on_putdown")
            else:
                self._record(
                    success=True,
                    reason="target_stored_at_bound_cell",
                    block=block,
                )
                return True
        if self.failed:
            if any(item.carrying for item in self.env.blocks):
                raise RuntimeError(
                    "Failed ExplicitAcceptOption may not terminate while "
                    "carrying"
                )
            self._record(
                success=False,
                reason=self.failure_reason,
                block=block,
            )
            return True
        return False

    def on_env_reset(self):
        self._reset_active()
        self._consumed = True
        self.last_outcome = None
        self.episode_success_count = 0
        self.episode_failure_count = 0
        self.episode_replans = 0
        self.episode_outcomes.clear()

    def __str__(self):
        return (
            f"ExplicitAcceptOption({self.target_label}->{self.destination})"
        )

    __repr__ = __str__

    def __hash__(self):
        return hash((type(self), self.target_label, self.destination))

    def __eq__(self, other):
        return bool(
            isinstance(other, ExplicitAcceptOption)
            and self.target_label == other.target_label
            and self.destination == other.destination
        )


__all__ = ["ExplicitAcceptOption"]
