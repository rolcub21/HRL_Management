"""One policy-independent, explicitly bound yard reconfiguration macro."""

from __future__ import annotations

from collections import deque

from option import BaseOption


class ReconfigureOption(BaseOption):
    """Move one stored block to one explicitly supplied storage cell.

    The block, destination, episode instance, and decision epoch are bound when
    the option is constructed.  This class deliberately contains no assignment
    or relocation selector: a caller must first choose the parameterized action
    ``(block_label, destination)`` and then instantiate the option.

    Both the unladen approach and laden transport legs use the environment's
    canonical A* planner.  The moved block is ignored and every other live block
    remains an obstacle.  ``storage_location`` is changed only when the final
    PUTDOWN action is emitted, preserving the environment's relocation contract
    and the block's original storage clock.
    """

    VERSION = "bound_reconfigure_option_v1"
    BINDING_CONTRACT = "block_cell_episode_epoch_bound_once_v1"
    PATH_CONTRACT = "canonical_other_blocks_obstruct_target_ignored_v1"
    CLOCK_CONTRACT = "preserve_original_storage_event_clock_v1"

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
            raise ValueError("destination must be a two-dimensional cell") from exc
        self.destination = (int(row), int(column))
        if self.destination != tuple(destination):
            raise ValueError("destination coordinates must be integral")
        self.max_replans = int(max_replans)
        if self.max_replans <= 0:
            raise ValueError("max_replans must be positive")

        target = self._block()
        if target is None:
            raise ValueError(f"unknown block label: {self.target_label}")
        self.source_cell = (
            None if target.position is None else tuple(target.position)
        )
        self.bound_storage_location = (
            None
            if target.storage_location is None
            else tuple(target.storage_location)
        )
        self.bound_stored_time_step = target.stored_time_step
        self.bound_storage_steps_elapsed = int(target.storage_steps_elapsed)
        self.bound_time_step = int(env.time_steps)
        instance = getattr(env, "current_episode_instance", None)
        self.bound_instance_id = (
            None if instance is None else instance.instance_id
        )

        cell_id = f"{self.destination[0]:02d}-{self.destination[1]:02d}"
        self.controller_identifier = (
            f"option:Reconfigure:{self.target_label}:cell-{cell_id}"
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

    def _block(self):
        return next(
            (
                block
                for block in self.env.blocks
                if block.label == self.target_label
            ),
            None,
        )

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
            return "bound_block_disappeared"
        if any(item.carrying for item in self.env.blocks):
            return "inventory_already_carried"
        if not block.stored or block.delivered or block.carrying:
            return "bound_block_not_stored"
        if block.position != self.source_cell:
            return "bound_block_source_changed"
        if block.storage_location != self.bound_storage_location:
            return "bound_block_assignment_changed"
        if self.bound_storage_location != self.source_cell:
            return "bound_block_storage_invariant_invalid"
        if block.stored_time_step != self.bound_stored_time_step:
            return "bound_block_storage_clock_changed"
        if self.destination not in self.env.storage_positions:
            return "destination_not_storage_cell"
        if self.destination == self.source_cell:
            return "destination_equals_source"
        if self._occupant(
            self.destination, ignore_label=self.target_label
        ) is not None:
            return "destination_occupied"
        return None

    def _strict_actions(self, block):
        """Plan against complete live occupancy, ignoring only ``block``."""

        if block.carrying:
            if self._occupant(
                self.destination, ignore_label=block.label
            ) is not None:
                return None
            transport = self.env.plan_path_heuristic(
                self.env.current_state,
                self.destination,
                ignore_block=block,
            )
            if self.env.current_state != self.destination and not transport:
                return None
            return list(transport) + [self.env.ACTION_IDS["PUTDOWN"]]

        if not block.stored or block.delivered or block.position is None:
            return None
        if block.position != self.source_cell:
            return None
        if self._occupant(
            self.destination, ignore_label=block.label
        ) is not None:
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
                "ReconfigureOption may not fail while carrying inventory"
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
            self._fail("bound_reconfiguration_route_unavailable")
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
            return bool(
                block.stored
                and not block.delivered
                and not block.carrying
                and block.position == self.env.current_state
                and not any(item.carrying for item in self.env.blocks)
            )
        if name == "PUTDOWN":
            return bool(
                block.stored
                and not block.delivered
                and block.carrying
                and block.position == self.env.current_state
                and self.env.current_state == self.destination
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
                    "ReconfigureOption exceeded its replan bound while carrying"
                )
            self._fail("reconfigure_replan_bound_exceeded")
            return
        actions = None if block is None else self._strict_actions(block)
        if not actions:
            if block is not None and block.carrying:
                raise RuntimeError(
                    "ReconfigureOption cannot reach its bound destination "
                    "while carrying inventory"
                )
            self._fail("reconfigure_live_replan_failed")
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
                    "ReconfigureOption produced an invalid action after replanning"
                )
        self._actions.popleft()
        if action == self.env.ACTION_IDS["PUTDOWN"]:
            # This is the sole mutation performed by planning/execution code.
            # The environment consumes it in the immediately following step.
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
            "bound_time_step": self.bound_time_step,
            "actual_steps": self.actual_steps,
            "replans": self.replan_count,
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
            if block.stored_time_step != self.bound_stored_time_step:
                self._fail("storage_clock_changed_during_reconfiguration")
            else:
                self._record(
                    success=True,
                    reason="target_reconfigured",
                    block=block,
                )
                return True
        if self.failed:
            if any(item.carrying for item in self.env.blocks):
                raise RuntimeError(
                    "Failed ReconfigureOption may not terminate while carrying"
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
            f"ReconfigureOption({self.target_label}->{self.destination})"
        )

    __repr__ = __str__

    def __hash__(self):
        return hash((type(self), self.target_label, self.destination))

    def __eq__(self, other):
        return bool(
            isinstance(other, ReconfigureOption)
            and self.target_label == other.target_label
            and self.destination == other.destination
        )


__all__ = ["ReconfigureOption"]
