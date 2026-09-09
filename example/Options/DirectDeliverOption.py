"""Execute one explicitly bound, direct stored-block delivery."""

from __future__ import annotations

from collections import deque

from option import BaseOption
from example.Options.certified_path import actions_from_cell_path
from PSLAP.viability import RecoveryAction, RecoveryActionKind


class DirectDeliverOption(BaseOption):
    """Carry one stored block to one exact exit without relocating inventory.

    Construction binds the block, exit, episode, and current decision epoch.
    There is intentionally no dispatcher, relocation selector, urgency rule, or
    alternate-exit fallback.  The only permissible live repair is replanning
    the two canonical path legs while retaining the bound block and exit.
    """

    VERSION = "bound_direct_deliver_option_v2"
    BINDING_CONTRACT = "stored_block_exit_episode_epoch_bound_once_v1"
    PATH_CONTRACT = "certified_initial_path_fixed_obstacle_repair_v2"
    WITNESS_CONTRACT = "RecoveryAction.DELIVERY_paths_executed_v2"

    def __init__(
        self,
        env,
        block_label: str,
        exit_cell,
        *,
        max_replans: int = 8,
        certified_approach_path=None,
        certified_transport_path=None,
        fixed_obstacles=(),
    ):
        super().__init__(is_primitive=False)
        self.env = env
        self.target_label = str(block_label)
        try:
            exit_cell = tuple(exit_cell)
            row, column = exit_cell
        except (TypeError, ValueError) as exc:
            raise ValueError("exit_cell must be a two-dimensional cell") from exc
        self.exit_cell = (int(row), int(column))
        if self.exit_cell != exit_cell:
            raise ValueError("exit coordinates must be integral")
        self.max_replans = int(max_replans)
        if self.max_replans <= 0:
            raise ValueError("max_replans must be positive")
        if (certified_approach_path is None) != (
            certified_transport_path is None
        ):
            raise ValueError("both certified path legs must be supplied together")
        self.bound_fixed_obstacles = frozenset(
            tuple(cell) for cell in fixed_obstacles
        )

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
        target = self._bound_block
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
        self.recovery_witness_steps = None
        self.certified_approach_path = (
            None
            if certified_approach_path is None
            else tuple(tuple(cell) for cell in certified_approach_path)
        )
        self.certified_transport_path = (
            None
            if certified_transport_path is None
            else tuple(tuple(cell) for cell in certified_transport_path)
        )
        self.certified_path_bound = self.certified_approach_path is not None
        if self.certified_path_bound:
            if self.certified_approach_path[0] != tuple(env.current_state):
                raise ValueError("certified approach starts at a different agent cell")
            if self.certified_approach_path[-1] != self.source_cell:
                raise ValueError("certified approach does not end at source")
            if self.certified_transport_path[0] != self.source_cell:
                raise ValueError("certified transport does not start at source")
            if self.certified_transport_path[-1] != self.exit_cell:
                raise ValueError("certified transport does not end at exit")
            if any(
                cell in self.bound_fixed_obstacles
                for cell in self.certified_approach_path
                + self.certified_transport_path
            ):
                raise ValueError("certified delivery path crosses a fixed obstacle")
            # Validate adjacency and live walls at binding time.
            actions_from_cell_path(
                env,
                self.certified_approach_path,
                start=env.current_state,
                goal=self.source_cell,
            )
            actions_from_cell_path(
                env,
                self.certified_transport_path,
                start=self.source_cell,
                goal=self.exit_cell,
            )

        cell_id = f"{self.exit_cell[0]:02d}-{self.exit_cell[1]:02d}"
        self.controller_identifier = (
            f"option:DirectDeliver:{self.target_label}:exit-{cell_id}"
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

    @classmethod
    def from_recovery_action(
        cls,
        env,
        action: RecoveryAction,
        *,
        max_replans: int = 8,
        fixed_obstacles=(),
    ):
        """Bind exactly one certified direct-delivery search transition."""

        if not isinstance(action, RecoveryAction):
            raise TypeError("action must be a RecoveryAction")
        if action.kind is not RecoveryActionKind.DELIVERY:
            raise ValueError("DirectDeliverOption requires a DELIVERY witness")
        option = cls(
            env,
            action.block_label,
            action.destination,
            max_replans=max_replans,
            certified_approach_path=action.approach_path,
            certified_transport_path=action.transport_path,
            fixed_obstacles=fixed_obstacles,
        )
        if option.source_cell != tuple(action.source):
            raise ValueError("delivery witness source does not match live block")
        if not action.approach_path:
            raise ValueError("delivery witness has an empty approach path")
        if not action.transport_path:
            raise ValueError("delivery witness has an empty transport path")
        if tuple(action.approach_path[0]) != tuple(env.current_state):
            raise ValueError("delivery witness starts at a different agent cell")
        if tuple(action.approach_path[-1]) != option.source_cell:
            raise ValueError("delivery witness approach does not end at source")
        if tuple(action.transport_path[0]) != option.source_cell:
            raise ValueError("delivery witness transport does not start at source")
        if tuple(action.transport_path[-1]) != option.exit_cell:
            raise ValueError("delivery witness transport does not end at exit")
        option.recovery_witness_steps = int(action.steps)
        return option

    def _reset_active(self):
        self.started = False
        self.failed = False
        self.failure_reason = None
        self.actual_steps = 0
        self.initial_estimated_steps = None
        self.replan_count = 0
        self._actions = deque()
        self._completion_recorded = False

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
        if self.exit_cell not in self.env.exit_cells:
            return "bound_destination_not_exit"
        if self._occupant(
            self.exit_cell, ignore_label=self.target_label
        ) is not None:
            return "bound_exit_occupied"
        return None

    def _strict_actions(self, block):
        """Plan both legs while treating every other block as an obstacle."""

        if (
            self.certified_path_bound
            and self.actual_steps == 0
            and self.replan_count == 0
            and not block.carrying
        ):
            approach = actions_from_cell_path(
                self.env,
                self.certified_approach_path,
                start=self.env.current_state,
                goal=self.source_cell,
            )
            transport = actions_from_cell_path(
                self.env,
                self.certified_transport_path,
                start=self.source_cell,
                goal=self.exit_cell,
            )
            return (
                approach
                + [self.env.ACTION_IDS["PICKUP"]]
                + transport
                + [self.env.ACTION_IDS["PUTDOWN"]]
            )

        if block.carrying:
            if self._occupant(
                self.exit_cell, ignore_label=block.label
            ) is not None:
                return None
            transport = self.env.plan_path_heuristic(
                self.env.current_state,
                self.exit_cell,
                ignore_block=block,
                extra_blocked=self.bound_fixed_obstacles,
            )
            if self.env.current_state != self.exit_cell and not transport:
                return None
            return list(transport) + [self.env.ACTION_IDS["PUTDOWN"]]

        if not block.stored or block.delivered or block.position is None:
            return None
        if block.position != self.source_cell:
            return None
        if self._occupant(
            self.exit_cell, ignore_label=block.label
        ) is not None:
            return None
        approach = self.env.plan_path_heuristic(
            self.env.current_state,
            block.position,
            ignore_block=block,
            extra_blocked=self.bound_fixed_obstacles,
        )
        if self.env.current_state != block.position and not approach:
            return None
        transport = self.env.plan_path_heuristic(
            block.position,
            self.exit_cell,
            ignore_block=block,
            extra_blocked=self.bound_fixed_obstacles,
        )
        if block.position != self.exit_cell and not transport:
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
        return bool(self._strict_actions(self._block()))

    def _fail(self, reason):
        if any(block.carrying for block in self.env.blocks):
            raise RuntimeError(
                "DirectDeliverOption may not fail while carrying inventory"
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
            self._fail("bound_direct_delivery_route_unavailable")
            return False
        self.initial_estimated_steps = len(actions)
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
            if intended in self.bound_fixed_obstacles:
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
                and self.env.current_state == self.exit_cell
                and self._occupant(
                    self.exit_cell, ignore_label=block.label
                ) is None
            )
        return False

    def _replan(self):
        self.replan_count += 1
        block = self._block()
        if self.replan_count > self.max_replans:
            if block is not None and block.carrying:
                raise RuntimeError(
                    "DirectDeliverOption exceeded its replan bound while carrying"
                )
            self._fail("direct_delivery_replan_bound_exceeded")
            return
        actions = None if block is None else self._strict_actions(block)
        if not actions:
            if block is not None and block.carrying:
                raise RuntimeError(
                    "DirectDeliverOption cannot reach its bound exit while carrying"
                )
            self._fail("direct_delivery_live_replan_failed")
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
        if block.delivered:
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
                    "DirectDeliverOption produced an invalid action after replanning"
                )
        self._actions.popleft()
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
            "exit_cell": self.exit_cell,
            "bound_instance_id": self.bound_instance_id,
            "bound_time_step": self.bound_time_step,
            "actual_steps": self.actual_steps,
            "initial_estimated_steps": self.initial_estimated_steps,
            "replans": self.replan_count,
            "recovery_witness_steps": self.recovery_witness_steps,
            "certified_path_bound": self.certified_path_bound,
            "bound_fixed_obstacles": tuple(sorted(self.bound_fixed_obstacles)),
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
            "delivery_error_time": (
                None if block is None else block.delivery_error_time
            ),
            "delivered_time_step": (
                None if block is None else block.delivered_time_step
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
            and block.delivered
            and not block.carrying
            and block.position == self.exit_cell
        ):
            if block.stored_time_step != self.bound_stored_time_step:
                self._fail("storage_clock_changed_during_direct_delivery")
            else:
                self._record(
                    success=True,
                    reason="target_delivered_to_bound_exit",
                    block=block,
                )
                return True
        if self.failed:
            if any(item.carrying for item in self.env.blocks):
                raise RuntimeError(
                    "Failed DirectDeliverOption may not terminate while carrying"
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
        return f"DirectDeliverOption({self.target_label}->{self.exit_cell})"

    __repr__ = __str__

    def __hash__(self):
        return hash((type(self), self.target_label, self.exit_cell))

    def __eq__(self, other):
        return bool(
            isinstance(other, DirectDeliverOption)
            and self.target_label == other.target_label
            and self.exit_cell == other.exit_cell
        )


__all__ = ["DirectDeliverOption"]
