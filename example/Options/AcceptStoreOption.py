"""Atomic inbound placement through a strict previewable assignment source."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter

from option import BaseOption
from example.Options.selector_v5 import StorageAssignmentPreview
from PSLAP.dynamic_yard import BlockView, YardSnapshot, shortest_clear_path
from PSLAP.track_a import shared_candidate_mask


@dataclass(frozen=True)
class AcceptStoreDurationEstimate:
    """Noncommitting current-epoch estimate of one atomic placement."""

    contract: str
    block_label: str
    time_step: int
    preview_storage_cell: tuple[int, int]
    preview_candidate_count: int
    preview_candidate_mask_id: str
    pickup_steps: int
    storage_steps: int
    total_steps: int
    preview_seconds: float
    preview_proposal: StorageAssignmentPreview


class AcceptStoreOption(BaseOption):
    """Assign, pick up, and store one arrived block as one SMDP action."""

    VERSION = "accept_store_option_v1"
    controller_identifier = "option:AcceptStore:v1"

    def __init__(self, env, selector, *, max_replans=8):
        super().__init__(is_primitive=False)
        if selector.env is not env:
            raise ValueError("AcceptStoreOption and selector must share an env")
        self.env = env
        self.selector = selector
        self.max_replans = int(max_replans)
        if self.max_replans <= 0:
            raise ValueError("max_replans must be positive")
        self.success_count = 0
        self.failure_count = 0
        self.episode_success_count = 0
        self.episode_failure_count = 0
        self.episode_outcomes = []
        self.last_outcome = None
        self._reset_active()

    def _reset_active(self):
        self.started = False
        self.failed = False
        self.failure_reason = None
        self.target_label = None
        self.phase = None
        self._actions = deque()
        self.actual_steps = 0
        self.replan_count = 0
        self._completion_recorded = False

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

    def _target(self):
        return next(
            (
                block
                for block in self.env.blocks
                if block.label == self.target_label
            ),
            None,
        )

    def _physical_candidates(self, block):
        yard = YardSnapshot.from_env(self.env)
        view = BlockView(
            block.label,
            block.position,
            float(block.get_remaining_storage_time()),
        )
        return shared_candidate_mask(yard, view, block.position)

    def initiation(self, state):
        if any(block.carrying for block in self.env.blocks):
            return False
        block = self._inbound_block()
        if block is None:
            return False
        if block.storage_location is not None:
            return False
        if not self._physical_candidates(block):
            return False
        actions = self.env.plan_path_heuristic(
            self.env.current_state,
            block.position,
            ignore_block=block,
        )
        return bool(self.env.current_state == block.position or actions)

    def estimate_duration(self):
        """Estimate full placement duration without changing live state.

        The selected assignment source is previewed at the current scheduler
        epoch. AcceptStore v1 makes its actual assignment after pickup, so this
        is an audited estimate rather than a promise that the later cell is
        equal.
        """

        if any(block.carrying for block in self.env.blocks):
            return None
        block = self._inbound_block()
        if block is None or block.storage_location is not None:
            return None
        preview_started = perf_counter()
        preview = self.selector.preview_assignment(
            block, source_cell=block.position
        )
        preview_seconds = perf_counter() - preview_started
        if preview is None:
            return None
        return self.estimate_duration_for_preview(
            preview, preview_seconds=preview_seconds
        )

    def estimate_duration_for_preview(
        self,
        preview: StorageAssignmentPreview,
        *,
        preview_seconds: float = 0.0,
    ):
        """Estimate one exact proposal without invoking an assignment scorer.

        This is the parameterized-action counterpart of ``estimate_duration``:
        the caller chooses a cell first, then this method validates and costs
        that same proposal. It is side-effect free and suitable for enumerating
        candidate descriptors before binding only the selected one.
        """

        if not isinstance(preview, StorageAssignmentPreview):
            raise ValueError("duration estimate requires a storage proposal")
        if any(block.carrying for block in self.env.blocks):
            return None
        block = self._inbound_block()
        if block is None or block.storage_location is not None:
            return None
        if preview.block_label != block.label:
            raise ValueError("duration proposal does not match inbound block")
        validator = getattr(self.selector, "validate_preview", None)
        if validator is None:
            raise RuntimeError("assignment source cannot validate proposals")
        validator(block, preview, require_current_epoch=True)
        pickup_actions = self.env.plan_path_heuristic(
            self.env.current_state,
            block.position,
            ignore_block=block,
        )
        if self.env.current_state != block.position and not pickup_actions:
            return None
        yard = YardSnapshot.from_env(self.env)
        storage_path = shortest_clear_path(
            yard,
            block.position,
            preview.chosen_cell,
            ignore_labels=(block.label,),
        )
        if storage_path is None:
            return None
        pickup_steps = len(pickup_actions) + 1
        storage_steps = len(storage_path) - 1 + 1
        return AcceptStoreDurationEstimate(
            contract=(
                "current_epoch_assignment_source_preview_stored_yard_"
                "duration_estimate_v2"
            ),
            block_label=block.label,
            time_step=int(self.env.time_steps),
            preview_storage_cell=preview.chosen_cell,
            preview_candidate_count=preview.candidate_count,
            preview_candidate_mask_id=preview.candidate_mask_id,
            pickup_steps=pickup_steps,
            storage_steps=storage_steps,
            total_steps=pickup_steps + storage_steps,
            preview_seconds=float(preview_seconds),
            preview_proposal=preview,
        )

    def _fail(self, reason):
        if any(block.carrying for block in self.env.blocks):
            raise RuntimeError(
                "AcceptStoreOption may not fail while carrying inventory"
            )
        self.failed = True
        self.failure_reason = str(reason)
        self._actions.clear()

    def _build_live_actions(self):
        block = self._target()
        if block is None or block.delivered:
            return None
        if block.stored and not block.carrying:
            return []
        if self.phase == "pickup":
            if block.carrying:
                return []
            path = self.env.plan_path_heuristic(
                self.env.current_state,
                block.position,
                ignore_block=block,
            )
            if self.env.current_state != block.position and not path:
                return None
            return list(path) + [self.env.ACTION_IDS["PICKUP"]]
        target = block.storage_location
        if self.phase == "store" and target is not None and block.carrying:
            path = self.env.plan_path_heuristic(
                self.env.current_state,
                target,
                ignore_block=block,
            )
            if self.env.current_state != target and not path:
                return None
            return list(path) + [self.env.ACTION_IDS["PUTDOWN"]]
        return None

    def _start(self):
        self._reset_active()
        block = self._inbound_block()
        if block is None:
            self._fail("inbound_block_disappeared")
            return False
        self.target_label = block.label
        self.phase = "pickup"
        actions = self._build_live_actions()
        if actions is None:
            self._fail("inbound_route_failed")
            return False
        self._actions = deque(actions)
        self.started = True
        return True

    def _occupant(self, position, target):
        return next(
            (
                block
                for block in self.env.blocks
                if block is not target
                and block.position == position
                and not block.delivered
                and not block.carrying
            ),
            None,
        )

    def _action_is_valid(self, action):
        block = self._target()
        if block is None:
            return False
        name = self.env.ACTION_NAMES[action]
        if name in ("UP", "DOWN", "LEFT", "RIGHT"):
            intended = self.env._get_intended_cell(
                self.env.current_state, action
            )
            if intended == self.env.current_state:
                return False
            return self._occupant(intended, block) is None
        if name == "PICKUP":
            return bool(
                block.position == self.env.current_state
                and not block.carrying
                and not any(item.carrying for item in self.env.blocks)
            )
        if name == "PUTDOWN":
            return bool(
                block.carrying
                and block.position == self.env.current_state
                and self.env.current_state == block.storage_location
                and self._occupant(block.storage_location, block) is None
            )
        return False

    def _replan(self):
        self.replan_count += 1
        if self.replan_count > self.max_replans:
            self._fail("inbound_replan_bound_exceeded")
            return
        actions = self._build_live_actions()
        if actions is None:
            self._fail("inbound_live_replan_failed")
            return
        self._actions = deque(actions)

    def policy(self, state, test=False):
        if not self.started and not self.failed:
            self._start()
        if self.failed:
            return self.env.ACTION_IDS["WAIT"]
        block = self._target()
        if self.phase == "assign":
            if block is None or not block.carrying:
                raise RuntimeError(
                    "AcceptStoreOption lost its target before assignment"
                )
            if block.storage_location is not None:
                raise RuntimeError("Inbound target was assigned more than once")
            chosen = self.selector.assign_block(
                block, source_cell=self.env.current_state
            )
            if chosen is None:
                raise RuntimeError(
                    "Frozen selector failed after atomic inbound pickup"
                )
            # Assignment is a zero-duration controller operation.  Continue
            # into storage in this same policy call so the learned method does
            # not pay an artificial WAIT step that Track-A assignment sources
            # do not incur.
            self.phase = "store"
            actions = self._build_live_actions()
            if actions is None:
                raise RuntimeError(
                    "No safe route to the selector's committed storage cell"
                )
            self._actions = deque(actions)
        if block is not None and block.stored and not block.carrying:
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
                    "AcceptStoreOption produced an invalid action after replanning"
                )
        self._actions.popleft()
        self.actual_steps += 1
        return action

    def termination(self, state):
        block = self._target()
        if (
            block is not None
            and self.phase == "pickup"
            and block.carrying
        ):
            self.phase = "assign"
            self._actions.clear()
            return False
        if block is not None and block.stored and not block.carrying:
            if not self._completion_recorded:
                self.success_count += 1
                self.episode_success_count += 1
                self.last_outcome = {
                    "success": True,
                    "reason": "target_stored",
                    "block_label": block.label,
                    "actual_steps": self.actual_steps,
                    "replans": self.replan_count,
                    "storage_location": block.storage_location,
                }
                self.episode_outcomes.append(dict(self.last_outcome))
                self._completion_recorded = True
            self.started = False
            return True
        if self.failed:
            if any(item.carrying for item in self.env.blocks):
                raise RuntimeError(
                    "Failed AcceptStoreOption may not terminate while carrying"
                )
            self.failure_count += 1
            self.episode_failure_count += 1
            self.last_outcome = {
                "success": False,
                "reason": self.failure_reason,
                "block_label": self.target_label,
                "actual_steps": self.actual_steps,
                "replans": self.replan_count,
                "storage_location": (
                    block.storage_location if block is not None else None
                ),
            }
            self.episode_outcomes.append(dict(self.last_outcome))
            self.started = False
            return True
        return False

    def on_env_reset(self):
        self._reset_active()
        self.last_outcome = None
        self.episode_success_count = 0
        self.episode_failure_count = 0
        self.episode_outcomes.clear()
        self.selector.on_env_reset()

    def __str__(self):
        return "AcceptStoreOption"

    __repr__ = __str__

    def __hash__(self):
        return hash(type(self))


class ReservedAcceptStoreOption(AcceptStoreOption):
    """Execute exactly one decision-epoch assignment proposal."""

    VERSION = "accept_store_reserved_cell_v2"
    controller_identifier = "option:AcceptStore:v2"
    COMMITMENT_CONTRACT = "decision_epoch_proposal_bound_once_v2"
    RESERVATION_CONTRACT = "bound_cell_parameterized_accept_store_v1"

    def __init__(self, env, selector, *, max_replans=8):
        super().__init__(env, selector, max_replans=max_replans)
        self.selector.deployment_assignment_contract = self.COMMITMENT_CONTRACT
        self._reservation = None
        self.episode_reservation_bound_count = 0
        self.episode_reservation_commit_count = 0
        self.episode_reservation_invalidation_count = 0
        self.episode_reservation_execution_match_count = 0
        self.episode_bound_proposal_ids = []
        self.episode_committed_proposal_ids = []

    @property
    def active_reservation(self):
        return self._reservation

    def bind_estimate(self, estimate: AcceptStoreDurationEstimate):
        """Bind a side-effect-free proposal to the next atomic placement."""

        if self.started or self._reservation is not None:
            raise RuntimeError("AcceptStore already has an active reservation")
        if not isinstance(estimate, AcceptStoreDurationEstimate):
            raise ValueError("reserved AcceptStore requires a duration estimate")
        block = self._inbound_block()
        if block is None:
            raise RuntimeError("reserved inbound block disappeared before binding")
        preview = estimate.preview_proposal
        if estimate.block_label != block.label:
            raise RuntimeError("duration estimate block does not match inbound block")
        if tuple(estimate.preview_storage_cell) != tuple(preview.chosen_cell):
            raise RuntimeError("duration estimate cell does not match proposal")
        if estimate.preview_candidate_mask_id != preview.candidate_mask_id:
            raise RuntimeError("duration estimate mask does not match proposal")
        bind_preview = getattr(self.selector, "bind_preview", None)
        if bind_preview is None:
            self.selector.validate_preview(
                block, preview, require_current_epoch=True
            )
        else:
            bind_preview(block, preview)
        self._reservation = estimate
        self.episode_reservation_bound_count += 1
        self.episode_bound_proposal_ids.append(preview.proposal_id)
        return preview

    def _start(self):
        reservation = self._reservation
        self._reset_active()
        if reservation is None:
            self._fail("missing_reserved_assignment")
            return False
        block = self._inbound_block()
        if block is None:
            self._fail("reserved_inbound_block_disappeared")
            return False
        preview = reservation.preview_proposal
        if int(self.env.time_steps) != int(preview.time_step):
            self.episode_reservation_invalidation_count += 1
            self._fail("reserved_assignment_not_started_at_bound_epoch")
            return False
        try:
            self.selector.validate_preview(
                block, preview, require_current_epoch=True
            )
        except ValueError as exc:
            self.episode_reservation_invalidation_count += 1
            self._fail(f"reserved_assignment_invalid_before_pickup:{exc}")
            return False
        self.target_label = block.label
        self.phase = "pickup"
        actions = self._build_live_actions()
        if actions is None:
            self._fail("reserved_inbound_route_failed")
            return False
        self._actions = deque(actions)
        self.started = True
        return True

    def policy(self, state, test=False):
        if self.phase == "reserved_commit":
            block = self._target()
            if block is None or not block.carrying:
                raise RuntimeError(
                    "ReservedAcceptStoreOption lost its carried target"
                )
            reservation = self._reservation
            if reservation is None:
                raise RuntimeError("ReservedAcceptStoreOption lost its proposal")
            chosen = self.selector.commit_preview(
                block, reservation.preview_proposal
            )
            if chosen is None:
                self.episode_reservation_invalidation_count += 1
                raise RuntimeError(
                    "Reserved assignment became invalid after atomic pickup"
                )
            if tuple(chosen) != tuple(reservation.preview_storage_cell):
                raise RuntimeError("Reserved assignment cell was substituted")
            self.episode_reservation_commit_count += 1
            self.episode_committed_proposal_ids.append(
                reservation.preview_proposal.proposal_id
            )
            self.phase = "store"
            actions = self._build_live_actions()
            if actions is None:
                raise RuntimeError("No safe route to the reserved storage cell")
            self._actions = deque(actions)
        return super().policy(state, test=test)

    def _reservation_metadata(self, block):
        reservation = self._reservation
        if reservation is None:
            return {
                "assignment_commitment_contract": self.COMMITMENT_CONTRACT,
                "reservation_contract": self.RESERVATION_CONTRACT,
                "proposal_id": None,
                "reserved_storage_cell": None,
                "proposal_candidate_mask_id": None,
                "proposal_time_step": None,
                "reservation_execution_match": False,
            }
        preview = reservation.preview_proposal
        executed = None if block is None else block.storage_location
        match = bool(
            executed is not None
            and tuple(executed) == tuple(preview.chosen_cell)
        )
        return {
            "assignment_commitment_contract": self.COMMITMENT_CONTRACT,
            "reservation_contract": self.RESERVATION_CONTRACT,
            "proposal_id": preview.proposal_id,
            "reserved_storage_cell": preview.chosen_cell,
            "proposal_candidate_mask_id": preview.candidate_mask_id,
            "proposal_time_step": preview.time_step,
            "reservation_execution_match": match,
        }

    def termination(self, state):
        block = self._target()
        if block is not None and self.phase == "pickup" and block.carrying:
            self.phase = "reserved_commit"
            self._actions.clear()
            return False

        terminated = super().termination(state)
        if not terminated:
            return False
        metadata = self._reservation_metadata(block)
        if self.last_outcome is not None:
            self.last_outcome.update(metadata)
            if self.last_outcome.get("success") and metadata[
                "reservation_execution_match"
            ]:
                self.episode_reservation_execution_match_count += 1
            if self.episode_outcomes:
                self.episode_outcomes[-1] = dict(self.last_outcome)
        self._reservation = None
        return True

    def on_env_reset(self):
        super().on_env_reset()
        self._reservation = None
        self.episode_reservation_bound_count = 0
        self.episode_reservation_commit_count = 0
        self.episode_reservation_invalidation_count = 0
        self.episode_reservation_execution_match_count = 0
        self.episode_bound_proposal_ids.clear()
        self.episode_committed_proposal_ids.clear()

    def __str__(self):
        return "ReservedAcceptStoreOption"

    __repr__ = __str__


__all__ = [
    "AcceptStoreDurationEstimate",
    "AcceptStoreOption",
    "ReservedAcceptStoreOption",
]
