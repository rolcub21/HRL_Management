"""Track B option adapter for the cardinality-invariant REG-v5 selector."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
from time import perf_counter

import torch

from option import BaseOption
from example.small_rooms_env import SmallRoomsEnv
from PSLAP.dynamic_yard import BlockView, YardSnapshot
from PSLAP.ga_policy import (
    DEFAULT_ROLLING_GA_CONFIG,
    CompleteSingletonDurationAwareRollingGAAssigner,
    DurationAwareRollingGAStorageAssigner,
    OperationalRollingGAStorageAssigner,
    RollingGAStorageAssigner,
)
from PSLAP.ga_capacity_aware import (
    CapacityAwareCompleteRollingGAAssigner,
    CapacityAwareDurationAwareRollingGAAssigner,
    CapacityAwareOperationalRollingGAAssigner,
    CapacityAwareRollingGAStorageAssigner,
)
from PSLAP.kim2020_a3c_spatial import (
    ARCHITECTURE_NAME as KIM2020_ARCHITECTURE_NAME,
    DEPLOYMENT_STOCHASTIC as KIM2020_DEPLOYMENT_STOCHASTIC,
    FEATURE_VERSION as KIM2020_FEATURE_VERSION,
    Kim2020A3CSpatialSource,
)
from PSLAP.reg_selector_v5 import (
    ARCHITECTURE_NAME,
    FEATURE_VERSION,
    REGV5AssignmentSource,
)
from PSLAP.track_a import (
    DYNAMIC_ASSIGNMENT_SOURCE_VERSION,
    NEAREST_FREE_ASSIGNMENT_SOURCE_VERSION,
    DynamicAssignmentSource,
    NearestFreeAssignmentSource,
    TRACK_A_DYNAMIC,
    TRACK_A_GA_ROLLING,
    TRACK_A_GA_ROLLING_DURATION_AWARE,
    TRACK_A_GA_ROLLING_OPERATIONAL,
    TRACK_A_GA_ROLLING_COMPLETE,
    TRACK_A_GA_ROLLING_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_DURATION_AWARE_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_OPERATIONAL_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_COMPLETE_CAPACITY_AWARE,
    TRACK_A_NEAREST_FREE,
    TRACK_A_KIM2020_A3C_SPATIAL,
    TRACK_A_REG_SELECTOR_V5,
    shared_candidate_mask,
)


TRACK_B_ASSIGNMENT_SOURCES = (
    TRACK_A_REG_SELECTOR_V5,
    TRACK_A_NEAREST_FREE,
    TRACK_A_DYNAMIC,
    TRACK_A_GA_ROLLING,
    TRACK_A_GA_ROLLING_DURATION_AWARE,
    TRACK_A_GA_ROLLING_OPERATIONAL,
    TRACK_A_GA_ROLLING_COMPLETE,
)
# The capacity-aware sources require the decision-epoch reservation ABI: the
# mandatory current block must still be the unique observable pickup head when
# the GA plan is formed.  Keep them out of the legacy post-pickup source list so
# generic callers cannot silently change that method definition.
TRACK_B_RESERVED_CAPACITY_AWARE_ASSIGNMENT_SOURCES = (
    TRACK_A_GA_ROLLING_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_DURATION_AWARE_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_OPERATIONAL_CAPACITY_AWARE,
    TRACK_A_GA_ROLLING_COMPLETE_CAPACITY_AWARE,
)
TRACK_B_ALL_ASSIGNMENT_SOURCES = (
    *TRACK_B_ASSIGNMENT_SOURCES,
    *TRACK_B_RESERVED_CAPACITY_AWARE_ASSIGNMENT_SOURCES,
)

JOINT_LEARNED_ASSIGNMENT_SOURCE = "joint_learned_parameterized"
JOINT_LEARNED_ASSIGNMENT_SOURCE_VERSION = (
    "explicit_cell_proposal_registry_v1"
)


def _candidate_mask_id(candidates) -> str:
    return hashlib.sha256(
        json.dumps(candidates, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _normalise_proposal(value):
    if value is None:
        return None, "no_proposal"
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or not isinstance(value[0], Integral)
        or not isinstance(value[1], Integral)
    ):
        return None, "malformed_cell"
    return (int(value[0]), int(value[1])), None


@dataclass(frozen=True)
class StorageAssignmentPreview:
    """Read-only frozen-selector proposal at the current decision epoch."""

    contract: str
    proposal_id: str
    instance_id: str
    assignment_source: str
    assignment_source_version: str
    block_label: str
    time_step: int
    source_cell: tuple[int, int]
    candidate_count: int
    candidate_mask_id: str
    chosen_cell: tuple[int, int]


def _storage_proposal_id(
    *,
    instance_id: str,
    assignment_source: str,
    assignment_source_version: str,
    block_label: str,
    time_step: int,
    source_cell: tuple[int, int],
    candidate_mask_id: str,
    chosen_cell: tuple[int, int],
) -> str:
    payload = {
        "instance_id": instance_id,
        "assignment_source": assignment_source,
        "assignment_source_version": assignment_source_version,
        "block_label": block_label,
        "time_step": int(time_step),
        "source_cell": tuple(source_cell),
        "candidate_mask_id": candidate_mask_id,
        "chosen_cell": tuple(chosen_cell),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:20]


class StorageSelectOptionV5(BaseOption):
    """Expose a REG-v5 assignment source through the HRL option contract.

    The adapter changes only the assignment mechanism.  The manager continues
    to see the historical ``StorageSelectOption`` output index, and the option
    still consumes one WAIT transition after committing its cell.  A frozen
    Track A checkpoint can therefore be inserted without changing the learned
    manager/primitive head dimensions or their controller-index ABI.
    """

    FEATURE_VERSION = FEATURE_VERSION
    ARCHITECTURE_NAME = ARCHITECTURE_NAME
    controller_identifier = "option:StorageSelectOption"
    is_storage_selector = True
    return_definition = "assignment_epoch_double_dqn_v5"
    requires_agent_gamma_match = False
    accepts_episode_outcome = True

    def __init__(self, env: SmallRoomsEnv, source: REGV5AssignmentSource):
        super().__init__(is_primitive=False)
        if source.env is not env:
            raise ValueError("REG-v5 source and option must share one environment")
        if source.feature_version != self.FEATURE_VERSION:
            raise ValueError("StorageSelectOptionV5 requires a REG-v5 source")
        self.env = env
        self.source = source
        self.assignment_source = TRACK_A_REG_SELECTOR_V5
        self.assignment_source_family = "learned_assignment_scorer"
        self.assignment_source_version = ARCHITECTURE_NAME
        self.information_regime = "online_arrived_only"
        self.preview_contract = (
            "current_epoch_frozen_reg_v5_greedy_preview_v1"
        )
        self.assignment_contract = (
            "post_pickup_frozen_reg_v5_assignment_v1"
        )
        self.assignment_source_learned = True
        self.block = None
        self.decision_count = 0
        self.infeasible_epoch_count = 0
        self.invalid_assignment_count = 0
        self.assignment_seconds = 0.0
        self.assignment_total_seconds = 0.0
        self.decisions: list[dict] = []
        self._committed_proposal_ids: set[str] = set()
        self.deployment_assignment_contract = self.assignment_contract

    @classmethod
    def from_checkpoint(
        cls,
        env: SmallRoomsEnv,
        checkpoint: str | Path | dict,
        *,
        device: str | torch.device = "cpu",
        seed: int = 0,
        learning_enabled: bool = False,
    ) -> "StorageSelectOptionV5":
        payload = (
            torch.load(checkpoint, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, (str, Path))
            else checkpoint
        )
        source = REGV5AssignmentSource.from_checkpoint(
            env,
            payload,
            learning_enabled=learning_enabled,
            device=device,
            seed=seed,
        )
        return cls(env, source)

    @property
    def gamma(self) -> float:
        config = getattr(self.source, "config", None)
        return float(getattr(config, "gamma", 1.0))

    @property
    def learning_enabled(self) -> bool:
        return bool(getattr(self.source, "learning_enabled", False))

    def set_learning_enabled(self, enabled: bool) -> None:
        setter = getattr(self.source, "set_learning_enabled", None)
        if setter is None:
            if enabled:
                raise RuntimeError(
                    "deterministic assignment sources cannot enable learning"
                )
            return
        setter(enabled)

    def on_env_reset(self) -> None:
        self.block = None
        self._committed_proposal_ids.clear()
        callback = getattr(self.source, "on_episode_start", None)
        if callback is not None:
            callback()

    def initiation(self, _state) -> bool:
        self.block = next(
            (
                block
                for block in self.env.blocks
                if block.carrying
                and not block.stored
                and not block.delivered
                and block.storage_location is None
            ),
            None,
        )
        return self.block is not None

    def policy(self, _state, test: bool = False):
        block = self.block
        if block is None:
            return SmallRoomsEnv.ACTION_IDS["WAIT"]

        self.assign_block(block, source_cell=self.env.current_state)
        return SmallRoomsEnv.ACTION_IDS["WAIT"]

    def assign_block(self, block, *, source_cell=None):
        """Commit one audited assignment without consuming an env action.

        The legacy option adapter calls this method and still emits its
        historical WAIT transition.  Atomic inbound controllers call the same
        assignment contract between pickup and the first storage action,
        matching the zero-duration decision interface used by Track A.
        """

        if source_cell is None:
            source_cell = self.env.current_state
        source_cell = tuple(source_cell)
        if (
            block is None
            or block.delivered
            or block.stored
            or block.position is None
        ):
            raise ValueError("selector assignment requires an observed inbound block")

        call_started = perf_counter()
        yard = YardSnapshot.from_env(self.env)
        view = BlockView(
            label=block.label,
            position=source_cell,
            remaining_time=float(block.get_remaining_storage_time()),
        )
        candidates = shared_candidate_mask(yard, view, source_cell)
        decision = {
            "decision_index": self.decision_count,
            "time_step": int(self.env.time_steps),
            "block_label": block.label,
            "candidate_count": len(candidates),
            "chosen_cell": None,
            "proposal_repr": None,
            "valid": False,
            "reason": None,
            "assignment_source": self.assignment_source,
            "assignment_source_version": self.assignment_source_version,
            "candidate_mask_id": _candidate_mask_id(candidates),
            "selection_time_step": int(self.env.time_steps),
            "commit_time_step": int(self.env.time_steps),
            "proposal_id": None,
            "commitment_contract": "post_pickup_recompute_v1",
        }
        self.decision_count += 1

        if not candidates:
            self.infeasible_epoch_count += 1
            decision["reason"] = "no_feasible_candidates"
            self.decisions.append(decision)
            self.assignment_total_seconds += perf_counter() - call_started
            return None

        started = perf_counter()
        proposal = self.source.propose(yard, view, source_cell, candidates)
        self.assignment_seconds += perf_counter() - started
        decision["proposal_repr"] = repr(proposal)
        chosen, reason = _normalise_proposal(proposal)
        if reason is None and chosen not in candidates:
            reason = "outside_shared_candidate_mask"
        if reason is not None:
            self.invalid_assignment_count += 1
            decision["reason"] = reason
            decision["chosen_cell"] = chosen
            self.decisions.append(decision)
            self.assignment_total_seconds += perf_counter() - call_started
            return None

        block.storage_location = chosen
        decision["chosen_cell"] = chosen
        decision["valid"] = True
        self.decisions.append(decision)
        self.env.store_events.append(
            {
                "episode": self.env.current_episode,
                "t_step": self.env.time_steps,
                "phase": "chosen",
                "selector": self.assignment_source_version,
                "assignment_source": self.assignment_source,
                "block_label": block.label,
                "row": chosen[0],
                "col": chosen[1],
                "candidate_count": len(candidates),
            }
        )
        self.assignment_total_seconds += perf_counter() - call_started
        return chosen

    def _validate_preview(self, block, preview, *, require_current_epoch):
        if not isinstance(preview, StorageAssignmentPreview):
            return (), "malformed_preview_token"
        instance = getattr(self.env, "current_episode_instance", None)
        instance_id = getattr(instance, "instance_id", None)
        if instance_id is None or preview.instance_id != instance_id:
            return (), "preview_instance_mismatch"
        if preview.assignment_source != self.assignment_source:
            return (), "preview_assignment_source_mismatch"
        if preview.assignment_source_version != self.assignment_source_version:
            return (), "preview_assignment_source_version_mismatch"
        if block is None or preview.block_label != getattr(block, "label", None):
            return (), "preview_block_mismatch"
        if require_current_epoch and preview.time_step != int(self.env.time_steps):
            return (), "stale_preview_epoch"
        if block.delivered or block.stored or block.position is None:
            return (), "preview_block_not_assignable"
        if block.storage_location is not None:
            return (), "preview_block_already_assigned"
        source_cell = tuple(preview.source_cell)
        if tuple(block.position) != source_cell:
            return (), "preview_source_cell_mismatch"
        yard = YardSnapshot.from_env(self.env)
        view = BlockView(
            label=block.label,
            position=source_cell,
            remaining_time=float(block.get_remaining_storage_time()),
        )
        candidates = shared_candidate_mask(yard, view, source_cell)
        mask_id = _candidate_mask_id(candidates)
        if preview.candidate_count != len(candidates):
            return candidates, "preview_candidate_count_mismatch"
        if preview.candidate_mask_id != mask_id:
            return candidates, "preview_candidate_mask_mismatch"
        if tuple(preview.chosen_cell) not in candidates:
            return candidates, "reserved_cell_outside_live_candidate_mask"
        expected_id = _storage_proposal_id(
            instance_id=preview.instance_id,
            assignment_source=preview.assignment_source,
            assignment_source_version=preview.assignment_source_version,
            block_label=preview.block_label,
            time_step=preview.time_step,
            source_cell=preview.source_cell,
            candidate_mask_id=preview.candidate_mask_id,
            chosen_cell=preview.chosen_cell,
        )
        if preview.proposal_id != expected_id:
            return candidates, "preview_proposal_id_mismatch"
        if preview.proposal_id in self._committed_proposal_ids:
            return candidates, "preview_proposal_already_committed"
        return candidates, None

    def validate_preview(self, block, preview, *, require_current_epoch=True):
        """Validate a frozen proposal without committing or rescoring it."""

        candidates, reason = self._validate_preview(
            block,
            preview,
            require_current_epoch=bool(require_current_epoch),
        )
        if reason is not None:
            raise ValueError(f"invalid reserved assignment proposal: {reason}")
        return candidates

    def bind_preview(self, block, preview):
        """Bind a validated proposal as the selected atomic reservation."""

        candidates = self.validate_preview(
            block, preview, require_current_epoch=True
        )
        reserved = getattr(self.source, "on_preview_reserved", None)
        if reserved is not None:
            reserved(
                proposal_id=preview.proposal_id,
                block_label=preview.block_label,
                chosen_cell=preview.chosen_cell,
                selection_time_step=int(preview.time_step),
                candidate_count=int(preview.candidate_count),
                candidate_mask_id=preview.candidate_mask_id,
            )
        return candidates

    def commit_preview(self, block, preview):
        """Commit one previously selected cell without a second source call."""

        call_started = perf_counter()
        candidates, reason = self._validate_preview(
            block, preview, require_current_epoch=False
        )
        decision = {
            "decision_index": self.decision_count,
            "time_step": int(preview.time_step),
            "selection_time_step": int(preview.time_step),
            "commit_time_step": int(self.env.time_steps),
            "block_label": getattr(block, "label", preview.block_label),
            "candidate_count": int(preview.candidate_count),
            "chosen_cell": tuple(preview.chosen_cell),
            "proposal_repr": repr(tuple(preview.chosen_cell)),
            "valid": False,
            "reason": reason,
            "assignment_source": self.assignment_source,
            "assignment_source_version": self.assignment_source_version,
            "candidate_mask_id": preview.candidate_mask_id,
            "live_candidate_mask_id": _candidate_mask_id(candidates),
            "proposal_id": preview.proposal_id,
            "commitment_contract": "decision_epoch_proposal_bound_once_v2",
        }
        self.decision_count += 1
        if reason is not None:
            self.invalid_assignment_count += 1
            self.decisions.append(decision)
            self.assignment_total_seconds += perf_counter() - call_started
            return None

        chosen = tuple(preview.chosen_cell)
        commit_metadata = {
            "proposal_id": preview.proposal_id,
            "block_label": block.label,
            "chosen_cell": chosen,
            "selection_time_step": int(preview.time_step),
            "commit_time_step": int(self.env.time_steps),
            "candidate_count": int(preview.candidate_count),
            "candidate_mask_id": preview.candidate_mask_id,
        }
        precommit = getattr(
            self.source, "validate_bound_assignment_commit", None
        )
        if precommit is not None:
            # Fail before mutating live assignment state if source provenance
            # differs from the reservation token.
            precommit(**commit_metadata)
        block.storage_location = chosen
        bound_committed = getattr(
            self.source, "on_bound_assignment_committed", None
        )
        committed = getattr(self.source, "on_assignment_committed", None)
        if bound_committed is not None:
            bound_committed(**commit_metadata)
        elif committed is not None:
            committed(
                block_label=block.label,
                chosen_cell=chosen,
                time_step=int(self.env.time_steps),
            )
        self._committed_proposal_ids.add(preview.proposal_id)
        decision["valid"] = True
        self.decisions.append(decision)
        self.env.store_events.append(
            {
                "episode": self.env.current_episode,
                "t_step": self.env.time_steps,
                "selection_t_step": preview.time_step,
                "phase": "reserved_chosen",
                "selector": self.assignment_source_version,
                "assignment_source": self.assignment_source,
                "block_label": block.label,
                "row": chosen[0],
                "col": chosen[1],
                "candidate_count": len(candidates),
                "candidate_mask_id": preview.candidate_mask_id,
                "proposal_id": preview.proposal_id,
            }
        )
        self.assignment_total_seconds += perf_counter() - call_started
        return chosen

    def preview_assignment(self, block, *, source_cell=None):
        """Score the current assignment state without committing or auditing.

        The preview is deliberately available only for a frozen selector.  It
        reflects information observable *now*; an AcceptStore v1 option makes
        its real assignment after the pickup leg, so a later proposal may
        differ if time or newly observed arrivals change its input.
        """

        if self.learning_enabled:
            raise RuntimeError(
                "assignment preview requires a frozen REG-v5 selector"
            )
        if block is None:
            raise ValueError(
                "assignment preview requires an unassigned observed inbound block"
            )
        if source_cell is None:
            source_cell = block.position
        if (
            block.delivered
            or block.stored
            or block.carrying
            or block.position is None
            or block.storage_location is not None
            or source_cell is None
        ):
            raise ValueError(
                "assignment preview requires an unassigned observed inbound block"
            )
        source_cell = tuple(source_cell)
        instance = getattr(self.env, "current_episode_instance", None)
        instance_id = getattr(instance, "instance_id", None)
        if instance_id is None:
            raise RuntimeError("assignment preview requires an active episode instance")
        yard = YardSnapshot.from_env(self.env)
        view = BlockView(
            label=block.label,
            position=source_cell,
            remaining_time=float(block.get_remaining_storage_time()),
        )
        candidates = shared_candidate_mask(yard, view, source_cell)
        if not candidates:
            return None
        preview = getattr(self.source, "preview", None)
        if preview is None:
            raise RuntimeError(
                "assignment source does not implement the side-effect-free "
                "preview contract"
            )
        proposal = preview(yard, view, source_cell, candidates)
        chosen, reason = _normalise_proposal(proposal)
        if reason is None and chosen not in candidates:
            reason = "outside_shared_candidate_mask"
        if reason is not None:
            raise RuntimeError(
                "assignment source preview returned no valid shared-mask "
                f"proposal ({reason}): {proposal!r}"
            )
        candidate_mask_id = _candidate_mask_id(candidates)
        proposal_id = _storage_proposal_id(
            instance_id=instance_id,
            assignment_source=self.assignment_source,
            assignment_source_version=self.assignment_source_version,
            block_label=block.label,
            time_step=int(self.env.time_steps),
            source_cell=source_cell,
            candidate_mask_id=candidate_mask_id,
            chosen_cell=tuple(chosen),
        )
        token = StorageAssignmentPreview(
            contract=self.preview_contract,
            proposal_id=proposal_id,
            instance_id=instance_id,
            assignment_source=self.assignment_source,
            assignment_source_version=self.assignment_source_version,
            block_label=block.label,
            time_step=int(self.env.time_steps),
            source_cell=source_cell,
            candidate_count=len(candidates),
            candidate_mask_id=candidate_mask_id,
            chosen_cell=tuple(chosen),
        )
        token_minted = getattr(self.source, "on_preview_token_minted", None)
        if token_minted is not None:
            token_minted(
                proposal_id=token.proposal_id,
                block_label=token.block_label,
                chosen_cell=token.chosen_cell,
                selection_time_step=token.time_step,
                candidate_count=token.candidate_count,
                candidate_mask_id=token.candidate_mask_id,
            )
        return token


    def on_step(self, reward: float, info: dict) -> None:
        callback = getattr(self.source, "on_step", None)
        if callback is not None:
            callback(reward, info)

    def on_episode_end(
        self, *, success: bool = False, truncated: bool = False
    ) -> None:
        callback = getattr(self.source, "on_episode_end", None)
        if callback is not None:
            callback(success=success, truncated=truncated)

    def checkpoint(self, **metadata) -> dict:
        callback = getattr(self.source, "checkpoint", None)
        if callback is None:
            raise RuntimeError(
                "deterministic assignment sources do not have checkpoints"
            )
        return callback(**metadata)

    def audit(self) -> dict:
        source_audit = getattr(self.source, "audit", None)
        return {
            "selector": self.assignment_source_version,
            "assignment_source": self.assignment_source,
            "assignment_source_family": self.assignment_source_family,
            "assignment_source_version": self.assignment_source_version,
            "assignment_source_learned": self.assignment_source_learned,
            "information_regime": self.information_regime,
            "preview_contract": self.preview_contract,
            "assignment_contract": self.assignment_contract,
            "deployment_assignment_contract": (
                self.deployment_assignment_contract
            ),
            "feature_version": getattr(self, "FEATURE_VERSION", None),
            "frozen": not self.learning_enabled,
            "decision_count": self.decision_count,
            "valid_assignment_count": sum(
                int(decision["valid"]) for decision in self.decisions
            ),
            "infeasible_epoch_count": self.infeasible_epoch_count,
            "invalid_assignment_count": self.invalid_assignment_count,
            "fallback_count": 0,
            "reserved_commit_count": sum(
                decision.get("commitment_contract")
                == "decision_epoch_proposal_bound_once_v2"
                and decision.get("valid")
                for decision in self.decisions
            ),
            "post_pickup_recompute_count": sum(
                decision.get("commitment_contract")
                == "post_pickup_recompute_v1"
                and decision.get("valid")
                for decision in self.decisions
            ),
            "assignment_seconds": self.assignment_seconds,
            "assignment_total_seconds": self.assignment_total_seconds,
            "decisions": list(self.decisions),
            "source_audit": source_audit() if callable(source_audit) else None,
        }

    def termination(self, _state) -> bool:
        return True

    def __str__(self) -> str:
        return "StorageSelectOption"

    __repr__ = __str__

    def __hash__(self):
        return hash("StorageSelectOptionV5")


class ExplicitCellAssignmentRegistry(StorageSelectOptionV5):
    """Issue strict proposal tokens for cells chosen by the joint controller.

    The wrapped frozen REG-v5 source is retained only as a read-only teacher
    and warm-start provenance. It is never asked to choose a cell when an
    explicit proposal is issued, and its independent replay remains disabled.
    """

    controller_identifier = "option:ExplicitCellAssignmentRegistry:v1"
    return_definition = "joint_macro_smdp_common_continuation_v1"

    def __init__(self, env: SmallRoomsEnv, source: REGV5AssignmentSource):
        if bool(getattr(source, "learning_enabled", False)):
            raise ValueError(
                "explicit-cell registry requires a frozen REG-v5 teacher"
            )
        super().__init__(env, source)
        self.assignment_source = JOINT_LEARNED_ASSIGNMENT_SOURCE
        self.assignment_source_family = "joint_learned_macro_controller"
        self.assignment_source_version = (
            JOINT_LEARNED_ASSIGNMENT_SOURCE_VERSION
        )
        self.preview_contract = "explicit_current_epoch_cell_proposal_v1"
        self.assignment_contract = "explicit_reserved_cell_commit_v1"
        self.deployment_assignment_contract = self.assignment_contract

    def preview_assignment_for_cell(
        self, block, chosen_cell, *, source_cell=None
    ) -> StorageAssignmentPreview:
        """Create one read-only proposal for an explicitly chosen live cell."""

        if block is None:
            raise ValueError("explicit proposal requires an inbound block")
        if source_cell is None:
            source_cell = block.position
        if (
            block.delivered
            or block.stored
            or block.carrying
            or block.position is None
            or block.storage_location is not None
            or source_cell is None
        ):
            raise ValueError(
                "explicit proposal requires an unassigned observed inbound block"
            )
        source_cell = tuple(source_cell)
        chosen, reason = _normalise_proposal(chosen_cell)
        if reason is not None:
            raise ValueError(f"invalid explicit storage cell: {reason}")
        instance = getattr(self.env, "current_episode_instance", None)
        instance_id = getattr(instance, "instance_id", None)
        if instance_id is None:
            raise RuntimeError(
                "explicit proposal requires an active episode instance"
            )
        yard = YardSnapshot.from_env(self.env)
        view = BlockView(
            label=block.label,
            position=source_cell,
            remaining_time=float(block.get_remaining_storage_time()),
        )
        candidates = shared_candidate_mask(yard, view, source_cell)
        if chosen not in candidates:
            raise ValueError("explicit cell is outside the shared candidate mask")
        candidate_mask_id = _candidate_mask_id(candidates)
        proposal_id = _storage_proposal_id(
            instance_id=instance_id,
            assignment_source=self.assignment_source,
            assignment_source_version=self.assignment_source_version,
            block_label=block.label,
            time_step=int(self.env.time_steps),
            source_cell=source_cell,
            candidate_mask_id=candidate_mask_id,
            chosen_cell=chosen,
        )
        return StorageAssignmentPreview(
            contract=self.preview_contract,
            proposal_id=proposal_id,
            instance_id=instance_id,
            assignment_source=self.assignment_source,
            assignment_source_version=self.assignment_source_version,
            block_label=block.label,
            time_step=int(self.env.time_steps),
            source_cell=source_cell,
            candidate_count=len(candidates),
            candidate_mask_id=candidate_mask_id,
            chosen_cell=chosen,
        )

    def preview_assignment(self, block, *, source_cell=None):
        """Return the frozen teacher cell as a joint-controller proposal."""

        if source_cell is None:
            source_cell = getattr(block, "position", None)
        if source_cell is None:
            raise ValueError("teacher preview requires an inbound source cell")
        source_cell = tuple(source_cell)
        yard = YardSnapshot.from_env(self.env)
        view = BlockView(
            label=block.label,
            position=source_cell,
            remaining_time=float(block.get_remaining_storage_time()),
        )
        candidates = shared_candidate_mask(yard, view, source_cell)
        if not candidates:
            return None
        chosen = self.source.preview(yard, view, source_cell, candidates)
        return self.preview_assignment_for_cell(
            block, chosen, source_cell=source_cell
        )


class StrictStorageAssignmentOption(StorageSelectOptionV5):
    """Track-B adapter for a previewable source with no fallback behavior."""

    FEATURE_VERSION = None
    ARCHITECTURE_NAME = "strict_track_b_assignment_adapter_v1"
    return_definition = "not_applicable_deterministic_assignment"

    def __init__(
        self,
        env: SmallRoomsEnv,
        source,
        *,
        assignment_source: str,
        assignment_source_version: str,
        preview_contract: str,
        assignment_contract: str,
    ):
        BaseOption.__init__(self, is_primitive=False)
        self.env = env
        self.source = source
        self.assignment_source = assignment_source
        self.assignment_source_family = "deterministic_assignment_rule"
        self.assignment_source_version = assignment_source_version
        self.information_regime = "online_arrived_only"
        self.preview_contract = preview_contract
        self.assignment_contract = assignment_contract
        self.assignment_source_learned = False
        self.block = None
        self.decision_count = 0
        self.infeasible_epoch_count = 0
        self.invalid_assignment_count = 0
        self.assignment_seconds = 0.0
        self.assignment_total_seconds = 0.0
        self.decisions: list[dict] = []
        self._committed_proposal_ids: set[str] = set()
        self.deployment_assignment_contract = self.assignment_contract


class Kim2020SpatialAssignmentOption(StrictStorageAssignmentOption):
    """Frozen Kim-adapted spatial policy under the shared reservation ABI."""

    FEATURE_VERSION = KIM2020_FEATURE_VERSION
    ARCHITECTURE_NAME = KIM2020_ARCHITECTURE_NAME
    return_definition = "episodic_discounted_placement_actor_critic_v1"

    def __init__(self, env: SmallRoomsEnv, source: Kim2020A3CSpatialSource):
        if source.env is not env:
            raise ValueError("Kim spatial source and option must share an environment")
        if bool(getattr(source, "learning_enabled", False)):
            raise ValueError("Track-B Kim spatial deployment requires a frozen source")
        mode = source.deployment_mode
        super().__init__(
            env,
            source,
            assignment_source=TRACK_A_KIM2020_A3C_SPATIAL,
            assignment_source_version=KIM2020_ARCHITECTURE_NAME,
            preview_contract=(
                f"current_epoch_frozen_kim2020_{mode}_stateless_preview_v1"
            ),
            assignment_contract=(
                "decision_epoch_bound_kim2020_spatial_assignment_v1"
            ),
        )
        self.assignment_source_family = "learned_spatial_assignment_policy"
        self.assignment_source_learned = True

    @classmethod
    def from_checkpoint(
        cls,
        env: SmallRoomsEnv,
        checkpoint: str | Path | dict,
        *,
        device: str | torch.device = "cpu",
        seed: int = 0,
        policy_seed: int | None = None,
        deployment_mode: str = KIM2020_DEPLOYMENT_STOCHASTIC,
    ) -> "Kim2020SpatialAssignmentOption":
        payload = (
            torch.load(checkpoint, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, (str, Path))
            else checkpoint
        )
        source = Kim2020A3CSpatialSource.from_checkpoint(
            env,
            payload,
            learning_enabled=False,
            device=device,
            seed=seed,
            policy_seed=seed if policy_seed is None else policy_seed,
            deployment_mode=deployment_mode,
        )
        return cls(env, source)


def make_track_b_assignment_option(
    env: SmallRoomsEnv,
    assignment_source: str,
    *,
    selector_payload=None,
    assignment_payload=None,
    device: str | torch.device = "cpu",
    seed: int = 0,
    policy_seed: int | None = None,
    deployment_mode: str = KIM2020_DEPLOYMENT_STOCHASTIC,
    rolling_ga_config=None,
    rolling_ga_egress_weight: int = 4,
):
    """Build one strict preview/commit adapter for the Track-B ablation."""

    if selector_payload is not None and assignment_payload is not None:
        raise ValueError("pass only one of selector_payload and assignment_payload")
    payload = assignment_payload if assignment_payload is not None else selector_payload
    if assignment_source == TRACK_A_REG_SELECTOR_V5:
        if payload is None:
            raise ValueError("REG-v5 assignment requires a selector checkpoint")
        return StorageSelectOptionV5.from_checkpoint(
            env,
            payload,
            device=device,
            seed=seed,
            learning_enabled=False,
        )
    if assignment_source == TRACK_A_KIM2020_A3C_SPATIAL:
        if payload is None:
            raise ValueError("Kim spatial assignment requires a checkpoint")
        return Kim2020SpatialAssignmentOption.from_checkpoint(
            env,
            payload,
            device=device,
            seed=seed,
            policy_seed=policy_seed,
            deployment_mode=deployment_mode,
        )
    if payload is not None:
        raise ValueError(
            "deterministic assignment sources may not receive a selector "
            "checkpoint"
        )
    if assignment_source == TRACK_A_NEAREST_FREE:
        return StrictStorageAssignmentOption(
            env,
            NearestFreeAssignmentSource(),
            assignment_source=assignment_source,
            assignment_source_version=NEAREST_FREE_ASSIGNMENT_SOURCE_VERSION,
            preview_contract=(
                "current_epoch_nearest_free_deterministic_preview_v1"
            ),
            assignment_contract=(
                "post_pickup_nearest_free_deterministic_assignment_v1"
            ),
        )
    if assignment_source == TRACK_A_DYNAMIC:
        return StrictStorageAssignmentOption(
            env,
            DynamicAssignmentSource(),
            assignment_source=assignment_source,
            assignment_source_version=DYNAMIC_ASSIGNMENT_SOURCE_VERSION,
            preview_contract=(
                "current_epoch_dynamic_pslap_deterministic_preview_v1"
            ),
            assignment_contract=(
                "post_pickup_dynamic_pslap_deterministic_assignment_v1"
            ),
        )
    if assignment_source == TRACK_A_GA_ROLLING:
        return StrictStorageAssignmentOption(
            env,
            RollingGAStorageAssigner(
                env,
                rolling_ga_config or DEFAULT_ROLLING_GA_CONFIG,
            ),
            assignment_source=assignment_source,
            assignment_source_version=RollingGAStorageAssigner.VERSION,
            preview_contract=(
                "current_epoch_rolling_ga_side_effect_free_preview_v1"
            ),
            assignment_contract=(
                "decision_epoch_bound_rolling_ga_assignment_v1"
            ),
        )
    if assignment_source == TRACK_A_GA_ROLLING_DURATION_AWARE:
        return StrictStorageAssignmentOption(
            env,
            DurationAwareRollingGAStorageAssigner(
                env,
                rolling_ga_config or DEFAULT_ROLLING_GA_CONFIG,
                egress_weight=rolling_ga_egress_weight,
            ),
            assignment_source=assignment_source,
            assignment_source_version=(
                DurationAwareRollingGAStorageAssigner.VERSION
            ),
            preview_contract=(
                "current_epoch_duration_aware_rolling_ga_preview_v1"
            ),
            assignment_contract=(
                "decision_epoch_bound_duration_aware_rolling_ga_v1"
            ),
        )
    if assignment_source == TRACK_A_GA_ROLLING_OPERATIONAL:
        return StrictStorageAssignmentOption(
            env,
            OperationalRollingGAStorageAssigner(
                env,
                rolling_ga_config or DEFAULT_ROLLING_GA_CONFIG,
                egress_weight=rolling_ga_egress_weight,
            ),
            assignment_source=assignment_source,
            assignment_source_version=OperationalRollingGAStorageAssigner.VERSION,
            preview_contract=(
                "current_epoch_operational_rolling_ga_preview_v2"
            ),
            assignment_contract=(
                "decision_epoch_bound_operational_rolling_ga_v2"
            ),
        )
    if assignment_source == TRACK_A_GA_ROLLING_COMPLETE:
        return StrictStorageAssignmentOption(
            env,
            CompleteSingletonDurationAwareRollingGAAssigner(
                env,
                rolling_ga_config or DEFAULT_ROLLING_GA_CONFIG,
                egress_weight=rolling_ga_egress_weight,
            ),
            assignment_source=assignment_source,
            assignment_source_version=(
                CompleteSingletonDurationAwareRollingGAAssigner.VERSION
            ),
            preview_contract=(
                "current_epoch_duration_aware_complete_preview_v3"
            ),
            assignment_contract=(
                "decision_epoch_bound_duration_aware_complete_v3"
            ),
        )
    if assignment_source == TRACK_A_GA_ROLLING_CAPACITY_AWARE:
        return StrictStorageAssignmentOption(
            env,
            CapacityAwareRollingGAStorageAssigner(
                env,
                rolling_ga_config or DEFAULT_ROLLING_GA_CONFIG,
            ),
            assignment_source=assignment_source,
            assignment_source_version=(
                CapacityAwareRollingGAStorageAssigner.VERSION
            ),
            preview_contract=(
                "current_epoch_capacity_aware_rolling_ga_bound_preview_v1"
            ),
            assignment_contract=(
                "decision_epoch_bound_capacity_aware_rolling_ga_assignment_v1"
            ),
        )
    if assignment_source == TRACK_A_GA_ROLLING_DURATION_AWARE_CAPACITY_AWARE:
        return StrictStorageAssignmentOption(
            env,
            CapacityAwareDurationAwareRollingGAAssigner(
                env,
                rolling_ga_config or DEFAULT_ROLLING_GA_CONFIG,
                egress_weight=rolling_ga_egress_weight,
            ),
            assignment_source=assignment_source,
            assignment_source_version=(
                CapacityAwareDurationAwareRollingGAAssigner.VERSION
            ),
            preview_contract=(
                "current_epoch_capacity_aware_duration_ga_bound_preview_v1"
            ),
            assignment_contract=(
                "decision_epoch_bound_capacity_aware_duration_ga_assignment_v1"
            ),
        )
    if assignment_source == TRACK_A_GA_ROLLING_OPERATIONAL_CAPACITY_AWARE:
        return StrictStorageAssignmentOption(
            env,
            CapacityAwareOperationalRollingGAAssigner(
                env,
                rolling_ga_config or DEFAULT_ROLLING_GA_CONFIG,
                egress_weight=rolling_ga_egress_weight,
            ),
            assignment_source=assignment_source,
            assignment_source_version=(
                CapacityAwareOperationalRollingGAAssigner.VERSION
            ),
            preview_contract=(
                "current_epoch_capacity_aware_operational_ga_bound_preview_v1"
            ),
            assignment_contract=(
                "decision_epoch_bound_capacity_aware_operational_ga_assignment_v1"
            ),
        )
    if assignment_source == TRACK_A_GA_ROLLING_COMPLETE_CAPACITY_AWARE:
        return StrictStorageAssignmentOption(
            env,
            CapacityAwareCompleteRollingGAAssigner(
                env,
                rolling_ga_config or DEFAULT_ROLLING_GA_CONFIG,
                egress_weight=rolling_ga_egress_weight,
            ),
            assignment_source=assignment_source,
            assignment_source_version=(
                CapacityAwareCompleteRollingGAAssigner.VERSION
            ),
            preview_contract=(
                "current_epoch_capacity_aware_complete_ga_bound_preview_v1"
            ),
            assignment_contract=(
                "decision_epoch_bound_capacity_aware_complete_ga_assignment_v1"
            ),
        )
    raise ValueError(
        f"Unknown Track-B assignment source {assignment_source!r}; "
        f"choose one of {TRACK_B_ALL_ASSIGNMENT_SOURCES}"
    )


__all__ = [
    "ExplicitCellAssignmentRegistry",
    "JOINT_LEARNED_ASSIGNMENT_SOURCE",
    "JOINT_LEARNED_ASSIGNMENT_SOURCE_VERSION",
    "Kim2020SpatialAssignmentOption",
    "StorageAssignmentPreview",
    "StorageSelectOptionV5",
    "StrictStorageAssignmentOption",
    "TRACK_B_ASSIGNMENT_SOURCES",
    "TRACK_B_ALL_ASSIGNMENT_SOURCES",
    "TRACK_B_RESERVED_CAPACITY_AWARE_ASSIGNMENT_SOURCES",
    "make_track_b_assignment_option",
]
