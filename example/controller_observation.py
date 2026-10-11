"""Versioned, online-safe observations for the Track-B controller.

Options continue to consume :meth:`SmallRoomsEnv.get_current_state`.  This
module builds a separate critic observation from information revealed by the
current environment state.  In particular it never exposes the sampled next
arrival time or processing durations belonging to unarrived blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from example.helper.tools import _astar, flat
from PSLAP.neutral_protocol import (
    NEUTRAL_RELOCATION_SELECTOR_VERSION,
    neutral_relocation_selector,
)
from PSLAP.retrieval_context import retrieval_planning_context


LEGACY_CONTROLLER_OBSERVATION = "legacy_flat_v1"
ONLINE_SIGNED_TIMING_OBSERVATION = "online_signed_timing_v2"
ONLINE_MANIFEST_TIMING_OBSERVATION = "online_manifest_timing_v3"
SUPPORTED_CONTROLLER_OBSERVATIONS = (
    LEGACY_CONTROLLER_OBSERVATION,
    ONLINE_SIGNED_TIMING_OBSERVATION,
    ONLINE_MANIFEST_TIMING_OBSERVATION,
)

GLOBAL_FEATURE_NAMES = (
    "agent_row",
    "agent_col",
    "current_step",
    "arrival_rate",
    "processing_mean",
    "observed_arrived_fraction",
    "known_unarrived_fraction",
    "waiting_fraction",
    "pickup_ready",
    "active_stored_fraction",
    "delivered_fraction",
    "storage_occupancy_fraction",
    "carrying_unstored",
    "carrying_stored",
    "steps_since_last_observed_arrival",
    "has_valid_retrieval_eta",
    "minimum_signed_retrieval_slack",
    "due_if_dispatched_fraction",
    "raw_overdue_fraction",
    "unavailable_retrieval_plan_fraction",
)

BLOCK_FEATURE_NAMES = (
    "available_unstored",
    "carrying_unstored",
    "stored_in_yard",
    "carrying_stored",
    "delivered",
    "at_waiting_cell",
    "at_pickup_cell",
    "row",
    "col",
    "assignment_valid",
    "assigned_row",
    "assigned_col",
    "observed_age",
    "required_storage_duration",
    "agent_path_valid",
    "agent_path_length",
    "assigned_storage_path_valid",
    "assigned_storage_path_length",
    "clear_egress_valid",
    "clear_egress_length",
    "deadline_valid",
    "signed_remaining_time",
    "retrieval_eta_valid",
    "retrieval_eta",
    "signed_eta_adjusted_slack",
    "due_if_dispatched_now",
    "already_overdue",
    "canonical_relocation_count",
    "observed_arrival_rank",
)

OBSERVATION_METADATA_KEYS = (
    "controller_observation_version",
    "controller_observation_global_size",
    "controller_observation_block_size",
    "controller_observation_max_blocks",
    "controller_observation_feature_size",
    "controller_observation_flatten_order",
    "controller_observation_timing_transform",
    "controller_observation_timing_scale",
    "controller_observation_path_scale",
    "controller_observation_episode_scale",
    "controller_observation_future_manifest_known",
)


def _clip_unsigned(value: float, scale: float) -> float:
    return float(np.clip(float(value) / scale, 0.0, 1.0))


def _clip_signed(value: float, scale: float) -> float:
    return float(np.clip(float(value) / scale, -1.0, 1.0))


def _coordinate(value: int, size: int) -> float:
    return float(value) / max(1, size - 1)


def _shortest_path_to_any_exit(env, start, blocked):
    candidates = [
        path
        for exit_cell in env.exit_cells
        if (path := _astar(env.rooms, start, exit_cell, blocked))
    ]
    return min(candidates, key=len) if candidates else []


@dataclass(frozen=True)
class ControllerObservationV2:
    """Structured observation plus raw timing values used for audits/tests."""

    global_features: np.ndarray
    block_mask: np.ndarray
    block_features: np.ndarray
    observed_labels: tuple[str, ...]
    global_raw: dict[str, float]
    remaining_steps: np.ndarray
    retrieval_eta_steps: np.ndarray
    signed_slack_steps: np.ndarray

    def flatten(self) -> np.ndarray:
        return np.concatenate(
            (
                self.global_features,
                self.block_mask.astype(np.float32),
                self.block_features.reshape(-1),
            )
        ).astype(np.float32, copy=False)


class LegacyControllerObservationEncoder:
    version = LEGACY_CONTROLLER_OBSERVATION

    def __init__(self, env):
        self.env = env
        self.feature_size = len(flat(env.get_current_state()))

    def __call__(self, state=None):
        if state is None:
            state = self.env.get_current_state()
        return np.asarray(flat(state), dtype=np.float32)

    def metadata(self):
        return {
            "controller_observation_version": self.version,
            "controller_observation_global_size": 0,
            "controller_observation_block_size": 0,
            "controller_observation_max_blocks": len(self.env.blocks),
            "controller_observation_feature_size": self.feature_size,
            "controller_observation_flatten_order": "legacy_state_flattener_v1",
            "controller_observation_timing_transform": "legacy_clamped_cap15",
            "controller_observation_timing_scale": 15.0,
            "controller_observation_path_scale": float(
                self.env.grid_rows + self.env.grid_cols
            ),
            "controller_observation_episode_scale": None,
            "controller_observation_future_manifest_known": True,
        }

class OnlineSignedTimingObservationEncoder:
    """Capture the revealed yard state with signed deadline information."""

    version = ONLINE_SIGNED_TIMING_OBSERVATION
    relocation_selector = None

    def __init__(self, env):
        self.env = env
        self.max_blocks = len(env.blocks)
        self.timing_scale = float(env.MAX_T)
        traversable = int(np.count_nonzero(env.rooms != "#"))
        self.path_scale = float(max(1, traversable - 1))
        self.episode_scale = float(max(1, self.max_blocks) * env.MAX_T)
        self.feature_size = (
            len(GLOBAL_FEATURE_NAMES)
            + self.max_blocks
            + self.max_blocks * len(BLOCK_FEATURE_NAMES)
        )
        self._cache_key = None
        self._cache_value = None

    def metadata(self):
        return {
            "controller_observation_version": self.version,
            "controller_observation_global_size": len(GLOBAL_FEATURE_NAMES),
            "controller_observation_block_size": len(BLOCK_FEATURE_NAMES),
            "controller_observation_max_blocks": self.max_blocks,
            "controller_observation_feature_size": self.feature_size,
            "controller_observation_flatten_order": (
                "global,block_mask,observed_arrival_order_block_rows"
            ),
            "controller_observation_timing_transform": "signed_linear_clip_v1",
            "controller_observation_timing_scale": self.timing_scale,
            "controller_observation_path_scale": self.path_scale,
            "controller_observation_episode_scale": self.episode_scale,
            "controller_observation_future_manifest_known": True,
        }

    @staticmethod
    def _is_observed(block):
        return bool(
            block.position is not None
            or block.carrying
            or block.stored
            or block.delivered
        )

    def _observed(self):
        observed = [
            (index, block)
            for index, block in enumerate(self.env.blocks)
            if self._is_observed(block)
        ]
        observed.sort(
            key=lambda item: (int(item[1].arrival_step), item[0])
        )
        return observed

    @staticmethod
    def _feature_row(original_index, observed_rank):
        return observed_rank

    def _state_key(self, agent_position, observed):
        # Only revealed block attributes participate in the cache key.  Future
        # schedule/duration changes therefore cannot affect even cache routing.
        block_state = tuple(
            (
                index,
                block.position,
                block.storage_location,
                bool(block.carrying),
                bool(block.stored),
                bool(block.delivered),
                block.stored_time_step,
                int(block.storage_steps_needed),
                int(block.arrival_step),
            )
            for index, block in observed
        )
        return (
            int(self.env.time_steps),
            tuple(agent_position),
            block_state,
            self.env.rooms.tobytes(),
        )

    def __call__(self, state=None):
        return self.capture(state).flatten()

    def capture(self, state=None):
        if state is None:
            state = self.env.get_current_state()
        agent_position = tuple(state[0])
        observed = self._observed()
        key = self._state_key(agent_position, observed)
        if key == self._cache_key:
            return self._cache_value

        snapshot = self._capture_uncached(agent_position, observed)
        self._cache_key = key
        self._cache_value = snapshot
        return snapshot

    def _capture_uncached(self, agent_position, observed):
        env = self.env
        n_blocks = max(1, self.max_blocks)
        block_mask = np.zeros(self.max_blocks, dtype=np.bool_)
        block_features = np.zeros(
            (self.max_blocks, len(BLOCK_FEATURE_NAMES)), dtype=np.float32
        )
        remaining_steps = np.full(self.max_blocks, np.nan, dtype=np.float32)
        retrieval_eta_steps = np.full(
            self.max_blocks, np.nan, dtype=np.float32
        )
        signed_slack_steps = np.full(
            self.max_blocks, np.nan, dtype=np.float32
        )

        carried = next((block for block in env.blocks if block.carrying), None)
        hands_free = carried is None
        plans = {}
        if hands_free:
            planning_context = retrieval_planning_context(
                env, relocation_selector=self.relocation_selector
            )
            for _, block in observed:
                if block.stored and not block.delivered and not block.carrying:
                    plans[block.label] = planning_context.plan(block.label)

        live_positions = {
            block.position
            for block in env.blocks
            if block.position is not None
            and not block.delivered
            and not block.carrying
        }

        valid_slacks = []
        due_count = 0
        overdue_count = 0
        unavailable_plan_count = 0
        active_stored = [
            block for block in env.blocks if block.stored and not block.delivered
        ]

        for observed_rank, (original_index, block) in enumerate(observed):
            row = self._feature_row(original_index, observed_rank)
            if row >= self.max_blocks:
                break
            block_mask[row] = True
            feature = block_features[row]
            available = bool(
                not block.carrying and not block.stored and not block.delivered
            )
            feature[0] = available
            feature[1] = bool(block.carrying and not block.stored)
            feature[2] = bool(
                block.stored and not block.carrying and not block.delivered
            )
            feature[3] = bool(
                block.stored and block.carrying and not block.delivered
            )
            feature[4] = bool(block.delivered)
            feature[5] = bool(block.position == env.waiting_cell)
            feature[6] = bool(block.position == env.pickup_cell)
            if block.position is not None:
                feature[7] = _coordinate(block.position[0], env.grid_rows)
                feature[8] = _coordinate(block.position[1], env.grid_cols)

            assignment_valid = bool(
                block.storage_location is not None and not block.delivered
            )
            feature[9] = assignment_valid
            if assignment_valid:
                feature[10] = _coordinate(
                    block.storage_location[0], env.grid_rows
                )
                feature[11] = _coordinate(
                    block.storage_location[1], env.grid_cols
                )

            age = max(0, int(env.time_steps) - int(block.arrival_step))
            feature[12] = _clip_unsigned(age, self.episode_scale)
            feature[13] = _clip_unsigned(
                block.storage_steps_needed, self.timing_scale
            )

            if block.position is not None and not block.delivered:
                blocked = set(live_positions)
                blocked.discard(block.position)
                agent_path = _astar(
                    env.rooms, agent_position, block.position, blocked
                )
                if agent_path:
                    feature[14] = 1.0
                    feature[15] = _clip_unsigned(
                        len(agent_path) - 1, self.path_scale
                    )

            if (
                block.carrying
                and not block.stored
                and block.storage_location is not None
            ):
                blocked = set(live_positions)
                storage_path = _astar(
                    env.rooms,
                    agent_position,
                    block.storage_location,
                    blocked,
                )
                if storage_path:
                    feature[16] = 1.0
                    feature[17] = _clip_unsigned(
                        len(storage_path) - 1, self.path_scale
                    )

            if block.stored and not block.delivered and block.position is not None:
                blocked = set(live_positions)
                blocked.discard(block.position)
                egress = _shortest_path_to_any_exit(
                    env, block.position, blocked
                )
                if egress:
                    feature[18] = 1.0
                    feature[19] = _clip_unsigned(
                        len(egress) - 1, self.path_scale
                    )

                remaining = env.signed_remaining_storage_time(block)
                if remaining is not None:
                    feature[20] = 1.0
                    feature[21] = _clip_signed(
                        remaining, self.timing_scale
                    )
                    remaining_steps[row] = remaining
                    if remaining < 0:
                        overdue_count += 1
                        feature[26] = 1.0

                eta = None
                relocations = 0
                if block.carrying:
                    exit_path = _shortest_path_to_any_exit(
                        env, agent_position, set(live_positions)
                    )
                    if exit_path:
                        eta = len(exit_path) - 1 + 1
                elif hands_free:
                    plan = plans.get(block.label)
                    if plan is not None:
                        eta = plan.estimated_steps
                        relocations = len(plan.relocations)

                if eta is not None and remaining is not None:
                    slack = remaining - eta
                    feature[22] = 1.0
                    feature[23] = _clip_unsigned(eta, self.timing_scale)
                    feature[24] = _clip_signed(slack, self.timing_scale)
                    feature[25] = float(slack <= 0)
                    feature[27] = float(relocations) / n_blocks
                    retrieval_eta_steps[row] = eta
                    signed_slack_steps[row] = slack
                    valid_slacks.append(float(slack))
                    due_count += int(slack <= 0)
                else:
                    unavailable_plan_count += 1

            feature[28] = float(observed_rank) / max(1, self.max_blocks - 1)

        observed_count = len(observed)
        waiting_count = sum(
            1
            for _, block in observed
            if block.position == env.waiting_cell
            and not block.carrying
            and not block.delivered
        )
        delivered_count = sum(block.delivered for _, block in observed)
        occupied_storage = {
            block.position
            for block in active_stored
            if block.position in env.storage_positions and not block.carrying
        }
        latest_observed_arrival = max(
            (int(block.arrival_step) for _, block in observed), default=0
        )
        since_arrival = max(0, int(env.time_steps) - latest_observed_arrival)
        active_count = len(active_stored)
        global_raw = {
            "current_step": float(env.time_steps),
            "observed_count": float(observed_count),
            "waiting_count": float(waiting_count),
            "active_stored_count": float(active_count),
            "delivered_count": float(delivered_count),
            "steps_since_last_observed_arrival": float(since_arrival),
            "minimum_signed_retrieval_slack": (
                min(valid_slacks) if valid_slacks else np.nan
            ),
        }
        global_features = np.asarray(
            (
                _coordinate(agent_position[0], env.grid_rows),
                _coordinate(agent_position[1], env.grid_cols),
                _clip_unsigned(env.time_steps, self.episode_scale),
                float(env.arrival_rate) / (1.0 + float(env.arrival_rate)),
                float(env.proc_mean)
                / (float(env.proc_mean) + self.timing_scale),
                observed_count / n_blocks,
                (self.max_blocks - observed_count) / n_blocks,
                waiting_count / n_blocks,
                float(
                    any(
                        block.position == env.pickup_cell
                        and not block.carrying
                        and not block.delivered
                        for _, block in observed
                    )
                ),
                active_count / n_blocks,
                delivered_count / n_blocks,
                len(occupied_storage) / max(1, len(env.storage_positions)),
                float(carried is not None and not carried.stored),
                float(carried is not None and carried.stored),
                _clip_unsigned(since_arrival, self.timing_scale),
                float(bool(valid_slacks)),
                (
                    _clip_signed(min(valid_slacks), self.timing_scale)
                    if valid_slacks
                    else 0.0
                ),
                due_count / max(1, active_count),
                overdue_count / max(1, active_count),
                unavailable_plan_count / max(1, active_count),
            ),
            dtype=np.float32,
        )
        return ControllerObservationV2(
            global_features=global_features,
            block_mask=block_mask,
            block_features=block_features,
            observed_labels=tuple(block.label for _, block in observed),
            global_raw=global_raw,
            remaining_steps=remaining_steps,
            retrieval_eta_steps=retrieval_eta_steps,
            signed_slack_steps=signed_slack_steps,
        )


class OnlineManifestTimingObservationEncoder(OnlineSignedTimingObservationEncoder):
    """Online-safe observation whose block rows match fixed manifest slots."""

    version = ONLINE_MANIFEST_TIMING_OBSERVATION
    relocation_selector = staticmethod(neutral_relocation_selector)

    @staticmethod
    def _feature_row(original_index, observed_rank):
        return original_index

    def metadata(self):
        return {
            **super().metadata(),
            "controller_observation_version": self.version,
            "controller_observation_flatten_order": (
                "global,block_mask,episode_manifest_slot_block_rows"
            ),
            "controller_observation_block_row_contract": (
                "episode_manifest_slot_v1"
            ),
            "controller_observation_relocation_selector": (
                NEUTRAL_RELOCATION_SELECTOR_VERSION
            ),
        }


def make_controller_observation_encoder(env, version):
    if version == LEGACY_CONTROLLER_OBSERVATION:
        return LegacyControllerObservationEncoder(env)
    if version == ONLINE_SIGNED_TIMING_OBSERVATION:
        return OnlineSignedTimingObservationEncoder(env)
    if version == ONLINE_MANIFEST_TIMING_OBSERVATION:
        return OnlineManifestTimingObservationEncoder(env)
    raise ValueError(f"Unknown controller observation version: {version!r}")


def controller_observation_encoder_from_checkpoint(env, payload):
    present = [key for key in OBSERVATION_METADATA_KEYS if key in payload]
    if not present:
        return LegacyControllerObservationEncoder(env)
    missing = [key for key in OBSERVATION_METADATA_KEYS if key not in payload]
    if missing:
        raise ValueError(
            "Incomplete controller-observation checkpoint metadata: "
            + ", ".join(missing)
        )
    encoder = make_controller_observation_encoder(
        env, payload["controller_observation_version"]
    )
    expected = encoder.metadata()
    for key in OBSERVATION_METADATA_KEYS:
        if payload[key] != expected[key]:
            raise ValueError(
                f"Controller observation metadata mismatch for {key}: "
                f"saved={payload[key]!r}, runtime={expected[key]!r}"
            )
    return encoder


__all__ = [
    "BLOCK_FEATURE_NAMES",
    "ControllerObservationV2",
    "GLOBAL_FEATURE_NAMES",
    "LEGACY_CONTROLLER_OBSERVATION",
    "ONLINE_SIGNED_TIMING_OBSERVATION",
    "ONLINE_MANIFEST_TIMING_OBSERVATION",
    "OBSERVATION_METADATA_KEYS",
    "SUPPORTED_CONTROLLER_OBSERVATIONS",
    "LegacyControllerObservationEncoder",
    "OnlineSignedTimingObservationEncoder",
    "OnlineManifestTimingObservationEncoder",
    "controller_observation_encoder_from_checkpoint",
    "make_controller_observation_encoder",
]
