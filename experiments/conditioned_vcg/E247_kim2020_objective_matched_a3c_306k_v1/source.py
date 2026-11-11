"""Kim-inspired placement actor-critic trained on the current operational J.

The spatial actor and five-channel adaptation are inherited from the audited
historical comparator.  Only credit assignment changes: exact primitive
rewards emitted by the current environment are accumulated between placement
decisions.  With gamma=1, every placement return is therefore the realized
future -0.01 J under the common non-certified baseline scheduler.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import torch

from PSLAP.kim2020_a3c_spatial import (
    ACTION_MAPPING,
    DEPLOYMENT_STOCHASTIC,
    INPUT_CHANNELS,
    Kim2020A3CSpatialSource,
    Kim2020Config,
    geometry_contract,
)


METHOD_NAME = "kim2020_objective_matched_placement_a3c"
FEATURE_VERSION = "kim2020_relative_dwell_spatial_five_channel_j_v1"
ARCHITECTURE_NAME = "kim2020_grid_actor_critic_objective_matched_v1"
REWARD_CONTRACT = "exact_primitive_negative_0p01_J_segmented_by_placement_v1"
RETURN_CONTRACT = "undiscounted_future_operational_J_from_placement_v1"
CHECKPOINT_SCHEMA_VERSION = 1
REWARD_SCALE = 0.01


class ObjectiveMatchedKimSource(Kim2020A3CSpatialSource):
    """Placement actor-critic with exact objective-aligned episodic returns."""

    feature_version = FEATURE_VERSION
    architecture_name = ARCHITECTURE_NAME
    reward_contract = REWARD_CONTRACT
    return_contract = RETURN_CONTRACT

    def __init__(self, env, config: Kim2020Config, **kwargs):
        if float(config.gamma) != 1.0:
            raise ValueError("objective-matched Kim source requires gamma=1")
        super().__init__(env, config, **kwargs)
        self._objective_records = []
        self._active_objective_record = None
        self._unassigned_scaled_reward = 0.0
        self._episode_scaled_reward = 0.0

    def on_episode_start(self) -> None:
        super().on_episode_start()
        self._objective_records = []
        self._active_objective_record = None
        self._unassigned_scaled_reward = 0.0
        self._episode_scaled_reward = 0.0

    def propose(self, yard, block, source, valid_candidates):
        chosen = super().propose(yard, block, source, valid_candidates)
        self._register_objective_record(block)
        return chosen

    def _register_objective_record(self, block) -> None:
        if self.learning_enabled:
            record = self.provisional_records[block.label]
            if not any(item is record for item in self._objective_records):
                record.reward = float(self._unassigned_scaled_reward)
                self._unassigned_scaled_reward = 0.0
                self._objective_records.append(record)
            self._active_objective_record = record

    def preview(self, yard, block, source, valid_candidates):
        """Select once at the common scheduler's reservation decision epoch.

        The Dynamic/GA scheduler uses an atomic reservation preview before the
        pickup leg.  During training that preview is the placement decision,
        so it must retain the policy record; generator retries reuse the same
        provisional record in ``_select`` and cannot create a second sample.
        Frozen evaluation remains side-effect free.
        """

        chosen = self._select(
            yard, block, source, valid_candidates, record=self.learning_enabled
        )
        self._register_objective_record(block)
        return chosen

    def on_step(self, reward: float, info: dict) -> None:
        if not self.learning_enabled:
            return
        scaled = REWARD_SCALE * float(reward)
        self._episode_scaled_reward += scaled
        if self._active_objective_record is None:
            self._unassigned_scaled_reward += scaled
        else:
            self._active_objective_record.reward = float(
                (self._active_objective_record.reward or 0.0) + scaled
            )

        stored_label = info.get("stored_block")
        if stored_label:
            record = self.provisional_records.pop(str(stored_label), None)
            if record is None:
                self._unmatched_store_count += 1
            else:
                record.stored = True
                record.storage_time_step = int(getattr(self.env, "time_steps", 0))
                self.pending_records[record.block_label] = record

        delivered_label = info.get("delivered_block")
        if delivered_label:
            record = self.pending_records.pop(str(delivered_label), None)
            if record is None:
                self._unmatched_delivery_count += 1
            else:
                self.completed_records.append(record)

    def on_episode_end(self, *, success: bool, truncated: bool) -> None:
        if not self.learning_enabled:
            self.last_episode_records = []
            self.last_episode_audit = {
                "success": bool(success), "truncated": bool(truncated),
                "trained": False, "outcome_tracking": False,
            }
            self.provisional_records = {}
            self.pending_records = {}
            self.completed_records = []
            self._objective_records = []
            self._active_objective_record = None
            return

        # Any objective cost before the first placement is action-independent;
        # attaching it to the first record makes the return telescope exactly.
        if self._objective_records and self._unassigned_scaled_reward:
            first = self._objective_records[0]
            first.reward = float((first.reward or 0.0) + self._unassigned_scaled_reward)
            self._unassigned_scaled_reward = 0.0

        censored = [
            *self.provisional_records.values(), *self.pending_records.values()
        ]
        integrity = bool(
            success and not truncated and self._objective_records and not censored
            and self._unmatched_store_count == 0
            and self._unmatched_delivery_count == 0
            and len(self.completed_records) == len(self._objective_records)
        )
        attributed = float(sum(float(item.reward or 0.0) for item in self._objective_records))
        telescoping_error = abs(attributed - self._episode_scaled_reward)
        telescopes = telescoping_error <= 1.0e-6
        trained = bool(integrity and telescopes)
        if trained:
            self._learn_records(list(self._objective_records))
            self.training_episode_count += 1
        else:
            self.skipped_episode_count += 1
        self.completed_outcome_count += len(self.completed_records)
        self.censored_outcome_count += len(censored)
        self.return_history.append(float(self._episode_scaled_reward))
        self.last_episode_records = list(self._objective_records)
        self.last_episode_audit = {
            "success": bool(success), "truncated": bool(truncated),
            "integrity": integrity, "reward_telescopes": telescopes,
            "trained": trained, "placement_count": len(self._objective_records),
            "episode_scaled_reward": float(self._episode_scaled_reward),
            "attributed_scaled_reward": attributed,
            "telescoping_absolute_error": telescoping_error,
            "unmatched_store_count": self._unmatched_store_count,
            "unmatched_delivery_count": self._unmatched_delivery_count,
            "censored_count": len(censored),
        }
        self.provisional_records = {}
        self.pending_records = {}
        self.completed_records = []
        self._objective_records = []
        self._active_objective_record = None

    def audit(self) -> dict:
        value = super().audit()
        value.update({
            "assignment_source": METHOD_NAME,
            "assignment_source_version": ARCHITECTURE_NAME,
            "selector_feature_version": FEATURE_VERSION,
            "reward_contract": REWARD_CONTRACT,
            "return_contract": RETURN_CONTRACT,
            "certification_used": False,
            "vcg_guard_used": False,
        })
        return value

    def checkpoint(self, **metadata) -> dict:
        state = self.network.state_dict()
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "assignment_source": METHOD_NAME,
            "selector_feature_version": FEATURE_VERSION,
            "selector_architecture": ARCHITECTURE_NAME,
            "adaptation_status": "kim2020_inspired_not_exact_reproduction",
            "paper_doi": "10.1080/00207543.2020.1748247",
            "reward_contract": REWARD_CONTRACT,
            "return_contract": RETURN_CONTRACT,
            "action_mapping": ACTION_MAPPING,
            "input_channels": INPUT_CHANNELS,
            "geometry_contract": self.geometry,
            "config": asdict(self.config),
            "network_state_dict": state,
            "deployment_state_dict": state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_rng_state": self.training_rng.bit_generator.state,
            "training_seed": self.training_seed,
            "assignment_count": self.assignment_count,
            "gradient_steps": self.gradient_steps,
            "training_episode_count": self.training_episode_count,
            "skipped_episode_count": self.skipped_episode_count,
            "loss_history": list(self.loss_history),
            "actor_loss_history": list(self.actor_loss_history),
            "value_loss_history": list(self.value_loss_history),
            "entropy_history": list(self.entropy_history),
            "return_history": list(self.return_history),
            "last_episode_audit": dict(self.last_episode_audit),
            "certification_used": False,
            "vcg_guard_used": False,
            **metadata,
        }

    @classmethod
    def from_checkpoint(
        cls, env, payload: dict, *, learning_enabled: bool = False,
        device: str | torch.device = "cpu", seed: int = 0,
        policy_seed: Optional[int] = None,
        deployment_mode: str = DEPLOYMENT_STOCHASTIC,
    ):
        expected = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "assignment_source": METHOD_NAME,
            "selector_feature_version": FEATURE_VERSION,
            "selector_architecture": ARCHITECTURE_NAME,
            "reward_contract": REWARD_CONTRACT,
            "return_contract": RETURN_CONTRACT,
            "action_mapping": ACTION_MAPPING,
            "input_channels": INPUT_CHANNELS,
        }
        bad = [key for key, value in expected.items() if payload.get(key) != value]
        if bad:
            raise ValueError("incompatible objective-matched Kim checkpoint: " + ", ".join(bad))
        if payload.get("geometry_contract") != geometry_contract(env):
            raise ValueError("objective-matched Kim checkpoint geometry differs")
        source = cls(
            env, Kim2020Config(**payload["config"]), seed=seed,
            policy_seed=seed if policy_seed is None else policy_seed,
            learning_enabled=learning_enabled, deployment_mode=deployment_mode,
            device=device,
        )
        source.network.load_state_dict(
            payload["network_state_dict"] if learning_enabled
            else payload.get("deployment_state_dict", payload["network_state_dict"])
        )
        if learning_enabled:
            source.optimizer.load_state_dict(payload["optimizer_state_dict"])
            for state in source.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(source.device)
            source.training_rng.bit_generator.state = payload["training_rng_state"]
        for name in ("assignment_count", "gradient_steps", "training_episode_count", "skipped_episode_count"):
            setattr(source, name, int(payload.get(name, 0)))
        for name in ("loss_history", "actor_loss_history", "value_loss_history", "entropy_history", "return_history"):
            setattr(source, name, [float(v) for v in payload.get(name, ())])
        source.set_learning_enabled(learning_enabled)
        return source
