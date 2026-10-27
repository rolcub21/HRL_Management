"""Observable queue context and representation-only E78 warm start for E187."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import math
from pathlib import Path
from typing import Optional

import torch

from viability_graph_episodic_audit import (
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
    EpisodicViabilityGraphConfig,
    EpisodicViabilityGraphHierarchyAgent,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    representation as e107,
)


ROOT = Path(__file__).resolve().parents[3]
CONTROLLER_ARCHITECTURE = "e187_wait_corrected_queue_observed_graph_smdp_q_v1"
NETWORK_ARCHITECTURE = "e187_e78_graph_warm_queue_context_scalar_q_v1"
TIME_CONTEXT_CONTRACT = (
    "bounded_current_public_queue_progress_context_no_future_schedule_v1"
)
QUEUE_CONTEXT_NAMES = (
    "waiting_fraction",
    "oldest_wait_age_fraction_of_horizon",
    "inbound_wait_age_fraction_of_horizon",
    "total_wait_age_fraction_of_horizon_capacity",
    "released_fraction",
    "accepted_fraction",
    "delivered_fraction",
    "elapsed_fraction",
)
QUEUE_CONTEXT_DIM = len(QUEUE_CONTEXT_NAMES)
if QUEUE_CONTEXT_DIM != e107.QUEUE_CONTEXT_DIM:
    raise RuntimeError("E187 queue context no longer matches the reused network width")

TRANSFER_PREFIXES = ("graph_encoder.",)
SOURCE_CHECKPOINTS = {
    0: {
        "path": (
            "results/vcg-conditioned-e70-original-vcg-reward-base-135k-v1/"
            "training/operational/empty_8x8_n36/seed-0/best.pth"
        ),
        "sha256": "eaaea1fad0618c57e52825c2172ee940ef41a8b6b568b989a25b686f582255cb",
        "protocol": "vcg_conditioned_e70_original_vcg_reward_base_135k_v1",
        "selected_episode": 36,
    },
    1: {
        "path": (
            "results/vcg-conditioned-e78-original-base80-full-8x8-143k-v1/"
            "training/operational/empty_8x8_n36/seed-1/best.pth"
        ),
        "sha256": "bd8cdf1cb6bfcad84def145e5ab9ea2cd7efb15d100b1308e96a433f2e5a4867",
        "protocol": "vcg_conditioned_e78_original_base80_full_8x8_143k_v1",
        "selected_episode": 108,
    },
    2: {
        "path": (
            "results/vcg-conditioned-e78-original-base80-full-8x8-143k-v1/"
            "training/operational/empty_8x8_n36/seed-2/best.pth"
        ),
        "sha256": "a2943f1c520776a76bb42608e19abc401ab840e67b0a4dc555b8214303d00d01",
        "protocol": "vcg_conditioned_e78_original_base80_full_8x8_143k_v1",
        "selected_episode": 84,
    },
}


@lru_cache(maxsize=None)
def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, dict]:
    return {
        str(seed): {
            **value,
            "path": value["path"],
            "transfer_prefixes": list(TRANSFER_PREFIXES),
            "source_completed_episodes": 112,
            "source_checkpoint_role": "best_deployment_finalized",
        }
        for seed, value in SOURCE_CHECKPOINTS.items()
    }


def authenticate_sources() -> dict[int, dict]:
    manifest = source_manifest()
    authenticated = {}
    for text_seed, value in manifest.items():
        seed = int(text_seed)
        path = (ROOT / value["path"]).resolve()
        if not path.is_file():
            raise RuntimeError(f"E187 source checkpoint is missing: {path}")
        observed = file_sha256(str(path))
        if observed != value["sha256"]:
            raise RuntimeError(f"E187 source checkpoint hash changed for seed {seed}")
        authenticated[seed] = value
    return authenticated


@dataclass(frozen=True)
class PublicQueueContext:
    values: tuple[float, ...]
    waiting_jobs: int
    oldest_wait_age: int
    inbound_wait_age: Optional[int]
    total_wait_age: int
    released_jobs: int
    accepted_jobs: int
    delivered_jobs: int
    time_step: int

    def __post_init__(self) -> None:
        if len(self.values) != QUEUE_CONTEXT_DIM:
            raise ValueError("E187 public context has the wrong width")
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in self.values
        ):
            raise ValueError("E187 public context must be finite and bounded")


def public_queue_context(
    env,
    snapshot,
    *,
    required_jobs: int,
    max_steps: int,
) -> PublicQueueContext:
    """Use current observable queue facts without reading future-job state."""

    if required_jobs <= 0 or max_steps <= 0:
        raise ValueError("invalid E187 context normalization contract")
    time_step = int(env.time_steps)
    if int(snapshot.decision_epoch) != time_step:
        raise ValueError("snapshot and E187 context are from different times")
    blocks = tuple(env.blocks)
    if len(blocks) != required_jobs:
        raise ValueError("E187 workload differs from its context contract")

    # Only arrival_step is inspected before release.  All other attributes are
    # read after the job has become observable to the controller.
    released = tuple(
        block for block in blocks if int(block.arrival_step) <= time_step
    )
    outside = tuple(
        block
        for block in released
        if getattr(block, "stored_time_step", None) is None
        and not bool(block.delivered)
    )
    ages = {
        str(block.label): max(0, time_step - int(block.arrival_step))
        for block in outside
    }
    oldest = max(ages.values(), default=0)
    inbound_age = ages.get(str(snapshot.inbound_label))
    total_age = sum(ages.values())
    accepted = sum(
        getattr(block, "stored_time_step", None) is not None
        for block in released
    )
    delivered = sum(bool(block.delivered) for block in released)
    jobs = float(required_jobs)
    values = (
        len(outside) / jobs,
        min(oldest / max_steps, 1.0),
        0.0 if inbound_age is None else min(inbound_age / max_steps, 1.0),
        min(total_age / (max_steps * jobs), 1.0),
        len(released) / jobs,
        accepted / jobs,
        delivered / jobs,
        min(time_step / max_steps, 1.0),
    )
    return PublicQueueContext(
        values=tuple(float(value) for value in values),
        waiting_jobs=len(outside),
        oldest_wait_age=oldest,
        inbound_wait_age=inbound_age,
        total_wait_age=total_age,
        released_jobs=len(released),
        accepted_jobs=accepted,
        delivered_jobs=delivered,
        time_step=time_step,
    )


def _tensor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class WaitCorrectedQueueAgent(e107.QueueContextAgent):
    """E107 queue architecture with a clean E78 graph-encoder transfer."""

    _suppress_warm_start = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.context_required_jobs = 36
        self.context_max_steps = self.config.episode_horizon_steps
        self.e187_warm_start_manifest = None
        if not self._suppress_warm_start and int(self.seed) in SOURCE_CHECKPOINTS:
            self.e187_warm_start_manifest = self._load_graph_encoder_source(
                int(self.seed)
            )

    def _load_graph_encoder_source(self, seed: int) -> dict:
        source = authenticate_sources()[seed]
        path = (ROOT / source["path"]).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "model_seed": seed,
            "completed_training_episodes": 112,
            "selected_checkpoint_episode": source["selected_episode"],
            "training_protocol": source["protocol"],
            "checkpoint_role": "best_deployment_finalized",
        }
        mismatches = {
            name: (payload.get(name), value)
            for name, value in expected.items()
            if payload.get(name) != value
        }
        if mismatches:
            raise RuntimeError(f"E187 source identity mismatch: {mismatches!r}")
        source_state = payload["agent_state"]["Q_local"]
        encoder_state = {
            name.removeprefix("graph_encoder."): value
            for name, value in source_state.items()
            if name.startswith("graph_encoder.")
        }
        expected_names = set(self.Q_local.graph_encoder.state_dict())
        if set(encoder_state) != expected_names:
            raise RuntimeError("E187 source graph encoder tensor set differs")
        self.Q_local.graph_encoder.load_state_dict(encoder_state, strict=True)
        self.Q_target.graph_encoder.load_state_dict(encoder_state, strict=True)
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()
        result = {
            **source,
            "transferred_tensor_count": len(encoder_state),
            "transferred_tensor_sha256": _tensor_digest(encoder_state),
            "transferred_modules": ["graph_encoder"],
            "reset_modules": ["action_encoder", "context_encoder", "q_head"],
            "optimizer_reset": True,
            "replay_reset": True,
            "counters_reset": True,
        }
        del payload
        return result

    def bind_context_environment(
        self,
        env,
        *,
        required_jobs: int,
        max_steps: int,
    ) -> None:
        self.context_environment = env
        self.context_required_jobs = int(required_jobs)
        self.context_max_steps = int(max_steps)

    def _admissible_prepared(self, snapshot):
        prepared, forced = (
            EpisodicViabilityGraphHierarchyAgent._admissible_prepared(
                self, snapshot
            )
        )
        if self.context_environment is None:
            raise RuntimeError("E187 agent is not bound to its live environment")
        context = public_queue_context(
            self.context_environment,
            snapshot,
            required_jobs=self.context_required_jobs,
            max_steps=self.context_max_steps,
        )
        current = e107.attach_queue_context(
            snapshot.recovery_state, context.values
        )
        records = tuple(
            replace(record, current_state=current)
            for record in prepared.records
        )
        return replace(prepared, records=records), forced

    def checkpoint_metadata(self) -> dict:
        result = super().checkpoint_metadata()
        result.update(
            controller_architecture=CONTROLLER_ARCHITECTURE,
            network_architecture=NETWORK_ARCHITECTURE,
            time_context_contract=TIME_CONTEXT_CONTRACT,
            remaining_budget_observed_by_q=True,
            elapsed_time_observed_by_q=True,
            representation_difference_from_vcg_v1=(
                "public queue/progress context plus graph-encoder warm start"
            ),
            queue_context_names=QUEUE_CONTEXT_NAMES,
            queue_context_dim=QUEUE_CONTEXT_DIM,
            queue_context_stored_in_replay=True,
            q_context_composition=(
                "current_graph_successor_graph_delta_action_current_queue_context_v1"
            ),
            e187_warm_start_manifest=self.e187_warm_start_manifest,
            e187_value_head_reset=True,
            e187_optimizer_replay_counters_reset=True,
        )
        return result

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict,
        *,
        device: str | torch.device = "cpu",
        resumable: bool = False,
        seed: int = 0,
    ) -> "WaitCorrectedQueueAgent":
        expected = {
            "checkpoint_schema_version": (
                EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION
            ),
            "checkpoint_family": EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "network_architecture": NETWORK_ARCHITECTURE,
            "time_context_contract": TIME_CONTEXT_CONTRACT,
            "queue_context_names": QUEUE_CONTEXT_NAMES,
            "queue_context_dim": QUEUE_CONTEXT_DIM,
            "queue_context_stored_in_replay": True,
            "e187_value_head_reset": True,
            "e187_optimizer_replay_counters_reset": True,
        }
        mismatches = {
            name: (payload.get(name), value)
            for name, value in expected.items()
            if payload.get(name) != value
        }
        if mismatches:
            raise ValueError(f"incompatible E187 checkpoint: {mismatches!r}")
        if payload.get("future_schedule_visible_to_policy", False) is not False:
            raise ValueError("E187 checkpoint exposes the future schedule")
        manifest = payload.get("e187_warm_start_manifest")
        if not isinstance(manifest, dict) or manifest.get("sha256") != (
            SOURCE_CHECKPOINTS[int(seed)]["sha256"]
        ):
            raise ValueError("E187 checkpoint has another warm-start identity")
        previous = cls._suppress_warm_start
        cls._suppress_warm_start = True
        try:
            agent = cls(
                config=EpisodicViabilityGraphConfig.from_dict(payload["config"]),
                seed=seed,
                device=device,
                epsilon=float(payload.get("agent_state", {}).get("epsilon", 0.0)),
            )
        finally:
            cls._suppress_warm_start = previous
        agent.e187_warm_start_manifest = dict(manifest)
        agent.load_checkpoint_state(payload["agent_state"], resumable=resumable)
        return agent


# E107's scorer operates on the same queue-state type and network class.  It
# performs exact identity deduplication and batched graph encoding.
attach_queue_context = e107.attach_queue_context
deduplicated_queue_scores = e107.deduplicated_queue_scores
acceleration_enabled = e107.acceleration_enabled


@contextmanager
def warm_start_disabled():
    previous = WaitCorrectedQueueAgent._suppress_warm_start
    WaitCorrectedQueueAgent._suppress_warm_start = True
    try:
        yield
    finally:
        WaitCorrectedQueueAgent._suppress_warm_start = previous


__all__ = [
    "CONTROLLER_ARCHITECTURE",
    "NETWORK_ARCHITECTURE",
    "PublicQueueContext",
    "QUEUE_CONTEXT_DIM",
    "QUEUE_CONTEXT_NAMES",
    "SOURCE_CHECKPOINTS",
    "TIME_CONTEXT_CONTRACT",
    "TRANSFER_PREFIXES",
    "WaitCorrectedQueueAgent",
    "acceleration_enabled",
    "attach_queue_context",
    "authenticate_sources",
    "deduplicated_queue_scores",
    "public_queue_context",
    "source_manifest",
    "warm_start_disabled",
]
