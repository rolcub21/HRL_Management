"""Fresh initialization with the exact E189 network/input architecture."""

from __future__ import annotations

import torch

from viability_graph_episodic_audit import (
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_SCHEMA_VERSION,
    EpisodicViabilityGraphConfig,
)
from experiments.conditioned_vcg.E187_wait_corrected_e78_246k_v1 import (
    representation as e187,
)
from experiments.conditioned_vcg.E188_mask_only_progress_guard_247k_v1 import (
    representation as e188,
)


# These identities are intentionally identical to E189/E188.  Initialization
# provenance changes; tensor topology and observation semantics do not.
CONTROLLER_ARCHITECTURE = e188.CONTROLLER_ARCHITECTURE
NETWORK_ARCHITECTURE = e188.NETWORK_ARCHITECTURE
TIME_CONTEXT_CONTRACT = e188.TIME_CONTEXT_CONTRACT
QUEUE_CONTEXT_NAMES = e188.QUEUE_CONTEXT_NAMES
QUEUE_CONTEXT_DIM = e188.QUEUE_CONTEXT_DIM


class FreshChoicePreservingQueueAgent(e188.ChoicePreservingQueueAgent):
    """E188 agent with every learned tensor initialized from its model seed."""

    # E187 checks this attribute through ``self``.  Keeping it true at the
    # subclass makes accidental E78 transfer impossible in every construction
    # path, including validation clones and resumptions.
    _suppress_warm_start = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.e187_warm_start_manifest is not None:
            raise RuntimeError("E191 unexpectedly loaded an E78 encoder")

    def checkpoint_metadata(self) -> dict:
        value = super().checkpoint_metadata()
        value.update(
            controller_architecture=CONTROLLER_ARCHITECTURE,
            network_architecture=NETWORK_ARCHITECTURE,
            time_context_contract=TIME_CONTEXT_CONTRACT,
            queue_context_names=QUEUE_CONTEXT_NAMES,
            queue_context_dim=QUEUE_CONTEXT_DIM,
            e187_warm_start_manifest=None,
            e191_fresh_random_initialization=True,
            e191_transferred_modules=(),
            e191_architecture_identical_to_e189=True,
            representation_difference_from_vcg_v1=(
                "unchanged E189 eight-field public queue/progress context; "
                "all network tensors freshly initialized"
            ),
        )
        return value

    @classmethod
    def from_checkpoint(
        cls,
        payload: dict,
        *,
        device: str | torch.device = "cpu",
        resumable: bool = False,
        seed: int = 0,
    ) -> "FreshChoicePreservingQueueAgent":
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
            "recovery_liveness_guard": e188.GUARD_CONTRACT,
            "e188_guard_contract": e188.GUARD_CONTRACT,
            "e188_choice_preserving_progress_guard": True,
            "e188_only_method_delta_from_e187": True,
            "e191_fresh_random_initialization": True,
            "e191_transferred_modules": (),
            "e191_architecture_identical_to_e189": True,
        }
        mismatches = {
            name: (payload.get(name), wanted)
            for name, wanted in expected.items()
            if payload.get(name) != wanted
        }
        if mismatches:
            raise ValueError(f"incompatible E191 checkpoint: {mismatches!r}")
        if payload.get("e187_warm_start_manifest") is not None:
            raise ValueError("E191 checkpoint contains transferred weights")
        if payload.get("future_schedule_visible_to_policy", False) is not False:
            raise ValueError("E191 checkpoint exposes the future schedule")

        agent = cls(
            config=EpisodicViabilityGraphConfig.from_dict(payload["config"]),
            seed=seed,
            device=device,
            epsilon=float(payload.get("agent_state", {}).get("epsilon", 0.0)),
        )
        agent.load_checkpoint_state(payload["agent_state"], resumable=resumable)
        return agent


def architecture_signature(agent) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(int(value) for value in tensor.shape)
        for name, tensor in agent.Q_local.state_dict().items()
    }


PublicQueueContext = e187.PublicQueueContext
public_queue_context = e187.public_queue_context
attach_queue_context = e187.attach_queue_context
deduplicated_queue_scores = e187.deduplicated_queue_scores
acceleration_enabled = e187.acceleration_enabled


__all__ = [
    "CONTROLLER_ARCHITECTURE", "FreshChoicePreservingQueueAgent",
    "NETWORK_ARCHITECTURE", "PublicQueueContext", "QUEUE_CONTEXT_DIM",
    "QUEUE_CONTEXT_NAMES", "TIME_CONTEXT_CONTRACT", "acceleration_enabled",
    "architecture_signature", "attach_queue_context",
    "deduplicated_queue_scores", "public_queue_context",
]
