"""E191 network and public observation, freshly initialized for E209."""

from __future__ import annotations

import torch

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    representation as e191,
)


CONTROLLER_ARCHITECTURE = e191.CONTROLLER_ARCHITECTURE
NETWORK_ARCHITECTURE = e191.NETWORK_ARCHITECTURE
TIME_CONTEXT_CONTRACT = e191.TIME_CONTEXT_CONTRACT
QUEUE_CONTEXT_NAMES = e191.QUEUE_CONTEXT_NAMES
QUEUE_CONTEXT_DIM = e191.QUEUE_CONTEXT_DIM


class FreshScaleMatched10x10Agent(e191.FreshChoicePreservingQueueAgent):
    """Exact E191 tensor topology with explicit E209 checkpoint identity."""

    def checkpoint_metadata(self) -> dict:
        value = super().checkpoint_metadata()
        value.update(
            e209_scale_matched_10x10=True,
            e209_architecture_identical_to_e191=True,
            e209_fresh_random_initialization=True,
            e209_transferred_modules=(),
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
    ) -> "FreshScaleMatched10x10Agent":
        expected = {
            "e209_scale_matched_10x10": True,
            "e209_architecture_identical_to_e191": True,
            "e209_fresh_random_initialization": True,
            "e209_transferred_modules": (),
        }
        mismatches = {
            name: (payload.get(name), wanted)
            for name, wanted in expected.items()
            if payload.get(name) != wanted
        }
        if mismatches:
            raise ValueError(f"incompatible E209 checkpoint: {mismatches!r}")
        return super().from_checkpoint(
            payload, device=device, resumable=resumable, seed=seed
        )


architecture_signature = e191.architecture_signature
PublicQueueContext = e191.PublicQueueContext
public_queue_context = e191.public_queue_context
attach_queue_context = e191.attach_queue_context
deduplicated_queue_scores = e191.deduplicated_queue_scores
acceleration_enabled = e191.acceleration_enabled


__all__ = [
    "CONTROLLER_ARCHITECTURE", "FreshScaleMatched10x10Agent",
    "NETWORK_ARCHITECTURE", "PublicQueueContext", "QUEUE_CONTEXT_DIM",
    "QUEUE_CONTEXT_NAMES", "TIME_CONTEXT_CONTRACT", "acceleration_enabled",
    "architecture_signature", "attach_queue_context",
    "deduplicated_queue_scores", "public_queue_context",
]
