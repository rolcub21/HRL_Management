"""Immediate-only centered rehandle controller."""

from __future__ import annotations

from torch import nn
import torch

from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1.centered import (
    CenteredHandlingAgent,
)


CONTROLLER_ARCHITECTURE = "e207_e191_centered_immediate_rehandle_only_v1"


class ZeroFutureHandlingNetwork(nn.Module):
    """Exact zero continuation term with no learned parameters."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, features, preference_lambda):
        self.config.normalized_lambda(
            preference_lambda,
            count=features.shape[0],
            dtype=features.dtype,
            device=features.device,
        )
        return torch.zeros(
            features.shape[0], dtype=features.dtype, device=features.device
        )


class ImmediateOnlyCenteredAgent(CenteredHandlingAgent):
    """Use E204's selector with future handling fixed exactly to zero."""

    def audit(self, *, include_decisions: bool = False) -> dict:
        value = super().audit(include_decisions=include_decisions)
        return {
            **value,
            "controller_architecture": CONTROLLER_ARCHITECTURE,
            "future_handling_source": "constant_exact_zero",
            "learned_future_handling_used": False,
            "deployment_merit": "Q_E191 - 0.01*(lambda-10)*I_reconfigure",
        }


def from_stage_b_agent(stage_b_agent, *, seed: int) -> ImmediateOnlyCenteredAgent:
    network = ZeroFutureHandlingNetwork(stage_b_agent.config).to(
        stage_b_agent.base_agent.device
    )
    network.requires_grad_(False).eval()
    return ImmediateOnlyCenteredAgent(
        stage_b_agent.base_agent,
        network,
        config=stage_b_agent.config,
        seed=int(seed),
        epsilon=0.0,
    )


__all__ = [
    "CONTROLLER_ARCHITECTURE", "ImmediateOnlyCenteredAgent",
    "ZeroFutureHandlingNetwork", "from_stage_b_agent",
]
