"""One-step on-policy evaluation of future rehandles."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import fmean
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from vcg_v11_conditioned_handling import FutureHandlingSample


@dataclass(frozen=True)
class SarsaTransition:
    feature: Tensor
    preference_lambda: float
    next_feature: Tensor
    next_immediate_rehandles: float
    terminal: bool
    mc_future_rehandles: float


def transitions_from_episode(
    samples: Sequence[FutureHandlingSample],
) -> tuple[SarsaTransition, ...]:
    """Build N(s,c)=I(c_next)+N(s_next,c_next) targets.

    The learned quantity excludes the current macro's exact immediate term.
    The next action is the one actually chosen by the same fixed-alpha behavior
    policy during collection (SARSA, not a max over another policy).
    """

    samples = tuple(samples)
    if not samples:
        raise ValueError("TD evaluation requires a complete nonempty episode")
    values = []
    zero = torch.zeros_like(samples[0].feature)
    for index, sample in enumerate(samples):
        terminal = index + 1 == len(samples)
        next_sample = None if terminal else samples[index + 1]
        if next_sample is not None and not math.isclose(
            float(next_sample.preference_lambda),
            float(sample.preference_lambda),
            abs_tol=0.0,
        ):
            raise ValueError("preference changed inside a TD episode")
        values.append(SarsaTransition(
            feature=sample.feature.detach().cpu().clone(),
            preference_lambda=float(sample.preference_lambda),
            next_feature=(
                zero.detach().cpu().clone()
                if terminal else next_sample.feature.detach().cpu().clone()
            ),
            next_immediate_rehandles=(
                0.0 if terminal else float(next_sample.immediate_rehandles)
            ),
            terminal=terminal,
            mc_future_rehandles=float(sample.future_rehandles),
        ))
    return tuple(values)


def _batch_tensors(transitions, device):
    features = torch.stack([item.feature for item in transitions]).to(device)
    next_features = torch.stack([item.next_feature for item in transitions]).to(device)
    lambdas = torch.as_tensor(
        [item.preference_lambda for item in transitions],
        dtype=features.dtype,
        device=device,
    )
    next_immediate = torch.as_tensor(
        [item.next_immediate_rehandles for item in transitions],
        dtype=features.dtype,
        device=device,
    )
    terminal = torch.as_tensor(
        [item.terminal for item in transitions],
        dtype=torch.bool,
        device=device,
    )
    mc = torch.as_tensor(
        [item.mc_future_rehandles for item in transitions],
        dtype=features.dtype,
        device=device,
    )
    return features, next_features, lambdas, next_immediate, terminal, mc


def diagnostics(network, transitions, *, device) -> dict:
    transitions = tuple(transitions)
    features, next_features, lambdas, next_immediate, terminal, mc = _batch_tensors(
        transitions, device
    )
    network.eval()
    with torch.no_grad():
        prediction = network(features, lambdas)
        continuation = network(next_features, lambdas)
        target = next_immediate + torch.where(
            terminal, torch.zeros_like(continuation), continuation
        )
    td_error = prediction - target
    mc_error = prediction - mc
    return {
        "sample_count": len(transitions),
        "td_mae": float(td_error.abs().mean().item()),
        "td_bias": float(td_error.mean().item()),
        "td_rmse": float(torch.sqrt((td_error * td_error).mean()).item()),
        "mc_mae": float(mc_error.abs().mean().item()),
        "mc_bias": float(mc_error.mean().item()),
        "mc_rmse": float(torch.sqrt((mc_error * mc_error).mean()).item()),
        "prediction_mean": float(prediction.mean().item()),
        "mc_target_mean": float(mc.mean().item()),
    }


def fit_episode_td(
    network,
    target_network,
    optimizer,
    transitions,
    *,
    device,
    epochs: int,
    batch_size: int,
    seed: int,
    target_tau: float,
) -> dict:
    transitions = tuple(transitions)
    if not transitions:
        raise ValueError("TD fit requires transitions")
    if not 0.0 < float(target_tau) <= 1.0:
        raise ValueError("target tau must be in (0,1]")
    before = diagnostics(network, transitions, device=device)
    rng = random.Random(int(seed))
    network.requires_grad_(True).train()
    target_network.requires_grad_(False).eval()
    losses = []
    gradient_steps = 0
    for _epoch in range(int(epochs)):
        order = list(range(len(transitions)))
        rng.shuffle(order)
        for start in range(0, len(order), int(batch_size)):
            batch = [transitions[i] for i in order[start : start + int(batch_size)]]
            features, next_features, lambdas, next_immediate, terminal, _mc = _batch_tensors(
                batch, device
            )
            with torch.no_grad():
                continuation = target_network(next_features, lambdas)
                target = next_immediate + torch.where(
                    terminal, torch.zeros_like(continuation), continuation
                )
            prediction = network(features, lambdas)
            loss = F.smooth_l1_loss(prediction, target, beta=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite Stage-B gradient")
            optimizer.step()
            losses.append(float(loss.detach().item()))
            gradient_steps += 1
    network.requires_grad_(False).eval()
    with torch.no_grad():
        for target_parameter, online_parameter in zip(
            target_network.parameters(), network.parameters()
        ):
            target_parameter.mul_(1.0 - float(target_tau)).add_(
                online_parameter, alpha=float(target_tau)
            )
    after = diagnostics(network, transitions, device=device)
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "gradient_steps": gradient_steps,
        "loss_mean": float(fmean(losses)),
        "target_tau": float(target_tau),
        "before": before,
        "after": after,
    }


__all__ = [
    "SarsaTransition", "diagnostics", "fit_episode_td",
    "transitions_from_episode",
]
