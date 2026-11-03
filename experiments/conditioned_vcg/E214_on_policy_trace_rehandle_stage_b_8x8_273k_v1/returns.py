"""Round-frozen on-policy trace returns for future physical rehandles."""

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
class TraceExample:
    feature: Tensor
    preference_lambda: float
    trace_target: float
    mc_future_rehandles: float


def examples_from_episode(
    samples: Sequence[FutureHandlingSample],
    target_network,
    *,
    trace_eta: float,
    device,
) -> tuple[TraceExample, ...]:
    """Return current-policy eta-returns for N(s,c), excluding current I(c)."""

    samples = tuple(samples)
    if not samples:
        raise ValueError("trace-return evaluation requires a nonempty episode")
    eta = float(trace_eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("trace eta must lie in [0,1]")
    alpha = float(samples[0].preference_lambda)
    if any(
        not math.isclose(float(item.preference_lambda), alpha, abs_tol=0.0)
        for item in samples
    ):
        raise ValueError("preference changed inside a trace-return episode")
    features = torch.stack([item.feature for item in samples]).to(device)
    lambdas = torch.full(
        (len(samples),), alpha, dtype=features.dtype, device=device
    )
    target_network.requires_grad_(False).eval()
    with torch.no_grad():
        bootstrap = target_network(features, lambdas).detach().cpu()
    targets = [0.0] * len(samples)
    for index in range(len(samples) - 2, -1, -1):
        next_immediate = float(samples[index + 1].immediate_rehandles)
        targets[index] = (
            next_immediate
            + (1.0 - eta) * float(bootstrap[index + 1].item())
            + eta * targets[index + 1]
        )
    return tuple(
        TraceExample(
            feature=item.feature.detach().cpu().clone(),
            preference_lambda=alpha,
            trace_target=float(targets[index]),
            mc_future_rehandles=float(item.future_rehandles),
        )
        for index, item in enumerate(samples)
    )


def _batch(examples, device):
    features = torch.stack([item.feature for item in examples]).to(device)
    lambdas = torch.as_tensor(
        [item.preference_lambda for item in examples],
        dtype=features.dtype,
        device=device,
    )
    trace = torch.as_tensor(
        [item.trace_target for item in examples],
        dtype=features.dtype,
        device=device,
    )
    mc = torch.as_tensor(
        [item.mc_future_rehandles for item in examples],
        dtype=features.dtype,
        device=device,
    )
    return features, lambdas, trace, mc


def diagnostics(network, examples, *, device) -> dict:
    examples = tuple(examples)
    if not examples:
        raise ValueError("trace-return diagnostics require examples")
    features, lambdas, trace, mc = _batch(examples, device)
    network.eval()
    with torch.no_grad():
        prediction = network(features, lambdas)
    trace_error = prediction - trace
    mc_error = prediction - mc
    zero_mc_mae = float(mc.abs().mean().item())
    mc_mae = float(mc_error.abs().mean().item())
    return {
        "sample_count": len(examples),
        "trace_mae": float(trace_error.abs().mean().item()),
        "trace_bias": float(trace_error.mean().item()),
        "trace_rmse": float(torch.sqrt((trace_error * trace_error).mean()).item()),
        "mc_mae": mc_mae,
        "mc_bias": float(mc_error.mean().item()),
        "mc_rmse": float(torch.sqrt((mc_error * mc_error).mean()).item()),
        "zero_mc_mae": zero_mc_mae,
        "mc_mae_gain_over_zero": zero_mc_mae - mc_mae,
        "mc_mae_ratio_to_zero": (
            mc_mae / zero_mc_mae if zero_mc_mae > 0.0 else None
        ),
        "prediction_mean": float(prediction.mean().item()),
        "trace_target_mean": float(trace.mean().item()),
        "mc_target_mean": float(mc.mean().item()),
    }


def fit_round(
    network,
    optimizer,
    examples,
    *,
    device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    examples = tuple(examples)
    if not examples:
        raise ValueError("trace-return fit requires examples")
    before = diagnostics(network, examples, device=device)
    rng = random.Random(int(seed))
    network.requires_grad_(True).train()
    losses = []
    gradient_steps = 0
    for _epoch in range(int(epochs)):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for start in range(0, len(order), int(batch_size)):
            selected = [
                examples[index] for index in order[start:start + int(batch_size)]
            ]
            features, lambdas, trace, _mc = _batch(selected, device)
            prediction = network(features, lambdas)
            loss = F.smooth_l1_loss(prediction, trace, beta=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite E214 Stage-B gradient")
            optimizer.step()
            losses.append(float(loss.detach().item()))
            gradient_steps += 1
    network.requires_grad_(False).eval()
    after = diagnostics(network, examples, device=device)
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "gradient_steps": gradient_steps,
        "loss_mean": float(fmean(losses)),
        "before": before,
        "after": after,
    }


__all__ = ["TraceExample", "diagnostics", "examples_from_episode", "fit_round"]
