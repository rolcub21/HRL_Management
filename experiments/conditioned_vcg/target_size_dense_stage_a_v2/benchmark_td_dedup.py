#!/usr/bin/env python3
"""Read-only parity/timing check for state-deduplicated Stage-A TD scoring."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter

import torch
from torch import nn
import torch.nn.functional as F

from PSLAP.viability import RecoveryState
from PSLAP.yard_graph import encode_recovery_state, pad_yard_graphs
from viability_graph_hierarchy import (
    ACTION_FEATURE_DIM,
    ViabilityGraphCandidateRecord,
    double_dqn_regularized_continuation,
)
from viability_graph_episodic_audit import (
    EpisodicViabilityGraphHierarchyAgent,
    episodic_common_smdp_target,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def deduplicate(states):
    unique = []
    indices = {}
    inverse = []
    for state in states:
        if not isinstance(state, RecoveryState):
            raise TypeError("every candidate state must be a RecoveryState")
        index = indices.get(state)
        if index is None:
            index = len(unique)
            unique.append(state)
            indices[state] = index
        inverse.append(index)
    return tuple(unique), tuple(inverse)


def deduplicated_forward(network, current_states, successor_states, action_features):
    current_states = tuple(current_states)
    successor_states = tuple(successor_states)
    if not current_states:
        raise ValueError("Q network requires at least one candidate state")
    if len(current_states) != len(successor_states):
        raise ValueError("current and successor state counts must match")
    parameter = next(network.parameters())
    features = torch.as_tensor(
        action_features, dtype=parameter.dtype, device=parameter.device
    )
    if features.ndim == 1:
        features = features.unsqueeze(0)
    if features.shape != (len(current_states), ACTION_FEATURE_DIM):
        raise ValueError("action feature shape mismatch")

    states = current_states + successor_states
    unique, inverse = deduplicate(states)
    graphs = tuple(
        encode_recovery_state(state, timing_scale=network.config.timing_scale)
        for state in unique
    )
    unique_embeddings = network.graph_encoder(pad_yard_graphs(graphs))
    gather = torch.as_tensor(inverse, dtype=torch.long, device=parameter.device)
    embeddings = unique_embeddings.index_select(0, gather)
    count = len(current_states)
    current = embeddings[:count]
    successor = embeddings[count:]
    action = network.action_encoder(features)
    return network.q_head(
        torch.cat((current, successor, successor - current, action), dim=-1)
    ).squeeze(-1)


def score_records(records, network, *, accelerated: bool, grad: bool):
    records = tuple(records)
    current = tuple(record.current_state for record in records)
    successor = tuple(record.successor_state for record in records)
    features = tuple(record.action_features for record in records)
    function = deduplicated_forward if accelerated else network
    if grad:
        return function(network, current, successor, features) if accelerated else function(current, successor, features)
    with torch.no_grad():
        return function(network, current, successor, features) if accelerated else function(current, successor, features)


def td_batch(agent, transitions, *, accelerated: bool):
    transitions = tuple(transitions)
    prediction = score_records(
        (item.chosen for item in transitions),
        agent.Q_local,
        accelerated=accelerated,
        grad=True,
    )
    next_values = torch.zeros(
        len(transitions), dtype=prediction.dtype, device=agent.device
    )
    records = [
        record
        for transition in transitions
        if not transition.done
        for record in transition.next_candidates
    ]
    if records:
        online = score_records(
            records, agent.Q_local, accelerated=accelerated, grad=False
        )
        target = score_records(
            records, agent.Q_target, accelerated=accelerated, grad=False
        )
        offset = 0
        for index, transition in enumerate(transitions):
            if transition.done:
                continue
            count = len(transition.next_candidates)
            modes = torch.as_tensor(
                [record.mode_id for record in transition.next_candidates],
                dtype=torch.long,
                device=agent.device,
            )
            next_values[index] = double_dqn_regularized_continuation(
                online[offset : offset + count],
                target[offset : offset + count],
                modes,
                agent.within_temperatures,
                agent.config.tau_mode,
            )
            offset += count
    rewards = torch.as_tensor(
        [item.reward for item in transitions],
        dtype=prediction.dtype,
        device=agent.device,
    )
    durations = torch.as_tensor(
        [item.duration for item in transitions],
        dtype=torch.long,
        device=agent.device,
    )
    dones = torch.as_tensor(
        [item.done for item in transitions],
        dtype=torch.bool,
        device=agent.device,
    )
    target = episodic_common_smdp_target(
        rewards,
        durations,
        dones,
        next_values,
        gamma=agent.config.gamma,
        reward_scale=agent.config.reward_scale,
    )
    return prediction, target.detach()


def one_update(agent, transitions, *, accelerated: bool):
    synchronize(agent.device)
    started = perf_counter()
    prediction, target = td_batch(
        agent, transitions, accelerated=accelerated
    )
    loss = F.smooth_l1_loss(
        prediction, target, beta=agent.config.huber_delta
    )
    agent.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(agent.Q_local.parameters(), agent.config.grad_clip)
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in agent.Q_local.named_parameters()
        if parameter.grad is not None
    }
    agent.optimizer.step()
    synchronize(agent.device)
    seconds = perf_counter() - started
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in agent.Q_local.state_dict().items()
    }
    return {
        "seconds": seconds,
        "prediction": prediction.detach().cpu(),
        "target": target.detach().cpu(),
        "loss": loss.detach().cpu(),
        "gradients": gradients,
        "state": state,
    }


def max_difference(left, right) -> float:
    return max(
        (float((left[name] - right[name]).abs().max()) for name in left),
        default=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(args.device)
    before_sha = sha256(args.checkpoint)
    before_mtime = args.checkpoint.stat().st_mtime_ns
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    reference = EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
        deepcopy(payload), device=device, resumable=True, seed=int(payload["model_seed"])
    )
    accelerated = EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
        deepcopy(payload), device=device, resumable=True, seed=int(payload["model_seed"])
    )
    if args.batch_size > len(reference.replay):
        raise ValueError("batch exceeds saved replay")
    transitions = reference.replay.sample_mode_balanced(
        args.batch_size, reference.rng
    )
    next_records = sum(
        len(transition.next_candidates)
        for transition in transitions
        if not transition.done
    )
    all_next = [
        record
        for transition in transitions
        if not transition.done
        for record in transition.next_candidates
    ]
    all_states = [record.current_state for record in all_next] + [
        record.successor_state for record in all_next
    ]
    unique_states, _ = deduplicate(all_states)

    native = one_update(reference, transitions, accelerated=False)
    fast = one_update(accelerated, transitions, accelerated=True)
    prediction_diff = float((native["prediction"] - fast["prediction"]).abs().max())
    target_diff = float((native["target"] - fast["target"]).abs().max())
    loss_diff = float((native["loss"] - fast["loss"]).abs())
    gradient_diff = max_difference(native["gradients"], fast["gradients"])
    state_diff = max_difference(native["state"], fast["state"])
    tolerances = {
        "prediction": 2e-5,
        "target": 2e-5,
        "loss": 2e-6,
        "gradient": 2e-5,
        "optimizer_state": 2e-5,
    }
    parity = bool(
        prediction_diff <= tolerances["prediction"]
        and target_diff <= tolerances["target"]
        and loss_diff <= tolerances["loss"]
        and gradient_diff <= tolerances["gradient"]
        and state_diff <= tolerances["optimizer_state"]
    )
    after_sha = sha256(args.checkpoint)
    after_mtime = args.checkpoint.stat().st_mtime_ns
    if before_sha != after_sha or before_mtime != after_mtime:
        raise RuntimeError("authoritative checkpoint changed during benchmark")
    report = {
        "status": "passed" if parity else "parity_failed",
        "purpose": "engineering benchmark; not experimental evidence",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256_before_after": [before_sha, after_sha],
        "checkpoint_unchanged": True,
        "device": str(device),
        "threads": args.threads,
        "batch_size": args.batch_size,
        "next_candidate_records": next_records,
        "native_state_occurrences": 2 * next_records,
        "deduplicated_next_states": len(unique_states),
        "native_seconds": native["seconds"],
        "deduplicated_seconds": fast["seconds"],
        "speedup": native["seconds"] / fast["seconds"],
        "maximum_absolute_differences": {
            "prediction": prediction_diff,
            "target": target_diff,
            "loss": loss_diff,
            "gradient": gradient_diff,
            "optimizer_state": state_diff,
        },
        "tolerances": tolerances,
        "training_runs": 0,
        "episode_runs": 0,
        "production_models_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not parity:
        raise RuntimeError("deduplicated TD update failed numerical parity")


if __name__ == "__main__":
    main()
