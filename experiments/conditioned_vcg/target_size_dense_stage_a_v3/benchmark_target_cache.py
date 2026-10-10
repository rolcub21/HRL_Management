#!/usr/bin/env python3
"""Read-only multi-update benchmark for a bounded frozen-target score cache."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import MethodType

import torch

from viability_graph_episodic_audit import EpisodicViabilityGraphHierarchyAgent
from experiments.conditioned_vcg.target_size_dense_stage_a_v3 import acceleration


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def state_copy(module):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def max_difference(left, right):
    return max(
        (float((left[name] - right[name]).abs().max()) for name in left),
        default=0.0,
    )


def cached_target_scores(agent, records):
    records = tuple(records)
    generation = agent.gradient_steps // agent.config.target_update_every
    if getattr(agent, "_target_score_cache_generation", None) != generation:
        agent._target_score_cache = {}
        agent._target_score_cache_generation = generation
        agent._target_score_cache_invalidations = int(
            getattr(agent, "_target_score_cache_invalidations", 0)
        ) + 1
    cache = agent._target_score_cache
    maximum = int(agent._target_score_cache_maximum)
    identities = [id(record) for record in records]
    missing = []
    seen_missing = set()
    hits = 0
    for identity, record in zip(identities, records):
        entry = cache.get(identity)
        if entry is not None and entry[0] is record:
            hits += 1
        elif identity not in seen_missing:
            missing.append(record)
            seen_missing.add(identity)
    if missing and len(cache) + len(missing) > maximum:
        cache.clear()
        agent._target_score_cache_capacity_clears += 1
        missing = []
        seen_missing.clear()
        hits = 0
        for identity, record in zip(identities, records):
            if identity not in seen_missing:
                missing.append(record)
                seen_missing.add(identity)
    if missing:
        with torch.no_grad():
            values = acceleration._deduplicated_no_grad_scores(
                missing, agent.Q_target
            ).detach().cpu().tolist()
        for record, value in zip(missing, values):
            cache[id(record)] = (record, float(value))
    agent._target_score_cache_hits += hits
    agent._target_score_cache_requests += len(records)
    agent._target_score_cache_misses += len(records) - hits
    parameter = next(agent.Q_target.parameters())
    return torch.as_tensor(
        [cache[id(record)][1] for record in records],
        dtype=parameter.dtype,
        device=parameter.device,
    )


def cached_td_batch(agent, transitions):
    # Retain v3's native gradient prediction and deduplicated online
    # continuation. Only the frozen target-network evaluation is cached.
    transitions = tuple(transitions)
    prediction = agent._score_records(
        tuple(item.chosen for item in transitions),
        network=agent.Q_local,
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
        with torch.no_grad():
            online = acceleration._deduplicated_no_grad_scores(
                records, agent.Q_local
            )
            target = cached_target_scores(agent, records)
            offset = 0
            from viability_graph_hierarchy import double_dqn_regularized_continuation
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
    from viability_graph_episodic_audit import episodic_common_smdp_target
    targets = episodic_common_smdp_target(
        rewards,
        durations,
        dones,
        next_values,
        gamma=agent.config.gamma,
        reward_scale=agent.config.reward_scale,
    )
    return prediction, targets.detach()


def run(agent, updates, *, cached):
    if cached:
        agent._td_batch = MethodType(cached_td_batch, agent)
        agent._target_score_cache = {}
        agent._target_score_cache_generation = None
        agent._target_score_cache_maximum = 500_000
        agent._target_score_cache_hits = 0
        agent._target_score_cache_misses = 0
        agent._target_score_cache_requests = 0
        agent._target_score_cache_invalidations = 0
        agent._target_score_cache_capacity_clears = 0
    else:
        agent._td_batch = MethodType(
            acceleration.accelerated_episodic_td_batch, agent
        )
    losses = []
    timings = []
    for _ in range(updates):
        synchronize(agent.device)
        started = perf_counter()
        loss = agent.learn()
        synchronize(agent.device)
        if loss is None:
            raise RuntimeError("saved replay did not produce a learning update")
        timings.append(perf_counter() - started)
        losses.append(float(loss))
    return {
        "losses": losses,
        "seconds": timings,
        "total_seconds": sum(timings),
        "q_local": state_copy(agent.Q_local),
        "q_target": state_copy(agent.Q_target),
        "optimizer": deepcopy(agent.optimizer.state_dict()),
        "rng_state": agent.rng.getstate(),
        "gradient_steps": agent.gradient_steps,
        "target_updates": agent.target_updates,
        "cache": (
            None
            if not cached
            else {
                "entries": len(agent._target_score_cache),
                "requests": agent._target_score_cache_requests,
                "hits": agent._target_score_cache_hits,
                "misses": agent._target_score_cache_misses,
                "hit_rate": (
                    agent._target_score_cache_hits
                    / agent._target_score_cache_requests
                ),
                "invalidations": agent._target_score_cache_invalidations,
                "capacity_clears": agent._target_score_cache_capacity_clears,
            }
        ),
    }


def optimizer_tensor_difference(left, right):
    differences = []
    for left_state, right_state in zip(
        left["state"].values(), right["state"].values()
    ):
        for name in left_state:
            a, b = left_state[name], right_state[name]
            if torch.is_tensor(a):
                differences.append(float((a.cpu() - b.cpu()).abs().max()))
            elif a != b:
                return float("inf")
    return max(differences, default=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--updates", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    before_sha = sha256(args.checkpoint)
    before_mtime = args.checkpoint.stat().st_mtime_ns
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    seed = int(payload["model_seed"])
    native = EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
        deepcopy(payload), device=args.device, resumable=True, seed=seed
    )
    cached = EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
        deepcopy(payload), device=args.device, resumable=True, seed=seed
    )
    reference = run(native, args.updates, cached=False)
    candidate = run(cached, args.updates, cached=True)
    loss_difference = max(
        abs(a - b) for a, b in zip(reference["losses"], candidate["losses"])
    )
    differences = {
        "loss": loss_difference,
        "q_local": max_difference(reference["q_local"], candidate["q_local"]),
        "q_target": max_difference(reference["q_target"], candidate["q_target"]),
        "optimizer": optimizer_tensor_difference(reference["optimizer"], candidate["optimizer"]),
    }
    tolerances = {"loss": 2e-6, "q_local": 2e-5, "q_target": 2e-5, "optimizer": 2e-5}
    parity = all(differences[name] <= value for name, value in tolerances.items())
    parity = bool(
        parity
        and reference["rng_state"] == candidate["rng_state"]
        and reference["gradient_steps"] == candidate["gradient_steps"]
        and reference["target_updates"] == candidate["target_updates"]
    )
    after_sha = sha256(args.checkpoint)
    after_mtime = args.checkpoint.stat().st_mtime_ns
    if before_sha != after_sha or before_mtime != after_mtime:
        raise RuntimeError("authoritative checkpoint changed")
    report = {
        "status": "passed" if parity else "parity_failed",
        "purpose": "engineering benchmark; not experimental evidence",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256_before_after": [before_sha, after_sha],
        "checkpoint_unchanged": True,
        "device": args.device,
        "threads": args.threads,
        "updates": args.updates,
        "dedup_only_seconds": reference["seconds"],
        "dedup_plus_target_cache_seconds": candidate["seconds"],
        "dedup_only_total_seconds": reference["total_seconds"],
        "dedup_plus_target_cache_total_seconds": candidate["total_seconds"],
        "speedup": reference["total_seconds"] / candidate["total_seconds"],
        "cache": candidate["cache"],
        "maximum_absolute_differences": differences,
        "tolerances": tolerances,
        "rng_and_counters_equal": bool(
            reference["rng_state"] == candidate["rng_state"]
            and reference["gradient_steps"] == candidate["gradient_steps"]
            and reference["target_updates"] == candidate["target_updates"]
        ),
        "training_runs": 0,
        "episode_runs": 0,
        "production_models_modified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not parity:
        raise RuntimeError("target-score cache failed numerical parity")


if __name__ == "__main__":
    main()
