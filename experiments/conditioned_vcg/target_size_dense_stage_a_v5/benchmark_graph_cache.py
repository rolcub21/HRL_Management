"""Read-only same-batch comparison of Stage-A v4 and v5 TD backups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from experiments.conditioned_vcg.E85_direct_10x10_cost_audit_150k_v1 import run as e85
from experiments.conditioned_vcg.target_size_dense_stage_a_v4 import acceleration as v4
from experiments.conditioned_vcg.target_size_dense_stage_a_v5 import acceleration as v5
import run_vcg_objective_gamma_audit as audit


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _run(agent, transitions, function):
    _sync(agent.device)
    started = perf_counter()
    with torch.no_grad():
        prediction, target = function(agent, transitions)
    _sync(agent.device)
    return perf_counter() - started, prediction.detach(), target.detach()


def _clear_v4(agent) -> None:
    for name in (
        "_target_score_cache", "_target_score_cache_generation",
        "_target_score_cache_audit", "_graph_input_cache",
        "_graph_topology_cache", "_graph_input_cache_signature",
        "_graph_input_cache_audit",
    ):
        if hasattr(agent, name):
            delattr(agent, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    agent, checkpoint, payload = e85._load_agent(
        args.source_output, args.model_seed, args.device
    )
    checkpoint_before = audit._sha256(checkpoint)
    transitions = agent.replay.sample_mode_balanced(
        agent.config.batch_size, agent.rng
    )
    candidate_records = sum(
        len(item.next_candidates) for item in transitions if not item.done
    )

    _clear_v4(agent)
    v4_cold, _, _ = _run(agent, transitions, v4.accelerated_cached_td_batch)
    v4_warm, baseline_prediction, baseline_target = _run(
        agent, transitions, v4.accelerated_cached_td_batch
    )
    _clear_v4(agent)
    v5_cold, _, _ = _run(
        agent, transitions, v5.accelerated_cached_tensor_td_batch
    )
    v5_warm, candidate_prediction, candidate_target = _run(
        agent, transitions, v5.accelerated_cached_tensor_td_batch
    )
    graph_audit = v5.audit_payload(agent)
    prediction_difference = float(
        (baseline_prediction - candidate_prediction).abs().max().item()
    )
    target_difference = float(
        (baseline_target - candidate_target).abs().max().item()
    )
    parity = (
        torch.allclose(baseline_prediction, candidate_prediction,
                       rtol=1e-5, atol=5e-5)
        and torch.allclose(baseline_target, candidate_target,
                           rtol=1e-5, atol=5e-5)
    )
    if not parity:
        raise RuntimeError(
            "v5 differs from v4: prediction max "
            f"{prediction_difference}, target max {target_difference}"
        )
    checkpoint_after = audit._sha256(checkpoint)
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("source checkpoint changed during benchmark")
    result = {
        "status": "passed",
        "model_seed": args.model_seed,
        "completed_training_episodes": int(
            payload["completed_training_episodes"]
        ),
        "candidate_records": candidate_records,
        "device": str(agent.device),
        "v4_cold_seconds": v4_cold,
        "v4_warm_seconds": v4_warm,
        "v5_cold_seconds": v5_cold,
        "v5_warm_seconds": v5_warm,
        "warm_speedup": v4_warm / v5_warm,
        "prediction_max_abs_difference": prediction_difference,
        "target_max_abs_difference": target_difference,
        "graph_cache_audit": graph_audit,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "optimizer_steps": 0,
        "training_runs": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
