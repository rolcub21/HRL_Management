"""Authenticate, benchmark, migrate, then resume cache-corrected E123."""

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1 import run as original
from . import protocol as p
from .acceleration import SharedProgressAgent, cache_session, compact_transitions

read, write, require = original.read, original.write, original.require


@contextmanager
def controller(arm):
    """Exactly the v1 frontier/proof contexts, without its per-episode clear."""
    joint, absolute = p.ARMS[arm]
    with ExitStack() as stack:
        verifier = original.JointVerifier() if joint else None
        stack.enter_context(patch.object(original.replay.audit, "ViabilityCertificateCache",
            original.JointCertificateCache if joint else original.replay.reuse.TimingInvariantCertificateCache))
        if joint:
            stack.enter_context(verifier.enabled())
            stack.enter_context(patch.object(original.replay.audit, "enumerate_viability_candidates", verifier.enumerate))
        stack.enter_context(original.replay.reuse.path_cleanup_active())
        stack.enter_context(original.replay.e115run.fast_inputs.enabled())
        stack.enter_context(patch.object(original.replay.model, "learning_rate_at", p.e118.learning_rate_at))
        stack.enter_context(patch.object(original.replay.instrumentation, "learning_rate_at", p.e118.learning_rate_at))
        yield original.AbsoluteProgressDeferRule() if absolute else original.BlockedPickupWaitRule()


@contextmanager
def implementation():
    with patch.object(original, "p", p), patch.object(original, "ProgressAgent", SharedProgressAgent), \
         patch.object(original, "controller", controller):
        yield


def prepare(output):
    with implementation():
        return original.prepare(output)


def source_payload(arm, seed):
    path = p.SOURCE_OUTPUT / arm / f"seed-{seed}/latest.pth"
    require(path.is_file(), f"missing E123-v1 resume checkpoint: {path}")
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(before == (path.stat().st_size, path.stat().st_mtime_ns), "source checkpoint changed while loading; stop v1 first")
    require(payload["e123_contract_sha256"] == p.source.contract()["contract_sha256"]
            and payload["e123_arm"] == arm and payload["model_seed"] == seed, "source checkpoint identity differs")
    require(all(row["strict_method_success"] for row in payload["training_history"]), "source contains failed training")
    require(payload["completed_training_episodes"] == len(payload["training_history"]), "source episode count differs")
    return path, payload


def migrate(output, arm, seed):
    contract = prepare(output)
    directory = output / arm / f"seed-{seed}"
    require(not (directory / "latest.pth").exists(), "v2 checkpoint exists; resume it, do not migrate again")
    source_path, payload = source_payload(arm, seed)
    source_sha = p.parent.parent.attribution.sha256(source_path)
    state = dict(payload["agent_state"])
    replay = dict(state["replay"])
    replay["memory"], sharing = compact_transitions(replay["memory"])
    state["replay"] = replay
    value = {**payload, "agent_state": state, "e123_protocol": p.PROTOCOL,
             "e123_contract_sha256": contract["contract_sha256"],
             "e123_execution_implementation": p.IMPLEMENTATION,
             "graph_input_cache_capacity": p.INPUT_CAPACITY,
             "v1_checkpoint_sha256": source_sha,
             "v1_completed_training_episodes": payload["completed_training_episodes"]}
    # Verify the unchanged model and resumable optimizer/RNG schema before writing.
    agent = SharedProgressAgent.from_checkpoint(value, resumable=True, seed=seed)
    require(agent.rng.getstate() == payload["agent_state"]["rng_state"], "migration changed sampler RNG")
    require(agent.decision_count == payload["agent_state"]["decision_count"], "migration changed exploration counter")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "latest.pth.tmp"
    torch.save(value, temporary)
    os.replace(temporary, directory / "latest.pth")
    write(directory / "summary.json", {"contract_sha256": contract["contract_sha256"], "arm": arm,
        "model_seed": seed, "completed_training_episodes": payload["completed_training_episodes"],
        "training_history": payload["training_history"], "validation_history": payload["validation_history"],
        "training_decisions": agent.decision_count, "epsilon": agent.epsilon,
        "paper_evidence": False, "deployment_checkpoint_eligible": False})
    result = {"source_checkpoint": str(source_path), "source_sha256": source_sha,
              "source_completed_episodes": payload["completed_training_episodes"], "sharing": sharing,
              "weights_optimizer_rng_counters_unchanged": True, "training_runs": 0}
    write(directory / "migration.json", result)
    require(p.parent.parent.attribution.sha256(source_path) == source_sha, "source changed during migration")
    return result


def train(output, arm, seed, through, device, authorize_absolute):
    require((output / arm / f"seed-{seed}/migration.json").is_file(), "migrate the stopped v1 checkpoint first")
    print(f"E123 v2 | cache-corrected resume | {arm} seed={seed} | endpoint={through}", flush=True)
    with cache_session(), implementation():
        return original.train_arm(output, arm, seed, through, device, authorize_absolute, False)


def analyze(output):
    with implementation():
        report = original.analyze(output)
    for arm, row in report["arms"].items():
        migration = read(output / arm / "seed-0/migration.json")
        summary = read(output / arm / "seed-0/summary.json")
        start = migration["source_completed_episodes"]
        new = summary["training_history"][start:]
        row.update(v1_completed_episodes=start, v2_completed_episodes=len(new),
                   v2_mean_training_episode_seconds=fmean(r["wall_seconds"] for r in new) if new else None)
    write(output / "e123-v2-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "migrate", "train-arm", "train-matched", "analyze"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--arm", choices=p.ARMS)
    parser.add_argument("--model-seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--through-episode", type=int, default=120)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--authorize-absolute-arm", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "preflight":
        from .verification import preflight
        result = preflight(output, args.device, args.model_seed)
    elif args.command == "migrate":
        result = {arm: migrate(output, arm, args.model_seed) for arm in
                  ((args.arm,) if args.arm else ("A_outflow_bounded", "B_joint_bounded"))}
    elif args.command == "train-arm":
        require(args.arm is not None, "train-arm needs --arm")
        result = train(output, args.arm, args.model_seed, args.through_episode, args.device, args.authorize_absolute_arm)
    elif args.command == "train-matched":
        result = {arm: train(output, arm, args.model_seed, args.through_episode, args.device, False)
                  for arm in ("A_outflow_bounded", "B_joint_bounded")}
    else:
        result = analyze(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
