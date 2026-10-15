"""Resumable one-episode-at-a-time E79 continuation using exact v6 TD input."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
from experiments.conditioned_vcg.E79_direct_legal_overdue_progress_full_10x10_v1 import (
    run as continuation,
)
from viability_graph_episodic_audit import (
    EpisodicViabilityGraphConfig,
    EpisodicViabilityGraphHierarchyAgent,
)


PROTOCOL = "vcg_e79_v6_direct_legal_overdue_progress_full_10x10_v1"
METHOD_VERSION = "vcg_dense_v1_1_direct_legal_overdue_progress_exact_v6_v1"
CHECKPOINT_FAMILY = "direct_legal_overdue_progress_resumable_exact_v6_v1"
SOURCE_THROUGH_EPISODE = 28
TOTAL_EPISODES = 63


def _load_v6():
    path = Path(__file__).resolve().with_name("acceleration.py")
    spec = importlib.util.spec_from_file_location(
        "e79_resumable_v6_acceleration", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exact-v6 acceleration from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contract(source_checkpoint_sha256: str) -> dict:
    directory = Path(__file__).resolve().parent
    roots = {
        "resumable_continuation.py": directory / "resumable_continuation.py",
        "acceleration.py": directory / "acceleration.py",
        "active_episode_runner.py": Path(continuation.__file__).resolve(),
    }
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "checkpoint_family": CHECKPOINT_FAMILY,
        "scope": "one_episode_at_a_time_resumable_v6_continuation",
        "source_through_episode": SOURCE_THROUGH_EPISODE,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "total_episodes": TOTAL_EPISODES,
        "checkpoint_after_every_successful_episode": True,
        "invocation_runs_exactly_one_episode": True,
        "recoverability_certification": False,
        "safety_guarantee": False,
        "paper_evidence": False,
        "td_backup_implementation": _load_v6().IMPLEMENTATION,
        "source_sha256": {
            name: _sha256(path) for name, path in sorted(roots.items())
        },
    }
    return {**payload, "contract_sha256": _digest(payload)}


def _summary(payload: dict, checkpoint: Path) -> dict:
    return {
        "status": (
            "complete"
            if int(payload["through_episode"]) == TOTAL_EPISODES
            else "paused"
        ),
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "through_episode": int(payload["through_episode"]),
        "total_episodes": TOTAL_EPISODES,
        "completed_v6_episodes": int(payload["completed_v6_episodes"]),
        "latest_checkpoint": str(checkpoint),
        "latest_checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_after_every_successful_episode": True,
        "invocation_runs_exactly_one_episode": True,
        "safety_guarantee": False,
        "paper_evidence": False,
    }


def _validate(payload: dict, contract: dict) -> None:
    expected = {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "total_training_episodes": TOTAL_EPISODES,
        "safety_guarantee": False,
        "paper_evidence": False,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"v6 checkpoint contract differs: {mismatches}")
    through = int(payload.get("through_episode", -1))
    if not SOURCE_THROUGH_EPISODE <= through <= TOTAL_EPISODES:
        raise RuntimeError("v6 checkpoint episode is outside its continuation")
    if int(payload.get("completed_v6_episodes", -1)) != (
        through - SOURCE_THROUGH_EPISODE
    ):
        raise RuntimeError("v6 completed-episode count differs")
    for key in ("config", "agent_state", "global_rng_state", "resume_contract"):
        if key not in payload:
            raise RuntimeError(f"v6 checkpoint is missing {key}")


def prepare(source_output: Path, output: Path) -> dict:
    source = source_output.resolve()
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite: {output}")
    source_checkpoint = source / "latest.pth"
    source_summary = json.loads(
        (source / "training-summary.json").read_text(encoding="utf-8")
    )
    source_sha = _sha256(source_checkpoint)
    if (
        int(source_summary.get("through_episode", -1))
        != SOURCE_THROUGH_EPISODE
        or source_summary.get("latest_checkpoint_sha256") != source_sha
    ):
        raise RuntimeError("source checkpoint 28 does not authenticate")
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if (
        int(payload.get("through_episode", -1)) != SOURCE_THROUGH_EPISODE
        or payload.get("protocol") != continuation.p.PROTOCOL
    ):
        raise RuntimeError("source payload is not the expected v5 checkpoint")
    contract = _contract(source_sha)
    migrated = dict(payload)
    migrated.update({
        "checkpoint_family": CHECKPOINT_FAMILY,
        "protocol": PROTOCOL,
        "method_version": METHOD_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "source_v5_checkpoint": str(source_checkpoint),
        "source_v5_checkpoint_sha256": source_sha,
        "source_v5_through_episode": SOURCE_THROUGH_EPISODE,
        "completed_v6_episodes": 0,
        "total_training_episodes": TOTAL_EPISODES,
        "safety_guarantee": False,
        "paper_evidence": False,
        "td_backup_acceleration_audit": {
            "implementation": contract["td_backup_implementation"],
            "status": "not_started_after_migration",
            "cache_checkpointed": False,
        },
        "checkpoint_role": "latest_resumable",
    })
    _validate(migrated, contract)
    output.mkdir(parents=True)
    (output / "heartbeats").mkdir()
    (output / "episode-reports").mkdir()
    latest = output / "latest.pth"
    audit._atomic_json_save(contract, output / "contract.json")
    audit._atomic_torch_save(migrated, latest)
    audit._atomic_json_save(_summary(migrated, latest), output / "training-summary.json")
    result = {
        "status": "prepared",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha,
        "latest_checkpoint": str(latest),
        "latest_checkpoint_sha256": _sha256(latest),
        "through_episode": SOURCE_THROUGH_EPISODE,
        "contract_sha256": contract["contract_sha256"],
        "agent_state_unchanged": True,
        "optimizer_state_unchanged": True,
        "replay_unchanged": True,
        "global_rng_state_unchanged": True,
    }
    audit._atomic_json_save(result, output / "migration.json")
    return result


def _load(output: Path, device: torch.device):
    output = output.resolve()
    contract = json.loads((output / "contract.json").read_text(encoding="utf-8"))
    source_sha = str(contract["source_checkpoint_sha256"])
    if contract != _contract(source_sha):
        raise RuntimeError("v6 continuation contract differs")
    latest = output / "latest.pth"
    summary = json.loads((output / "training-summary.json").read_text(encoding="utf-8"))
    if summary.get("latest_checkpoint_sha256") != _sha256(latest):
        raise RuntimeError("v6 summary does not authenticate checkpoint")
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    _validate(payload, contract)
    config = EpisodicViabilityGraphConfig.from_dict(payload["config"])
    agent = EpisodicViabilityGraphHierarchyAgent(
        config=config,
        seed=int(getattr(continuation.p, "MODEL_SEED", 0)),
        device=device,
        epsilon=float(payload["agent_state"].get("epsilon", 0.0)),
    )
    agent.load_checkpoint_state(payload["agent_state"], resumable=True)
    audit._restore_global_rng_state(payload["global_rng_state"])
    return contract, latest, payload, agent


def _protocol_proxy():
    values = {
        name: value
        for name, value in vars(continuation.p).items()
        if not name.startswith("__")
    }
    values.update({
        "PROTOCOL": PROTOCOL,
        "METHOD_VERSION": METHOD_VERSION,
        "TOTAL_EPISODES": TOTAL_EPISODES,
    })
    return SimpleNamespace(**values)


def train_next(output: Path, device_name: str) -> dict:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    contract, latest, payload, agent = _load(output, device)
    completed = int(payload["through_episode"])
    if completed >= TOTAL_EPISODES:
        return _summary(payload, latest)
    episode_number = completed + 1
    v6 = _load_v6()
    proxy = _protocol_proxy()
    with (
        patch.object(continuation, "acceleration", v6),
        patch.object(continuation, "p", proxy),
        patch.object(builtins, "print"),
    ):
        result = continuation._run_episode(
            agent,
            payload["resume_contract"],
            output.resolve(),
            episode_number,
            device,
        )
    if not result["success"] or not result["training_state_advanced"]:
        failure = output.resolve() / "episode-reports" / (
            f"episode-{episode_number:04d}-failed.json"
        )
        audit._atomic_json_save(result, failure)
        raise RuntimeError(f"v6 episode failed; details: {failure}")
    history = list(payload.get("training_history", ()))
    history.append(result)
    checkpoint = dict(payload)
    checkpoint.update({
        "through_episode": episode_number,
        "completed_v6_episodes": episode_number - SOURCE_THROUGH_EPISODE,
        "agent_state": agent.checkpoint_state(include_replay=True),
        "global_rng_state": audit._capture_global_rng_state(),
        "training_history": tuple(history),
        "td_backup_acceleration_audit": v6.audit_payload(agent),
        "checkpoint_role": "latest_resumable",
    })
    _validate(checkpoint, contract)
    audit._atomic_torch_save(checkpoint, latest)
    latest_sha = _sha256(latest)
    report = {
        **result,
        "status": "complete",
        "checkpoint": str(latest),
        "checkpoint_sha256": latest_sha,
    }
    audit._atomic_json_save(
        report,
        output.resolve() / "episode-reports" / f"episode-{episode_number:04d}.json",
    )
    audit._atomic_json_save(
        _summary(checkpoint, latest), output.resolve() / "training-summary.json"
    )
    line = continuation._episode_console_line(result, agent)
    with (output.resolve() / "episode-console.log").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write(line + "\n")
    print(line, flush=True)
    return _summary(checkpoint, latest)


def status(output: Path) -> dict:
    contract, latest, payload, _ = _load(output, torch.device("cpu"))
    _validate(payload, contract)
    return _summary(payload, latest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "status", "train-next"))
    parser.add_argument("--source-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.source_output is None:
            raise RuntimeError("prepare requires --source-output")
        result = prepare(args.source_output, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "status":
        print(json.dumps(status(args.output), indent=2, sort_keys=True))
    else:
        train_next(args.output, args.device)


if __name__ == "__main__":
    main()
