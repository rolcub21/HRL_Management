#!/usr/bin/env python3
"""Migrate and continue E188 with a bounded numerical audit repair."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
import train_vcg_dense_proper as dense
from experiments.conditioned_vcg.E188_mask_only_progress_guard_247k_v1 import (
    run as old_run,
)

from . import protocol as p
from .roundoff import repaired_audit


class E189Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E189Error(message)


def _read(path: Path) -> dict:
    require(path.is_file(), f"E189 migration input is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"E189 migration input is invalid: {path}")
    return value


@contextmanager
def implementation():
    # E188 owns the learner/guard implementation.  Rebinding only its frozen
    # protocol identity gives E189 the same training path under a new contract.
    with patch.object(old_run, "p", p):
        yield


def prepare(output: Path) -> dict:
    with implementation():
        result = old_run.prepare(output)
    return {
        **result,
        "protocol": p.PROTOCOL,
        "numerical_parent_protocol": p.v1.PROTOCOL,
        "only_change": "bounded_end_of_episode_roundoff_audit",
    }


def authenticate(output: Path) -> dict:
    with implementation():
        return old_run.authenticate(output)


def _new_resume_contract(old_resume: dict, contract: dict) -> dict:
    result = dict(old_resume)
    result.update({
        "training_protocol": p.PROTOCOL,
        "method_version": p.METHOD_VERSION,
        "target_contract_sha256": contract["contract_sha256"],
        # The inherited E187 runner uses this historical key for the active
        # extension contract.  In E188 and E189 it contains the current arm's
        # hash, not E187's literal hash.
        "e187_contract_sha256": contract["contract_sha256"],
    })
    return result


def _validate_migrated_checkpoint(payload: dict, resume: dict) -> None:
    with ExitStack() as stack:
        stack.enter_context(patch.object(dense, "TRAINING_PROTOCOL", p.PROTOCOL))
        stack.enter_context(patch.object(dense, "METHOD_VERSION", p.METHOD_VERSION))
        stack.enter_context(patch.object(
            dense, "FROZEN_TOTAL_EPISODES", p.TOTAL_EPISODES
        ))
        stack.enter_context(patch.object(dense, "FROZEN_OBJECTIVE_SPEC", p.SPEC))
        stack.enter_context(patch.object(dense, "FROZEN_GAMMA", p.GAMMA))
        stack.enter_context(patch.object(
            dense,
            "FROZEN_VALIDATION_SEEDS",
            tuple(p.VALIDATION_SEEDS[p.SCENARIO]),
        ))
        dense._validate_resume(payload, resume)
    expected = {
        "e187_contract_sha256": p.contract()["contract_sha256"],
        "e187_objective": p.contract()["objective"],
        "e187_training_budget_rule": old_run.BUDGET_RULE,
        "e187_epsilon_decay_decisions": p.EPSILON_DECAY_DECISIONS,
        "td_backup_implementation": p.TD_BACKUP_IMPLEMENTATION,
        "queue_context_stored_in_replay": True,
        "future_schedule_visible_to_policy": False,
    }
    mismatches = {
        name: (payload.get(name), wanted)
        for name, wanted in expected.items()
        if payload.get(name) != wanted
    }
    if mismatches:
        raise E189Error(f"E189 migrated metadata differs: {mismatches!r}")


def migrate_seed(source_output: Path, output: Path, seed: int) -> dict:
    """Copy one authenticated unfinished E188 state into the E189 contract."""

    source_output = source_output.resolve()
    output = output.resolve()
    require(source_output != output, "E188 and E189 outputs must differ")
    require(seed in p.MODEL_SEEDS, "E189 model seed must be 0, 1, or 2")

    source_contract = _read(source_output / "contract.json")
    expected_source_contract = p.v1.contract()
    require(
        source_contract == expected_source_contract,
        "E188 contract or source files changed; migration refused",
    )
    prepare(output)
    destination_contract = authenticate(output)

    source_dir = (
        source_output / "training/operational" / p.SCENARIO / f"seed-{seed}"
    )
    source_manifest = _read(source_dir / "training-contract.json")
    source_summary = _read(source_dir / "training-summary.json")
    source_latest_path = source_dir / "latest.pth"
    require(source_latest_path.is_file(), "E188 latest checkpoint is missing")
    source_latest_sha = audit._sha256(source_latest_path)
    checkpoint = torch.load(
        source_latest_path, map_location="cpu", weights_only=False
    )
    require(isinstance(checkpoint, dict), "E188 checkpoint is not a mapping")
    source_resume = checkpoint.get("resume_contract")
    require(isinstance(source_resume, dict), "E188 checkpoint lacks resume contract")
    source_resume_hash = audit._contract_hash(source_resume)
    # JSON serialization normalizes tuples to lists, so authenticate the
    # manifest's contract by its canonical hash instead of Python equality.
    require(
        audit._contract_hash(source_manifest.get("contract", {}))
        == source_resume_hash
        and source_manifest.get("resume_contract_sha256") == source_resume_hash
        and checkpoint.get("resume_contract_sha256") == source_resume_hash,
        "E188 checkpoint and training manifest do not authenticate",
    )
    completed = int(checkpoint.get("completed_training_episodes", -1))
    require(
        1 <= completed < p.TOTAL_EPISODES,
        "only unfinished nonempty E188 checkpoints can migrate",
    )
    require(
        source_resume.get("training_protocol") == p.v1.PROTOCOL
        and source_resume.get("method_version") == p.v1.METHOD_VERSION
        and source_resume.get("target_contract_sha256")
        == source_contract["contract_sha256"],
        "source resume contract is not E188",
    )
    require(
        checkpoint.get("training_protocol") == p.v1.PROTOCOL
        and checkpoint.get("method_version") == p.v1.METHOD_VERSION
        and checkpoint.get("checkpoint_role") == "latest_resumable"
        and checkpoint.get("td_backup_implementation")
        == p.TD_BACKUP_IMPLEMENTATION,
        "E188 checkpoint metadata differs",
    )
    require(
        len(checkpoint.get("training_history", ())) == completed,
        "E188 checkpoint training history is incomplete",
    )
    require(
        source_summary.get("completed_training_episodes") == completed
        and source_summary.get("resume_contract_sha256") == source_resume_hash,
        "E188 summary does not match latest checkpoint",
    )

    candidate_name = checkpoint.get("best_candidate_checkpoint_filename")
    candidate_sha = checkpoint.get("best_candidate_checkpoint_sha256")
    best_record = checkpoint.get("best_validation_record")
    migrated_candidate = None
    if best_record is None:
        require(
            candidate_name is None and candidate_sha is None,
            "E188 names a candidate without a best record",
        )
    else:
        require(
            candidate_name in dense.CANDIDATE_FILENAMES
            and isinstance(candidate_sha, str),
            "E188 best candidate metadata is invalid",
        )
        source_candidate_path = source_dir / candidate_name
        require(
            source_candidate_path.is_file()
            and audit._sha256(source_candidate_path) == candidate_sha,
            "E188 best candidate does not match its hash",
        )
        migrated_candidate = torch.load(
            source_candidate_path, map_location="cpu", weights_only=False
        )
        require(
            isinstance(migrated_candidate, dict)
            and migrated_candidate.get("training_protocol") == p.v1.PROTOCOL
            and migrated_candidate.get("method_version") == p.v1.METHOD_VERSION
            and migrated_candidate.get("resume_contract") == source_resume
            and dense._best_record_identity(
                migrated_candidate.get("best_validation_record")
            ) == dense._best_record_identity(best_record),
            "E188 best candidate identity differs",
        )
        migrated_candidate = dict(migrated_candidate)

    new_resume = _new_resume_contract(source_resume, destination_contract)
    new_resume_hash = audit._contract_hash(new_resume)
    new_checkpoint = dict(checkpoint)
    new_checkpoint.update({
        "protocol": p.PROTOCOL,
        "training_protocol": p.PROTOCOL,
        "method_version": p.METHOD_VERSION,
        "resume_contract": new_resume,
        "resume_contract_sha256": new_resume_hash,
        "e187_contract_sha256": destination_contract["contract_sha256"],
        "e189_numerical_audit_repair": True,
        "e189_source_checkpoint_sha256": source_latest_sha,
    })
    if migrated_candidate is not None:
        migrated_candidate.update({
            "protocol": p.PROTOCOL,
            "training_protocol": p.PROTOCOL,
            "method_version": p.METHOD_VERSION,
            "resume_contract": new_resume,
            "resume_contract_sha256": new_resume_hash,
            "e187_contract_sha256": destination_contract["contract_sha256"],
            "e189_numerical_audit_repair": True,
        })
    _validate_migrated_checkpoint(new_checkpoint, new_resume)

    destination_dir = (
        output / "training/operational" / p.SCENARIO / f"seed-{seed}"
    )
    require(not destination_dir.exists(), "E189 seed directory already exists")
    new_manifest = dict(source_manifest)
    new_manifest.update({
        "training_protocol": p.PROTOCOL,
        "method_version": p.METHOD_VERSION,
        "resume_contract_sha256": new_resume_hash,
        "contract": new_resume,
    })
    new_summary = dict(source_summary)
    new_summary.update({
        "training_protocol": p.PROTOCOL,
        "method_version": p.METHOD_VERSION,
        "target_contract_sha256": destination_contract["contract_sha256"],
        "e187_contract_sha256": destination_contract["contract_sha256"],
        "resume_contract": new_resume,
        "resume_contract_sha256": new_resume_hash,
        "latest_checkpoint": str(destination_dir / "latest.pth"),
        "e189_numerical_audit_repair": True,
    })
    provenance = {
        "status": "migrated",
        "source_protocol": p.v1.PROTOCOL,
        "source_contract_sha256": source_contract["contract_sha256"],
        "source_checkpoint": str(source_latest_path),
        "source_checkpoint_sha256": source_latest_sha,
        "destination_protocol": p.PROTOCOL,
        "destination_contract_sha256": destination_contract["contract_sha256"],
        "completed_training_episodes": completed,
        "next_training_episode": completed + 1,
        "agent_state_unchanged": True,
        "optimizer_state_unchanged": True,
        "replay_unchanged": True,
        "rng_state_unchanged": True,
        "history_unchanged": True,
        "training_runs": 0,
        "episode_runs": 0,
    }

    parent = destination_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".seed-{seed}-migration-", dir=parent
    ) as temporary:
        staged = Path(temporary) / f"seed-{seed}"
        staged.mkdir()
        audit._atomic_json_save(new_manifest, staged / "training-contract.json")
        if migrated_candidate is not None:
            candidate_path = staged / candidate_name
            audit._atomic_torch_save(migrated_candidate, candidate_path)
            new_candidate_sha = audit._sha256(candidate_path)
            new_checkpoint["best_candidate_checkpoint_sha256"] = new_candidate_sha
            new_summary["best_candidate_checkpoint_sha256"] = new_candidate_sha
            new_summary["best_candidate_checkpoint"] = str(
                destination_dir / candidate_name
            )
            provenance["source_candidate_sha256"] = candidate_sha
            provenance["destination_candidate_sha256"] = new_candidate_sha
        audit._atomic_torch_save(new_checkpoint, staged / "latest.pth")
        audit._atomic_json_save(new_summary, staged / "training-summary.json")
        provenance["destination_checkpoint_sha256"] = audit._sha256(
            staged / "latest.pth"
        )
        audit._atomic_json_save(provenance, staged / "e189-migration.json")
        os.replace(staged, destination_dir)
    return {
        **provenance,
        "migration_record": str(destination_dir / "e189-migration.json"),
    }


def self_check(output: Path) -> dict:
    with implementation():
        result = old_run.self_check(output)
    parent = p.v1.contract()
    contract = p.contract()
    require(
        contract["numerical_parent_contract_sha256"]
        == parent["contract_sha256"],
        "E189 parent contract identity differs",
    )
    return {
        **result,
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "numerical_audit_repair": contract["numerical_audit_repair"],
    }


def train_seed(output: Path, seed: int, through_episode: int, device: str) -> dict:
    with implementation(), repaired_audit():
        result = old_run.train_seed(output, seed, through_episode, device)
    return {
        **result,
        "e189_protocol": p.PROTOCOL,
        "e189_contract_sha256": p.contract()["contract_sha256"],
        "numerical_audit_repair": True,
    }


def status(output: Path) -> dict:
    with implementation():
        return old_run.status(output)


def analyze(output: Path) -> dict:
    with implementation():
        return old_run.analyze(output)


def compare(output: Path, parent_output: Path) -> dict:
    with implementation():
        return old_run.compare(output, parent_output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare", "migrate-seed", "self-check", "train-seed",
            "status", "analyze", "compare",
        ),
    )
    parser.add_argument("--source-output", type=Path, default=p.SOURCE_OUTPUT)
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--parent-output", type=Path, default=p.PARENT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--through-episode", type=int, default=p.TOTAL_EPISODES)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "migrate-seed":
        require(args.model_seed is not None, "migrate-seed requires --model-seed")
        result = migrate_seed(
            args.source_output, args.output, args.model_seed
        )
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "train-seed":
        require(args.model_seed is not None, "train-seed requires --model-seed")
        result = train_seed(
            args.output, args.model_seed, args.through_episode, args.device
        )
    elif args.command == "status":
        result = status(args.output)
    elif args.command == "analyze":
        result = analyze(args.output)
    else:
        result = compare(args.output, args.parent_output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
