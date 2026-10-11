"""Resume E79 with a versioned numerical audit; no learner change."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
import train_vcg_dense_proper as dense
from experiments.conditioned_vcg.E79_original_vcg_base100_full_10x10_144k_v1 import protocol as old_p
from experiments.conditioned_vcg.E79_original_vcg_base100_full_10x10_144k_v1 import run as old_run

from . import protocol as p
from .roundoff import repaired_audit


class E79V2Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E79V2Error(message)


def _read_json(path: Path) -> dict:
    require(path.is_file(), f"missing migration input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid migration input: {path}")
    return value


def migrate_seed(source_output: Path, output: Path, seed: int) -> dict:
    """Rebind an unfinished v1 checkpoint, preserving agent state and history.

    The original output is read-only. A new protocol contract, checkpoint,
    manifest, and summary are committed to a distinct v2 output directory.
    No episode, optimizer update, or quality-based checkpoint choice occurs.
    """

    source_output, output = source_output.resolve(), output.resolve()
    require(source_output != output, "v1 and v2 outputs must differ")
    require(seed in p.MODEL_SEEDS, "model seed must be 0, 1, or 2")
    old_contract = _read_json(source_output / "contract.json")
    require(old_contract == old_p.contract(),
            "v1 contract or source files differ; migration refused")
    require(old_contract["contract_sha256"] ==
            "f25a890d6b16f2735ac092bb9af3d43363fffcac826c1a7bdcce2cdfaa78f620",
            "v1 source contract is not the recorded E79 release")
    with patch.object(old_run, "p", p):
        old_run.prepare(output)
        new_contract = old_run.authenticate(output)

    old_dir = source_output / "training/operational" / p.SCENARIO / f"seed-{seed}"
    old_manifest = _read_json(old_dir / "training-contract.json")
    old_summary = _read_json(old_dir / "training-summary.json")
    old_latest_path = old_dir / "latest.pth"
    require(old_latest_path.is_file(), "v1 latest checkpoint is missing")
    old_latest_sha = audit._sha256(old_latest_path)
    checkpoint = torch.load(old_latest_path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), "v1 latest checkpoint is not a mapping")
    old_resume = checkpoint.get("resume_contract")
    require(isinstance(old_resume, dict), "v1 checkpoint lacks resume contract")
    old_resume_hash = audit._contract_hash(old_resume)
    require(old_manifest.get("contract") == old_resume
            and old_manifest.get("resume_contract_sha256") == old_resume_hash
            and checkpoint.get("resume_contract_sha256") == old_resume_hash,
            "v1 checkpoint and training manifest do not authenticate")
    completed = int(checkpoint.get("completed_training_episodes", -1))
    require(1 <= completed < p.EPISODES_BY_SCENARIO[p.SCENARIO],
            "only unfinished v1 checkpoints can migrate")
    require(old_resume.get("training_protocol") == old_p.PROTOCOL
            and old_resume.get("method_version") == old_p.METHOD_VERSION
            and old_resume.get("target_contract_sha256")
            == old_contract["contract_sha256"],
            "v1 resume contract is not E79-v1")
    require(checkpoint.get("training_protocol") == old_p.PROTOCOL
            and checkpoint.get("method_version") == old_p.METHOD_VERSION
            and checkpoint.get("checkpoint_role") == "latest_resumable"
            and checkpoint.get("td_backup_implementation")
            == p.TD_BACKUP_IMPLEMENTATION,
            "v1 checkpoint metadata differs")
    require(len(checkpoint.get("training_history", ())) == completed,
            "v1 checkpoint training history is incomplete")
    require(old_summary.get("completed_training_episodes") == completed
            and old_summary.get("resume_contract_sha256") == old_resume_hash,
            "v1 summary does not match latest checkpoint")

    candidate_name = checkpoint.get("best_candidate_checkpoint_filename")
    candidate_sha = checkpoint.get("best_candidate_checkpoint_sha256")
    best_record = checkpoint.get("best_validation_record")
    migrated_candidate = None
    if best_record is None:
        require(candidate_name is None and candidate_sha is None,
                "v1 checkpoint names a candidate without a best record")
    else:
        require(candidate_name in dense.CANDIDATE_FILENAMES
                and isinstance(candidate_sha, str),
                "v1 best candidate metadata is invalid")
        old_candidate_path = old_dir / candidate_name
        require(old_candidate_path.is_file()
                and audit._sha256(old_candidate_path) == candidate_sha,
                "v1 best candidate does not match its checkpoint hash")
        old_candidate = torch.load(old_candidate_path, map_location="cpu",
                                   weights_only=False)
        require(isinstance(old_candidate, dict)
                and old_candidate.get("training_protocol") == old_p.PROTOCOL
                and old_candidate.get("method_version") == old_p.METHOD_VERSION
                and old_candidate.get("resume_contract") == old_resume
                and dense._best_record_identity(
                    old_candidate.get("best_validation_record")
                ) == dense._best_record_identity(best_record),
                "v1 best candidate metadata differs from latest checkpoint")
        migrated_candidate = dict(old_candidate)

    new_resume = dict(old_resume)
    new_resume.update({
        "training_protocol": p.PROTOCOL,
        "method_version": p.METHOD_VERSION,
        "target_contract_sha256": new_contract["contract_sha256"],
    })
    new_resume_hash = audit._contract_hash(new_resume)
    new_checkpoint = dict(checkpoint)
    new_checkpoint.update({
        "protocol": p.PROTOCOL,
        "training_protocol": p.PROTOCOL,
        "method_version": p.METHOD_VERSION,
        "resume_contract": new_resume,
        "resume_contract_sha256": new_resume_hash,
    })
    if migrated_candidate is not None:
        migrated_candidate.update({
            "protocol": p.PROTOCOL,
            "training_protocol": p.PROTOCOL,
            "method_version": p.METHOD_VERSION,
            "resume_contract": new_resume,
            "resume_contract_sha256": new_resume_hash,
        })
    with ExitStack() as stack:
        stack.enter_context(patch.object(dense, "TRAINING_PROTOCOL", p.PROTOCOL))
        stack.enter_context(patch.object(dense, "METHOD_VERSION", p.METHOD_VERSION))
        stack.enter_context(patch.object(dense, "FROZEN_TOTAL_EPISODES", 63))
        stack.enter_context(patch.object(dense, "FROZEN_OBJECTIVE_SPEC", p.SPEC))
        stack.enter_context(patch.object(dense, "FROZEN_VALIDATION_SEEDS",
                                         tuple(p.VALIDATION_SEEDS[p.SCENARIO])))
        dense._validate_resume(new_checkpoint, new_resume)

    new_dir = output / "training/operational" / p.SCENARIO / f"seed-{seed}"
    require(not new_dir.exists(), "v2 seed directory already exists")
    new_manifest = dict(old_manifest)
    new_manifest.update({
        "training_protocol": p.PROTOCOL,
        "method_version": p.METHOD_VERSION,
        "resume_contract_sha256": new_resume_hash,
        "contract": new_resume,
    })
    new_summary = dict(old_summary)
    new_summary.update({
        "training_protocol": p.PROTOCOL,
        "method_version": p.METHOD_VERSION,
        "target_contract_sha256": new_contract["contract_sha256"],
        "resume_contract": new_resume,
        "resume_contract_sha256": new_resume_hash,
        "latest_checkpoint": str(new_dir / "latest.pth"),
    })
    provenance = {
        "status": "migrated",
        "source_protocol": old_p.PROTOCOL,
        "source_contract_sha256": old_contract["contract_sha256"],
        "source_checkpoint": str(old_latest_path),
        "source_checkpoint_sha256": old_latest_sha,
        "destination_protocol": p.PROTOCOL,
        "destination_contract_sha256": new_contract["contract_sha256"],
        "completed_training_episodes": completed,
        "agent_state_unchanged": True,
        "training_runs": 0,
        "episode_runs": 0,
    }
    parent = new_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".seed-{seed}-migration-", dir=parent) as temp:
        staged = Path(temp) / f"seed-{seed}"
        staged.mkdir()
        audit._atomic_json_save(new_manifest, staged / "training-contract.json")
        if migrated_candidate is not None:
            candidate_path = staged / candidate_name
            audit._atomic_torch_save(migrated_candidate, candidate_path)
            new_candidate_sha = audit._sha256(candidate_path)
            new_checkpoint["best_candidate_checkpoint_sha256"] = new_candidate_sha
            new_summary["best_candidate_checkpoint_sha256"] = new_candidate_sha
            new_summary["best_candidate_checkpoint"] = str(new_dir / candidate_name)
            provenance["source_candidate_sha256"] = candidate_sha
            provenance["destination_candidate_sha256"] = new_candidate_sha
        audit._atomic_torch_save(new_checkpoint, staged / "latest.pth")
        audit._atomic_json_save(new_summary, staged / "training-summary.json")
        provenance["destination_checkpoint_sha256"] = audit._sha256(staged / "latest.pth")
        audit._atomic_json_save(provenance, staged / "e79-v2-migration.json")
        os.replace(staged, new_dir)
    return {**provenance, "migration_record": str(new_dir / "e79-v2-migration.json")}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "migrate-seed":
        parser = argparse.ArgumentParser(description="Migrate pre-validation E79-v1 checkpoint")
        parser.add_argument("command", choices=("migrate-seed",))
        parser.add_argument("--source-output", type=Path, default=old_p.DEFAULT_OUTPUT)
        parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
        parser.add_argument("--model-seed", type=int, required=True, choices=p.MODEL_SEEDS)
        args = parser.parse_args()
        result = migrate_seed(args.source_output, args.output, args.model_seed)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    with patch.object(old_run, "p", p):
        if len(sys.argv) > 1 and sys.argv[1] == "train-seed":
            with repaired_audit():
                old_run.main()
        else:
            old_run.main()


if __name__ == "__main__":
    main()
