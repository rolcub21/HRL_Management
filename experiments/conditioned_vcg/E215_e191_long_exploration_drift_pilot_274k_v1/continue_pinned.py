#!/usr/bin/env python3
"""Resume E215 under the exact revision-pinned contract validated at launch.

This is an administrative continuation adapter.  It does not update E215 to
the current method revision and it does not modify the saved experiment
contract.  It permits only the shared conformance validator to have changed;
every source that implements E215's training behavior must still match the
hash recorded before episode 1.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
from unittest.mock import patch

import torch

from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
)

from . import protocol as p
from . import run as e215_run


class PinnedContinuationError(RuntimeError):
    pass


ADMINISTRATIVE_SOURCE = (
    "projects/temporal_consequence_realization/method_conformance.py"
)
CURRENT_REFERENCE = (
    p.ROOT / "projects/temporal_consequence_realization/METHOD_REFERENCE.md"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PinnedContinuationError(message)


def _read(path: Path) -> dict:
    require(path.is_file(), f"required E215 artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_revision() -> str:
    text = CURRENT_REFERENCE.read_text(encoding="utf-8")
    match = re.search(r"^\*\*Revision:\*\* `([^`]+)`", text, re.MULTILINE)
    require(match is not None, "current method reference has no revision")
    return match.group(1)


def _declaration(contract: dict) -> dict:
    keys = (
        "method_reference",
        "method_revision",
        "claim_level",
        "declared_deviations",
        "oracle_inputs",
        "nonconforming_shortcuts",
        "conformance_audit",
    )
    value = {key: deepcopy(contract[key]) for key in keys}
    # This validates the recorded declaration without comparing it with the
    # living reference.  That is the intended path for a run validated before
    # the reference advanced.
    render_validated_conformance_audit(value)
    return value


def authenticate(output: Path) -> tuple[dict, dict]:
    output = output.resolve()
    contract = _read(output / "contract.json")
    require(contract.get("protocol") == p.PROTOCOL,
            "saved contract is not the E215 protocol")
    require(contract.get("method_revision") == p.METHOD_REVISION,
            "saved contract is not pinned to E215's launch revision")

    recorded_hash = contract.get("contract_sha256")
    payload = deepcopy(contract)
    payload.pop("contract_sha256", None)
    require(recorded_hash == p.digest(payload),
            "saved E215 contract hash is invalid")

    changed: dict[str, dict[str, str]] = {}
    for relative, expected in contract.get("source_sha256", {}).items():
        path = p.ROOT / relative
        require(path.is_file(), f"saved E215 source is missing: {relative}")
        observed = _sha256(path)
        if observed != expected:
            changed[relative] = {"recorded": expected, "current": observed}
    require(
        set(changed) <= {ADMINISTRATIVE_SOURCE},
        "E215 implementation sources changed after launch: "
        + ", ".join(sorted(set(changed) - {ADMINISTRATIVE_SOURCE})),
    )
    require(
        ADMINISTRATIVE_SOURCE in changed,
        "pinned continuation is unnecessary: all launch sources still match",
    )

    declaration = _declaration(contract)
    summary_path = (
        output / "training/operational" / p.SCENARIO
        / "seed-0/training-summary.json"
    )
    summary = _read(summary_path)
    completed = int(summary.get(
        "completed_training_episodes",
        len(summary.get("training", {}).get("runs", [])),
    ))
    require(summary.get("target_contract_sha256") == recorded_hash,
            "training summary is not bound to the saved E215 contract")
    checkpoint_contract_hash = summary.get("e187_contract_sha256")
    require(
        isinstance(checkpoint_contract_hash, str)
        and re.fullmatch(r"[0-9a-f]{64}", checkpoint_contract_hash) is not None,
        "training summary has no valid checkpoint provenance hash",
    )
    training_contract = _read(summary_path.parent / "training-contract.json")
    embedded_resume = training_contract.get("contract", {})
    require(
        embedded_resume.get("target_contract_sha256") == recorded_hash
        and embedded_resume.get("e187_contract_sha256") == recorded_hash,
        "the checkpoint resume contract is not bound to the launch contract",
    )
    latest = summary_path.parent / "latest.pth"
    require(latest.is_file(), "E215 resumable latest.pth is missing")

    audit = {
        "status": "authenticated",
        "protocol": p.PROTOCOL,
        "saved_contract_sha256": recorded_hash,
        "saved_method_revision": contract["method_revision"],
        "current_method_revision": _current_revision(),
        "completed_training_episodes": completed,
        "latest_checkpoint": str(latest),
        "latest_checkpoint_bytes": latest.stat().st_size,
        "legacy_checkpoint_provenance_sha256": checkpoint_contract_hash,
        "checkpoint_resume_contract_sha256": training_contract.get(
            "resume_contract_sha256"
        ),
        "allowed_administrative_source_change": changed,
        "behavioral_source_changes": [],
        "continuation_semantics": (
            "resume the launch-validated E215 historical diagnostic; do not "
            "reinterpret it as evidence under the current method revision"
        ),
        "test_panel_opened": False,
    }
    return contract, {**audit, "declaration": declaration}


def _write_audit(output: Path, value: dict) -> Path:
    path = output.resolve() / "pinned-continuation-audit.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _inside_e187_resume_validator() -> bool:
    """Identify the one legacy metadata comparison that drifted in flight."""

    frame = inspect.currentframe()
    try:
        frame = None if frame is None else frame.f_back
        expected_file = Path(e215_run.e187_run.__file__).resolve()
        while frame is not None:
            if (
                frame.f_code.co_name == "validate_resume"
                and Path(frame.f_code.co_filename).resolve() == expected_file
            ):
                return True
            frame = frame.f_back
        return False
    finally:
        del frame


@contextmanager
def pinned_launch_contract(
    contract: dict,
    declaration: dict,
    legacy_checkpoint_contract_sha256: str | None = None,
):
    """Expose the immutable launch contract through E215's extension seams."""

    def contract_view() -> dict:
        value = deepcopy(contract)
        # During the original 23-hour invocation the living conformance file
        # advanced after contract preparation but before episode 120 was
        # checkpointed.  E187 consequently copied the then-current E215 hash
        # into one redundant top-level provenance field.  The authoritative
        # resume_contract inside the same checkpoint remained correctly bound
        # to the launch contract.  Present that recorded value only to E187's
        # comparison of this one field; all other consumers see the immutable
        # launch contract, and the next checkpoint writes the launch hash.
        if (
            legacy_checkpoint_contract_sha256 is not None
            and _inside_e187_resume_validator()
        ):
            value["contract_sha256"] = legacy_checkpoint_contract_sha256
        return value

    with (
        patch.object(p, "contract", contract_view),
        patch.object(e215_run, "_conformance", lambda: deepcopy(declaration)),
    ):
        yield


def train_seed(output: Path, through_episode: int, device: str) -> dict:
    contract, audit = authenticate(output)
    completed = int(audit["completed_training_episodes"])
    require(through_episode > completed,
            f"through-episode must exceed completed episode {completed}")
    require(through_episode <= p.TOTAL_EPISODES,
            f"through-episode cannot exceed {p.TOTAL_EPISODES}")
    require(device == "cuda" and torch.cuda.is_available(),
            "E215 continuation requires CUDA")
    audit_path = _write_audit(output, audit)
    with pinned_launch_contract(
        contract,
        audit["declaration"],
        audit["legacy_checkpoint_provenance_sha256"],
    ):
        result = e215_run.train_seed(output, through_episode, device)
    return {
        **result,
        "pinned_continuation": True,
        "pinned_continuation_audit": str(audit_path),
        "saved_method_revision": contract["method_revision"],
        "current_method_revision": audit["current_method_revision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "train-seed"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=(0,), default=0)
    parser.add_argument("--through-episode", type=int, default=220)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    require(args.model_seed == 0, "E215 is predeclared for seed 0 only")
    if args.command == "check":
        _contract, result = authenticate(args.output)
        result.pop("declaration", None)
    else:
        result = train_seed(args.output, args.through_episode, args.device)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
