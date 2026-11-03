#!/usr/bin/env python3
"""Verify and materialize the immutable E214 Stage-B procedure freeze."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)

from . import protocol as p


class E216Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E216Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing frozen artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid JSON artifact: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def verify() -> dict:
    contract = p.contract()
    declaration = validate_protocol(contract, source=p.PROTOCOL)
    for relative, expected in contract["frozen_artifacts"].items():
        path = p.ROOT / relative
        require(p.sha256(path) == expected, f"frozen artifact changed: {relative}")

    report = _read(p.E214_REPORT)
    require(report.get("status") == "complete", "E214 report is incomplete")
    require(
        report.get("verdict") == "candidate_ready_for_expansion"
        and report.get("summary", {}).get("all_gates_passed") is True,
        "E214 did not pass its predeclared gates",
    )
    require(
        report["summary"]["price_exact_rate"] == 1.0 / 3.0
        and report["summary"]["price_near_5pct_rate"] == 1.0 / 3.0,
        "E214 price-interpretation limitation changed",
    )
    require(
        report["causal_replays"]["forced_post_placement"]["physical_rehandles"]
        == 29,
        "E214 forced-placement recovery anchor changed",
    )
    return {
        "status": "verified",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "source_stage_b_protocol": contract["source_stage_b_protocol"],
        "stage_b_procedure": contract["stage_b_procedure"],
        "evidence_boundary": contract["evidence_boundary"],
        "conformance": declaration,
        "training_runs": 0,
        "episode_runs": 0,
        "test_panel_opened": False,
    }


def freeze(output: Path) -> dict:
    value = verify()
    output = output.resolve()
    _write(output / "contract.json", p.contract())
    _write(output / "freeze-manifest.json", value)

    audit = render_validated_conformance_audit(value["conformance"])
    procedure = audit + "\nFROZEN STAGE-B PROCEDURE\n" + "=" * 72 + "\n\n"
    procedure += json.dumps({
        "stage_b_procedure": value["stage_b_procedure"],
        "evidence_boundary": value["evidence_boundary"],
    }, indent=2, sort_keys=True) + "\n"
    temporary = output / "PROCEDURE.txt.tmp"
    temporary.write_text(procedure, encoding="utf-8")
    os.replace(temporary, output / "PROCEDURE.txt")
    return {
        "status": "frozen",
        "protocol": p.PROTOCOL,
        "contract_sha256": value["contract_sha256"],
        "procedure": str(output / "PROCEDURE.txt"),
        "manifest": str(output / "freeze-manifest.json"),
        "training_runs": 0,
        "episode_runs": 0,
        "test_panel_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "freeze"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = verify() if args.command == "verify" else freeze(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
