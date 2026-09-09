#!/usr/bin/env python3
"""Interim A-versus-B view of the contracted 85k architecture screen.

This sidecar executes only the already-contracted conditioned (B) evaluation
cells.  It does not alter the architecture-screen contract or write the final
A/B/C report.  Its result is explicitly provisional until the matched
unconditioned arm C has trained and been evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

import run_vcg_preference_conditioned_architecture_screen_85k as screen


PROTOCOL = "vcg_preference_conditioned_B_interim_85k_v1"
REPORT_NAME = "conditioned-B-interim-report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(
    project_root: Path, output_root: Path, *, device_name: str
) -> dict:
    contract = screen._require_contract(project_root, output_root)
    jobs = []
    for model_seed in screen.MODEL_SEEDS:
        for value in screen.LAMBDA_GRID:
            ledger = screen.evaluate_cell(
                project_root,
                output_root,
                architecture="conditioned",
                model_seed=model_seed,
                preference_lambda=value,
                device_name=device_name,
            )
            jobs.append(
                {
                    "model_seed": model_seed,
                    "lambda": value,
                    "strict_safe_complete_rows": ledger[
                        "strict_safe_complete_rows"
                    ],
                }
            )
    return {
        "status": "complete",
        "scope": "contracted_conditioned_B_cells_only",
        "contract_sha256": contract["contract_sha256"],
        "job_count": len(jobs),
        "row_count": len(jobs) * len(screen.INSTANCE_SEEDS),
        "jobs": jobs,
    }


def analyze(project_root: Path, output_root: Path) -> dict:
    contract = screen._require_contract(project_root, output_root)
    _arm, _sources, anchor_report, anchor_sha = screen._load_historical_anchor(
        project_root
    )
    rows = screen._load_architecture_rows(output_root, "conditioned")
    arm_a = screen._historical_summary(anchor_report)
    arm_b = screen._architecture_summary(rows)
    performance_gate = screen._advancement_gate(arm_a, arm_b)
    performance_passed = bool(performance_gate["passed"])
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": (
            "complete"
            if arm_b["whole_method_eligible"]
            else "complete_with_B_suppression"
        ),
        "scope": "opened_85k_interim_A_versus_B_development_view",
        "provisional": True,
        "final_architecture_selection_authorized": False,
        "contract_sha256": contract["contract_sha256"],
        "sidecar_source_sha256": _sha256(Path(__file__).resolve()),
        "historical_A_report_sha256": anchor_sha,
        "lambda_grid": list(screen.LAMBDA_GRID),
        "model_seeds": list(screen.MODEL_SEEDS),
        "instance_seeds": list(screen.INSTANCE_SEEDS),
        "new_evaluation_row_count": screen.EXPECTED_ROWS_PER_ARCHITECTURE,
        "arms": {"A": arm_a, "B": arm_b},
        "A_B_nondominated_points": screen._combined_nondominated(
            {"A": arm_a, "B": arm_b}
        ),
        "B_performance_gate": {
            **performance_gate,
            "decision": (
                "B_performance_gate_passed_pending_matched_C"
                if performance_passed
                else "B_performance_gate_failed"
            ),
        },
        "pending_work": {
            "matched_C_training_and_evaluation": True,
            "conditioning_ablation": True,
            "final_A_B_C_report": True,
        },
        "interpretation": (
            "This interim report can determine whether B clears the declared "
            "performance requirements against historical A. It cannot show "
            "that lambda conditioning caused any improvement; that conclusion "
            "requires the matched masked-preference arm C."
        ),
    }
    path = output_root / REPORT_NAME
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != report:
            raise screen.ArchitectureScreenError(
                "existing conditioned-B interim report changed"
            )
    else:
        screen._atomic_json(path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "analyze", "run"))
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root
        / "results/vcg-preference-conditioned-architecture-screen-85k-development"
    )
    torch.set_num_threads(1)
    if args.command == "evaluate":
        result = evaluate(
            project_root, output_root, device_name=args.device
        )
    elif args.command == "analyze":
        result = analyze(project_root, output_root)
    else:
        evaluate(project_root, output_root, device_name=args.device)
        result = analyze(project_root, output_root)
    display = result
    if "B_performance_gate" in result:
        display = {
            "status": result["status"],
            "B_performance_gate": result["B_performance_gate"],
            "final_architecture_selection_authorized": False,
            "report": str((output_root / REPORT_NAME).resolve()),
        }
    print(json.dumps(display, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
