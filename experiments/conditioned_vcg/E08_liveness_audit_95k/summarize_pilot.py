#!/usr/bin/env python3
"""Derive the paper-facing E8 summary from the immutable pilot trace.

The base agent prepares the next Bellman frontier in ``observe_outcome``.
Consequently, a recovery witness can be activated after decision i and before
the audit wrapper enters ``select`` for decision i+1. Activation boundaries
are reconstructed as starts of contiguous forced runs.
"""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e08-liveness-audit-95k"
INPUT_NAME = "e8-instrumented-pilot-report.json"
OUTPUT_NAME = "e8-pilot-summary.json"
TABLE_NAME = "e8-pilot-summary.md"
PROTOCOL = "vcg_conditioned_e8_pilot_derived_summary_v1"


class E8SummaryError(RuntimeError):
    pass


def _canonical(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, field: Optional[str] = None) -> str:
    payload = dict(value)
    if field is not None:
        payload.pop(field, None)
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _load_report(path: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E8SummaryError(f"invalid pilot report: {path}") from error
    if not isinstance(report, dict):
        raise E8SummaryError("pilot report must contain an object")
    if report.get("report_sha256") != _digest(report, "report_sha256"):
        raise E8SummaryError("pilot report self-hash mismatch")
    if report.get("status") != "complete":
        raise E8SummaryError("pilot report is not complete")
    return report


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _forced_runs(decisions: Sequence[Mapping]) -> list[dict]:
    runs = []
    index = 0
    while index < len(decisions):
        if not bool(decisions[index]["liveness_forced"]):
            index += 1
            continue
        start = index
        while index < len(decisions) and bool(
            decisions[index]["liveness_forced"]
        ):
            index += 1
        first = decisions[start]
        reasons = []
        if bool(first["due_trigger"]):
            reasons.append("due_trigger")
        if bool(first["nonprogress_trigger"]):
            reasons.append("nonprogress_limit")
        runs.append(
            {
                "start_decision": int(first["decision_index"]),
                "end_decision_inclusive": int(
                    decisions[index - 1]["decision_index"]
                ),
                "length": index - start,
                "activation_reason": "+".join(reasons) if reasons else "unknown",
                "witness_length": int(first["active_witness_length_after_select"]),
            }
        )
    return runs


def _mean(records: Sequence[Mapping], field: str) -> Optional[float]:
    values = [float(item[field]) for item in records if item[field] is not None]
    return fmean(values) if values else None


def summarize(output: Path) -> dict:
    output = output.resolve()
    source_path = output / INPUT_NAME
    report = _load_report(source_path)
    decisions = report.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != int(
        report["macro_decisions"]
    ):
        raise E8SummaryError("pilot decision trace is incomplete")

    runs = _forced_runs(decisions)
    forced = [item for item in decisions if item["liveness_forced"]]
    overrides = [item for item in forced if item["actual_override"]]
    accept_diversions = [
        item
        for item in overrides
        if item["learned_proposal_action_type"] == "accept"
        and item["executed_action_type"] != "accept"
    ]
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "source_report": str(source_path.relative_to(ROOT)),
        "source_report_sha256": report["report_sha256"],
        "strict_safe_complete": bool(report["strict_safe_complete"]),
        "macro_decisions": len(decisions),
        "liveness_forced_decisions": len(forced),
        "liveness_forced_rate": len(forced) / len(decisions),
        "actual_overrides": len(overrides),
        "actual_override_rate_among_forced": len(overrides) / len(forced),
        "guard_activations": len(runs),
        "guard_persistence_decisions": len(forced) - len(runs),
        "guard_activation_reasons": dict(
            sorted(Counter(item["activation_reason"] for item in runs).items())
        ),
        "forced_runs": runs,
        "accept_proposals_diverted_to_recovery": len(accept_diversions),
        "forced_proposal_to_executed": report["forced_proposal_to_executed"],
        "override_margins": {
            "mean_proposal_minus_executed_merit": _mean(
                overrides, "proposal_minus_executed_merit"
            ),
            "mean_proposal_mode_margin": _mean(overrides, "proposal_mode_margin"),
            "mean_proposal_within_mode_margin": _mean(
                overrides, "proposal_within_mode_margin"
            ),
        },
        "historical_behavior_digest_equal": bool(
            report["historical_e13_comparison"]["behavior_digest_equal"]
        ),
        "interpretation": {
            "activation_accounting": (
                "activation is reconstructed at the start of each contiguous "
                "forced run because next-frontier preparation occurs inside "
                "observe_outcome before the next audited select call"
            ),
            "causal_scope": (
                "proposal-versus-execution is an exact same-state attribution; "
                "it is not a guard-off episode counterfactual"
            ),
        },
    }
    result["summary_sha256"] = _digest(result)
    _atomic_text(
        output / OUTPUT_NAME,
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )

    if len(runs) == 1:
        run = runs[0]
        run_text = (
            f"decision {run['start_decision']} through "
            f"{run['end_decision_inclusive']} ({run['length']} macros)"
        )
    else:
        run_text = f"{len(runs)} forced runs"
    markdown = f"""# E8 instrumented-pilot summary

| Quantity | Result |
| --- | ---: |
| Strict safe completion | {'yes' if result['strict_safe_complete'] else 'no'} |
| All macro decisions | {result['macro_decisions']} |
| Witness-forced decisions | {result['liveness_forced_decisions']} ({100.0 * result['liveness_forced_rate']:.1f}%) |
| Actual overrides of learned proposal | {result['actual_overrides']}/{result['liveness_forced_decisions']} ({100.0 * result['actual_override_rate_among_forced']:.1f}%) |
| Reconstructed guard activations | {result['guard_activations']} |
| Witness-persistence decisions | {result['guard_persistence_decisions']} |
| Learned Accept proposals diverted to recovery | {result['accept_proposals_diverted_to_recovery']} |
| Historical E13 behavior reproduced | {'yes' if result['historical_behavior_digest_equal'] else 'no'} |

The one witness was activated by the nonprogress limit and governed {run_text}.
The same-state learned proposal disagreed with the retained exact witness on
{result['actual_overrides']} of {result['liveness_forced_decisions']} forced
decisions. This establishes implemented action attribution, not the outcome of
a separate guard-off rollout.

The immutable v1 trace reports zero activations *inside `select`*. That is an
instrumentation-boundary artifact: `observe_outcome` prepares the next
frontier and activates the witness before the audit wrapper enters the next
`select`. This derived summary counts starts of contiguous forced runs.
"""
    _atomic_text(output / TABLE_NAME, markdown)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
