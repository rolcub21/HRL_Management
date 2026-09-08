"""Bounded offline relocation-family screen over authenticated E14 queries."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

from PSLAP import viability as v
from PSLAP.viability_dataset import recovery_state_from_dict
from experiments.conditioned_vcg.E14_certification_scalability_95k import (
    capture, program as e14, replay, reuse,
)
from .proofs import ValidatedAnchor, try_relocation


ROOT = Path(__file__).resolve().parents[4]
SCENARIO = "size_10x10_occ_medium"


def select_queries(trace, per_frontier=6):
    """Evenly spaced ordered relocation queries in each completed frontier."""
    completed = {f["frontier_index"] for f in trace["completed_frontiers"]}
    groups = defaultdict(list)
    for query in trace["queries"]:
        if (query["frontier_index"] in completed
                and query["check_role"] == "reconfigure"
                and query["result"] is not None):
            groups[query["frontier_index"]].append(query)
    selected = []
    for frontier in sorted(groups):
        group = groups[frontier]
        count = len(group) if per_frontier is None else min(per_frontier, len(group))
        indices = ([len(group) // 2] if count == 1 else
                   [i * (len(group) - 1) // (count - 1) for i in range(count)])
        selected.extend(group[i] for i in indices)
    return selected


def run(source, output, per_frontier=6, max_seconds=180.0, validation_mode="full"):
    # Authenticate saved provenance without mutating or re-preparing E14.
    contract, _ = e14.authenticate(source)
    _, trace = capture.authenticate_capture(source, contract, SCENARIO)
    records = replay._load_records(source / "path-replay" / f"{SCENARIO}.jsonl", contract, SCENARIO)
    selected = select_queries(trace, per_frontier)
    roots = {
        q["frontier_index"]: q for q in trace["queries"]
        if q["check_role"] == "current_state" and q["result"] is not None
    }
    output.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), Path(__file__).with_name("proofs.py"),
                    ROOT / "PSLAP/viability.py"]
    screen_contract = {
        "protocol": "d12_relocation_family_screen_v2", "scenario": SCENARIO,
        "validation_mode": validation_mode,
        "trace_sha256": trace["trace_sha256"],
        "e14_contract_sha256": contract["contract_sha256"],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_files},
        "selected_query_indices": [q["query_index"] for q in selected],
        "per_frontier": per_frontier, "max_seconds": max_seconds,
        "selection": ("all_completed_relocation_queries" if per_frontier is None else
                      "evenly_spaced_relocations_per_completed_frontier"),
        "fallback": "authenticated_saved_path_cleanup_result;not_reexecuted",
        "platform": platform.platform(), "python": platform.python_version(),
        "online_policy_changed": False, "d11_imported_or_modified": False,
    }
    e14.atomic_json(output / "contract.json", e14.with_hash(screen_contract, "contract_sha256"))
    anchors = {}
    rows = []
    setup_seconds = 0.0
    started = perf_counter()
    censored = False
    with (output / "queries.jsonl").open("w") as stream:
        for query in selected:
            if perf_counter() - started >= max_seconds:
                censored = True
                break
            frontier = query["frontier_index"]
            if frontier not in anchors:
                before = perf_counter()
                root = roots.get(frontier)
                if root is None or root["result"]["status"] != "SAFE":
                    anchors[frontier] = None
                else:
                    state = recovery_state_from_dict(trace["physical_states"][root["physical_state_id"]])
                    anchor = ValidatedAnchor(state, reuse.certificate_from_dict(root["result"]))
                    # Enumeration is performed once per family and charged to setup.
                    successors = {
                        v._apply_legal_action(state, action): action
                        for action in reuse.legal_recovery_actions_optimized(state)
                        if action.kind is v.RecoveryActionKind.RELOCATION
                    }
                    anchors[frontier] = (anchor, successors)
                setup_seconds += perf_counter() - before
            successor = recovery_state_from_dict(trace["physical_states"][query["physical_state_id"]])
            baseline = records.get(replay._query_key(query))
            if baseline is None or not baseline["exact_certificate_and_witness_match"]:
                raise ValueError("missing or non-equivalent E14 path-cleanup baseline")
            baseline_certificate = reuse.certificate_from_dict(baseline["certificate"])
            before = perf_counter()
            family = anchors[frontier]
            action = family[1].get(successor) if family else None
            attempt = (try_relocation(family[0], action, successor, reuse.config_from_dict(query["search"]))
                       if action is not None else None)
            construction_seconds = perf_counter() - before
            certificate = attempt.certificate if attempt else None
            validation_seconds = 0.0
            if certificate is not None:
                before = perf_counter()
                # Independent native full replay is an audit, not needed for suffix composition.
                current = successor
                steps = certificate.witness if validation_mode == "full" else certificate.witness[:2]
                for step in steps:
                    current = v.apply_recovery_action(current, step)
                valid = not current.blocks if validation_mode == "full" else current == family[0].first_successor
                if not valid:
                    raise ValueError("family certificate failed independent completion replay")
                if baseline_certificate.status is v.ViabilityStatus.UNSAFE:
                    raise ValueError("constructive proof contradicts exact UNSAFE baseline")
                validation_seconds = perf_counter() - before
            row = {
                "query_index": query["query_index"], "frontier_index": frontier,
                "physical_state_id": query["physical_state_id"],
                "reason": attempt.reason if attempt else "no_anchor_or_relocation",
                "family_safe": certificate is not None,
                "baseline_status": baseline_certificate.status.value,
                "baseline_search_seconds": baseline["search_seconds"],
                "construction_seconds": construction_seconds,
                "independent_native_replay_seconds": validation_seconds,
                "baseline_witness_macros": baseline_certificate.witness_macro_count,
                "baseline_witness_steps": baseline_certificate.witness_primitive_steps,
                "added_macros": (len(certificate.witness) - len(baseline_certificate.witness)
                                 if certificate and baseline_certificate.is_safe else None),
                "added_steps": (certificate.witness_primitive_steps - baseline_certificate.witness_primitive_steps
                                if certificate and baseline_certificate.is_safe else None),
                "certificate": reuse.certificate_to_dict(certificate) if certificate else None,
            }
            rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
    hits = [r for r in rows if r["family_safe"]]
    saved = sum(r["baseline_search_seconds"] for r in hits)
    construction = sum(r["construction_seconds"] for r in rows)
    validation = sum(r["independent_native_replay_seconds"] for r in rows)
    comparable_hits = [r for r in hits if r["added_macros"] is not None]
    report = {
        "protocol": screen_contract["protocol"], "selected_queries": len(selected),
        "validation_mode": validation_mode,
        "completed_queries": len(rows), "censored": censored,
        "family_safe": len(hits), "exact_fallback_queries": len(rows) - len(hits),
        "reasons": dict(Counter(r["reason"] for r in rows)),
        "coverage": len(hits) / len(rows) if rows else None,
        "sampled_baseline_search_seconds": sum(r["baseline_search_seconds"] for r in rows),
        "avoided_search_seconds_estimate": saved,
        "anchor_validation_and_family_enumeration_seconds": setup_seconds,
        "connection_construction_seconds": construction,
        "independent_native_replay_seconds": validation,
        "net_seconds_estimate_excluding_independent_replay": saved - setup_seconds - construction,
        "net_seconds_estimate_including_independent_replay": saved - setup_seconds - construction - validation,
        "mean_added_macros": (sum(r["added_macros"] for r in comparable_hits)
                              / len(comparable_hits)) if comparable_hits else None,
        "max_added_steps": max((r["added_steps"] for r in hits if r["added_steps"] is not None), default=None),
        "screen_wall_seconds": perf_counter() - started,
        "interpretation": "sampled offline estimate against historical E14 timings; no online speedup or liveness claim",
        "soundness": ("every served witness independently replayed with native legal-action validation"
                      if validation_mode == "full" else
                      "every connection independently replayed with native validation to exact anchor join; anchor suffix validated once"),
    }
    e14.atomic_json(output / "report.json", e14.with_hash(report, "report_sha256"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/vcg-conditioned-e14-certificate-reuse-95k")
    parser.add_argument("--output", type=Path, default=ROOT / "results/vcg-d12-relocation-family-screen-95k")
    parser.add_argument("--per-frontier", type=int, default=6)
    parser.add_argument("--all-relocations", action="store_true")
    parser.add_argument("--validation-mode", choices=("full", "prefix"), default="full")
    parser.add_argument("--max-seconds", type=float, default=180)
    args = parser.parse_args()
    if args.per_frontier < 1 or args.max_seconds <= 0:
        parser.error("sample count and time budget must be positive")
    print(json.dumps(run(args.source, args.output, None if args.all_relocations else args.per_frontier,
                         args.max_seconds, args.validation_mode), indent=2))


if __name__ == "__main__":
    main()
