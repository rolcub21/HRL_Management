#!/usr/bin/env python3
"""Select event milestones from a longer controlled VCG filmstrip trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _canonical_sha(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _indices(value: str) -> list[int]:
    result = [int(item) for item in value.split(",")]
    if len(result) != 5 or result != sorted(set(result)) or result[0] < 1:
        raise argparse.ArgumentTypeError("event indices must be five increasing 1-based integers")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda0", type=_indices, required=True)
    parser.add_argument("--lambda005", type=_indices, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("protocol") != "vcg_unified_controlled_policy_filmstrip_trace_v1":
        raise ValueError("unexpected source trace protocol")
    result = {key: value for key, value in source.items() if key not in {"arms", "trace_sha256"}}
    result["macro_count_per_arm"] = 5
    result["event_selected_from_trace"] = {
        "path": str(args.input.resolve()),
        "trace_sha256": source["trace_sha256"],
        "lambda0_indices_1_based": args.lambda0,
        "lambda005_indices_1_based": args.lambda005,
    }
    result["arms"] = {}
    for label, selected_indices in (("lambda0", args.lambda0), ("lambda005", args.lambda005)):
        arm = source["arms"][label]
        rows = arm["macros"]
        if selected_indices[-1] > len(rows):
            raise ValueError(f"{label} event index exceeds source trace")
        selected = []
        previous = 0
        for index in selected_indices:
            row = dict(rows[index - 1])
            row["source_macro_index_1_based"] = index
            row["omitted_before"] = index - previous - 1
            selected.append(row)
            previous = index
        result["arms"][label] = {
            **{key: value for key, value in arm.items() if key != "macros"},
            "macros": selected,
        }
    result["trace_sha256"] = _canonical_sha(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "trace_sha256": result["trace_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
