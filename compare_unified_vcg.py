#!/usr/bin/env python3
"""Compare the two terminal unified-VCG development ledgers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Mapping, Sequence

from train_vcg_unified import (
    METHOD_VERSION,
    VCG,
    VCG_HANDLING_CONSTRAINT,
)


T_95_DF_11 = 2.200985160082949


def _read_mapping(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _load_arm(root: Path, expected_variant: str) -> tuple[dict, list[dict]]:
    summary = _read_mapping(root / "training-summary.json")
    ledger = _read_mapping(root / "validation-ledger" / "episode-0200.json")
    if summary.get("status") != "complete":
        raise ValueError(f"{expected_variant} training is not complete")
    if summary.get("method_version") != METHOD_VERSION:
        raise ValueError(f"{expected_variant} has the wrong method version")
    if summary.get("variant") != expected_variant:
        raise ValueError(f"expected {expected_variant}, found {summary.get('variant')!r}")
    if ledger.get("unified_variant") != expected_variant:
        raise ValueError(f"{expected_variant} ledger binding failed")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != 48 or ledger.get("row_count") != 48:
        raise ValueError(f"{expected_variant} requires all 48 terminal rows")
    if summary.get("final_validation", {}).get("strict_integrity_gate") is not True:
        raise ValueError(f"{expected_variant} terminal validation is unsafe/incomplete")
    return summary, rows


def _instance_points(rows: Sequence[Mapping]) -> dict[int, dict]:
    groups: dict[int, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[int(row["instance_seed"])].append(row)
    if len(groups) != 12 or any(len(group) != 4 for group in groups.values()):
        raise ValueError("expected 12 EpisodeInstances crossed with four action RNGs")
    points = {}
    for seed, group in groups.items():
        delivered = sum(int(row["required_deliveries"]) for row in group)
        deviations = []
        for row in group:
            row_deviations = row.get("delivery_deviations")
            if (
                not isinstance(row_deviations, (list, tuple))
                or len(row_deviations) != int(row["required_deliveries"])
                or any(
                    isinstance(value, bool) or not math.isfinite(float(value))
                    for value in row_deviations
                )
            ):
                raise ValueError("a terminal row has invalid delivery deviations")
            observed_mae = fmean(abs(float(value)) for value in row_deviations)
            if not math.isclose(
                observed_mae,
                float(row["mean_absolute_error"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("stored MAE disagrees with delivery deviations")
            deviations.extend(float(value) for value in row_deviations)
        points[seed] = {
            "dense_return": fmean(float(row["dense_return"]) for row in group),
            "mean_absolute_error": fmean(float(row["mean_absolute_error"]) for row in group),
            "physical_rehandles_per_100": 100.0
            * sum(int(row["physical_rehandles"]) for row in group)
            / delivered,
            "steps": fmean(float(row["steps"]) for row in group),
            "within_window_percentage": 100.0
            * sum(abs(deviation) <= 20.0 for deviation in deviations)
            / delivered,
            "mean_earliness": fmean(max(-deviation, 0.0) for deviation in deviations),
            "mean_tardiness": fmean(max(deviation, 0.0) for deviation in deviations),
        }
    return points


def _paired_difference(
    constrained: Mapping[int, Mapping],
    unconstrained: Mapping[int, Mapping],
    field: str,
) -> dict:
    seeds = sorted(unconstrained)
    if seeds != sorted(constrained):
        raise ValueError("paired EpisodeInstance seeds do not match")
    values = [float(constrained[seed][field]) - float(unconstrained[seed][field]) for seed in seeds]
    mean = float(fmean(values))
    se = float(stdev(values) / math.sqrt(len(values)))
    return {
        "direction": "VCG + handling constraint minus VCG",
        "mean_difference": mean,
        "standard_error": se,
        "nominal_95_ci": (mean - T_95_DF_11 * se, mean + T_95_DF_11 * se),
        "paired_episode_instances": len(values),
    }


def compare(unconstrained_dir: Path, constrained_dir: Path) -> dict:
    u_summary, u_rows = _load_arm(unconstrained_dir, VCG)
    c_summary, c_rows = _load_arm(constrained_dir, VCG_HANDLING_CONSTRAINT)
    if u_summary.get("shared_configuration_sha256") != c_summary.get(
        "shared_configuration_sha256"
    ):
        raise ValueError("the two arms do not share an identical configuration")
    u = _instance_points(u_rows)
    c = _instance_points(c_rows)
    fields = (
        "dense_return",
        "mean_absolute_error",
        "physical_rehandles_per_100",
        "steps",
        "within_window_percentage",
        "mean_earliness",
        "mean_tardiness",
    )
    result = {
        "status": "complete",
        "method_version": METHOD_VERSION,
        "shared_configuration_sha256": u_summary["shared_configuration_sha256"],
        "statistical_unit": "EpisodeInstance; four action RNGs reduced within instance first",
        "terminal_checkpoint_episode": 200,
        "arms": {
            VCG: {field: fmean(point[field] for point in u.values()) for field in fields},
            VCG_HANDLING_CONSTRAINT: {
                field: fmean(point[field] for point in c.values()) for field in fields
            },
        },
        "paired_differences": {
            field: _paired_difference(c, u, field) for field in fields
        },
        "development_only": True,
        "final_86xxx_panel_opened": False,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcg-dir", type=Path, required=True)
    parser.add_argument("--constrained-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = compare(args.vcg_dir, args.constrained_dir)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
