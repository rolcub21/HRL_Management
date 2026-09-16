#!/usr/bin/env python3
"""Aggregate common-evaluation CSV rows without conflating timing metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from example.helper.timing_metrics import summarize_delivery_timing


GROUP_FIELDS = ("method", "lambda", "mu", "target_window")


def _finite_float(value):
    number = float(value)
    return number if np.isfinite(number) else None


def summarize_group(rows):
    if not rows:
        raise ValueError("cannot summarize an empty evaluation group")
    target_window = float(rows[0]["target_window"])
    deviations = [
        float(value)
        for row in rows
        for value in json.loads(row["delivery_deviations"])
    ]
    returns = np.asarray([float(row["return"]) for row in rows], dtype=float)
    successes = np.asarray([float(row["success"]) for row in rows], dtype=float)
    episode_signed = [
        value
        for row in rows
        if (value := _finite_float(row["mean_signed_deviation"])) is not None
    ]
    episode_absolute = [
        value
        for row in rows
        if (value := _finite_float(row["mean_absolute_error"])) is not None
    ]
    timing = summarize_delivery_timing(deviations, target_window)
    return {
        **{field: rows[0][field] for field in GROUP_FIELDS},
        "episode_count": len(rows),
        "mean_return": float(returns.mean()),
        "return_std_across_episodes": float(returns.std(ddof=0)),
        "success_rate": float(successes.mean()),
        "mean_episode_signed_deviation": (
            float(np.mean(episode_signed)) if episode_signed else np.nan
        ),
        "signed_deviation_std_across_episodes": (
            float(np.std(episode_signed, ddof=0)) if episode_signed else np.nan
        ),
        "mean_episode_absolute_error": (
            float(np.mean(episode_absolute)) if episode_absolute else np.nan
        ),
        "absolute_error_std_across_episodes": (
            float(np.std(episode_absolute, ddof=0)) if episode_absolute else np.nan
        ),
        **timing,
    }


def summarize_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in GROUP_FIELDS)
        groups[key].append(row)
    return [summarize_group(groups[key]) for key in sorted(groups)]


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = summarize_rows(rows)
    rendered = json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"Results: {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
