#!/usr/bin/env python3
"""Replay selected confirmation rows and render macro-level behavior GIFs.

This is an explanatory, inference-only replay.  It neither changes nor
replaces any confirmation row.  Every replay must exactly reproduce its
persisted confirmation outcome before a GIF is written.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Mapping, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

import evaluate_vcg_unified_frozen_lambda_confirmation as confirmation
import evaluate_vcg_unified_frozen_lambda_seed_stability as development
import evaluate_vcg_unified_frozen_lambda_sweep as base
import train_vcg_constrained_v2_3 as v23


PROTOCOL = "vcg_unified_frozen_lambda_confirmation_87k_behavior_casebook_v1"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "results"
    / "vcg-unified-frozen-lambda-confirmation-87k-behavior-view"
)
CASES = (
    {
        "id": "handling_avoidance",
        "title": "Handling avoidance: waiting replaces reconfiguration",
        "model_seed": 17,
        "instance_seed": 87_011,
        "rng_index": 0,
    },
    {
        "id": "counterexample",
        "title": "Counterexample: stochastic path increases handling",
        "model_seed": 17,
        "instance_seed": 87_015,
        "rng_index": 3,
    },
    {
        "id": "timing_without_rehandles",
        "title": "Same rehandles, different timing behavior",
        "model_seed": 16,
        "instance_seed": 87_026,
        "rng_index": 1,
    },
)
LAMBDAS = (0.0, 0.05)
WIDTH = 1500
HEIGHT = 720


class BehaviorViewError(RuntimeError):
    pass


def _font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


F_TITLE = _font(30, bold=True)
F_HEADING = _font(22, bold=True)
F_BODY = _font(17)
F_SMALL = _font(14)
F_CELL = _font(14, bold=True)


def _snapshot(environment) -> dict:
    return {
        "time": int(environment.time_steps),
        "agent": tuple(environment.current_state),
        "rooms": tuple("".join(row) for row in environment.rooms),
        "storage_positions": tuple(tuple(cell) for cell in environment.storage_positions),
        "pickup_cell": tuple(environment.pickup_cell),
        "waiting_cell": tuple(environment.waiting_cell),
        "exit_cells": tuple(tuple(cell) for cell in environment.exit_cells),
        "blocks": tuple(
            {
                "label": block.label,
                "position": None if block.position is None else tuple(block.position),
                "storage_location": (
                    None if block.storage_location is None else tuple(block.storage_location)
                ),
                "stored": bool(block.stored),
                "delivered": bool(block.delivered),
                "carrying": bool(block.carrying),
                "arrival_step": int(block.arrival_step),
                "storage_needed": int(block.storage_steps_needed),
                "storage_elapsed": int(block.storage_steps_elapsed),
            }
            for block in environment.blocks
        ),
    }


def _option_metadata(candidate) -> dict:
    option = candidate.option
    labels = []
    for name in ("block", "target", "_bound_block"):
        value = getattr(option, name, None)
        label = getattr(value, "label", None)
        if label is not None and label not in labels:
            labels.append(str(label))
    cells = {}
    for name in ("source_cell", "destination", "bound_storage_location"):
        value = getattr(option, name, None)
        if value is not None:
            try:
                cells[name] = tuple(value)
            except TypeError:
                pass
    return {
        "block_labels": tuple(labels),
        "cells": cells,
    }


def _saved_row(output: Path, model_seed: int, fixed_lambda: float, instance_seed: int, rng_index: int) -> dict:
    path = (
        output
        / "validation-ledger"
        / f"seed-{model_seed}"
        / f"lambda-{fixed_lambda:.2f}.json"
    )
    ledger = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row for row in ledger["rows"]
        if row["instance_seed"] == instance_seed and row["policy_rng_index"] == rng_index
    ]
    if len(rows) != 1:
        raise BehaviorViewError("saved confirmation row is missing or duplicated")
    return rows[0]


def _assert_replay(saved: Mapping, observed: Mapping) -> None:
    exact = (
        "instance_seed",
        "episode_instance_id",
        "schedule_id",
        "policy_rng_index",
        "policy_rng_seed",
        "steps",
        "physical_rehandles",
        "delivery_count",
        "required_deliveries",
        "selected_action_counts",
        "delivery_deviations",
        "strict_method_success",
        "method_failure_reason",
    )
    for field in exact:
        if v23._json_safe(observed.get(field)) != v23._json_safe(saved.get(field)):
            raise BehaviorViewError(f"inference replay disagrees on {field}")
    for field in ("dense_return", "mean_absolute_error"):
        if not math.isclose(
            float(observed[field]), float(saved[field]), rel_tol=0.0, abs_tol=1e-9
        ):
            raise BehaviorViewError(f"inference replay disagrees on {field}")


def _replay(case: Mapping, fixed_lambda: float, *, device: str) -> dict:
    output = confirmation.DEFAULT_OUTPUT
    parent_root = confirmation.DEFAULT_PARENT
    contract = confirmation.prepare(output, parent_root, device=device)
    manifest = confirmation._validate_manifest(output, contract)
    entries = {entry["instance_seed"]: entry for entry in manifest["instances"]}
    entry = entries[int(case["instance_seed"])]
    parents = {
        seed: development._parent(parent_root, seed) for seed in confirmation.MODEL_SEEDS
    }
    model_seed = int(case["model_seed"])
    trace = []
    with confirmation._runtime_context(model_seed):
        runtime = base._runtime(parents[model_seed], device=device)
        schedule = v23._install_schedule(
            runtime, v23.schedule_for_episode(confirmation.EPISODE, validation=True)
        )
        runtime.set_dual_lambda(fixed_lambda)
        instance = confirmation._load_instance(
            output / entry["relative_path"], entry, runtime.env
        )
        original_execute = runtime._execute_macro
        cumulative_rehandles = 0
        cumulative_deviations = []

        def traced_execute(candidate, *, remaining_steps: int, evaluation: bool):
            nonlocal cumulative_rehandles
            if not trace:
                trace.append(
                    {
                        "macro": 0,
                        "action": "START",
                        "duration": 0,
                        "macro_rehandles": 0,
                        "cumulative_rehandles": 0,
                        "delivery_deviations": (),
                        "cumulative_delivery_deviations": (),
                        "metadata": {},
                        "snapshot": _snapshot(runtime.env),
                    }
                )
            execution = original_execute(
                candidate, remaining_steps=remaining_steps, evaluation=evaluation
            )
            cumulative_rehandles += int(execution.relocations)
            cumulative_deviations.extend(float(value) for value in execution.delivery_deviations)
            trace.append(
                {
                    "macro": len(trace),
                    "action": str(execution.action_type),
                    "duration": int(execution.duration),
                    "macro_rehandles": int(execution.relocations),
                    "cumulative_rehandles": cumulative_rehandles,
                    "delivery_deviations": tuple(execution.delivery_deviations),
                    "cumulative_delivery_deviations": tuple(cumulative_deviations),
                    "metadata": _option_metadata(candidate),
                    "snapshot": _snapshot(runtime.env),
                }
            )
            return execution

        runtime._execute_macro = traced_execute
        runtime.begin_validation_batch()
        original_environment = runtime.env
        runtime.env = confirmation._BoundEnvironment(original_environment, instance)
        try:
            observed = dict(
                runtime.run_episode(
                    instance_seed=int(case["instance_seed"]),
                    training=False,
                    max_steps=confirmation.MAX_STEPS,
                    policy_rng_index=int(case["rng_index"]),
                    policy_rng_seed=confirmation._policy_rng(
                        int(case["instance_seed"]) - confirmation.INSTANCE_SEEDS[0],
                        int(case["rng_index"]),
                    ),
                )
            )
        finally:
            runtime.env = original_environment
            runtime._execute_macro = original_execute
            batch = dict(runtime.end_validation_batch())
        if batch.get("training_agent_unchanged") is not True:
            raise BehaviorViewError("explanatory replay mutated the frozen agent")
    saved = _saved_row(output, model_seed, fixed_lambda, int(case["instance_seed"]), int(case["rng_index"]))
    _assert_replay(saved, observed)
    return {
        "lambda": fixed_lambda,
        "trace": trace,
        "outcome": {
            "dense_return": float(observed["dense_return"]),
            "mean_absolute_error": float(observed["mean_absolute_error"]),
            "steps": int(observed["steps"]),
            "physical_rehandles": int(observed["physical_rehandles"]),
            "delivery_deviations": tuple(float(value) for value in observed["delivery_deviations"]),
            "selected_action_counts": dict(observed["selected_action_counts"]),
        },
        "exactly_reproduces_confirmation_row": True,
    }


def _cell_fill(snapshot: Mapping, cell: tuple[int, int]) -> tuple[int, int, int]:
    row, col = cell
    if snapshot["rooms"][row][col] == "#":
        return (55, 61, 72)
    if cell in snapshot["exit_cells"]:
        return (190, 235, 195)
    if cell == snapshot["pickup_cell"]:
        return (255, 220, 150)
    if cell == snapshot["waiting_cell"]:
        return (220, 205, 245)
    if cell in snapshot["storage_positions"]:
        return (205, 228, 245)
    return (248, 248, 246)


def _draw_arm(draw: ImageDraw.ImageDraw, replay: Mapping, frame_index: int, x0: int, title: str) -> None:
    trace = replay["trace"]
    event = trace[min(frame_index, len(trace) - 1)]
    snap = event["snapshot"]
    draw.text((x0, 80), title, font=F_HEADING, fill=(25, 30, 38))
    draw.text(
        (x0, 112),
        f"macro {event['macro']}/{len(trace)-1}  |  t={snap['time']}  |  action={event['action']}",
        font=F_BODY,
        fill=(40, 45, 55),
    )
    grid_x, grid_y, cell_size = x0, 155, 82
    for row in range(len(snap["rooms"])):
        for col in range(len(snap["rooms"][row])):
            cell = (row, col)
            box = (
                grid_x + col * cell_size,
                grid_y + row * cell_size,
                grid_x + (col + 1) * cell_size,
                grid_y + (row + 1) * cell_size,
            )
            draw.rectangle(box, fill=_cell_fill(snap, cell), outline=(120, 125, 135), width=2)
            if cell == snap["pickup_cell"]:
                draw.text((box[0] + 4, box[1] + 4), "PICK", font=F_SMALL, fill=(70, 55, 20))
            elif cell == snap["waiting_cell"]:
                draw.text((box[0] + 4, box[1] + 4), "WAIT", font=F_SMALL, fill=(60, 40, 90))
            elif cell in snap["exit_cells"]:
                draw.text((box[0] + 4, box[1] + 4), "EXIT", font=F_SMALL, fill=(25, 80, 35))
    by_position = defaultdict(list)
    for block in snap["blocks"]:
        if block["position"] is not None and not block["delivered"]:
            by_position[tuple(block["position"])].append(block)
    for (row, col), blocks in by_position.items():
        cx = grid_x + col * cell_size + cell_size // 2
        cy = grid_y + row * cell_size + cell_size // 2 + 7
        labels = ",".join(block["label"] for block in blocks[:3])
        if len(blocks) > 3:
            labels += f"+{len(blocks)-3}"
        draw.rounded_rectangle((cx - 31, cy - 17, cx + 31, cy + 17), radius=7, fill=(70, 130, 195), outline=(25, 70, 120), width=2)
        draw.text((cx, cy), labels, anchor="mm", font=F_CELL, fill="white")
    agent_row, agent_col = snap["agent"]
    ax = grid_x + agent_col * cell_size + cell_size - 18
    ay = grid_y + agent_row * cell_size + 19
    draw.ellipse((ax - 13, ay - 13, ax + 13, ay + 13), fill=(220, 65, 65), outline=(115, 20, 20), width=2)
    draw.text((ax, ay), "A", anchor="mm", font=F_CELL, fill="white")

    panel_x = grid_x + 5 * cell_size + 22
    labels = event["metadata"].get("block_labels", ())
    target = ", ".join(labels) if labels else "-"
    lines = [
        f"target block: {target}",
        f"macro duration: {event['duration']}",
        f"macro rehandles: {event['macro_rehandles']}",
        f"cumulative rehandles: {event['cumulative_rehandles']}",
        f"deliveries: {len(event['cumulative_delivery_deviations'])}/8",
        "",
        "delivery deviations:",
    ]
    deviations = event["cumulative_delivery_deviations"]
    lines.extend(
        f"  B{index+1}: {value:+.0f}"
        for index, value in enumerate(deviations)
    )
    for index, line in enumerate(lines):
        draw.text((panel_x, grid_y + index * 27), line, font=F_BODY, fill=(35, 40, 48))
    outcome = replay["outcome"]
    draw.text(
        (x0, 590),
        f"final: rehandles={outcome['physical_rehandles']}  MAE={outcome['mean_absolute_error']:.2f}  "
        f"return={outcome['dense_return']:.2f}  steps={outcome['steps']}",
        font=F_BODY,
        fill=(20, 25, 32),
    )


def _render_case(case: Mapping, replays: Mapping[float, Mapping], path: Path) -> None:
    frame_count = max(len(replays[value]["trace"]) for value in LAMBDAS)
    frames = []
    for frame_index in range(frame_count):
        image = Image.new("RGB", (WIDTH, HEIGHT), (244, 246, 249))
        draw = ImageDraw.Draw(image)
        draw.text((WIDTH // 2, 22), case["title"], anchor="ma", font=F_TITLE, fill=(20, 25, 35))
        draw.text(
            (WIDTH // 2, 58),
            f"model seed {case['model_seed']} | EpisodeInstance {case['instance_seed']} | RNG {case['rng_index']}",
            anchor="ma",
            font=F_BODY,
            fill=(60, 65, 75),
        )
        _draw_arm(draw, replays[0.0], frame_index, 35, "VCG base  (lambda = 0)")
        _draw_arm(draw, replays[0.05], frame_index, 770, "VCG + handling cost  (lambda = 0.05)")
        draw.line((750, 80, 750, 640), fill=(185, 188, 195), width=2)
        draw.text(
            (WIDTH // 2, 686),
            "Frames align macro-decision index; each panel reports its own simulation time.",
            anchor="mm",
            font=F_SMALL,
            fill=(80, 85, 95),
        )
        frames.append(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = [900] * len(frames)
    durations[-1] = 3000
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise BehaviorViewError(
            "ffmpeg is required for MP4 output; install it or retain the GIFs"
        )
    return executable


def _to_mp4(gif_path: Path, mp4_path: Path) -> None:
    subprocess.run(
        (
            _ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(gif_path),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ),
        check=True,
    )


def _combine_mp4(inputs: Sequence[Path], output: Path) -> None:
    command = [_ffmpeg(), "-y", "-loglevel", "error"]
    for path in inputs:
        command.extend(("-i", str(path)))
    streams = "".join(f"[{index}:v]" for index in range(len(inputs)))
    command.extend(
        (
            "-filter_complex",
            f"{streams}concat=n={len(inputs)}:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        )
    )
    subprocess.run(command, check=True)


def run(output_dir: Path, *, device: str) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    mp4_paths = []
    for case in CASES:
        replays = {value: _replay(case, value, device=device) for value in LAMBDAS}
        gif_path = output / f"{case['id']}.gif"
        mp4_path = output / f"{case['id']}.mp4"
        _render_case(case, replays, gif_path)
        _to_mp4(gif_path, mp4_path)
        mp4_paths.append(mp4_path)
        records.append(
            {
                **case,
                "gif": str(gif_path),
                "mp4": str(mp4_path),
                "lambda0": replays[0.0]["outcome"],
                "lambda005": replays[0.05]["outcome"],
                "lambda0_macro_count": len(replays[0.0]["trace"]) - 1,
                "lambda005_macro_count": len(replays[0.05]["trace"]) - 1,
                "both_replays_exactly_match_confirmation_rows": True,
            }
        )
    combined_path = output / "behavior_casebook_combined.mp4"
    _combine_mp4(mp4_paths, combined_path)
    summary = {
        "protocol": PROTOCOL,
        "role": "qualitative_inference_only_replay_not_confirmation_replacement",
        "cases": records,
        "combined_mp4": str(combined_path),
        "training_or_learning": False,
        "confirmation_rows_modified": False,
    }
    (output / "behavior-casebook.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output_dir, device=args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
