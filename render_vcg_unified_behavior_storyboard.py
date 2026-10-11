#!/usr/bin/env python3
"""Render one concise, reader-facing VCG behavior comparison.

The detailed renderer preserves every macro decision.  This companion view
uses the same authenticated replay but keeps only five explanatory moments so
the storage constraint and the policy difference are visible at a glance.
It is an illustration, not a new evaluation row or statistical result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

import render_vcg_unified_behavior_gifs as detailed


PROTOCOL = "vcg_unified_frozen_lambda_confirmation_87k_reader_storyboard_v1"
CASE = {
    "id": "single_rehandle_avoided",
    "title": "One physical move avoided under capacity-tight demand",
    "model_seed": 16,
    "instance_seed": 87_013,
    "rng_index": 3,
}
LAMBDAS = detailed.LAMBDAS
DEFAULT_OUTPUT = detailed.DEFAULT_OUTPUT

WIDTH = 1600
HEIGHT = 900
PANEL_WIDTH = 750
GRID_CELL = 88
GRID_SIZE = GRID_CELL * 5

STAGES = (
    {
        "id": "common_placement",
        "title": "1  Same controller and same early placements",
        "caption": "Both policies place B1-B3 identically; the hard exact-safety layer is unchanged.",
        "indices": {0.0: 3, 0.05: 3},
    },
    {
        "id": "capacity_tight",
        "title": "2  Capacity-tight demand",
        "caption": "Seven of eight storage cells are occupied; the eighth block is still inbound.",
        "indices": {0.0: 8, 0.05: 8},
    },
    {
        "id": "pre_decision",
        "title": "3  Nearly identical state before the only action difference",
        "caption": "Seven deliveries are complete; B8 is the sole stored block in both yards.",
        "indices": {0.0: 20, 0.05: 20},
    },
    {
        "id": "key_access_decision",
        "title": "4  Key access decision: move a block or deliver it",
        "caption": "Base VCG moves B8 once; the handling-cost augmentation delivers B8 directly.",
        "indices": {0.0: 21, 0.05: 21},
    },
    {
        "id": "outcome",
        "title": "5  Same safe task, one unnecessary move removed",
        "caption": "Both finish with nearly identical timing; physical rehandles fall from 1 to 0.",
        "indices": {0.0: 22, 0.05: 21},
    },
)


class StoryboardError(RuntimeError):
    pass


def _font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return (
        ImageFont.truetype(str(path), size=size)
        if path.is_file()
        else ImageFont.load_default()
    )


F_TITLE = _font(30, bold=True)
F_STAGE = _font(25, bold=True)
F_HEADING = _font(21, bold=True)
F_BODY = _font(18)
F_SMALL = _font(15)
F_CELL = _font(16, bold=True)
F_BADGE = _font(18, bold=True)


def _block_at(snapshot: Mapping, cell) -> Optional[str]:
    if cell is None:
        return None
    position = tuple(cell)
    for block in snapshot["blocks"]:
        if (
            block["position"] is not None
            and tuple(block["position"]) == position
            and not block["delivered"]
        ):
            return str(block["label"])
    return None


def _event_context(replay: Mapping, index: int) -> dict:
    trace = replay["trace"]
    if index < 0 or index >= len(trace):
        raise StoryboardError(f"storyboard index {index} is outside the replay")
    event = trace[index]
    before = trace[max(0, index - 1)]["snapshot"]
    cells = dict(event["metadata"].get("cells", {}))
    labels = list(event["metadata"].get("block_labels", ()))
    source = cells.get("source_cell")
    if not labels:
        inferred = _block_at(before, source)
        if inferred is not None:
            labels.append(inferred)
    return {
        "event": event,
        "before": before,
        "source": None if source is None else tuple(source),
        "destination": (
            None
            if cells.get("destination") is None
            else tuple(cells["destination"])
        ),
        "labels": tuple(labels),
    }


def _occupancy(snapshot: Mapping) -> tuple[int, int]:
    storage = {tuple(cell) for cell in snapshot["storage_positions"]}
    occupied = {
        tuple(block["position"])
        for block in snapshot["blocks"]
        if block["position"] is not None
        and tuple(block["position"]) in storage
        and not block["delivered"]
    }
    return len(occupied), len(storage)


def _cell_center(x0: int, y0: int, cell: tuple[int, int]) -> tuple[int, int]:
    row, col = cell
    return (
        x0 + col * GRID_CELL + GRID_CELL // 2,
        y0 + row * GRID_CELL + GRID_CELL // 2,
    )


def _arrow(draw: ImageDraw.ImageDraw, start, end, *, color, width: int = 7) -> None:
    draw.line((*start, *end), fill=color, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    left = (end[0] - 18 * ux + 10 * uy, end[1] - 18 * uy - 10 * ux)
    right = (end[0] - 18 * ux - 10 * uy, end[1] - 18 * uy + 10 * ux)
    draw.polygon((end, left, right), fill=color)


def _action_text(context: Mapping) -> str:
    event = context["event"]
    action = str(event["action"])
    target = ", ".join(context["labels"]) or "yard"
    if action == "accept":
        return f"STORE {target}"
    if action == "deliver":
        return f"DELIVER {target}"
    if action == "reconfigure":
        return f"MOVE {target} AGAIN"
    if action == "defer":
        return "WAIT FOR NEXT EVENT"
    if action == "START":
        return "INITIAL STATE"
    return action.upper()


def _draw_yard(
    draw: ImageDraw.ImageDraw,
    context: Mapping,
    *,
    x0: int,
    y0: int,
) -> None:
    event = context["event"]
    snapshot = event["snapshot"]
    source = context["source"]
    destination = context["destination"]
    focus = set(context["labels"])
    action_color = {
        "accept": (32, 115, 190),
        "reconfigure": (195, 52, 48),
        "deliver": (38, 145, 72),
    }.get(str(event["action"]), (110, 115, 125))

    for row in range(len(snapshot["rooms"])):
        for col in range(len(snapshot["rooms"][row])):
            cell = (row, col)
            box = (
                x0 + col * GRID_CELL,
                y0 + row * GRID_CELL,
                x0 + (col + 1) * GRID_CELL,
                y0 + (row + 1) * GRID_CELL,
            )
            draw.rectangle(
                box,
                fill=detailed._cell_fill(snapshot, cell),
                outline=(120, 125, 135),
                width=2,
            )
            if cell == snapshot["pickup_cell"]:
                draw.text((box[0] + 5, box[1] + 5), "IN", font=F_SMALL, fill=(75, 52, 15))
            elif cell == snapshot["waiting_cell"]:
                draw.text((box[0] + 5, box[1] + 5), "WAIT", font=F_SMALL, fill=(65, 42, 95))
            elif cell in snapshot["exit_cells"]:
                draw.text((box[0] + 5, box[1] + 5), "OUT", font=F_SMALL, fill=(22, 88, 40))

    by_position = defaultdict(list)
    for block in snapshot["blocks"]:
        if block["position"] is not None and not block["delivered"]:
            by_position[tuple(block["position"])].append(block)
    for cell, blocks in by_position.items():
        cx, cy = _cell_center(x0, y0, cell)
        labels = ",".join(str(block["label"]) for block in blocks)
        highlighted = any(str(block["label"]) in focus for block in blocks)
        fill = (238, 150, 45) if highlighted else (92, 130, 168)
        outline = (135, 70, 5) if highlighted else (40, 75, 110)
        draw.rounded_rectangle(
            (cx - 31, cy - 19, cx + 31, cy + 19),
            radius=8,
            fill=fill,
            outline=outline,
            width=3,
        )
        draw.text((cx, cy), labels, anchor="mm", font=F_CELL, fill="white")

    if source is not None:
        sx, sy = _cell_center(x0, y0, source)
        draw.ellipse(
            (sx - 38, sy - 38, sx + 38, sy + 38),
            outline=action_color,
            width=6,
        )
        if destination is not None:
            dx, dy = _cell_center(x0, y0, destination)
            _arrow(draw, (sx, sy), (dx, dy), color=action_color)
        elif event["action"] == "deliver":
            exit_cell = tuple(snapshot["exit_cells"][0])
            _arrow(
                draw,
                (sx, sy),
                _cell_center(x0, y0, exit_cell),
                color=(38, 145, 72),
            )
    if destination is not None:
        dx, dy = _cell_center(x0, y0, destination)
        draw.rectangle((dx - 39, dy - 39, dx + 39, dy + 39), outline=(32, 115, 190), width=5)

    agent_row, agent_col = snapshot["agent"]
    ax, ay = _cell_center(x0, y0, (agent_row, agent_col))
    ax += 28
    ay -= 27
    draw.ellipse((ax - 14, ay - 14, ax + 14, ay + 14), fill=(220, 65, 65), outline=(110, 25, 25), width=2)
    draw.text((ax, ay), "A", anchor="mm", font=F_CELL, fill="white")


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    replay: Mapping,
    index: int,
    *,
    x0: int,
    title: str,
) -> dict:
    context = _event_context(replay, index)
    event = context["event"]
    snapshot = event["snapshot"]
    occupied, capacity = _occupancy(snapshot)
    draw.rounded_rectangle(
        (x0, 155, x0 + PANEL_WIDTH, 746),
        radius=18,
        fill=(252, 253, 254),
        outline=(185, 190, 198),
        width=2,
    )
    draw.text((x0 + 20, 173), title, font=F_HEADING, fill=(24, 30, 38))
    draw.text(
        (x0 + 20, 207),
        f"t={snapshot['time']}   macro {event['macro']}   {_action_text(context)}",
        font=F_BODY,
        fill=(45, 50, 60),
    )
    grid_x, grid_y = x0 + 18, 250
    _draw_yard(draw, context, x0=grid_x, y0=grid_y)

    info_x = grid_x + GRID_SIZE + 22
    ratio = occupied / capacity if capacity else 0.0
    draw.text((info_x, grid_y), "YARD CAPACITY", font=F_SMALL, fill=(70, 75, 85))
    draw.rounded_rectangle(
        (info_x, grid_y + 28, info_x + 225, grid_y + 59),
        radius=8,
        fill=(226, 230, 235),
    )
    draw.rounded_rectangle(
        (info_x, grid_y + 28, info_x + int(225 * ratio), grid_y + 59),
        radius=8,
        fill=(230, 155, 55) if occupied >= capacity - 1 else (67, 145, 205),
    )
    draw.text(
        (info_x + 112, grid_y + 43),
        f"{occupied}/{capacity} occupied",
        anchor="mm",
        font=F_BADGE,
        fill=(20, 28, 38),
    )

    rehandle_color = (180, 42, 42) if event["macro_rehandles"] else (42, 120, 66)
    details = (
        ("selected action", _action_text(context), (35, 40, 48)),
        ("physical move now", str(event["macro_rehandles"]), rehandle_color),
        ("physical moves total", str(event["cumulative_rehandles"]), rehandle_color),
        ("deliveries", f"{len(event['cumulative_delivery_deviations'])}/8", (35, 40, 48)),
    )
    for row, (label, value, color) in enumerate(details):
        y = grid_y + 90 + row * 66
        draw.text((info_x, y), label.upper(), font=F_SMALL, fill=(90, 94, 103))
        draw.text((info_x, y + 24), value, font=F_BADGE, fill=color)

    outcome = replay["outcome"]
    draw.text(
        (x0 + 20, 710),
        f"final: {outcome['physical_rehandles']} rehandles  |  MAE {outcome['mean_absolute_error']:.2f}  |  {outcome['steps']} steps",
        font=F_BODY,
        fill=(28, 34, 42),
    )
    return {
        "macro": int(event["macro"]),
        "time": int(snapshot["time"]),
        "action": str(event["action"]),
        "action_label": _action_text(context),
        "occupied": occupied,
        "capacity": capacity,
        "cumulative_rehandles": int(event["cumulative_rehandles"]),
    }


def _frame(stage: Mapping, replays: Mapping[float, Mapping]) -> tuple[Image.Image, dict]:
    image = Image.new("RGB", (WIDTH, HEIGHT), (240, 243, 247))
    draw = ImageDraw.Draw(image)
    draw.text((WIDTH // 2, 25), stage["title"], anchor="ma", font=F_STAGE, fill=(18, 24, 32))
    draw.text((WIDTH // 2, 65), stage["caption"], anchor="ma", font=F_BODY, fill=(55, 60, 70))
    draw.rounded_rectangle((425, 104, 1175, 139), radius=10, fill=(220, 239, 224))
    draw.text(
        (WIDTH // 2, 121),
        "IDENTICAL EXACT-SAFETY CONSTRAINTS  |  SAME CHECKPOINT + INSTANCE + RNG",
        anchor="mm",
        font=F_SMALL,
        fill=(30, 92, 48),
    )
    panels = {
        "lambda0": _draw_panel(
            draw,
            replays[0.0],
            int(stage["indices"][0.0]),
            x0=25,
            title="VCG  (lambda = 0)",
        ),
        "lambda005": _draw_panel(
            draw,
            replays[0.05],
            int(stage["indices"][0.05]),
            x0=825,
            title="VCG + handling augmentation  (lambda = 0.05)",
        ),
    }
    draw.text(
        (WIDTH // 2, 875),
        "Orange block = current focus  |  blue = initial placement  |  red = physical relocation  |  green = delivery",
        anchor="mm",
        font=F_SMALL,
        fill=(75, 80, 90),
    )
    return image, panels


def run(output_dir: Path, *, device: str) -> dict:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    replays = {value: detailed._replay(CASE, value, device=device) for value in LAMBDAS}
    if not all(replay["exactly_reproduces_confirmation_row"] for replay in replays.values()):
        raise StoryboardError("a replay did not reproduce its saved confirmation row")

    frames = []
    stage_records = []
    for stage in STAGES:
        image, panels = _frame(stage, replays)
        frames.append(image)
        stage_records.append(
            {
                "id": stage["id"],
                "title": stage["title"],
                "caption": stage["caption"],
                "panels": panels,
            }
        )

    gif_path = output / "vcg_handling_augmentation_storyboard.gif"
    mp4_path = output / "vcg_handling_augmentation_storyboard.mp4"
    poster_path = output / "vcg_handling_augmentation_key_decision.png"
    durations = [3600, 4200, 4200, 6000, 6000]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    detailed._to_mp4(gif_path, mp4_path)
    frames[3].save(poster_path)

    summary = {
        "protocol": PROTOCOL,
        "role": "reader_facing_qualitative_storyboard_not_new_evidence",
        "case": dict(CASE),
        "stages": stage_records,
        "lambda0_outcome": replays[0.0]["outcome"],
        "lambda005_outcome": replays[0.05]["outcome"],
        "both_replays_exactly_match_confirmation_rows": True,
        "training_or_learning": False,
        "confirmation_rows_modified": False,
        "gif": str(gif_path),
        "mp4": str(mp4_path),
        "key_decision_png": str(poster_path),
    }
    summary_path = output / "vcg_handling_augmentation_storyboard.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
