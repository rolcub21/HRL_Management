#!/usr/bin/env python3
"""Render a text-light two-row VCG policy filmstrip from a primitive trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

import render_vcg_unified_decision_mechanism as video


PROTOCOL = "vcg_unified_policy_filmstrip_render_v2"
LEGACY_TRACE_PROTOCOL = "vcg_unified_controlled_policy_filmstrip_trace_v1"
CONDITIONED_TRACE_PROTOCOL = "vcg_conditioned_controlled_policy_filmstrip_trace_v1"
WIDTH = 2440
HEIGHT = 900
LEFT = 160
TOP = 72
PANEL = 348
GAP = 30
ROW_GAP = 72
CELL = 60
YARD = CELL * 5
YARD_PAD_X = (PANEL - YARD) // 2
YARD_PAD_Y = 25
BG = (247, 246, 242)
PANEL_BG = (255, 255, 253)
GRID = (191, 199, 207)
WALL = (90, 101, 112)
STORAGE = (229, 239, 248)
EXIT = (201, 237, 217)
PICKUP = (255, 225, 164)
WAITING = (222, 210, 242)
PATH = (248, 248, 245)
LAMBDA0 = (41, 104, 185)
LAMBDA005 = (197, 62, 126)
REHANDLE = (207, 61, 61)
DELIVER = (18, 145, 112)
HOLD = (124, 87, 184)
APPROACH = (65, 77, 90)
AMBER = (237, 170, 53)
TIMING_EARLY = (64, 139, 196)
TIMING_IMMINENT = (235, 169, 55)
TIMING_OVERDUE = (207, 70, 70)
TIMING_UNSET = (139, 148, 158)


class FilmstripRenderError(ValueError):
    pass


def _font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


F_LAMBDA = _font(24, bold=True)
F_BLOCK = _font(18, bold=True)
F_LEGEND = _font(19, bold=True)


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(child) for child in value]
    return value


def _arm_keys(trace: Mapping) -> tuple[str, str]:
    if trace.get("protocol") == LEGACY_TRACE_PROTOCOL:
        return "lambda0", "lambda005"
    if trace.get("protocol") == CONDITIONED_TRACE_PROTOCOL:
        return "lambda0", "lambda_positive"
    raise FilmstripRenderError("unexpected trace protocol")


def validate(trace: Mapping) -> int:
    row_keys = _arm_keys(trace)
    if trace.get("training_or_learning") is not False:
        raise FilmstripRenderError("filmstrip source must be inference-only")
    arms = trace.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != set(row_keys):
        raise FilmstripRenderError("both policy arms are required")
    count = int(trace.get("macro_count_per_arm", 0))
    if count < 5:
        raise FilmstripRenderError("filmstrip requires at least five decisions per row")
    if _json_safe(arms[row_keys[0]]["common_snapshot"]) != _json_safe(
        arms[row_keys[1]]["common_snapshot"]
    ):
        raise FilmstripRenderError("rows do not start from the same state")
    lambdas = tuple(float(arms[key].get("lambda")) for key in row_keys)
    if lambdas[0] != 0.0 or not math.isfinite(lambdas[1]) or lambdas[1] <= 0.0:
        raise FilmstripRenderError("filmstrip lambda coordinates are invalid")
    for label, arm in arms.items():
        rows = arm.get("macros")
        if not isinstance(rows, Sequence) or len(rows) != count:
            raise FilmstripRenderError(f"{label} macro grid is incomplete")
        for macro in rows:
            if (
                macro.get("exact_safe") is not True
                or macro.get("option_success") is not True
                or macro.get("failure_reason") is not None
                or macro.get("truncated") is not False
                or int(macro.get("duration", -1)) != len(macro.get("primitive_steps", ()))
            ):
                raise FilmstripRenderError(f"{label} contains an invalid macro")
            _validate_primitive_path(macro)
    if arms[row_keys[0]]["macros"][0].get("selected_key") == arms[row_keys[1]]["macros"][0].get("selected_key"):
        raise FilmstripRenderError("the first displayed decision does not diverge")
    return count


def _validate_primitive_path(macro: Mapping) -> None:
    for step in macro["primitive_steps"]:
        before = tuple(step["before"]["agent"])
        after = tuple(step["after"]["agent"])
        distance = abs(before[0] - after[0]) + abs(before[1] - after[1])
        action = str(step["action"])
        if action in {"UP", "DOWN", "LEFT", "RIGHT"} and distance != 1:
            raise FilmstripRenderError("movement primitive is not cell-adjacent")
        if action in {"PICKUP", "PUTDOWN", "WAIT"} and distance != 0:
            raise FilmstripRenderError("stationary primitive changed agent cell")


def _panel_origin(column: int, row: int) -> tuple[int, int]:
    return LEFT + column * (PANEL + GAP), TOP + row * (PANEL + ROW_GAP)


def _yard_origin(column: int, row: int) -> tuple[int, int]:
    x, y = _panel_origin(column, row)
    return x + YARD_PAD_X, y + YARD_PAD_Y


def _center(column: int, row: int, cell: Sequence[int]) -> tuple[float, float]:
    x, y = _yard_origin(column, row)
    r, c = int(cell[0]), int(cell[1])
    return x + (c + 0.5) * CELL, y + (r + 0.5) * CELL


def _cell_kind(snapshot: Mapping, cell: tuple[int, int]) -> tuple[int, int, int]:
    r, c = cell
    rooms = snapshot["rooms"]
    if rooms[r][c] == "#":
        return WALL
    if cell in {tuple(value) for value in snapshot["exit_cells"]}:
        return EXIT
    if cell == tuple(snapshot["pickup_cell"]):
        return PICKUP
    if cell == tuple(snapshot["waiting_cell"]):
        return WAITING
    if cell in {tuple(value) for value in snapshot["storage_positions"]}:
        return STORAGE
    return PATH


def _blocks(snapshot: Mapping) -> dict[tuple[int, int], list[Mapping]]:
    result: dict[tuple[int, int], list[Mapping]] = {}
    for block in snapshot["blocks"]:
        if block.get("delivered") or block.get("position") is None:
            continue
        result.setdefault(tuple(block["position"]), []).append(block)
    return result


def _timing_color(block: Mapping) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if not bool(block.get("stored")):
        return TIMING_UNSET, (255, 255, 255)
    remaining = int(block["storage_needed"]) - int(block["storage_elapsed"])
    if remaining < 0:
        return TIMING_OVERDUE, (255, 255, 255)
    if remaining <= 20:
        return TIMING_IMMINENT, (42, 42, 38)
    return TIMING_EARLY, (255, 255, 255)


def _timing_outline(fill: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return a darker, saturated border from the block's timing color."""
    return {
        TIMING_EARLY: (29, 95, 143),
        TIMING_IMMINENT: (163, 105, 14),
        TIMING_OVERDUE: (150, 35, 35),
        TIMING_UNSET: (78, 88, 98),
    }[fill]


def _draw_grid(draw: ImageDraw.ImageDraw, snapshot: Mapping, column: int, row: int) -> None:
    x0, y0 = _yard_origin(column, row)
    for r in range(5):
        for c in range(5):
            box = (x0 + c * CELL, y0 + r * CELL, x0 + (c + 1) * CELL, y0 + (r + 1) * CELL)
            draw.rectangle(box, fill=_cell_kind(snapshot, (r, c)), outline=GRID, width=2)


def _draw_blocks(
    draw: ImageDraw.ImageDraw,
    snapshot: Mapping,
    column: int,
    row: int,
    *,
    target: str | None = None,
    target_color: tuple[int, int, int] | None = None,
) -> None:
    for cell, blocks in _blocks(snapshot).items():
        cx, cy = _center(column, row, cell)
        if len(blocks) > 2:
            # Several not-yet-admitted blocks can share the waiting cell.
            # Compress them into a small card stack instead of letting labels
            # spill across neighboring yard cells.
            for layer in range(2, -1, -1):
                shift = layer * 4
                box = (cx - 23 + shift, cy - 19 - shift, cx + 23 + shift, cy + 19 - shift)
                draw.rounded_rectangle(
                    box,
                    radius=6,
                    fill=TIMING_UNSET,
                    outline=_timing_outline(TIMING_UNSET),
                    width=3,
                )
            draw.text(
                (cx + 4, cy - 4),
                f"×{len(blocks)}",
                anchor="mm",
                font=F_BLOCK,
                fill=(255, 255, 255),
            )
            continue
        for offset, block in enumerate(blocks):
            label = str(block["label"])
            shift = (offset - (len(blocks) - 1) / 2) * 18
            radius = 20
            box = (cx - radius + shift, cy - radius, cx + radius + shift, cy + radius)
            fill, text_fill = _timing_color(block)
            if target == label and target_color:
                halo = (box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4)
                draw.rounded_rectangle(halo, radius=10, outline=target_color, width=4)
            draw.rounded_rectangle(
                box,
                radius=7,
                fill=fill,
                outline=_timing_outline(fill),
                width=3,
            )
            draw.text((cx + shift, cy), label, anchor="mm", font=F_BLOCK, fill=text_fill)


def _carried(step: Mapping) -> bool:
    for snapshot_name in ("before", "after"):
        if any(bool(block.get("carrying")) for block in step[snapshot_name]["blocks"]):
            return True
    return False


def _dashed_line(draw: ImageDraw.ImageDraw, start, end, fill, width=4) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    dash, gap = 10, 7
    cursor = 0.0
    while cursor < length:
        stop = min(length, cursor + dash)
        draw.line((x1 + ux * cursor, y1 + uy * cursor, x1 + ux * stop, y1 + uy * stop), fill=fill, width=width)
        cursor += dash + gap


def _arrowhead(draw: ImageDraw.ImageDraw, start, end, fill, size=8) -> None:
    x1, y1 = start
    x2, y2 = end
    angle = math.atan2(y2 - y1, x2 - x1)
    left = (x2 - size * math.cos(angle - 0.55), y2 - size * math.sin(angle - 0.55))
    right = (x2 - size * math.cos(angle + 0.55), y2 - size * math.sin(angle + 0.55))
    draw.polygon((end, left, right), fill=fill)


def _move_toward(start, end, distance: float):
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= distance:
        return start
    return (
        x1 + distance * (x2 - x1) / length,
        y1 + distance * (y2 - y1) / length,
    )


def _trim_route(points: Sequence[tuple[float, float]], *, margin: float = 22.0):
    """Keep a route outside the source and destination symbols."""
    trimmed = list(points)
    if len(trimmed) < 2:
        return trimmed
    trimmed[0] = _move_toward(trimmed[0], trimmed[1], margin)
    trimmed[-1] = _move_toward(trimmed[-1], trimmed[-2], margin)
    return trimmed


def _action_color(action: str) -> tuple[int, int, int]:
    if action == "reconfigure":
        return REHANDLE
    if action == "deliver":
        return DELIVER
    if action == "defer":
        return HOLD
    return AMBER


def _draw_route(draw: ImageDraw.ImageDraw, macro: Mapping, column: int, row: int) -> None:
    action = str(macro["selected_action_type"])
    action_color = _action_color(action)
    approach_runs: list[list[tuple[float, float]]] = []
    carried_runs: list[list[tuple[float, float]]] = []
    for step in macro["primitive_steps"]:
        before = tuple(step["before"]["agent"])
        after = tuple(step["after"]["agent"])
        if before == after:
            continue
        p1 = _center(column, row, before)
        p2 = _center(column, row, after)
        runs = carried_runs if _carried(step) else approach_runs
        if runs and runs[-1][-1] == p1:
            runs[-1].append(p2)
        else:
            runs.append([p1, p2])

    # Keep the unladen approach quiet: one dashed cell-by-cell path without
    # repeated arrowheads that can protrude through a retraced carried path.
    for raw_points in approach_runs:
        points = _trim_route(raw_points)
        for p1, p2 in zip(points, points[1:]):
            _dashed_line(draw, p1, p2, APPROACH, width=4)
        _arrowhead(draw, points[-2], points[-1], APPROACH, size=14)

    # The colored route starts outside its source symbol and terminates at its
    # arrowhead outside the destination symbol.  A single terminal arrowhead
    # communicates direction without clutter or a line continuing behind it.
    for raw_points in carried_runs:
        points = _trim_route(raw_points)
        draw.line(points, fill=action_color, width=10, joint="curve")
        _arrowhead(draw, points[-2], points[-1], action_color, size=18)


def _draw_ghost_origin(draw: ImageDraw.ImageDraw, macro: Mapping, column: int, row: int) -> None:
    label = macro.get("target_label")
    source = macro.get("source")
    if not label or source is None:
        return
    cx, cy = _center(column, row, source)
    color = _action_color(str(macro["selected_action_type"]))
    radius = 23
    draw.rounded_rectangle((cx - radius, cy - radius, cx + radius, cy + radius), radius=8, outline=color, width=3)


def _draw_hold_symbol(draw: ImageDraw.ImageDraw, column: int, row: int) -> None:
    # A compact pause badge replaces the former clock/dot count.  The
    # unchanged yard and changing timing colors show the effect of waiting.
    cx, cy = _center(column, row, (0, 0))
    radius = 22
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=HOLD, outline=(255, 255, 255), width=3)
    draw.rounded_rectangle((cx - 9, cy - 12, cx - 3, cy + 12), radius=2, fill=(255, 255, 255))
    draw.rounded_rectangle((cx + 3, cy - 12, cx + 9, cy + 12), radius=2, fill=(255, 255, 255))


def _draw_agent(draw: ImageDraw.ImageDraw, snapshot: Mapping, column: int, row: int, rail_color) -> None:
    cx, cy = _center(column, row, snapshot["agent"])
    cx += CELL * 0.28
    cy -= CELL * 0.28
    draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=rail_color, outline=(255, 255, 255), width=3)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    snapshot: Mapping,
    column: int,
    row: int,
    rail_color,
    macro: Mapping | None = None,
) -> None:
    x, y = _panel_origin(column, row)
    border = AMBER if column == 1 else (218, 221, 223)
    draw.rounded_rectangle(
        (x + 4, y + 6, x + PANEL + 4, y + PANEL + 6),
        radius=18,
        fill=(226, 225, 220),
    )
    draw.rounded_rectangle((x, y, x + PANEL, y + PANEL), radius=18, fill=PANEL_BG, outline=border, width=7 if column == 1 else 2)
    _draw_grid(draw, snapshot, column, row)
    target = None if macro is None else macro.get("target_label")
    target_color = None if macro is None else _action_color(str(macro["selected_action_type"]))
    if macro is not None:
        _draw_route(draw, macro, column, row)
        _draw_ghost_origin(draw, macro, column, row)
    _draw_blocks(draw, snapshot, column, row, target=target, target_color=target_color)
    if macro is not None and macro["selected_action_type"] == "defer":
        _draw_hold_symbol(draw, column, row)
    _draw_agent(draw, snapshot, column, row, rail_color)


def _draw_connector(
    draw: ImageDraw.ImageDraw,
    column: int,
    row: int,
    *,
    omitted: bool = False,
) -> None:
    if column == 0:
        return
    x, y = _panel_origin(column, row)
    cy = y + PANEL / 2
    start = (x - GAP + 6, cy)
    end = (x - 8, cy)
    if omitted:
        center = (start[0] + end[0]) / 2
        for offset in (-8, 0, 8):
            draw.ellipse(
                (center + offset - 3, cy - 3, center + offset + 3, cy + 3),
                fill=(150, 156, 163),
            )
        return
    connector = (125, 132, 139)
    draw.line((start, end), fill=connector, width=5)
    _arrowhead(draw, start, end, connector, size=14)


def _draw_timing_legend(draw: ImageDraw.ImageDraw) -> None:
    # Minimal notation only: a clock followed by remaining-time color bins.
    base_x, cy = 1042, 34
    draw.ellipse((base_x - 13, cy - 13, base_x + 13, cy + 13), outline=(104, 111, 119), width=3)
    draw.line((base_x, cy, base_x, cy - 8), fill=(104, 111, 119), width=3)
    draw.line((base_x, cy, base_x + 7, cy + 4), fill=(104, 111, 119), width=3)
    items = ((TIMING_EARLY, ">20"), (TIMING_IMMINENT, "0–20"), (TIMING_OVERDUE, "<0"))
    x = base_x + 42
    for color, label in items:
        draw.rounded_rectangle((x, cy - 13, x + 27, cy + 13), radius=7, fill=color, outline=(255, 255, 255), width=2)
        draw.text((x + 38, cy), label, anchor="lm", font=F_LEGEND, fill=(73, 79, 86))
        x += 105


def render(trace: Mapping, *, reveal: int | None = None) -> Image.Image:
    count = validate(trace)
    columns = count + 1
    if columns != 6:
        raise FilmstripRenderError("this paper layout is frozen to one common plus five decisions")
    if reveal is None:
        reveal = columns - 1
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    _draw_timing_legend(draw)
    arms = trace["arms"]
    row_specs = (
        (_arm_keys(trace)[0], LAMBDA0),
        (_arm_keys(trace)[1], LAMBDA005),
    )
    for row, (label, rail_color) in enumerate(row_specs):
        rail_label = f"λ = {float(arms[label]['lambda']):g}"
        _, y = _panel_origin(0, row)
        draw.text((76, y + PANEL / 2), rail_label, anchor="mm", font=F_LAMBDA, fill=rail_color)
        arm = arms[label]
        _draw_panel(draw, arm["common_snapshot"], 0, row, rail_color)
        for index, macro in enumerate(arm["macros"], start=1):
            if index > reveal:
                continue
            _draw_connector(
                draw,
                index,
                row,
                omitted=int(macro.get("omitted_before", 0)) > 0,
            )
            _draw_panel(draw, macro["after"], index, row, rail_color, macro=macro)
    return image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="vcg-policy-filmstrip")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--keep-stages", action="store_true")
    parser.add_argument("--keep-final-stage", action="store_true")
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    validate(trace)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / f"{args.stem}.png"
    pdf = args.output_dir / f"{args.stem}.pdf"
    mp4 = args.output_dir / f"{args.stem}.mp4"
    final = render(trace)
    final.save(png)
    final.save(pdf, "PDF", resolution=180.0)
    final_stage = args.output_dir / f"{args.stem}-stage-5.png"
    if args.keep_final_stage:
        final.save(final_stage)
    stage_paths = []
    if not args.no_video:
        with tempfile.TemporaryDirectory(prefix="vcg-filmstrip-") as temporary:
            temp = Path(temporary)
            for reveal in range(6):
                path = temp / f"stage-{reveal}.png"
                render(trace, reveal=reveal).save(path)
                stage_paths.append(path)
            video.write_video(stage_paths, mp4, fps=24, hold=1.15, fade=0.22)
            if args.keep_stages:
                for index, path in enumerate(stage_paths):
                    (args.output_dir / f"{args.stem}-stage-{index}.png").write_bytes(path.read_bytes())
    manifest = {
        "protocol": PROTOCOL,
        "trace": str(args.trace.resolve()),
        "trace_sha256": trace["trace_sha256"],
        "outputs": {
            "png": {"path": str(png.resolve()), "sha256": _sha256(png)},
            "pdf": {"path": str(pdf.resolve()), "sha256": _sha256(pdf)},
            "mp4": None if args.no_video else {"path": str(mp4.resolve()), "sha256": _sha256(mp4)},
            "final_stage_png": (
                {"path": str(final_stage.resolve()), "sha256": _sha256(final_stage)}
                if args.keep_final_stage
                else None
            ),
        },
        "contains_explanatory_text": False,
        "rows": list(_arm_keys(trace)),
        "columns": 6,
    }
    manifest_path = args.output_dir / f"{args.stem}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(png)
    print(pdf)
    if not args.no_video:
        print(mp4)
    print(manifest_path)


if __name__ == "__main__":
    main()
