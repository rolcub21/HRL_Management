#!/usr/bin/env python3
"""Render the text-light E3 certification companion trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw

import render_vcg_unified_policy_filmstrip as film
from experiments.conditioned_vcg.E03_certification_ablation_90k import run as e3
from experiments.conditioned_vcg.E03_certification_ablation_90k import (
    trace_companion,
)


PROTOCOL = "vcg_conditioned_e03_certification_companion_figure_v2"
DEFAULT_TRACE = e3.DEFAULT_OUTPUT / "e03-companion-trace.json"
DEFAULT_STEM = "e03-c-certified-decision-case"
PHYSICAL_COLOR = (41, 104, 185)
CERTIFIED_COLOR = (18, 145, 112)
DEADLOCK_COLOR = (194, 45, 55)


class CompanionFigureError(ValueError):
    pass


def _canonical_sha(value: Mapping, *, hash_field: str | None = None) -> str:
    payload = dict(value)
    if hash_field is not None:
        payload.pop(hash_field, None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(trace: Mapping) -> None:
    if trace.get("protocol") != trace_companion.PROTOCOL:
        raise CompanionFigureError("unexpected companion trace protocol")
    if trace.get("trace_sha256") != _canonical_sha(
        trace, hash_field="trace_sha256"
    ):
        raise CompanionFigureError("companion trace self hash mismatch")
    if trace.get("training_or_learning") is not False:
        raise CompanionFigureError("companion trace must be inference-only")
    arms = trace.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {e3.PHYSICAL, e3.CERTIFIED}:
        raise CompanionFigureError("both E3 candidate-source arms are required")
    physical = arms[e3.PHYSICAL]
    certified = arms[e3.CERTIFIED]
    if physical.get("common_snapshot") != certified.get("common_snapshot"):
        raise CompanionFigureError("candidate-source rows do not share a state")
    physical_macros = physical.get("macros")
    certified_macros = certified.get("macros")
    if not isinstance(physical_macros, list) or len(physical_macros) != 1:
        raise CompanionFigureError("physical-only row must stop after one macro")
    if not isinstance(certified_macros, list) or len(certified_macros) < 2:
        raise CompanionFigureError("certified row has no consequence trajectory")
    unsafe = physical_macros[0]
    if (
        unsafe.get("selected_action_type") != "accept"
        or unsafe.get("shadow_status") != "UNSAFE"
        or unsafe.get("exact_safe") is not False
    ):
        raise CompanionFigureError("physical-only fork is not the unsafe acceptance")
    first_safe = certified_macros[0]
    if (
        first_safe.get("selected_action_type") != "reconfigure"
        or first_safe.get("shadow_status") != "SAFE"
    ):
        raise CompanionFigureError("certified fork is not the safe reconfiguration")
    for macro in (*physical_macros, *certified_macros):
        if (
            macro.get("option_success") is not True
            or macro.get("failure_reason") is not None
            or macro.get("truncated") is not False
            or int(macro.get("duration", -1))
            != len(macro.get("primitive_steps", ()))
        ):
            raise CompanionFigureError("companion contains an invalid macro")
        film._validate_primitive_path(macro)
    summary = certified.get("full_episode_summary")
    if not isinstance(summary, Mapping) or (
        summary.get("strict_method_success") is not True
        or summary.get("terminal") is not True
        or int(summary.get("completed_deliveries", -1)) != e3.EXPECTED_BLOCKS
    ):
        raise CompanionFigureError("certified companion is not strictly complete")
    terminal = certified.get("terminal_snapshot")
    if not isinstance(terminal, Mapping) or any(
        not block.get("delivered") for block in terminal.get("blocks", ())
    ):
        raise CompanionFigureError("certified terminal snapshot is incomplete")


def _draw_stop(
    draw: ImageDraw.ImageDraw,
    *,
    after_column: int,
    row: int,
) -> None:
    x, y = film._panel_origin(after_column + 1, row)
    previous_x, _ = film._panel_origin(after_column, row)
    cy = y + film.PANEL / 2
    start = previous_x + film.PANEL + 10
    center_x = x + 72
    draw.line((start, cy, center_x - 34, cy), fill=DEADLOCK_COLOR, width=8)
    film._arrowhead(
        draw,
        (start, cy),
        (center_x - 34, cy),
        DEADLOCK_COLOR,
        size=18,
    )
    radius = 32
    points = []
    for index in range(8):
        angle = math.pi / 8 + index * math.pi / 4
        points.append(
            (
                center_x + radius * math.cos(angle),
                cy + radius * math.sin(angle),
            )
        )
    draw.polygon(points, fill=DEADLOCK_COLOR)
    draw.rounded_rectangle(
        (center_x - 19, cy - 5, center_x + 19, cy + 5),
        radius=5,
        fill=(255, 255, 255),
    )


def _draw_fork_arrow(
    draw: ImageDraw.ImageDraw,
    macro: Mapping,
    *,
    column: int,
    row: int,
) -> None:
    """Keep the adjacent-cell fork direction visible between block symbols."""
    source = macro.get("source")
    destination = macro.get("destination")
    if source is None or destination is None:
        return
    start_center = film._center(column, row, source)
    end_center = film._center(column, row, destination)
    start = film._move_toward(start_center, end_center, 24.0)
    end = film._move_toward(end_center, start_center, 24.0)
    color = film._action_color(str(macro["selected_action_type"]))
    draw.line((start, end), fill=color, width=9)
    film._arrowhead(draw, start, end, color, size=8)


def _draw_rejection(
    draw: ImageDraw.ImageDraw,
    macro: Mapping,
    *,
    column: int,
    row: int,
) -> None:
    destination = macro.get("destination")
    if destination is None:
        return
    cx, cy = film._center(column, row, destination)
    radius = 29
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline=DEADLOCK_COLOR,
        width=7,
    )
    delta = 20
    draw.line(
        (cx - delta, cy + delta, cx + delta, cy - delta),
        fill=DEADLOCK_COLOR,
        width=8,
    )


def _draw_complete_badge(
    draw: ImageDraw.ImageDraw,
    *,
    column: int,
    row: int,
) -> None:
    x, y = film._panel_origin(column, row)
    box = (x + 91, y + 294, x + 257, y + 334)
    draw.rounded_rectangle(box, radius=19, fill=CERTIFIED_COLOR)
    draw.text(
        ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
        "COMPLETE",
        anchor="mm",
        font=film.F_LEGEND,
        fill=(255, 255, 255),
    )


def render(trace: Mapping) -> Image.Image:
    validate(trace)
    image = Image.new("RGB", (film.WIDTH, film.HEIGHT), film.BG)
    draw = ImageDraw.Draw(image)
    film._draw_timing_legend(draw)
    arms = trace["arms"]
    rows = (
        (e3.PHYSICAL, "Physical", PHYSICAL_COLOR),
        (e3.CERTIFIED, "Certified", CERTIFIED_COLOR),
    )
    for row, (source, label, rail_color) in enumerate(rows):
        arm = arms[source]
        _, y = film._panel_origin(0, row)
        draw.text(
            (76, y + film.PANEL / 2),
            label,
            anchor="mm",
            font=film.F_LAMBDA,
            fill=rail_color,
        )
        common = arm["common_snapshot"]
        film._draw_panel(
            draw,
            common,
            0,
            row,
            rail_color,
        )
        if source == e3.PHYSICAL:
            fork = arm["macros"][0]
            film._draw_connector(draw, 1, row)
            film._draw_panel(
                draw,
                fork["before"],
                1,
                row,
                rail_color,
                macro=fork,
            )
            _draw_fork_arrow(draw, fork, column=1, row=row)
            film._draw_connector(draw, 2, row)
            film._draw_panel(draw, fork["after"], 2, row, rail_color)
            _draw_stop(draw, after_column=2, row=row)
            continue
        unsafe = arms[e3.PHYSICAL]["macros"][0]
        safe = arm["macros"][0]
        film._draw_connector(draw, 1, row)
        film._draw_panel(
            draw,
            unsafe["before"],
            1,
            row,
            rail_color,
            macro=unsafe,
        )
        _draw_fork_arrow(draw, unsafe, column=1, row=row)
        _draw_rejection(draw, unsafe, column=1, row=row)
        film._draw_connector(draw, 2, row)
        film._draw_panel(
            draw,
            safe["before"],
            2,
            row,
            rail_color,
            macro=safe,
        )
        _draw_fork_arrow(draw, safe, column=2, row=row)
        film._draw_connector(draw, 3, row)
        film._draw_panel(draw, safe["after"], 3, row, rail_color)
        first_delivery = arm["macros"][1]
        film._draw_connector(draw, 4, row)
        film._draw_panel(
            draw,
            first_delivery["after"],
            4,
            row,
            rail_color,
            macro=first_delivery,
        )
        film._draw_connector(draw, 5, row, omitted=True)
        film._draw_panel(
            draw,
            arm["terminal_snapshot"],
            5,
            row,
            rail_color,
        )
        _draw_complete_badge(draw, column=5, row=row)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output-dir", type=Path, default=e3.DEFAULT_OUTPUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    args = parser.parse_args()
    trace_path = args.trace.resolve()
    output_dir = args.output_dir.resolve()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    validate(trace)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{args.stem}.png"
    pdf = output_dir / f"{args.stem}.pdf"
    final = render(trace)
    final.save(png)
    final.save(pdf, "PDF", resolution=180.0)
    manifest = {
        "protocol": PROTOCOL,
        "source_trace": str(trace_path),
        "source_trace_sha256": trace["trace_sha256"],
        "post_hoc_explanation_not_performance_evidence": True,
        "contains_explanatory_text_panels": False,
        "rows": [e3.PHYSICAL, e3.CERTIFIED],
        "outputs": {
            "png": {"path": str(png), "sha256": _sha256(png)},
            "pdf": {"path": str(pdf), "sha256": _sha256(pdf)},
        },
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    manifest_path = output_dir / f"{args.stem}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(png)
    print(pdf)
    print(manifest_path)


if __name__ == "__main__":
    main()
