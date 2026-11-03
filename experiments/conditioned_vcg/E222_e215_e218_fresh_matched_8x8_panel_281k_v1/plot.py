#!/usr/bin/env python3
"""Render the completed E222 matched comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from matplotlib.axes import Axes

from experiments.conditioned_vcg.E203_e191_matched_8x8_test_panel_262k_v1 import (
    plot as e203_plot,
)

from . import protocol as p


def _bar(original):
    def renamed(self, *args, **kwargs):
        if kwargs.get("label") == "VCG (E191 pooled)":
            kwargs["label"] = "VCG (long-exploration pooled)"
        return original(self, *args, **kwargs)
    return renamed


def render(report: Path, output: Path, kpi_output: Path) -> tuple[Path, Path]:
    with patch.object(Axes, "bar", _bar(Axes.bar)):
        main_plot = e203_plot.plot(report, output)
        kpi_plot = e203_plot.plot_kpis(report, kpi_output)
    return main_plot, kpi_plot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=p.DEFAULT_OUTPUT / p.REPORT_NAME)
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT / p.PLOT_NAME)
    parser.add_argument("--kpi-output", type=Path, default=p.DEFAULT_OUTPUT / p.KPI_PLOT_NAME)
    args = parser.parse_args()
    for value in render(args.report.resolve(), args.output.resolve(), args.kpi_output.resolve()):
        print(value)


if __name__ == "__main__":
    main()
