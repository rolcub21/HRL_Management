#!/usr/bin/env python3
"""Matplotlib compatibility entry point for the frozen 88k plotter.

The source-bound plotter passed seven explicit marker positions as a tuple.
Matplotlib 3.10 reserves tuples for its two-item ``(start, stride)`` form.
Convert only that presentation argument to a list, then run the unchanged
authenticated data/plot pipeline.
"""

from __future__ import annotations

from matplotlib.axes import Axes


_ORIGINAL_PLOT = Axes.plot


def _compatible_plot(self, *args, **kwargs):
    markevery = kwargs.get("markevery")
    if isinstance(markevery, tuple) and len(markevery) != 2:
        kwargs["markevery"] = list(markevery)
    return _ORIGINAL_PLOT(self, *args, **kwargs)


Axes.plot = _compatible_plot

from plot_vcg_v11_nested_handling_confirmation_88k import main  # noqa: E402


if __name__ == "__main__":
    main()
