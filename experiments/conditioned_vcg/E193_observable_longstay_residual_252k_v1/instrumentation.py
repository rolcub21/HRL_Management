"""E193 telemetry layered on the matched E191 workload instrumentation."""

from __future__ import annotations

import builtins
from unittest.mock import patch

from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    instrumentation as e191,
)


single_objective_rescore = e191.single_objective_rescore
summarize_runs = e191.summarize_runs


def run_episode(*args, **kwargs) -> dict:
    inherited_print = builtins.print

    def e193_print(*values, **keywords):
        if values and isinstance(values[0], str):
            first = values[0]
            if first.startswith("E191 "):
                values = ("E193 " + first[len("E191 "):], *values[1:])
        return inherited_print(*values, **keywords)

    with patch.object(builtins, "print", e193_print):
        run = e191.run_episode(*args, **kwargs)
    run["e193_learning_interface"] = {
        "observable_arrival_context": True,
        "event_defer": True,
        "nominal_cost_residual": True,
    }
    return run


__all__ = ["run_episode", "single_objective_rescore", "summarize_runs"]

