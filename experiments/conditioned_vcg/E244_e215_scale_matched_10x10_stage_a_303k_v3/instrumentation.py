"""E209 telemetry with failure-atomic E244 v3 episode commits."""

from __future__ import annotations

from experiments.conditioned_vcg.E209_e191_method_10x10_longstay_268k_v1 import (
    instrumentation as e209,
)


class E244V3EpisodeFailure(RuntimeError):
    """A strict episode failed and must not advance the resumable checkpoint."""


single_objective_rescore = e209.single_objective_rescore
summarize_runs = e209.summarize_runs


def run_episode(*args, **kwargs) -> dict:
    run = e209.run_episode(*args, **kwargs)
    if not bool(run.get("strict_method_success")):
        reason = run.get("method_failure_reason") or "strict completion failed"
        raise E244V3EpisodeFailure(
            "E244 v3 refuses to commit a failed episode: " + str(reason)
        )
    return run


__all__ = [
    "E244V3EpisodeFailure", "run_episode", "single_objective_rescore",
    "summarize_runs",
]
