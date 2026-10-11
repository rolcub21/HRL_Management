"""Delivery-timing metrics with explicit, non-overloaded semantics."""

from __future__ import annotations

from numbers import Integral

import numpy as np


STORAGE_FLOW_METRIC_CONTRACT = (
    "arrival_to_first_completed_storage_right_censor_v1"
)


def _step(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def summarize_storage_flow(
    arrival_steps,
    stored_steps,
    observation_end_step,
    labels=None,
):
    """Summarize arrival-to-first-storage flow with explicit censoring.

    Flow time is the elapsed number of environment steps from an exogenous
    arrival through the first successful storage PUTDOWN.  An arrived block
    without that event at ``observation_end_step`` is right-censored.  A block
    whose arrival is later than the observation end has not entered the risk
    set and is counted separately.

    Primary distribution statistics are defined only for a fully observed
    manifest.  Completed times and manifest-aligned records remain available
    for diagnostics on incomplete episodes; censor ages are never imputed as
    completed flow times.
    """

    arrivals = tuple(arrival_steps)
    completions = tuple(stored_steps)
    if len(arrivals) != len(completions):
        raise ValueError("arrival_steps and stored_steps must have equal length")
    if labels is None:
        block_labels = (None,) * len(arrivals)
    else:
        block_labels = tuple(labels)
        if len(block_labels) != len(arrivals):
            raise ValueError("labels must have the same length as arrival_steps")

    end_step = _step(observation_end_step, "observation_end_step")
    normalised_arrivals = tuple(
        _step(value, f"arrival_steps[{index}]")
        for index, value in enumerate(arrivals)
    )
    if tuple(sorted(normalised_arrivals)) != normalised_arrivals:
        raise ValueError("arrival_steps must be nondecreasing")

    normalised_completions = []
    for index, (arrival, completion) in enumerate(
        zip(normalised_arrivals, completions)
    ):
        if completion is None:
            normalised_completions.append(None)
            continue
        completion = _step(completion, f"stored_steps[{index}]")
        if completion < arrival:
            raise ValueError("a storage completion cannot precede arrival")
        if completion > end_step:
            raise ValueError(
                "a storage completion cannot exceed observation_end_step"
            )
        normalised_completions.append(completion)

    completed_times = []
    censor_ages = []
    records = []
    arrived_count = 0
    completed_count = 0
    right_censored_count = 0
    not_yet_arrived_count = 0
    for index, (arrival, completion, label) in enumerate(
        zip(normalised_arrivals, normalised_completions, block_labels)
    ):
        record = {
            "manifest_index": index,
            "block_label": None if label is None else str(label),
            "arrival_step": arrival,
            "storage_completion_step": completion,
            "storage_flow_time": None,
            "status": None,
            "censor_step": None,
            "censor_age": None,
        }
        if completion is not None:
            flow_time = completion - arrival
            completed_count += 1
            arrived_count += 1
            completed_times.append(flow_time)
            record["storage_flow_time"] = flow_time
            record["status"] = "completed"
        elif arrival <= end_step:
            censor_age = end_step - arrival
            arrived_count += 1
            right_censored_count += 1
            censor_ages.append(censor_age)
            record["status"] = "right_censored"
            record["censor_step"] = end_step
            record["censor_age"] = censor_age
        else:
            not_yet_arrived_count += 1
            record["status"] = "not_yet_arrived"
        records.append(record)

    manifest_count = len(normalised_arrivals)
    unfinished_count = manifest_count - completed_count
    fully_observed = bool(
        manifest_count > 0 and completed_count == manifest_count
    )
    flow_values = np.asarray(completed_times, dtype=np.float64)
    if fully_observed:
        mean_flow = float(flow_values.mean())
        median_flow = float(np.quantile(flow_values, 0.50, method="linear"))
        p90_flow = float(np.quantile(flow_values, 0.90, method="linear"))
        p95_flow = float(np.quantile(flow_values, 0.95, method="linear"))
        max_flow = float(flow_values.max())
    else:
        mean_flow = np.nan
        median_flow = np.nan
        p90_flow = np.nan
        p95_flow = np.nan
        max_flow = np.nan

    return {
        "storage_flow_metric_contract": STORAGE_FLOW_METRIC_CONTRACT,
        "storage_flow_observation_end_step": end_step,
        "storage_flow_manifest_count": manifest_count,
        "storage_flow_arrived_count": arrived_count,
        "storage_flow_completed_count": completed_count,
        "storage_flow_right_censored_count": right_censored_count,
        "storage_flow_not_yet_arrived_count": not_yet_arrived_count,
        "storage_flow_unfinished_count": unfinished_count,
        "storage_flow_completion_rate": (
            float(completed_count / manifest_count)
            if manifest_count
            else np.nan
        ),
        "storage_flow_arrived_completion_rate": (
            float(completed_count / arrived_count)
            if arrived_count
            else np.nan
        ),
        "storage_flow_fully_observed": fully_observed,
        "mean_storage_flow_time": mean_flow,
        "median_storage_flow_time": median_flow,
        "p90_storage_flow_time": p90_flow,
        "p95_storage_flow_time": p95_flow,
        "max_storage_flow_time": max_flow,
        "completed_storage_flow_times": completed_times,
        "right_censored_storage_flow_ages": censor_ages,
        "storage_flow_records": records,
    }


def summarize_block_storage_flow(blocks, observation_end_step):
    """Apply :func:`summarize_storage_flow` to manifest-ordered blocks."""

    blocks = tuple(blocks)
    return summarize_storage_flow(
        [block.arrival_step for block in blocks],
        [block.stored_time_step for block in blocks],
        observation_end_step,
        labels=[getattr(block, "label", None) for block in blocks],
    )


def summarize_storage_flow_runs(runs):
    """Pool a collection of episode-level storage-flow summaries.

    Block-level distribution statistics remain undefined unless every episode
    has a fully observed manifest.  Completion/censor counts are always
    aggregated, so failed methods cannot look fast through survivor bias.
    """

    runs = tuple(runs)
    manifest_count = int(
        sum(item["storage_flow_manifest_count"] for item in runs)
    )
    arrived_count = int(
        sum(item["storage_flow_arrived_count"] for item in runs)
    )
    completed_count = int(
        sum(item["storage_flow_completed_count"] for item in runs)
    )
    right_censored_count = int(
        sum(item["storage_flow_right_censored_count"] for item in runs)
    )
    not_yet_arrived_count = int(
        sum(item["storage_flow_not_yet_arrived_count"] for item in runs)
    )
    unfinished_count = manifest_count - completed_count
    fully_observed_episodes = int(
        sum(bool(item["storage_flow_fully_observed"]) for item in runs)
    )
    fully_observed = bool(
        runs and fully_observed_episodes == len(runs)
    )
    completed_times = [
        float(value)
        for item in runs
        for value in item["completed_storage_flow_times"]
    ]
    values = np.asarray(completed_times, dtype=np.float64)
    if fully_observed and values.size:
        mean_flow = float(values.mean())
        median_flow = float(np.quantile(values, 0.50, method="linear"))
        p90_flow = float(np.quantile(values, 0.90, method="linear"))
        p95_flow = float(np.quantile(values, 0.95, method="linear"))
        max_flow = float(values.max())
    else:
        mean_flow = np.nan
        median_flow = np.nan
        p90_flow = np.nan
        p95_flow = np.nan
        max_flow = np.nan

    return {
        "storage_flow_metric_contract": STORAGE_FLOW_METRIC_CONTRACT,
        "storage_flow_episode_count": len(runs),
        "storage_flow_manifest_count": manifest_count,
        "storage_flow_arrived_count": arrived_count,
        "storage_flow_completed_count": completed_count,
        "storage_flow_right_censored_count": right_censored_count,
        "storage_flow_not_yet_arrived_count": not_yet_arrived_count,
        "storage_flow_unfinished_count": unfinished_count,
        "storage_flow_completion_rate": (
            float(completed_count / manifest_count)
            if manifest_count
            else np.nan
        ),
        "storage_flow_arrived_completion_rate": (
            float(completed_count / arrived_count)
            if arrived_count
            else np.nan
        ),
        "storage_flow_fully_observed": fully_observed,
        "fully_observed_storage_flow_episodes": fully_observed_episodes,
        "fully_observed_storage_flow_episode_rate": (
            float(fully_observed_episodes / len(runs)) if runs else np.nan
        ),
        "mean_storage_flow_time": mean_flow,
        "median_storage_flow_time": median_flow,
        "p90_storage_flow_time": p90_flow,
        "p95_storage_flow_time": p95_flow,
        "max_storage_flow_time": max_flow,
    }


def summarize_delivery_timing(deviations, target_window: float):
    """Summarize signed delivery deviations for one or more episodes.

    A deviation is ``delivery_time - target_time``: negative values are early
    and positive values are tardy. Tardiness and earliness are averaged across
    all deliveries, including zeros for deliveries in the opposite direction.
    """
    target_window = float(target_window)
    if target_window < 0:
        raise ValueError("target_window must be non-negative")

    values = np.asarray(deviations, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "delivery_count": 0,
            "mean_signed_deviation": np.nan,
            "mean_absolute_error": np.nan,
            "mean_tardiness": np.nan,
            "mean_earliness": np.nan,
            "within_target_window_rate": np.nan,
            "tardy_delivery_rate": np.nan,
            "mean_tardiness_when_tardy": np.nan,
            "p90_tardiness": np.nan,
            "p90_absolute_error": np.nan,
            "signed_deviation_std": np.nan,
            "absolute_error_std": np.nan,
        }

    absolute = np.abs(values)
    tardiness = np.maximum(values, 0.0)
    earliness = np.maximum(-values, 0.0)
    tardy_values = values[values > 0.0]
    return {
        "delivery_count": int(values.size),
        "mean_signed_deviation": float(values.mean()),
        "mean_absolute_error": float(absolute.mean()),
        "mean_tardiness": float(tardiness.mean()),
        "mean_earliness": float(earliness.mean()),
        "within_target_window_rate": float((absolute <= target_window).mean()),
        "tardy_delivery_rate": float((values > 0.0).mean()),
        "mean_tardiness_when_tardy": (
            float(tardy_values.mean()) if tardy_values.size else np.nan
        ),
        "p90_tardiness": float(np.quantile(tardiness, 0.90)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "signed_deviation_std": float(values.std(ddof=0)),
        "absolute_error_std": float(absolute.std(ddof=0)),
    }


__all__ = [
    "STORAGE_FLOW_METRIC_CONTRACT",
    "summarize_block_storage_flow",
    "summarize_delivery_timing",
    "summarize_storage_flow",
    "summarize_storage_flow_runs",
]
