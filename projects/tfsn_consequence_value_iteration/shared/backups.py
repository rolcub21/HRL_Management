"""Finite-horizon road-map backups shared across domains."""

from __future__ import annotations

import numpy as np


def _check_edges(node_count, source, destination, reliability, gain):
    source = np.asarray(source, np.int64).reshape(-1)
    destination = np.asarray(destination, np.int64).reshape(-1)
    reliability = np.asarray(reliability, np.float32).reshape(-1)
    gain = np.asarray(gain, np.float32).reshape(-1)
    if not (
        len(source) == len(destination) == len(reliability) == len(gain)
    ):
        raise ValueError("road-map edge arrays are not aligned")
    if np.any(source < 0) or np.any(source >= int(node_count)):
        raise ValueError("road-map source index is outside the graph")
    if np.any(destination < 0) or np.any(destination >= int(node_count)):
        raise ValueError("road-map destination index is outside the graph")
    if np.any(reliability < 0.0) or np.any(reliability > 1.0):
        raise ValueError("edge reliability must be in [0, 1]")
    return source, destination, reliability, gain


def bounded_search_backup(
    node_count,
    source,
    destination,
    reliability,
    path_gain,
    horizon,
):
    """R28 search backup: consumable gain and failure stays at source."""
    source, destination, reliability, gain = _check_edges(
        node_count, source, destination, reliability, path_gain
    )
    gain = np.clip(gain, 0.0, 1.0)
    values = np.zeros((int(horizon) + 1, int(node_count)), np.float32)
    for remaining in range(1, int(horizon) + 1):
        previous = values[remaining - 1]
        success = gain + (1.0 - gain) * previous[destination]
        edge_value = (
            reliability * success
            + (1.0 - reliability) * previous[source]
        )
        row = previous.copy()
        np.maximum.at(row, source, edge_value)
        values[remaining] = np.clip(row, 0.0, 1.0)
    return values


def additive_field_backup(
    node_count,
    source,
    destination,
    reliability,
    destination_field,
    horizon,
):
    """Persistent return/supplied-goal backup with failure at the source."""
    source, destination, reliability, _unused = _check_edges(
        node_count,
        source,
        destination,
        reliability,
        np.zeros(len(np.asarray(source).reshape(-1)), np.float32),
    )
    field = np.asarray(destination_field, np.float32).reshape(int(node_count))
    values = np.zeros((int(horizon) + 1, int(node_count)), np.float32)
    for remaining in range(1, int(horizon) + 1):
        previous = values[remaining - 1]
        success = field[destination] + previous[destination]
        edge_value = (
            reliability * success
            + (1.0 - reliability) * previous[source]
        )
        row = previous.copy()
        np.maximum.at(row, source, edge_value)
        values[remaining] = row
    return values


def ensemble_reduce(member_values, mode):
    values = np.asarray(member_values, np.float32)
    if values.ndim < 2:
        raise ValueError("ensemble values need a member axis")
    if mode == "mean":
        return values.mean(axis=-1)
    if mode == "pessimistic":
        return values.min(axis=-1)
    raise ValueError("ensemble reduction must be mean or pessimistic")

