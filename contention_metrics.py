"""Canonical physical-storage relocation metrics and invariants.

The simulator's ``relocated_block`` event records a completed storage-to-
storage move.  It does not, by itself, establish why the move occurred.  This
module keeps the common physical burden separate from controller-specific
mechanism labels and fails closed when a reported decomposition is incomplete.
"""

from __future__ import annotations

from typing import Mapping


CONTENTION_METRIC_SCHEMA_VERSION = (
    "physical_storage_relocation_decomposition_v1"
)

CANONICAL_CONTENTION_FIELDS = (
    "contention_metric_schema_version",
    "physical_storage_relocations",
    "target_bound_obstruction_clearances",
    "standalone_reconfigurations",
    "standalone_with_direct_delivery_available",
    "standalone_without_direct_delivery_available",
    "directly_deliverable_self_reconfigurations",
)


def _count(name: str, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer count")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer count") from exc
    try:
        exact = float(value) == float(number)
    except (TypeError, ValueError, OverflowError):
        exact = False
    if number < 0 or not exact:
        raise ValueError(f"{name} must be a nonnegative integer count")
    return number


def contention_metric_record(
    *,
    physical_storage_relocations: int,
    target_bound_obstruction_clearances: int,
    standalone_reconfigurations: int,
    standalone_with_direct_delivery_available: int,
    standalone_without_direct_delivery_available: int,
    directly_deliverable_self_reconfigurations: int,
) -> dict:
    """Build and validate one lossless mechanism decomposition.

    ``directly_deliverable_self_reconfigurations`` is a diagnostic subset of
    ``standalone_with_direct_delivery_available``; it is not another additive
    component of the physical relocation total.
    """

    record = {
        "contention_metric_schema_version": CONTENTION_METRIC_SCHEMA_VERSION,
        "physical_storage_relocations": _count(
            "physical_storage_relocations", physical_storage_relocations
        ),
        "target_bound_obstruction_clearances": _count(
            "target_bound_obstruction_clearances",
            target_bound_obstruction_clearances,
        ),
        "standalone_reconfigurations": _count(
            "standalone_reconfigurations", standalone_reconfigurations
        ),
        "standalone_with_direct_delivery_available": _count(
            "standalone_with_direct_delivery_available",
            standalone_with_direct_delivery_available,
        ),
        "standalone_without_direct_delivery_available": _count(
            "standalone_without_direct_delivery_available",
            standalone_without_direct_delivery_available,
        ),
        "directly_deliverable_self_reconfigurations": _count(
            "directly_deliverable_self_reconfigurations",
            directly_deliverable_self_reconfigurations,
        ),
    }
    validate_contention_metric_record(record)
    return record


def validate_contention_metric_record(
    record: Mapping,
    *,
    require_legacy_alias: bool = False,
) -> dict:
    """Return normalized counts or reject a lossy/ambiguous decomposition."""

    if record.get("contention_metric_schema_version") != (
        CONTENTION_METRIC_SCHEMA_VERSION
    ):
        raise ValueError(
            "contention metric schema must be "
            f"{CONTENTION_METRIC_SCHEMA_VERSION!r}"
        )
    normalized = {
        name: _count(name, record[name])
        for name in CANONICAL_CONTENTION_FIELDS
        if name != "contention_metric_schema_version"
    }
    physical = normalized["physical_storage_relocations"]
    target_bound = normalized["target_bound_obstruction_clearances"]
    standalone = normalized["standalone_reconfigurations"]
    with_delivery = normalized[
        "standalone_with_direct_delivery_available"
    ]
    without_delivery = normalized[
        "standalone_without_direct_delivery_available"
    ]
    directly_deliverable_self = normalized[
        "directly_deliverable_self_reconfigurations"
    ]
    if physical != target_bound + standalone:
        raise ValueError(
            "physical storage relocations must equal target-bound obstruction "
            "clearances plus standalone reconfigurations"
        )
    if standalone != with_delivery + without_delivery:
        raise ValueError(
            "standalone reconfigurations must be partitioned by direct-delivery "
            "availability"
        )
    if directly_deliverable_self > with_delivery:
        raise ValueError(
            "directly-deliverable self reconfigurations must be a subset of "
            "standalone reconfigurations with direct delivery available"
        )
    if require_legacy_alias:
        if _count("relocations", record["relocations"]) != physical:
            raise ValueError(
                "legacy relocations must be an exact alias of physical storage "
                "relocations"
            )
        if _count("obstructive_moves", record["obstructive_moves"]) != target_bound:
            raise ValueError(
                "legacy obstructive_moves must be an exact alias of target-bound "
                "obstruction clearances"
            )
    return {
        "contention_metric_schema_version": CONTENTION_METRIC_SCHEMA_VERSION,
        **normalized,
    }
