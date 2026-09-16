"""Canonical mixed-domain manifests for Track-B generalization studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

from example.yard_geometry import geometry_metadata, make_shipyard_env


REGIME_MANIFEST_VERSION = "track_b_mixed_regime_manifest_v1"


def _digest(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value):
    """Return the canonical JSON-native representation of ``value``.

    Geometry metadata deliberately uses tuples internally.  A persisted
    manifest necessarily reads those arrays back as lists, so emitting the
    JSON-native form here keeps strict manifest validation stable across a
    write/read round trip.
    """

    return json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"))
    )


@dataclass(frozen=True)
class TrackBRegime:
    """One exact environment family used to sample paired episodes."""

    regime_id: str
    arrival_rate: float
    proc_mean: float
    grid_rows: int
    grid_cols: int
    exit_width: int | None
    number_blocks: int

    def __post_init__(self):
        object.__setattr__(self, "regime_id", str(self.regime_id))
        object.__setattr__(self, "arrival_rate", float(self.arrival_rate))
        object.__setattr__(self, "proc_mean", float(self.proc_mean))
        object.__setattr__(self, "grid_rows", int(self.grid_rows))
        object.__setattr__(self, "grid_cols", int(self.grid_cols))
        object.__setattr__(
            self,
            "exit_width",
            None if self.exit_width is None else int(self.exit_width),
        )
        object.__setattr__(self, "number_blocks", int(self.number_blocks))
        if not self.regime_id:
            raise ValueError("regime_id must be nonempty")
        if self.arrival_rate < 0.0:
            raise ValueError("arrival_rate must be nonnegative")
        if self.proc_mean <= 0.0:
            raise ValueError("proc_mean must be positive")
        if self.number_blocks <= 0:
            raise ValueError("number_blocks must be positive")
        # Geometry construction is the canonical feasibility validator.
        self.make_env()

    def make_env(self):
        return make_shipyard_env(
            arrival_rate=self.arrival_rate,
            proc_mean=self.proc_mean,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            exit_width=self.exit_width,
            number_blocks=self.number_blocks,
        )

    @property
    def regime_signature(self) -> str:
        # ``regime_id`` is a human-readable label, not part of the environment
        # family.  Excluding it makes the signature useful for rejecting the
        # same profile under two aliases.
        profile = self.to_dict()
        profile.pop("regime_id")
        return _digest(profile)[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    def provenance(self) -> dict:
        env = self.make_env()
        return {
            **self.to_dict(),
            "regime_signature": self.regime_signature,
            "geometry": geometry_metadata(
                env, requested_exit_width=self.exit_width
            ),
        }

    @classmethod
    def from_dict(cls, value: dict):
        expected = set(cls.__dataclass_fields__)
        unknown = set(value) - expected
        missing = expected - set(value)
        if unknown or missing:
            raise ValueError(
                f"noncanonical regime fields: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(**value)


def canonical_manifest(regimes: Iterable[TrackBRegime]) -> dict:
    regimes = tuple(regimes)
    if not regimes:
        raise ValueError("a regime manifest cannot be empty")
    ids = [item.regime_id for item in regimes]
    signatures = [item.regime_signature for item in regimes]
    if len(ids) != len(set(ids)):
        raise ValueError("regime ids must be unique")
    if len(signatures) != len(set(signatures)):
        raise ValueError("regime signatures must be unique")
    payload = _json_value(
        {
            "manifest_version": REGIME_MANIFEST_VERSION,
            "regimes": [item.provenance() for item in regimes],
        }
    )
    payload["manifest_sha256"] = _digest(payload)
    return payload


def manifest_regimes(payload: dict) -> tuple[TrackBRegime, ...]:
    if payload.get("manifest_version") != REGIME_MANIFEST_VERSION:
        raise ValueError("unsupported regime manifest version")
    raw = payload.get("regimes")
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest has no regimes")
    regimes = tuple(
        TrackBRegime.from_dict(
            {key: value for key, value in item.items() if key in TrackBRegime.__dataclass_fields__}
        )
        for item in raw
    )
    expected = canonical_manifest(regimes)
    if payload != expected:
        raise ValueError("regime manifest is noncanonical or was modified")
    return regimes


MIXED_TRAIN_V1 = (
    TrackBRegime("anchor_10x10_b40", 0.50, 50.0, 10, 10, None, 40),
    TrackBRegime("narrow_10x10_b40", 0.50, 50.0, 10, 10, 2, 40),
    TrackBRegime("compact_9x9_b36", 0.65, 60.0, 9, 9, 2, 36),
    TrackBRegime("large_11x11_b48", 0.40, 40.0, 11, 11, 3, 48),
)


MIXED_OOD_V1 = (
    TrackBRegime("composed_10x10_b44", 0.55, 55.0, 10, 10, 3, 44),
    TrackBRegime("rectangular_9x11_b40", 0.55, 55.0, 9, 11, 2, 40),
    TrackBRegime("heavy_11x11_b52", 0.60, 60.0, 11, 11, 2, 52),
)


STRESS_V1 = (
    # Compact profiles were selected on pressure/integrity only.  They are a
    # challenge panel, not one-factor causal ablations; the paired egress
    # protocol remains the geometry-effect experiment.
    TrackBRegime("load_5x5_b24", 0.80, 100.0, 5, 5, 1, 24),
    TrackBRegime("count_5x5_b52", 0.10, 100.0, 5, 5, 1, 52),
    TrackBRegime("egress_5x5_b40", 0.40, 100.0, 5, 5, 1, 40),
    TrackBRegime("compound_5x5_b64", 0.80, 100.0, 5, 5, 1, 64),
)


BUILTIN_REGIME_SUITES = {
    "mixed_train_v1": MIXED_TRAIN_V1,
    "mixed_ood_v1": MIXED_OOD_V1,
    "stress_v1": STRESS_V1,
}


__all__ = [
    "BUILTIN_REGIME_SUITES",
    "MIXED_OOD_V1",
    "MIXED_TRAIN_V1",
    "REGIME_MANIFEST_VERSION",
    "STRESS_V1",
    "TrackBRegime",
    "canonical_manifest",
    "manifest_regimes",
]
