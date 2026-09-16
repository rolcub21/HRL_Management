#!/usr/bin/env python3
"""Run one fresh paired-seed replicate of unified VCG.

This is a thin profile wrapper around :mod:`train_vcg_unified`.  It changes
only the model seed and the three training RNG namespaces.  Within a model
seed, the unconstrained and constrained arms use the same initialization,
EpisodeInstances, behavior RNGs, replay RNG, and validation grid.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional, Sequence

import train_vcg_unified as unified


MODEL_SEEDS = (15, 16, 17)
EPISODES = 200


class UnifiedSeedStabilityError(ValueError):
    """Raised when a paired stability seed is outside the frozen grid."""


@dataclass(frozen=True)
class PairedSeedProfile:
    model_seed: int
    train_seed_base: int
    behavior_rng_base: int
    replay_rng_seed: int

    @property
    def train_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.train_seed_base, self.train_seed_base + EPISODES))

    @property
    def behavior_rng_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.behavior_rng_base, self.behavior_rng_base + EPISODES))

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "train_seed_range": (self.train_seed_base, self.train_seed_base + EPISODES - 1),
            "behavior_rng_range": (
                self.behavior_rng_base,
                self.behavior_rng_base + EPISODES - 1,
            ),
            "validation_policy_rng_range": (
                unified.VALIDATION_POLICY_RNG_BASE,
                unified.VALIDATION_POLICY_RNG_BASE + 47,
            ),
        }


SEED_PROFILES = {
    15: PairedSeedProfile(15, 61_005_000, 610_005_000, 610_105_010),
    16: PairedSeedProfile(16, 61_006_000, 610_006_000, 610_106_010),
    17: PairedSeedProfile(17, 61_007_000, 610_007_000, 610_107_010),
}


def _validate_profiles() -> None:
    if tuple(sorted(SEED_PROFILES)) != MODEL_SEEDS:
        raise UnifiedSeedStabilityError("paired seed grid drifted")
    # Include seed 14 so the fresh stability streams cannot overlap the
    # completed development run.
    namespaces: list[set[int]] = [
        set(range(61_004_000, 61_004_200)),
        set(range(610_004_000, 610_004_200)),
        {610_104_010},
        set(range(85_000, 85_012)),
        set(range(621_000_000, 621_000_048)),
    ]
    for seed in MODEL_SEEDS:
        profile = SEED_PROFILES[seed]
        if profile.model_seed != seed:
            raise UnifiedSeedStabilityError("model seed/profile mismatch")
        namespaces.extend(
            (
                set(profile.train_seeds),
                set(profile.behavior_rng_seeds),
                {profile.replay_rng_seed},
            )
        )
    for index, left in enumerate(namespaces):
        for right in namespaces[index + 1 :]:
            if left & right:
                raise UnifiedSeedStabilityError("paired seed/RNG namespaces overlap")


def profile_for_seed(model_seed: int) -> PairedSeedProfile:
    if type(model_seed) is not int or model_seed not in SEED_PROFILES:
        raise UnifiedSeedStabilityError(f"model seed must be one of {MODEL_SEEDS}")
    return SEED_PROFILES[model_seed]


@contextmanager
def activated_paired_profile(profile: PairedSeedProfile):
    """Patch only the unified trainer's seed constants, then restore them."""

    if profile != profile_for_seed(profile.model_seed):
        raise UnifiedSeedStabilityError("profile is not part of the frozen grid")
    names = (
        "MODEL_SEED",
        "TRAIN_SEED_BASE",
        "TRAINING_POLICY_RNG_BASE",
        "REPLAY_RNG_SEED",
    )
    snapshot = {name: getattr(unified, name) for name in names}
    try:
        unified.MODEL_SEED = profile.model_seed
        unified.TRAIN_SEED_BASE = profile.train_seed_base
        unified.TRAINING_POLICY_RNG_BASE = profile.behavior_rng_base
        unified.REPLAY_RNG_SEED = profile.replay_rng_seed
        yield
    finally:
        for name, value in snapshot.items():
            setattr(unified, name, value)


def _unified_args(args: argparse.Namespace) -> argparse.Namespace:
    profile = profile_for_seed(args.model_seed)
    forwarded = [
        "--output-dir",
        str(Path(args.output_dir)),
        "--variant",
        str(args.variant),
        "--device",
        str(args.device),
    ]
    if args.contract_only:
        forwarded.append("--contract-only")
    with activated_paired_profile(profile):
        parsed = unified.build_parser().parse_args(forwarded)
    return parsed


def run(args: argparse.Namespace) -> dict:
    profile = profile_for_seed(args.model_seed)
    with activated_paired_profile(profile):
        summary = unified.run_unified_training(_unified_args(args))
    result = dict(summary)
    result["stability_model_seed"] = profile.model_seed
    result["paired_seed_profile"] = profile.to_dict()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one unified-VCG paired seed-stability arm"
    )
    parser.add_argument("--model-seed", type=int, choices=MODEL_SEEDS, required=True)
    parser.add_argument("--variant", choices=unified.VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contract-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    _validate_profiles()
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


_validate_profiles()


if __name__ == "__main__":
    main()

