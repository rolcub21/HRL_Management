#!/usr/bin/env python3
"""Run one fixed-handling-weight unified-VCG development arm.

Each arm shares seed 14, the 300 training EpisodeInstances, behavior/replay
RNGs, architecture, and terminal 85xxx validation grid with the completed
lambda-zero parent.  Lambda is zero for the common 20-episode critic warm-up
and then stays fixed for the rest of training and evaluation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Sequence

import train_vcg_constrained_v2_1 as atomic_io
import train_vcg_constrained_v2_3 as v23
import train_vcg_unified as unified


METHOD_VERSION = "vcg_unified_fixed_lambda_sweep_v1"
TRAINING_PROTOCOL = "vcg_unified_fixed_lambda_sweep_seed14_300ep_development_v1"
CHECKPOINT_FAMILY = "vcg_unified_fixed_lambda_terminal_development_v1"
EPISODES = 300
TOTAL_BLOCKS = EPISODES // v23.BLOCK_EPISODES
MODEL_SEED = 14
FIXED_LAMBDAS = (0.05, 0.10, 0.20, 0.30)


class FixedLambdaSweepError(ValueError):
    """Raised when the fixed-weight sweep contract is violated."""


def lambda_key(value: float) -> str:
    fixed = validate_lambda(value)
    return f"lambda-{fixed:.2f}"


def validate_lambda(value: float) -> float:
    fixed = float(value)
    if fixed not in FIXED_LAMBDAS:
        raise FixedLambdaSweepError(
            f"fixed lambda must be one of {FIXED_LAMBDAS!r}; received {value!r}"
        )
    return fixed


@dataclass(frozen=True)
class FixedLambdaArm:
    fixed_lambda: float

    def __post_init__(self) -> None:
        validate_lambda(self.fixed_lambda)

    @property
    def name(self) -> str:
        return lambda_key(self.fixed_lambda)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "variant": unified.VCG_FIXED_HANDLING_WEIGHT,
            "fixed_lambda_after_warmup": float(self.fixed_lambda),
            "warmup_episodes_at_lambda_zero": v23.WARMUP_EPISODES,
            "dual_updates_applied": False,
        }


ARMS = tuple(FixedLambdaArm(value) for value in FIXED_LAMBDAS)


@contextmanager
def activated_fixed_lambda_protocol():
    """Install the 300-episode fixed-weight protocol, then restore modules."""

    v23_names = ("TOTAL_EPISODES", "TOTAL_BLOCKS")
    unified_names = (
        "METHOD_VERSION",
        "TRAINING_PROTOCOL",
        "CHECKPOINT_FAMILY",
        "CONTRACT_SCHEMA_VERSION",
        "CHECKPOINT_SCHEMA_VERSION",
    )
    v23_snapshot = {name: getattr(v23, name) for name in v23_names}
    unified_snapshot = {name: getattr(unified, name) for name in unified_names}
    try:
        v23.TOTAL_EPISODES = EPISODES
        v23.TOTAL_BLOCKS = TOTAL_BLOCKS
        unified.METHOD_VERSION = METHOD_VERSION
        unified.TRAINING_PROTOCOL = TRAINING_PROTOCOL
        unified.CHECKPOINT_FAMILY = CHECKPOINT_FAMILY
        unified.CONTRACT_SCHEMA_VERSION = 3
        unified.CHECKPOINT_SCHEMA_VERSION = 3
        yield
    finally:
        for name, value in unified_snapshot.items():
            setattr(unified, name, value)
        for name, value in v23_snapshot.items():
            setattr(v23, name, value)


def _unified_args(args: argparse.Namespace, arm: FixedLambdaArm) -> argparse.Namespace:
    forwarded = [
        "--output-dir",
        str(Path(args.output_dir)),
        "--variant",
        unified.VCG_FIXED_HANDLING_WEIGHT,
        "--fixed-lambda",
        str(arm.fixed_lambda),
        "--episodes",
        str(EPISODES),
        # This is monitor-only for fixed-weight arms and has no policy effect.
        "--rehandle-budget-per-100",
        "0",
        "--device",
        str(args.device),
    ]
    if args.contract_only:
        forwarded.append("--contract-only")
    return unified.build_parser().parse_args(forwarded)


def run(args: argparse.Namespace) -> dict:
    arm = FixedLambdaArm(validate_lambda(args.fixed_lambda))
    with activated_fixed_lambda_protocol():
        result = dict(unified.run_unified_training(_unified_args(args, arm)))
    result.update(
        fixed_lambda_sweep_protocol=TRAINING_PROTOCOL,
        fixed_lambda_sweep_arm=arm.to_dict(),
        fixed_training_horizon=EPISODES,
        lambda_zero_parent_reused=True,
        developmental_point_estimate_only=True,
        post_hoc_after_dual_budget_sweep=True,
        final_86xxx_panel_opened=False,
    )
    atomic_io._atomic_json(result, Path(args.output_dir) / "training-summary.json")
    return result


def plan() -> dict:
    return {
        "protocol": TRAINING_PROTOCOL,
        "model_seed": MODEL_SEED,
        "episodes_per_new_arm": EPISODES,
        "lambda_zero_parent": (
            "results/vcg-unified-budget-sweep-seed14-300ep-85k-development/lambda0"
        ),
        "new_arms": tuple(arm.to_dict() for arm in ARMS),
        "lambda_schedule": (
            "episodes 1..20: lambda=0; episodes 21..300 and terminal "
            "validation: declared fixed lambda"
        ),
        "training_instance_seed_range": (
            unified.TRAIN_SEED_BASE,
            unified.TRAIN_SEED_BASE + EPISODES - 1,
        ),
        "behavior_rng_range": (
            unified.TRAINING_POLICY_RNG_BASE,
            unified.TRAINING_POLICY_RNG_BASE + EPISODES - 1,
        ),
        "replay_rng_seed": unified.REPLAY_RNG_SEED,
        "validation_instance_seeds": tuple(v23.DEFAULT_VALIDATION_SEEDS),
        "validation_action_rng_range": (
            unified.VALIDATION_POLICY_RNG_BASE,
            unified.VALIDATION_POLICY_RNG_BASE + 47,
        ),
        "terminal_evaluation_only": True,
        "budget_or_kkt_claim": False,
        "post_hoc_after_dual_budget_sweep": True,
        "new_final_panel_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one unified-VCG fixed-lambda development arm"
    )
    parser.add_argument("--fixed-lambda", type=float, choices=FIXED_LAMBDAS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contract-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
