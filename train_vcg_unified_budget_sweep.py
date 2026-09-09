#!/usr/bin/env python3
"""Run one arm of the unified-VCG developmental handling-budget sweep.

The four arms share model seed 14, initialization, training EpisodeInstances,
behavior RNGs, replay RNG, architecture, and the opened 85xxx validation grid.
They differ only in whether lambda is fixed at zero or updated against a
declared physical-rehandle budget.  Every arm trains for the same fixed
300-episode horizon and is evaluated only at its terminal checkpoint.
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


METHOD_VERSION = "vcg_unified_budget_sweep_v1"
TRAINING_PROTOCOL = "vcg_unified_budget_sweep_seed14_300ep_development_v1"
CHECKPOINT_FAMILY = "vcg_unified_budget_sweep_terminal_development_v1"
EPISODES = 300
TOTAL_BLOCKS = EPISODES // v23.BLOCK_EPISODES
MODEL_SEED = 14

LAMBDA_ZERO = "lambda0"
BUDGET_10 = "budget10"
BUDGET_8 = "budget8"
BUDGET_5 = "budget5"


class UnifiedBudgetSweepError(ValueError):
    """Raised when the developmental sweep contract is violated."""


@dataclass(frozen=True)
class SweepArm:
    name: str
    variant: str
    budget_per_100: float

    @property
    def constrained(self) -> bool:
        return self.variant == unified.VCG_HANDLING_CONSTRAINT

    @property
    def output_name(self) -> str:
        return self.name.replace("budget", "budget-")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "output_name": self.output_name,
            "variant": self.variant,
            "constrained": self.constrained,
            "handling_budget_per_100": (
                self.budget_per_100 if self.constrained else None
            ),
            "monitor_reference_budget_per_100": (
                self.budget_per_100 if not self.constrained else None
            ),
        }


ARMS = {
    LAMBDA_ZERO: SweepArm(LAMBDA_ZERO, unified.VCG, 10.0),
    BUDGET_10: SweepArm(BUDGET_10, unified.VCG_HANDLING_CONSTRAINT, 10.0),
    BUDGET_8: SweepArm(BUDGET_8, unified.VCG_HANDLING_CONSTRAINT, 8.0),
    BUDGET_5: SweepArm(BUDGET_5, unified.VCG_HANDLING_CONSTRAINT, 5.0),
}
ARM_ORDER = (LAMBDA_ZERO, BUDGET_10, BUDGET_8, BUDGET_5)


def arm_for_name(name: str) -> SweepArm:
    if name not in ARMS:
        raise UnifiedBudgetSweepError(
            f"arm must be one of {ARM_ORDER!r}; received {name!r}"
        )
    return ARMS[name]


def validate_sweep() -> None:
    if EPISODES % v23.BLOCK_EPISODES:
        raise UnifiedBudgetSweepError("training horizon must end on a dual block")
    if TOTAL_BLOCKS != 30:
        raise UnifiedBudgetSweepError("the developmental sweep requires 30 blocks")
    if tuple(ARMS) != ARM_ORDER:
        raise UnifiedBudgetSweepError("sweep arm order drifted")
    if tuple(ARMS[name].budget_per_100 for name in ARM_ORDER[1:]) != (
        10.0,
        8.0,
        5.0,
    ):
        raise UnifiedBudgetSweepError("handling-budget grid drifted")
    if ARMS[LAMBDA_ZERO].variant != unified.VCG:
        raise UnifiedBudgetSweepError("lambda-zero arm must use unified VCG")


@contextmanager
def activated_sweep_protocol():
    """Install the 300-episode protocol and restore imported modules."""

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
        unified.CONTRACT_SCHEMA_VERSION = 2
        unified.CHECKPOINT_SCHEMA_VERSION = 2
        yield
    finally:
        for name, value in unified_snapshot.items():
            setattr(unified, name, value)
        for name, value in v23_snapshot.items():
            setattr(v23, name, value)


def _unified_args(args: argparse.Namespace, arm: SweepArm) -> argparse.Namespace:
    forwarded = [
        "--output-dir",
        str(Path(args.output_dir)),
        "--variant",
        arm.variant,
        "--episodes",
        str(EPISODES),
        "--rehandle-budget-per-100",
        str(arm.budget_per_100),
        "--device",
        str(args.device),
    ]
    if args.contract_only:
        forwarded.append("--contract-only")
    return unified.build_parser().parse_args(forwarded)


def run(args: argparse.Namespace) -> dict:
    validate_sweep()
    arm = arm_for_name(args.arm)
    with activated_sweep_protocol():
        result = dict(unified.run_unified_training(_unified_args(args, arm)))
    result.update(
        sweep_protocol=TRAINING_PROTOCOL,
        sweep_arm=arm.to_dict(),
        fixed_training_horizon=EPISODES,
        same_seed_and_rng_streams_across_arms=True,
        developmental_point_estimate_only=True,
        post_hoc_after_initial_unified_pair=True,
        final_86xxx_panel_opened=False,
    )
    atomic_io._atomic_json(result, Path(args.output_dir) / "training-summary.json")
    return result


def plan() -> dict:
    validate_sweep()
    return {
        "protocol": TRAINING_PROTOCOL,
        "model_seed": MODEL_SEED,
        "episodes_per_arm": EPISODES,
        "arms": tuple(ARMS[name].to_dict() for name in ARM_ORDER),
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
        "new_final_panel_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one arm of the unified-VCG developmental budget sweep"
    )
    parser.add_argument("--arm", choices=ARM_ORDER, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contract-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


validate_sweep()


if __name__ == "__main__":
    main()

