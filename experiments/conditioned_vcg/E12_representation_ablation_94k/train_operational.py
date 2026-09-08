#!/usr/bin/env python3
"""Matched operational-Q training wrapper for one E12 representation arm."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from typing import Optional, Sequence
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.conditioned_vcg.E12_representation_ablation_94k import program
from methods.conditioned_vcg.representation_ablation import (
    REPRESENTATION_VARIANTS,
    agent_class_for_variant,
)
import train_viability_graph_smdp_proper as trainer


TRAINING_PROTOCOL = "vcg_e12_matched_operational_representation_training_v1"


def output_path(output_dir: Path, variant: str, model_seed: int) -> Path:
    return output_dir / "training" / "operational" / variant / f"seed-{model_seed}"


def run(
    output_dir: Path,
    *,
    variant: str,
    model_seed: int,
    device: str,
    stop_after_episode: Optional[int],
) -> dict:
    contract, _manifest = program.authenticate(output_dir)
    if variant not in REPRESENTATION_VARIANTS:
        raise program.E12Error(f"unknown representation: {variant}")
    if model_seed not in program.MODEL_SEEDS:
        raise program.E12Error("model seed must be 0, 1, or 2")
    agent_class = agent_class_for_variant(variant)
    destination = output_path(output_dir, variant, model_seed)
    latest = destination / "latest.pth"
    arguments = [
        "--output-dir", str(destination),
        "--model-seed", str(model_seed),
        "--total-episodes", str(program.OPERATIONAL_EPISODES),
        "--train-instance-seed-base",
        str(
            program.OPERATIONAL_TRAIN_SEED_BASE
            + model_seed * program.OPERATIONAL_MODEL_SEED_STRIDE
        ),
        "--validation-seeds",
        *[str(seed) for seed in program.OPERATIONAL_VALIDATION_SEEDS],
        "--device", device,
        "--max-steps", str(program.MAX_STEPS),
    ]
    if latest.is_file():
        arguments.extend(("--resume", str(latest)))
    if stop_after_episode is not None:
        arguments.extend(("--stop-after-episode", str(stop_after_episode)))

    original_resume_contract = trainer._resume_contract
    original_result_payload = trainer._result_payload

    def e12_resume_contract(*args, **kwargs):
        result = original_resume_contract(*args, **kwargs)
        result.update(
            {
                "training_protocol": TRAINING_PROTOCOL,
                "e12_contract_sha256": contract["contract_sha256"],
                "representation_variant": variant,
                "representation_only_difference": True,
                "parameter_matched": True,
            }
        )
        return result

    def e12_result_payload(*args, **kwargs):
        result = original_result_payload(*args, **kwargs)
        result.update(
            {
                "training_protocol": TRAINING_PROTOCOL,
                "canonical_500_episode_replication_complete": False,
                "e12_contract_sha256": contract["contract_sha256"],
                "representation_variant": variant,
                "representation_ablation_only": True,
                "development_pilot_opened": False,
                "e11_confirmation_opened": False,
            }
        )
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(trainer, "TRAINING_PROTOCOL", TRAINING_PROTOCOL))
        stack.enter_context(
            patch.object(trainer, "ViabilityGraphHierarchyAgent", agent_class)
        )
        stack.enter_context(
            patch.object(trainer, "_resume_contract", e12_resume_contract)
        )
        stack.enter_context(
            patch.object(trainer, "_result_payload", e12_result_payload)
        )
        result = trainer.main(arguments)
    return {
        "status": result["status"],
        "variant": variant,
        "model_seed": model_seed,
        "completed_training_episodes": result["completed_training_episodes"],
        "best_checkpoint": result["best_checkpoint"],
        "latest_checkpoint": result["latest_checkpoint"],
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=program.DEFAULT_OUTPUT)
    parser.add_argument("--variant", choices=REPRESENTATION_VARIANTS, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stop-after-episode", type=int)
    args = parser.parse_args(argv)
    result = run(
        args.output_dir.resolve(),
        variant=args.variant,
        model_seed=args.model_seed,
        device=args.device,
        stop_after_episode=args.stop_after_episode,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
