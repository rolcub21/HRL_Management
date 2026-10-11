#!/usr/bin/env python3
"""Shared immutable program and instance contract for E12."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E11_distribution_shift_93k import run as e11
from methods.conditioned_vcg.representation_ablation import (
    FULL_RELATIONAL_SUCCESSOR,
    NONRELATIONAL_SUCCESSOR,
    RELATIONAL_CURRENT_CANDIDATE,
    REPRESENTATION_VARIANTS,
)


PROTOCOL = "vcg_conditioned_e12_representation_ablation_94k_v1"
SCHEMA_VERSION = 1
CONTRACT_NAME = "e12-contract.json"
MANIFEST_NAME = "development-instance-manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e12-representation-ablation-94k"

MODEL_SEEDS = (0, 1, 2)
PILOT_MODEL_SEEDS = (0,)
DEPLOYMENT_LAMBDAS = (0.0, 0.05, 0.10, 0.20)
DEVELOPMENT_INSTANCE_SEEDS = tuple(range(94_000, 94_003))
DEVELOPMENT_REGIMES = (
    "reference",
    "arrival_spread",
    "dwell_long",
    "mirrored_entry",
    "combined_shift",
)
OPERATIONAL_EPISODES = 500
OPERATIONAL_TRAIN_SEED_BASE = 94_000_000
OPERATIONAL_MODEL_SEED_STRIDE = 1_000_000
OPERATIONAL_VALIDATION_SEEDS = tuple(range(94_900, 94_920))
HANDLING_ROUNDS = 8
HANDLING_EPISODES_PER_ROUND = 50
HANDLING_EPISODES = HANDLING_ROUNDS * HANDLING_EPISODES_PER_ROUND
HANDLING_TRAIN_SEED_BASE = 104_000_000
HANDLING_MODEL_SEED_STRIDE = 1_000_000
HANDLING_ROUND_SEED_STRIDE = 10_000
HANDLING_VALIDATION_EPISODES = 10
HANDLING_FIT_EPOCHS = 8
HANDLING_FIT_BATCH_SIZE = 256
HANDLING_FIT_LEARNING_RATE = 5.0e-4
MAX_STEPS = 2_000


class E12Error(RuntimeError):
    pass


def canonical_bytes(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: Mapping, *, hash_field: Optional[str] = None) -> str:
    payload = dict(value)
    if hash_field is not None:
        payload.pop(hash_field, None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = digest(result)
    return result


def sha256(path: Path) -> str:
    path = Path(path).resolve()
    if not path.is_file() or path.is_symlink():
        raise E12Error(f"missing regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path, *, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise E12Error(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E12Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E12Error(f"{label} must contain an object")
    return value


def _source_contract() -> dict:
    relative_paths = (
        "methods/conditioned_vcg/representation_ablation.py",
        "experiments/conditioned_vcg/E12_representation_ablation_94k/program.py",
        "experiments/conditioned_vcg/E12_representation_ablation_94k/train_operational.py",
        "experiments/conditioned_vcg/E12_representation_ablation_94k/train_handling.py",
        "experiments/conditioned_vcg/E12_representation_ablation_94k/evaluate.py",
        "train_viability_graph_smdp_proper.py",
        "train_vcg_v11_conditioned_handling_iterative.py",
        "vcg_v11_conditioned_handling.py",
        "viability_graph_hierarchy.py",
        "PSLAP/yard_graph.py",
        "experiments/conditioned_vcg/E11_distribution_shift_93k/run.py",
    )
    return {
        "source_sha256": {
            path: sha256(PROJECT_ROOT / path) for path in relative_paths
        }
    }


def contract() -> dict:
    e11_manifest = e11.DEFAULT_OUTPUT / e11.MANIFEST_NAME
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scientific_question": (
            "do_relational_aggregation_and_counterfactual_successor_"
            "representation_reduce_degradation_under_distribution_shift"
        ),
        "representations": list(REPRESENTATION_VARIANTS),
        "full_arm": FULL_RELATIONAL_SUCCESSOR,
        "ablations": [NONRELATIONAL_SUCCESSOR, RELATIONAL_CURRENT_CANDIDATE],
        "model_seeds": list(MODEL_SEEDS),
        "pilot_model_seeds": list(PILOT_MODEL_SEEDS),
        "deployment_lambdas": list(DEPLOYMENT_LAMBDAS),
        "development_instance_seeds": list(DEVELOPMENT_INSTANCE_SEEDS),
        "development_regimes": list(DEVELOPMENT_REGIMES),
        "operational_training": {
            "episodes": OPERATIONAL_EPISODES,
            "train_seed_base": OPERATIONAL_TRAIN_SEED_BASE,
            "model_seed_stride": OPERATIONAL_MODEL_SEED_STRIDE,
            "validation_seeds": list(OPERATIONAL_VALIDATION_SEEDS),
            "original_recipe_otherwise_unchanged": True,
        },
        "handling_training": {
            "rounds": HANDLING_ROUNDS,
            "episodes_per_round": HANDLING_EPISODES_PER_ROUND,
            "episodes": HANDLING_EPISODES,
            "train_seed_base": HANDLING_TRAIN_SEED_BASE,
            "model_seed_stride": HANDLING_MODEL_SEED_STRIDE,
            "round_seed_stride": HANDLING_ROUND_SEED_STRIDE,
            "validation_episodes_per_round": HANDLING_VALIDATION_EPISODES,
            "fit_epochs_per_round": HANDLING_FIT_EPOCHS,
            "fit_batch_size": HANDLING_FIT_BATCH_SIZE,
            "fit_learning_rate": HANDLING_FIT_LEARNING_RATE,
            "random_initialization_no_warm_start": True,
            "operational_critic_frozen": True,
        },
        "controlled_constants": [
            "exact_checker",
            "candidate_generation",
            "liveness_guard",
            "action_interface",
            "temporal_mode_aggregation",
            "Bellman_target",
            "training_instances",
            "validation_instances",
            "training_budget",
            "deployment_preferences",
            "evaluation_instances",
            "parameter_count",
        ],
        "nonrelational_definition": (
            "same_three_encoder_layers_and_parameters_with_empty_edge_set"
        ),
        "successor_free_definition": (
            "current_graph_repeated_in_successor_slot_zero_difference_equal_width"
        ),
        "pilot_is_development_only": True,
        "pilot_does_not_open_e11_93k_instances": True,
        "full_confirmation_reuses_frozen_e11_93k_episode_instances": True,
        "e11_manifest_sha256": sha256(e11_manifest),
        "primary_shift_estimand": (
            "loss_shift_minus_reference_ablation_minus_full_on_matched_coordinates"
        ),
        "report_absolute_performance_alongside_shift_degradation": True,
        "complete_case_filtering_allowed": False,
        "max_steps": MAX_STEPS,
        **_source_contract(),
    }
    return with_hash(semantic, "contract_sha256")


def instance_path(output_dir: Path, regime_id: str, seed: int) -> Path:
    return output_dir / "development-instances" / regime_id / f"seed-{seed}.json"


def prepare(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    expected = contract()
    contract_path = output_dir / CONTRACT_NAME
    if contract_path.is_file():
        observed = load_json(contract_path, label="E12 contract")
        if observed.get("contract_sha256") != digest(
            observed, hash_field="contract_sha256"
        ) or observed != expected:
            raise E12Error("E12 contract, sources, or E11 manifest changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise E12Error("nonempty E12 output has no contract")
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(contract_path, expected)

    records = []
    for regime_id in DEVELOPMENT_REGIMES:
        regime = e11.REGIME_BY_ID[regime_id]
        for index, seed in enumerate(DEVELOPMENT_INSTANCE_SEEDS):
            path = instance_path(output_dir, regime_id, seed)
            expected_instance = e11._instance(regime, seed)
            if path.is_file():
                observed_instance = EpisodeInstance.from_json(
                    path.read_text(encoding="utf-8")
                )
                if observed_instance != expected_instance:
                    raise E12Error(f"development instance changed: {regime_id}/{seed}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(expected_instance.to_json(), encoding="utf-8")
                observed_instance = EpisodeInstance.from_json(
                    path.read_text(encoding="utf-8")
                )
            records.append(
                {
                    "regime_id": regime_id,
                    "seed": int(seed),
                    "instance_index": index,
                    "relative_path": str(path.relative_to(output_dir)),
                    "raw_sha256": sha256(path),
                    "episode_instance_id": observed_instance.instance_id,
                    "schedule_id": observed_instance.schedule_id,
                    "workload": e11._workload_diagnostics(observed_instance),
                }
            )
    manifest = with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "contract_sha256": expected["contract_sha256"],
            "development_only": True,
            "records": records,
        },
        "manifest_sha256",
    )
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.is_file():
        observed = load_json(manifest_path, label="E12 development manifest")
        if observed != manifest:
            raise E12Error("E12 development instance manifest changed")
    else:
        atomic_json(manifest_path, manifest)
    return {
        "status": "prepared",
        "output_dir": str(output_dir),
        "representations": len(REPRESENTATION_VARIANTS),
        "development_instances": len(records),
        "seed0_operational_runs": len(REPRESENTATION_VARIANTS),
        "seed0_handling_runs": len(REPRESENTATION_VARIANTS),
        "seed0_pilot_rollouts": (
            len(REPRESENTATION_VARIANTS)
            * len(DEPLOYMENT_LAMBDAS)
            * len(records)
        ),
    }


def authenticate(output_dir: Path) -> tuple[dict, dict]:
    output_dir = output_dir.resolve()
    observed = load_json(output_dir / CONTRACT_NAME, label="E12 contract")
    if observed.get("contract_sha256") != digest(
        observed, hash_field="contract_sha256"
    ) or observed != contract():
        raise E12Error("E12 contract, inputs, or sources changed")
    manifest = load_json(
        output_dir / MANIFEST_NAME, label="E12 development manifest"
    )
    if (
        manifest.get("manifest_sha256")
        != digest(manifest, hash_field="manifest_sha256")
        or manifest.get("contract_sha256") != observed["contract_sha256"]
    ):
        raise E12Error("E12 development manifest authentication failed")
    return observed, manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "status"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output_dir)
    else:
        contract_value, manifest = authenticate(args.output_dir)
        result = {
            "status": "authenticated",
            "contract_sha256": contract_value["contract_sha256"],
            "development_instances": len(manifest["records"]),
        }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
