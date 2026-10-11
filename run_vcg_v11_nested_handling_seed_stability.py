#!/usr/bin/env python3
"""Additive seed-1/2 continuation of the strictly nested VCG 1.1 pilot."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence

import torch

import benchmark_viability_critic_priority as benchmark
import compare_vcg_v2_3_matched_baselines as matched
import run_vcg_final86_four_method as final86
import run_vcg_v11_nested_handling_pilot as pilot
from vcg_v11_nested_handling import (
    DetachedHandlingCostNetwork,
    HandlingAugmentedV11Agent,
    cache_cost_dataset,
    cost_checkpoint,
    fit_cached_cost_network,
    load_cost_checkpoint,
)
from vcg_v11_replay_cost_dataset import build_replay_cost_dataset


PROTOCOL = "vcg_v1_1_nested_handling_lambda0025_seed_stability_85k_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (0, 1, 2)
NEW_MODEL_SEEDS = (1, 2)
SELECTED_EPISODES = {0: 500, 1: 450, 2: 375}
SELECTED_LAMBDA = 0.025
INSTANCE_SEEDS = tuple(range(85_000, 85_012))
LATEST_SHA256 = {
    1: "d2e9324222c706a77446cf26018bec24df7422d5d8ae4d8d2184423284b40861",
    2: "da6e7828f3d2f1862483d2899c29691413fd5e539d51fbd3a03fc7940e2b201b",
}
COST_SEEDS = {1: 81_101_000, 2: 81_102_000}
SPLIT_SEEDS = {1: 81_101_001, 2: 81_102_001}

PARENT_ARTIFACT_SHA256 = {
    "pilot-contract.json": "a057ab31be91545b1f4b7898ed31782b96d193c76b9f53a3a98dd823a9ac370f",
    "lambda-zero-parity.json": "59ce63949d8cb788591bf7796bb7eb2c875d4422110fbb1c9a6bb6a07a7b4eb7",
    "handling-cost-head.pth": "8caedeff2354a43227449c38b7eacaf47b40a6915ee275f8c37e83eb9679fc0c",
    "handling-cost-training.json": "5ec9ed2930a63e9a16e536a4a1f14afbd62a6f507b2bad27edffcc39c2240d9a",
    "pilot-sweep.json": "f92b0b12ac83423c2312f51f867e7c20e77802592ef23f0594b144d8e6209db1",
}

CONTRACT_NAME = "stability-contract.json"
REPORT_NAME = "stability-report.json"
EXPECTED_BLOCKS = 8
_AUTH_CACHE = {}


class StabilityError(RuntimeError):
    pass


def _paths(project_root: Path) -> dict:
    root = project_root.resolve()
    return {
        "parent": root
        / "results/vcg-v1-1-nested-handling-seed0-85k-development",
        "v23_source": root
        / "results/vcg-constrained-v2-3-gamma1-ablation-seed10-200ep",
        "v11_control": root
        / "results/vcg-dense-v1-1-v2-2-panel-control-12instance",
        "v22_source": root
        / "results/vcg-constrained-v2-2-development-seed10-200ep",
    }


def _latest_path(project_root: Path, seed: int) -> Path:
    return (
        project_root.resolve()
        / f"results/vcg-dense-v1-1-seed{seed}-500ep/latest.pth"
    )


def _authenticate_parent(parent: Path) -> dict:
    for name, expected in PARENT_ARTIFACT_SHA256.items():
        path = parent / name
        if not path.is_file() or pilot._sha256_file(path) != expected:
            raise StabilityError(f"seed-0 parent artifact drifted: {name}")
    contract = json.loads((parent / "pilot-contract.json").read_text())
    parity = json.loads((parent / "lambda-zero-parity.json").read_text())
    sweep = json.loads((parent / "pilot-sweep.json").read_text())
    if contract.get("protocol") != pilot.PROTOCOL:
        raise StabilityError("seed-0 parent protocol drifted")
    live_parent_sources = {
        "run_vcg_v11_nested_handling_pilot.py": pilot._sha256_file(
            Path(pilot.__file__).resolve()
        ),
        "vcg_v11_nested_handling.py": pilot._sha256_file(
            Path(__import__("vcg_v11_nested_handling").__file__).resolve()
        ),
        "vcg_v11_replay_cost_dataset.py": pilot._sha256_file(
            Path(__import__("vcg_v11_replay_cost_dataset").__file__).resolve()
        ),
    }
    if contract.get("source_sha256") != live_parent_sources:
        raise StabilityError("seed-0 parent runtime sources changed")
    if (
        parity.get("status") != "passed"
        or parity.get("lambda_zero_cost_forward_calls") != 0
        or parity.get("sentinel_behavior_digest_exact") is not True
    ):
        raise StabilityError("seed-0 lambda-zero parity did not authenticate")
    if (
        sweep.get("status") != "complete"
        or sweep.get("selected_smallest_qualifying_lambda") != SELECTED_LAMBDA
        or sweep.get("advance_to_seeds_1_and_2") is not True
    ):
        raise StabilityError("seed-0 pilot did not authorize this continuation")
    for key in ("0.0", str(SELECTED_LAMBDA)):
        rows = sweep.get("rows", {}).get(key)
        if not isinstance(rows, list) or len(rows) != len(INSTANCE_SEEDS):
            raise StabilityError(f"seed-0 parent row grid is incomplete: {key}")
        if not all(row.get("strict_safe_complete") is True for row in rows):
            raise StabilityError(f"seed-0 parent row is unsafe: {key}")
    return {"contract": contract, "parity": parity, "sweep": sweep}


def _authenticate_inputs(project_root: Path):
    key = str(project_root.resolve())
    if key in _AUTH_CACHE:
        return _AUTH_CACHE[key]
    paths = _paths(project_root)
    parent = _authenticate_parent(paths["parent"])
    arms, records = final86._authenticate_v11()
    arms = {int(arm.model_seed): arm for arm in arms}
    if tuple(sorted(arms)) != MODEL_SEEDS:
        raise StabilityError("VCG 1.1 selected checkpoint grid drifted")
    sources = matched.authenticate_sources(
        v23_source_dir=paths["v23_source"],
        v11_control_dir=paths["v11_control"],
        v22_source_dir=paths["v22_source"],
    )
    if tuple(sorted(sources.instances)) != INSTANCE_SEEDS:
        raise StabilityError("authenticated 85k instance grid drifted")
    for arm_key in ("0.0", str(SELECTED_LAMBDA)):
        parent_rows = {
            int(row["instance_seed"]): row
            for row in parent["sweep"]["rows"][arm_key]
        }
        if tuple(sorted(parent_rows)) != INSTANCE_SEEDS:
            raise StabilityError(f"seed-0 parent identity grid drifted: {arm_key}")
        for instance_seed in INSTANCE_SEEDS:
            row = parent_rows[instance_seed]
            instance = sources.instances[instance_seed]
            if (
                row.get("episode_instance_id") != instance.instance_id
                or row.get("schedule_id") != instance.schedule_id
            ):
                raise StabilityError(
                    f"seed-0 parent instance identity drifted: {arm_key}/{instance_seed}"
                )
    latest = {}
    datasets = {}
    for seed in NEW_MODEL_SEEDS:
        path = _latest_path(project_root, seed)
        data = path.read_bytes()
        if pilot._sha256_bytes(data) != LATEST_SHA256[seed]:
            raise StabilityError(f"VCG 1.1 seed-{seed} latest SHA drifted")
        payload = torch.load(
            BytesIO(data), map_location="cpu", weights_only=False
        )
        if not isinstance(payload, Mapping):
            raise StabilityError(f"VCG 1.1 seed-{seed} latest is not a mapping")
        expected = {
            "model_seed": seed,
            "completed_training_episodes": 500,
            "selected_checkpoint_episode": SELECTED_EPISODES[seed],
            "checkpoint_role": "latest_resumable",
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise StabilityError(f"seed-{seed} latest mismatch: {name}")
        arm = arms[seed]
        if arm.checkpoint_weight_episode != SELECTED_EPISODES[seed]:
            raise StabilityError(f"seed-{seed} selected best episode drifted")
        latest[seed] = payload
        datasets[seed] = build_replay_cost_dataset(
            payload,
            split_seed=SPLIT_SEEDS[seed],
            validation_fraction=pilot.VALIDATION_FRACTION,
            selected_episode_limit=SELECTED_EPISODES[seed],
        )
    result = (arms, sources, latest, datasets, parent, records)
    _AUTH_CACHE[key] = result
    return result


def _source_hashes() -> dict:
    paths = {
        "stability_runner": Path(__file__).resolve(),
        "seed0_runner": Path(pilot.__file__).resolve(),
        "nested_controller": Path(
            __import__("vcg_v11_nested_handling").__file__
        ).resolve(),
        "replay_dataset": Path(
            __import__("vcg_v11_replay_cost_dataset").__file__
        ).resolve(),
        "benchmark_executor": Path(benchmark.__file__).resolve(),
    }
    return {name: pilot._sha256_file(path) for name, path in sorted(paths.items())}


def _contract(project_root: Path) -> dict:
    arms, sources, latest, datasets, parent, records = _authenticate_inputs(
        project_root
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "development_seed_stability_on_opened_85000_85011",
        "final_86xxx_opened": False,
        "final_87xxx_used": False,
        "new_environment_training": False,
        "model_seeds": list(MODEL_SEEDS),
        "new_model_seeds": list(NEW_MODEL_SEEDS),
        "selected_lambda": SELECTED_LAMBDA,
        "selected_lambda_source": (
            "smallest_predeclared_qualifying_value_from_authenticated_seed0_pilot"
        ),
        "selected_episodes": {
            str(seed): SELECTED_EPISODES[seed] for seed in MODEL_SEEDS
        },
        "selected_checkpoints": {
            str(seed): {
                "raw_sha256": arms[seed].checkpoint_sha256,
                "policy_digest": arms[seed].deployment_policy_digest,
            }
            for seed in MODEL_SEEDS
        },
        "latest_replay_sha256": {
            str(seed): LATEST_SHA256[seed] for seed in NEW_MODEL_SEEDS
        },
        "dataset_audits": {
            str(seed): datasets[seed].audit_dict() for seed in NEW_MODEL_SEEDS
        },
        "cost_seeds": {str(seed): COST_SEEDS[seed] for seed in NEW_MODEL_SEEDS},
        "split_seeds": {str(seed): SPLIT_SEEDS[seed] for seed in NEW_MODEL_SEEDS},
        "instance_seeds": list(INSTANCE_SEEDS),
        "new_evaluation_rows": len(NEW_MODEL_SEEDS) * len(INSTANCE_SEEDS),
        "lambda_zero_rows_reused": len(MODEL_SEEDS) * len(INSTANCE_SEEDS),
        "seed0_positive_rows_reused": len(INSTANCE_SEEDS),
        "stability_gate": {
            "all_72_reused_and_new_rows_strict_safe_complete": True,
            "aggregate_physical_rehandles_per_100_strictly_lower": True,
            "aggregate_maximum_mae_increase": 2.0,
            "aggregate_maximum_dense_return_decrease": 20.0,
            "both_new_seeds_pass_original_gate": True,
            "individual_minimum_total_rehandles_saved": 2,
        },
        "parent_artifact_sha256": dict(PARENT_ARTIFACT_SHA256),
        "source_sha256": _source_hashes(),
        "runtime_at_prepare": pilot._runtime(),
    }
    # Normalize tuples and integer dictionary keys exactly as persisted JSON.
    return json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))


def prepare(project_root: Path, output_dir: Path) -> dict:
    contract = _contract(project_root)
    path = output_dir / CONTRACT_NAME
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != contract:
            raise StabilityError("existing stability contract changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise StabilityError("nonempty output directory lacks stability contract")
        pilot._atomic_json(path, contract)
    return {
        "status": "prepared",
        "selected_lambda": SELECTED_LAMBDA,
        "new_evaluation_rows": 24,
        "contract": str(path),
    }


def _require_contract(project_root: Path, output_dir: Path) -> dict:
    path = output_dir / CONTRACT_NAME
    if not path.is_file():
        raise StabilityError("run prepare first")
    observed = json.loads(path.read_text())
    if observed != _contract(project_root):
        raise StabilityError("stability contract or a bound input changed")
    return observed


def _cost_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "cost-head" / f"seed-{seed}.pth"


def _cost_report_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "cost-head" / f"seed-{seed}.json"


def _ledger_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "evaluation-ledger" / f"seed-{seed}.json"


def train_seed(
    project_root: Path, output_dir: Path, *, seed: int, device_name: str
) -> dict:
    _require_contract(project_root, output_dir)
    if seed not in NEW_MODEL_SEEDS:
        raise StabilityError("only seeds 1 and 2 are new training jobs")
    report_path = _cost_report_path(output_dir, seed)
    checkpoint_path = _cost_path(output_dir, seed)
    if report_path.is_file() and checkpoint_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("cost_checkpoint_sha256") != pilot._sha256_file(
            checkpoint_path
        ):
            raise StabilityError(f"seed-{seed} cost checkpoint/report mismatch")
        return report

    arms, sources, latest, datasets, parent, records = _authenticate_inputs(
        project_root
    )
    arm = arms[seed]
    device = pilot._device(device_name)
    base = pilot._fresh_base(arm, device)
    local_before = pilot._state_digest(base.Q_local.state_dict())
    target_before = pilot._state_digest(base.Q_target.state_dict())
    dataset = datasets[seed]
    training = cache_cost_dataset(
        base.Q_local, dataset.train_samples, batch_size=pilot.COST_BATCH_SIZE
    )
    validation = cache_cost_dataset(
        base.Q_local,
        dataset.validation_samples,
        batch_size=pilot.COST_BATCH_SIZE,
    )
    network = DetachedHandlingCostNetwork(
        base.config, seed=COST_SEEDS[seed]
    ).to(device)
    initial = pilot._state_digest(network.state_dict())
    fit = fit_cached_cost_network(
        network,
        training,
        validation,
        epochs=pilot.COST_EPOCHS,
        batch_size=pilot.COST_BATCH_SIZE,
        learning_rate=pilot.COST_LEARNING_RATE,
        seed=COST_SEEDS[seed],
    )
    final = pilot._state_digest(network.state_dict())
    local_after = pilot._state_digest(base.Q_local.state_dict())
    target_after = pilot._state_digest(base.Q_target.state_dict())
    if initial == final:
        raise StabilityError(f"seed-{seed} cost head did not learn")
    if (local_before, target_before) != (local_after, target_after):
        raise StabilityError(f"seed-{seed} cost fitting changed frozen Qop")
    record = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        "model_seed": seed,
        "selected_episode": SELECTED_EPISODES[seed],
        "dataset": dataset.audit_dict(),
        "fit": fit,
        "qop_local_before_sha256": local_before,
        "qop_local_after_sha256": local_after,
        "qop_target_before_sha256": target_before,
        "qop_target_after_sha256": target_after,
        "cost_initial_sha256": initial,
        "cost_final_sha256": final,
        "device": str(device),
    }
    payload = cost_checkpoint(
        network,
        source_checkpoint_sha256=arm.checkpoint_sha256,
        source_policy_digest=arm.deployment_policy_digest,
        model_seed=seed,
        training_record=record,
    )
    pilot._atomic_torch(checkpoint_path, payload)
    record["cost_checkpoint_sha256"] = pilot._sha256_file(checkpoint_path)
    pilot._atomic_json(report_path, record)
    return record


def _load_cost(output_dir: Path, arm, *, seed: int, device, config):
    path = _cost_path(output_dir, seed)
    report_path = _cost_report_path(output_dir, seed)
    if not path.is_file() or not report_path.is_file():
        raise StabilityError(f"train seed {seed} first")
    report = json.loads(report_path.read_text())
    if report.get("cost_checkpoint_sha256") != pilot._sha256_file(path):
        raise StabilityError(f"seed-{seed} cost checkpoint drifted")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return load_cost_checkpoint(
        payload,
        config=config,
        device=device,
        expected_source_checkpoint_sha256=arm.checkpoint_sha256,
        expected_source_policy_digest=arm.deployment_policy_digest,
        expected_model_seed=seed,
    )


def _lambda_zero_rows(sources, seed: int) -> list[dict]:
    rows = [
        row for row in sources.v11_rows if int(row.get("model_seed", -1)) == seed
    ]
    rows.sort(key=lambda row: int(row["instance_seed"]))
    if [int(row["instance_seed"]) for row in rows] != list(INSTANCE_SEEDS):
        raise StabilityError(f"seed-{seed} lambda-zero source grid is incomplete")
    output = []
    for row in rows:
        physical = int(row["physical_storage_relocations"])
        safe = bool(
            row["strict_method_success"] == 1.0
            and row["completion_rate"] == 1.0
            and row["method_failure_reason"] is None
            and row["illegal_drops"] == 0
            and row["fallbacks"] == 0
        )
        if not safe:
            raise StabilityError(f"seed-{seed} lambda-zero source row is unsafe")
        output.append(
            {
                "instance_seed": int(row["instance_seed"]),
                "episode_instance_id": row["instance_id"],
                "schedule_id": row["schedule_id"],
                "strict_safe_complete": True,
                "dense_return": float(row["dense_rescored_return"]),
                "mean_absolute_error": float(row["mean_absolute_error"]),
                "steps": int(row["steps"]),
                "physical_rehandles": physical,
                "physical_rehandles_per_100": 100.0
                * physical
                / EXPECTED_BLOCKS,
                "delivery_deviations": list(row["delivery_deviations"]),
                "execution_reused": True,
            }
        )
    return output


def evaluate_seed(
    project_root: Path, output_dir: Path, *, seed: int, device_name: str
) -> dict:
    _require_contract(project_root, output_dir)
    if seed not in NEW_MODEL_SEEDS:
        raise StabilityError("only seeds 1 and 2 need new evaluation")
    path = _ledger_path(output_dir, seed)
    if path.is_file():
        ledger = json.loads(path.read_text())
        if (
            ledger.get("model_seed") != seed
            or ledger.get("fixed_lambda") != SELECTED_LAMBDA
            or len(ledger.get("rows", ())) != len(INSTANCE_SEEDS)
            or ledger.get("cost_checkpoint_sha256")
            != pilot._sha256_file(_cost_path(output_dir, seed))
            or not all(
                row.get("strict_safe_complete") is True
                for row in ledger.get("rows", ())
            )
        ):
            raise StabilityError(f"seed-{seed} evaluation ledger drifted")
        return ledger
    arms, sources, latest, datasets, parent, records = _authenticate_inputs(
        project_root
    )
    arm = arms[seed]
    device = pilot._device(device_name)
    probe = pilot._fresh_base(arm, device)
    _load_cost(output_dir, arm, seed=seed, device=device, config=probe.config)
    rows = []
    for instance_seed in INSTANCE_SEEDS:
        instance = sources.instances[instance_seed]

        def wrapper_factory(base):
            cost = _load_cost(
                output_dir,
                arm,
                seed=seed,
                device=device,
                config=base.config,
            )
            return HandlingAugmentedV11Agent(
                base, cost, handling_lambda=SELECTED_LAMBDA
            )

        raw = pilot._run_raw(
            arm, instance, device=device, wrapper_factory=wrapper_factory
        )
        rows.append(pilot._compact_row(raw, instance))
    summary = pilot._summary(rows)
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        "model_seed": seed,
        "fixed_lambda": SELECTED_LAMBDA,
        "cost_checkpoint_sha256": pilot._sha256_file(_cost_path(output_dir, seed)),
        "row_count": len(rows),
        "summary": summary,
        "rows": rows,
    }
    pilot._atomic_json(path, ledger)
    return ledger


def _metrics_gate(base: Mapping, augmented: Mapping) -> dict:
    if base is None or augmented is None:
        return {
            "whole_method_eligible": False,
            "saves_at_least_two_rehandles": False,
            "mae_increase_at_most_two": False,
            "dense_return_decrease_at_most_twenty": False,
            "passed": False,
        }
    conditions = {
        "whole_method_eligible": True,
        "saves_at_least_two_rehandles": (
            int(base["total_physical_rehandles"])
            - int(augmented["total_physical_rehandles"])
            >= 2
        ),
        "mae_increase_at_most_two": (
            float(augmented["mean_absolute_error"])
            <= float(base["mean_absolute_error"]) + 2.0
        ),
        "dense_return_decrease_at_most_twenty": (
            float(augmented["mean_dense_return"])
            >= float(base["mean_dense_return"]) - 20.0
        ),
    }
    return {**conditions, "passed": all(conditions.values())}


def _aggregate(seed_summaries: Mapping[int, Mapping]) -> dict | None:
    if any(
        not summary.get("whole_method_eligible")
        or summary.get("metrics") is None
        for summary in seed_summaries.values()
    ):
        return None
    metrics = [seed_summaries[seed]["metrics"] for seed in MODEL_SEEDS]
    total = sum(int(value["total_physical_rehandles"]) for value in metrics)
    return {
        "mean_dense_return": fmean(value["mean_dense_return"] for value in metrics),
        "mean_absolute_error": fmean(
            value["mean_absolute_error"] for value in metrics
        ),
        "mean_steps": fmean(value["mean_steps"] for value in metrics),
        "total_physical_rehandles": total,
        "physical_rehandles_per_100": 100.0
        * total
        / (len(MODEL_SEEDS) * len(INSTANCE_SEEDS) * EXPECTED_BLOCKS),
    }


def analyze(project_root: Path, output_dir: Path) -> dict:
    contract = _require_contract(project_root, output_dir)
    arms, sources, latest, datasets, parent, records = _authenticate_inputs(
        project_root
    )
    seed0 = parent["sweep"]["rows"]
    lambda_zero_rows = {0: seed0["0.0"]}
    augmented_rows = {0: seed0[str(SELECTED_LAMBDA)]}
    for seed in NEW_MODEL_SEEDS:
        lambda_zero_rows[seed] = _lambda_zero_rows(sources, seed)
        path = _ledger_path(output_dir, seed)
        if not path.is_file():
            raise StabilityError(f"evaluate seed {seed} first")
        ledger = json.loads(path.read_text())
        if ledger.get("cost_checkpoint_sha256") != pilot._sha256_file(
            _cost_path(output_dir, seed)
        ):
            raise StabilityError(f"seed-{seed} evaluation/cost binding drifted")
        augmented_rows[seed] = ledger["rows"]
    base_summaries = {
        seed: pilot._summary(lambda_zero_rows[seed]) for seed in MODEL_SEEDS
    }
    augmented_summaries = {
        seed: pilot._summary(augmented_rows[seed]) for seed in MODEL_SEEDS
    }
    individual_gates = {
        seed: _metrics_gate(
            base_summaries[seed]["metrics"],
            augmented_summaries[seed]["metrics"],
        )
        for seed in MODEL_SEEDS
    }
    base_aggregate = _aggregate(base_summaries)
    augmented_aggregate = _aggregate(augmented_summaries)
    individual_passes = sum(gate["passed"] for gate in individual_gates.values())
    all_72_safe = all(
        row["strict_safe_complete"]
        for seed in MODEL_SEEDS
        for arm_rows in (lambda_zero_rows[seed], augmented_rows[seed])
        for row in arm_rows
    )
    if base_aggregate is None or augmented_aggregate is None:
        aggregate_conditions = {
            "all_72_reused_and_new_rows_strict_safe_complete": all_72_safe,
            "aggregate_physical_rehandles_per_100_strictly_lower": False,
            "mae_increase_at_most_two": False,
            "dense_return_decrease_at_most_twenty": False,
            "both_new_seeds_pass_original_gate": all(
                individual_gates[seed]["passed"] for seed in NEW_MODEL_SEEDS
            ),
        }
    else:
        aggregate_conditions = {
            "all_72_reused_and_new_rows_strict_safe_complete": all_72_safe,
            "aggregate_physical_rehandles_per_100_strictly_lower": (
                augmented_aggregate["physical_rehandles_per_100"]
                < base_aggregate["physical_rehandles_per_100"]
            ),
            "mae_increase_at_most_two": (
                augmented_aggregate["mean_absolute_error"]
                <= base_aggregate["mean_absolute_error"] + 2.0
            ),
            "dense_return_decrease_at_most_twenty": (
                augmented_aggregate["mean_dense_return"]
                >= base_aggregate["mean_dense_return"] - 20.0
            ),
            "both_new_seeds_pass_original_gate": all(
                individual_gates[seed]["passed"] for seed in NEW_MODEL_SEEDS
            ),
        }
    passed = all(aggregate_conditions.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "passed" if passed else "failed",
        "scope": "development_seed_stability",
        "selected_lambda": SELECTED_LAMBDA,
        "lambda_zero_exact_vcg_1_1": True,
        "base_seed_summaries": base_summaries,
        "augmented_seed_summaries": augmented_summaries,
        "individual_seed_gates": individual_gates,
        "individual_seed_pass_count": individual_passes,
        "base_equal_seed_aggregate": base_aggregate,
        "augmented_equal_seed_aggregate": augmented_aggregate,
        "stability_gate": {**aggregate_conditions, "passed": passed},
        "new_evaluation_row_count": 24,
        "new_environment_training": False,
        "final_86xxx_opened": False,
        "final_87xxx_used": False,
        "interpretation": (
            "three-seed development stability of frozen VCG 1.1 plus a "
            "detached Monte Carlo handling predictor; not a dual-convergence claim"
        ),
    }
    path = output_dir / REPORT_NAME
    if path.exists():
        persisted_report = json.loads(
            json.dumps(report, sort_keys=True, allow_nan=False)
        )
        if json.loads(path.read_text()) != persisted_report:
            raise StabilityError("existing stability report changed")
    else:
        pilot._atomic_json(path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "train-seed1",
            "train-seed2",
            "evaluate-seed1",
            "evaluate-seed2",
            "analyze",
            "run-all",
        ),
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.project_root.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root
        / "results/vcg-v1-1-nested-handling-seed-stability-85k-development"
    )
    torch.set_num_threads(1)
    command = args.command
    if command == "prepare":
        result = prepare(root, output)
    elif command.startswith("train-seed"):
        result = train_seed(
            root, output, seed=int(command[-1]), device_name=args.device
        )
    elif command.startswith("evaluate-seed"):
        result = evaluate_seed(
            root, output, seed=int(command[-1]), device_name=args.device
        )
    elif command == "analyze":
        result = analyze(root, output)
    else:
        prepare(root, output)
        training = {
            seed: train_seed(
                root, output, seed=seed, device_name=args.device
            )
            for seed in NEW_MODEL_SEEDS
        }
        evaluation = {
            seed: evaluate_seed(
                root, output, seed=seed, device_name=args.device
            )
            for seed in NEW_MODEL_SEEDS
        }
        result = analyze(root, output)
    if "stability_gate" in result:
        display = {
            "status": result["status"],
            "selected_lambda": result["selected_lambda"],
            "individual_seed_pass_count": result[
                "individual_seed_pass_count"
            ],
            "base_equal_seed_aggregate": result["base_equal_seed_aggregate"],
            "augmented_equal_seed_aggregate": result[
                "augmented_equal_seed_aggregate"
            ],
            "stability_gate": result["stability_gate"],
            "report": str(output / REPORT_NAME),
        }
    elif "fit" in result:
        display = {
            "status": result["status"],
            "model_seed": result["model_seed"],
            "validation_mae_rehandles": result["fit"]["validation"][
                "mae_rehandles"
            ],
        }
    elif "summary" in result:
        display = {
            "status": result["status"],
            "model_seed": result["model_seed"],
            "summary": result["summary"],
        }
    else:
        display = result
    print(json.dumps(display, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
