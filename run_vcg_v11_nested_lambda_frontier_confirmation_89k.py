#!/usr/bin/env python3
"""Unseen 89k confirmation of the frozen nested-VCG lambda frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
import tempfile
from typing import Mapping, Optional, Sequence

import torch

import run_vcg_final86_four_method as final86
import run_vcg_v11_nested_handling_pilot as pilot
from vcg_v11_nested_handling import HandlingAugmentedV11Agent, load_cost_checkpoint


PROTOCOL = "vcg_v1_1_nested_fixed_lambda_frontier_confirmation_89k_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (0, 1, 2)
INSTANCE_SEEDS = tuple(range(89_000, 89_030))
LAMBDA_GRID = (0.0, 0.025, 0.05, 0.1, 0.2)
EXPECTED_BLOCKS = 8
EXPECTED_ROWS = len(MODEL_SEEDS) * len(INSTANCE_SEEDS) * len(LAMBDA_GRID)
OPEN_CONFIRMATION = "OPEN_NESTED_LAMBDA_FRONTIER_89XXX"

CONTRACT_NAME = "confirmation-contract.json"
ACTIVATION_NAME = "panel-activation.json"
INSTANCE_MANIFEST_NAME = "episode-instance-manifest.json"
REPORT_NAME = "confirmation-report.json"

PARENT_SHA256 = {
    "frontier-contract.json": (
        "476988ea979ed4dac65cd977ee9cd89b3485a64dfcc1cc4ce6d17b80b846739c"
    ),
    "frontier-report.json": (
        "7009b78bab75ff14c4986d123231d7774625883e4b2c9d34cf4dd00c8bde192f"
    ),
}
SELECTED_CHECKPOINT_SHA256 = {
    0: "aa52ee14612ae39bb40fb9bdfecba8b7a44c40665628d26bb51f9f8f7a5355e8",
    1: "575326ddb3dbd535f4849e0f9d85bc55b8063279ebcfb6a872107f459dc56bb4",
    2: "3f681965feaca098a8d7fc2b58b41a8f0784191417e964a1fe7d5be2f5781917",
}
COST_HEAD_SHA256 = {
    0: "8caedeff2354a43227449c38b7eacaf47b40a6915ee275f8c37e83eb9679fc0c",
    1: "a81eeac765c7012d2f43ce28d91894568998cdfb9fb9e27da316556487cce887",
    2: "5cdd9ee650c4e92c665d4b7cc842cfb01d1ffd25e4088b2926493cca06c84122",
}

_AUTH_CACHE = {}


class ConfirmationError(RuntimeError):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ConfirmationError(f"expected a canonical regular file: {path}")
    return _sha256_bytes(path.read_bytes())


def _digest(payload: Mapping, *, hash_field: Optional[str] = None) -> str:
    value = dict(payload)
    if hash_field is not None:
        value.pop(hash_field, None)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _with_hash(payload: Mapping, field: str) -> dict:
    value = dict(payload)
    value[field] = _digest(value)
    return value


def _verify_hash(payload: Mapping, field: str, *, label: str) -> None:
    if payload.get(field) != _digest(payload, hash_field=field):
        raise ConfirmationError(f"{label} self hash mismatch")


def _load_json(path: Path, *, label: str) -> dict:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ConfirmationError(f"{label} is not a canonical regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfirmationError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ConfirmationError(f"{label} must contain an object")
    return value


def _atomic_json(path: Path, payload: Mapping) -> None:
    final86._atomic_json(path, payload)


def _atomic_text(path: Path, value: str) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _lambda_key(value: float) -> str:
    return str(float(value))


def _lambda_token(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def _paths(project_root: Path) -> dict:
    root = project_root.resolve()
    return {
        "parent": root
        / "results/vcg-v1-1-nested-lambda-frontier-85k-development",
        "cost_heads": {
            0: root
            / "results/vcg-v1-1-nested-handling-seed0-85k-development/handling-cost-head.pth",
            1: root
            / "results/vcg-v1-1-nested-handling-seed-stability-85k-development/cost-head/seed-1.pth",
            2: root
            / "results/vcg-v1-1-nested-handling-seed-stability-85k-development/cost-head/seed-2.pth",
        },
    }


def _source_paths(project_root: Path) -> dict[str, Path]:
    root = project_root.resolve()
    return {
        "benchmark_executor": root / "benchmark_viability_critic_priority.py",
        "confirmation_runner": Path(__file__).resolve(),
        "episode_instance": root / "example/episode_instance.py",
        "environment": root / "example/small_rooms_env.py",
        "nested_controller": root / "vcg_v11_nested_handling.py",
        "pilot_executor": root / "run_vcg_v11_nested_handling_pilot.py",
        "v11_authenticator": root / "run_vcg_final86_four_method.py",
    }


def _source_hashes(project_root: Path) -> dict[str, str]:
    return {
        name: _sha256_file(path)
        for name, path in sorted(_source_paths(project_root).items())
    }


def _authenticate_inputs(project_root: Path) -> tuple[dict, dict]:
    root = project_root.resolve()
    key = str(root)
    if key in _AUTH_CACHE:
        return _AUTH_CACHE[key]

    paths = _paths(root)
    for name, expected in PARENT_SHA256.items():
        if _sha256_file(paths["parent"] / name) != expected:
            raise ConfirmationError(f"development frontier artifact drifted: {name}")
    parent_contract = _load_json(
        paths["parent"] / "frontier-contract.json", label="frontier contract"
    )
    parent_report = _load_json(
        paths["parent"] / "frontier-report.json", label="frontier report"
    )
    if (
        parent_contract.get("lambda_grid") != list(LAMBDA_GRID)
        or parent_contract.get("model_seeds") != list(MODEL_SEEDS)
        or parent_contract.get("training_or_learning") is not False
        or parent_report.get("status") != "complete"
        or parent_report.get("lambda_grid") != list(LAMBDA_GRID)
        or parent_report.get("advance_all_five_levels_to_unseen_89k") is not True
        or not all(parent_report.get("sampled_frontier_criteria", {}).values())
    ):
        raise ConfirmationError("development frontier did not authorize 89k")

    arms, records = final86._authenticate_v11()
    arms = {int(arm.model_seed): arm for arm in arms}
    if tuple(sorted(arms)) != MODEL_SEEDS:
        raise ConfirmationError("VCG 1.1 selected checkpoint grid changed")
    for seed in MODEL_SEEDS:
        arm = arms[seed]
        expected_checkpoint = SELECTED_CHECKPOINT_SHA256[seed]
        if (
            arm.checkpoint_sha256 != expected_checkpoint
            or parent_contract["selected_checkpoints"][str(seed)]["raw_sha256"]
            != expected_checkpoint
            or parent_contract["selected_checkpoints"][str(seed)]["policy_digest"]
            != arm.deployment_policy_digest
        ):
            raise ConfirmationError(f"seed-{seed} operational checkpoint drifted")
        cost_path = paths["cost_heads"][seed]
        expected_cost = COST_HEAD_SHA256[seed]
        if (
            _sha256_file(cost_path) != expected_cost
            or parent_contract["cost_head_sha256"][str(seed)] != expected_cost
        ):
            raise ConfirmationError(f"seed-{seed} handling head drifted")
        payload = torch.load(cost_path, map_location="cpu", weights_only=True)
        base = pilot._fresh_base(arm, torch.device("cpu"))
        load_cost_checkpoint(
            payload,
            config=base.config,
            device=torch.device("cpu"),
            expected_source_checkpoint_sha256=arm.checkpoint_sha256,
            expected_source_policy_digest=arm.deployment_policy_digest,
            expected_model_seed=seed,
        )

    result = (arms, {"records": records, "paths": paths})
    _AUTH_CACHE[key] = result
    return result


def _contract(project_root: Path, output_dir: Path) -> dict:
    arms, _ = _authenticate_inputs(project_root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "prepared_panel_unopened",
        "scope": "unseen_89k_confirmation_of_frozen_nested_vcg_lambda_frontier",
        "output_dir": str(output_dir.absolute()),
        "panel_opened": False,
        "instance_seeds": list(INSTANCE_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "lambda_grid": list(LAMBDA_GRID),
        "expected_row_count": EXPECTED_ROWS,
        "training_or_learning": False,
        "checkpoint_or_lambda_selection": False,
        "same_frozen_qop_and_cost_head_within_each_model_seed": True,
        "lambda_changed_only_in_action_merit": True,
        "lambda_zero_direct_vcg_1_1_delegation": True,
        "deterministic_frozen_policy_evaluation": True,
        "all_rows_use_exact_shared_serialized_episode_instances": True,
        "environment": {
            "grid_rows": 5,
            "grid_cols": 5,
            "number_blocks": EXPECTED_BLOCKS,
            "arrival_rate_lambda": 10.0,
            "exponential_mean_interarrival": 0.1,
            "processing_time_poisson_mean": 80,
            "max_steps": 2000,
        },
        "primary_plane": ["mean_absolute_error", "physical_rehandles_per_100"],
        "confirmation_rule": {
            "all_450_rows_strict_safe_complete": True,
            "minimum_distinct_aggregate_nondominated_lambdas": 3,
            "minimum_seed_support_for_each_claimed_lambda": 2,
            "aggregate_rehandles_nonincreasing": True,
            "material_rehandle_reversal_in_at_most_one_seed": True,
            "complete_case_filtering_allowed": False,
        },
        "selected_checkpoints": {
            str(seed): {
                "raw_sha256": arms[seed].checkpoint_sha256,
                "deployment_policy_digest": arms[seed].deployment_policy_digest,
            }
            for seed in MODEL_SEEDS
        },
        "cost_head_sha256": {
            str(seed): COST_HEAD_SHA256[seed] for seed in MODEL_SEEDS
        },
        "development_parent_sha256": dict(PARENT_SHA256),
        "source_sha256": _source_hashes(project_root),
        "runtime_at_prepare": final86._runtime_versions(),
    }
    return _with_hash(payload, "contract_sha256")


def prepare(project_root: Path, output_dir: Path) -> dict:
    output = output_dir.absolute()
    if output.is_symlink():
        raise ConfirmationError("output directory must not be a symlink")
    expected = _contract(project_root, output)
    path = output / CONTRACT_NAME
    if path.exists():
        observed = _load_json(path, label="confirmation contract")
        _verify_hash(observed, "contract_sha256", label="confirmation contract")
        if observed != expected:
            raise ConfirmationError("existing confirmation contract changed")
    else:
        if output.exists() and any(output.iterdir()):
            raise ConfirmationError("nonempty output directory lacks a contract")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, expected)
    return {
        "status": "prepared_panel_unopened",
        "panel_opened": False,
        "expected_rows": EXPECTED_ROWS,
        "contract": str(path),
    }


def authenticate_contract(project_root: Path, output_dir: Path) -> dict:
    observed = _load_json(output_dir / CONTRACT_NAME, label="confirmation contract")
    _verify_hash(observed, "contract_sha256", label="confirmation contract")
    expected = _contract(project_root, output_dir.absolute())
    if observed != expected:
        raise ConfirmationError("contract, parent artifacts, or evaluator sources changed")
    return observed


def _instance_path(output_dir: Path, instance_seed: int) -> Path:
    return output_dir / "episode-instances" / f"seed-{instance_seed}.json"


def _new_environment():
    return final86._new_environment()


def open_panel(
    project_root: Path, output_dir: Path, *, confirmation: str
) -> dict:
    if confirmation != OPEN_CONFIRMATION:
        raise ConfirmationError(
            f"opening 89k requires --confirm {OPEN_CONFIRMATION}"
        )
    contract = authenticate_contract(project_root, output_dir)
    final86._configure_runtime()
    activation_path = output_dir / ACTIVATION_NAME
    if activation_path.exists():
        activation = _load_json(activation_path, label="panel activation")
        _verify_hash(activation, "activation_sha256", label="panel activation")
        if activation.get("contract_sha256") != contract["contract_sha256"]:
            raise ConfirmationError("activation/contract binding changed")
    else:
        activation = _with_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "panel_opened": True,
                "marker_written_before_first_instance_sample": True,
                "contract_sha256": contract["contract_sha256"],
                "contract_raw_sha256": _sha256_file(output_dir / CONTRACT_NAME),
                "authorized_seeds": list(INSTANCE_SEEDS),
                "runtime_versions": final86._runtime_versions(),
            },
            "activation_sha256",
        )
        _atomic_json(activation_path, activation)

    if (output_dir / INSTANCE_MANIFEST_NAME).is_file():
        return authenticate_instance_manifest(project_root, output_dir)

    env = _new_environment()
    records = []
    for index, instance_seed in enumerate(INSTANCE_SEEDS):
        path = _instance_path(output_dir, instance_seed)
        sampled = env.sample_episode_instance(instance_seed)
        if path.exists():
            observed = final86.EpisodeInstance.from_json(
                path.read_text(encoding="utf-8")
            )
            if observed != sampled:
                raise ConfirmationError(
                    f"serialized instance changed: {instance_seed}"
                )
        else:
            _atomic_text(path, sampled.to_json())
            observed = final86.EpisodeInstance.from_json(
                path.read_text(encoding="utf-8")
            )
            if observed != sampled:
                raise ConfirmationError(
                    f"instance serialization failed: {instance_seed}"
                )
        observed.validate_for(env)
        raw = path.read_bytes()
        records.append(
            {
                "instance_index": index,
                "instance_seed": instance_seed,
                "relative_path": str(path.relative_to(output_dir)),
                "raw_sha256": _sha256_bytes(raw),
                "canonical_sha256": _sha256_bytes(observed.to_json().encode()),
                "episode_instance_id": observed.instance_id,
                "schedule_id": observed.schedule_id,
            }
        )
    manifest = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "panel_opened": True,
            "contract_sha256": contract["contract_sha256"],
            "activation_sha256": activation["activation_sha256"],
            "instance_count": len(records),
            "instances": records,
            "all_450_rows_load_these_exact_serialized_instances": True,
        },
        "manifest_sha256",
    )
    _atomic_json(output_dir / INSTANCE_MANIFEST_NAME, manifest)
    return authenticate_instance_manifest(project_root, output_dir)


def authenticate_instance_manifest(project_root: Path, output_dir: Path) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    activation = _load_json(output_dir / ACTIVATION_NAME, label="panel activation")
    _verify_hash(activation, "activation_sha256", label="panel activation")
    if (
        activation.get("panel_opened") is not True
        or activation.get("contract_sha256") != contract["contract_sha256"]
        or activation.get("authorized_seeds") != list(INSTANCE_SEEDS)
    ):
        raise ConfirmationError("panel activation changed")
    manifest = _load_json(
        output_dir / INSTANCE_MANIFEST_NAME, label="instance manifest"
    )
    _verify_hash(manifest, "manifest_sha256", label="instance manifest")
    records = manifest.get("instances")
    if (
        manifest.get("contract_sha256") != contract["contract_sha256"]
        or manifest.get("activation_sha256") != activation["activation_sha256"]
        or not isinstance(records, list)
        or len(records) != len(INSTANCE_SEEDS)
        or [record.get("instance_seed") for record in records]
        != list(INSTANCE_SEEDS)
    ):
        raise ConfirmationError("instance manifest grid changed")
    env = _new_environment()
    expected_paths = set()
    for record in records:
        instance_seed = int(record["instance_seed"])
        path = output_dir / str(record["relative_path"])
        expected_paths.add(path.absolute())
        if path != _instance_path(output_dir, instance_seed):
            raise ConfirmationError("instance path changed")
        raw = path.read_bytes()
        if _sha256_bytes(raw) != record["raw_sha256"]:
            raise ConfirmationError(f"instance bytes changed: {instance_seed}")
        instance = final86.EpisodeInstance.from_json(raw.decode("utf-8"))
        instance.validate_for(env)
        if (
            instance.seed != instance_seed
            or instance.instance_id != record["episode_instance_id"]
            or instance.schedule_id != record["schedule_id"]
            or _sha256_bytes(instance.to_json().encode())
            != record["canonical_sha256"]
        ):
            raise ConfirmationError(f"instance identity changed: {instance_seed}")
    directory = output_dir / "episode-instances"
    observed_paths = {
        path.absolute()
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if observed_paths != expected_paths:
        raise ConfirmationError("instance file tree is not exact")
    return manifest


def _load_instance(output_dir: Path, record: Mapping):
    path = output_dir / str(record["relative_path"])
    raw = path.read_bytes()
    if _sha256_bytes(raw) != record["raw_sha256"]:
        raise ConfirmationError("instance raw SHA changed before evaluation")
    instance = final86.EpisodeInstance.from_json(raw.decode("utf-8"))
    if (
        instance.instance_id != record["episode_instance_id"]
        or instance.schedule_id != record["schedule_id"]
        or _sha256_bytes(instance.to_json().encode())
        != record["canonical_sha256"]
    ):
        raise ConfirmationError("instance identity changed before evaluation")
    return instance


def _ledger_path(
    output_dir: Path, value: float, model_seed: int, instance_seed: int
) -> Path:
    return (
        output_dir
        / "run-ledger"
        / f"lambda-{_lambda_token(value)}"
        / f"seed-{model_seed}"
        / f"instance-{instance_seed}.json"
    )


def _load_bound_cost(project_root: Path, arm, *, device, config):
    seed = int(arm.model_seed)
    path = _paths(project_root)["cost_heads"][seed]
    if _sha256_file(path) != COST_HEAD_SHA256[seed]:
        raise ConfirmationError("cost head changed before evaluation")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return load_cost_checkpoint(
        payload,
        config=config,
        device=device,
        expected_source_checkpoint_sha256=arm.checkpoint_sha256,
        expected_source_policy_digest=arm.deployment_policy_digest,
        expected_model_seed=seed,
    )


def _validate_ledger(
    ledger: Mapping,
    *,
    contract: Mapping,
    manifest: Mapping,
    value: float,
    model_seed: int,
    record: Mapping,
) -> dict:
    _verify_hash(ledger, "ledger_sha256", label="run ledger")
    expected = {
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "handling_lambda": value,
        "model_seed": model_seed,
        "instance_seed": record["instance_seed"],
        "episode_instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        "checkpoint_sha256": SELECTED_CHECKPOINT_SHA256[model_seed],
        "cost_head_sha256": (
            None if value == 0.0 else COST_HEAD_SHA256[model_seed]
        ),
    }
    for name, expected_value in expected.items():
        if ledger.get(name) != expected_value:
            raise ConfirmationError(f"run ledger identity changed: {name}")
    row = ledger.get("row")
    if not isinstance(row, Mapping):
        raise ConfirmationError("run ledger has no row")
    if (
        row.get("instance_seed") != record["instance_seed"]
        or row.get("episode_instance_id") != record["episode_instance_id"]
        or row.get("schedule_id") != record["schedule_id"]
    ):
        raise ConfirmationError("persisted row identity changed")
    return dict(row)


def evaluate(project_root: Path, output_dir: Path, *, device_name: str) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    manifest = authenticate_instance_manifest(project_root, output_dir)
    device = pilot._device(device_name)
    if device.type != "cuda":
        raise ConfirmationError("confirmation requires CUDA")
    final86._configure_runtime()
    arms, _ = _authenticate_inputs(project_root)

    costs = {}
    for model_seed in MODEL_SEEDS:
        arm = arms[model_seed]
        base = pilot._fresh_base(arm, device)
        costs[model_seed] = _load_bound_cost(
            project_root, arm, device=device, config=base.config
        )

    completed = 0
    for value in LAMBDA_GRID:
        for model_seed in MODEL_SEEDS:
            arm = arms[model_seed]
            for record_index, record in enumerate(manifest["instances"], start=1):
                instance_seed = int(record["instance_seed"])
                path = _ledger_path(output_dir, value, model_seed, instance_seed)
                if path.exists():
                    _validate_ledger(
                        _load_json(path, label="run ledger"),
                        contract=contract,
                        manifest=manifest,
                        value=value,
                        model_seed=model_seed,
                        record=record,
                    )
                    completed += 1
                    if record_index % 10 == 0:
                        print(
                            json.dumps(
                                {
                                    "progress": f"{completed}/{EXPECTED_ROWS}",
                                    "handling_lambda": value,
                                    "model_seed": model_seed,
                                    "instances_in_cell": record_index,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    continue
                instance = _load_instance(output_dir, record)
                try:
                    if value == 0.0:
                        raw = pilot._run_raw(arm, instance, device=device)
                    else:
                        def wrapper_factory(base, *, seed=model_seed, fixed=value):
                            return HandlingAugmentedV11Agent(
                                base,
                                costs[seed],
                                handling_lambda=fixed,
                            )

                        raw = pilot._run_raw(
                            arm,
                            instance,
                            device=device,
                            wrapper_factory=wrapper_factory,
                        )
                    row = pilot._compact_row(raw, instance)
                    row.update(
                        {
                            "handling_lambda": value,
                            "model_seed": model_seed,
                            "required_deliveries": EXPECTED_BLOCKS,
                            "evaluation_learning": False,
                            "cost_head_forward_calls": 0 if value == 0.0 else None,
                        }
                    )
                except Exception as exc:
                    row = {
                        "handling_lambda": value,
                        "model_seed": model_seed,
                        "instance_seed": instance_seed,
                        "episode_instance_id": record["episode_instance_id"],
                        "schedule_id": record["schedule_id"],
                        "strict_safe_complete": False,
                        "failure_type": type(exc).__name__,
                        "failure_reason": str(exc),
                        "evaluation_learning": False,
                    }
                    ledger = _with_hash(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "protocol": PROTOCOL,
                            "contract_sha256": contract["contract_sha256"],
                            "manifest_sha256": manifest["manifest_sha256"],
                            "handling_lambda": value,
                            "model_seed": model_seed,
                            "instance_seed": instance_seed,
                            "episode_instance_id": record["episode_instance_id"],
                            "schedule_id": record["schedule_id"],
                            "checkpoint_sha256": arm.checkpoint_sha256,
                            "cost_head_sha256": (
                                None if value == 0.0 else COST_HEAD_SHA256[model_seed]
                            ),
                            "row": row,
                        },
                        "ledger_sha256",
                    )
                    _atomic_json(path, ledger)
                    raise
                ledger = _with_hash(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "contract_sha256": contract["contract_sha256"],
                        "manifest_sha256": manifest["manifest_sha256"],
                        "handling_lambda": value,
                        "model_seed": model_seed,
                        "instance_seed": instance_seed,
                        "episode_instance_id": record["episode_instance_id"],
                        "schedule_id": record["schedule_id"],
                        "checkpoint_sha256": arm.checkpoint_sha256,
                        "cost_head_sha256": (
                            None if value == 0.0 else COST_HEAD_SHA256[model_seed]
                        ),
                        "row": row,
                    },
                    "ledger_sha256",
                )
                _atomic_json(path, ledger)
                completed += 1
                if record_index % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "progress": f"{completed}/{EXPECTED_ROWS}",
                                "handling_lambda": value,
                                "model_seed": model_seed,
                                "instances_in_cell": record_index,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    return {"status": "evaluation_complete", "completed_rows": completed}


def _summary(rows: Sequence[Mapping]) -> dict:
    total_rehandles = sum(int(row["physical_rehandles"]) for row in rows)
    return {
        "mean_dense_return": float(fmean(float(row["dense_return"]) for row in rows)),
        "mean_absolute_error": float(
            fmean(float(row["mean_absolute_error"]) for row in rows)
        ),
        "mean_steps": float(fmean(float(row["steps"]) for row in rows)),
        "total_physical_rehandles": total_rehandles,
        "physical_rehandles_per_100": float(
            100.0 * total_rehandles / (len(rows) * EXPECTED_BLOCKS)
        ),
    }


def _nondominated(metrics: Mapping[float, Mapping]) -> list[float]:
    result = []
    for value, point in metrics.items():
        dominated = False
        for other_value, other in metrics.items():
            if other_value == value:
                continue
            no_worse = (
                other["mean_absolute_error"] <= point["mean_absolute_error"]
                and other["physical_rehandles_per_100"]
                <= point["physical_rehandles_per_100"]
            )
            strict = (
                other["mean_absolute_error"] < point["mean_absolute_error"]
                or other["physical_rehandles_per_100"]
                < point["physical_rehandles_per_100"]
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            result.append(value)
    return sorted(result)


def _monotonicity(metrics: Mapping[float, Mapping]) -> dict:
    inversions = []
    for lower, upper in zip(LAMBDA_GRID, LAMBDA_GRID[1:]):
        delta = (
            metrics[upper]["physical_rehandles_per_100"]
            - metrics[lower]["physical_rehandles_per_100"]
        )
        if delta > 1e-12:
            inversions.append({"from": lower, "to": upper, "increase": delta})
    return {
        "inversion_count": len(inversions),
        "total_inversion_magnitude": sum(item["increase"] for item in inversions),
        "inversions": inversions,
    }


def _read_all_rows(
    output_dir: Path, contract: Mapping, manifest: Mapping
) -> dict[float, dict[int, dict[int, dict]]]:
    result = {
        value: {seed: {} for seed in MODEL_SEEDS} for value in LAMBDA_GRID
    }
    for value in LAMBDA_GRID:
        for model_seed in MODEL_SEEDS:
            for record in manifest["instances"]:
                instance_seed = int(record["instance_seed"])
                path = _ledger_path(output_dir, value, model_seed, instance_seed)
                if not path.is_file():
                    raise ConfirmationError(f"missing run ledger: {path}")
                result[value][model_seed][instance_seed] = _validate_ledger(
                    _load_json(path, label="run ledger"),
                    contract=contract,
                    manifest=manifest,
                    value=value,
                    model_seed=model_seed,
                    record=record,
                )
    return result


def analyze(project_root: Path, output_dir: Path) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    manifest = authenticate_instance_manifest(project_root, output_dir)
    rows = _read_all_rows(output_dir, contract, manifest)
    flat = [
        rows[value][seed][instance]
        for value in LAMBDA_GRID
        for seed in MODEL_SEEDS
        for instance in INSTANCE_SEEDS
    ]
    all_safe = len(flat) == EXPECTED_ROWS and all(
        row.get("strict_safe_complete") is True for row in flat
    )
    if not all_safe:
        report = _with_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "status": "failed_incomplete_or_unsafe",
                "whole_method_metrics_suppressed": True,
                "expected_rows": EXPECTED_ROWS,
                "observed_rows": len(flat),
                "strict_safe_rows": sum(
                    row.get("strict_safe_complete") is True for row in flat
                ),
                "contract_sha256": contract["contract_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            "report_sha256",
        )
        _atomic_json(output_dir / REPORT_NAME, report)
        return report

    seed_metrics = {
        seed: {
            value: _summary(
                [rows[value][seed][instance] for instance in INSTANCE_SEEDS]
            )
            for value in LAMBDA_GRID
        }
        for seed in MODEL_SEEDS
    }
    aggregate = {
        value: _summary(
            [
                rows[value][seed][instance]
                for seed in MODEL_SEEDS
                for instance in INSTANCE_SEEDS
            ]
        )
        for value in LAMBDA_GRID
    }
    aggregate_nd = _nondominated(aggregate)
    seed_nd = {seed: _nondominated(seed_metrics[seed]) for seed in MODEL_SEEDS}
    support = {
        value: sum(value in seed_nd[seed] for seed in MODEL_SEEDS)
        for value in LAMBDA_GRID
    }
    supported_nd = [value for value in aggregate_nd if support[value] >= 2]
    monotonicity = {
        "aggregate": _monotonicity(aggregate),
        "per_seed": {seed: _monotonicity(seed_metrics[seed]) for seed in MODEL_SEEDS},
    }
    distinct_points = len(
        {
            (
                round(aggregate[value]["mean_absolute_error"], 12),
                round(aggregate[value]["physical_rehandles_per_100"], 12),
            )
            for value in LAMBDA_GRID
        }
    )
    material_seed_inversions = sum(
        monotonicity["per_seed"][seed]["total_inversion_magnitude"] > 1.0
        for seed in MODEL_SEEDS
    )
    criteria = {
        "all_450_rows_strict_safe_complete": all_safe,
        "lambda0_zero_cost_head_calls": all(
            rows[0.0][seed][instance].get("cost_head_forward_calls") == 0
            for seed in MODEL_SEEDS
            for instance in INSTANCE_SEEDS
        ),
        "at_least_three_distinct_aggregate_nondominated_points": (
            len(aggregate_nd) >= 3 and distinct_points >= 3
        ),
        "at_least_three_nondominated_points_supported_in_two_of_three_seeds": (
            len(supported_nd) >= 3
        ),
        "aggregate_rehandles_nonincreasing": (
            monotonicity["aggregate"]["inversion_count"] == 0
        ),
        "material_rehandle_reversal_in_at_most_one_seed": (
            material_seed_inversions <= 1
        ),
        "frozen_input_hashes_unchanged": (
            _contract(project_root, output_dir) == contract
        ),
    }
    passed = all(criteria.values())
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "passed" if passed else "failed",
            "scope": "unseen_89k_frozen_policy_frontier_confirmation",
            "training_or_learning": False,
            "lambda_grid": list(LAMBDA_GRID),
            "row_count": len(flat),
            "strict_safe_row_count": len(flat),
            "seed_metrics": {
                str(seed): {
                    _lambda_key(value): seed_metrics[seed][value]
                    for value in LAMBDA_GRID
                }
                for seed in MODEL_SEEDS
            },
            "aggregate_metrics": {
                _lambda_key(value): aggregate[value] for value in LAMBDA_GRID
            },
            "aggregate_nondominated_lambdas": aggregate_nd,
            "per_seed_nondominated_lambdas": {
                str(seed): seed_nd[seed] for seed in MODEL_SEEDS
            },
            "frontier_support_count": {
                _lambda_key(value): support[value] for value in LAMBDA_GRID
            },
            "supported_aggregate_nondominated_lambdas": supported_nd,
            "handling_monotonicity": monotonicity,
            "confirmation_gate": {**criteria, "passed": passed},
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "interpretation": (
                "one frozen VCG 1.1 Qop plus one frozen detached handling head "
                "per model seed; lambda changes only the inference-time action merit"
            ),
        },
        "report_sha256",
    )
    path = output_dir / REPORT_NAME
    if path.exists() and _load_json(path, label="confirmation report") != report:
        raise ConfirmationError("existing confirmation report changed")
    if not path.exists():
        _atomic_json(path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "open-panel", "evaluate", "analyze", "run")
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--confirm", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    output = (
        args.output_dir.absolute()
        if args.output_dir is not None
        else root / "results/vcg-v1-1-nested-lambda-frontier-confirmation-89k"
    )
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(root, output)
    elif args.command == "open-panel":
        result = open_panel(root, output, confirmation=args.confirm)
    elif args.command == "evaluate":
        result = evaluate(root, output, device_name=args.device)
    elif args.command == "analyze":
        result = analyze(root, output)
    else:
        prepare(root, output)
        open_panel(root, output, confirmation=args.confirm)
        evaluate(root, output, device_name=args.device)
        result = analyze(root, output)
    display = result
    if "aggregate_metrics" in result:
        display = {
            "status": result["status"],
            "aggregate_metrics": result["aggregate_metrics"],
            "aggregate_nondominated_lambdas": result[
                "aggregate_nondominated_lambdas"
            ],
            "confirmation_gate": result["confirmation_gate"],
            "report": str(output / REPORT_NAME),
        }
    print(json.dumps(display, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
