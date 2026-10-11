#!/usr/bin/env python3
"""Unseen-panel confirmation of VCG 1.1 with a nested handling cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev
import tempfile
from typing import Mapping, Optional, Sequence

import torch

import run_vcg_final86_four_method as final86
import run_vcg_v11_nested_handling_pilot as pilot
from vcg_v11_nested_handling import HandlingAugmentedV11Agent, load_cost_checkpoint


PROTOCOL = "vcg_v1_1_nested_handling_lambda0025_confirmation_88k_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (0, 1, 2)
INSTANCE_SEEDS = tuple(range(88_000, 88_030))
ARMS = ("lambda0", "lambda0025")
ARM_LAMBDAS = {"lambda0": 0.0, "lambda0025": 0.025}
EXPECTED_BLOCKS = 8
EXPECTED_ROWS = len(MODEL_SEEDS) * len(INSTANCE_SEEDS) * len(ARMS)
OPEN_CONFIRMATION = "OPEN_NESTED_HANDLING_88XXX"

CONTRACT_NAME = "confirmation-contract.json"
ACTIVATION_NAME = "panel-activation.json"
INSTANCE_MANIFEST_NAME = "episode-instance-manifest.json"
REPORT_NAME = "confirmation-report.json"

SIMULTANEOUS_ONE_SIDED_T = 2.2343299587959504
NOMINAL_TWO_SIDED_T = 2.0452296421

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
PARENT_SHA256 = {
    "stability-contract.json": "184159c358d78478eb38de0d60613375cfea38ddcddb772b20d5e362084aa639",
    "stability-report.json": "46e502927b3baafec6cac9ae3c0d714ad8dece8b729c459e213482528882acae",
}


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


def _atomic_text(path: Path, text: str) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _paths(project_root: Path) -> dict:
    root = project_root.resolve()
    return {
        "stability": root
        / "results/vcg-v1-1-nested-handling-seed-stability-85k-development",
        "seed0": root
        / "results/vcg-v1-1-nested-handling-seed0-85k-development",
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
    paths = {
        f"inherited:{name}": Path(path).resolve()
        for name, path in final86._source_paths().items()
    }
    paths.update({
        "confirmation_runner": Path(__file__).resolve(),
        "confirmation_plot": root
        / "plot_vcg_v11_nested_handling_confirmation_88k.py",
        "nested_controller": root / "vcg_v11_nested_handling.py",
        "pilot_executor": root / "run_vcg_v11_nested_handling_pilot.py",
        "stability_runner": root
        / "run_vcg_v11_nested_handling_seed_stability.py",
        "v11_authenticator": root / "run_vcg_final86_four_method.py",
        "benchmark_executor": root / "benchmark_viability_critic_priority.py",
        "episode_instance": root / "example/episode_instance.py",
        "environment": root / "example/small_rooms_env.py",
    })
    return dict(sorted(paths.items()))


def _source_hashes(project_root: Path) -> dict[str, str]:
    return {
        name: _sha256_file(path)
        for name, path in _source_paths(project_root).items()
    }


def _authenticate_inputs(project_root: Path) -> tuple[dict, dict]:
    root = project_root.resolve()
    paths = _paths(root)
    stability = paths["stability"]
    for name, expected in PARENT_SHA256.items():
        path = stability / name
        if _sha256_file(path) != expected:
            raise ConfirmationError(f"selected stability artifact drifted: {name}")
    report = _load_json(stability / "stability-report.json", label="stability report")
    if (
        report.get("status") != "passed"
        or report.get("selected_lambda") != ARM_LAMBDAS["lambda0025"]
        or report.get("stability_gate", {}).get("passed") is not True
    ):
        raise ConfirmationError("development stability did not authorize confirmation")

    arms, records = final86._authenticate_v11()
    arms = {int(arm.model_seed): arm for arm in arms}
    if tuple(sorted(arms)) != MODEL_SEEDS:
        raise ConfirmationError("VCG 1.1 selected checkpoint grid changed")
    for seed in MODEL_SEEDS:
        if arms[seed].checkpoint_sha256 != SELECTED_CHECKPOINT_SHA256[seed]:
            raise ConfirmationError(f"VCG 1.1 seed-{seed} checkpoint drifted")
        cost_path = paths["cost_heads"][seed]
        if _sha256_file(cost_path) != COST_HEAD_SHA256[seed]:
            raise ConfirmationError(f"seed-{seed} handling head drifted")
        payload = torch.load(cost_path, map_location="cpu", weights_only=True)
        base = pilot._fresh_base(arms[seed], torch.device("cpu"))
        load_cost_checkpoint(
            payload,
            config=base.config,
            device=torch.device("cpu"),
            expected_source_checkpoint_sha256=arms[seed].checkpoint_sha256,
            expected_source_policy_digest=arms[seed].deployment_policy_digest,
            expected_model_seed=seed,
        )
    return arms, {"records": records, "paths": paths}


def _contract(project_root: Path, output_dir: Path) -> dict:
    arms, auth = _authenticate_inputs(project_root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "prepared_panel_unopened",
        "scope": "unseen_88k_confirmation_of_fixed_three_controller_ensemble",
        "output_dir": str(output_dir.resolve()),
        "panel_opened": False,
        "instance_seeds": list(INSTANCE_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "arms": [
            {"arm": arm, "handling_lambda": ARM_LAMBDAS[arm]}
            for arm in ARMS
        ],
        "expected_row_count": EXPECTED_ROWS,
        "training_or_learning": False,
        "checkpoint_or_lambda_selection": False,
        "environment": {
            "grid_rows": 5,
            "grid_cols": 5,
            "number_blocks": EXPECTED_BLOCKS,
            "arrival_rate_lambda": 10.0,
            "exponential_mean_interarrival": 0.1,
            "processing_time_poisson_mean": 80,
            "max_steps": 2000,
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
        "primary_estimand": (
            "within-instance equal mean over three fixed model seeds, then "
            "paired EpisodeInstance differences lambda0025-minus-lambda0"
        ),
        "primary_gate": {
            "all_180_rows_strict_safe_complete": True,
            "familywise_alpha": 0.05,
            "endpoint_count": 3,
            "df": 29,
            "one_sided_bonferroni_t": SIMULTANEOUS_ONE_SIDED_T,
            "rehandles_delta_upper_bound_below": 0.0,
            "mae_delta_upper_bound_below": 2.0,
            "dense_return_delta_lower_bound_above": -20.0,
        },
        "nominal_two_sided_t_95": NOMINAL_TWO_SIDED_T,
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
        raise ConfirmationError("contract, parent artifacts, or sources changed")
    return observed


def _instance_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "episode-instances" / f"seed-{seed}.json"


def _new_environment():
    return final86._new_environment()


def open_panel(
    project_root: Path, output_dir: Path, *, confirmation: str
) -> dict:
    if confirmation != OPEN_CONFIRMATION:
        raise ConfirmationError(
            f"opening 88k requires --confirm {OPEN_CONFIRMATION}"
        )
    contract = authenticate_contract(project_root, output_dir)
    final86._configure_runtime()
    final86._assert_cuda()
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

    # A completed panel is authenticated and reused byte-for-byte.  In
    # particular, rerunning the wrapper does not resample the 30 instances.
    if (output_dir / INSTANCE_MANIFEST_NAME).is_file():
        return authenticate_instance_manifest(project_root, output_dir)

    env = _new_environment()
    records = []
    for seed in INSTANCE_SEEDS:
        path = _instance_path(output_dir, seed)
        sampled = env.sample_episode_instance(seed)
        if path.exists():
            observed = final86.EpisodeInstance.from_json(
                path.read_text(encoding="utf-8")
            )
            if observed != sampled:
                raise ConfirmationError(f"serialized instance changed: {seed}")
        else:
            _atomic_text(path, sampled.to_json())
            observed = final86.EpisodeInstance.from_json(
                path.read_text(encoding="utf-8")
            )
            if observed != sampled:
                raise ConfirmationError(f"instance serialization failed: {seed}")
        observed.validate_for(env)
        raw = path.read_bytes()
        records.append(
            {
                "instance_index": seed - INSTANCE_SEEDS[0],
                "instance_seed": seed,
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
            "all_rows_load_these_exact_serialized_instances": True,
        },
        "manifest_sha256",
    )
    path = output_dir / INSTANCE_MANIFEST_NAME
    if path.exists():
        if _load_json(path, label="instance manifest") != manifest:
            raise ConfirmationError("instance manifest changed")
    else:
        _atomic_json(path, manifest)
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
        or not isinstance(records, list)
        or len(records) != len(INSTANCE_SEEDS)
        or [record.get("instance_seed") for record in records]
        != list(INSTANCE_SEEDS)
    ):
        raise ConfirmationError("instance manifest grid changed")
    env = _new_environment()
    expected_paths = set()
    for record in records:
        seed = int(record["instance_seed"])
        path = output_dir / str(record["relative_path"])
        expected_paths.add(path.absolute())
        if path != _instance_path(output_dir, seed):
            raise ConfirmationError("instance path changed")
        raw = path.read_bytes()
        if _sha256_bytes(raw) != record["raw_sha256"]:
            raise ConfirmationError(f"instance bytes changed: {seed}")
        instance = final86.EpisodeInstance.from_json(raw.decode("utf-8"))
        instance.validate_for(env)
        if (
            instance.seed != seed
            or instance.instance_id != record["episode_instance_id"]
            or instance.schedule_id != record["schedule_id"]
            or _sha256_bytes(instance.to_json().encode())
            != record["canonical_sha256"]
        ):
            raise ConfirmationError(f"instance identity changed: {seed}")
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


def _ledger_path(output_dir: Path, arm: str, model_seed: int, instance_seed: int) -> Path:
    return (
        output_dir
        / "run-ledger"
        / arm
        / f"seed-{model_seed}"
        / f"instance-{instance_seed}.json"
    )


def _load_bound_cost(project_root: Path, arm, *, device, config):
    path = _paths(project_root)["cost_heads"][int(arm.model_seed)]
    if _sha256_file(path) != COST_HEAD_SHA256[int(arm.model_seed)]:
        raise ConfirmationError("cost head changed before evaluation")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return load_cost_checkpoint(
        payload,
        config=config,
        device=device,
        expected_source_checkpoint_sha256=arm.checkpoint_sha256,
        expected_source_policy_digest=arm.deployment_policy_digest,
        expected_model_seed=int(arm.model_seed),
    )


def _validate_ledger(
    ledger: Mapping,
    *,
    contract: Mapping,
    manifest: Mapping,
    arm_name: str,
    model_seed: int,
    record: Mapping,
) -> dict:
    _verify_hash(ledger, "ledger_sha256", label="run ledger")
    expected = {
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "arm": arm_name,
        "handling_lambda": ARM_LAMBDAS[arm_name],
        "model_seed": model_seed,
        "instance_seed": record["instance_seed"],
        "episode_instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        "checkpoint_sha256": SELECTED_CHECKPOINT_SHA256[model_seed],
        "cost_head_sha256": (
            None if arm_name == "lambda0" else COST_HEAD_SHA256[model_seed]
        ),
    }
    for name, value in expected.items():
        if ledger.get(name) != value:
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
        raise ConfirmationError("confirmation requires CUDA, matching VCG 1.1 deployment")
    final86._configure_runtime()
    arms, _ = _authenticate_inputs(project_root)
    completed = 0
    for arm_name in ARMS:
        for model_seed in MODEL_SEEDS:
            arm = arms[model_seed]
            for record in manifest["instances"]:
                path = _ledger_path(
                    output_dir, arm_name, model_seed, int(record["instance_seed"])
                )
                if path.exists():
                    _validate_ledger(
                        _load_json(path, label="run ledger"),
                        contract=contract,
                        manifest=manifest,
                        arm_name=arm_name,
                        model_seed=model_seed,
                        record=record,
                    )
                    completed += 1
                    continue
                instance = _load_instance(output_dir, record)
                try:
                    if arm_name == "lambda0":
                        raw = pilot._run_raw(arm, instance, device=device)
                    else:
                        def wrapper_factory(base):
                            cost = _load_bound_cost(
                                project_root,
                                arm,
                                device=device,
                                config=base.config,
                            )
                            return HandlingAugmentedV11Agent(
                                base,
                                cost,
                                handling_lambda=ARM_LAMBDAS[arm_name],
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
                            "arm": arm_name,
                            "handling_lambda": ARM_LAMBDAS[arm_name],
                            "model_seed": model_seed,
                            "required_deliveries": EXPECTED_BLOCKS,
                            "evaluation_learning": False,
                            "cost_head_forward_calls": (
                                0 if arm_name == "lambda0" else None
                            ),
                        }
                    )
                except Exception as exc:
                    row = {
                        "arm": arm_name,
                        "handling_lambda": ARM_LAMBDAS[arm_name],
                        "model_seed": model_seed,
                        "instance_seed": int(record["instance_seed"]),
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
                            "arm": arm_name,
                            "handling_lambda": ARM_LAMBDAS[arm_name],
                            "model_seed": model_seed,
                            "instance_seed": int(record["instance_seed"]),
                            "episode_instance_id": record["episode_instance_id"],
                            "schedule_id": record["schedule_id"],
                            "checkpoint_sha256": arm.checkpoint_sha256,
                            "cost_head_sha256": (
                                None
                                if arm_name == "lambda0"
                                else COST_HEAD_SHA256[model_seed]
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
                        "arm": arm_name,
                        "handling_lambda": ARM_LAMBDAS[arm_name],
                        "model_seed": model_seed,
                        "instance_seed": int(record["instance_seed"]),
                        "episode_instance_id": record["episode_instance_id"],
                        "schedule_id": record["schedule_id"],
                        "checkpoint_sha256": arm.checkpoint_sha256,
                        "cost_head_sha256": (
                            None
                            if arm_name == "lambda0"
                            else COST_HEAD_SHA256[model_seed]
                        ),
                        "row": row,
                    },
                    "ledger_sha256",
                )
                _atomic_json(path, ledger)
                completed += 1
    return {"status": "evaluation_complete", "completed_rows": completed}


def _mean(rows: Sequence[Mapping], name: str) -> float:
    return float(fmean(float(row[name]) for row in rows))


def _summary(rows: Sequence[Mapping]) -> dict:
    return {
        "dense_return": _mean(rows, "dense_return"),
        "mean_absolute_error": _mean(rows, "mean_absolute_error"),
        "steps": _mean(rows, "steps"),
        "total_physical_rehandles": sum(
            int(row["physical_rehandles"]) for row in rows
        ),
        "physical_rehandles_per_100": 100.0
        * sum(int(row["physical_rehandles"]) for row in rows)
        / (len(rows) * EXPECTED_BLOCKS),
    }


def _paired_interval(values: Sequence[float]) -> dict:
    if len(values) != len(INSTANCE_SEEDS):
        raise ConfirmationError("paired inference requires exactly 30 instances")
    mean = float(fmean(values))
    sd = float(stdev(values))
    se = sd / math.sqrt(len(values))
    return {
        "mean": mean,
        "standard_deviation": sd,
        "standard_error": se,
        "nominal_two_sided_95_ci": [
            mean - NOMINAL_TWO_SIDED_T * se,
            mean + NOMINAL_TWO_SIDED_T * se,
        ],
        "simultaneous_one_sided_lower_bound": (
            mean - SIMULTANEOUS_ONE_SIDED_T * se
        ),
        "simultaneous_one_sided_upper_bound": (
            mean + SIMULTANEOUS_ONE_SIDED_T * se
        ),
    }


def _read_all_rows(
    project_root: Path, output_dir: Path, contract: Mapping, manifest: Mapping
) -> dict[str, dict[int, dict[int, dict]]]:
    result = {arm: {seed: {} for seed in MODEL_SEEDS} for arm in ARMS}
    for arm_name in ARMS:
        for model_seed in MODEL_SEEDS:
            for record in manifest["instances"]:
                path = _ledger_path(
                    output_dir, arm_name, model_seed, int(record["instance_seed"])
                )
                if not path.is_file():
                    raise ConfirmationError(f"missing run ledger: {path}")
                row = _validate_ledger(
                    _load_json(path, label="run ledger"),
                    contract=contract,
                    manifest=manifest,
                    arm_name=arm_name,
                    model_seed=model_seed,
                    record=record,
                )
                result[arm_name][model_seed][int(record["instance_seed"])] = row
    return result


def analyze(project_root: Path, output_dir: Path) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    manifest = authenticate_instance_manifest(project_root, output_dir)
    rows = _read_all_rows(project_root, output_dir, contract, manifest)
    all_rows = [
        rows[arm][seed][instance]
        for arm in ARMS
        for seed in MODEL_SEEDS
        for instance in INSTANCE_SEEDS
    ]
    all_safe = len(all_rows) == EXPECTED_ROWS and all(
        row.get("strict_safe_complete") is True for row in all_rows
    )
    if not all_safe:
        report = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "failed_incomplete_or_unsafe",
            "whole_method_metrics_suppressed": True,
            "expected_rows": EXPECTED_ROWS,
            "observed_rows": len(all_rows),
            "strict_safe_rows": sum(
                row.get("strict_safe_complete") is True for row in all_rows
            ),
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
        }
        report = _with_hash(report, "report_sha256")
        _atomic_json(output_dir / REPORT_NAME, report)
        return report

    arm_summaries = {
        arm: _summary(
            [
                rows[arm][seed][instance]
                for seed in MODEL_SEEDS
                for instance in INSTANCE_SEEDS
            ]
        )
        for arm in ARMS
    }
    seed_summaries = {
        arm: {
            str(seed): _summary(
                [rows[arm][seed][instance] for instance in INSTANCE_SEEDS]
            )
            for seed in MODEL_SEEDS
        }
        for arm in ARMS
    }
    instance_points = []
    differences = {
        "dense_return": [],
        "mean_absolute_error": [],
        "physical_rehandles_per_100": [],
        "steps": [],
    }
    for instance in INSTANCE_SEEDS:
        point = {"instance_seed": instance}
        for arm in ARMS:
            cluster = [rows[arm][seed][instance] for seed in MODEL_SEEDS]
            point[arm] = {
                "dense_return": _mean(cluster, "dense_return"),
                "mean_absolute_error": _mean(cluster, "mean_absolute_error"),
                "steps": _mean(cluster, "steps"),
                "physical_rehandles_per_100": _mean(
                    cluster, "physical_rehandles_per_100"
                ),
            }
        point["difference_lambda0025_minus_lambda0"] = {
            name: point["lambda0025"][name] - point["lambda0"][name]
            for name in differences
        }
        for name in differences:
            differences[name].append(
                point["difference_lambda0025_minus_lambda0"][name]
            )
        instance_points.append(point)

    paired = {name: _paired_interval(values) for name, values in differences.items()}
    conditions = {
        "all_180_rows_strict_safe_complete": all_safe,
        "lambda0_zero_cost_head_calls": all(
            rows["lambda0"][seed][instance].get("cost_head_forward_calls") == 0
            for seed in MODEL_SEEDS
            for instance in INSTANCE_SEEDS
        ),
        "rehandles_superiority": (
            paired["physical_rehandles_per_100"][
                "simultaneous_one_sided_upper_bound"
            ]
            < 0.0
        ),
        "mae_noninferiority_margin_2": (
            paired["mean_absolute_error"][
                "simultaneous_one_sided_upper_bound"
            ]
            < 2.0
        ),
        "dense_return_noninferiority_margin_20": (
            paired["dense_return"]["simultaneous_one_sided_lower_bound"]
            > -20.0
        ),
        "frozen_input_hashes_unchanged": (
            _contract(project_root, output_dir) == contract
        ),
    }
    passed = all(conditions.values())
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "passed" if passed else "failed",
            "scope": "unseen_88k_frozen_policy_confirmation",
            "training_or_learning": False,
            "selected_lambda": ARM_LAMBDAS["lambda0025"],
            "row_count": len(all_rows),
            "strict_safe_row_count": len(all_rows),
            "arm_summaries": arm_summaries,
            "seed_summaries": seed_summaries,
            "instance_points": instance_points,
            "paired_differences_lambda0025_minus_lambda0": paired,
            "confirmation_gate": {**conditions, "passed": passed},
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "interpretation": (
                "confirmation of a fixed VCG 1.1 controller ensemble with an "
                "optional frozen handling-cost augmentation on unseen instances"
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
        else root / "results/vcg-v1-1-nested-handling-confirmation-88k"
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
    if "instance_points" in result:
        display = {
            "status": result["status"],
            "arm_summaries": result["arm_summaries"],
            "paired_differences": result[
                "paired_differences_lambda0025_minus_lambda0"
            ],
            "confirmation_gate": result["confirmation_gate"],
            "report": str(output / REPORT_NAME),
        }
    print(json.dumps(display, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
