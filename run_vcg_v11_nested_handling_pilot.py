#!/usr/bin/env python3
"""Development pilot for VCG 1.1 plus a strictly nested handling-cost head.

The experiment does not retrain the operational controller.  It authenticates
the frozen VCG 1.1 seed-0 checkpoint, learns a detached handling predictor from
that checkpoint's completed historical replay, proves that ``lambda=0`` is an
exact delegation to VCG 1.1, and evaluates a small fixed-lambda sweep on the
already-opened 85000--85011 EpisodeInstances.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from types import MethodType
from typing import Mapping, Sequence
from unittest.mock import patch

import numpy as np
import torch

import benchmark_viability_critic_priority as benchmark
import compare_vcg_dense_pareto as v11
import compare_vcg_v2_3_matched_baselines as matched
import compare_viability_graph_baselines as normalized
import run_vcg_final86_four_method as final86

from vcg_v11_nested_handling import (
    DetachedHandlingCostNetwork,
    HandlingAugmentedV11Agent,
    cache_cost_dataset,
    cost_checkpoint,
    fit_cached_cost_network,
    load_cost_checkpoint,
)
from vcg_v11_replay_cost_dataset import build_replay_cost_dataset


PROTOCOL = "vcg_v1_1_strictly_nested_handling_seed0_85k_development_v1"
SCHEMA_VERSION = 1
MODEL_SEED = 0
SELECTED_EPISODE = 500
INSTANCE_SEEDS = tuple(range(85_000, 85_012))
PARITY_INSTANCE_SEED = 85_000
LAMBDA_GRID = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20)
LATEST_SHA256 = (
    "b31a24ce518ccd0f8782f9ae51b6fceeb82f9073d6a32ed6702e9b54c018cf5b"
)
COST_SEED = 81_100_000
SPLIT_SEED = 81_100_001
VALIDATION_FRACTION = 0.20
COST_EPOCHS = 12
COST_BATCH_SIZE = 256
COST_LEARNING_RATE = 1e-3
EXPECTED_BLOCKS = 8
_AUTH_CACHE = {}

CONTRACT_NAME = "pilot-contract.json"
PARITY_NAME = "lambda-zero-parity.json"
COST_NAME = "handling-cost-head.pth"
COST_REPORT_NAME = "handling-cost-training.json"
SWEEP_NAME = "pilot-sweep.json"


class PilotError(RuntimeError):
    pass


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise PilotError(f"state[{name!r}] is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(_canonical_json(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _runtime() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }


def _paths(project_root: Path) -> dict:
    project_root = project_root.resolve()
    return {
        "latest": project_root
        / "results/vcg-dense-v1-1-seed0-500ep/latest.pth",
        "v23_source": project_root
        / "results/vcg-constrained-v2-3-gamma1-ablation-seed10-200ep",
        "v11_control": project_root
        / "results/vcg-dense-v1-1-v2-2-panel-control-12instance",
        "v22_source": project_root
        / "results/vcg-constrained-v2-2-development-seed10-200ep",
    }


def _authenticate_inputs(project_root: Path):
    cache_key = str(project_root.resolve())
    if cache_key in _AUTH_CACHE:
        return _AUTH_CACHE[cache_key]
    arms, records = final86._authenticate_v11()
    selected = [arm for arm in arms if arm.model_seed == MODEL_SEED]
    if len(selected) != 1:
        raise PilotError("could not resolve exact VCG 1.1 seed-0 selected checkpoint")
    arm = selected[0]
    if arm.checkpoint_weight_episode != SELECTED_EPISODE:
        raise PilotError("VCG 1.1 selected episode drifted")
    paths = _paths(project_root)
    latest_data = paths["latest"].read_bytes()
    if _sha256_bytes(latest_data) != LATEST_SHA256:
        raise PilotError("VCG 1.1 seed-0 latest checkpoint raw SHA drifted")
    # The resumable replay contains immutable project dataclasses.  It is
    # deserialized only after the exact external byte hash above is matched.
    latest = torch.load(BytesIO(latest_data), map_location="cpu", weights_only=False)
    if not isinstance(latest, Mapping):
        raise PilotError("VCG 1.1 latest checkpoint is not a mapping")
    for name, expected in (
        ("model_seed", MODEL_SEED),
        ("selected_checkpoint_episode", SELECTED_EPISODE),
        ("completed_training_episodes", SELECTED_EPISODE),
    ):
        if latest.get(name) != expected:
            raise PilotError(f"VCG 1.1 latest checkpoint mismatch: {name}")
    best_q = _state_digest(arm.payload["agent_state"]["Q_local"])
    latest_q = _state_digest(latest["agent_state"]["Q_local"])
    if best_q != latest_q:
        raise PilotError("selected best and resumable latest Qop weights differ")
    sources = matched.authenticate_sources(
        v23_source_dir=paths["v23_source"],
        v11_control_dir=paths["v11_control"],
        v22_source_dir=paths["v22_source"],
    )
    if tuple(sorted(sources.instances)) != INSTANCE_SEEDS:
        raise PilotError("authenticated 85k instance grid drifted")
    result = (arm, latest, sources, best_q, records)
    _AUTH_CACHE[cache_key] = result
    return result


def _contract(project_root: Path) -> dict:
    arm, latest, sources, q_digest, records = _authenticate_inputs(project_root)
    source_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("vcg_v11_nested_handling.py").resolve(),
        Path(__file__).with_name("vcg_v11_replay_cost_dataset.py").resolve(),
    )
    identities = []
    for seed in INSTANCE_SEEDS:
        instance = sources.instances[seed]
        identities.append(
            {
                "instance_seed": seed,
                "episode_instance_id": instance.instance_id,
                "schedule_id": instance.schedule_id,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "development_only_already_opened_85000_85011",
        "no_new_environment_training": True,
        "final_86xxx_opened": False,
        "final_87xxx_used": False,
        "model_seed": MODEL_SEED,
        "selected_checkpoint_episode": SELECTED_EPISODE,
        "selected_checkpoint_sha256": arm.checkpoint_sha256,
        "selected_policy_digest": arm.deployment_policy_digest,
        "selected_qop_state_sha256": q_digest,
        "latest_checkpoint_sha256": LATEST_SHA256,
        "latest_replay_transition_count": len(
            latest["agent_state"]["replay"]["memory"]
        ),
        "lambda_grid": list(LAMBDA_GRID),
        "lambda_zero_semantics": "direct_original_vcg_1_1_selector_delegation",
        "cost_training": {
            "source": "stored_completed_vcg_1_1_behavior_replay",
            "label": "inclusive_remaining_physical_reconfigure_count",
            "split_seed": SPLIT_SEED,
            "cost_seed": COST_SEED,
            "validation_fraction": VALIDATION_FRACTION,
            "epochs": COST_EPOCHS,
            "batch_size": COST_BATCH_SIZE,
            "learning_rate": COST_LEARNING_RATE,
            "interpretation": (
                "frozen-policy Monte Carlo handling predictor; not a converged "
                "dual or lambda-induced-policy Bellman critic"
            ),
        },
        "instance_identities": identities,
        "parity_gate": {
            "structural_direct_delegation": True,
            "sentinel_instance_seed": PARITY_INSTANCE_SEED,
            "sentinel_behavior_digest_exact": True,
            "authenticated_lambda_zero_source_rows": len(INSTANCE_SEEDS),
            "qop_state_unchanged": True,
            "lambda_zero_cost_forward_calls": 0,
        },
        "pilot_advancement_gate": {
            "all_12_rows_strict_safe_complete": True,
            "minimum_total_rehandles_saved": 2,
            "maximum_mae_increase": 2.0,
            "maximum_dense_return_decrease": 20.0,
            "selection": "smallest_positive_lambda_meeting_every_condition",
        },
        "source_sha256": {
            path.name: _sha256_file(path) for path in source_paths
        },
        "runtime_at_prepare": _runtime(),
    }


def prepare(project_root: Path, output_dir: Path) -> dict:
    value = _contract(project_root)
    path = output_dir / CONTRACT_NAME
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise PilotError("existing pilot contract does not match current inputs")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise PilotError("pilot output directory is nonempty without its contract")
        _atomic_json(path, value)
    return {
        "status": "prepared",
        "contract": str(path),
        "lambda_grid": list(LAMBDA_GRID),
        "new_environment_training": False,
    }


def _require_contract(project_root: Path, output_dir: Path) -> dict:
    path = output_dir / CONTRACT_NAME
    if not path.is_file():
        raise PilotError("run prepare first")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = _contract(project_root)
    if observed != expected:
        raise PilotError("pilot contract or its bound inputs changed")
    return observed


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PilotError("CUDA was requested but is unavailable")
    return device


def _fresh_base(arm, device: torch.device):
    return benchmark._freeze_agent(
        arm.payload, device=device, seed=int(arm.model_seed)
    )


@contextmanager
def _agent_factory(factory):
    original = benchmark._freeze_agent

    def replacement(payload, *, device, seed):
        return factory(original(payload, device=device, seed=seed))

    with patch.object(benchmark, "_freeze_agent", replacement):
        yield


def _run_raw(arm, instance, *, device: torch.device, wrapper_factory=None) -> dict:
    kwargs = {
        "arm": benchmark.EXACT_FULL,
        "controller_payload": arm.payload,
        "instance": instance,
        "instance_seed": int(instance.seed),
        "search_config": benchmark._search_config(arm.payload),
        "liveness_rule": benchmark._liveness_rule(arm.payload),
        "prioritizer": None,
        "max_steps": v11.MAX_STEPS,
        "device": device,
    }
    if wrapper_factory is None:
        return benchmark.run_arm(**kwargs)
    with _agent_factory(wrapper_factory):
        return benchmark.run_arm(**kwargs)


def _compact_row(raw: Mapping, instance) -> dict:
    row = normalized._normalize_vcg(
        dict(raw), instance, EXPECTED_BLOCKS, v11.FROZEN_OBJECTIVE_SPEC
    )
    return {
        "instance_seed": int(instance.seed),
        "episode_instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "behavior_digest": raw["behavior_digest"],
        "strict_safe_complete": bool(
            raw["strict_method_success"]
            and raw["terminal"]
            and raw["method_failure_reason"] is None
            and raw["complete_frontier_exactly_verified"]
            and raw["illegal_drops"] == 0
            and raw["macro_failures"] == 0
            and len(raw["delivery_deviations"]) == EXPECTED_BLOCKS
        ),
        "dense_return": float(row["dense_rescored_return"]),
        "mean_absolute_error": float(row["mean_absolute_error"]),
        "steps": int(row["steps"]),
        "physical_rehandles": int(row["physical_storage_relocations"]),
        "physical_rehandles_per_100": float(
            100.0 * row["physical_storage_relocations"] / EXPECTED_BLOCKS
        ),
        "delivery_deviations": list(raw["delivery_deviations"]),
    }


def _authenticated_lambda_zero_rows(sources) -> list[dict]:
    rows = [
        row for row in sources.v11_rows if int(row.get("model_seed", -1)) == MODEL_SEED
    ]
    rows.sort(key=lambda row: int(row["instance_seed"]))
    if [int(row["instance_seed"]) for row in rows] != list(INSTANCE_SEEDS):
        raise PilotError("authenticated seed-0 VCG 1.1 row grid is incomplete")
    output = []
    for row in rows:
        physical = int(row["physical_storage_relocations"])
        strict = bool(
            row["strict_method_success"] == 1.0
            and row["completion_rate"] == 1.0
            and row["method_failure_reason"] is None
            and row["illegal_drops"] == 0
            and row["fallbacks"] == 0
        )
        if not strict:
            raise PilotError(
                f"authenticated VCG 1.1 row {row['instance_seed']} is not strict"
            )
        output.append(
            {
                "instance_seed": int(row["instance_seed"]),
                "episode_instance_id": row["instance_id"],
                "schedule_id": row["schedule_id"],
                "behavior_digest": None,
                "strict_safe_complete": True,
                "dense_return": float(row["dense_rescored_return"]),
                "mean_absolute_error": float(row["mean_absolute_error"]),
                "steps": int(row["steps"]),
                "physical_rehandles": physical,
                "physical_rehandles_per_100": float(
                    100.0 * physical / EXPECTED_BLOCKS
                ),
                "delivery_deviations": list(row["delivery_deviations"]),
                "execution_reused": True,
                "source_protocol": row["protocol"],
            }
        )
    return output


def parity(project_root: Path, output_dir: Path, *, device_name: str) -> dict:
    _require_contract(project_root, output_dir)
    arm, _latest, sources, q_digest, _records = _authenticate_inputs(project_root)
    device = _device(device_name)
    source_rows = _authenticated_lambda_zero_rows(sources)
    instance = sources.instances[PARITY_INSTANCE_SEED]
    direct = _run_raw(arm, instance, device=device)

    def wrapper_factory(base):
        cost = DetachedHandlingCostNetwork(base.config, seed=COST_SEED)
        wrapper = HandlingAugmentedV11Agent(
            base, cost, handling_lambda=0.0
        )

        def forbidden_forward(self, _features):
            raise AssertionError("lambda=0 called the handling-cost head")

        wrapper.cost_network.forward = MethodType(
            forbidden_forward, wrapper.cost_network
        )
        return wrapper

    nested = _run_raw(
        arm, instance, device=device, wrapper_factory=wrapper_factory
    )
    exact_fields = (
        "behavior_digest",
        "return",
        "steps",
        "terminal",
        "success",
        "strict_method_success",
        "method_failure_reason",
        "delivery_deviations",
        "relocations",
        "illegal_drops",
        "macro_failures",
    )
    mismatches = [
        name for name in exact_fields if direct.get(name) != nested.get(name)
    ]
    if mismatches:
        raise PilotError(
            f"lambda=0 parity failed for {PARITY_INSTANCE_SEED}: {mismatches}"
        )
    compact = _compact_row(direct, instance)
    source = source_rows[0]
    for name in (
        "dense_return",
        "mean_absolute_error",
        "steps",
        "physical_rehandles",
        "delivery_deviations",
    ):
        if compact[name] != source[name]:
            raise PilotError(f"sentinel rollout/source ledger mismatch: {name}")
    if not compact["strict_safe_complete"]:
        raise PilotError("direct VCG 1.1 sentinel row is not strict-safe-complete")
    fresh = _fresh_base(arm, device)
    after_digest = _state_digest(fresh.Q_local.state_dict())
    if after_digest != q_digest:
        raise PilotError("Qop state digest changed during parity evaluation")
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "passed",
        "device": str(device),
        "sentinel_pair_count": 1,
        "sentinel_instance_seed": PARITY_INSTANCE_SEED,
        "sentinel_behavior_digest_exact": True,
        "sentinel_terminal_metrics_exact": True,
        "authenticated_lambda_zero_row_count": len(source_rows),
        "lambda_zero_cost_forward_calls": 0,
        "qop_state_before_sha256": q_digest,
        "qop_state_after_sha256": after_digest,
        "sentinel": compact,
        "rows": source_rows,
    }
    _atomic_json(output_dir / PARITY_NAME, result)
    return result


def train_cost(project_root: Path, output_dir: Path, *, device_name: str) -> dict:
    _require_contract(project_root, output_dir)
    arm, latest, _sources, q_digest, _records = _authenticate_inputs(project_root)
    device = _device(device_name)
    dataset = build_replay_cost_dataset(
        latest,
        split_seed=SPLIT_SEED,
        validation_fraction=VALIDATION_FRACTION,
        selected_episode_limit=SELECTED_EPISODE,
    )
    base = _fresh_base(arm, device)
    before_local = _state_digest(base.Q_local.state_dict())
    before_target = _state_digest(base.Q_target.state_dict())
    if before_local != q_digest:
        raise PilotError("loaded base Qop does not match selected checkpoint")
    train_data = cache_cost_dataset(
        base.Q_local, dataset.train_samples, batch_size=COST_BATCH_SIZE
    )
    validation_data = cache_cost_dataset(
        base.Q_local, dataset.validation_samples, batch_size=COST_BATCH_SIZE
    )
    network = DetachedHandlingCostNetwork(base.config, seed=COST_SEED).to(device)
    initial_cost_digest = _state_digest(network.state_dict())
    training = fit_cached_cost_network(
        network,
        train_data,
        validation_data,
        epochs=COST_EPOCHS,
        batch_size=COST_BATCH_SIZE,
        learning_rate=COST_LEARNING_RATE,
        seed=COST_SEED,
    )
    final_cost_digest = _state_digest(network.state_dict())
    if final_cost_digest == initial_cost_digest:
        raise PilotError("handling-cost weights did not change")
    after_local = _state_digest(base.Q_local.state_dict())
    after_target = _state_digest(base.Q_target.state_dict())
    if (after_local, after_target) != (before_local, before_target):
        raise PilotError("cost fitting changed frozen operational weights")
    record = {
        "dataset": dataset.audit_dict(),
        "fit": training,
        "qop_local_before_sha256": before_local,
        "qop_local_after_sha256": after_local,
        "qop_target_before_sha256": before_target,
        "qop_target_after_sha256": after_target,
        "cost_initial_sha256": initial_cost_digest,
        "cost_final_sha256": final_cost_digest,
        "device": str(device),
    }
    payload = cost_checkpoint(
        network,
        source_checkpoint_sha256=arm.checkpoint_sha256,
        source_policy_digest=arm.deployment_policy_digest,
        model_seed=MODEL_SEED,
        training_record=record,
    )
    _atomic_torch(output_dir / COST_NAME, payload)
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        **record,
        "cost_checkpoint_sha256": _sha256_file(output_dir / COST_NAME),
    }
    _atomic_json(output_dir / COST_REPORT_NAME, report)
    return report


def _load_bound_cost(path: Path, arm, *, device: torch.device, config):
    if not path.is_file():
        raise PilotError("run train-cost first")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return load_cost_checkpoint(
        payload,
        config=config,
        device=device,
        expected_source_checkpoint_sha256=arm.checkpoint_sha256,
        expected_source_policy_digest=arm.deployment_policy_digest,
        expected_model_seed=MODEL_SEED,
    )


def _mean(rows: Sequence[Mapping], name: str) -> float:
    return float(sum(float(row[name]) for row in rows) / len(rows))


def _summary(rows: Sequence[Mapping]) -> dict:
    if len(rows) != len(INSTANCE_SEEDS) or not all(
        row["strict_safe_complete"] for row in rows
    ):
        return {"whole_method_eligible": False, "metrics": None}
    return {
        "whole_method_eligible": True,
        "metrics": {
            "mean_dense_return": _mean(rows, "dense_return"),
            "mean_absolute_error": _mean(rows, "mean_absolute_error"),
            "mean_steps": _mean(rows, "steps"),
            "total_physical_rehandles": int(
                sum(int(row["physical_rehandles"]) for row in rows)
            ),
            "physical_rehandles_per_100": float(
                100.0
                * sum(int(row["physical_rehandles"]) for row in rows)
                / (len(rows) * EXPECTED_BLOCKS)
            ),
        },
    }


def sweep(project_root: Path, output_dir: Path, *, device_name: str) -> dict:
    _require_contract(project_root, output_dir)
    parity_path = output_dir / PARITY_NAME
    if not parity_path.is_file():
        raise PilotError("run parity first")
    parity_result = json.loads(parity_path.read_text(encoding="utf-8"))
    if parity_result.get("status") != "passed":
        raise PilotError("lambda-zero parity did not pass")
    arm, _latest, sources, _q_digest, _records = _authenticate_inputs(project_root)
    device = _device(device_name)
    probe_base = _fresh_base(arm, device)
    cost_path = output_dir / COST_NAME
    # Authenticate checkpoint/base binding once, then load a fresh head per row.
    _load_bound_cost(cost_path, arm, device=device, config=probe_base.config)

    lambda_rows = {"0.0": parity_result["rows"]}
    summaries = {"0.0": _summary(parity_result["rows"])}
    for value in LAMBDA_GRID[1:]:
        rows = []
        for seed in INSTANCE_SEEDS:
            instance = sources.instances[seed]

            def wrapper_factory(base, value=value):
                cost = _load_bound_cost(
                    cost_path, arm, device=device, config=base.config
                )
                return HandlingAugmentedV11Agent(
                    base, cost, handling_lambda=value
                )

            raw = _run_raw(
                arm, instance, device=device, wrapper_factory=wrapper_factory
            )
            rows.append(_compact_row(raw, instance))
        key = str(value)
        lambda_rows[key] = rows
        summaries[key] = _summary(rows)

    base_metrics = summaries["0.0"]["metrics"]
    qualifying = []
    for value in LAMBDA_GRID[1:]:
        key = str(value)
        summary = summaries[key]
        metrics = summary["metrics"]
        conditions = {
            "whole_method_eligible": summary["whole_method_eligible"],
            "saves_at_least_two_rehandles": bool(
                metrics is not None
                and base_metrics["total_physical_rehandles"]
                - metrics["total_physical_rehandles"]
                >= 2
            ),
            "mae_increase_at_most_two": bool(
                metrics is not None
                and metrics["mean_absolute_error"]
                <= base_metrics["mean_absolute_error"] + 2.0
            ),
            "dense_return_decrease_at_most_twenty": bool(
                metrics is not None
                and metrics["mean_dense_return"]
                >= base_metrics["mean_dense_return"] - 20.0
            ),
        }
        passed = all(conditions.values())
        summaries[key]["pilot_gate"] = {**conditions, "passed": passed}
        if passed:
            qualifying.append(value)
    selected = min(qualifying) if qualifying else None
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": "development_only",
        "interpretation": (
            "strictly nested frozen VCG 1.1 with an offline Monte Carlo "
            "handling predictor; not a converged active-budget result"
        ),
        "lambda_grid": list(LAMBDA_GRID),
        "lambda_zero_reused_from_exact_parity": True,
        "summaries": summaries,
        "selected_smallest_qualifying_lambda": selected,
        "advance_to_seeds_1_and_2": selected is not None,
        "rows": lambda_rows,
    }
    _atomic_json(output_dir / SWEEP_NAME, result)
    return result


def _parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "parity", "train-cost", "sweep", "run-all")
    )
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main() -> None:
    args = _parser().parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root
        / "results/vcg-v1-1-nested-handling-seed0-85k-development"
    )
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(project_root, output_dir)
    elif args.command == "parity":
        result = parity(project_root, output_dir, device_name=args.device)
    elif args.command == "train-cost":
        result = train_cost(project_root, output_dir, device_name=args.device)
    elif args.command == "sweep":
        result = sweep(project_root, output_dir, device_name=args.device)
    else:
        result = {
            "prepare": prepare(project_root, output_dir),
            "parity": parity(project_root, output_dir, device_name=args.device),
            "train_cost": train_cost(
                project_root, output_dir, device_name=args.device
            ),
            "sweep": sweep(project_root, output_dir, device_name=args.device),
        }
    if args.command == "prepare":
        display = result
    elif args.command == "parity":
        display = {
            "status": result["status"],
            "sentinel_instance_seed": result["sentinel_instance_seed"],
            "sentinel_behavior_digest_exact": result[
                "sentinel_behavior_digest_exact"
            ],
            "lambda_zero_cost_forward_calls": result[
                "lambda_zero_cost_forward_calls"
            ],
            "report": str(output_dir / PARITY_NAME),
        }
    elif args.command == "train-cost":
        display = {
            "status": result["status"],
            "training_samples": result["dataset"]["train_sample_count"],
            "validation_samples": result["dataset"][
                "validation_sample_count"
            ],
            "selected_epoch": result["fit"]["selected_epoch"],
            "validation_mae_rehandles": result["fit"]["validation"][
                "mae_rehandles"
            ],
            "report": str(output_dir / COST_REPORT_NAME),
        }
    elif args.command == "sweep":
        display = {
            "status": result["status"],
            "selected_smallest_qualifying_lambda": result[
                "selected_smallest_qualifying_lambda"
            ],
            "advance_to_seeds_1_and_2": result["advance_to_seeds_1_and_2"],
            "summaries": result["summaries"],
            "report": str(output_dir / SWEEP_NAME),
        }
    else:
        display = {
            "status": "complete",
            "parity_passed": result["parity"]["status"] == "passed",
            "cost_validation_mae_rehandles": result["train_cost"]["fit"][
                "validation"
            ]["mae_rehandles"],
            "selected_smallest_qualifying_lambda": result["sweep"][
                "selected_smallest_qualifying_lambda"
            ],
            "advance_to_seeds_1_and_2": result["sweep"][
                "advance_to_seeds_1_and_2"
            ],
            "report": str(output_dir / SWEEP_NAME),
        }
    print(json.dumps(display, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
