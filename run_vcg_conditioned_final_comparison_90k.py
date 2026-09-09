#!/usr/bin/env python3
"""Prospective final comparison of conditioned VCG on a sealed 90k panel.

The contract freezes all policies, lambda coordinates, nuisance replications,
and analysis rules before the 30 EpisodeInstances are materialized.  Every arm
is ledgered independently and is resumable.  A failed cell is retained and is
never silently redrawn or removed from whole-method aggregation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import fmean, stdev
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

import final86_v23_runtime as v23_adapter
import run_kim2020_final86_comparison as kim_final
import run_vcg_final86_four_method as final86
import run_vcg_v11_conditioned_handling_damped_evaluation_85k as seed0_eval
import run_vcg_v11_conditioned_handling_seed0_85k as seed0_parent
import run_vcg_v11_conditioned_handling_two_phase_seeds12 as seeds12
import run_vcg_v11_nested_handling_pilot as pilot
import train_vcg_v11_conditioned_handling_iterative as conditioned_training
import train_vcg_v11_conditioned_handling_seed1_convergence_continuation as seed1_cont
import vcg_v2_3_kim2020_supplement_evaluation as kim85
from vcg_v11_conditioned_handling import ConditionedHandlingAgent


PROTOCOL = "vcg_conditioned_final_matched_comparison_90k_cpu_v3"
KIM_PROTOCOL = PROTOCOL + "_kim2020"
SCHEMA_VERSION = 1
MODEL_SEEDS = (0, 1, 2)
INSTANCE_SEEDS = tuple(range(90_000, 90_030))
LAMBDA_GRID = (0.0, *conditioned_training.PROBE_LAMBDAS)
EXPECTED_BLOCKS = 8
MAX_STEPS = 2_000
TARGET_WINDOW = 20.0

CONDITIONED_METHOD = "vcg_conditioned"
V23_METHOD = final86.V23_METHOD
DYNAMIC_METHOD = final86.DYNAMIC_METHOD
GA_METHOD = final86.GA_METHOD
KIM_METHOD = kim85.KIM_STOCHASTIC
BASELINE_METHODS = (V23_METHOD, DYNAMIC_METHOD, GA_METHOD, KIM_METHOD)

# Preserve the action-RNG grid of the authenticated frozen VCG 2.3 final
# adapter.  Reusing random streams by panel position is intentional; the 90k
# EpisodeInstances and the enclosing contract are new and disjoint.
V23_POLICY_RNG_BASE = 622_000_000
GA_RNG_BASE = 635_000_000
KIM_POLICY_RNG_BASE = 636_000_000

CONDITIONED_ROWS = len(MODEL_SEEDS) * len(INSTANCE_SEEDS) * len(LAMBDA_GRID)
V23_ROWS = len(final86.V23_MODEL_SEEDS) * len(final86.RNG_INDICES) * len(INSTANCE_SEEDS)
DYNAMIC_ROWS = len(INSTANCE_SEEDS)
GA_ROWS = len(final86.RNG_INDICES) * len(INSTANCE_SEEDS)
KIM_ROWS = len(kim_final.MODEL_SEEDS) * len(kim_final.ROLLOUTS) * len(INSTANCE_SEEDS)
EXPECTED_ROWS = CONDITIONED_ROWS + V23_ROWS + DYNAMIC_ROWS + GA_ROWS + KIM_ROWS

CONTRACT_NAME = "final90-contract.json"
ACTIVATION_NAME = "panel-activation.json"
INSTANCE_MANIFEST_NAME = "episode-instance-manifest.json"
REPORT_NAME = "final90-report.json"
OPEN_CONFIRMATION = "OPEN_CONDITIONED_FINAL_90K_CPU_V3"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "results/vcg-conditioned-final-comparison-90k-cpu-v3"
)
SEEDS12_RELATIVE = Path(
    "results/vcg-v1-1-conditioned-handling-seeds12-two-phase-development"
)

METRICS = (
    "dense_return",
    "mean_absolute_error",
    "mean_signed_deviation",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "steps",
    "physical_rehandles_per_100_required_deliveries",
)
NOMINAL_T_95 = 2.0452296421
_AUTH_CACHE: dict[str, dict] = {}


class Final90Error(RuntimeError):
    pass


def _canonical_bytes(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, *, hash_field: Optional[str] = None) -> str:
    payload = dict(value)
    if hash_field is not None:
        payload.pop(hash_field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256(path: Path) -> str:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise Final90Error(f"expected a canonical regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise Final90Error(f"missing canonical {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Final90Error(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise Final90Error(f"{label} must contain an object")
    return value


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _verify_hash(value: Mapping, field: str, *, label: str) -> None:
    if value.get(field) != _digest(value, hash_field=field):
        raise Final90Error(f"{label} self hash mismatch")


def _atomic_json(path: Path, value: Mapping) -> None:
    final86._atomic_json(path, value)


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


def _lambda_token(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _source_hashes(project_root: Path) -> dict[str, str]:
    sources = {
        "final90_runner": Path(__file__).resolve(),
        "conditioned_controller": project_root / "vcg_v11_conditioned_handling.py",
        "conditioned_seed0_loader": project_root
        / "run_vcg_v11_conditioned_handling_seed0_85k.py",
        "conditioned_seeds12_loader": project_root
        / "run_vcg_v11_conditioned_handling_two_phase_seeds12.py",
        "seed1_continuation": project_root
        / "train_vcg_v11_conditioned_handling_seed1_convergence_continuation.py",
        "pilot_executor": project_root / "run_vcg_v11_nested_handling_pilot.py",
        "baseline_executor": project_root / "run_vcg_final86_four_method.py",
        "v23_final_adapter": Path(v23_adapter.__file__).resolve(),
        "kim_executor": project_root / "run_kim2020_final86_comparison.py",
        "kim_evaluator": Path(kim85.__file__).resolve(),
        "episode_instance": project_root / "example/episode_instance.py",
        "environment": project_root / "example/small_rooms_env.py",
    }
    return {name: _sha256(path) for name, path in sorted(sources.items())}


def _software_runtime_contract() -> dict[str, Optional[str]]:
    """Return reproducibility-relevant software versions, not host state.

    CUDA visibility, GPU identity, environment variables, and the thread count
    observed in the calling shell are descriptive runtime provenance.  They
    must not change the identity of this explicitly CPU-only protocol.
    """

    return {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "torch": str(torch.__version__),
        "torch_cuda_build": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
    }


def _authenticate_kim_source_manifest(project_root: Path) -> dict[str, Any]:
    path = (
        project_root
        / "results/vcg-v2-3-kim2020-supplement-evaluation-85k"
        / "evaluation-run-manifest.json"
    )
    manifest = kim85._load_json(path, expected_type=dict)
    kim85._verify_self_hash(
        manifest, "manifest_sha256", label="Kim evaluation run manifest"
    )
    sources = manifest.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise Final90Error("Kim evaluation source registry is missing")
    observed = {}
    for relative, expected in sorted(sources.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise Final90Error("Kim evaluation source registry is malformed")
        current = _sha256(project_root / relative)
        if current != expected:
            raise Final90Error(f"Kim authenticated source changed: {relative}")
        observed[relative] = current
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": observed,
    }


def _cpu_runtime_provenance() -> dict[str, Any]:
    """Describe the active CPU execution context without probing CUDA."""

    return {
        **_software_runtime_contract(),
        "execution_device": "cpu",
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cuda_runtime_was_probed": False,
    }


def _authenticate_conditioned(project_root: Path) -> dict:
    seed0 = seed0_eval._damped_artifacts(project_root)
    seeds12_root = project_root / SEEDS12_RELATIVE
    seeds12_contract = seeds12._require_contract(
        project_root, seeds12_root, max_steps=MAX_STEPS
    )
    inputs = seeds12._authenticate_inputs(project_root)

    seed2_paths = seeds12._seed_paths(seeds12_root, 2)
    seed2 = torch.load(
        seed2_paths["terminal"], map_location="cpu", weights_only=False
    )
    seeds12._validate_checkpoint(
        seed2, seeds12_contract, seed=2, terminal=True
    )
    seed2_summary = _load_json(seed2_paths["summary"], label="seed-2 summary")
    if seed2_summary.get("stability_gate", {}).get("passed") is not True:
        raise Final90Error("seed 2 did not pass its training convergence gate")

    seed1_root = project_root / seed1_cont.OUTPUT_RELATIVE
    seed1_contract = seed1_cont._require_contract(
        project_root, seed1_root, max_steps=MAX_STEPS
    )
    seed1_artifacts = seed1_cont._parent_artifacts(
        project_root, max_steps=MAX_STEPS
    )
    seed1_bank_sha = _sha256(seed1_root / seed1_cont.BANK_NAME)
    seed1_path = seed1_root / seed1_cont.TERMINAL_NAME
    seed1 = torch.load(seed1_path, map_location="cpu", weights_only=False)
    seed1_cont._validate_checkpoint(
        seed1,
        seed1_contract,
        artifacts=seed1_artifacts,
        bank_sha256=seed1_bank_sha,
        terminal=True,
    )
    if seed1.get("stopping_assessment", {}).get("passed") is not True:
        raise Final90Error("seed 1 did not pass convergence continuation")

    if seed0["summary"].get("convergence_assessment", {}).get("passed") is not True:
        raise Final90Error("seed 0 did not pass its training convergence gate")
    terminals = {0: seed0["terminal"], 1: seed1, 2: seed2}
    terminal_paths = {
        0: seed0["root"] / "terminal.pth",
        1: seed1_path,
        2: seed2_paths["terminal"],
    }
    environments = {
        json.dumps(inputs["arms"][seed].payload["environment"], sort_keys=True)
        for seed in MODEL_SEEDS
    }
    if len(environments) != 1:
        raise Final90Error("conditioned model environments differ")
    return {
        "seed0": seed0,
        "seeds12_root": seeds12_root,
        "seeds12_contract": seeds12_contract,
        "inputs": inputs,
        "seed1_contract": seed1_contract,
        "seed1_artifacts": seed1_artifacts,
        "terminals": terminals,
        "terminal_paths": terminal_paths,
        "terminal_sha256": {
            seed: _sha256(path) for seed, path in terminal_paths.items()
        },
    }


def _kim_completions(project_root: Path) -> dict[int, dict]:
    completions = {}
    for seed in kim_final.MODEL_SEEDS:
        path = (
            kim_final.KIM85_ROOT
            / "training-completion"
            / f"seed-{seed}.json"
        )
        completion = kim85._load_json(path, expected_type=dict)
        kim85._verify_self_hash(
            completion, "completion_sha256", label=f"Kim seed {seed} completion"
        )
        checkpoint = (
            kim_final.TRAINING_ROOT / "training" / f"seed-{seed}" / "best.pth"
        )
        if kim85._sha256(checkpoint) != completion.get(
            "selected_checkpoint_raw_sha256"
        ):
            raise Final90Error(f"Kim seed-{seed} checkpoint changed")
        completions[seed] = completion
    return completions


def _authenticate_inputs(project_root: Path) -> dict:
    key = str(project_root.resolve())
    if key in _AUTH_CACHE:
        return _AUTH_CACHE[key]
    conditioned = _authenticate_conditioned(project_root)
    baseline = final86._authenticate_all()
    completions = _kim_completions(project_root)
    kim_sources = _authenticate_kim_source_manifest(project_root)
    result = {
        "conditioned": conditioned,
        "baseline": baseline,
        "kim_completions": completions,
        "kim_sources": kim_sources,
    }
    _AUTH_CACHE[key] = result
    return result


def _contract(project_root: Path, output_dir: Path) -> dict:
    auth = _authenticate_inputs(project_root)
    conditioned = auth["conditioned"]
    baseline = auth["baseline"]
    completions = auth["kim_completions"]
    kim_sources = auth["kim_sources"]
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "prepared_panel_unopened",
        "scientific_role": "prospective_final_conditioned_vcg_comparison",
        "output_dir": str(output_dir.resolve()),
        "panel_opened": False,
        "instance_seeds": list(INSTANCE_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "lambda_grid": list(LAMBDA_GRID),
        "conditioned_rows": CONDITIONED_ROWS,
        "baseline_rows": {
            V23_METHOD: V23_ROWS,
            DYNAMIC_METHOD: DYNAMIC_ROWS,
            GA_METHOD: GA_ROWS,
            KIM_METHOD: KIM_ROWS,
        },
        "total_expected_rows": EXPECTED_ROWS,
        "all_rows_use_same_serialized_episode_instances": True,
        "training_or_learning": False,
        "checkpoint_selection": False,
        "lambda_selection": False,
        "complete_case_filtering_allowed": False,
        "failed_method_or_lambda_has_aggregate_metrics_suppressed": True,
        "aggregation_unit": "EpisodeInstance",
        "nuisance_averaging": {
            CONDITIONED_METHOD: "three model seeds within EpisodeInstance",
            V23_METHOD: "three model seeds x four policy RNGs within EpisodeInstance",
            DYNAMIC_METHOD: "one deterministic run per EpisodeInstance",
            GA_METHOD: "four optimizer RNGs within EpisodeInstance",
            KIM_METHOD: "three model seeds x five policy RNGs within EpisodeInstance",
        },
        "environment": {
            "grid_rows": 5,
            "grid_cols": 5,
            "number_blocks": EXPECTED_BLOCKS,
            "arrival_rate": 10.0,
            "proc_mean": 80,
            "max_steps": MAX_STEPS,
        },
        "execution_devices": {
            CONDITIONED_METHOD: "cpu",
            V23_METHOD: "cpu",
            DYNAMIC_METHOD: "cpu",
            GA_METHOD: "cpu",
            KIM_METHOD: "cpu",
        },
        "metrics_are_separate_not_collapsed": True,
        "primary_tradeoff_plane": [
            "mean_absolute_error",
            "physical_rehandles_per_100_required_deliveries",
        ],
        "secondary_metrics": [
            "dense_return",
            "steps",
            "within_target_window_rate",
        ],
        "predeclared_cumulative_plot_conditioned_lambdas": [0.0, 0.2],
        "conditioned_terminal_sha256": {
            str(seed): conditioned["terminal_sha256"][seed]
            for seed in MODEL_SEEDS
        },
        "conditioned_terminal_training_rounds": {"0": 8, "1": 10, "2": 8},
        "lambda_zero_exact_vcg_1_1_delegation": True,
        "baseline_policy_rng": {
            "v23_base": V23_POLICY_RNG_BASE,
            "v23_grid_reuses_authenticated_frozen_adapter_streams": True,
            "ga_base": GA_RNG_BASE,
            "kim_base": KIM_POLICY_RNG_BASE,
        },
        "historical_v23_checkpoints": {
            str(seed): final86.V23_SELECTED[seed][1]
            for seed in final86.V23_MODEL_SEEDS
        },
        "kim_checkpoints": {
            str(seed): completions[seed]["selected_checkpoint_raw_sha256"]
            for seed in kim_final.MODEL_SEEDS
        },
        "authenticated_baseline_source_sha256": dict(
            sorted(baseline["source_hashes"].items())
        ),
        "authenticated_kim_evaluation_manifest_sha256": kim_sources[
            "manifest_sha256"
        ],
        "authenticated_kim_source_sha256": kim_sources["source_sha256"],
        "source_sha256": _source_hashes(project_root),
        "runtime_contract": {
            "execution_device": "cpu",
            "torch_num_threads": 1,
            "software_versions": _software_runtime_contract(),
            "host_runtime_metadata_is_descriptive_only": True,
        },
    }
    return _with_hash(semantic, "contract_sha256")


def prepare(project_root: Path, output_dir: Path) -> dict:
    output_dir = output_dir.absolute()
    if output_dir.is_symlink():
        raise Final90Error("output directory must not be a symlink")
    expected = _contract(project_root, output_dir)
    path = output_dir / CONTRACT_NAME
    if path.is_file():
        observed = _load_json(path, label="final90 contract")
        _verify_hash(observed, "contract_sha256", label="final90 contract")
        if observed != expected:
            raise Final90Error("final90 contract, inputs, or sources changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise Final90Error("nonempty final90 output has no contract")
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, expected)
    return {
        "status": "prepared_panel_unopened",
        "panel_opened": False,
        "conditioned_rows": CONDITIONED_ROWS,
        "baseline_rows": EXPECTED_ROWS - CONDITIONED_ROWS,
        "total_rows": EXPECTED_ROWS,
        "contract": str(path.resolve()),
    }


def authenticate_contract(project_root: Path, output_dir: Path) -> dict:
    observed = _load_json(output_dir / CONTRACT_NAME, label="final90 contract")
    _verify_hash(observed, "contract_sha256", label="final90 contract")
    if observed != _contract(project_root, output_dir.absolute()):
        raise Final90Error("final90 contract, inputs, or sources changed")
    return observed


def _instance_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "episode-instances" / f"seed-{seed}.json"


def open_panel(
    project_root: Path, output_dir: Path, *, confirmation: str
) -> dict:
    if confirmation != OPEN_CONFIRMATION:
        raise Final90Error(f"opening final90 requires --confirm {OPEN_CONFIRMATION}")
    contract = authenticate_contract(project_root, output_dir)
    activation_path = output_dir / ACTIVATION_NAME
    if activation_path.is_file():
        activation = _load_json(activation_path, label="panel activation")
        _verify_hash(activation, "activation_sha256", label="panel activation")
        if activation.get("contract_sha256") != contract["contract_sha256"]:
            raise Final90Error("panel activation contract changed")
    else:
        activation = _with_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "panel_opened": True,
                "marker_written_before_first_instance_sample": True,
                "contract_sha256": contract["contract_sha256"],
                "contract_raw_sha256": _sha256(output_dir / CONTRACT_NAME),
                "authorized_seeds": list(INSTANCE_SEEDS),
                "runtime_provenance_at_activation": _cpu_runtime_provenance(),
            },
            "activation_sha256",
        )
        _atomic_json(activation_path, activation)
    if (output_dir / INSTANCE_MANIFEST_NAME).is_file():
        return authenticate_manifest(project_root, output_dir)
    env = final86._new_environment()
    records = []
    for index, seed in enumerate(INSTANCE_SEEDS):
        path = _instance_path(output_dir, seed)
        sampled = env.sample_episode_instance(seed)
        if path.is_file():
            observed = final86.EpisodeInstance.from_json(
                path.read_text(encoding="utf-8")
            )
            if observed != sampled:
                raise Final90Error(f"serialized EpisodeInstance changed: {seed}")
        else:
            _atomic_text(path, sampled.to_json())
            observed = final86.EpisodeInstance.from_json(
                path.read_text(encoding="utf-8")
            )
        observed.validate_for(env)
        raw = path.read_bytes()
        records.append(
            {
                "seed": seed,
                "instance_index": index,
                "relative_path": str(path.relative_to(output_dir)),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "canonical_sha256": hashlib.sha256(
                    observed.to_json().encode("utf-8")
                ).hexdigest(),
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
            "all_rows_load_exact_serialized_instances": True,
        },
        "manifest_sha256",
    )
    _atomic_json(output_dir / INSTANCE_MANIFEST_NAME, manifest)
    return authenticate_manifest(project_root, output_dir)


def authenticate_manifest(project_root: Path, output_dir: Path) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    activation = _load_json(output_dir / ACTIVATION_NAME, label="panel activation")
    _verify_hash(activation, "activation_sha256", label="panel activation")
    if (
        activation.get("contract_sha256") != contract["contract_sha256"]
        or activation.get("authorized_seeds") != list(INSTANCE_SEEDS)
    ):
        raise Final90Error("panel activation identity changed")
    manifest = _load_json(
        output_dir / INSTANCE_MANIFEST_NAME, label="EpisodeInstance manifest"
    )
    _verify_hash(manifest, "manifest_sha256", label="EpisodeInstance manifest")
    records = manifest.get("instances")
    if (
        manifest.get("contract_sha256") != contract["contract_sha256"]
        or not isinstance(records, list)
        or [record.get("seed") for record in records] != list(INSTANCE_SEEDS)
    ):
        raise Final90Error("EpisodeInstance manifest grid changed")
    env = final86._new_environment()
    for record in records:
        path = output_dir / str(record["relative_path"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
            raise Final90Error("EpisodeInstance bytes changed")
        instance = final86.EpisodeInstance.from_json(raw.decode("utf-8"))
        instance.validate_for(env)
        if (
            instance.seed != record["seed"]
            or instance.instance_id != record["episode_instance_id"]
            or instance.schedule_id != record["schedule_id"]
            or hashlib.sha256(instance.to_json().encode("utf-8")).hexdigest()
            != record["canonical_sha256"]
        ):
            raise Final90Error("EpisodeInstance semantic identity changed")
    return manifest


def _load_instance(output_dir: Path, record: Mapping):
    path = output_dir / str(record["relative_path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
        raise Final90Error("EpisodeInstance changed before execution")
    return final86.EpisodeInstance.from_json(raw.decode("utf-8"))


def _identity(record: Mapping) -> dict:
    return {
        "instance_seed": int(record["seed"]),
        "instance_index": int(record["instance_index"]),
        "episode_instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        "episode_instance_sha256": record["canonical_sha256"],
    }


def _conditioned_ledger_path(
    output_dir: Path, value: float, model_seed: int, instance_seed: int
) -> Path:
    return (
        output_dir
        / "run-ledger"
        / CONDITIONED_METHOD
        / f"lambda-{_lambda_token(value)}"
        / f"seed-{model_seed}"
        / f"instance-{instance_seed}.json"
    )


def _timing(deviations: Sequence[float]) -> dict:
    values = tuple(float(value) for value in deviations)
    if len(values) != EXPECTED_BLOCKS or not all(math.isfinite(v) for v in values):
        raise Final90Error("complete conditioned row needs eight timing deviations")
    return {
        "mean_signed_deviation": float(fmean(values)),
        "mean_absolute_error": float(fmean(abs(value) for value in values)),
        "mean_tardiness": float(fmean(max(value, 0.0) for value in values)),
        "mean_earliness": float(fmean(max(-value, 0.0) for value in values)),
        "within_target_window_rate": float(
            fmean(abs(value) <= TARGET_WINDOW for value in values)
        ),
    }


def _conditioned_row(
    raw: Mapping,
    compact: Mapping,
    *,
    identity: Mapping,
    model_seed: int,
    value: float,
    checkpoint_sha256: str,
) -> dict:
    timing = _timing(raw["delivery_deviations"])
    strict = bool(compact["strict_safe_complete"])
    issues = []
    if not strict:
        issues.append("not_strict_safe_complete")
    if raw.get("method_failure_reason") is not None:
        issues.append("method_failure_reason_present")
    if int(raw.get("illegal_drops", 0)):
        issues.append("nonzero_illegal_drops")
    if int(raw.get("macro_failures", 0)):
        issues.append("nonzero_macro_failures")
    row = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "method": CONDITIONED_METHOD,
        **dict(identity),
        "model_seed": model_seed,
        "preference_lambda": float(value),
        "rng_index": None,
        "rng_seed": None,
        "checkpoint_sha256": checkpoint_sha256,
        "dense_return": float(compact["dense_return"]),
        "delivery_deviations": list(raw["delivery_deviations"]),
        **timing,
        "steps": int(compact["steps"]),
        "required_deliveries": EXPECTED_BLOCKS,
        "physical_storage_relocations": int(compact["physical_rehandles"]),
        "physical_rehandles_per_100_required_deliveries": float(
            compact["physical_rehandles_per_100"]
        ),
        "strict_safe_complete": strict,
        "all_selected_candidates_exact_safe": bool(
            raw.get("complete_frontier_exactly_verified", False)
        ),
        "method_failure_reason": raw.get("method_failure_reason"),
        "safety_issues": sorted(set(issues)),
        "behavior_digest": raw.get("behavior_digest"),
        "evaluation_learning": False,
        "lambda_zero_exact_vcg_1_1_delegation": value == 0.0,
    }
    _validate_conditioned_row(row, identity=identity, value=value, seed=model_seed)
    return row


def _failed_conditioned_row(
    error: Exception,
    *,
    identity: Mapping,
    model_seed: int,
    value: float,
    checkpoint_sha256: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "method": CONDITIONED_METHOD,
        **dict(identity),
        "model_seed": model_seed,
        "preference_lambda": float(value),
        "rng_index": None,
        "rng_seed": None,
        "checkpoint_sha256": checkpoint_sha256,
        "dense_return": None,
        "delivery_deviations": [],
        "mean_signed_deviation": None,
        "mean_absolute_error": None,
        "mean_tardiness": None,
        "mean_earliness": None,
        "within_target_window_rate": None,
        "steps": None,
        "required_deliveries": EXPECTED_BLOCKS,
        "physical_storage_relocations": None,
        "physical_rehandles_per_100_required_deliveries": None,
        "strict_safe_complete": False,
        "all_selected_candidates_exact_safe": None,
        "method_failure_reason": f"{type(error).__name__}: {error}",
        "safety_issues": ["execution_exception"],
        "behavior_digest": None,
        "evaluation_learning": False,
        "lambda_zero_exact_vcg_1_1_delegation": value == 0.0,
    }


def _validate_conditioned_row(
    row: Mapping, *, identity: Mapping, value: float, seed: int
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "method": CONDITIONED_METHOD,
        **dict(identity),
        "model_seed": seed,
        "preference_lambda": float(value),
    }
    for name, expected_value in expected.items():
        if row.get(name) != expected_value:
            raise Final90Error(f"conditioned row binding changed: {name}")
    if type(row.get("strict_safe_complete")) is not bool:
        raise Final90Error("conditioned strict-safe flag is invalid")
    if row["strict_safe_complete"]:
        observed = _timing(row["delivery_deviations"])
        for name, expected_value in observed.items():
            if not math.isclose(
                float(row[name]), expected_value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise Final90Error(f"conditioned timing metric changed: {name}")
        if row.get("safety_issues") != []:
            raise Final90Error("strict conditioned row has safety issues")
    elif not row.get("safety_issues"):
        raise Final90Error("failed conditioned row lacks an issue")


def _load_conditioned_agent(
    project_root: Path,
    auth: Mapping,
    *,
    model_seed: int,
    base,
    device,
):
    conditioned = auth["conditioned"]
    terminal = conditioned["terminals"][model_seed]
    if model_seed == 0:
        return seed0_parent._new_terminal_agent(
            project_root,
            conditioned["inputs"]["arms"][0],
            terminal,
            device,
            base=base,
        )
    seed_input = conditioned["inputs"]["seed_inputs"][model_seed]
    return ConditionedHandlingAgent.from_checkpoint(
        terminal["agent_checkpoint"],
        base_agent=base,
        expected_base_checkpoint_sha256=seed_input["arm"].checkpoint_sha256,
        expected_base_policy_digest=seed_input["arm"].deployment_policy_digest,
        expected_source_cost_sha256=seed_input["cost_sha256"],
        seed=model_seed,
    )


def run_conditioned(
    project_root: Path,
    output_dir: Path,
    *,
    device_name: str,
    selected_seed: Optional[int] = None,
) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    manifest = authenticate_manifest(project_root, output_dir)
    device = pilot._device(device_name)
    if device.type != "cpu":
        raise Final90Error("final90 CPU-v3 conditioned evaluation requires CPU")
    auth = _authenticate_inputs(project_root)
    conditioned = auth["conditioned"]
    seeds = MODEL_SEEDS if selected_seed is None else (selected_seed,)
    if any(seed not in MODEL_SEEDS for seed in seeds):
        raise Final90Error("invalid conditioned model seed")
    completed = safe = 0
    for value in LAMBDA_GRID:
        for model_seed in seeds:
            arm = conditioned["inputs"]["arms"][model_seed]
            checkpoint_sha = conditioned["terminal_sha256"][model_seed]
            for record in manifest["instances"]:
                identity = _identity(record)
                path = _conditioned_ledger_path(
                    output_dir, value, model_seed, identity["instance_seed"]
                )
                if path.is_file():
                    ledger = _load_json(path, label="conditioned ledger")
                    _verify_hash(ledger, "ledger_sha256", label="conditioned ledger")
                    if ledger.get("contract_sha256") != contract["contract_sha256"]:
                        raise Final90Error("conditioned ledger contract changed")
                    row = ledger.get("run")
                    _validate_conditioned_row(
                        row, identity=identity, value=value, seed=model_seed
                    )
                else:
                    instance = _load_instance(output_dir, record)

                    def factory(base):
                        agent = _load_conditioned_agent(
                            project_root,
                            auth,
                            model_seed=model_seed,
                            base=base,
                            device=device,
                        )
                        agent.set_epsilon(0.0)
                        return seed0_parent._FixedLambdaAgent(agent, value)

                    try:
                        raw = pilot._run_raw(
                            arm,
                            instance,
                            device=device,
                            wrapper_factory=factory,
                        )
                        compact = pilot._compact_row(raw, instance)
                        row = _conditioned_row(
                            raw,
                            compact,
                            identity=identity,
                            model_seed=model_seed,
                            value=value,
                            checkpoint_sha256=checkpoint_sha,
                        )
                    except Exception as error:
                        row = _failed_conditioned_row(
                            error,
                            identity=identity,
                            model_seed=model_seed,
                            value=value,
                            checkpoint_sha256=checkpoint_sha,
                        )
                    ledger = _with_hash(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "protocol": PROTOCOL,
                            "contract_sha256": contract["contract_sha256"],
                            "manifest_sha256": manifest["manifest_sha256"],
                            "method": CONDITIONED_METHOD,
                            "preference_lambda": float(value),
                            "model_seed": model_seed,
                            "instance_seed": identity["instance_seed"],
                            "checkpoint_sha256": checkpoint_sha,
                            "run": row,
                        },
                        "ledger_sha256",
                    )
                    _atomic_json(path, ledger)
                    ledger = _load_json(path, label="serialized conditioned ledger")
                    _verify_hash(
                        ledger, "ledger_sha256", label="serialized conditioned ledger"
                    )
                    row = ledger["run"]
                completed += 1
                safe += int(row["strict_safe_complete"])
                if completed % 25 == 0:
                    print(
                        f"Conditioned VCG seed-set {completed}/"
                        f"{len(seeds) * len(LAMBDA_GRID) * len(INSTANCE_SEEDS)} "
                        f"| safe={safe}",
                        flush=True,
                    )
    return {
        "status": "complete",
        "model_seeds": list(seeds),
        "rows": completed,
        "strict_safe_complete_rows": safe,
    }


@contextmanager
def _final90_baseline_context():
    old = {
        "protocol": final86.PROTOCOL,
        "instance_seeds": final86.INSTANCE_SEEDS,
        "matched_seeds": final86.matched.PANEL_SEEDS,
        "adapter_seeds": v23_adapter.FINAL_INSTANCE_SEEDS,
        "v23_rng_base": final86.V23_POLICY_RNG_BASE,
        "ga_rng_base": final86.GA_RNG_BASE,
        "assert_cuda": final86._assert_cuda,
        "load_v23_runtime": final86._load_v23_runtime,
        "input_contract": final86._input_contract,
        "common_row": final86._common_row,
    }

    def cpu_input_contract(**kwargs):
        result = old["input_contract"](**kwargs)
        if kwargs.get("method") == V23_METHOD:
            result["learned_device"] = "cpu"
        return result

    def cpu_common_row(**kwargs):
        result = old["common_row"](**kwargs)
        if kwargs.get("method") == V23_METHOD:
            result["evidence"]["device"] = "cpu"
        return result

    try:
        final86.PROTOCOL = PROTOCOL
        final86.INSTANCE_SEEDS = INSTANCE_SEEDS
        final86.matched.PANEL_SEEDS = INSTANCE_SEEDS
        v23_adapter.FINAL_INSTANCE_SEEDS = INSTANCE_SEEDS
        final86.V23_POLICY_RNG_BASE = V23_POLICY_RNG_BASE
        final86.GA_RNG_BASE = GA_RNG_BASE
        final86._assert_cuda = lambda: None
        final86._load_v23_runtime = _load_v23_runtime_cpu
        final86._input_contract = cpu_input_contract
        final86._common_row = cpu_common_row
        yield
    finally:
        final86._common_row = old["common_row"]
        final86._input_contract = old["input_contract"]
        final86._load_v23_runtime = old["load_v23_runtime"]
        final86._assert_cuda = old["assert_cuda"]
        final86.GA_RNG_BASE = old["ga_rng_base"]
        final86.V23_POLICY_RNG_BASE = old["v23_rng_base"]
        v23_adapter.FINAL_INSTANCE_SEEDS = old["adapter_seeds"]
        final86.matched.PANEL_SEEDS = old["matched_seeds"]
        final86.INSTANCE_SEEDS = old["instance_seeds"]
        final86.PROTOCOL = old["protocol"]


def _load_v23_runtime_cpu(*, seed: int, auth: Mapping):
    binding = auth["v23_binding"]
    completion = auth["v23_completions"][final86.V23_MODEL_SEEDS.index(seed)]
    seed_dir = final86.STABILITY_OUTPUT / f"seed-{seed}"
    profile = final86.v23_stability._frozen_profile(seed)
    loaded = final86.v23_stability.load_seed_best_candidate(
        seed,
        seed_dir / "best-development-candidate.pth",
        binding=binding,
        expected_best_sha256=completion["selected_best"][
            "best_checkpoint_raw_sha256"
        ],
        manifest_path=seed_dir / "candidate-look-checkpoint-manifest.json",
        expected_manifest_sha256=completion["candidate_manifest_sha256"],
        validation_instance_manifest_path=seed_dir
        / "validation-instance-manifest.json",
        contract=seed_dir / "training-contract.json",
        device="cpu",
    )
    activation = final86.v23_stability.activated_seed_profile(profile, binding)
    activation.__enter__()
    try:
        contract = final86._load_json(
            seed_dir / "training-contract.json", name="VCG 2.3 training contract"
        )
        runtime_contract = dict(contract)
        runtime_contract["device"] = "cpu"
        runtime = v23_adapter.FinalV23Runtime(
            args=SimpleNamespace(device="cpu"), contract=runtime_contract
        )
        runtime.agent = loaded["agent"]
        runtime.set_schedule_state(loaded["checkpoint"]["schedule_state"])
        final86._require_equal(
            "VCG 2.3 runtime lambda",
            runtime.dual_lambda,
            loaded["checkpoint"]["validated_policy_lambda"],
        )
        return runtime, loaded, activation
    except Exception:
        activation.__exit__(None, None, None)
        raise


def run_v23(project_root: Path, output_dir: Path) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    manifest = authenticate_manifest(project_root, output_dir)
    auth = _authenticate_inputs(project_root)["baseline"]
    with _final90_baseline_context():
        rows, missing = final86._v23_rows(
            root=output_dir,
            contract=contract,
            manifest=manifest,
            auth=auth,
            execute=True,
        )
    return {
        "method": V23_METHOD,
        "rows": len(rows),
        "safe": sum(row["strict_safe_complete"] for row in rows),
        "missing": len(missing),
    }


def run_baselines(project_root: Path, output_dir: Path) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    manifest = authenticate_manifest(project_root, output_dir)
    with _final90_baseline_context():
        dynamic, dynamic_missing = final86._dynamic_rows(
            root=output_dir,
            contract=contract,
            manifest=manifest,
            auth={},
            execute=True,
        )
        ga, ga_missing = final86._ga_rows(
            root=output_dir,
            contract=contract,
            manifest=manifest,
            auth={},
            execute=True,
        )
    return {
        DYNAMIC_METHOD: {
            "rows": len(dynamic),
            "safe": sum(row["strict_safe_complete"] for row in dynamic),
            "missing": len(dynamic_missing),
        },
        GA_METHOD: {
            "rows": len(ga),
            "safe": sum(row["strict_safe_complete"] for row in ga),
            "missing": len(ga_missing),
        },
    }


def _kim_inputs(project_root: Path, output_dir: Path):
    manifest = authenticate_manifest(project_root, output_dir)
    completions = _kim_completions(project_root)
    records = {int(record["seed"]): dict(record) for record in manifest["instances"]}
    return manifest, completions, records


def _kim_config(manifest: Mapping, completions: Mapping[int, Mapping]) -> dict:
    payload = {
        "protocol": KIM_PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_no_kim_final90_rows_executed",
        "scientific_role": "prospective_matched_final90_baseline",
        "model_seeds": list(kim_final.MODEL_SEEDS),
        "instance_seeds": list(INSTANCE_SEEDS),
        "rollouts_per_model_instance": len(kim_final.ROLLOUTS),
        "expected_rows": KIM_ROWS,
        "policy_rng_base": KIM_POLICY_RNG_BASE,
        "source_instance_manifest_sha256": manifest["manifest_sha256"],
        "checkpoints": {
            str(seed): {
                "selected_episode": completions[seed]["selected_episode"],
                "raw_sha256": completions[seed]["selected_checkpoint_raw_sha256"],
                "deployment_sha256": completions[seed][
                    "selected_checkpoint_deployment_sha256"
                ],
            }
            for seed in kim_final.MODEL_SEEDS
        },
        "aggregation": (
            "five policy RNGs within model-instance, three models within instance"
        ),
        "complete_case_filtering_allowed": False,
        "kim_retraining": False,
    }
    payload["config_sha256"] = final86._digest(payload)
    return payload


@contextmanager
def _kim90_context(project_root: Path, output_dir: Path):
    old = {
        "protocol": kim_final.PROTOCOL,
        "panel_seeds": kim_final.PANEL_SEEDS,
        "final_root": kim_final.FINAL86_ROOT,
        "output_root": kim_final.OUTPUT_ROOT,
        "expected_rows": kim_final.EXPECTED_ROWS,
        "seed_base": kim_final.POLICY_SEED_BASE,
        "inputs": kim_final._inputs,
        "config": kim_final._config,
    }
    try:
        kim_final.PROTOCOL = KIM_PROTOCOL
        kim_final.PANEL_SEEDS = INSTANCE_SEEDS
        kim_final.FINAL86_ROOT = output_dir
        kim_final.OUTPUT_ROOT = output_dir
        kim_final.EXPECTED_ROWS = KIM_ROWS
        kim_final.POLICY_SEED_BASE = KIM_POLICY_RNG_BASE
        kim_final._inputs = lambda: _kim_inputs(project_root, output_dir)
        kim_final._config = _kim_config
        yield
    finally:
        kim_final._config = old["config"]
        kim_final._inputs = old["inputs"]
        kim_final.POLICY_SEED_BASE = old["seed_base"]
        kim_final.EXPECTED_ROWS = old["expected_rows"]
        kim_final.OUTPUT_ROOT = old["output_root"]
        kim_final.FINAL86_ROOT = old["final_root"]
        kim_final.PANEL_SEEDS = old["panel_seeds"]
        kim_final.PROTOCOL = old["protocol"]


def run_kim(project_root: Path, output_dir: Path) -> dict:
    authenticate_contract(project_root, output_dir)
    authenticate_manifest(project_root, output_dir)
    with _kim90_context(project_root, output_dir):
        return kim_final.run(output_dir)


def _load_conditioned_rows(output_dir: Path, manifest: Mapping) -> list[dict]:
    rows = []
    for value in LAMBDA_GRID:
        for model_seed in MODEL_SEEDS:
            for record in manifest["instances"]:
                path = _conditioned_ledger_path(
                    output_dir, value, model_seed, int(record["seed"])
                )
                ledger = _load_json(path, label="conditioned ledger")
                _verify_hash(ledger, "ledger_sha256", label="conditioned ledger")
                row = dict(ledger["run"])
                _validate_conditioned_row(
                    row,
                    identity=_identity(record),
                    value=value,
                    seed=model_seed,
                )
                rows.append(row)
    return rows


def _baseline_grid(method: str):
    if method == V23_METHOD:
        return (
            (seed, model_seed, rng)
            for model_seed in final86.V23_MODEL_SEEDS
            for seed in INSTANCE_SEEDS
            for rng in final86.RNG_INDICES
        )
    if method == DYNAMIC_METHOD:
        return ((seed, None, None) for seed in INSTANCE_SEEDS)
    if method == GA_METHOD:
        return (
            (seed, None, rng)
            for seed in INSTANCE_SEEDS
            for rng in final86.RNG_INDICES
        )
    raise Final90Error(f"unknown baseline method: {method}")


def _load_baseline_rows(output_dir: Path, method: str) -> list[dict]:
    rows = []
    with _final90_baseline_context():
        for seed, model_seed, rng in _baseline_grid(method):
            path = final86._ledger_path(output_dir, method, seed, model_seed, rng)
            ledger = final86._load_json(path, name="final90 baseline ledger")
            final86._self_hash(
                ledger, "ledger_sha256", name="final90 baseline ledger"
            )
            row = dict(ledger["run"])
            if (
                row.get("instance_seed") != seed
                or row.get("model_seed") != model_seed
                or row.get("rng_index") != rng
            ):
                raise Final90Error("baseline row grid identity changed")
            rows.append(row)
    return rows


def _load_kim_rows(project_root: Path, output_dir: Path) -> list[dict]:
    with _kim90_context(project_root, output_dir):
        _manifest, completions, records = kim_final._inputs()
        return kim_final._load_rows(output_dir, completions, records)


def _metric(row: Mapping, name: str) -> float:
    key = "dense_objective_return" if (
        name == "dense_return" and "dense_return" not in row
    ) else name
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Final90Error(f"missing numeric metric {name}")
    value = float(value)
    if not math.isfinite(value):
        raise Final90Error(f"nonfinite metric {name}")
    return value


def _aggregate(
    rows: Sequence[Mapping], *, expected_per_instance: int
) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["instance_seed"])].append(row)
    complete_grid = set(grouped) == set(INSTANCE_SEEDS) and all(
        len(grouped[seed]) == expected_per_instance for seed in INSTANCE_SEEDS
    )
    safe_rows = sum(row.get("strict_safe_complete") is True for row in rows)
    eligible = bool(
        complete_grid
        and len(rows) == expected_per_instance * len(INSTANCE_SEEDS)
        and safe_rows == len(rows)
    )
    points = None
    metrics = None
    if eligible:
        points = {
            seed: {
                metric: float(fmean(_metric(row, metric) for row in grouped[seed]))
                for metric in METRICS
            }
            for seed in INSTANCE_SEEDS
        }
        metrics = {
            metric: float(fmean(points[seed][metric] for seed in INSTANCE_SEEDS))
            for metric in METRICS
        }
    return {
        "expected_rows": expected_per_instance * len(INSTANCE_SEEDS),
        "observed_rows": len(rows),
        "strict_safe_complete_rows": safe_rows,
        "whole_method_eligible": eligible,
        "numeric_metrics_suppressed": not eligible,
        "metrics": metrics,
        "instance_points": (
            [{"instance_seed": seed, **points[seed]} for seed in INSTANCE_SEEDS]
            if points is not None
            else None
        ),
        "failures": [
            {
                "model_seed": row.get("model_seed"),
                "instance_seed": row.get("instance_seed"),
                "rng_index": row.get("rng_index", row.get("rollout_index")),
                "reason": row.get("method_failure_reason"),
                "issues": row.get("safety_issues"),
            }
            for row in rows
            if row.get("strict_safe_complete") is not True
        ],
    }


def _paired_ci(values: Sequence[float]) -> dict:
    values = tuple(float(value) for value in values)
    if len(values) != len(INSTANCE_SEEDS):
        raise Final90Error("paired interval requires 30 EpisodeInstances")
    mean = float(fmean(values))
    standard_deviation = float(stdev(values))
    standard_error = standard_deviation / math.sqrt(len(values))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "nominal_two_sided_95_ci": [
            mean - NOMINAL_T_95 * standard_error,
            mean + NOMINAL_T_95 * standard_error,
        ],
    }


def _point_map(record: Mapping) -> dict[int, Mapping]:
    return {
        int(point["instance_seed"]): point
        for point in record["instance_points"]
    }


def _contrasts(left: Mapping, right: Mapping) -> dict:
    left_points = _point_map(left)
    right_points = _point_map(right)
    return {
        metric: _paired_ci(
            [
                float(left_points[seed][metric])
                - float(right_points[seed][metric])
                for seed in INSTANCE_SEEDS
            ]
        )
        for metric in (
            "dense_return",
            "mean_absolute_error",
            "steps",
            "physical_rehandles_per_100_required_deliveries",
        )
    }


def _dominates(left: Mapping, right: Mapping) -> bool:
    fields = (
        "mean_absolute_error",
        "physical_rehandles_per_100_required_deliveries",
    )
    return all(float(left[field]) <= float(right[field]) for field in fields) and any(
        float(left[field]) < float(right[field]) for field in fields
    )


def analyze(project_root: Path, output_dir: Path) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    manifest = authenticate_manifest(project_root, output_dir)
    conditioned_rows = _load_conditioned_rows(output_dir, manifest)
    methods = []
    conditioned_records = {}
    for value in LAMBDA_GRID:
        rows = [
            row
            for row in conditioned_rows
            if float(row["preference_lambda"]) == float(value)
        ]
        record = {
            "method": CONDITIONED_METHOD,
            "method_key": f"{CONDITIONED_METHOD}:lambda={value}",
            "display_name": f"VCG (lambda={value:g})",
            "preference_lambda": float(value),
            **_aggregate(rows, expected_per_instance=len(MODEL_SEEDS)),
        }
        methods.append(record)
        conditioned_records[value] = record
    baseline_specs = (
        (V23_METHOD, "Historical VCG 2.3", V23_ROWS // len(INSTANCE_SEEDS)),
        (DYNAMIC_METHOD, "Dynamic PSLAP", 1),
        (GA_METHOD, "Capacity-aware GA", len(final86.RNG_INDICES)),
    )
    for method, display, per_instance in baseline_specs:
        rows = _load_baseline_rows(output_dir, method)
        methods.append(
            {
                "method": method,
                "method_key": method,
                "display_name": display,
                "preference_lambda": None,
                **_aggregate(rows, expected_per_instance=per_instance),
            }
        )
    kim_rows = _load_kim_rows(project_root, output_dir)
    methods.append(
        {
            "method": KIM_METHOD,
            "method_key": KIM_METHOD,
            "display_name": "Kim2020 adaptation",
            "preference_lambda": None,
            **_aggregate(
                kim_rows,
                expected_per_instance=(
                    len(kim_final.MODEL_SEEDS) * len(kim_final.ROLLOUTS)
                ),
            ),
        }
    )
    eligible = [record for record in methods if record["whole_method_eligible"]]
    frontier = [
        record["method_key"]
        for record in eligible
        if not any(
            other["method_key"] != record["method_key"]
            and _dominates(other["metrics"], record["metrics"])
            for other in eligible
        )
    ]
    contrasts = {}
    lambda0 = conditioned_records[0.0]
    if lambda0["whole_method_eligible"]:
        for value, record in conditioned_records.items():
            if value != 0.0 and record["whole_method_eligible"]:
                contrasts[f"lambda={value}:minus:lambda=0"] = _contrasts(
                    record, lambda0
                )
        for record in methods:
            if (
                record["preference_lambda"] is None
                and record["whole_method_eligible"]
            ):
                contrasts[f"lambda=0:minus:{record['method_key']}"] = _contrasts(
                    lambda0, record
                )
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "expected_total_rows": EXPECTED_ROWS,
        "observed_total_rows": sum(record["observed_rows"] for record in methods),
        "all_conditioned_lambdas_eligible": all(
            conditioned_records[value]["whole_method_eligible"]
            for value in LAMBDA_GRID
        ),
        "methods": methods,
        "point_estimate_mae_rehandles_frontier": frontier,
        "paired_nominal_95_contrasts": contrasts,
        "complete_case_filtering_used": False,
        "metrics_collapsed_to_one_score": False,
        "statistical_unit": (
            "30 EpisodeInstances after within-instance nuisance averaging"
        ),
        "evaluation_learning": False,
        "checkpoint_or_lambda_selection": False,
    }
    report = _with_hash(report, "report_sha256")
    path = output_dir / REPORT_NAME
    if path.is_file():
        if _load_json(path, label="final90 report") != report:
            raise Final90Error("existing final90 report changed")
    else:
        _atomic_json(path, report)
    return report


def inspect(output_dir: Path) -> dict:
    counts = {}
    for value in LAMBDA_GRID:
        directory = (
            output_dir / "run-ledger" / CONDITIONED_METHOD / f"lambda-{_lambda_token(value)}"
        )
        count = len(list(directory.glob("seed-*/instance-*.json"))) if directory.is_dir() else 0
        counts[f"conditioned_lambda_{value}"] = {
            "expected": len(MODEL_SEEDS) * len(INSTANCE_SEEDS),
            "observed": count,
        }
    for method, expected in (
        (V23_METHOD, V23_ROWS),
        (DYNAMIC_METHOD, DYNAMIC_ROWS),
        (GA_METHOD, GA_ROWS),
    ):
        directory = output_dir / "run-ledger" / method
        count = len(list(directory.glob("*.json"))) if directory.is_dir() else 0
        counts[method] = {"expected": expected, "observed": count}
    kim_count = sum(
        len(
            list(
                (output_dir / "run-ledger" / f"seed-{seed}").glob(
                    "instance-*-roll-*.json"
                )
            )
        )
        for seed in kim_final.MODEL_SEEDS
    )
    counts[KIM_METHOD] = {"expected": KIM_ROWS, "observed": kim_count}
    return {
        "protocol": PROTOCOL,
        "rows": counts,
        "all_complete": all(
            item["expected"] == item["observed"] for item in counts.values()
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "open-panel",
            "run-vcg",
            "run-v23",
            "run-baselines",
            "run-kim",
            "analyze",
            "inspect",
        ),
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confirm")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model-seed", type=int, choices=MODEL_SEEDS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(project_root, output_dir)
    elif args.command == "open-panel":
        result = open_panel(
            project_root, output_dir, confirmation=args.confirm or ""
        )
    elif args.command == "run-vcg":
        result = run_conditioned(
            project_root,
            output_dir,
            device_name=args.device,
            selected_seed=args.model_seed,
        )
    elif args.command == "run-v23":
        result = run_v23(project_root, output_dir)
    elif args.command == "run-baselines":
        result = run_baselines(project_root, output_dir)
    elif args.command == "run-kim":
        result = run_kim(project_root, output_dir)
    elif args.command == "analyze":
        result = analyze(project_root, output_dir)
    else:
        result = inspect(output_dir)
    compact = result
    if args.command == "open-panel":
        compact = {
            "status": "panel_opened",
            "instance_count": result["instance_count"],
            "manifest_sha256": result["manifest_sha256"],
        }
    elif args.command == "analyze":
        compact = {
            "status": result["status"],
            "all_conditioned_lambdas_eligible": result[
                "all_conditioned_lambdas_eligible"
            ],
            "point_estimate_mae_rehandles_frontier": result[
                "point_estimate_mae_rehandles_frontier"
            ],
            "report": str((output_dir / REPORT_NAME).resolve()),
        }
    print(json.dumps(compact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
