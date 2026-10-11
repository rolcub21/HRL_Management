#!/usr/bin/env python3
"""Prospective four-method comparison on the sealed 86xxx panel.

``prepare`` authenticates and freezes everything without constructing an
EpisodeInstance.  ``open-panel`` is the only operation allowed to materialize
the final panel.  All later executions reload those exact serialized values.

The compared methods are the frozen VCG 1.1 three-seed ensemble, the frozen
VCG 2.3 three-seed stability ensemble, duration-aware Dynamic PSLAP, and the
capacity-aware Park--Seo-2009 rolling GA.  The workload unit is always the
EpisodeInstance (n=30); model and policy/optimizer RNGs are nested nuisance
replications.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from statistics import fmean, stdev
import sys
import tempfile
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

import compare_vcg_dense_pareto as v11
import compare_vcg_v2_3_matched_baselines as matched
import compare_vcg_v2_3_capacity_aware_ga_repair_v2 as repair_v2
import train_vcg_constrained_v2 as v2_train
import train_vcg_constrained_v2_3 as v23_train
import train_vcg_constrained_v2_3_stability as v23_stability
import vcg_v2_3_seed_stability as stability
import vcg_v2_3_seed_stability_analysis_repair_v2 as stability_analysis
import vcg_v2_3_seed_stability_protocol as stability_protocol
from example.episode_instance import EpisodeInstance
from final86_v23_runtime import FinalV23Runtime
from track_b_urgency_evaluate import evaluate_assignment_ablation_one
from vcg_objective_audit import ObjectiveAuditSmallRoomsEnv


PROTOCOL = "vcg_v1_1_v2_3_pslap_ga_final_86xxx_confirmation_v1"
SCHEMA_VERSION = 1
CONTRACT_NAME = "final86-contract.json"
ACTIVATION_NAME = "panel-activation.json"
INSTANCE_MANIFEST_NAME = "final-instance-manifest.json"
REPORT_NAME = "final-report.json"
AUDIT_NAME = "final-audit.json"
OPEN_CONFIRMATION = "OPEN_FINAL_86XXX_ONCE"

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/home/ai_diagnosis/HRL_Management")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "vcg-final86-four-method-confirmation"
STABILITY_OUTPUT = PROJECT_ROOT / "results" / "vcg-v2-3-seed-stability-85k"
STABILITY_ANALYSIS_OUTPUT = (
    PROJECT_ROOT / "results" / "vcg-v2-3-seed-stability-analysis-repair-v2-85k"
)
REPAIR_OUTPUT = PROJECT_ROOT / "results" / "vcg-v2-3-capacity-aware-ga-repair-v2-85k"
FREEZE_SPEC = PROJECT_ROOT / "experiments" / "vcg_v2_3_seed_stability_85k" / "freeze-spec.json"

INSTANCE_SEEDS = tuple(range(86_000, 86_030))
V11_MODEL_SEEDS = (0, 1, 2)
V23_MODEL_SEEDS = (11, 12, 13)
RNG_INDICES = (0, 1, 2, 3)
V23_POLICY_RNG_BASE = 622_000_000
GA_RNG_BASE = 630_000_000
GA_RNG_STRIDE = 2_001
MAX_STEPS = 2_000
EXPECTED_DELIVERIES = 8
TARGET_WINDOW = 20.0

V11_METHOD = "vcg_1_1"
V23_METHOD = "vcg_2_3"
DYNAMIC_METHOD = "duration_aware_dynamic_pslap"
GA_METHOD = "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial"
METHODS = (V11_METHOD, V23_METHOD, DYNAMIC_METHOD, GA_METHOD)

V11_SELECTED = {
    0: (500, "aa52ee14612ae39bb40fb9bdfecba8b7a44c40665628d26bb51f9f8f7a5355e8"),
    1: (450, "575326ddb3dbd535f4849e0f9d85bc55b8063279ebcfb6a872107f459dc56bb4"),
    2: (375, "3f681965feaca098a8d7fc2b58b41a8f0784191417e964a1fe7d5be2f5781917"),
}
V23_SELECTED = {
    11: (120, "95e4e4812b0a358c95b8856a25227e9a16eb7f727b8fd3d6b5366c809b1331e6"),
    12: (160, "e53c805b98c7bc105d59f79c4900c68f6982cdc272c724aba3f5b73084e38566"),
    13: (100, "3c089b3d79abdebc9c3a54341efe7d2aed41459cd348c7e5059afebe456f5f53"),
}

# Paired df=29 constants frozen before final outcomes.
PRIMARY_BONFERRONI_T = 2.8315526875
NOMINAL_T_95 = 2.0452296421
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


class Final86Error(RuntimeError):
    pass


def _canonical_bytes(value: Mapping) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Mapping, *, hash_field: Optional[str] = None) -> str:
    payload = dict(value)
    if hash_field is not None:
        payload.pop(hash_field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_bytes(path: Path, *, name: str) -> bytes:
    requested = Path(path).absolute()
    if requested.is_symlink() or not requested.is_file() or requested.resolve() != requested:
        raise Final86Error(f"{name} must be a canonical regular non-symlink file")
    return requested.read_bytes()


def _sha256_file(path: Path, *, name: str = "file") -> str:
    return _sha256_bytes(_read_bytes(path, name=name))


def _load_json(path: Path, *, name: str) -> dict:
    try:
        value = json.loads(_read_bytes(path, name=name).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Final86Error(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise Final86Error(f"{name} must contain an object")
    return value


def _self_hash(payload: Mapping, field: str, *, name: str) -> str:
    observed = payload.get(field)
    expected = _digest(payload, hash_field=field)
    if not isinstance(observed, str) or observed != expected:
        raise Final86Error(f"{name} self hash mismatch")
    return expected


def _atomic_json(path: Path, payload: Mapping) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _require_equal(name: str, observed, expected) -> None:
    if observed != expected:
        raise Final86Error(f"{name} mismatch: observed={observed!r}, expected={expected!r}")


def _runtime_versions() -> dict:
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu,
        "torch_num_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _configure_runtime() -> dict:
    """Apply and report the process-wide state frozen for every panel phase."""

    torch.set_num_threads(1)
    return _runtime_versions()


def _assert_cuda() -> None:
    if not torch.cuda.is_available():
        raise Final86Error("final learned-method evaluation requires CUDA")


def _source_paths() -> dict[str, Path]:
    """Return the stable union of both authenticated runtime closures.

    A live ``sys.modules`` inventory is deliberately not used: lazy imports
    during a long evaluation could otherwise change the registry after rows
    had already been written.  The two parent protocols expose their explicit
    transitive closures; this prospective layer adds only its new runtime
    sources.
    """

    paths: dict[str, Path] = {}

    def bind(logical_name: str, value: Path) -> None:
        path = Path(value).resolve()
        prior = paths.get(logical_name)
        if prior is not None and prior != path:
            raise Final86Error(
                f"source registry collision for {logical_name}: {prior} != {path}"
            )
        paths[logical_name] = path

    for registry in (
        repair_v2._current_source_paths(),
        stability._source_paths(),
    ):
        for name, path in registry.items():
            bind(f"project:{name}", path)
    # V1.1's historical authenticator imports these runtime/controller parents
    # outside both newer protocols' explicit registries.  Keep the list stable
    # and reviewable rather than discovering it from process import state.
    for name in (
        "PSLAP/kim2020_a3c_spatial.py",
        "PSLAP/reg_selector_v4.py",
        "PSLAP/reg_selector_v5.py",
        "benchmark_viability_critic_priority.py",
        "example/Options/selector.py",
        "example/Options/waitOption.py",
        "example/controller_observation.py",
        "primitive_option.py",
        "relational_scheduler.py",
        "run_vcg_objective_gamma_audit.py",
        "train_vcg_dense_proper.py",
        "train_viability_graph_smdp.py",
        "train_viability_graph_smdp_proper.py",
    ):
        bind(f"project:{name}", PROJECT_ROOT / name)
    bind("prospective:runner", Path(__file__))
    bind("prospective:v23_final_adapter", HERE / "final86_v23_runtime.py")
    bind(
        "prospective:stability_analysis_repair",
        Path(stability_analysis.__file__),
    )
    return dict(sorted(paths.items()))


def _authenticate_v11() -> tuple:
    arms = []
    records = []
    for seed in V11_MODEL_SEEDS:
        directory = PROJECT_ROOT / "results" / f"vcg-dense-v1-1-seed{seed}-500ep"
        best_path = (directory / "best.pth").absolute()
        episode, expected_sha = V11_SELECTED[seed]
        data = _read_bytes(best_path, name="VCG 1.1 selected checkpoint")
        # Hash-first: no checkpoint byte reaches the restricted deserializer
        # before it matches the prospective external trust root.
        _require_equal("VCG 1.1 selected raw SHA", _sha256_bytes(data), expected_sha)
        payload = torch.load(BytesIO(data), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise Final86Error("VCG 1.1 selected checkpoint payload is not an object")
        top_schema = hashlib.sha256(
            json.dumps(sorted(payload), separators=(",", ":")).encode()
        ).hexdigest()
        state = payload.get("agent_state")
        if not isinstance(state, dict):
            raise Final86Error("VCG 1.1 selected checkpoint has no agent state")
        state_schema = hashlib.sha256(
            json.dumps(sorted(state), separators=(",", ":")).encode()
        ).hexdigest()
        _require_equal(
            "VCG 1.1 outer schema",
            top_schema,
            "22bd409612bc4d7d6a757ad037cbfa8c34513966ae115faa247ff1366f8a5ba2",
        )
        _require_equal(
            "VCG 1.1 agent-state schema",
            state_schema,
            "f20901e95b426773b0c57b5d3118a695c8f12eb475488b96cb912fcd12fcacc0",
        )
        _require_equal("VCG 1.1 model seed", payload.get("model_seed"), seed)
        _require_equal("VCG 1.1 selected episode", payload.get("selected_checkpoint_episode"), episode)
        _require_equal("VCG 1.1 checkpoint role", payload.get("checkpoint_role"), "best_deployment_finalized")
        _require_equal("VCG 1.1 deployment eligibility", payload.get("deployment_checkpoint_eligible"), True)
        _require_equal("VCG 1.1 training protocol", payload.get("training_protocol"), v11.TRAINING_PROTOCOL)
        _require_equal("VCG 1.1 method version", payload.get("method_version"), v11.METHOD_VERSION)
        _require_equal("VCG 1.1 timing objective", payload.get("timing_objective"), "dense_piecewise_v1")
        _require_equal("VCG 1.1 objective", payload.get("timing_objective_spec"), v11.FROZEN_OBJECTIVE_SPEC.to_dict())
        _require_equal("VCG 1.1 gamma", payload.get("gamma"), v11.FROZEN_GAMMA)
        _require_equal("VCG 1.1 baseline query", payload.get("baseline_policy_query"), False)
        _require_equal("VCG 1.1 future schedule", payload.get("future_schedule_visible_to_policy"), False)
        contract = payload.get("resume_contract")
        if not isinstance(contract, Mapping):
            raise Final86Error("VCG 1.1 selected checkpoint lacks a training contract")
        v11._validate_frozen_resume_contract(contract, seed)
        readiness = v11._checkpoint_readiness(payload)
        _require_equal("VCG 1.1 finalized readiness", readiness["finalized_best_deployment_artifact"], True)
        policy_digest = v11._policy_digest(payload)
        arm = v11.PolicyArm(
            method_id=f"vcg_dense_seed{seed}_selected_best",
            policy_group=v11.SELECTED_BEST,
            model_seed=seed,
            checkpoint_variant="best",
            checkpoint_weight_episode=episode,
            checkpoint_path=best_path,
            checkpoint_sha256=expected_sha,
            deployment_policy_digest=policy_digest,
            primary_analysis=True,
            diagnostic_only=False,
            readiness=readiness,
            payload=v11._compact_deployment_payload(payload),
        )
        arms.append(arm)
        records.append(
            {
                "model_seed": seed,
                "selected_episode": episode,
                "checkpoint_path": str(best_path),
                "checkpoint_raw_sha256": expected_sha,
                "deployment_policy_digest": policy_digest,
                "source_deployment_checkpoint_eligible": True,
            }
        )
    return tuple(arms), tuple(records)


def _authenticate_v23(*, load_device: str = "cpu") -> tuple:
    manifest, binding, repair = stability.authenticate_run_manifest(
        output_dir=STABILITY_OUTPUT,
        freeze_spec_path=FREEZE_SPEC,
        repair_dir=REPAIR_OUTPUT,
    )
    completions = []
    selected = []
    for seed in V23_MODEL_SEEDS:
        completion, loaded = stability.authenticate_seed_completion(
            seed,
            output_dir=STABILITY_OUTPUT,
            run_manifest=manifest,
            binding=binding,
        )
        if loaded is None:
            raise Final86Error(f"VCG 2.3 seed {seed} has no eligible selected candidate")
        episode, checkpoint_sha = V23_SELECTED[seed]
        _require_equal("VCG 2.3 selected episode", loaded["selected_episode"], episode)
        _require_equal("VCG 2.3 selected raw SHA", loaded["checkpoint_sha256"], checkpoint_sha)
        _require_equal(
            "VCG 2.3 upstream deployment flag",
            loaded["checkpoint"].get("deployment_checkpoint_eligible"),
            False,
        )
        completions.append(completion)
        selected.append(loaded)
    repaired = stability_analysis.validate(
        output_dir=STABILITY_ANALYSIS_OUTPUT,
        original_output_dir=STABILITY_OUTPUT,
        freeze_spec_path=FREEZE_SPEC,
        repair_dir=REPAIR_OUTPUT,
    )
    if repaired.get("status") != "passed" or repaired.get("strong_result") is not True:
        raise Final86Error("VCG 2.3 strong three-seed stability gate did not authenticate")
    return manifest, binding, repair, tuple(completions), tuple(selected), repaired


def _authenticate_all() -> dict:
    v11_arms, v11_records = _authenticate_v11()
    manifest, binding, repair, completions, selected, stable_report = _authenticate_v23()
    repaired_ga = stability_protocol.authenticate_repair_v2(REPAIR_OUTPUT)
    source_paths = _source_paths()
    source_hashes = {
        name: _sha256_file(path, name=f"source {name}") for name, path in source_paths.items()
    }
    return {
        "v11_arms": v11_arms,
        "v11_records": v11_records,
        "v23_manifest": manifest,
        "v23_binding": binding,
        "v23_repair": repair,
        "v23_completions": completions,
        "v23_selected": selected,
        "stability_report": stable_report,
        "repair_v2": repaired_ga,
        "source_paths": source_paths,
        "source_hashes": source_hashes,
    }


def _contract_payload(auth: Mapping, *, output_dir: Path) -> dict:
    v23_records = []
    for seed, completion, selected in zip(
        V23_MODEL_SEEDS, auth["v23_completions"], auth["v23_selected"]
    ):
        v23_records.append(
            {
                "model_seed": seed,
                "selected_episode": selected["selected_episode"],
                "checkpoint_raw_sha256": selected["checkpoint_sha256"],
                "completion_sha256": completion["completion_sha256"],
                "source_deployment_checkpoint_eligible": False,
                "external_role": "exact_final_confirmation_candidate",
            }
        )
    payload = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_panel_still_sealed",
        "output_dir": str(Path(output_dir).absolute()),
        "methods": list(METHODS),
        "method_definitions": {
            V11_METHOD: {"model_seeds": list(V11_MODEL_SEEDS), "rows": 90, "device": "cuda"},
            V23_METHOD: {"model_seeds": list(V23_MODEL_SEEDS), "policy_rng_count": 4, "rows": 360, "device": "cuda"},
            DYNAMIC_METHOD: {"rows": 30, "device": "cpu", "deterministic": True},
            GA_METHOD: {
                "rows": 120,
                "device": "cpu",
                "optimizer_rng_count": 4,
                "source_version": repair_v2.CapacityAwareRollingGAStorageAssigner.VERSION,
            },
        },
        "expected_total_rows": 600,
        "panel": {
            "seeds": list(INSTANCE_SEEDS),
            "grid_rows": 5,
            "grid_cols": 5,
            "number_blocks": 8,
            "arrival_distribution": "exponential",
            "arrival_rate_lambda": 10.0,
            "mean_interarrival": 0.1,
            "arrival_numpy_scale": 0.1,
            "processing_distribution": "poisson",
            "processing_mean": 80,
            "max_steps": MAX_STEPS,
            "target_window": TARGET_WINDOW,
            "sealed_at_prepare": True,
        },
        "rng": {
            "v23_formula": "622000000 + 4*instance_index + rng_index",
            "v23_range": [622_000_000, 622_000_119],
            "ga_rollout_base_formula": "630000000 + 2001*(4*instance_index + rng_index)",
            "ga_committed_decision_formula": "ga_rollout_base + committed_decision_index",
            "ga_rollout_count_per_instance": 4,
            "ga_best_of_replications_forbidden": True,
        },
        "aggregation": {
            "unit": "EpisodeInstance",
            "instance_count": 30,
            "v11": "equal_mean_over_3_model_seeds_within_instance",
            "v23": "mean_4_action_rng_within_model_instance_then_equal_mean_3_models",
            "dynamic_pslap": "one_deterministic_row_per_instance",
            "ga": "equal_mean_4_optimizer_rng_realizations_per_instance",
            "row_level_pseudoreplication_forbidden": True,
        },
        "analysis": {
            "primary_coordinates": [
                "mean_absolute_error",
                "physical_rehandles_per_100_required_deliveries",
            ],
            "scalarization": None,
            "primary_bonferroni_t_df29": PRIMARY_BONFERRONI_T,
            "primary_contrast_family": "VCG2.3_vs_three_comparators_x_two_primary_metrics",
            "secondary_nominal_t_df29": NOMINAL_T_95,
            "complete_case_filtering_permitted": False,
        },
        "v11_candidates": list(auth["v11_records"]),
        "v23_candidates": v23_records,
        "v23_evaluation_authorization": {
            "upstream_artifacts_relabelled": False,
            "authorized_as_exact_final_confirmation_candidates": True,
            "authorization_basis": "authenticated_strong_three_seed_stability_result",
            "deployment_readiness_claimed_before_final_result": False,
        },
        "stability_trust": {
            "manifest_sha256": auth["v23_manifest"]["manifest_sha256"],
            "analysis_report_sha256": auth["stability_report"]["report_sha256"],
            "strong_result": True,
        },
        "repair_v2_trust": {
            "contract_sha256": auth["repair_v2"]["contract_sha256"],
            "report_sha256": auth["repair_v2"]["report_sha256"],
            "passed": True,
        },
        "source_paths": {name: str(path) for name, path in auth["source_paths"].items()},
        "source_sha256": dict(auth["source_hashes"]),
        "runtime_device_contract": {
            "learned_methods": "cuda",
            "baseline_methods": "cpu",
            "auto_device_forbidden": True,
            "torch_threads": 1,
        },
        "panel_opened": False,
        "performance_outcomes_observed": False,
    }
    payload["contract_sha256"] = _digest(payload)
    return payload


def _canonical_output_root(output_dir: Path, *, require_exists: bool) -> Path:
    requested = Path(output_dir).absolute()
    if requested.is_symlink():
        raise Final86Error("output root must not be a symlink")
    if require_exists and not requested.is_dir():
        raise Final86Error("output root does not exist")
    resolved = requested.resolve()
    if resolved != requested:
        raise Final86Error("output root must be canonical")
    protected = (PROJECT_ROOT / "results" / "vcg-v2-3-seed-stability-85k").resolve()
    if resolved == protected or resolved.is_relative_to(protected) or protected.is_relative_to(resolved):
        raise Final86Error("output root overlaps the stability artifact tree")
    if resolved == REPAIR_OUTPUT.resolve() or resolved.is_relative_to(REPAIR_OUTPUT.resolve()):
        raise Final86Error("output root overlaps the repaired baseline tree")
    return resolved


def _validate_root_entries(root: Path) -> None:
    allowed = {
        CONTRACT_NAME,
        ACTIVATION_NAME,
        "instances",
        INSTANCE_MANIFEST_NAME,
        "run-ledger",
        REPORT_NAME,
        AUDIT_NAME,
    }
    extras = {path.name for path in root.iterdir()} - allowed
    if extras:
        raise Final86Error(f"unexpected output entries: {sorted(extras)!r}")
    for path in root.iterdir():
        if path.is_symlink():
            raise Final86Error(f"output entry is a symlink: {path.name}")


def prepare(*, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    root = _canonical_output_root(output_dir, require_exists=False)
    if root.exists() and any(root.iterdir()):
        return authenticate_contract(output_dir=root)
    root.mkdir(parents=True, exist_ok=True)
    auth = _authenticate_all()
    contract = _contract_payload(auth, output_dir=root)
    _atomic_json(root / CONTRACT_NAME, contract)
    return authenticate_contract(output_dir=root)


def authenticate_contract(*, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    root = _canonical_output_root(output_dir, require_exists=True)
    _validate_root_entries(root)
    contract = _load_json(root / CONTRACT_NAME, name="final86 contract")
    _self_hash(contract, "contract_sha256", name="final86 contract")
    auth = _authenticate_all()
    expected = _contract_payload(auth, output_dir=root)
    _require_equal("final86 frozen contract", contract, expected)
    return contract


def _new_environment() -> ObjectiveAuditSmallRoomsEnv:
    return ObjectiveAuditSmallRoomsEnv(
        timing_objective=v23_train.FROZEN_OBJECTIVE_SPEC,
        grid_rows=5,
        grid_cols=5,
        number_blocks=8,
        choose_storage=False,
        arrival_rate=10.0,
        proc_mean=80,
    )


def _instance_file(root: Path, seed: int) -> Path:
    return root / "instances" / f"seed-{seed}.json"


def _validate_instance_tree(root: Path, *, require_complete: bool) -> None:
    directory = root / "instances"
    if directory.is_symlink():
        raise Final86Error("instances directory must not be a symlink")
    if not directory.exists():
        if require_complete:
            raise Final86Error("instances directory is missing")
        return
    if not directory.is_dir():
        raise Final86Error("instances path must be a directory")
    expected = {_instance_file(root, seed).absolute() for seed in INSTANCE_SEEDS}
    observed = set()
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise Final86Error("instances directory contains a non-regular entry")
        observed.add(path.absolute())
    extras = observed - expected
    if extras:
        raise Final86Error(
            f"instances directory contains unexpected files: {sorted(map(str, extras))[:3]!r}"
        )
    if require_complete and observed != expected:
        raise Final86Error(f"final instance tree is incomplete: {len(observed)}/30")


def _instance_record(path: Path, instance: EpisodeInstance) -> dict:
    raw = _read_bytes(path, name="final EpisodeInstance")
    canonical_sha = _sha256_bytes(instance.to_json().encode("utf-8"))
    return {
        "seed": int(instance.seed),
        "relative_path": str(path.relative_to(path.parent.parent)),
        "raw_sha256": _sha256_bytes(raw),
        "canonical_sha256": canonical_sha,
        "episode_instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        "arrival_steps": list(instance.arrival_steps),
        "storage_steps_needed": list(instance.storage_steps_needed),
    }


def open_panel(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR, confirmation: str,
) -> dict:
    if confirmation != OPEN_CONFIRMATION:
        raise Final86Error(f"opening the final panel requires --confirm {OPEN_CONFIRMATION}")
    contract = authenticate_contract(output_dir=output_dir)
    root = Path(contract["output_dir"])
    frozen_runtime = _configure_runtime()
    _assert_cuda()
    activation_path = root / ACTIVATION_NAME
    if not activation_path.exists():
        activation = {
            "protocol": PROTOCOL,
            "schema_version": SCHEMA_VERSION,
            "panel_opened": True,
            "marker_written_before_first_instance_sample": True,
            "contract_sha256": contract["contract_sha256"],
            "contract_raw_sha256": _sha256_file(root / CONTRACT_NAME, name="final86 contract"),
            "authorized_seeds": list(INSTANCE_SEEDS),
            "runtime_versions": frozen_runtime,
        }
        activation["activation_sha256"] = _digest(activation)
        _atomic_json(activation_path, activation)
    else:
        activation = _load_json(activation_path, name="panel activation")
        _self_hash(activation, "activation_sha256", name="panel activation")
        _require_equal("activation contract", activation["contract_sha256"], contract["contract_sha256"])
        _require_equal(
            "activation runtime", activation.get("runtime_versions"), frozen_runtime,
        )

    env = _new_environment()
    instances_dir = root / "instances"
    if instances_dir.is_symlink():
        raise Final86Error("instances directory must not be a symlink")
    instances_dir.mkdir(exist_ok=True)
    _validate_instance_tree(root, require_complete=False)
    records = []
    for seed in INSTANCE_SEEDS:
        path = _instance_file(root, seed)
        sampled = env.sample_episode_instance(seed)
        if path.exists():
            observed = EpisodeInstance.from_json(_read_bytes(path, name="final EpisodeInstance").decode())
            _require_equal("deterministic final EpisodeInstance", observed, sampled)
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=f".seed-{seed}.", suffix=".tmp", dir=instances_dir)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(sampled.to_json())
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            observed = EpisodeInstance.from_json(_read_bytes(path, name="final EpisodeInstance").decode())
            _require_equal("serialized final EpisodeInstance", observed, sampled)
        observed.validate_for(env)
        records.append(_instance_record(path, observed))
    _validate_instance_tree(root, require_complete=True)
    manifest = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "panel_opened": True,
        "contract_sha256": contract["contract_sha256"],
        "activation_sha256": activation["activation_sha256"],
        "instance_count": len(records),
        "instances": records,
        "all_methods_must_load_these_exact_serialized_values": True,
    }
    manifest["manifest_sha256"] = _digest(manifest)
    manifest_path = root / INSTANCE_MANIFEST_NAME
    if manifest_path.exists():
        observed_manifest = _load_json(manifest_path, name="final instance manifest")
        _require_equal("final instance manifest", observed_manifest, manifest)
    else:
        _atomic_json(manifest_path, manifest)
    return authenticate_instance_manifest(output_dir=root)


def authenticate_instance_manifest(*, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    contract = authenticate_contract(output_dir=output_dir)
    root = Path(contract["output_dir"])
    frozen_runtime = _configure_runtime()
    activation = _load_json(root / ACTIVATION_NAME, name="panel activation")
    _self_hash(activation, "activation_sha256", name="panel activation")
    _require_equal("activation protocol", activation.get("protocol"), PROTOCOL)
    _require_equal("activation schema", activation.get("schema_version"), SCHEMA_VERSION)
    _require_equal("activation opened flag", activation.get("panel_opened"), True)
    _require_equal("activation contract", activation.get("contract_sha256"), contract["contract_sha256"])
    _require_equal(
        "activation contract raw SHA",
        activation.get("contract_raw_sha256"),
        _sha256_file(root / CONTRACT_NAME, name="final86 contract"),
    )
    _require_equal("activation seed grid", activation.get("authorized_seeds"), list(INSTANCE_SEEDS))
    _require_equal("activation runtime", activation.get("runtime_versions"), frozen_runtime)
    _validate_instance_tree(root, require_complete=True)
    manifest = _load_json(root / INSTANCE_MANIFEST_NAME, name="final instance manifest")
    _self_hash(manifest, "manifest_sha256", name="final instance manifest")
    _require_equal("instance manifest contract", manifest["contract_sha256"], contract["contract_sha256"])
    records = manifest.get("instances")
    if not isinstance(records, list) or len(records) != 30:
        raise Final86Error("final instance manifest must contain exactly 30 records")
    observed_seeds = []
    env = _new_environment()
    for record in records:
        seed = record.get("seed")
        if type(seed) is not int or seed not in INSTANCE_SEEDS:
            raise Final86Error("final instance record has an invalid seed")
        observed_seeds.append(seed)
        path = root / str(record.get("relative_path"))
        _require_equal("final EpisodeInstance path", path, _instance_file(root, seed))
        raw = _read_bytes(path, name="final EpisodeInstance")
        _require_equal("final EpisodeInstance raw SHA", _sha256_bytes(raw), record["raw_sha256"])
        instance = EpisodeInstance.from_json(raw.decode())
        instance.validate_for(env)
        _require_equal("final EpisodeInstance seed", instance.seed, seed)
        _require_equal("final EpisodeInstance ID", instance.instance_id, record["episode_instance_id"])
        _require_equal("final schedule ID", instance.schedule_id, record["schedule_id"])
        _require_equal("final canonical SHA", _sha256_bytes(instance.to_json().encode()), record["canonical_sha256"])
    _require_equal("final panel seed order", observed_seeds, list(INSTANCE_SEEDS))
    return manifest


def _identity_map(manifest: Mapping) -> dict[int, dict]:
    result = {}
    for index, record in enumerate(manifest["instances"]):
        seed = int(record["seed"])
        result[seed] = {
            "instance_seed": seed,
            "instance_index": index,
            "episode_instance_id": record["episode_instance_id"],
            "schedule_id": record["schedule_id"],
            "episode_instance_sha256": record["canonical_sha256"],
        }
    return result


def _load_instance(root: Path, record: Mapping) -> EpisodeInstance:
    path = root / str(record["relative_path"])
    raw = _read_bytes(path, name="final EpisodeInstance")
    _require_equal("final EpisodeInstance raw SHA", _sha256_bytes(raw), record["raw_sha256"])
    instance = EpisodeInstance.from_json(raw.decode())
    _require_equal("final EpisodeInstance canonical SHA", _sha256_bytes(instance.to_json().encode()), record["canonical_sha256"])
    _require_equal("final EpisodeInstance ID", instance.instance_id, record["episode_instance_id"])
    _require_equal("final schedule ID", instance.schedule_id, record["schedule_id"])
    return instance


def _timing_metrics(deviations: Sequence[float]) -> dict:
    values = tuple(float(value) for value in deviations)
    if len(values) != EXPECTED_DELIVERIES or not all(math.isfinite(value) for value in values):
        raise Final86Error("a complete final row requires eight finite delivery deviations")
    return {
        "mean_signed_deviation": float(fmean(values)),
        "mean_absolute_error": float(fmean(abs(value) for value in values)),
        "mean_tardiness": float(fmean(max(value, 0.0) for value in values)),
        "mean_earliness": float(fmean(max(-value, 0.0) for value in values)),
        "within_target_window_rate": float(fmean(abs(value) <= TARGET_WINDOW for value in values)),
    }


def _strict_bool(value, *, name: str) -> bool:
    if type(value) is not bool:
        raise Final86Error(f"{name} must be boolean")
    return value


def _strict_int(value, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Final86Error(f"{name} must be an integer >= {minimum}")
    return value


def _common_row(
    *,
    method: str,
    raw: Mapping,
    identity: Mapping,
    model_seed: Optional[int],
    rng_index: Optional[int],
    rng_seed: Optional[int],
    checkpoint_sha256: Optional[str],
    evidence: Optional[Mapping] = None,
) -> dict:
    if method not in METHODS:
        raise Final86Error("unknown final method")
    deviations = tuple(float(value) for value in raw.get("delivery_deviations", ()))
    timing = _timing_metrics(deviations)
    dense_key = "dense_return" if "dense_return" in raw else "dense_objective_return"
    dense_return = float(raw[dense_key])
    if not math.isfinite(dense_return):
        raise Final86Error("dense return is nonfinite")
    steps = _strict_int(raw.get("steps"), name="steps")
    deliveries = _strict_int(raw.get("delivery_count"), name="delivery_count")
    required = raw.get("required_deliveries", EXPECTED_DELIVERIES)
    if required is None:
        required = EXPECTED_DELIVERIES
    required = _strict_int(required, name="required_deliveries", minimum=1)
    physical_raw = raw.get("physical_rehandles")
    if physical_raw is None:
        physical_raw = raw.get("physical_storage_relocations")
    physical = _strict_int(physical_raw, name="physical_storage_relocations")
    strict_raw = raw.get("strict_method_success")
    if type(strict_raw) is bool:
        strict = strict_raw
    elif isinstance(strict_raw, (int, float)) and not isinstance(strict_raw, bool) and float(strict_raw) in (0.0, 1.0):
        strict = bool(strict_raw)
    else:
        raise Final86Error("strict_method_success has an invalid type")
    completion = float(raw.get("completion_rate"))
    if not math.isfinite(completion):
        raise Final86Error("completion rate is nonfinite")
    illegal = _strict_int(raw.get("illegal_drops", 0), name="illegal_drops")
    invalid = _strict_int(raw.get("invalid_assignments", 0), name="invalid_assignments")
    fallbacks = _strict_int(raw.get("fallbacks", 0), name="fallbacks")
    witness = _strict_int(raw.get("witness_mismatches", 0), name="witness_mismatches")
    failure_reason = raw.get("method_failure_reason")
    exact = raw.get("all_selected_candidates_exact_safe")
    if method in (V11_METHOD, V23_METHOD):
        exact = _strict_bool(exact, name="all_selected_candidates_exact_safe")
    elif exact is not None and type(exact) is not bool:
        raise Final86Error("baseline exact-safe field must be boolean or null")
    issues = []
    if not strict:
        issues.append("strict_method_failure")
    if completion != 1.0 or deliveries != required or required != EXPECTED_DELIVERIES:
        issues.append("incomplete_required_workload")
    if failure_reason is not None:
        issues.append("method_failure_reason_present")
    if method in (V11_METHOD, V23_METHOD) and exact is not True:
        issues.append("learned_candidate_not_exact_safe")
    for name, value in (
        ("illegal_drops", illegal),
        ("invalid_assignments", invalid),
        ("fallbacks", fallbacks),
        ("witness_mismatches", witness),
    ):
        if value:
            issues.append(f"nonzero_{name}")
    if method == V23_METHOD:
        for name, expected in (
            ("evaluation_learning", False),
            ("fresh_evaluation_clone", True),
            ("training_agent_unchanged", True),
            ("validation_batch_state_unchanged", True),
            ("stochastic_selection_only", True),
            ("map_selection_used", False),
        ):
            if raw.get(name) is not expected:
                issues.append(f"v23_{name}")
    if method == GA_METHOD:
        if raw.get("capacity_aware_reservation_integrity") is not True:
            issues.append("ga_reservation_integrity")
        if raw.get("capacity_aware_source_audit_issues") not in ((), []):
            issues.append("ga_source_audit_issue")
    row = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "instance_seed": identity["instance_seed"],
        "instance_index": identity["instance_index"],
        "episode_instance_id": identity["episode_instance_id"],
        "schedule_id": identity["schedule_id"],
        "episode_instance_sha256": identity["episode_instance_sha256"],
        "model_seed": model_seed,
        "rng_index": rng_index,
        "rng_seed": rng_seed,
        "checkpoint_sha256": checkpoint_sha256,
        "dense_return": dense_return,
        "delivery_deviations": list(deviations),
        **timing,
        "steps": steps,
        "required_deliveries": required,
        "delivery_count": deliveries,
        "physical_storage_relocations": physical,
        "physical_rehandles_per_100_required_deliveries": 100.0 * physical / required,
        "strict_method_success": strict,
        "completion_rate": completion,
        "all_selected_candidates_exact_safe": exact,
        "illegal_drops": illegal,
        "invalid_assignments": invalid,
        "fallbacks": fallbacks,
        "witness_mismatches": witness,
        "method_failure_reason": failure_reason,
        "safety_issues": sorted(set(issues)),
        "strict_safe_complete": not issues,
        "evidence": {} if evidence is None else dict(evidence),
    }
    validate_row(row)
    return row


def _failed_row(
    *, method: str, identity: Mapping, model_seed: Optional[int],
    rng_index: Optional[int], rng_seed: Optional[int], checkpoint_sha256: Optional[str],
    error: Exception,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "instance_seed": identity["instance_seed"],
        "instance_index": identity["instance_index"],
        "episode_instance_id": identity["episode_instance_id"],
        "schedule_id": identity["schedule_id"],
        "episode_instance_sha256": identity["episode_instance_sha256"],
        "model_seed": model_seed,
        "rng_index": rng_index,
        "rng_seed": rng_seed,
        "checkpoint_sha256": checkpoint_sha256,
        "dense_return": None,
        "delivery_deviations": [],
        "mean_signed_deviation": None,
        "mean_absolute_error": None,
        "mean_tardiness": None,
        "mean_earliness": None,
        "within_target_window_rate": None,
        "steps": None,
        "required_deliveries": EXPECTED_DELIVERIES,
        "delivery_count": 0,
        "physical_storage_relocations": None,
        "physical_rehandles_per_100_required_deliveries": None,
        "strict_method_success": False,
        "completion_rate": 0.0,
        "all_selected_candidates_exact_safe": None,
        "illegal_drops": None,
        "invalid_assignments": None,
        "fallbacks": None,
        "witness_mismatches": None,
        "method_failure_reason": f"{type(error).__name__}: {error}",
        "safety_issues": ["execution_exception"],
        "strict_safe_complete": False,
        "evidence": {},
    }


def validate_row(row: Mapping) -> None:
    if row.get("protocol") != PROTOCOL or row.get("schema_version") != SCHEMA_VERSION:
        raise Final86Error("final row protocol mismatch")
    method = row.get("method")
    if method not in METHODS:
        raise Final86Error("final row method mismatch")
    seed = row.get("instance_seed")
    if type(seed) is not int or seed not in INSTANCE_SEEDS:
        raise Final86Error("final row instance seed mismatch")
    if row.get("instance_index") != seed - INSTANCE_SEEDS[0]:
        raise Final86Error("final row instance index mismatch")
    if type(row.get("strict_safe_complete")) is not bool:
        raise Final86Error("final row strict flag must be boolean")
    if row["strict_safe_complete"]:
        if row.get("safety_issues") != []:
            raise Final86Error("strict row contains safety issues")
        timing = _timing_metrics(row["delivery_deviations"])
        for name, expected in timing.items():
            if not math.isclose(float(row[name]), expected, rel_tol=0.0, abs_tol=1e-12):
                raise Final86Error(f"final row timing mismatch: {name}")
        required = _strict_int(row["required_deliveries"], name="required_deliveries", minimum=1)
        physical = _strict_int(row["physical_storage_relocations"], name="physical_storage_relocations")
        expected_rate = 100.0 * physical / required
        if not math.isclose(float(row["physical_rehandles_per_100_required_deliveries"]), expected_rate, rel_tol=0.0, abs_tol=1e-12):
            raise Final86Error("final row physical rate mismatch")


def _run_key(method: str, seed: int, model_seed: Optional[int], rng_index: Optional[int]) -> str:
    model = "none" if model_seed is None else str(model_seed)
    rng = "none" if rng_index is None else str(rng_index)
    return f"{method}:instance={seed}:model={model}:rng={rng}"


def _ledger_path(root: Path, method: str, seed: int, model_seed: Optional[int], rng_index: Optional[int]) -> Path:
    model = "none" if model_seed is None else str(model_seed)
    rng = "none" if rng_index is None else str(rng_index)
    return root / "run-ledger" / method / f"instance-{seed}-model-{model}-rng-{rng}.json"


def _input_contract(
    *, contract: Mapping, manifest: Mapping, method: str, identity: Mapping,
    model_seed: Optional[int], rng_index: Optional[int], rng_seed: Optional[int],
    checkpoint_sha256: Optional[str],
) -> dict:
    return {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "instance_manifest_sha256": manifest["manifest_sha256"],
        "method": method,
        "instance_seed": identity["instance_seed"],
        "instance_index": identity["instance_index"],
        "episode_instance_id": identity["episode_instance_id"],
        "schedule_id": identity["schedule_id"],
        "episode_instance_sha256": identity["episode_instance_sha256"],
        "model_seed": model_seed,
        "rng_index": rng_index,
        "rng_seed": rng_seed,
        "checkpoint_sha256": checkpoint_sha256,
        "max_steps": MAX_STEPS,
        "learned_device": "cuda" if method in (V11_METHOD, V23_METHOD) else None,
        "baseline_device": "cpu" if method in (DYNAMIC_METHOD, GA_METHOD) else None,
    }


def _load_or_execute(
    *, root: Path, input_contract: Mapping, execute: bool,
    executor: Callable[[], Mapping],
) -> Optional[dict]:
    path = _ledger_path(
        root,
        input_contract["method"],
        input_contract["instance_seed"],
        input_contract["model_seed"],
        input_contract["rng_index"],
    )
    fingerprint = _digest(input_contract)
    if path.exists():
        ledger = _load_json(path, name="final run ledger")
        _self_hash(ledger, "ledger_sha256", name="final run ledger")
        _require_equal("ledger run key", ledger.get("run_key"), _run_key(
            input_contract["method"], input_contract["instance_seed"],
            input_contract["model_seed"], input_contract["rng_index"],
        ))
        _require_equal("ledger input contract", ledger.get("input_contract"), dict(input_contract))
        _require_equal("ledger input fingerprint", ledger.get("input_fingerprint"), fingerprint)
        row = ledger.get("run")
        if not isinstance(row, Mapping):
            raise Final86Error("final ledger run is missing")
        _validate_row_binding(row, input_contract)
        return dict(row)
    if not execute:
        return None
    row = dict(executor())
    _validate_row_binding(row, input_contract)
    ledger = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "run_key": _run_key(
            input_contract["method"], input_contract["instance_seed"],
            input_contract["model_seed"], input_contract["rng_index"],
        ),
        "input_contract": dict(input_contract),
        "input_fingerprint": fingerprint,
        "run": row,
    }
    ledger["ledger_sha256"] = _digest(ledger)
    _atomic_json(path, ledger)
    # Serialized authority: always return the fully reloaded ledger.
    return _load_or_execute(
        root=root, input_contract=input_contract, execute=False,
        executor=lambda: (_ for _ in ()).throw(AssertionError("ledger reload executed policy")),
    )


def _validate_row_binding(row: Mapping, input_contract: Mapping) -> None:
    if not isinstance(row, Mapping):
        raise Final86Error("final row must be an object")
    for field in (
        "protocol", "schema_version", "method", "instance_seed", "instance_index",
        "episode_instance_id", "schedule_id", "episode_instance_sha256",
        "model_seed", "rng_index", "rng_seed", "checkpoint_sha256",
    ):
        expected_field = "protocol" if field == "protocol" else field
        expected = (
            PROTOCOL if field == "protocol" else
            SCHEMA_VERSION if field == "schema_version" else
            input_contract.get(expected_field)
        )
        _require_equal(f"final row/input {field}", row.get(field), expected)
    if type(row.get("strict_safe_complete")) is not bool:
        raise Final86Error("final row strict_safe_complete must be boolean")
    issues = row.get("safety_issues")
    if not isinstance(issues, list) or not all(isinstance(item, str) for item in issues):
        raise Final86Error("final row safety issues must be a string list")
    if row["strict_safe_complete"]:
        validate_row(row)
    else:
        if not issues or not isinstance(row.get("method_failure_reason"), str):
            raise Final86Error("failed final row must retain issues and a failure reason")


def _record_by_seed(manifest: Mapping) -> dict[int, dict]:
    return {int(record["seed"]): dict(record) for record in manifest["instances"]}


def _v11_exact_safety_audit(row: Mapping) -> tuple[bool, dict]:
    """Derive the V1.1 exact-safety flag from its authoritative frontiers."""

    audit = row.get("method_audit")
    if not isinstance(audit, Mapping):
        raise Final86Error("VCG 1.1 row lacks its method audit")
    frontiers = audit.get("frontiers")
    if not isinstance(frontiers, (list, tuple)) or not frontiers:
        raise Final86Error("VCG 1.1 row lacks its exact frontier records")
    complete = audit.get("complete_frontier_exactly_verified") is True
    candidates = []
    frontiers_complete = True
    for frontier in frontiers:
        if not isinstance(frontier, Mapping):
            raise Final86Error("VCG 1.1 frontier is not an object")
        frontiers_complete = frontiers_complete and (
            frontier.get("complete_frontier_exactly_verified") is True
        )
        observed = frontier.get("candidates")
        if not isinstance(observed, (list, tuple)):
            raise Final86Error("VCG 1.1 frontier candidates are malformed")
        candidates.extend(observed)
    if not candidates:
        raise Final86Error("VCG 1.1 exact frontiers contain no candidates")
    candidates_safe = all(
        isinstance(candidate, Mapping)
        and isinstance(candidate.get("certificate"), Mapping)
        and candidate["certificate"].get("status") == "SAFE"
        for candidate in candidates
    )
    compact = dict(audit)
    compact.pop("frontiers", None)
    compact.update(
        {
            "frontier_count": len(frontiers),
            "frontier_candidate_count": len(candidates),
            "all_frontiers_complete_exact": bool(frontiers_complete),
            "all_frontier_candidates_exact_safe": bool(candidates_safe),
        }
    )
    return bool(complete and frontiers_complete and candidates_safe), compact


def _v11_rows(
    *, root: Path, contract: Mapping, manifest: Mapping, auth: Mapping, execute: bool,
) -> tuple[list[dict], list[dict]]:
    rows, missing = [], []
    identities = _identity_map(manifest)
    records = _record_by_seed(manifest)
    _assert_cuda()
    torch.set_num_threads(1)
    arms = {arm.model_seed: arm for arm in auth["v11_arms"]}
    for model_seed in V11_MODEL_SEEDS:
        arm = arms[model_seed]
        for seed in INSTANCE_SEEDS:
            identity = identities[seed]
            input_contract = _input_contract(
                contract=contract, manifest=manifest, method=V11_METHOD,
                identity=identity, model_seed=model_seed, rng_index=None, rng_seed=None,
                checkpoint_sha256=arm.checkpoint_sha256,
            )

            def run_one(seed=seed, identity=identity, arm=arm, model_seed=model_seed):
                try:
                    instance = _load_instance(root, records[seed])
                    raw = v11._run_policy_once(
                        arm,
                        instance,
                        seed,
                        block_count=EXPECTED_DELIVERIES,
                        device=torch.device("cuda"),
                    )
                    exact_safe, compact_audit = _v11_exact_safety_audit(raw)
                    raw["all_selected_candidates_exact_safe"] = exact_safe
                    return _common_row(
                        method=V11_METHOD,
                        raw=raw,
                        identity=identity,
                        model_seed=model_seed,
                        rng_index=None,
                        rng_seed=None,
                        checkpoint_sha256=arm.checkpoint_sha256,
                        evidence={
                            "deployment_policy_digest": arm.deployment_policy_digest,
                            "selected_episode": arm.checkpoint_weight_episode,
                            "evaluation_policy": "deterministic_greedy_epsilon_zero",
                            "device": "cuda",
                            "method_audit": compact_audit,
                        },
                    )
                except Exception as error:
                    return _failed_row(
                        method=V11_METHOD, identity=identity, model_seed=model_seed,
                        rng_index=None, rng_seed=None,
                        checkpoint_sha256=arm.checkpoint_sha256, error=error,
                    )

            row = _load_or_execute(
                root=root, input_contract=input_contract, execute=execute, executor=run_one,
            )
            if row is None:
                missing.append(dict(input_contract))
            else:
                rows.append(row)
    return rows, missing


def _load_v23_runtime(
    *, seed: int, auth: Mapping,
) -> tuple[FinalV23Runtime, dict, object]:
    binding = auth["v23_binding"]
    completion = auth["v23_completions"][V23_MODEL_SEEDS.index(seed)]
    seed_dir = STABILITY_OUTPUT / f"seed-{seed}"
    profile = v23_stability._frozen_profile(seed)
    loaded = v23_stability.load_seed_best_candidate(
        seed,
        seed_dir / "best-development-candidate.pth",
        binding=binding,
        expected_best_sha256=completion["selected_best"]["best_checkpoint_raw_sha256"],
        manifest_path=seed_dir / "candidate-look-checkpoint-manifest.json",
        expected_manifest_sha256=completion["candidate_manifest_sha256"],
        validation_instance_manifest_path=seed_dir / "validation-instance-manifest.json",
        contract=seed_dir / "training-contract.json",
        device="cuda",
    )
    activation = v23_stability.activated_seed_profile(profile, binding)
    activation.__enter__()
    try:
        contract_payload = _load_json(seed_dir / "training-contract.json", name="VCG 2.3 training contract")
        runtime = FinalV23Runtime(
            args=SimpleNamespace(device="cuda"), contract=contract_payload,
        )
        runtime.agent = loaded["agent"]
        runtime.set_schedule_state(loaded["checkpoint"]["schedule_state"])
        _require_equal("VCG 2.3 runtime lambda", runtime.dual_lambda, loaded["checkpoint"]["validated_policy_lambda"])
        return runtime, loaded, activation
    except Exception:
        activation.__exit__(*sys.exc_info())
        raise


def _v23_rows(
    *, root: Path, contract: Mapping, manifest: Mapping, auth: Mapping, execute: bool,
) -> tuple[list[dict], list[dict]]:
    rows, missing = [], []
    identities = _identity_map(manifest)
    records = _record_by_seed(manifest)
    _assert_cuda()
    torch.set_num_threads(1)
    for model_seed in V23_MODEL_SEEDS:
        runtime = loaded = profile_context = None
        try:
            if execute:
                runtime, loaded, profile_context = _load_v23_runtime(
                    seed=model_seed, auth=auth,
                )
            checkpoint_sha = V23_SELECTED[model_seed][1]
            for seed in INSTANCE_SEEDS:
                instance_index = seed - INSTANCE_SEEDS[0]
                identity = identities[seed]
                for rng_index in RNG_INDICES:
                    rng_seed = V23_POLICY_RNG_BASE + 4 * instance_index + rng_index
                    input_contract = _input_contract(
                        contract=contract, manifest=manifest, method=V23_METHOD,
                        identity=identity, model_seed=model_seed, rng_index=rng_index,
                        rng_seed=rng_seed, checkpoint_sha256=checkpoint_sha,
                    )

                    def run_one(
                        seed=seed, identity=identity, model_seed=model_seed,
                        rng_index=rng_index, rng_seed=rng_seed,
                    ):
                        assert runtime is not None and loaded is not None
                        try:
                            instance = _load_instance(root, records[seed])
                            runtime.begin_validation_batch()
                            try:
                                raw = dict(runtime.run_final_episode(
                                    episode_instance=instance,
                                    instance_seed=seed,
                                    max_steps=MAX_STEPS,
                                    policy_rng_index=rng_index,
                                    policy_rng_seed=rng_seed,
                                ))
                            finally:
                                batch = runtime.end_validation_batch()
                            raw["validation_batch_state_unchanged"] = bool(
                                batch["training_agent_unchanged"]
                            )
                            raw = v2_train.normalize_v2_run(raw, rng_index)
                            return _common_row(
                                method=V23_METHOD,
                                raw=raw,
                                identity=identity,
                                model_seed=model_seed,
                                rng_index=rng_index,
                                rng_seed=rng_seed,
                                checkpoint_sha256=checkpoint_sha,
                                evidence={
                                    "selected_episode": loaded["selected_episode"],
                                    "policy_realization": raw["policy_mode"],
                                    "dual_lambda": raw["dual_lambda"],
                                    "schedule_state": loaded["checkpoint"]["schedule_state"],
                                    "selection_source_counts": raw["selection_source_counts"],
                                    "device": "cuda",
                                },
                            )
                        except Exception as error:
                            # If execution failed between batch boundaries, make
                            # a best effort to close the batch before preserving
                            # the failure row.  Never redraw this cell.
                            if runtime._validation_batch_signature_before is not None:
                                try:
                                    runtime.end_validation_batch()
                                except Exception:
                                    pass
                            return _failed_row(
                                method=V23_METHOD, identity=identity,
                                model_seed=model_seed, rng_index=rng_index,
                                rng_seed=rng_seed, checkpoint_sha256=checkpoint_sha,
                                error=error,
                            )

                    row = _load_or_execute(
                        root=root, input_contract=input_contract,
                        execute=execute, executor=run_one,
                    )
                    if row is None:
                        missing.append(dict(input_contract))
                    else:
                        rows.append(row)
        finally:
            if profile_context is not None:
                profile_context.__exit__(None, None, None)
    return rows, missing


def _baseline_runtime_args(auth: Mapping) -> SimpleNamespace:
    environment = {
        "grid_rows": 5,
        "grid_cols": 5,
        "number_blocks": 8,
        "arrival_rate": 10.0,
        "proc_mean": 80,
        "max_steps": MAX_STEPS,
    }
    return matched._baseline_args({"environment": environment}, torch.device("cpu"))


def _dynamic_rows(
    *, root: Path, contract: Mapping, manifest: Mapping, auth: Mapping, execute: bool,
) -> tuple[list[dict], list[dict]]:
    rows, missing = [], []
    identities = _identity_map(manifest)
    records = _record_by_seed(manifest)
    args = _baseline_runtime_args(auth)
    args.device = "cpu"
    for seed in INSTANCE_SEEDS:
        identity = identities[seed]
        input_contract = _input_contract(
            contract=contract, manifest=manifest, method=DYNAMIC_METHOD,
            identity=identity, model_seed=None, rng_index=None, rng_seed=None,
            checkpoint_sha256=None,
        )

        def run_one(seed=seed, identity=identity):
            try:
                instance = _load_instance(root, records[seed])
                raw = evaluate_assignment_ablation_one(
                    args,
                    seed,
                    None,
                    assignment_source=matched.DETERMINISTIC_METHOD_TO_SOURCE[matched.DYNAMIC_METHOD],
                    episode_instance=instance,
                )
                normalized = matched._normalize_deterministic_baseline(
                    matched.DYNAMIC_METHOD, raw, instance, identities,
                )
                return _common_row(
                    method=DYNAMIC_METHOD, raw=normalized, identity=identity,
                    model_seed=None, rng_index=None, rng_seed=None,
                    checkpoint_sha256=None,
                    evidence={
                        "assignment_source": normalized["assignment_source"],
                        "method_audit": normalized.get("method_audit"),
                        "device": "cpu",
                    },
                )
            except Exception as error:
                return _failed_row(
                    method=DYNAMIC_METHOD, identity=identity, model_seed=None,
                    rng_index=None, rng_seed=None, checkpoint_sha256=None, error=error,
                )

        row = _load_or_execute(
            root=root, input_contract=input_contract, execute=execute, executor=run_one,
        )
        if row is None:
            missing.append(dict(input_contract))
        else:
            rows.append(row)
    return rows, missing


@contextmanager
def _ga_validation_seed_base(value: int):
    implementation = repair_v2._impl
    original = implementation.base.GA_SEED_BASE
    implementation.base.GA_SEED_BASE = int(value)
    try:
        yield
    finally:
        implementation.base.GA_SEED_BASE = original


def _ga_rows(
    *, root: Path, contract: Mapping, manifest: Mapping, auth: Mapping, execute: bool,
) -> tuple[list[dict], list[dict]]:
    rows, missing = [], []
    identities = _identity_map(manifest)
    records = _record_by_seed(manifest)
    base_args = _baseline_runtime_args(auth)
    base_args.device = "cpu"
    implementation = repair_v2._impl
    method = implementation.REPAIRED_2009_METHOD
    _require_equal("final GA method identity", method, GA_METHOD)
    for seed in INSTANCE_SEEDS:
        instance_index = seed - INSTANCE_SEEDS[0]
        identity = identities[seed]
        for rng_index in RNG_INDICES:
            rng_seed = GA_RNG_BASE + GA_RNG_STRIDE * (4 * instance_index + rng_index)
            input_contract = _input_contract(
                contract=contract, manifest=manifest, method=GA_METHOD,
                identity=identity, model_seed=None, rng_index=rng_index,
                rng_seed=rng_seed, checkpoint_sha256=None,
            )

            def run_one(seed=seed, identity=identity, rng_index=rng_index, rng_seed=rng_seed):
                try:
                    instance = _load_instance(root, records[seed])
                    args = deepcopy(base_args)
                    # The evaluator adds the instance seed.  This offset makes
                    # the optimizer's actual base seed exactly the frozen value.
                    args.ga_seed_base = rng_seed - seed
                    raw = evaluate_assignment_ablation_one(
                        args,
                        seed,
                        None,
                        assignment_source=implementation.REPAIRED_METHOD_TO_SOURCE[method],
                        episode_instance=instance,
                    )
                    with _ga_validation_seed_base(rng_seed - seed):
                        normalized = implementation._normalize_repaired_run(
                            method, raw, instance=instance, identities=identities,
                        )
                    return _common_row(
                        method=GA_METHOD, raw=normalized, identity=identity,
                        model_seed=None, rng_index=rng_index, rng_seed=rng_seed,
                        checkpoint_sha256=None,
                        evidence={
                            "assignment_source": normalized["assignment_source"],
                            "assignment_source_version": normalized["assignment_source_version"],
                            "capacity_aware_source_audit": normalized["capacity_aware_source_audit"],
                            "capacity_aware_selector_audit": normalized["capacity_aware_selector_audit"],
                            "capacity_aware_reservation_audit": normalized["capacity_aware_reservation_audit"],
                            "device": "cpu",
                            "ga_rollout_base_seed": rng_seed,
                        },
                    )
                except Exception as error:
                    return _failed_row(
                        method=GA_METHOD, identity=identity, model_seed=None,
                        rng_index=rng_index, rng_seed=rng_seed,
                        checkpoint_sha256=None, error=error,
                    )

            row = _load_or_execute(
                root=root, input_contract=input_contract, execute=execute, executor=run_one,
            )
            if row is None:
                missing.append(dict(input_contract))
            else:
                rows.append(row)
    return rows, missing


def evaluate(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR, method: str = "all", execute: bool,
) -> dict:
    contract = authenticate_contract(output_dir=output_dir)
    manifest = authenticate_instance_manifest(output_dir=output_dir)
    root = Path(contract["output_dir"])
    auth = _authenticate_all()
    selected_methods = METHODS if method == "all" else (method,)
    if any(item not in METHODS for item in selected_methods):
        raise Final86Error("unknown method selection")
    dispatch = {
        V11_METHOD: _v11_rows,
        V23_METHOD: _v23_rows,
        DYNAMIC_METHOD: _dynamic_rows,
        GA_METHOD: _ga_rows,
    }
    summaries = {}
    for item in selected_methods:
        rows, missing = dispatch[item](
            root=root, contract=contract, manifest=manifest, auth=auth,
            execute=execute,
        )
        summaries[item] = {
            "loaded_or_executed": len(rows),
            "missing": len(missing),
            "strict_safe_complete": sum(bool(row["strict_safe_complete"]) for row in rows),
        }
    # Close the long execution trust window before returning.
    authenticate_contract(output_dir=root)
    authenticate_instance_manifest(output_dir=root)
    return {
        "protocol": PROTOCOL,
        "execute": bool(execute),
        "methods": summaries,
        "panel_opened": True,
    }


def _expected_ledger_paths(root: Path) -> set[Path]:
    paths = set()
    for seed in INSTANCE_SEEDS:
        for model_seed in V11_MODEL_SEEDS:
            paths.add(_ledger_path(root, V11_METHOD, seed, model_seed, None))
        for model_seed in V23_MODEL_SEEDS:
            for rng_index in RNG_INDICES:
                paths.add(_ledger_path(root, V23_METHOD, seed, model_seed, rng_index))
        paths.add(_ledger_path(root, DYNAMIC_METHOD, seed, None, None))
        for rng_index in RNG_INDICES:
            paths.add(_ledger_path(root, GA_METHOD, seed, None, rng_index))
    if len(paths) != 600:
        raise AssertionError("final ledger plan must contain 600 unique paths")
    return paths


def _validate_ledger_tree(root: Path, *, require_complete: bool) -> None:
    ledger_root = root / "run-ledger"
    if ledger_root.is_symlink():
        raise Final86Error("run-ledger must not be a symlink")
    if not ledger_root.exists():
        if require_complete:
            raise Final86Error("run-ledger is missing")
        return
    expected = _expected_ledger_paths(root)
    expected_directories = {ledger_root.absolute()} | {
        path.parent.absolute() for path in expected
    }
    observed = set()
    observed_directories = {ledger_root.absolute()}
    for path in ledger_root.rglob("*"):
        if path.is_symlink():
            raise Final86Error("run-ledger contains a symlink")
        if path.is_file():
            if path.suffix != ".json":
                raise Final86Error("run-ledger contains a non-JSON file")
            observed.add(path.absolute())
        elif path.is_dir():
            observed_directories.add(path.absolute())
        else:
            raise Final86Error("run-ledger contains an invalid entry")
    extra_directories = observed_directories - expected_directories
    if extra_directories:
        raise Final86Error(
            "run-ledger contains unexpected directories: "
            f"{sorted(map(str, extra_directories))[:3]!r}"
        )
    extras = observed - expected
    if extras:
        raise Final86Error(f"run-ledger contains unexpected files: {sorted(map(str, extras))[:3]!r}")
    if require_complete and observed != expected:
        raise Final86Error(f"final grid is incomplete: {len(observed)}/600 ledgers")
    if require_complete and observed_directories != expected_directories:
        raise Final86Error("final run-ledger directory tree is incomplete")


def _load_complete_grid(
    *, root: Path, contract: Mapping, manifest: Mapping, auth: Mapping,
) -> dict[str, list[dict]]:
    _validate_ledger_tree(root, require_complete=True)
    dispatch = {
        V11_METHOD: _v11_rows,
        V23_METHOD: _v23_rows,
        DYNAMIC_METHOD: _dynamic_rows,
        GA_METHOD: _ga_rows,
    }
    result = {}
    expected_counts = {V11_METHOD: 90, V23_METHOD: 360, DYNAMIC_METHOD: 30, GA_METHOD: 120}
    for method, loader in dispatch.items():
        rows, missing = loader(
            root=root, contract=contract, manifest=manifest, auth=auth, execute=False,
        )
        if missing or len(rows) != expected_counts[method]:
            raise Final86Error(f"{method} grid authentication failed")
        result[method] = rows
    return result


def _method_instance_points(method: str, rows: Sequence[Mapping]) -> dict[int, dict]:
    by_instance: dict[int, list[Mapping]] = defaultdict(list)
    for row in rows:
        by_instance[int(row["instance_seed"])].append(row)
    if set(by_instance) != set(INSTANCE_SEEDS):
        raise Final86Error(f"{method} does not cover the exact final panel")
    points = {}
    for seed in INSTANCE_SEEDS:
        cluster = by_instance[seed]
        if method == V11_METHOD:
            _require_equal("V1.1 instance model grid", sorted(row["model_seed"] for row in cluster), list(V11_MODEL_SEEDS))
            reduced_rows = cluster
        elif method == V23_METHOD:
            expected = {(model, rng) for model in V23_MODEL_SEEDS for rng in RNG_INDICES}
            observed = {(row["model_seed"], row["rng_index"]) for row in cluster}
            _require_equal("V2.3 instance nuisance grid", observed, expected)
            # First average four RNGs per model, then equally average models.
            model_points = []
            for model in V23_MODEL_SEEDS:
                model_rows = [row for row in cluster if row["model_seed"] == model]
                model_points.append({
                    metric: float(fmean(float(row[metric]) for row in model_rows))
                    for metric in METRICS
                })
            points[seed] = {
                metric: float(fmean(point[metric] for point in model_points))
                for metric in METRICS
            }
            continue
        elif method == GA_METHOD:
            _require_equal("GA instance RNG grid", sorted(row["rng_index"] for row in cluster), list(RNG_INDICES))
            reduced_rows = cluster
        elif method == DYNAMIC_METHOD:
            _require_equal("Dynamic PSLAP instance row count", len(cluster), 1)
            reduced_rows = cluster
        else:
            raise Final86Error("unknown method reduction")
        points[seed] = {
            metric: float(fmean(float(row[metric]) for row in reduced_rows))
            for metric in METRICS
        }
    return points


def _mean_ci(values: Sequence[float], critical: float) -> dict:
    values = tuple(float(value) for value in values)
    if len(values) != 30 or not all(math.isfinite(value) for value in values):
        raise Final86Error("paired final contrast requires 30 finite instance values")
    mean = float(fmean(values))
    standard_error = float(stdev(values) / math.sqrt(len(values)))
    half = float(critical * standard_error)
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower": mean - half,
        "upper": mean + half,
        "critical": float(critical),
        "df": 29,
        "n_episode_instances": 30,
    }


def _point_dominates(left: Mapping, right: Mapping) -> bool:
    coordinates = ("mean_absolute_error", "physical_rehandles_per_100_required_deliveries")
    return bool(
        all(float(left[name]) <= float(right[name]) for name in coordinates)
        and any(float(left[name]) < float(right[name]) for name in coordinates)
    )


def _build_report(
    *, contract: Mapping, manifest: Mapping, rows_by_method: Mapping[str, Sequence[Mapping]],
) -> dict:
    method_records = {}
    instance_points = {}
    for method in METHODS:
        rows = list(rows_by_method[method])
        safe = all(row["strict_safe_complete"] is True for row in rows)
        record = {
            "method": method,
            "expected_rows": {V11_METHOD: 90, V23_METHOD: 360, DYNAMIC_METHOD: 30, GA_METHOD: 120}[method],
            "observed_rows": len(rows),
            "strict_safe_complete_rows": sum(row["strict_safe_complete"] is True for row in rows),
            "whole_method_eligible": safe,
            "numeric_metrics_suppressed": not safe,
            "failures": [
                {
                    "instance_seed": row["instance_seed"],
                    "model_seed": row["model_seed"],
                    "rng_index": row["rng_index"],
                    "reason": row["method_failure_reason"],
                    "issues": row["safety_issues"],
                }
                for row in rows if not row["strict_safe_complete"]
            ],
        }
        if safe:
            points = _method_instance_points(method, rows)
            instance_points[method] = points
            record["metrics"] = {
                metric: float(fmean(points[seed][metric] for seed in INSTANCE_SEEDS))
                for metric in METRICS
            }
            record["instance_points"] = [
                {"instance_seed": seed, **points[seed]} for seed in INSTANCE_SEEDS
            ]
        else:
            record["metrics"] = None
            record["instance_points"] = None
        method_records[method] = record

    eligible = [method for method in METHODS if method_records[method]["whole_method_eligible"]]
    frontier = [
        method for method in eligible
        if not any(
            other != method
            and _point_dominates(method_records[other]["metrics"], method_records[method]["metrics"])
            for other in eligible
        )
    ]
    contrasts = {}
    if V23_METHOD in instance_points:
        for comparator in (V11_METHOD, DYNAMIC_METHOD, GA_METHOD):
            if comparator not in instance_points:
                contrasts[comparator] = {"evaluable": False}
                continue
            metrics = {}
            for metric in METRICS:
                differences = [
                    instance_points[V23_METHOD][seed][metric]
                    - instance_points[comparator][seed][metric]
                    for seed in INSTANCE_SEEDS
                ]
                critical = (
                    PRIMARY_BONFERRONI_T
                    if metric in (
                        "mean_absolute_error",
                        "physical_rehandles_per_100_required_deliveries",
                    )
                    else NOMINAL_T_95
                )
                metrics[metric] = _mean_ci(differences, critical)
            primary = metrics["mean_absolute_error"], metrics[
                "physical_rehandles_per_100_required_deliveries"
            ]
            v23_statistically_dominates = bool(
                all(item["upper"] <= 0.0 for item in primary)
                and any(item["upper"] < 0.0 for item in primary)
            )
            comparator_statistically_dominates = bool(
                all(item["lower"] >= 0.0 for item in primary)
                and any(item["lower"] > 0.0 for item in primary)
            )
            contrasts[comparator] = {
                "evaluable": True,
                "orientation": "VCG2.3_minus_comparator",
                "metrics": metrics,
                "v23_statistically_dominates_on_primary_coordinates": v23_statistically_dominates,
                "comparator_statistically_dominates_on_primary_coordinates": comparator_statistically_dominates,
                "absence_of_statistical_dominance_is_not_proof_of_nondominance": True,
            }
    report = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "contract_sha256": contract["contract_sha256"],
        "instance_manifest_sha256": manifest["manifest_sha256"],
        "run_count": sum(len(rows_by_method[method]) for method in METHODS),
        "episode_instance_count": 30,
        "inference_unit": "paired_EpisodeInstance_after_nested_nuisance_averaging",
        "methods": [method_records[method] for method in METHODS],
        "primary_pareto_coordinates": [
            "mean_absolute_error", "physical_rehandles_per_100_required_deliveries",
        ],
        "point_estimate_pareto_front": frontier,
        "paired_v23_contrasts": contrasts,
        "scalarization_used": False,
        "complete_case_filtering_used": False,
        "arrival_semantics": "exponential_rate_lambda_10_mean_gap_0.1",
        "performance_claim_scope": "prospective_final_panel_conditional_on_frozen_models_and_workload_generator",
    }
    report["report_sha256"] = _digest(report)
    return report


def analyze(*, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    contract = authenticate_contract(output_dir=output_dir)
    manifest = authenticate_instance_manifest(output_dir=output_dir)
    root = Path(contract["output_dir"])
    auth = _authenticate_all()
    rows_by_method = _load_complete_grid(
        root=root, contract=contract, manifest=manifest, auth=auth,
    )
    report = _build_report(
        contract=contract, manifest=manifest, rows_by_method=rows_by_method,
    )
    report_path = root / REPORT_NAME
    if report_path.exists():
        observed = _load_json(report_path, name="final report")
        _require_equal("final report deterministic reconstruction", observed, report)
    else:
        _atomic_json(report_path, report)
    # Reauthenticate every immutable input after report construction.
    authenticate_contract(output_dir=root)
    authenticate_instance_manifest(output_dir=root)
    _validate_ledger_tree(root, require_complete=True)
    audit = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "contract_raw_sha256": _sha256_file(root / CONTRACT_NAME, name="final86 contract"),
        "contract_sha256": contract["contract_sha256"],
        "activation_raw_sha256": _sha256_file(root / ACTIVATION_NAME, name="panel activation"),
        "instance_manifest_raw_sha256": _sha256_file(root / INSTANCE_MANIFEST_NAME, name="final instance manifest"),
        "instance_manifest_sha256": manifest["manifest_sha256"],
        "report_raw_sha256": _sha256_file(report_path, name="final report"),
        "report_sha256": report["report_sha256"],
        "ledger_count": 600,
        "all_sources_and_candidates_reauthenticated_after_analysis": True,
        "no_complete_case_filtering": True,
    }
    audit["audit_sha256"] = _digest(audit)
    audit_path = root / AUDIT_NAME
    if audit_path.exists():
        observed_audit = _load_json(audit_path, name="final audit")
        _require_equal("final audit deterministic reconstruction", observed_audit, audit)
    else:
        _atomic_json(audit_path, audit)
    return report


def status(*, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    contract = authenticate_contract(output_dir=output_dir)
    root = Path(contract["output_dir"])
    opened = (root / ACTIVATION_NAME).is_file()
    result = {
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "panel_opened": opened,
        "panel_materialized": (root / INSTANCE_MANIFEST_NAME).is_file(),
        "report_complete": (root / REPORT_NAME).is_file() and (root / AUDIT_NAME).is_file(),
    }
    if opened and result["panel_materialized"]:
        manifest = authenticate_instance_manifest(output_dir=root)
        result["instance_manifest_sha256"] = manifest["manifest_sha256"]
        expected = _expected_ledger_paths(root)
        observed = set((root / "run-ledger").rglob("*.json")) if (root / "run-ledger").is_dir() else set()
        result["ledger_count"] = len(observed & expected)
        result["expected_ledger_count"] = 600
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="freeze/authenticate without opening 86xxx")
    opener = subparsers.add_parser("open-panel", help="explicitly open and materialize 86xxx")
    opener.add_argument("--confirm", required=True)
    evaluator = subparsers.add_parser("evaluate", help="run or inspect the frozen row grid")
    evaluator.add_argument("--method", choices=("all", *METHODS), default="all")
    evaluator.add_argument("--execute", action="store_true")
    subparsers.add_parser("analyze", help="authenticate and aggregate the complete 600-row grid")
    subparsers.add_parser("status", help="show authenticated lifecycle state")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(output_dir=args.output_dir)
    elif args.command == "open-panel":
        result = open_panel(output_dir=args.output_dir, confirmation=args.confirm)
    elif args.command == "evaluate":
        result = evaluate(
            output_dir=args.output_dir, method=args.method, execute=args.execute,
        )
    elif args.command == "analyze":
        result = analyze(output_dir=args.output_dir)
    elif args.command == "status":
        result = status(output_dir=args.output_dir)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
