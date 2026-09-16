#!/usr/bin/env python3
"""Authenticate, evaluate, and analyze the matched Kim2020 supplement.

The completed training tree is immutable input.  ``prepare`` creates a new,
source-bound evaluation root before any rollout.  Evaluation is exactly the
frozen 3 x 12 x 5 stochastic grid; analysis is hierarchical and never uses
complete-case filtering.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import shutil
import stat
import statistics
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import vcg_v2_3_kim2020_supplement_protocol as training_protocol
from compare_kim2020_a3c_spatial import (
    KIM_STOCHASTIC,
    validate_locked_kim2020_checkpoint,
)
from contention_metrics import validate_contention_metric_record
from example.episode_instance import EpisodeInstance
from example.helper.timing_metrics import summarize_delivery_timing
from example.yard_geometry import geometry_metadata, make_shipyard_env
from PSLAP.kim2020_a3c_spatial import (
    DEPLOYMENT_STOCHASTIC,
    kim2020_deployment_digest,
)
from PSLAP.track_a import TRACK_A_KIM2020_A3C_SPATIAL
from track_b_urgency_evaluate import (
    DECISION_EPOCH_RESERVED,
    DURATION_AWARE_VARIANT,
    _evaluate_one,
    resolve_device,
)
from vcg_objective_audit import (
    DENSE_PIECEWISE,
    LEGACY_CLIPPED,
    TimingObjectiveSpec,
    delivery_reward,
)


PROTOCOL = "vcg_v2_3_kim2020_matched_supplement_evaluation_85xxx_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (0, 1, 2)
PANEL_SEEDS = tuple(range(85_000, 85_012))
ROLLOUTS = tuple(range(5))
EXPECTED_ROWS_PER_MODEL = 60
EXPECTED_ROWS = 180
POLICY_SEED_BASE = 632_000_000
SEALED_INSTANCE_SEEDS = frozenset(range(86_000, 86_030))
SEALED_POLICY_START = 622_000_000
SEALED_POLICY_STOP = 623_000_000
EXPECTED_DELIVERIES = 8

TRAINING_RELATIVE = Path("results/vcg-v2-3-kim2020-supplement-85k")
OUTPUT_RELATIVE = Path("results/vcg-v2-3-kim2020-supplement-evaluation-85k")
V23_SOURCE_RELATIVE = Path("results/vcg-constrained-v2-3-gamma1-ablation-seed10-200ep")
V11_CONTROL_RELATIVE = Path("results/vcg-dense-v1-1-v2-2-panel-control-12instance")
V22_SOURCE_RELATIVE = Path("results/vcg-constrained-v2-2-development-seed10-200ep")
REPAIR_REPORT_RELATIVE = Path(
    "results/vcg-v2-3-capacity-aware-ga-repair-v2-85k/expanded-report.json"
)
STABILITY_REPORT_RELATIVE = Path(
    "results/vcg-v2-3-seed-stability-analysis-repair-v2-85k/analysis-v2-report.json"
)
STABILITY_AUDIT_RELATIVE = Path(
    "results/vcg-v2-3-seed-stability-analysis-repair-v2-85k/analysis-v2-audit.json"
)
STABILITY_CONTRACT_RELATIVE = Path(
    "results/vcg-v2-3-seed-stability-analysis-repair-v2-85k/analysis-repair-contract.json"
)

EXPECTED_PARENT_RAW_SHA256 = {
    str(REPAIR_REPORT_RELATIVE): "7223fbeca579f78f06f7dbb68c9a65147201e7d9e66422f9ae73ec8e23129ac5",
    str(STABILITY_REPORT_RELATIVE): "bc6192ddc8188e12af37379f800f2d1a6c3a20b09bb0856685eaf1f892aa98a5",
    str(STABILITY_AUDIT_RELATIVE): "29c9c6423fa424467c7d63fc956a37b59c01fa51e092fa45f3292296a2e780dd",
    str(STABILITY_CONTRACT_RELATIVE): "2a8f92b8d067b2fd0e0fdcea00950316414782241e474b144a62000e081de8c3",
}
EXPECTED_PARENT_SELF_SHA256 = {
    str(REPAIR_REPORT_RELATIVE): ("report_sha256", "8da7fe1bc6f2356ae05e4d518bd07c850f766dd23ae11e36c5f600a8434e2b89"),
    str(STABILITY_REPORT_RELATIVE): ("report_sha256", "d4ae636d7e0d5cdaa1c7d6665d53c61ab9725affbab0f4be4b1f4522922cba94"),
    str(STABILITY_AUDIT_RELATIVE): ("audit_sha256", "5f458b1178e20bfe82604f794d4416878eab2cd69f25dacdb5b8da7ea5fde15f"),
    str(STABILITY_CONTRACT_RELATIVE): ("contract_sha256", "ed2a120228d8267f42b08473d2bfa72e3b335437c46c7e74599b174ae24fb104"),
}

EXPECTED_TRAINING_TREE_SHA256 = (
    "82e92ffa19f2e97d918157c5fce480fde7aaaff9fab2716fee59717e05cbd9b4"
)
EXPECTED_TRAINING_RAW_SHA256 = {
    "supplement-run-manifest.json": "805d1c0f57596cd0ea733aefe512a012e6fb4293fb08d2e90b3478a083323437",
    "training/seed-0/best.pth": "6a9f567768d0b7a35de6806c28e581acdf922678be07b31545e1842456bc2aae",
    "training/seed-0/latest.pth": "7a0878ec7fec2e54c2d2202190d09f2ed6641a588c1429b226150cac404f1966",
    "training/seed-0/training-history.json": "a70ba354adcf4c5a29aa3a1ec00554f3c810fba277ba206ed8bffc9aa566a473",
    "training/seed-0/training-summary.json": "92342d295c69abe83c7794ae254c31e6a56d3dfd11285c933c218c0bb6a48f9f",
    "training/seed-0/validation-instances.json": "8691b926a580ce7ebdff8721fd3126f2cba5ab3cf3b12f98b1d403f1e1302041",
    "training/seed-0/validation-instances/seed-73000000.json": "fd7b5cfe39f43fdf0a20a6c8cddbdbee9cf960fb584acc5590bd6bfe4f8907a7",
    "training/seed-0/validation-instances/seed-73000001.json": "0fa0684fab0be4d0ee38c35d6f15e0b9edb4f3cc4003939d87005100e393220c",
    "training/seed-0/validation-instances/seed-73000002.json": "dc93933956b2eee1d44f0346a633580f3274c2d172a1cdd45108b96db66348be",
    "training/seed-0/validation-instances/seed-73000003.json": "5cb71ee57fbd5625cb62f9f2491a909b21a21b318f3a47c59f999cb0fc3a4639",
    "training/seed-0/validation-instances/seed-73000004.json": "fcf9bb3ec7567d08a5908354aac90990ae36021e54d8778b2bf43a63f2f796c7",
    "training/seed-1/best.pth": "02fe55b37c330840f80337acff5efe4fb4dfdec9d6b3d771727579e0b82185a0",
    "training/seed-1/latest.pth": "f4c607cff3a3cd3b4ccdebba8a0c0fd9117e2b9ae50ca813a8c01a2447ee679c",
    "training/seed-1/training-history.json": "4bd0067f76ac5a85aa1e31d46f98d1353bc11328b581ded815c18d90672bb000",
    "training/seed-1/training-summary.json": "f94d39895ed4b213659627465e1e0b986584c1ba83e85f59676a87aefde429ec",
    "training/seed-1/validation-instances.json": "ca913045fee922b6c1068751e7ff9a4d7dbe561cd5bf2305326841de5e379abc",
    "training/seed-1/validation-instances/seed-73000000.json": "fd7b5cfe39f43fdf0a20a6c8cddbdbee9cf960fb584acc5590bd6bfe4f8907a7",
    "training/seed-1/validation-instances/seed-73000001.json": "0fa0684fab0be4d0ee38c35d6f15e0b9edb4f3cc4003939d87005100e393220c",
    "training/seed-1/validation-instances/seed-73000002.json": "dc93933956b2eee1d44f0346a633580f3274c2d172a1cdd45108b96db66348be",
    "training/seed-1/validation-instances/seed-73000003.json": "5cb71ee57fbd5625cb62f9f2491a909b21a21b318f3a47c59f999cb0fc3a4639",
    "training/seed-1/validation-instances/seed-73000004.json": "fcf9bb3ec7567d08a5908354aac90990ae36021e54d8778b2bf43a63f2f796c7",
    "training/seed-2/best.pth": "137db88a87accd95116ecaa07a7c52301cfccf9bf02bd91951755f794055b839",
    "training/seed-2/latest.pth": "91f51557e419a905975e600c8d043aa21331ba4fc675f563c15574f2371a04ea",
    "training/seed-2/training-history.json": "75b688bbfaa400b9842846333e22767d1ac749bd5377384fd6cec0196bc0b33c",
    "training/seed-2/training-summary.json": "f6d7f965189c3588f85bcca2f2cf497184f3f2ef9cc9525a1bd3ea21d1617875",
    "training/seed-2/validation-instances.json": "99d7f2227b6e309a55e5a2cf3560b6c4a78c9567da842d24d744930b17e341ea",
    "training/seed-2/validation-instances/seed-73000000.json": "fd7b5cfe39f43fdf0a20a6c8cddbdbee9cf960fb584acc5590bd6bfe4f8907a7",
    "training/seed-2/validation-instances/seed-73000001.json": "0fa0684fab0be4d0ee38c35d6f15e0b9edb4f3cc4003939d87005100e393220c",
    "training/seed-2/validation-instances/seed-73000002.json": "dc93933956b2eee1d44f0346a633580f3274c2d172a1cdd45108b96db66348be",
    "training/seed-2/validation-instances/seed-73000003.json": "5cb71ee57fbd5625cb62f9f2491a909b21a21b318f3a47c59f999cb0fc3a4639",
    "training/seed-2/validation-instances/seed-73000004.json": "fcf9bb3ec7567d08a5908354aac90990ae36021e54d8778b2bf43a63f2f796c7",
}
EXPECTED_BEST_EPISODES = {0: 100, 1: 200, 2: 700}
EXPECTED_BEST_DEPLOYMENT_DIGESTS = {
    0: "742bdb6819f9f2163c5a83b455cde8d24bb4488295b3725c9c166519995d1615",
    1: "05f56fd73967f4130b030c32ac9b23e3448f8a18abc5de52f6d24d1f28983b35",
    2: "cc72f6a6360b53ebcf7b3ceb098e31459564cbc778699eef24432fba90a0c85e",
}

INSTANCE_MANIFEST_RAW_SHA256 = (
    "23a835d58da883be8761ad4fe7d8177f7b7a0db4bb135e7acbd183f2470f4f8e"
)
INSTANCE_MANIFEST_SELF_SHA256 = (
    "0ead7930ea4be1eeea36efcc3422f88f4b55100d3eb7c2325eba7294db0f9559"
)

EXPECTED_FIXED_SOURCE_SHA256 = {
    "vcg_objective_audit.py": "50581f31f497c9adacc42d57214eb58f995988ab19abd33064f11e51951454d0",
    "compare_kim2020_a3c_spatial.py": "60d6bf99902a7d04b403fbe8a232852dd0e53874d996311771f7032244766298",
    "track_b_urgency_evaluate.py": "49fd55a96b24c41392cb00d55e44621c902723a7d5b6ff049e41c7df41e22dc5",
}

NEW_SOURCE_PATHS = (
    "vcg_v2_3_kim2020_supplement_evaluation.py",
    "experiments/vcg_v2_3_kim2020_supplement_evaluation_85k/run.sh",
    "vcg_objective_audit.py",
    "compare_kim2020_a3c_spatial.py",
    "track_b_urgency_evaluate.py",
)

FROZEN_OBJECTIVE_SPEC = TimingObjectiveSpec.dense(
    dense_b=40.0,
    lambda_abs=1.5,
    lambda_outside=0.5,
    window=20.0,
)

DISPLAY_LABELS = {
    "vcg_constrained_v2_3_equal_stability_seeds_11_13": (
        "VCG constrained V2.3 (equal clean seeds 11-13)"
    ),
    "vcg_dense_v1_1_selected_three_seed": "VCG dense V1.1 (three selected training seeds)",
    "duration_aware_nearest_free": "Duration-aware nearest-free",
    "duration_aware_dynamic_pslap": "Duration-aware dynamic PSLAP",
    "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial": (
        "Park-Seo (2009)-inspired rolling GA, capacity-aware"
    ),
    "duration_aware_pslap_ga_duration_aware_rolling_capacity_aware_partial": (
        "Duration-aware rolling GA, capacity-aware"
    ),
    "duration_aware_pslap_ga_operational_rolling_capacity_aware_partial": (
        "Operational rolling GA, capacity-aware"
    ),
    "duration_aware_enhanced_complete_rolling_ga_capacity_aware_partial": (
        "Enhanced-complete rolling GA, capacity-aware"
    ),
    KIM_STOCHASTIC: "Kim et al. (2020)-inspired spatial A3C (stochastic)",
}

ELIGIBLE_BASELINE_IDS = (
    "vcg_dense_v1_1_selected_three_seed",
    "duration_aware_nearest_free",
    "duration_aware_dynamic_pslap",
    "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial",
    "duration_aware_pslap_ga_duration_aware_rolling_capacity_aware_partial",
    "duration_aware_pslap_ga_operational_rolling_capacity_aware_partial",
    "duration_aware_enhanced_complete_rolling_ga_capacity_aware_partial",
)

TRACK_B_METHOD = "reserved_cell_assignment_source_ablation"
TRACK_B_METHOD_VARIANT = (
    "reserved_cell_assignment_source_ablation__kim2020_a3c_spatial_adapted"
)
TRACK_B_EVALUATION_SCOPE = "repaired_learning_augmented_scheduler_candidate"
TRACK_B_CONTROLLER_ARCHITECTURE = (
    "deterministic_duration_aware_reserved_cell_hierarchy_v3"
)
TRACK_B_CONTROLLER_ACTION_INTERFACE = (
    "interleaved_atomic_scheduler_reserved_cell_v4"
)
TRACK_B_POLICY_REALIZATION = "deterministic_bound_assignment_duration_lookahead_v3"
TRACK_B_DECISION_EPOCH_CONTRACT = (
    "due_then_exact_bound_cell_lookahead_then_capacity_then_event_v3"
)
TRACK_B_ASSIGNMENT_COMMITMENT_CONTRACT = "decision_epoch_proposal_bound_once_v2"
TRACK_B_RESERVATION_CONTRACT = "bound_cell_parameterized_accept_store_v1"

REPORT_METRICS = (
    "mean_dense_objective_return",
    "mean_absolute_error",
    "mean_signed_deviation",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "mean_steps",
    "physical_rehandles_per_100_required_deliveries",
)
TABLE_METRICS = tuple(
    metric for metric in REPORT_METRICS if metric != "mean_signed_deviation"
)
GATE_METRICS = (
    "mean_absolute_error",
    "physical_rehandles_per_100_required_deliveries",
)
NUMERIC_TABLE_METHOD_IDS = (
    "vcg_constrained_v2_3_equal_stability_seeds_11_13",
    *ELIGIBLE_BASELINE_IDS,
    KIM_STOCHASTIC,
)

FROZEN_GEOMETRY = {
    "geometry_contract": "open_yard_right_aligned_bottom_gate_v1",
    "grid_rows": 5,
    "grid_cols": 5,
    "start_state": [1, 1],
    "door_cell": [0, 3],
    "pickup_cell": [1, 3],
    "waiting_cell": [0, 3],
    "exit_cells": [[4, 1], [4, 2], [4, 3]],
    "storage_positions": [
        [1, 1], [1, 2], [2, 1], [2, 2],
        [2, 3], [3, 1], [3, 2], [3, 3],
    ],
    "room_rows": ["###.#", "#...#", "#...#", "#...#", "#...#"],
    "geometry_signature": "f4d984dd3f4c27c7",
    "geometry_regime": "ordinary_default_gate",
    "requested_exit_width": None,
    "actual_exit_width": 3,
    "storage_cell_count": 8,
    "block_count": 8,
    "nominal_storage_density": 1.0,
}


class EvaluationProtocolError(RuntimeError):
    """A fail-closed supplement contract violation."""


def _reject_json_constant(value: str) -> None:
    raise EvaluationProtocolError(f"non-finite JSON constant is forbidden: {value}")


def _reject_symlink_components(path: Path) -> None:
    """Reject a symlink at the leaf or in any existing ancestor component."""

    absolute = path.absolute()
    chain = (absolute, *absolute.parents)
    for component in chain:
        if component.is_symlink():
            raise EvaluationProtocolError(
                f"symlink component forbidden in authenticated path: {component}"
            )


def _load_json(path: Path, *, expected_type: type | None = None) -> Any:
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise EvaluationProtocolError(f"missing/nonregular JSON artifact: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationProtocolError(f"invalid JSON artifact: {path}") from exc
    if expected_type is not None and not isinstance(value, expected_type):
        raise EvaluationProtocolError(
            f"{path} must contain {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise EvaluationProtocolError("non-finite value may not enter an authenticated artifact")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    _reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise EvaluationProtocolError(f"missing/nonregular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_regular_bytes(path: Path) -> tuple[bytes, str]:
    """Open with O_NOFOLLOW and bind parsing to the exact hashed byte string."""

    _reject_symlink_components(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvaluationProtocolError(f"cannot open authenticated file: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise EvaluationProtocolError(f"authenticated path is not a file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    return payload, hashlib.sha256(payload).hexdigest()


def _load_checkpoint_bound(path: Path, expected_raw_sha256: str) -> dict[str, Any]:
    payload, observed = _read_regular_bytes(path)
    _require_equal(f"checkpoint raw SHA {path}", observed, expected_raw_sha256)
    loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise EvaluationProtocolError(f"checkpoint payload is not a mapping: {path}")
    return loaded


def _add_self_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    if field in value:
        raise EvaluationProtocolError(f"self-hash field already present: {field}")
    result = dict(value)
    result[field] = _digest_json(result)
    return result


def _verify_self_hash(
    value: Mapping[str, Any], field: str, *, label: str
) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise EvaluationProtocolError(f"{label} has no valid {field}")
    unhashed = dict(value)
    unhashed.pop(field, None)
    observed = _digest_json(unhashed)
    if observed != claimed:
        raise EvaluationProtocolError(
            f"{label} self-hash mismatch: {observed} != {claimed}"
        )
    return claimed


def _json_payload(value: Any) -> bytes:
    return (
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_new_json(path: Path, value: Any) -> str:
    """Atomically create one write-once JSON artifact without overwriting."""

    if path.exists() or path.is_symlink():
        raise EvaluationProtocolError(f"refusing to overwrite artifact: {path}")
    payload = _json_payload(value)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise EvaluationProtocolError(f"stale temporary artifact: {temporary}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise EvaluationProtocolError(f"concurrent artifact creation: {path}") from exc
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def _strict_equivalent(observed: Any, expected: Any) -> bool:
    """JSON-like equality with bool kept distinct from numeric 0/1."""

    observed = _json_safe(observed)
    expected = _json_safe(expected)
    if isinstance(observed, bool) or isinstance(expected, bool):
        return type(observed) is bool and type(expected) is bool and observed == expected
    if isinstance(observed, (int, float)) or isinstance(expected, (int, float)):
        return type(observed) is type(expected) and observed == expected
    if isinstance(observed, Mapping) or isinstance(expected, Mapping):
        return (
            isinstance(observed, Mapping)
            and isinstance(expected, Mapping)
            and set(observed) == set(expected)
            and all(
                _strict_equivalent(observed[key], expected[key])
                for key in observed
            )
        )
    if isinstance(observed, (list, tuple)) or isinstance(expected, (list, tuple)):
        return (
            isinstance(observed, (list, tuple))
            and isinstance(expected, (list, tuple))
            and len(observed) == len(expected)
            and all(
                _strict_equivalent(left, right)
                for left, right in zip(observed, expected)
            )
        )
    if isinstance(observed, (set, frozenset)) or isinstance(
        expected, (set, frozenset)
    ):
        return (
            isinstance(observed, (set, frozenset))
            and isinstance(expected, (set, frozenset))
            and observed == expected
        )
    return observed == expected


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if not _strict_equivalent(observed, expected):
        raise EvaluationProtocolError(
            f"{label} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise EvaluationProtocolError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationProtocolError(f"{label} must be finite")
    return result


def _count(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise EvaluationProtocolError(f"{label} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise EvaluationProtocolError(f"{label} must be a nonnegative integer")
    return result


def _tree_file_hashes(root: Path) -> dict[str, str]:
    _reject_symlink_components(root)
    if root.is_symlink() or not root.is_dir():
        raise EvaluationProtocolError(f"tree root is missing/nonregular: {root}")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise EvaluationProtocolError(f"symlink forbidden in authenticated tree: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise EvaluationProtocolError(f"non-file tree entry: {path}")
        result[str(path.relative_to(root))] = _sha256(path)
    return result


def _fixed_runtime() -> dict[str, Any]:
    deterministic = torch.are_deterministic_algorithms_enabled()
    thread_count = torch.get_num_threads()
    if deterministic is not True or thread_count != 1:
        raise EvaluationProtocolError(
            "runtime is not fixed: deterministic algorithms and one Torch thread are required"
        )
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "evaluation_device": "cpu",
        "deterministic_algorithms": deterministic,
        "torch_num_threads": thread_count,
    }


def _configure_runtime() -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    _fixed_runtime()


def _resolve_roots(
    project_root: Path, training_root: Path, output_root: Path
) -> tuple[Path, Path, Path]:
    requested = (project_root.absolute(), training_root.absolute(), output_root.absolute())
    for label, path in zip(("project", "training", "output"), requested):
        if path.is_symlink():
            raise EvaluationProtocolError(f"{label} root argument may not be a symlink")
        if path.resolve() != path:
            raise EvaluationProtocolError(f"{label} root argument must be canonical")
    project, training, output = (path.resolve() for path in requested)
    _require_equal("training root", training, project / TRAINING_RELATIVE)
    _require_equal("evaluation output root", output, project / OUTPUT_RELATIVE)
    if output == training or output.is_relative_to(training) or training.is_relative_to(output):
        raise EvaluationProtocolError("training and evaluation roots must be disjoint")
    immutable_roots = (
        project / V23_SOURCE_RELATIVE,
        project / V11_CONTROL_RELATIVE,
        project / V22_SOURCE_RELATIVE,
        project / REPAIR_REPORT_RELATIVE.parent,
        project / STABILITY_REPORT_RELATIVE.parent,
    )
    for source in immutable_roots:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise EvaluationProtocolError(
                f"evaluation output must be disjoint from immutable source: {source}"
            )
    return project, training, output


def policy_seed(model_seed: int, panel_index: int, rollout: int) -> int:
    if model_seed not in MODEL_SEEDS:
        raise EvaluationProtocolError(f"invalid model seed: {model_seed}")
    if not 0 <= panel_index < len(PANEL_SEEDS):
        raise EvaluationProtocolError(f"invalid panel index: {panel_index}")
    if rollout not in ROLLOUTS:
        raise EvaluationProtocolError(f"invalid rollout: {rollout}")
    return POLICY_SEED_BASE + 1000 * model_seed + 10 * panel_index + rollout


def expected_grid(model_seed: int | None = None) -> list[dict[str, int]]:
    seeds = MODEL_SEEDS if model_seed is None else (model_seed,)
    rows: list[dict[str, int]] = []
    for selected_seed in seeds:
        for panel_index, instance_seed in enumerate(PANEL_SEEDS):
            for rollout in ROLLOUTS:
                rows.append(
                    {
                        "model_seed": selected_seed,
                        "panel_index": panel_index,
                        "instance_seed": instance_seed,
                        "rollout": rollout,
                        "policy_seed": policy_seed(selected_seed, panel_index, rollout),
                    }
                )
    expected = EXPECTED_ROWS if model_seed is None else EXPECTED_ROWS_PER_MODEL
    if len(rows) != expected or len({row["policy_seed"] for row in rows}) != expected:
        raise EvaluationProtocolError("evaluation-grid cardinality/uniqueness failure")
    if any(row["instance_seed"] in SEALED_INSTANCE_SEEDS for row in rows):
        raise EvaluationProtocolError("sealed 86xxx instance entered evaluation grid")
    if any(SEALED_POLICY_START <= row["policy_seed"] < SEALED_POLICY_STOP for row in rows):
        raise EvaluationProtocolError("sealed 622m policy seed entered evaluation grid")
    return rows


def _authenticate_parent_artifacts(project_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for relative, expected_raw in EXPECTED_PARENT_RAW_SHA256.items():
        path = project_root / relative
        observed_raw = _sha256(path)
        _require_equal(f"parent raw SHA {relative}", observed_raw, expected_raw)
        payload = _load_json(path, expected_type=dict)
        field, expected_self = EXPECTED_PARENT_SELF_SHA256[relative]
        observed_self = _verify_self_hash(payload, field, label=relative)
        _require_equal(f"parent self SHA {relative}", observed_self, expected_self)
        loaded[relative] = payload
        result[relative] = {
            "raw_sha256": observed_raw,
            "self_hash_field": field,
            "self_sha256": observed_self,
        }

    repair = loaded[str(REPAIR_REPORT_RELATIVE)]
    stability = loaded[str(STABILITY_REPORT_RELATIVE)]
    stability_audit = loaded[str(STABILITY_AUDIT_RELATIVE)]
    stability_contract = loaded[str(STABILITY_CONTRACT_RELATIVE)]
    for label, payload in (("repair report", repair), ("stability report", stability)):
        _require_equal(f"{label} final panel", payload.get("final_86xxx_panel_opened"), False)
        _require_equal(f"{label} performance claim", payload.get("performance_claim_authorized"), False)
        _require_equal(f"{label} confirmatory claim", payload.get("confirmatory_claim_authorized"), False)
    _require_equal("stability status", stability.get("status"), "passed")
    _require_equal("stability strong result", stability.get("strong_result"), True)
    equal_seed = stability.get("equal_seed_aggregate")
    if not isinstance(equal_seed, dict):
        raise EvaluationProtocolError("stability equal-seed aggregate is missing")
    _require_equal("stability equal-seed availability", equal_seed.get("available"), True)
    _require_equal("stability equal-seed model seeds", equal_seed.get("model_seeds"), [11, 12, 13])
    _require_equal("stability equal-seed non-dominance", equal_seed.get("non_dominated"), True)
    _require_equal("stability audit status", stability_audit.get("status"), "complete")
    _require_equal("stability audit rollout count", stability_audit.get("training_or_evaluation_rollouts_executed"), 0)
    _require_equal("stability audit complete-case filtering", stability_audit.get("complete_case_filtering_used"), False)
    _require_equal("stability audit final panel", stability_audit.get("final_86xxx_panel_opened"), False)
    _require_equal("stability audit protected policy panel", stability_audit.get("protected_622xxx_policy_rng_opened"), False)
    _require_equal("stability contract development only", stability_contract.get("development_only"), True)
    _require_equal("stability contract final panel", stability_contract.get("final_86xxx_panel_opened"), False)
    _require_equal("stability contract protected policy panel", stability_contract.get("protected_622xxx_policy_rng_opened"), False)
    _require_equal("stability contract no rollout authorization", stability_contract.get("training_or_evaluation_rollouts_authorized"), False)
    _require_equal("stability contract no complete cases", stability_contract.get("complete_case_filtering_authorized"), False)
    _require_equal("seed10 not a gate comparator", stability_contract.get("development_seed10_is_not_a_gate_comparator"), True)
    _require_equal(
        "frozen stability comparator registry",
        stability_contract.get("frozen_baseline_method_ids"),
        list(ELIGIBLE_BASELINE_IDS),
    )
    repair_gate = repair.get("development_method_capacity_or_pareto_screen")
    if not isinstance(repair_gate, Mapping):
        raise EvaluationProtocolError("repair report frozen Pareto screen is missing")
    _require_equal(
        "frozen Pareto coordinates",
        repair_gate.get("dominance_coordinates"),
        list(GATE_METRICS),
    )
    _require_equal("frozen Pareto scalarization", repair_gate.get("scalarization_used"), False)
    repair_trust = stability_contract.get("repair_v2_trust")
    if not isinstance(repair_trust, Mapping):
        raise EvaluationProtocolError("stability contract repair-V2 trust is missing")
    repair_root = Path(repair_trust.get("root", ""))
    _require_equal(
        "repair-V2 trust root",
        repair_root.resolve(),
        (project_root / REPAIR_REPORT_RELATIVE.parent).resolve(),
    )
    raw_registry = repair_trust.get("raw_sha256")
    ledger_registry = repair_trust.get("repair_ledger_raw_sha256")
    if not isinstance(raw_registry, Mapping) or not isinstance(
        ledger_registry, Mapping
    ):
        raise EvaluationProtocolError("repair-V2 raw registries are missing")
    fixed_repair_paths = {
        "audit": "expanded-audit.json",
        "contract": "repair-contract.json",
        "preflight": "preflight.json",
        "report": "expanded-report.json",
        "runs": "expanded-runs.csv",
    }
    _require_equal("repair-V2 fixed raw registry", set(raw_registry), set(fixed_repair_paths))
    expected_repair_tree = {
        relative: raw_registry[label]
        for label, relative in fixed_repair_paths.items()
    }
    for identity, expected_raw in ledger_registry.items():
        if not isinstance(identity, str) or ":" not in identity:
            raise EvaluationProtocolError("repair-V2 ledger identity is malformed")
        method_id, seed_text = identity.rsplit(":", 1)
        seed = int(seed_text)
        if method_id not in ELIGIBLE_BASELINE_IDS[3:] or seed not in PANEL_SEEDS:
            raise EvaluationProtocolError("repair-V2 ledger is outside the frozen grid")
        relative = f"run-ledger/{method_id}/seed-{seed}.json"
        if relative in expected_repair_tree:
            raise EvaluationProtocolError("duplicate repair-V2 ledger path")
        expected_repair_tree[relative] = expected_raw
    _require_equal("repair-V2 ledger count", len(ledger_registry), 48)
    _require_equal(
        "repair-V2 exact pinned tree",
        _tree_file_hashes(repair_root),
        expected_repair_tree,
    )
    repair_contract = _load_json(repair_root / "repair-contract.json", expected_type=dict)
    repair_audit = _load_json(repair_root / "expanded-audit.json", expected_type=dict)
    _require_equal(
        "repair-V2 contract self SHA",
        _verify_self_hash(repair_contract, "contract_sha256", label="repair-V2 contract"),
        repair_trust.get("contract_sha256"),
    )
    _require_equal(
        "repair-V2 audit self SHA",
        _verify_self_hash(repair_audit, "audit_sha256", label="repair-V2 audit"),
        repair_trust.get("audit_sha256"),
    )
    _require_equal(
        "repair-V2 report self SHA",
        repair.get("report_sha256"),
        repair_trust.get("report_sha256"),
    )
    _require_equal(
        "repair-V2 whole-safe comparator registry",
        repair_trust.get("whole_safe_baseline_method_ids"),
        list(ELIGIBLE_BASELINE_IDS),
    )
    return result


def _authenticate_panel(
    project_root: Path,
) -> tuple[dict[str, Any], dict[int, EpisodeInstance]]:
    control_root = project_root / V11_CONTROL_RELATIVE
    manifest_path = control_root / "instance-manifest.json"
    _require_equal("instance manifest raw SHA", _sha256(manifest_path), INSTANCE_MANIFEST_RAW_SHA256)
    manifest = _load_json(manifest_path, expected_type=dict)
    _require_equal(
        "instance manifest self SHA",
        _verify_self_hash(manifest, "manifest_sha256", label="85k instance manifest"),
        INSTANCE_MANIFEST_SELF_SHA256,
    )
    _require_equal("instance manifest seeds", manifest.get("seeds"), list(PANEL_SEEDS))
    _require_equal("instance manifest development flag", manifest.get("development_only"), True)
    _require_equal("instance manifest final panel", manifest.get("final_86xxx_panel_opened"), False)
    records = manifest.get("instances")
    if not isinstance(records, list) or len(records) != len(PANEL_SEEDS):
        raise EvaluationProtocolError("instance manifest must contain the exact 12 records")
    by_seed = {int(item.get("instance_seed")): item for item in records}
    _require_equal("instance record seed set", sorted(by_seed), list(PANEL_SEEDS))

    template_env = make_shipyard_env(
        arrival_rate=10.0,
        proc_mean=80.0,
        grid_rows=5,
        grid_cols=5,
        exit_width=None,
        number_blocks=EXPECTED_DELIVERIES,
    )
    instances: dict[int, EpisodeInstance] = {}
    normalized_records: list[dict[str, Any]] = []
    for panel_index, seed in enumerate(PANEL_SEEDS):
        record = by_seed[seed]
        path = control_root / "instances" / f"seed-{seed}.json"
        _require_equal("saved instance path", Path(record.get("saved_path", "")).resolve(), path.resolve())
        instance_bytes, raw_sha = _read_regular_bytes(path)
        _require_equal(f"seed {seed} saved raw SHA", raw_sha, record.get("saved_file_sha256"))
        try:
            instance = EpisodeInstance.from_json(instance_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
            raise EvaluationProtocolError(f"seed {seed} saved instance is invalid") from exc
        instance.validate_for(template_env)
        canonical_sha = hashlib.sha256(instance.to_json().encode("utf-8")).hexdigest()
        _require_equal(f"seed {seed} embedded seed", instance.seed, seed)
        _require_equal(f"seed {seed} instance id", instance.instance_id, record.get("episode_instance_id"))
        _require_equal(f"seed {seed} schedule id", instance.schedule_id, record.get("schedule_id"))
        _require_equal(f"seed {seed} canonical SHA", canonical_sha, record.get("episode_instance_sha256"))
        instances[seed] = instance
        normalized_records.append(
            {
                "panel_index": panel_index,
                "instance_seed": seed,
                "episode_instance_id": instance.instance_id,
                "schedule_id": instance.schedule_id,
                "episode_instance_sha256": canonical_sha,
                "saved_file_raw_sha256": raw_sha,
                "saved_path": str(path.resolve()),
            }
        )

    # Cross-bind the panel identities to every row in the immutable repair-V2
    # baseline universe, not merely to the earlier instance manifest.
    repair = _load_json(project_root / REPAIR_REPORT_RELATIVE, expected_type=dict)
    repair_rows = repair.get("runs")
    if not isinstance(repair_rows, list):
        raise EvaluationProtocolError("repair-V2 report has no run registry")
    expected_by_seed = {row["instance_seed"]: row for row in normalized_records}
    for seed in PANEL_SEEDS:
        identities = {
            (
                row.get("episode_instance_id"),
                row.get("schedule_id"),
                row.get("episode_instance_sha256"),
            )
            for row in repair_rows
            if row.get("instance_seed") == seed
        }
        expected_identity = expected_by_seed[seed]
        _require_equal(
            f"seed {seed} repair identity set",
            identities,
            {
                (
                    expected_identity["episode_instance_id"],
                    expected_identity["schedule_id"],
                    expected_identity["episode_instance_sha256"],
                )
            },
        )
    panel_record = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_raw_sha256": INSTANCE_MANIFEST_RAW_SHA256,
        "manifest_self_sha256": INSTANCE_MANIFEST_SELF_SHA256,
        "records": normalized_records,
        "record_count": len(normalized_records),
        "records_sha256": _digest_json(normalized_records),
    }
    return panel_record, instances


def _authenticate_sources(
    project_root: Path, training_manifest: Mapping[str, Any]
) -> dict[str, str]:
    inherited = training_manifest.get("source_sha256")
    if not isinstance(inherited, dict):
        raise EvaluationProtocolError("training manifest source registry is missing")
    observed: dict[str, str] = {}
    for relative, expected in inherited.items():
        path = project_root / relative
        current = _sha256(path)
        _require_equal(f"inherited source {relative}", current, expected)
        observed[relative] = current
    for relative in NEW_SOURCE_PATHS:
        path = project_root / relative
        current = _sha256(path)
        hard_expected = EXPECTED_FIXED_SOURCE_SHA256.get(relative)
        if hard_expected is not None:
            _require_equal(f"fixed evaluator dependency {relative}", current, hard_expected)
        if relative in observed:
            _require_equal(f"overlapping source registry {relative}", current, observed[relative])
        observed[relative] = current
    installed = project_root / "vcg_v2_3_kim2020_supplement_evaluation.py"
    if Path(__file__).resolve() != installed.resolve():
        raise EvaluationProtocolError("evaluator must run from its installed project path")
    return dict(sorted(observed.items()))


def _validate_common_checkpoint(
    payload: Mapping[str, Any], *, model_seed: int, kind: str, episode: int
) -> None:
    expected = {
        "checkpoint_kind": kind,
        "trainer_version": "kim2020_a3c_spatial_adapted_trainer_v7",
        "adaptation_contract": "kim_jeong_shin_2020_spatial_a3c_conceptual_adaptation_v6",
        "validation_contract": "fixed_instances_map_once_stochastic_fixed_policy_seed_rollouts_v1",
        "model_selection_contract": "primary_stochastic_strict_success_then_mean_obstructive_moves_v2",
        "exact_recovery_fallback_contract": "overdue_greedy_stall_goal_directed_strict_macro_witness_v1",
        "exact_recovery_max_nodes": 100000,
        "adaptation_status": "adaptation_not_exact_reproduction",
        "exact_paper_reproduction": False,
        "assignment_source": "kim2020_a3c_spatial_adapted",
        "training_lambda": 10.0,
        "training_mu": 80.0,
        "training_seed": model_seed,
        "completed_training_episodes": episode,
        "training_instance_seed_base": 70_000_000,
        "validation_instance_seeds": list(training_protocol.VALIDATION_INSTANCE_SEEDS),
        "stochastic_rollouts": 5,
        "validation_policy_seed_base": 74_000_000,
        "deterministic_algorithms": True,
        "resumed_from_checkpoint": None,
        "resume_start_episode": 0,
        "deployment_mode": DEPLOYMENT_STOCHASTIC,
    }
    for field, expected_value in expected.items():
        _require_equal(f"seed {model_seed} {kind} checkpoint {field}", payload.get(field), expected_value)


def _authenticate_training_seed(
    project_root: Path, training_root: Path, model_seed: int
) -> dict[str, Any]:
    seed_dir = training_root / "training" / f"seed-{model_seed}"
    summary_path = seed_dir / "training-summary.json"
    history_path = seed_dir / "training-history.json"
    best_path = seed_dir / "best.pth"
    latest_path = seed_dir / "latest.pth"
    validation_manifest_path = seed_dir / "validation-instances.json"
    summary = _load_json(summary_path, expected_type=dict)
    history = _load_json(history_path, expected_type=list)
    validation_manifest = _load_json(validation_manifest_path, expected_type=list)

    expected_summary = {
        "status": "completed",
        "trainer_version": "kim2020_a3c_spatial_adapted_trainer_v7",
        "method": "kim2020_a3c_spatial_adapted",
        "adaptation_contract": "kim_jeong_shin_2020_spatial_a3c_conceptual_adaptation_v6",
        "exact_paper_reproduction": False,
        "model_selection_contract": "primary_stochastic_strict_success_then_mean_obstructive_moves_v2",
        "validation_contract": "fixed_instances_map_once_stochastic_fixed_policy_seed_rollouts_v1",
        "training_seed": model_seed,
        "python_hash_seed": str(model_seed),
        "deterministic_algorithms": True,
        "training_device": "cuda",
        "cublas_workspace_config": ":4096:8",
        "episodes": 1000,
        "resumed_from_checkpoint": None,
        "resume_start_episode": 0,
        "max_steps": 2000,
        "training_instance_seed_start": 70_000_001 + model_seed * 1_000_000,
        "training_instance_seed_end": 70_001_000 + model_seed * 1_000_000,
        "validation_policy_seed_base": 74_000_000,
        "stochastic_rollouts": 5,
        "best_episode": EXPECTED_BEST_EPISODES[model_seed],
        "best_checkpoint": str(best_path.resolve()),
        "latest_checkpoint": str(latest_path.resolve()),
        "history": str(history_path.resolve()),
    }
    for field, expected_value in expected_summary.items():
        _require_equal(f"seed {model_seed} summary {field}", summary.get(field), expected_value)
    _require_equal("summary lambda", float(summary.get("lambda")), 10.0)
    _require_equal("summary mu", float(summary.get("mu")), 80.0)
    config = summary.get("config")
    _require_equal(
        f"seed {model_seed} trainer config",
        config,
        {
            "hidden_channels": 32,
            "learning_rate": 0.0001,
            "weight_decay": 0.0,
            "gamma": 0.99,
            "reward_scale": 1.0,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "grad_clip": 5.0,
            "updates_per_episode": 1,
        },
    )

    if len(history) != 10:
        raise EvaluationProtocolError(f"seed {model_seed} history must have 10 validation looks")
    _require_equal("history episodes", [item.get("episode") for item in history], list(range(100, 1001, 100)))
    _require_equal(
        "history schedule endpoints",
        [item.get("training_instance_seed") for item in history],
        [70_000_000 + model_seed * 1_000_000 + episode for episode in range(100, 1001, 100)],
    )
    for item in history:
        if not isinstance(item.get("validation"), dict):
            raise EvaluationProtocolError(f"seed {model_seed} history validation is missing")
    eligible_history = [
        item
        for item in history
        if isinstance(item["validation"].get("selection"), Mapping)
        and item["validation"]["selection"].get("eligible") is True
    ]
    if not eligible_history:
        raise EvaluationProtocolError(f"seed {model_seed} has no eligible validation look")
    selected = max(
        eligible_history,
        key=lambda item: (
            tuple(_finite(value, label="selection key") for value in item["validation"]["selection_key"]),
            -int(item["episode"]),
        ),
    )
    _require_equal("selected history episode", selected["episode"], summary["best_episode"])
    _require_equal("selected history key", selected["validation"]["selection_key"], summary.get("best_selection_key"))
    _require_equal("selected history validation", selected["validation"], summary.get("best_validation"))
    _require_equal("final history validation", history[-1]["validation"], summary.get("final_validation"))
    selection = summary["best_validation"].get("selection", {})
    _require_equal("best validation eligible", selection.get("eligible"), True)
    _require_equal("best stochastic strict rate", selection.get("primary_strict_method_success_rate"), 1.0)

    _require_equal("validation index", validation_manifest, summary.get("validation_instances"))
    if len(validation_manifest) != 5:
        raise EvaluationProtocolError(f"seed {model_seed} validation manifest must have five rows")
    template_env = make_shipyard_env(
        arrival_rate=10.0, proc_mean=80.0, grid_rows=5, grid_cols=5,
        exit_width=None, number_blocks=EXPECTED_DELIVERIES,
    )
    _require_equal(
        "training geometry",
        summary.get("geometry"),
        geometry_metadata(template_env, requested_exit_width=None),
    )
    validation_records: list[dict[str, Any]] = []
    for expected_seed, record in zip(training_protocol.VALIDATION_INSTANCE_SEEDS, validation_manifest):
        path = seed_dir / "validation-instances" / f"seed-{expected_seed}.json"
        _require_equal("validation record seed", record.get("seed"), expected_seed)
        _require_equal("validation record path", Path(record.get("path", "")).resolve(), path.resolve())
        instance_bytes, instance_raw_sha = _read_regular_bytes(path)
        instance = EpisodeInstance.from_json(instance_bytes.decode("utf-8"))
        instance.validate_for(template_env)
        _require_equal("validation instance seed", instance.seed, expected_seed)
        _require_equal("validation instance id", instance.instance_id, record.get("instance_id"))
        _require_equal("validation schedule id", instance.schedule_id, record.get("schedule_id"))
        validation_records.append(
            {
                "seed": expected_seed,
                "instance_id": instance.instance_id,
                "schedule_id": instance.schedule_id,
                "raw_sha256": instance_raw_sha,
            }
        )

    best_payload = _load_checkpoint_bound(
        best_path,
        EXPECTED_TRAINING_RAW_SHA256[
            f"training/seed-{model_seed}/best.pth"
        ],
    )
    latest_payload = _load_checkpoint_bound(
        latest_path,
        EXPECTED_TRAINING_RAW_SHA256[
            f"training/seed-{model_seed}/latest.pth"
        ],
    )
    _validate_common_checkpoint(
        best_payload, model_seed=model_seed, kind="best", episode=summary["best_episode"]
    )
    _validate_common_checkpoint(
        latest_payload, model_seed=model_seed, kind="latest", episode=1000
    )
    _require_equal("best checkpoint validation", best_payload.get("validation_evaluation"), summary.get("best_validation"))
    _require_equal("latest checkpoint validation", latest_payload.get("validation_evaluation"), summary.get("final_validation"))
    validate_locked_kim2020_checkpoint(
        best_payload,
        env=template_env,
        requested_exit_width=None,
        expected_deployment_digest=EXPECTED_BEST_DEPLOYMENT_DIGESTS[model_seed],
    )
    deployment_digest = kim2020_deployment_digest(best_payload)
    _require_equal("best deployment digest", deployment_digest, EXPECTED_BEST_DEPLOYMENT_DIGESTS[model_seed])
    diagnostics = summary.get("source_diagnostics")
    if not isinstance(diagnostics, dict):
        raise EvaluationProtocolError(f"seed {model_seed} source diagnostics are missing")
    for field in (
        "assignment_count", "gradient_steps", "completed_outcome_count",
        "censored_outcome_count", "training_episode_count", "skipped_episode_count",
    ):
        _require_equal(f"latest/source diagnostic {field}", latest_payload.get(field), diagnostics.get(field))
    _require_equal(
        "trained plus skipped episodes",
        _count(diagnostics.get("training_episode_count"), label="training episode count")
        + _count(diagnostics.get("skipped_episode_count"), label="skipped episode count"),
        1000,
    )

    artifact_hashes = {
        str(path.relative_to(seed_dir)): _sha256(path)
        for path in sorted(seed_dir.rglob("*"))
        if path.is_file()
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "authenticated_complete_eligible",
        "model_seed": model_seed,
        "training_was_resumed": False,
        "completed_training_episodes": 1000,
        "training_schedule_range": list(training_protocol.training_schedule_range(model_seed)),
        "validation_look_episodes": list(range(100, 1001, 100)),
        "selected_episode": summary["best_episode"],
        "selected_checkpoint_path": str(best_path.resolve()),
        "selected_checkpoint_raw_sha256": _sha256(best_path),
        "selected_checkpoint_deployment_sha256": deployment_digest,
        "selected_checkpoint_inference_origin_counters": {
            field: _count(best_payload.get(field), label=f"best checkpoint {field}")
            for field in (
                "assignment_count",
                "gradient_steps",
                "completed_outcome_count",
                "censored_outcome_count",
            )
        },
        "latest_checkpoint_raw_sha256": _sha256(latest_path),
        "artifact_raw_sha256": artifact_hashes,
        "artifact_tree_sha256": _digest_json(artifact_hashes),
        "validation_instances": validation_records,
        "best_validation_eligible": True,
        "best_validation_primary_strict_success_rate": 1.0,
        "exact_paper_reproduction": False,
        "adaptation_status": "adaptation_not_exact_reproduction",
    }
    return _add_self_hash(record, "completion_sha256")


def _authenticate_training(
    project_root: Path, training_root: Path
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[str, str]]:
    training_protocol._validate_output_tree(training_root)
    manifest_path = training_root / "supplement-run-manifest.json"
    manifest = _load_json(manifest_path, expected_type=dict)
    training_protocol.validate_manifest(manifest, project_root, training_root)
    tree = _tree_file_hashes(training_root)
    _require_equal("completed training tree", tree, EXPECTED_TRAINING_RAW_SHA256)
    _require_equal("completed training tree digest", _digest_json(tree), EXPECTED_TRAINING_TREE_SHA256)
    completions = {
        seed: _authenticate_training_seed(project_root, training_root, seed)
        for seed in MODEL_SEEDS
    }
    source_hashes = _authenticate_sources(project_root, manifest)
    return manifest, completions, source_hashes


def _completion_filename(model_seed: int) -> str:
    return f"seed-{model_seed}.json"


def _lock_filename(model_seed: int) -> str:
    return f"seed-{model_seed}.lock"


@contextmanager
def _exclusive_seed_lock(output_root: Path, model_seed: int):
    path = output_root / "run-locks" / _lock_filename(model_seed)
    _reject_symlink_components(path)
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise EvaluationProtocolError("seed run lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvaluationProtocolError(
                f"model seed {model_seed} is already being evaluated"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _ledger_path(output_root: Path, model_seed: int) -> Path:
    return output_root / "run-ledger" / _completion_filename(model_seed)


def _seed_completion_path(output_root: Path, model_seed: int) -> Path:
    return output_root / "seed-completion" / _completion_filename(model_seed)


def _validate_output_tree(output_root: Path, *, require_manifest: bool = True) -> None:
    if output_root.is_symlink():
        raise EvaluationProtocolError("evaluation output root may not be a symlink")
    if not output_root.exists():
        if require_manifest:
            raise EvaluationProtocolError("evaluation output root is missing")
        return
    if not output_root.is_dir():
        raise EvaluationProtocolError("evaluation output root must be a directory")
    entries = {entry.name: entry for entry in output_root.iterdir()}
    required = {
        "evaluation-run-manifest.json",
        "training-completion",
        "run-ledger",
        "seed-completion",
        "run-locks",
    }
    optional = {"extended-report.json", "extended-audit.json"}
    unexpected = sorted(set(entries) - required - optional)
    if unexpected:
        raise EvaluationProtocolError(f"unexpected evaluation-root entries: {unexpected}")
    if require_manifest and not required.issubset(entries):
        raise EvaluationProtocolError(
            f"evaluation output is incomplete: {sorted(required - set(entries))}"
        )
    if "extended-audit.json" in entries and "extended-report.json" not in entries:
        raise EvaluationProtocolError("extended audit may not exist without its report")
    for filename in ("evaluation-run-manifest.json", "extended-report.json", "extended-audit.json"):
        path = entries.get(filename)
        if path is not None and (path.is_symlink() or not path.is_file()):
            raise EvaluationProtocolError(f"{filename} must be a regular non-symlink file")
    allowed_seed_files = {_completion_filename(seed) for seed in MODEL_SEEDS}
    for directory_name in (
        "training-completion", "run-ledger", "seed-completion", "run-locks"
    ):
        directory = entries.get(directory_name)
        if directory is None:
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise EvaluationProtocolError(f"{directory_name} must be a regular directory")
        children = {child.name: child for child in directory.iterdir()}
        directory_allowed = (
            {_lock_filename(seed) for seed in MODEL_SEEDS}
            if directory_name == "run-locks"
            else allowed_seed_files
        )
        if not set(children).issubset(directory_allowed):
            raise EvaluationProtocolError(
                f"unexpected {directory_name} files: {sorted(set(children) - directory_allowed)}"
            )
        if any(child.is_symlink() or not child.is_file() for child in children.values()):
            raise EvaluationProtocolError(f"invalid artifact in {directory_name}")
        if directory_name in ("training-completion", "run-locks") and set(children) != directory_allowed:
            raise EvaluationProtocolError(
                f"{directory_name} must contain exactly three fixed files"
            )
        if directory_name == "run-locks":
            for child in children.values():
                if child.stat().st_size != 0:
                    raise EvaluationProtocolError("run lock files must remain empty")
    ledgers = {
        child.name for child in (output_root / "run-ledger").iterdir()
    } if (output_root / "run-ledger").is_dir() else set()
    completions = {
        child.name for child in (output_root / "seed-completion").iterdir()
    } if (output_root / "seed-completion").is_dir() else set()
    if not completions.issubset(ledgers):
        raise EvaluationProtocolError("seed completion exists without its run ledger")


def _prepared_state(
    project_root: Path, training_root: Path, output_root: Path
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[int, EpisodeInstance],
]:
    training_manifest, completions, source_hashes = _authenticate_training(
        project_root, training_root
    )
    parent_artifacts = _authenticate_parent_artifacts(project_root)
    panel, instances = _authenticate_panel(project_root)
    completion_files: dict[str, dict[str, Any]] = {}
    for seed, completion in completions.items():
        payload = _json_payload(completion)
        completion_files[str(seed)] = {
            "path": f"training-completion/{_completion_filename(seed)}",
            "raw_sha256": hashlib.sha256(payload).hexdigest(),
            "completion_sha256": completion["completion_sha256"],
            "selected_episode": completion["selected_episode"],
            "selected_checkpoint_raw_sha256": completion[
                "selected_checkpoint_raw_sha256"
            ],
            "selected_checkpoint_deployment_sha256": completion[
                "selected_checkpoint_deployment_sha256"
            ],
        }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "prepared_no_85k_kim_rollouts_executed",
        "development_only": True,
        "performance_claim_authorized": False,
        "confirmatory_claim_authorized": False,
        "original_v2_3_stability_verdict_mutated": False,
        "final_86xxx_panel_opened": False,
        "sealed_622m_policy_namespace_opened": False,
        "post_training_pre_evaluation_implementation": True,
        "kim_85k_outcomes_observed_before_freeze": False,
        "training_outcomes_observed_before_evaluator_implementation": True,
        "predeclared_design_realized_without_outcome_change": {
            "source": "vcg_v2_3_kim2020_matched_supplement_85xxx_v1",
            "grid_unchanged": True,
            "evaluation_config_unchanged": True,
            "aggregation_order_unchanged": True,
            "extended_gate_unchanged": True,
        },
        "project_root": str(project_root),
        "immutable_training_root": str(training_root),
        "output_root": str(output_root),
        "training_manifest": {
            "path": str((training_root / "supplement-run-manifest.json").resolve()),
            "raw_sha256": EXPECTED_TRAINING_RAW_SHA256["supplement-run-manifest.json"],
            "manifest_sha256": training_manifest["manifest_sha256"],
        },
        "training_tree_file_count": len(EXPECTED_TRAINING_RAW_SHA256),
        "training_tree_raw_sha256": EXPECTED_TRAINING_TREE_SHA256,
        "training_completion_files": completion_files,
        "parent_artifacts": parent_artifacts,
        "episode_instance_panel": panel,
        "model_seeds": list(MODEL_SEEDS),
        "panel_seeds": list(PANEL_SEEDS),
        "rollouts": list(ROLLOUTS),
        "evaluation_grid": expected_grid(),
        "evaluation_config": {
            "device": "cpu",
            "deterministic_algorithms": True,
            "torch_num_threads": 1,
            "deployment": "stochastic_primary",
            "scheduler_variant": DURATION_AWARE_VARIANT,
            "source_neutral_scheduler": True,
            "assignment_commitment": DECISION_EPOCH_RESERVED,
            "assignment_source": TRACK_A_KIM2020_A3C_SPATIAL,
            "max_steps": 2000,
            "max_defer_steps": 10,
            "lookahead_margin_steps": 2.0,
            "target_window": 20.0,
            "required_deliveries": EXPECTED_DELIVERIES,
            "policy_seed_formula": (
                "632000000 + 1000*model_seed + 10*panel_index + rollout"
            ),
            "map_rollouts": 0,
            "baseline_rollouts": 0,
            "instance_resampling": False,
            "complete_case_filtering_allowed": False,
        },
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "reported_return": "undiscounted_dense_fixed_trajectory_rescore",
        "environment_return_role": "legacy_environment_return_diagnostic_only",
        "primary_contention_metric": (
            "total_physical_storage_relocations_per_100_required_deliveries"
        ),
        "aggregation": {
            "order": "equal_5_policy_rolls_then_equal_12_instances_then_equal_3_model_seeds",
            "model_instance_point_count": 36,
            "model_summary_count": 3,
            "instance_cluster_count_for_descriptive_statistics": 12,
            "raw_rows_are_not_independent_statistical_units": True,
        },
        "numeric_table_method_ids": [
            "vcg_constrained_v2_3_equal_stability_seeds_11_13",
            *ELIGIBLE_BASELINE_IDS,
            KIM_STOCHASTIC,
        ],
        "seed10_role": "historical_development_diagnostic_not_gate_comparator",
        "extended_gate": {
            "coordinates": [
                "mean_absolute_error",
                "physical_rehandles_per_100_required_deliveries",
            ],
            "dominance": "weak_both_strict_at_least_one",
            "equal_seed_v2_3_must_be_nondominated": True,
            "minimum_nondominated_individual_v2_3_seeds": 2,
            "scalarization_used": False,
            "return_and_steps_reported_not_gated": True,
            "kim_required_to_win": False,
        },
        "source_sha256": source_hashes,
        "source_count": len(source_hashes),
        "runtime": _fixed_runtime(),
    }
    return _add_self_hash(manifest, "manifest_sha256"), completions, instances


def _prepare(project_root: Path, training_root: Path, output_root: Path) -> Path:
    _validate_output_tree(output_root, require_manifest=False)
    if output_root.exists():
        raise EvaluationProtocolError(f"evaluation output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    manifest, completions, _ = _prepared_state(project_root, training_root, output_root)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.prepare-", dir=str(output_root.parent)
        )
    )
    try:
        for directory in (
            "training-completion", "run-ledger", "seed-completion", "run-locks"
        ):
            (temporary / directory).mkdir()
        for seed in MODEL_SEEDS:
            lock_path = temporary / "run-locks" / _lock_filename(seed)
            descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            os.close(descriptor)
        for seed, completion in completions.items():
            _write_new_json(
                temporary / "training-completion" / _completion_filename(seed),
                completion,
            )
        _write_new_json(temporary / "evaluation-run-manifest.json", manifest)
        if output_root.exists() or output_root.is_symlink():
            raise EvaluationProtocolError("evaluation output appeared during prepare")
        os.rename(temporary, output_root)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
    _authenticate_prepared(project_root, training_root, output_root)
    return output_root / "evaluation-run-manifest.json"


def _authenticate_prepared(
    project_root: Path, training_root: Path, output_root: Path
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[int, EpisodeInstance]]:
    _validate_output_tree(output_root)
    expected_manifest, expected_completions, instances = _prepared_state(
        project_root, training_root, output_root
    )
    manifest_path = output_root / "evaluation-run-manifest.json"
    manifest = _load_json(manifest_path, expected_type=dict)
    _verify_self_hash(manifest, "manifest_sha256", label="evaluation run manifest")
    _require_equal("evaluation run manifest", manifest, expected_manifest)
    for seed, expected in expected_completions.items():
        path = output_root / "training-completion" / _completion_filename(seed)
        observed = _load_json(path, expected_type=dict)
        _verify_self_hash(observed, "completion_sha256", label=f"seed {seed} training completion")
        _require_equal(f"seed {seed} training completion", observed, expected)
        reference = manifest["training_completion_files"][str(seed)]
        _require_equal(f"seed {seed} completion raw SHA", _sha256(path), reference["raw_sha256"])
    return manifest, expected_completions, instances


def _dense_rescore(
    legacy_return: Any, delivery_deviations: Sequence[float]
) -> tuple[float, float]:
    legacy = _finite(legacy_return, label="legacy environment return")
    deviations = tuple(
        _finite(value, label="delivery deviation") for value in delivery_deviations
    )
    legacy_delivery = sum(
        delivery_reward(value, FROZEN_OBJECTIVE_SPEC, LEGACY_CLIPPED)
        for value in deviations
    )
    dense_delivery = sum(
        delivery_reward(value, FROZEN_OBJECTIVE_SPEC, DENSE_PIECEWISE)
        for value in deviations
    )
    return legacy, float(legacy - legacy_delivery + dense_delivery)


def _timing_metrics(deviations: Sequence[float]) -> dict[str, float | None]:
    values = tuple(_finite(value, label="delivery deviation") for value in deviations)
    if not values:
        return {
            "mean_signed_deviation": None,
            "mean_absolute_error": None,
            "mean_tardiness": None,
            "mean_earliness": None,
            "within_target_window_rate": None,
        }
    return {
        "mean_signed_deviation": float(statistics.fmean(values)),
        "mean_absolute_error": float(statistics.fmean(abs(value) for value in values)),
        "mean_tardiness": float(statistics.fmean(max(value, 0.0) for value in values)),
        "mean_earliness": float(statistics.fmean(max(-value, 0.0) for value in values)),
        "within_target_window_rate": float(
            statistics.fmean(abs(value) <= FROZEN_OBJECTIVE_SPEC.window for value in values)
        ),
    }


def _require_close(label: str, observed: Any, expected: float, *, tolerance: float = 1e-9) -> None:
    value = _finite(observed, label=label)
    if abs(value - float(expected)) > tolerance:
        raise EvaluationProtocolError(
            f"{label} mismatch: observed={value!r}, expected={expected!r}"
        )


def _audit_count(mapping: Mapping[str, Any], field: str) -> int | None:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    result = int(value)
    return result if result >= 0 else None


def _normalize_run(
    raw: Mapping[str, Any],
    *,
    grid_row: Mapping[str, int],
    instance_record: Mapping[str, Any],
    training_completion: Mapping[str, Any],
    evaluation_manifest_sha256: str,
) -> dict[str, Any]:
    model_seed = int(grid_row["model_seed"])
    panel_index = int(grid_row["panel_index"])
    instance_seed = int(grid_row["instance_seed"])
    rollout = int(grid_row["rollout"])
    rng_seed = int(grid_row["policy_seed"])
    expected_geometry = FROZEN_GEOMETRY
    fatal_expected = {
        "method": TRACK_B_METHOD,
        "method_variant": TRACK_B_METHOD_VARIANT,
        "evaluation_scope": TRACK_B_EVALUATION_SCOPE,
        "controller_architecture": TRACK_B_CONTROLLER_ARCHITECTURE,
        "controller_action_interface": TRACK_B_CONTROLLER_ACTION_INTERFACE,
        "policy_realization": TRACK_B_POLICY_REALIZATION,
        "assignment_source_family": "learned_spatial_assignment_policy",
        "assignment_source_version": "kim2020_grid_actor_critic_adapted_v1",
        "assignment_commitment": DECISION_EPOCH_RESERVED,
        "assignment_source": TRACK_A_KIM2020_A3C_SPATIAL,
        "track": "B",
        "information_regime": "online_arrived_only",
        "eval_seed": instance_seed,
        "instance_id": instance_record["episode_instance_id"],
        "schedule_id": instance_record["schedule_id"],
        "geometry": expected_geometry,
        "assignment_policy_realization": DEPLOYMENT_STOCHASTIC,
        "assignment_policy_seed": rng_seed,
        "selector_deployment_digest": training_completion[
            "selected_checkpoint_deployment_sha256"
        ],
    }
    for field, expected in fatal_expected.items():
        _require_equal(f"row {model_seed}/{instance_seed}/{rollout} {field}", raw.get(field), expected)
    _require_close("row lambda", raw.get("lambda"), 10.0)
    _require_close("row mu", raw.get("mu"), 80.0)
    _require_close("row target window", raw.get("target_window"), 20.0)
    deviations_raw = raw.get("delivery_deviations")
    if not isinstance(deviations_raw, (list, tuple)):
        raise EvaluationProtocolError("row delivery deviations must be a sequence")
    deviations = tuple(
        _finite(value, label="row delivery deviation") for value in deviations_raw
    )
    if len(deviations) > EXPECTED_DELIVERIES:
        raise EvaluationProtocolError("row has more delivery deviations than required blocks")
    delivery_count = _count(raw.get("delivery_count"), label="row delivery count")
    _require_equal("row deviation count", len(deviations), delivery_count)
    timing = _timing_metrics(deviations)
    if deviations:
        for field, expected in timing.items():
            assert expected is not None
            _require_close(f"row {field}", raw.get(field), expected)
    legacy_return, dense_return = _dense_rescore(raw.get("return"), deviations)
    steps = _count(raw.get("steps"), label="row steps")
    try:
        contention = validate_contention_metric_record(raw, require_legacy_alias=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationProtocolError("row contention decomposition is invalid") from exc
    physical = int(contention["physical_storage_relocations"])
    physical_rate = 100.0 * physical / EXPECTED_DELIVERIES
    selector = raw.get("selector_audit")
    scheduler = raw.get("scheduler_audit")
    urgency = raw.get("urgency_scheduler_audit")
    if (
        not isinstance(selector, Mapping)
        or not isinstance(scheduler, Mapping)
        or not isinstance(urgency, Mapping)
    ):
        raise EvaluationProtocolError("row selector/scheduler audits are missing")
    urgency_expected = {
        "method": TRACK_B_METHOD,
        "scheduler_architecture": TRACK_B_CONTROLLER_ARCHITECTURE,
        "policy_realization": TRACK_B_POLICY_REALIZATION,
        "controller_action_interface": TRACK_B_CONTROLLER_ACTION_INTERFACE,
        "decision_epoch_contract": TRACK_B_DECISION_EPOCH_CONTRACT,
        "assignment_commitment_contract": TRACK_B_ASSIGNMENT_COMMITMENT_CONTRACT,
        "reservation_contract": TRACK_B_RESERVATION_CONTRACT,
        "assignment_source": TRACK_A_KIM2020_A3C_SPATIAL,
        "assignment_source_family": "learned_spatial_assignment_policy",
        "assignment_source_version": "kim2020_grid_actor_critic_adapted_v1",
    }
    for field, expected_value in urgency_expected.items():
        _require_equal(f"row urgency scheduler {field}", urgency.get(field), expected_value)
    _require_close(
        "row urgency scheduler lookahead margin",
        urgency.get("lookahead_margin_steps"),
        2.0,
    )
    source_audit = selector.get("source_audit")
    if not isinstance(source_audit, Mapping):
        raise EvaluationProtocolError("row nested assignment-source audit is missing")
    last_episode = source_audit.get("last_episode")
    if not isinstance(last_episode, Mapping):
        raise EvaluationProtocolError("row assignment-source last-episode audit is missing")
    selector_committed_ids = [
        decision.get("proposal_id")
        for decision in selector.get("decisions", ())
        if isinstance(decision, Mapping)
        and decision.get("valid")
        and decision.get("commitment_contract")
        == "decision_epoch_proposal_bound_once_v2"
    ]

    method_audit = {
        "method": raw.get("method"),
        "method_variant": raw.get("method_variant"),
        "evaluation_scope": raw.get("evaluation_scope"),
        "target_window": raw.get("target_window"),
        "assignment_source_family": raw.get("assignment_source_family"),
        "assignment_source_version": raw.get("assignment_source_version"),
        "controller_architecture": raw.get("controller_architecture"),
        "controller_action_interface": raw.get("controller_action_interface"),
        "policy_realization": raw.get("policy_realization"),
        "urgency_method": urgency.get("method"),
        "urgency_scheduler_architecture": urgency.get("scheduler_architecture"),
        "urgency_policy_realization": urgency.get("policy_realization"),
        "urgency_controller_action_interface": urgency.get(
            "controller_action_interface"
        ),
        "urgency_decision_epoch_contract": urgency.get(
            "decision_epoch_contract"
        ),
        "urgency_assignment_commitment_contract": urgency.get(
            "assignment_commitment_contract"
        ),
        "urgency_reservation_contract": urgency.get("reservation_contract"),
        "urgency_assignment_source": urgency.get("assignment_source"),
        "urgency_assignment_source_family": urgency.get(
            "assignment_source_family"
        ),
        "urgency_assignment_source_version": urgency.get(
            "assignment_source_version"
        ),
        "urgency_lookahead_margin_steps": urgency.get("lookahead_margin_steps"),
        "reservation_integrity": raw.get("reservation_integrity"),
        "selector_frozen": selector.get("frozen"),
        "selector_assignment_source": selector.get("assignment_source"),
        "selector_assignment_source_family": selector.get("assignment_source_family"),
        "selector_assignment_source_learned": selector.get("assignment_source_learned"),
        "selector_assignment_source_version": selector.get("assignment_source_version"),
        "selector_decision_count": _audit_count(selector, "decision_count"),
        "selector_valid_assignment_count": _audit_count(
            selector, "valid_assignment_count"
        ),
        "selector_reserved_commit_count": _audit_count(
            selector, "reserved_commit_count"
        ),
        "selector_post_pickup_recompute_count": _audit_count(
            selector, "post_pickup_recompute_count"
        ),
        "selector_committed_proposal_ids": selector_committed_ids,
        "invalid_assignment_count": _audit_count(selector, "invalid_assignment_count"),
        "fallback_count": _audit_count(selector, "fallback_count"),
        "source_audit_schema": sorted(source_audit),
        "source_assignment_source": source_audit.get("assignment_source"),
        "source_assignment_source_family": source_audit.get(
            "assignment_source_family"
        ),
        "source_assignment_source_version": source_audit.get(
            "assignment_source_version"
        ),
        "source_selector_feature_version": source_audit.get(
            "selector_feature_version"
        ),
        "source_reward_contract": source_audit.get("reward_contract"),
        "source_return_contract": source_audit.get("return_contract"),
        "source_action_mapping": source_audit.get("action_mapping"),
        "learning_enabled": source_audit.get("learning_enabled"),
        "deployment_mode": source_audit.get("deployment_mode"),
        "policy_seed": source_audit.get("policy_seed"),
        "source_assignment_count": _audit_count(source_audit, "assignment_count"),
        "source_gradient_steps": _audit_count(source_audit, "gradient_steps"),
        "source_completed_outcome_count": _audit_count(
            source_audit, "completed_outcome_count"
        ),
        "source_censored_outcome_count": _audit_count(
            source_audit, "censored_outcome_count"
        ),
        "source_last_episode": _json_safe(dict(last_episode)),
        "inbound_successes": _audit_count(scheduler, "inbound_successes"),
        "inbound_failures": _audit_count(scheduler, "inbound_failures"),
        "retrieve_successes": _audit_count(scheduler, "retrieve_successes"),
        "retrieve_failures": _audit_count(scheduler, "retrieve_failures"),
        "retrieve_relocations": _audit_count(scheduler, "retrieve_relocations"),
        "reservation_bound_count": _audit_count(scheduler, "reservation_bound_count"),
        "reservation_commit_count": _audit_count(scheduler, "reservation_commit_count"),
        "reservation_execution_match_count": _audit_count(
            scheduler, "reservation_execution_match_count"
        ),
        "reservation_invalidation_count": _audit_count(
            scheduler, "reservation_invalidation_count"
        ),
        "bound_proposal_ids": _json_safe(scheduler.get("bound_proposal_ids")),
        "committed_proposal_ids": _json_safe(
            scheduler.get("committed_proposal_ids")
        ),
        "storage_flow_fully_observed": raw.get("storage_flow_fully_observed"),
        "storage_flow_manifest_count": raw.get("storage_flow_manifest_count"),
        "storage_flow_completed_count": raw.get("storage_flow_completed_count"),
        "storage_flow_unfinished_count": raw.get("storage_flow_unfinished_count"),
    }
    success = _finite(raw.get("success"), label="row success")
    raw_strict = _finite(
        raw.get("strict_method_success"), label="row strict method success"
    )
    truncated = _finite(raw.get("truncated"), label="row truncated")
    illegal_drops = _count(raw.get("illegal_drops"), label="row illegal drops")
    safety_checks = {
        "success": success == 1.0,
        "raw_strict_method_success": raw_strict == 1.0,
        "complete_required_deliveries": delivery_count == EXPECTED_DELIVERIES,
        "not_truncated": truncated == 0.0,
        "no_method_failure": raw.get("method_failure_reason") is None,
        "reservation_integrity": raw.get("reservation_integrity") is True,
        "no_illegal_drops": illegal_drops == 0,
        "no_invalid_assignments": method_audit["invalid_assignment_count"] == 0,
        "no_fallbacks": method_audit["fallback_count"] == 0,
        "no_inbound_failures": method_audit["inbound_failures"] == 0,
        "no_retrieve_failures": method_audit["retrieve_failures"] == 0,
        "no_reservation_invalidations": (
            method_audit["reservation_invalidation_count"] == 0
        ),
        "selector_frozen": method_audit["selector_frozen"] is True,
        "selector_counts_complete": (
            method_audit["selector_decision_count"] == EXPECTED_DELIVERIES
            and method_audit["selector_valid_assignment_count"] == EXPECTED_DELIVERIES
            and method_audit["selector_reserved_commit_count"] == EXPECTED_DELIVERIES
            and method_audit["selector_post_pickup_recompute_count"] == 0
        ),
        "scheduler_counts_complete": (
            method_audit["inbound_successes"] == EXPECTED_DELIVERIES
            and method_audit["retrieve_successes"] == EXPECTED_DELIVERIES
            and method_audit["reservation_bound_count"] == EXPECTED_DELIVERIES
            and method_audit["reservation_commit_count"] == EXPECTED_DELIVERIES
            and method_audit["reservation_execution_match_count"]
            == EXPECTED_DELIVERIES
        ),
        "proposal_id_integrity": (
            isinstance(method_audit["bound_proposal_ids"], list)
            and len(method_audit["bound_proposal_ids"]) == EXPECTED_DELIVERIES
            and len(set(method_audit["bound_proposal_ids"])) == EXPECTED_DELIVERIES
            and method_audit["bound_proposal_ids"]
            == method_audit["committed_proposal_ids"]
            == method_audit["selector_committed_proposal_ids"]
        ),
        "selector_source_identity": (
            method_audit["selector_assignment_source"]
            == TRACK_A_KIM2020_A3C_SPATIAL
            and method_audit["selector_assignment_source_family"]
            == "learned_spatial_assignment_policy"
            and method_audit["selector_assignment_source_learned"] is True
            and method_audit["selector_assignment_source_version"]
            == "kim2020_grid_actor_critic_adapted_v1"
            and method_audit["source_assignment_source"]
            == TRACK_A_KIM2020_A3C_SPATIAL
            and method_audit["source_assignment_source_family"]
            == "learned_spatial_assignment_policy"
            and method_audit["source_assignment_source_version"]
            == "kim2020_grid_actor_critic_adapted_v1"
        ),
        "evaluation_learning_disabled": method_audit["learning_enabled"] is False,
        "stochastic_deployment": method_audit["deployment_mode"] == DEPLOYMENT_STOCHASTIC,
        "exact_policy_seed": method_audit["policy_seed"] == rng_seed,
        "checkpoint_counters_unchanged": all(
            method_audit[f"source_{field}"]
            == training_completion["selected_checkpoint_inference_origin_counters"][field]
            for field in (
                "assignment_count", "gradient_steps", "completed_outcome_count",
                "censored_outcome_count",
            )
        ),
        "source_last_episode_is_evaluation_only": (
            method_audit["source_last_episode"].get("instance_id")
            == instance_record["episode_instance_id"]
            and method_audit["source_last_episode"].get("success") is True
            and method_audit["source_last_episode"].get("truncated") is False
            and method_audit["source_last_episode"].get("outcome_tracking") is False
            and method_audit["source_last_episode"].get("integrity") is None
            and method_audit["source_last_episode"].get("trained") is False
            and method_audit["source_last_episode"].get("selection_count") == 0
            and method_audit["source_last_episode"].get("completed_count") == 0
            and method_audit["source_last_episode"].get("censored_count") == 0
            and method_audit["source_last_episode"].get("records") == []
        ),
        "storage_manifest_fully_observed": (
            method_audit["storage_flow_fully_observed"] is True
            and method_audit["storage_flow_manifest_count"] == EXPECTED_DELIVERIES
            and method_audit["storage_flow_completed_count"] == EXPECTED_DELIVERIES
            and method_audit["storage_flow_unfinished_count"] == 0
        ),
        "all_physical_relocations_target_bound": (
            contention["target_bound_obstruction_clearances"] == physical
            and contention["standalone_reconfigurations"] == 0
        ),
        "retrieve_relocation_audit_matches": (
            method_audit["retrieve_relocations"] == physical
        ),
    }
    safety_issues = sorted(name for name, passed in safety_checks.items() if not passed)
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "method_id": KIM_STOCHASTIC,
        "method_category": "stochastic_online_primary",
        "replication_design": "five_policy_rolls_within_fixed_episode_instance",
        "model_seed": model_seed,
        "panel_index": panel_index,
        "instance_seed": instance_seed,
        "rollout_index": rollout,
        "policy_rng_seed": rng_seed,
        "episode_instance_id": instance_record["episode_instance_id"],
        "schedule_id": instance_record["schedule_id"],
        "episode_instance_sha256": instance_record["episode_instance_sha256"],
        "episode_instance_file_raw_sha256": instance_record["saved_file_raw_sha256"],
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "training_completion_sha256": training_completion["completion_sha256"],
        "source_checkpoint_raw_sha256": training_completion[
            "selected_checkpoint_raw_sha256"
        ],
        "source_checkpoint_deployment_sha256": training_completion[
            "selected_checkpoint_deployment_sha256"
        ],
        "selected_training_episode": training_completion["selected_episode"],
        "assignment_source": TRACK_A_KIM2020_A3C_SPATIAL,
        "assignment_commitment": DECISION_EPOCH_RESERVED,
        "scheduler_variant": DURATION_AWARE_VARIANT,
        "source_neutral_scheduler": True,
        "deployment_mode": DEPLOYMENT_STOCHASTIC,
        "evaluation_learning": False,
        "fresh_evaluation_clone": True,
        "arrival_rate": 10.0,
        "mean_stay": 80.0,
        "target_window": 20.0,
        "geometry": _json_safe(FROZEN_GEOMETRY),
        "legacy_environment_return": legacy_return,
        "dense_objective_return": dense_return,
        "delivery_deviations": list(deviations),
        **timing,
        "steps": steps,
        "required_deliveries": EXPECTED_DELIVERIES,
        "delivery_count": delivery_count,
        "completion_rate": delivery_count / EXPECTED_DELIVERIES,
        **contention,
        "physical_rehandles_per_100_required_deliveries": physical_rate,
        "success": success,
        "raw_strict_method_success": raw_strict,
        "method_failure_reason": raw.get("method_failure_reason"),
        "truncated": truncated,
        "illegal_drops": illegal_drops,
        "method_audit": method_audit,
        "safety_checks": safety_checks,
        "safety_issues": safety_issues,
        "strict_safe_complete": not safety_issues,
    }
    # Recompute from the persisted primitive evidence, using the same routine
    # that authenticates a reloaded ledger.  No persisted safety claim is its
    # own evidence.
    row["safety_checks"] = _recompute_row_safety(
        row,
        training_completion=training_completion,
        instance_record=instance_record,
    )
    row["safety_issues"] = sorted(
        name for name, passed in row["safety_checks"].items() if passed is not True
    )
    row["strict_safe_complete"] = not row["safety_issues"]
    return _add_self_hash(row, "row_sha256")


ROW_FIELDS = frozenset(
    {
        "schema_version", "protocol", "method_id", "method_category",
        "replication_design", "model_seed", "panel_index", "instance_seed",
        "rollout_index", "policy_rng_seed", "episode_instance_id", "schedule_id",
        "episode_instance_sha256", "episode_instance_file_raw_sha256",
        "evaluation_manifest_sha256", "training_completion_sha256",
        "source_checkpoint_raw_sha256", "source_checkpoint_deployment_sha256",
        "selected_training_episode", "assignment_source", "assignment_commitment",
        "scheduler_variant", "source_neutral_scheduler", "deployment_mode",
        "evaluation_learning", "fresh_evaluation_clone", "legacy_environment_return",
        "arrival_rate", "mean_stay", "target_window", "geometry",
        "dense_objective_return", "delivery_deviations", "mean_signed_deviation",
        "mean_absolute_error", "mean_tardiness", "mean_earliness",
        "within_target_window_rate", "steps", "required_deliveries",
        "delivery_count", "completion_rate", "contention_metric_schema_version",
        "physical_storage_relocations", "target_bound_obstruction_clearances",
        "standalone_reconfigurations", "standalone_with_direct_delivery_available",
        "standalone_without_direct_delivery_available",
        "directly_deliverable_self_reconfigurations",
        "physical_rehandles_per_100_required_deliveries", "success",
        "raw_strict_method_success", "method_failure_reason", "truncated",
        "illegal_drops", "method_audit", "safety_checks", "safety_issues",
        "strict_safe_complete", "row_sha256",
    }
)

METHOD_AUDIT_FIELDS = frozenset(
    {
        "method", "method_variant", "evaluation_scope", "target_window",
        "assignment_source_family", "assignment_source_version",
        "controller_architecture", "controller_action_interface", "policy_realization",
        "urgency_method", "urgency_scheduler_architecture",
        "urgency_policy_realization", "urgency_controller_action_interface",
        "urgency_decision_epoch_contract",
        "urgency_assignment_commitment_contract", "urgency_reservation_contract",
        "urgency_assignment_source", "urgency_assignment_source_family",
        "urgency_assignment_source_version", "urgency_lookahead_margin_steps",
        "reservation_integrity", "selector_frozen", "selector_assignment_source",
        "selector_assignment_source_family", "selector_assignment_source_learned",
        "selector_assignment_source_version", "selector_decision_count",
        "selector_valid_assignment_count", "selector_reserved_commit_count",
        "selector_post_pickup_recompute_count", "selector_committed_proposal_ids",
        "invalid_assignment_count", "fallback_count", "source_audit_schema",
        "source_assignment_source", "source_assignment_source_family",
        "source_assignment_source_version", "source_selector_feature_version",
        "source_reward_contract", "source_return_contract", "source_action_mapping",
        "learning_enabled", "deployment_mode", "policy_seed", "source_assignment_count",
        "source_gradient_steps", "source_completed_outcome_count",
        "source_censored_outcome_count", "source_last_episode", "inbound_successes",
        "inbound_failures", "retrieve_successes", "retrieve_failures",
        "retrieve_relocations", "reservation_bound_count", "reservation_commit_count",
        "reservation_execution_match_count", "reservation_invalidation_count",
        "bound_proposal_ids", "committed_proposal_ids",
        "storage_flow_fully_observed", "storage_flow_manifest_count",
        "storage_flow_completed_count", "storage_flow_unfinished_count",
    }
)

SOURCE_AUDIT_SCHEMA = sorted(
    {
        "action_mapping", "assignment_count", "assignment_source",
        "assignment_source_family", "assignment_source_version",
        "censored_outcome_count", "completed_outcome_count", "deployment_mode",
        "gradient_steps", "last_episode", "learning_enabled", "policy_seed",
        "return_contract", "reward_contract", "selector_feature_version",
    }
)

LAST_EPISODE_FIELDS = frozenset(
    {
        "instance_id", "success", "truncated", "outcome_tracking", "integrity",
        "trained", "selection_count", "completed_count", "censored_count", "records",
    }
)


def _recompute_row_safety(
    row: Mapping[str, Any],
    *,
    training_completion: Mapping[str, Any],
    instance_record: Mapping[str, Any],
) -> dict[str, bool]:
    method = row.get("method_audit")
    if not isinstance(method, Mapping):
        raise EvaluationProtocolError("normalized method audit must be a mapping")
    _require_equal("normalized method-audit schema", set(method), METHOD_AUDIT_FIELDS)
    for field in (
        "selector_decision_count", "selector_valid_assignment_count",
        "selector_reserved_commit_count", "selector_post_pickup_recompute_count",
        "invalid_assignment_count", "fallback_count", "source_assignment_count",
        "source_gradient_steps", "source_completed_outcome_count",
        "source_censored_outcome_count", "inbound_successes", "inbound_failures",
        "retrieve_successes", "retrieve_failures", "retrieve_relocations",
        "reservation_bound_count", "reservation_commit_count",
        "reservation_execution_match_count", "reservation_invalidation_count",
        "storage_flow_manifest_count", "storage_flow_completed_count",
        "storage_flow_unfinished_count",
    ):
        if method.get(field) is not None:
            _count(method[field], label=f"method audit {field}")
    for field in (
        "bound_proposal_ids", "committed_proposal_ids",
        "selector_committed_proposal_ids",
    ):
        if not isinstance(method.get(field), list) or any(
            not isinstance(value, str) or not value for value in method[field]
        ):
            raise EvaluationProtocolError(f"method audit {field} must be a string list")
    last_episode = method.get("source_last_episode")
    if not isinstance(last_episode, Mapping):
        raise EvaluationProtocolError("source last-episode evidence must be a mapping")
    _require_equal("source last-episode schema", set(last_episode), LAST_EPISODE_FIELDS)
    rng_seed = row.get("policy_rng_seed")
    physical = row.get("physical_storage_relocations")
    expected_counters = training_completion.get(
        "selected_checkpoint_inference_origin_counters"
    )
    if not isinstance(expected_counters, Mapping):
        raise EvaluationProtocolError("training completion inference counters are missing")
    return {
        "exact_track_b_executor": (
            method.get("method") == TRACK_B_METHOD
            and method.get("method_variant") == TRACK_B_METHOD_VARIANT
            and method.get("evaluation_scope") == TRACK_B_EVALUATION_SCOPE
            and method.get("target_window") == 20.0
            and method.get("controller_architecture")
            == TRACK_B_CONTROLLER_ARCHITECTURE
            and method.get("controller_action_interface")
            == TRACK_B_CONTROLLER_ACTION_INTERFACE
            and method.get("policy_realization") == TRACK_B_POLICY_REALIZATION
            and method.get("assignment_source_family")
            == "learned_spatial_assignment_policy"
            and method.get("assignment_source_version")
            == "kim2020_grid_actor_critic_adapted_v1"
        ),
        "exact_urgency_scheduler_contract": (
            method.get("urgency_method") == TRACK_B_METHOD
            and method.get("urgency_scheduler_architecture")
            == TRACK_B_CONTROLLER_ARCHITECTURE
            and method.get("urgency_policy_realization")
            == TRACK_B_POLICY_REALIZATION
            and method.get("urgency_controller_action_interface")
            == TRACK_B_CONTROLLER_ACTION_INTERFACE
            and method.get("urgency_decision_epoch_contract")
            == TRACK_B_DECISION_EPOCH_CONTRACT
            and method.get("urgency_assignment_commitment_contract")
            == TRACK_B_ASSIGNMENT_COMMITMENT_CONTRACT
            and method.get("urgency_reservation_contract")
            == TRACK_B_RESERVATION_CONTRACT
            and method.get("urgency_assignment_source")
            == TRACK_A_KIM2020_A3C_SPATIAL
            and method.get("urgency_assignment_source_family")
            == "learned_spatial_assignment_policy"
            and method.get("urgency_assignment_source_version")
            == "kim2020_grid_actor_critic_adapted_v1"
            and method.get("urgency_lookahead_margin_steps") == 2.0
        ),
        "success": row.get("success") == 1.0,
        "raw_strict_method_success": row.get("raw_strict_method_success") == 1.0,
        "complete_required_deliveries": row.get("delivery_count") == EXPECTED_DELIVERIES,
        "within_frozen_step_limit": (
            isinstance(row.get("steps"), int)
            and not isinstance(row.get("steps"), bool)
            and 0 <= row["steps"] <= 2000
        ),
        "not_truncated": row.get("truncated") == 0.0,
        "no_method_failure": row.get("method_failure_reason") is None,
        "reservation_integrity": method.get("reservation_integrity") is True,
        "no_illegal_drops": row.get("illegal_drops") == 0,
        "no_invalid_assignments": method.get("invalid_assignment_count") == 0,
        "no_fallbacks": method.get("fallback_count") == 0,
        "no_inbound_failures": method.get("inbound_failures") == 0,
        "no_retrieve_failures": method.get("retrieve_failures") == 0,
        "no_reservation_invalidations": (
            method.get("reservation_invalidation_count") == 0
        ),
        "selector_frozen": method.get("selector_frozen") is True,
        "selector_counts_complete": (
            method.get("selector_decision_count") == EXPECTED_DELIVERIES
            and method.get("selector_valid_assignment_count") == EXPECTED_DELIVERIES
            and method.get("selector_reserved_commit_count") == EXPECTED_DELIVERIES
            and method.get("selector_post_pickup_recompute_count") == 0
        ),
        "scheduler_counts_complete": (
            method.get("inbound_successes") == EXPECTED_DELIVERIES
            and method.get("retrieve_successes") == EXPECTED_DELIVERIES
            and method.get("reservation_bound_count") == EXPECTED_DELIVERIES
            and method.get("reservation_commit_count") == EXPECTED_DELIVERIES
            and method.get("reservation_execution_match_count")
            == EXPECTED_DELIVERIES
        ),
        "proposal_id_integrity": (
            len(method["bound_proposal_ids"]) == EXPECTED_DELIVERIES
            and len(set(method["bound_proposal_ids"])) == EXPECTED_DELIVERIES
            and method["bound_proposal_ids"]
            == method["committed_proposal_ids"]
            == method["selector_committed_proposal_ids"]
        ),
        "selector_source_identity": (
            method.get("selector_assignment_source")
            == TRACK_A_KIM2020_A3C_SPATIAL
            and method.get("selector_assignment_source_family")
            == "learned_spatial_assignment_policy"
            and method.get("selector_assignment_source_learned") is True
            and method.get("selector_assignment_source_version")
            == "kim2020_grid_actor_critic_adapted_v1"
            and method.get("source_assignment_source")
            == TRACK_A_KIM2020_A3C_SPATIAL
            and method.get("source_assignment_source_family")
            == "learned_spatial_assignment_policy"
            and method.get("source_assignment_source_version")
            == "kim2020_grid_actor_critic_adapted_v1"
            and method.get("source_selector_feature_version")
            == "kim2020_relative_dwell_spatial_v1"
            and method.get("source_reward_contract")
            == "kim2020_delayed_target_retrieval_relocations_v1"
            and method.get("source_return_contract")
            == "kim2020_episodic_discounted_placement_return_v1"
            and method.get("source_action_mapping")
            == "row_major_grid_cell_masked_by_shared_candidates_v1"
            and method.get("source_audit_schema") == SOURCE_AUDIT_SCHEMA
        ),
        "evaluation_learning_disabled": (
            row.get("evaluation_learning") is False
            and method.get("learning_enabled") is False
        ),
        "stochastic_deployment": (
            row.get("deployment_mode") == DEPLOYMENT_STOCHASTIC
            and method.get("deployment_mode") == DEPLOYMENT_STOCHASTIC
        ),
        "exact_policy_seed": method.get("policy_seed") == rng_seed,
        "checkpoint_counters_unchanged": all(
            method.get(f"source_{field}") == expected_counters.get(field)
            for field in (
                "assignment_count", "gradient_steps", "completed_outcome_count",
                "censored_outcome_count",
            )
        ),
        "source_last_episode_is_evaluation_only": (
            last_episode.get("instance_id") == instance_record["episode_instance_id"]
            and last_episode.get("success") is True
            and last_episode.get("truncated") is False
            and last_episode.get("outcome_tracking") is False
            and last_episode.get("integrity") is None
            and last_episode.get("trained") is False
            and last_episode.get("selection_count") == 0
            and last_episode.get("completed_count") == 0
            and last_episode.get("censored_count") == 0
            and last_episode.get("records") == []
        ),
        "storage_manifest_fully_observed": (
            method.get("storage_flow_fully_observed") is True
            and method.get("storage_flow_manifest_count") == EXPECTED_DELIVERIES
            and method.get("storage_flow_completed_count") == EXPECTED_DELIVERIES
            and method.get("storage_flow_unfinished_count") == 0
        ),
        "all_physical_relocations_target_bound": (
            row.get("target_bound_obstruction_clearances") == physical
            and row.get("standalone_reconfigurations") == 0
        ),
        "retrieve_relocation_audit_matches": method.get("retrieve_relocations") == physical,
        "pinned_fresh_selector_execution": (
            row.get("fresh_evaluation_clone") is True
            and method.get("selector_frozen") is True
            and method.get("learning_enabled") is False
            and all(
                method.get(f"source_{field}") == expected_counters.get(field)
                for field in (
                    "assignment_count", "gradient_steps", "completed_outcome_count",
                    "censored_outcome_count",
                )
            )
        ),
    }


def _validate_normalized_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, int],
    manifest: Mapping[str, Any],
    training_completion: Mapping[str, Any],
) -> None:
    _require_equal("normalized row schema", set(row), ROW_FIELDS)
    _verify_self_hash(row, "row_sha256", label="Kim evaluation row")
    panel_index = int(expected["panel_index"])
    panel_records = manifest["episode_instance_panel"]["records"]
    if not 0 <= panel_index < len(panel_records):
        raise EvaluationProtocolError("normalized row panel index is invalid")
    instance_record = panel_records[panel_index]
    _require_equal("normalized panel seed", instance_record["instance_seed"], expected["instance_seed"])
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "model_seed": expected["model_seed"],
        "panel_index": expected["panel_index"],
        "instance_seed": expected["instance_seed"],
        "rollout_index": expected["rollout"],
        "policy_rng_seed": expected["policy_seed"],
        "evaluation_manifest_sha256": manifest["manifest_sha256"],
        "training_completion_sha256": training_completion["completion_sha256"],
        "source_checkpoint_raw_sha256": training_completion[
            "selected_checkpoint_raw_sha256"
        ],
        "source_checkpoint_deployment_sha256": training_completion[
            "selected_checkpoint_deployment_sha256"
        ],
        "selected_training_episode": training_completion["selected_episode"],
        "method_id": KIM_STOCHASTIC,
        "method_category": "stochastic_online_primary",
        "replication_design": "five_policy_rolls_within_fixed_episode_instance",
        "episode_instance_id": instance_record["episode_instance_id"],
        "schedule_id": instance_record["schedule_id"],
        "episode_instance_sha256": instance_record["episode_instance_sha256"],
        "episode_instance_file_raw_sha256": instance_record["saved_file_raw_sha256"],
        "assignment_source": TRACK_A_KIM2020_A3C_SPATIAL,
        "assignment_commitment": DECISION_EPOCH_RESERVED,
        "scheduler_variant": DURATION_AWARE_VARIANT,
        "source_neutral_scheduler": True,
        "deployment_mode": DEPLOYMENT_STOCHASTIC,
        "evaluation_learning": False,
        "fresh_evaluation_clone": True,
        "arrival_rate": 10.0,
        "mean_stay": 80.0,
        "target_window": 20.0,
        "geometry": FROZEN_GEOMETRY,
        "required_deliveries": EXPECTED_DELIVERIES,
    }
    for field, value in expected_identity.items():
        _require_equal(f"normalized row {field}", row.get(field), value)
    deviations = row.get("delivery_deviations")
    if not isinstance(deviations, list):
        raise EvaluationProtocolError("normalized row deviations must be a list")
    _require_equal("normalized delivery count", row.get("delivery_count"), len(deviations))
    timing = _timing_metrics(deviations)
    for field, expected_value in timing.items():
        _require_equal(f"normalized row {field}", row.get(field), expected_value)
    legacy, dense = _dense_rescore(row.get("legacy_environment_return"), deviations)
    _require_close("normalized legacy return", row.get("legacy_environment_return"), legacy)
    _require_close("normalized dense return", row.get("dense_objective_return"), dense)
    physical = _count(
        row.get("physical_storage_relocations"), label="normalized physical relocations"
    )
    try:
        contention = validate_contention_metric_record(row)
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationProtocolError("normalized contention evidence is invalid") from exc
    _require_equal("normalized physical count", contention["physical_storage_relocations"], physical)
    _require_close(
        "normalized physical relocation rate",
        row.get("physical_rehandles_per_100_required_deliveries"),
        100.0 * physical / EXPECTED_DELIVERIES,
    )
    delivery_count = _count(row.get("delivery_count"), label="normalized delivery count")
    _require_close(
        "normalized completion rate",
        row.get("completion_rate"),
        delivery_count / EXPECTED_DELIVERIES,
    )
    steps = _count(row.get("steps"), label="normalized steps")
    if steps > 2000:
        raise EvaluationProtocolError("normalized steps exceed frozen max_steps=2000")
    _count(row.get("illegal_drops"), label="normalized illegal drops")
    for field in ("success", "raw_strict_method_success", "truncated"):
        value = _finite(row.get(field), label=f"normalized {field}")
        if value not in (0.0, 1.0):
            raise EvaluationProtocolError(f"normalized {field} must be binary")
    if row.get("method_failure_reason") is not None and not isinstance(
        row.get("method_failure_reason"), str
    ):
        raise EvaluationProtocolError("normalized failure reason must be null or text")
    checks = row.get("safety_checks")
    issues = row.get("safety_issues")
    if not isinstance(checks, dict) or not isinstance(issues, list):
        raise EvaluationProtocolError("normalized row safety evidence is missing")
    recomputed = _recompute_row_safety(
        row,
        training_completion=training_completion,
        instance_record=instance_record,
    )
    _require_equal("normalized safety checks", checks, recomputed)
    _require_equal(
        "normalized safety issues",
        issues,
        sorted(key for key, passed in recomputed.items() if passed is not True),
    )
    _require_equal("normalized strict-safe flag", row.get("strict_safe_complete"), not issues)


LEDGER_FIELDS = frozenset(
    {
        "schema_version", "protocol", "status", "development_only",
        "final_86xxx_panel_opened", "sealed_622m_policy_namespace_opened",
        "model_seed", "evaluation_manifest_sha256", "training_completion_sha256",
        "selected_checkpoint_raw_sha256", "selected_checkpoint_deployment_sha256",
        "input_fingerprint", "expected_row_count", "observed_row_count",
        "grid", "rows", "row_sha256", "all_rows_strict_safe_complete",
        "unsafe_row_count", "kim_policy_executions", "map_policy_executions",
        "baseline_policy_executions", "training_tree_sha256_before",
        "training_tree_sha256_after", "checkpoint_raw_sha256_before",
        "checkpoint_raw_sha256_after", "checkpoint_deployment_sha256_before",
        "checkpoint_deployment_sha256_after", "complete_case_filtering_used",
        "ledger_sha256",
    }
)

SEED_COMPLETION_FIELDS = frozenset(
    {
        "schema_version", "protocol", "status", "development_only",
        "final_86xxx_panel_opened", "sealed_622m_policy_namespace_opened",
        "model_seed", "evaluation_manifest_sha256", "training_completion_sha256",
        "ledger_path", "ledger_raw_sha256", "ledger_sha256", "row_count",
        "row_sha256", "all_rows_strict_safe_complete", "unsafe_row_count",
        "kim_policy_executions", "input_tree_reauthenticated_after_execution",
        "source_and_parent_roots_reauthenticated_after_execution",
        "serialized_ledger_reloaded_and_validated", "complete_case_filtering_used",
        "completion_sha256",
    }
)


def _input_fingerprint(
    manifest: Mapping[str, Any], training_completion: Mapping[str, Any], model_seed: int
) -> str:
    return _digest_json(
        {
            "evaluation_manifest_sha256": manifest["manifest_sha256"],
            "training_completion_sha256": training_completion["completion_sha256"],
            "episode_instance_panel_sha256": manifest["episode_instance_panel"][
                "records_sha256"
            ],
            "model_seed": model_seed,
            "grid": expected_grid(model_seed),
            "objective_spec": manifest["objective_spec"],
            "evaluation_config": manifest["evaluation_config"],
        }
    )


def _authenticate_seed_ledger(
    output_root: Path,
    *,
    model_seed: int,
    manifest: Mapping[str, Any],
    training_completion: Mapping[str, Any],
) -> dict[str, Any]:
    path = _ledger_path(output_root, model_seed)
    ledger = _load_json(path, expected_type=dict)
    _require_equal("Kim ledger schema", set(ledger), LEDGER_FIELDS)
    _verify_self_hash(ledger, "ledger_sha256", label=f"seed {model_seed} ledger")
    exact_grid = expected_grid(model_seed)
    expected_fixed = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "completed_all_rows_persisted",
        "development_only": True,
        "final_86xxx_panel_opened": False,
        "sealed_622m_policy_namespace_opened": False,
        "model_seed": model_seed,
        "evaluation_manifest_sha256": manifest["manifest_sha256"],
        "training_completion_sha256": training_completion["completion_sha256"],
        "selected_checkpoint_raw_sha256": training_completion[
            "selected_checkpoint_raw_sha256"
        ],
        "selected_checkpoint_deployment_sha256": training_completion[
            "selected_checkpoint_deployment_sha256"
        ],
        "input_fingerprint": _input_fingerprint(
            manifest, training_completion, model_seed
        ),
        "expected_row_count": EXPECTED_ROWS_PER_MODEL,
        "observed_row_count": EXPECTED_ROWS_PER_MODEL,
        "grid": exact_grid,
        "kim_policy_executions": EXPECTED_ROWS_PER_MODEL,
        "map_policy_executions": 0,
        "baseline_policy_executions": 0,
        "training_tree_sha256_before": EXPECTED_TRAINING_TREE_SHA256,
        "training_tree_sha256_after": EXPECTED_TRAINING_TREE_SHA256,
        "checkpoint_raw_sha256_before": training_completion[
            "selected_checkpoint_raw_sha256"
        ],
        "checkpoint_raw_sha256_after": training_completion[
            "selected_checkpoint_raw_sha256"
        ],
        "checkpoint_deployment_sha256_before": training_completion[
            "selected_checkpoint_deployment_sha256"
        ],
        "checkpoint_deployment_sha256_after": training_completion[
            "selected_checkpoint_deployment_sha256"
        ],
        "complete_case_filtering_used": False,
    }
    for field, expected_value in expected_fixed.items():
        _require_equal(f"seed ledger {field}", ledger.get(field), expected_value)
    for field in (
        "model_seed", "expected_row_count", "observed_row_count",
        "unsafe_row_count", "kim_policy_executions", "map_policy_executions",
        "baseline_policy_executions",
    ):
        _count(ledger.get(field), label=f"seed ledger {field}")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS_PER_MODEL:
        raise EvaluationProtocolError("seed ledger does not contain exactly 60 rows")
    for row, grid_row in zip(rows, exact_grid):
        if not isinstance(row, dict):
            raise EvaluationProtocolError("seed ledger row must be a mapping")
        _validate_normalized_row(
            row,
            expected=grid_row,
            manifest=manifest,
            training_completion=training_completion,
        )
    row_hashes = [row["row_sha256"] for row in rows]
    _require_equal("seed ledger row hash registry", ledger.get("row_sha256"), row_hashes)
    unsafe = sum(not row["strict_safe_complete"] for row in rows)
    _require_equal("seed ledger unsafe row count", ledger.get("unsafe_row_count"), unsafe)
    _require_equal(
        "seed ledger strict-safe flag",
        ledger.get("all_rows_strict_safe_complete"),
        unsafe == 0,
    )
    return ledger


def _build_seed_completion(
    output_root: Path,
    *,
    model_seed: int,
    manifest: Mapping[str, Any],
    training_completion: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    ledger_path = _ledger_path(output_root, model_seed)
    eligible = bool(ledger["all_rows_strict_safe_complete"])
    completion: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": (
            "completed_whole_seed_strict_safe"
            if eligible
            else "completed_whole_seed_ineligible_no_numeric_filtering"
        ),
        "development_only": True,
        "final_86xxx_panel_opened": False,
        "sealed_622m_policy_namespace_opened": False,
        "model_seed": model_seed,
        "evaluation_manifest_sha256": manifest["manifest_sha256"],
        "training_completion_sha256": training_completion["completion_sha256"],
        "ledger_path": str(ledger_path.resolve()),
        "ledger_raw_sha256": _sha256(ledger_path),
        "ledger_sha256": ledger["ledger_sha256"],
        "row_count": EXPECTED_ROWS_PER_MODEL,
        "row_sha256": list(ledger["row_sha256"]),
        "all_rows_strict_safe_complete": eligible,
        "unsafe_row_count": int(ledger["unsafe_row_count"]),
        "kim_policy_executions": EXPECTED_ROWS_PER_MODEL,
        "input_tree_reauthenticated_after_execution": True,
        "source_and_parent_roots_reauthenticated_after_execution": True,
        "serialized_ledger_reloaded_and_validated": True,
        "complete_case_filtering_used": False,
    }
    return _add_self_hash(completion, "completion_sha256")


def _authenticate_seed_completion(
    output_root: Path,
    *,
    model_seed: int,
    manifest: Mapping[str, Any],
    training_completion: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = _authenticate_seed_ledger(
        output_root,
        model_seed=model_seed,
        manifest=manifest,
        training_completion=training_completion,
    )
    expected = _build_seed_completion(
        output_root,
        model_seed=model_seed,
        manifest=manifest,
        training_completion=training_completion,
        ledger=ledger,
    )
    observed = _load_json(
        _seed_completion_path(output_root, model_seed), expected_type=dict
    )
    _require_equal("seed completion schema", set(observed), SEED_COMPLETION_FIELDS)
    _verify_self_hash(
        observed, "completion_sha256", label=f"seed {model_seed} evaluation completion"
    )
    for field in ("model_seed", "row_count", "unsafe_row_count", "kim_policy_executions"):
        _count(observed.get(field), label=f"seed completion {field}")
    _require_equal(f"seed {model_seed} evaluation completion", observed, expected)
    return observed, ledger


def _evaluation_args() -> SimpleNamespace:
    return SimpleNamespace(
        lam=10.0,
        mu=80.0,
        grid_rows=5,
        grid_cols=5,
        exit_width=None,
        number_blocks=EXPECTED_DELIVERIES,
        max_steps=2000,
        max_defer_steps=10,
        lookahead_margin_steps=2.0,
        target_window=20.0,
        assignment_commitment=DECISION_EPOCH_RESERVED,
        allow_selector_regime_shift=False,
        instance=None,
        save_instances_dir=None,
        device="cpu",
    )


def _evaluate_seed(
    project_root: Path,
    training_root: Path,
    output_root: Path,
    model_seed: int,
) -> dict[str, Any]:
    _validate_output_tree(output_root)
    with _exclusive_seed_lock(output_root, model_seed):
        ledger_path = _ledger_path(output_root, model_seed)
        completion_path = _seed_completion_path(output_root, model_seed)
        ledger_exists = ledger_path.exists() or ledger_path.is_symlink()
        completion_exists = completion_path.exists() or completion_path.is_symlink()
        if completion_exists:
            raise EvaluationProtocolError(
                f"model seed {model_seed} already has evaluation output; no rerun/overwrite"
            )
        manifest, training_completions, instances = _authenticate_prepared(
            project_root, training_root, output_root
        )
        training_completion = training_completions[model_seed]
        if ledger_exists:
            # Crash-safe resume point: a complete write-once ledger is already
            # the only row authority.  Authenticate it, execute zero rollouts,
            # and derive only the missing completion marker.
            persisted_ledger = _authenticate_seed_ledger(
                output_root,
                model_seed=model_seed,
                manifest=manifest,
                training_completion=training_completion,
            )
            _authenticate_prepared(project_root, training_root, output_root)
            seed_completion = _build_seed_completion(
                output_root,
                model_seed=model_seed,
                manifest=manifest,
                training_completion=training_completion,
                ledger=persisted_ledger,
            )
            _write_new_json(completion_path, seed_completion)
            persisted_completion, _ = _authenticate_seed_completion(
                output_root,
                model_seed=model_seed,
                manifest=manifest,
                training_completion=training_completion,
            )
            _authenticate_prepared(project_root, training_root, output_root)
            return persisted_completion
        checkpoint_path = (
            training_root / "training" / f"seed-{model_seed}" / "best.pth"
        )
        checkpoint_payload = _load_checkpoint_bound(
            checkpoint_path,
            training_completion["selected_checkpoint_raw_sha256"],
        )
        checkpoint_digest = kim2020_deployment_digest(checkpoint_payload)
        _require_equal(
            "pre-execution checkpoint deployment digest",
            checkpoint_digest,
            training_completion["selected_checkpoint_deployment_sha256"],
        )
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        _require_equal("resolved evaluation device", str(resolve_device("cpu")), "cpu")
        args = _evaluation_args()
        panel_records = {
            int(record["instance_seed"]): record
            for record in manifest["episode_instance_panel"]["records"]
        }
        normalized_rows: list[dict[str, Any]] = []
        for grid_row in expected_grid(model_seed):
            instance_seed = grid_row["instance_seed"]
            raw = _evaluate_one(
                args,
                instance_seed,
                checkpoint_payload,
                episode_instance=instances[instance_seed],
                scheduler_variant=DURATION_AWARE_VARIANT,
                assignment_source=TRACK_A_KIM2020_A3C_SPATIAL,
                source_neutral_scheduler=True,
                assignment_commitment=DECISION_EPOCH_RESERVED,
                deployment_mode=DEPLOYMENT_STOCHASTIC,
                policy_seed=grid_row["policy_seed"],
            )
            if not isinstance(raw, Mapping):
                raise EvaluationProtocolError("Track-B evaluator returned a non-mapping row")
            normalized_rows.append(
                _normalize_run(
                    raw,
                    grid_row=grid_row,
                    instance_record=panel_records[instance_seed],
                    training_completion=training_completion,
                    evaluation_manifest_sha256=manifest["manifest_sha256"],
                )
            )
        _require_equal(
            "post-execution in-memory checkpoint digest",
            kim2020_deployment_digest(checkpoint_payload),
            checkpoint_digest,
        )

        # Reconstruct every input/source/parent root after the final rollout and
        # before committing any output.  The exact checkpoint bytes are loaded
        # again rather than trusting the in-memory payload.
        post_manifest, post_completions, _ = _authenticate_prepared(
            project_root, training_root, output_root
        )
        _require_equal("pre/post evaluation manifest", post_manifest, manifest)
        _require_equal(
            "pre/post training completion",
            post_completions[model_seed],
            training_completion,
        )
        post_checkpoint = _load_checkpoint_bound(
            checkpoint_path,
            training_completion["selected_checkpoint_raw_sha256"],
        )
        _require_equal(
            "post-execution checkpoint deployment digest",
            kim2020_deployment_digest(post_checkpoint),
            checkpoint_digest,
        )
        unsafe = sum(not row["strict_safe_complete"] for row in normalized_rows)
        ledger: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "completed_all_rows_persisted",
            "development_only": True,
            "final_86xxx_panel_opened": False,
            "sealed_622m_policy_namespace_opened": False,
            "model_seed": model_seed,
            "evaluation_manifest_sha256": manifest["manifest_sha256"],
            "training_completion_sha256": training_completion["completion_sha256"],
            "selected_checkpoint_raw_sha256": training_completion[
                "selected_checkpoint_raw_sha256"
            ],
            "selected_checkpoint_deployment_sha256": checkpoint_digest,
            "input_fingerprint": _input_fingerprint(
                manifest, training_completion, model_seed
            ),
            "expected_row_count": EXPECTED_ROWS_PER_MODEL,
            "observed_row_count": len(normalized_rows),
            "grid": expected_grid(model_seed),
            "rows": normalized_rows,
            "row_sha256": [row["row_sha256"] for row in normalized_rows],
            "all_rows_strict_safe_complete": unsafe == 0,
            "unsafe_row_count": unsafe,
            "kim_policy_executions": len(normalized_rows),
            "map_policy_executions": 0,
            "baseline_policy_executions": 0,
            "training_tree_sha256_before": EXPECTED_TRAINING_TREE_SHA256,
            "training_tree_sha256_after": EXPECTED_TRAINING_TREE_SHA256,
            "checkpoint_raw_sha256_before": training_completion[
                "selected_checkpoint_raw_sha256"
            ],
            "checkpoint_raw_sha256_after": _sha256(checkpoint_path),
            "checkpoint_deployment_sha256_before": checkpoint_digest,
            "checkpoint_deployment_sha256_after": kim2020_deployment_digest(
                post_checkpoint
            ),
            "complete_case_filtering_used": False,
        }
        ledger = _add_self_hash(ledger, "ledger_sha256")
        _write_new_json(ledger_path, ledger)

        # Only the deserialized, fully revalidated ledger is authoritative.
        persisted_ledger = _authenticate_seed_ledger(
            output_root,
            model_seed=model_seed,
            manifest=manifest,
            training_completion=training_completion,
        )
        _authenticate_prepared(project_root, training_root, output_root)
        seed_completion = _build_seed_completion(
            output_root,
            model_seed=model_seed,
            manifest=manifest,
            training_completion=training_completion,
            ledger=persisted_ledger,
        )
        _write_new_json(completion_path, seed_completion)
        persisted_completion, _ = _authenticate_seed_completion(
            output_root,
            model_seed=model_seed,
            manifest=manifest,
            training_completion=training_completion,
        )
        _authenticate_prepared(project_root, training_root, output_root)
        return persisted_completion


def _mean(values: Iterable[Any], *, label: str) -> float:
    finite_values = [_finite(value, label=label) for value in values]
    if not finite_values:
        raise EvaluationProtocolError(f"cannot average an empty sequence: {label}")
    return float(statistics.fmean(finite_values))


def _row_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "mean_dense_objective_return": _finite(
            row.get("dense_objective_return"), label="row dense return"
        ),
        "mean_absolute_error": _finite(
            row.get("mean_absolute_error"), label="row MAE"
        ),
        "mean_signed_deviation": _finite(
            row.get("mean_signed_deviation"), label="row signed deviation"
        ),
        "mean_tardiness": _finite(
            row.get("mean_tardiness"), label="row tardiness"
        ),
        "mean_earliness": _finite(
            row.get("mean_earliness"), label="row earliness"
        ),
        "within_target_window_rate": _finite(
            row.get("within_target_window_rate"), label="row window rate"
        ),
        "mean_steps": _finite(row.get("steps"), label="row steps"),
        "physical_rehandles_per_100_required_deliveries": _finite(
            row.get("physical_rehandles_per_100_required_deliveries"),
            label="row physical rehandle rate",
        ),
    }


def _mean_metric_mappings(
    metrics: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, float]:
    if not metrics:
        raise EvaluationProtocolError(f"no metric records for {label}")
    for record in metrics:
        _require_equal(f"{label} metric schema", set(record), set(REPORT_METRICS))
    return {
        metric: _mean(
            (record[metric] for record in metrics), label=f"{label} {metric}"
        )
        for metric in REPORT_METRICS
    }


def _descriptive_variability(
    metric_records: Sequence[Mapping[str, Any]], *, unit: str
) -> dict[str, Any]:
    if len(metric_records) < 2:
        raise EvaluationProtocolError("descriptive variability needs at least two units")
    result: dict[str, Any] = {}
    for metric in REPORT_METRICS:
        values = [
            _finite(record[metric], label=f"{unit} {metric}")
            for record in metric_records
        ]
        result[metric] = {
            f"n_{unit}": len(values),
            "minimum": min(values),
            "maximum": max(values),
            "range": max(values) - min(values),
            "sample_standard_deviation": float(statistics.stdev(values)),
            "inferential_claim_authorized": False,
        }
    return result


def _cluster_statistics(
    cluster_points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(cluster_points) != len(PANEL_SEEDS):
        raise EvaluationProtocolError("instance-cluster statistics require exactly 12 points")
    result: dict[str, Any] = {}
    for metric in REPORT_METRICS:
        values = [
            _finite(point["metrics"][metric], label=f"cluster {metric}")
            for point in cluster_points
        ]
        standard_deviation = float(statistics.stdev(values))
        result[metric] = {
            "statistical_unit": "EpisodeInstance_cluster",
            "n_episode_instance_clusters": len(values),
            "mean": float(statistics.fmean(values)),
            "sample_standard_deviation": standard_deviation,
            "standard_error": standard_deviation / math.sqrt(len(values)),
            "inferential_claim_authorized": False,
            "raw_rows_treated_as_independent_units": False,
        }
    return result


def _summarize_kim_ledgers(
    ledgers: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    _require_equal("Kim ledger seed set", set(ledgers), set(MODEL_SEEDS))
    rows = [row for seed in MODEL_SEEDS for row in ledgers[seed]["rows"]]
    if len(rows) != EXPECTED_ROWS:
        raise EvaluationProtocolError("Kim analysis requires exactly 180 persisted rows")
    unsafe_rows = [
        {
            "model_seed": row["model_seed"],
            "instance_seed": row["instance_seed"],
            "rollout_index": row["rollout_index"],
            "policy_rng_seed": row["policy_rng_seed"],
            "safety_issues": list(row["safety_issues"]),
        }
        for row in rows
        if row["strict_safe_complete"] is not True
    ]
    if unsafe_rows:
        return {
            "method_id": KIM_STOCHASTIC,
            "status": "whole_method_ineligible_metrics_suppressed",
            "whole_method_numeric_eligible": False,
            "expected_row_count": EXPECTED_ROWS,
            "observed_row_count": len(rows),
            "strict_safe_complete_row_count": len(rows) - len(unsafe_rows),
            "unsafe_row_count": len(unsafe_rows),
            "unsafe_rows": unsafe_rows,
            "complete_case_filtering_used": False,
            "aggregation_order": (
                "equal_5_policy_rolls_then_equal_12_instances_then_equal_3_model_seeds"
            ),
            "model_instance_points": [],
            "model_summaries": [],
            "equal_model_summary": None,
            "instance_cluster_points": [],
            "instance_cluster_statistics": None,
            "descriptive_model_seed_variability": None,
            "metrics": None,
        }

    model_instance_points: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    for model_seed in MODEL_SEEDS:
        seed_rows = [row for row in rows if row["model_seed"] == model_seed]
        if len(seed_rows) != EXPECTED_ROWS_PER_MODEL:
            raise EvaluationProtocolError(f"seed {model_seed} does not have 60 rows")
        seed_points: list[dict[str, Any]] = []
        for panel_index, instance_seed in enumerate(PANEL_SEEDS):
            group = [
                row
                for row in seed_rows
                if row["panel_index"] == panel_index
                and row["instance_seed"] == instance_seed
            ]
            _require_equal(
                f"seed {model_seed} instance {instance_seed} rollout set",
                [row["rollout_index"] for row in group],
                list(ROLLOUTS),
            )
            point_metrics = _mean_metric_mappings(
                [_row_metrics(row) for row in group],
                label=f"seed {model_seed} instance {instance_seed}",
            )
            physical_total = sum(
                _count(
                    row["physical_storage_relocations"],
                    label="physical storage relocations",
                )
                for row in group
            )
            _require_close(
                "hierarchical instance physical rate",
                point_metrics[
                    "physical_rehandles_per_100_required_deliveries"
                ],
                100.0 * physical_total / (len(ROLLOUTS) * EXPECTED_DELIVERIES),
            )
            point = {
                "model_seed": model_seed,
                "panel_index": panel_index,
                "instance_seed": instance_seed,
                "episode_instance_id": group[0]["episode_instance_id"],
                "policy_roll_count": len(group),
                "policy_rng_seeds": [row["policy_rng_seed"] for row in group],
                "total_physical_storage_relocations": physical_total,
                "required_delivery_denominator": len(ROLLOUTS)
                * EXPECTED_DELIVERIES,
                "metrics": point_metrics,
            }
            seed_points.append(point)
            model_instance_points.append(point)
        model_metrics = _mean_metric_mappings(
            [point["metrics"] for point in seed_points],
            label=f"seed {model_seed} equal-instance",
        )
        model_summaries.append(
            {
                "model_seed": model_seed,
                "policy_rolls_per_instance": len(ROLLOUTS),
                "instance_count": len(seed_points),
                "row_count": len(seed_rows),
                "whole_seed_strict_safe_complete": True,
                "metrics": model_metrics,
            }
        )
    _require_equal("model-instance point count", len(model_instance_points), 36)
    equal_model_metrics = _mean_metric_mappings(
        [summary["metrics"] for summary in model_summaries],
        label="equal-model Kim aggregate",
    )
    cluster_points: list[dict[str, Any]] = []
    for panel_index, instance_seed in enumerate(PANEL_SEEDS):
        points = [
            point
            for point in model_instance_points
            if point["panel_index"] == panel_index
        ]
        _require_equal(
            f"instance cluster {instance_seed} model set",
            [point["model_seed"] for point in points],
            list(MODEL_SEEDS),
        )
        cluster_points.append(
            {
                "panel_index": panel_index,
                "instance_seed": instance_seed,
                "episode_instance_id": points[0]["episode_instance_id"],
                "model_seed_count": len(points),
                "policy_roll_count_per_model_instance": len(ROLLOUTS),
                "raw_row_count": len(points) * len(ROLLOUTS),
                "metrics": _mean_metric_mappings(
                    [point["metrics"] for point in points],
                    label=f"instance cluster {instance_seed}",
                ),
            }
        )
    return {
        "method_id": KIM_STOCHASTIC,
        "status": "whole_method_strict_safe_numeric_eligible",
        "whole_method_numeric_eligible": True,
        "expected_row_count": EXPECTED_ROWS,
        "observed_row_count": len(rows),
        "strict_safe_complete_row_count": len(rows),
        "unsafe_row_count": 0,
        "unsafe_rows": [],
        "complete_case_filtering_used": False,
        "aggregation_order": (
            "equal_5_policy_rolls_then_equal_12_instances_then_equal_3_model_seeds"
        ),
        "model_instance_points": model_instance_points,
        "model_summaries": model_summaries,
        "equal_model_summary": {
            "model_seeds": list(MODEL_SEEDS),
            "model_seed_count": len(MODEL_SEEDS),
            "metrics": equal_model_metrics,
        },
        "instance_cluster_points": cluster_points,
        "instance_cluster_statistics": _cluster_statistics(cluster_points),
        "descriptive_model_seed_variability": _descriptive_variability(
            [summary["metrics"] for summary in model_summaries],
            unit="model_seeds",
        ),
        "metrics": equal_model_metrics,
    }


def _table_metrics(metrics: Mapping[str, Any], *, label: str) -> dict[str, float]:
    return {
        metric: _finite(metrics.get(metric), label=f"{label} {metric}")
        for metric in TABLE_METRICS
    }


def _comparison_table(
    repair: Mapping[str, Any],
    stability: Mapping[str, Any],
    kim_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = repair.get("method_summaries")
    if not isinstance(summaries, list):
        raise EvaluationProtocolError("repair report has no method summaries")
    by_id: dict[str, Mapping[str, Any]] = {}
    for summary in summaries:
        if not isinstance(summary, Mapping) or not isinstance(
            summary.get("method_id"), str
        ):
            raise EvaluationProtocolError("repair method summary is malformed")
        method_id = summary["method_id"]
        if method_id in by_id:
            raise EvaluationProtocolError(f"duplicate repair method: {method_id}")
        by_id[method_id] = summary
    expected_registry = [
        "vcg_constrained_v2_3_gamma1_episode160",
        "vcg_dense_v1_1_selected_three_seed",
        "duration_aware_nearest_free",
        "duration_aware_dynamic_pslap",
        "duration_aware_pslap_ga_2009_rolling",
        "duration_aware_pslap_ga_duration_aware_rolling",
        "duration_aware_pslap_ga_operational_rolling",
        "duration_aware_enhanced_complete_rolling_ga",
        "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial",
        "duration_aware_pslap_ga_duration_aware_rolling_capacity_aware_partial",
        "duration_aware_pslap_ga_operational_rolling_capacity_aware_partial",
        "duration_aware_enhanced_complete_rolling_ga_capacity_aware_partial",
    ]
    _require_equal("repair method registry", list(by_id), expected_registry)
    failed_ids = expected_registry[4:8]
    _require_equal(
        "repair whole-method exclusions",
        repair.get("whole_method_safety_exclusions"),
        failed_ids,
    )
    suppressed = []
    for method_id in failed_ids:
        summary = by_id[method_id]
        _require_equal(f"failed method {method_id} eligibility", summary.get("whole_method_numeric_eligible"), False)
        _require_equal(f"failed method {method_id} metrics", summary.get("metrics"), None)
        suppressed.append(
            {
                "method_id": method_id,
                "status": "historical_failed_adapter_retained_metrics_suppressed",
                "repaired_successor": repair["method_registry"][
                    "historical_overconstrained_ga_adapters_retained"
                ][failed_ids.index(method_id)]["repaired_successor"],
            }
        )

    equal = stability.get("equal_seed_aggregate")
    if not isinstance(equal, Mapping):
        raise EvaluationProtocolError("stability equal-seed aggregate is missing")
    _require_equal("equal VCG available", equal.get("available"), True)
    _require_equal("equal VCG seeds", equal.get("model_seeds"), [11, 12, 13])
    rows: list[dict[str, Any]] = [
        {
            "table_order": 1,
            "method_id": NUMERIC_TABLE_METHOD_IDS[0],
            "display_name": DISPLAY_LABELS[NUMERIC_TABLE_METHOD_IDS[0]],
            "source": "authenticated_stability_equal_seed_aggregate",
            "whole_method_numeric_eligible": True,
            "metrics": _table_metrics(equal["metrics"], label="equal VCG"),
        }
    ]
    for method_id in ELIGIBLE_BASELINE_IDS:
        summary = by_id[method_id]
        _require_equal(
            f"baseline {method_id} eligibility",
            summary.get("whole_method_numeric_eligible"),
            True,
        )
        if not isinstance(summary.get("metrics"), Mapping):
            raise EvaluationProtocolError(f"baseline {method_id} metrics are absent")
        rows.append(
            {
                "table_order": len(rows) + 1,
                "method_id": method_id,
                "display_name": DISPLAY_LABELS[method_id],
                "source": "authenticated_repair_v2_summary_reused_once",
                "whole_method_numeric_eligible": True,
                "metrics": _table_metrics(
                    summary["metrics"], label=f"baseline {method_id}"
                ),
            }
        )
    kim_eligible = kim_summary.get("whole_method_numeric_eligible") is True
    rows.append(
        {
            "table_order": len(rows) + 1,
            "method_id": KIM_STOCHASTIC,
            "display_name": DISPLAY_LABELS[KIM_STOCHASTIC],
            "source": "new_authenticated_180_row_hierarchical_aggregate",
            "whole_method_numeric_eligible": kim_eligible,
            "metrics": (
                _table_metrics(kim_summary["metrics"], label="Kim aggregate")
                if kim_eligible
                else None
            ),
        }
    )
    _require_equal(
        "all-method comparison order",
        [row["method_id"] for row in rows],
        list(NUMERIC_TABLE_METHOD_IDS),
    )
    _require_equal("all-method comparison row count", len(rows), 9)
    return rows, suppressed


def _dominates(
    challenger: Mapping[str, Any], candidate: Mapping[str, Any]
) -> bool:
    challenger_values = [
        _finite(challenger.get(metric), label=f"challenger {metric}")
        for metric in GATE_METRICS
    ]
    candidate_values = [
        _finite(candidate.get(metric), label=f"candidate {metric}")
        for metric in GATE_METRICS
    ]
    weak_both = all(
        challenger_value <= candidate_value
        for challenger_value, candidate_value in zip(
            challenger_values, candidate_values
        )
    )
    strict_one = any(
        challenger_value < candidate_value
        for challenger_value, candidate_value in zip(
            challenger_values, candidate_values
        )
    )
    return weak_both and strict_one


def _screen_candidate(
    *,
    candidate_id: str,
    candidate_metrics: Mapping[str, Any],
    comparators: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dominators = sorted(
        comparator["method_id"]
        for comparator in comparators
        if _dominates(comparator["metrics"], candidate_metrics)
    )
    return {
        "candidate_id": candidate_id,
        "coordinates": {
            metric: _finite(candidate_metrics.get(metric), label=f"candidate {metric}")
            for metric in GATE_METRICS
        },
        "dominator_method_ids": dominators,
        "non_dominated": not dominators,
    }


def _extended_gate(
    comparison_table: Sequence[Mapping[str, Any]],
    stability: Mapping[str, Any],
    kim_summary: Mapping[str, Any],
) -> dict[str, Any]:
    fixed = {
        "coordinates": list(GATE_METRICS),
        "dominance": "weak_both_strict_at_least_one",
        "comparison_tolerance": 0.0,
        "scalarization_used": False,
        "return_and_steps_are_reported_not_gated": True,
        "kim_required_to_win": False,
        "required_equal_seed_vcg_nondominated": True,
        "minimum_nondominated_individual_vcg_seeds": 2,
        "eligible_comparator_count_without_vcg": 8,
    }
    if kim_summary.get("whole_method_numeric_eligible") is not True:
        return {
            **fixed,
            "status": "unevaluable_kim_whole_method_ineligible",
            "comparator_method_ids": [],
            "equal_seed_screen": None,
            "individual_seed_screens": [],
            "nondominated_individual_seed_count": None,
            "passed": None,
        }
    comparator_rows = [
        row
        for row in comparison_table
        if row["method_id"] != NUMERIC_TABLE_METHOD_IDS[0]
        and row["whole_method_numeric_eligible"] is True
    ]
    _require_equal("extended comparator count", len(comparator_rows), 8)
    _require_equal(
        "extended comparator ids",
        [row["method_id"] for row in comparator_rows],
        [*ELIGIBLE_BASELINE_IDS, KIM_STOCHASTIC],
    )
    equal = stability.get("equal_seed_aggregate")
    if not isinstance(equal, Mapping) or not isinstance(equal.get("metrics"), Mapping):
        raise EvaluationProtocolError("equal-seed VCG metrics are missing")
    equal_screen = _screen_candidate(
        candidate_id="vcg_constrained_v2_3_equal_stability_seeds_11_13",
        candidate_metrics=equal["metrics"],
        comparators=comparator_rows,
    )
    individual = stability.get("individual_seed_summaries")
    if not isinstance(individual, list) or len(individual) != 3:
        raise EvaluationProtocolError("stability report lacks three seed summaries")
    by_seed = {summary.get("model_seed"): summary for summary in individual}
    _require_equal("individual stability seeds", set(by_seed), {11, 12, 13})
    individual_screens = []
    for seed in (11, 12, 13):
        summary = by_seed[seed]
        _require_equal(f"VCG seed {seed} eligible", summary.get("development_candidate_eligible"), True)
        if not isinstance(summary.get("metrics"), Mapping):
            raise EvaluationProtocolError(f"VCG seed {seed} metrics are missing")
        individual_screens.append(
            _screen_candidate(
                candidate_id=f"vcg_constrained_v2_3_seed_{seed}",
                candidate_metrics=summary["metrics"],
                comparators=comparator_rows,
            )
        )
    nondominated_count = sum(
        screen["non_dominated"] is True for screen in individual_screens
    )
    passed = equal_screen["non_dominated"] is True and nondominated_count >= 2
    return {
        **fixed,
        "status": "passed" if passed else "failed",
        "comparator_method_ids": [row["method_id"] for row in comparator_rows],
        "equal_seed_screen": equal_screen,
        "individual_seed_screens": individual_screens,
        "nondominated_individual_seed_count": nondominated_count,
        "passed": passed,
    }


def _load_parent_payloads(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _authenticate_parent_artifacts(project_root)
    repair = _load_json(project_root / REPAIR_REPORT_RELATIVE, expected_type=dict)
    stability = _load_json(project_root / STABILITY_REPORT_RELATIVE, expected_type=dict)
    return repair, stability


def _build_extended_report(
    *,
    project_root: Path,
    manifest: Mapping[str, Any],
    evaluation_completions: Mapping[int, Mapping[str, Any]],
    ledgers: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    repair, stability = _load_parent_payloads(project_root)
    kim_summary = _summarize_kim_ledgers(ledgers)
    table, suppressed = _comparison_table(repair, stability, kim_summary)
    gate = _extended_gate(table, stability, kim_summary)
    completion_registry = [
        {
            "model_seed": seed,
            "evaluation_completion_sha256": evaluation_completions[seed][
                "completion_sha256"
            ],
            "evaluation_completion_raw_sha256": _sha256(
                _seed_completion_path(Path(manifest["output_root"]), seed)
            ),
            "ledger_sha256": ledgers[seed]["ledger_sha256"],
            "ledger_raw_sha256": _sha256(
                _ledger_path(Path(manifest["output_root"]), seed)
            ),
            "row_count": ledgers[seed]["observed_row_count"],
            "all_rows_strict_safe_complete": ledgers[seed][
                "all_rows_strict_safe_complete"
            ],
        }
        for seed in MODEL_SEEDS
    ]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": (
            "extended_development_gate_evaluable"
            if kim_summary["whole_method_numeric_eligible"]
            else "kim_whole_method_ineligible_metrics_and_gate_suppressed"
        ),
        "development_only": True,
        "performance_claim_authorized": False,
        "confirmatory_claim_authorized": False,
        "original_v2_3_stability_verdict_mutated": False,
        "final_86xxx_panel_opened": False,
        "sealed_622m_policy_namespace_opened": False,
        "evaluation_manifest_sha256": manifest["manifest_sha256"],
        "authenticated_parent_artifacts": manifest["parent_artifacts"],
        "training_completion_files": manifest["training_completion_files"],
        "evaluation_completion_registry": completion_registry,
        "kim_policy_executions": EXPECTED_ROWS,
        "baseline_policy_executions": 0,
        "map_policy_executions": 0,
        "analysis_policy_executions": 0,
        "complete_case_filtering_used": False,
        "aggregation_order": (
            "equal_5_policy_rolls_then_equal_12_instances_then_equal_3_model_seeds"
        ),
        "raw_180_rows_are_not_independent_statistical_units": True,
        "kim_summary": kim_summary,
        "comparison_table": table,
        "comparison_table_method_ids": [row["method_id"] for row in table],
        "historical_failed_ga_variants_retained_and_suppressed": suppressed,
        "baseline_summaries_reloaded_not_copied_or_multiplied": True,
        "baseline_summary_reuse_count_per_method": 1,
        "development_seed10_role": (
            "historical_diagnostic_only_not_a_gate_comparator"
        ),
        "extended_gate": gate,
        "return_metric_relationship_note": (
            "dense return is the frozen fixed-trajectory timing rescore; MAE and "
            "timing components are reported jointly, while the gate uses only MAE "
            "and total physical rehandles without scalarization"
        ),
        "scope_note": (
            "development evidence is specific to the fixed 5x5, eight-block, "
            "arrival-rate-10, mean-stay-80, 85xxx EpisodeInstance panel"
        ),
    }
    return _add_self_hash(report, "report_sha256")


def _build_extended_audit(
    *,
    project_root: Path,
    training_root: Path,
    output_root: Path,
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    report_path = output_root / "extended-report.json"
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete_fail_closed_development_analysis",
        "development_only": True,
        "performance_claim_authorized": False,
        "confirmatory_claim_authorized": False,
        "original_v2_3_stability_verdict_mutated": False,
        "final_86xxx_panel_opened": False,
        "sealed_622m_policy_namespace_opened": False,
        "evaluation_manifest_sha256": manifest["manifest_sha256"],
        "report_path": str(report_path.resolve()),
        "report_raw_sha256": _sha256(report_path),
        "report_sha256": report["report_sha256"],
        "immutable_training_root": str(training_root),
        "immutable_training_tree_sha256": EXPECTED_TRAINING_TREE_SHA256,
        "project_root": str(project_root),
        "output_root": str(output_root),
        "training_or_evaluation_rollouts_executed_during_analysis": 0,
        "kim_policy_executions_authenticated": EXPECTED_ROWS,
        "baseline_policy_executions": 0,
        "map_policy_executions": 0,
        "complete_case_filtering_used": False,
        "all_three_serialized_ledgers_reloaded_and_semantically_validated": True,
        "all_180_rows_rebound_to_grid_checkpoint_instance_and_executor": True,
        "inputs_sources_and_parent_artifacts_reauthenticated_pre_and_post": True,
        "hierarchical_aggregation_recomputed_from_serialized_rows": True,
        "comparison_table_recomputed_from_authenticated_parent_summaries": True,
        "extended_gate_recomputed_without_scalarization": True,
        "report_is_development_only_no_performance_or_confirmatory_claim": True,
    }
    return _add_self_hash(audit, "audit_sha256")


def _load_all_evaluation_completions(
    *,
    output_root: Path,
    manifest: Mapping[str, Any],
    training_completions: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    completions: dict[int, dict[str, Any]] = {}
    ledgers: dict[int, dict[str, Any]] = {}
    for seed in MODEL_SEEDS:
        completion, ledger = _authenticate_seed_completion(
            output_root,
            model_seed=seed,
            manifest=manifest,
            training_completion=training_completions[seed],
        )
        completions[seed] = completion
        ledgers[seed] = ledger
    return completions, ledgers


def _authenticate_analysis(
    project_root: Path,
    training_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, training_completions, _ = _authenticate_prepared(
        project_root, training_root, output_root
    )
    completions, ledgers = _load_all_evaluation_completions(
        output_root=output_root,
        manifest=manifest,
        training_completions=training_completions,
    )
    expected_report = _build_extended_report(
        project_root=project_root,
        manifest=manifest,
        evaluation_completions=completions,
        ledgers=ledgers,
    )
    report = _load_json(output_root / "extended-report.json", expected_type=dict)
    _verify_self_hash(report, "report_sha256", label="extended report")
    _require_equal("extended report", report, expected_report)
    expected_audit = _build_extended_audit(
        project_root=project_root,
        training_root=training_root,
        output_root=output_root,
        manifest=manifest,
        report=report,
    )
    audit = _load_json(output_root / "extended-audit.json", expected_type=dict)
    _verify_self_hash(audit, "audit_sha256", label="extended audit")
    _require_equal("extended audit", audit, expected_audit)
    return report, audit


def _analyze(
    project_root: Path, training_root: Path, output_root: Path
) -> dict[str, Any]:
    report_path = output_root / "extended-report.json"
    audit_path = output_root / "extended-audit.json"
    report_exists = report_path.exists() or report_path.is_symlink()
    audit_exists = audit_path.exists() or audit_path.is_symlink()
    if audit_exists:
        raise EvaluationProtocolError("analysis output already exists; no overwrite")
    manifest, training_completions, _ = _authenticate_prepared(
        project_root, training_root, output_root
    )
    completions, ledgers = _load_all_evaluation_completions(
        output_root=output_root,
        manifest=manifest,
        training_completions=training_completions,
    )
    # Reauthenticate after all serialized inputs were read and before commit.
    post_manifest, post_training_completions, _ = _authenticate_prepared(
        project_root, training_root, output_root
    )
    _require_equal("pre/post analysis manifest", post_manifest, manifest)
    _require_equal(
        "pre/post analysis training completions",
        post_training_completions,
        training_completions,
    )
    report = _build_extended_report(
        project_root=project_root,
        manifest=manifest,
        evaluation_completions=completions,
        ledgers=ledgers,
    )
    if not report_exists:
        _write_new_json(report_path, report)
    persisted_report = _load_json(report_path, expected_type=dict)
    _verify_self_hash(persisted_report, "report_sha256", label="persisted report")
    _require_equal("persisted report", persisted_report, report)
    audit = _build_extended_audit(
        project_root=project_root,
        training_root=training_root,
        output_root=output_root,
        manifest=manifest,
        report=persisted_report,
    )
    _write_new_json(audit_path, audit)
    authenticated_report, authenticated_audit = _authenticate_analysis(
        project_root, training_root, output_root
    )
    _authenticate_prepared(project_root, training_root, output_root)
    _require_equal("authenticated persisted report", authenticated_report, report)
    _require_equal("authenticated persisted audit", authenticated_audit, audit)
    return authenticated_report


def _validate(
    project_root: Path, training_root: Path, output_root: Path
) -> dict[str, Any]:
    manifest, training_completions, _ = _authenticate_prepared(
        project_root, training_root, output_root
    )
    seed_status: list[dict[str, Any]] = []
    for seed in MODEL_SEEDS:
        ledger_exists = _ledger_path(output_root, seed).exists()
        completion_exists = _seed_completion_path(output_root, seed).exists()
        if completion_exists and not ledger_exists:
            raise EvaluationProtocolError("evaluation completion exists without ledger")
        if completion_exists:
            completion, _ = _authenticate_seed_completion(
                output_root,
                model_seed=seed,
                manifest=manifest,
                training_completion=training_completions[seed],
            )
            status = completion["status"]
        elif ledger_exists:
            _authenticate_seed_ledger(
                output_root,
                model_seed=seed,
                manifest=manifest,
                training_completion=training_completions[seed],
            )
            status = "serialized_ledger_valid_completion_recoverable_without_rollout"
        else:
            status = "not_evaluated"
        seed_status.append({"model_seed": seed, "status": status})
    report_exists = (output_root / "extended-report.json").exists()
    audit_exists = (output_root / "extended-audit.json").exists()
    if report_exists and audit_exists:
        report, audit = _authenticate_analysis(
            project_root, training_root, output_root
        )
        analysis_status = report["status"]
        audit_sha256 = audit["audit_sha256"]
    elif report_exists:
        completions, ledgers = _load_all_evaluation_completions(
            output_root=output_root,
            manifest=manifest,
            training_completions=training_completions,
        )
        expected_report = _build_extended_report(
            project_root=project_root,
            manifest=manifest,
            evaluation_completions=completions,
            ledgers=ledgers,
        )
        report = _load_json(output_root / "extended-report.json", expected_type=dict)
        _verify_self_hash(report, "report_sha256", label="recoverable report")
        _require_equal("recoverable extended report", report, expected_report)
        analysis_status = "serialized_report_valid_audit_recoverable_without_rollout"
        audit_sha256 = None
    else:
        analysis_status = "not_analyzed"
        audit_sha256 = None
    _authenticate_prepared(project_root, training_root, output_root)
    return {
        "protocol": PROTOCOL,
        "status": "authenticated",
        "evaluation_manifest_sha256": manifest["manifest_sha256"],
        "seed_status": seed_status,
        "analysis_status": analysis_status,
        "audit_sha256": audit_sha256,
        "development_only": True,
        "final_86xxx_panel_opened": False,
        "sealed_622m_policy_namespace_opened": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate-seed", "analyze", "validate"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "evaluate-seed" and args.seed not in MODEL_SEEDS:
        raise EvaluationProtocolError("evaluate-seed requires --seed 0, 1, or 2")
    if args.command != "evaluate-seed" and args.seed is not None:
        raise EvaluationProtocolError("--seed is valid only for evaluate-seed")
    _configure_runtime()
    project_root, training_root, output_root = _resolve_roots(
        args.project_root, args.training_root, args.output_root
    )
    if args.command == "prepare":
        result: Any = {
            "status": "prepared",
            "manifest_path": str(
                _prepare(project_root, training_root, output_root).resolve()
            ),
        }
    elif args.command == "evaluate-seed":
        assert args.seed is not None
        result = _evaluate_seed(
            project_root, training_root, output_root, int(args.seed)
        )
    elif args.command == "analyze":
        result = _analyze(project_root, training_root, output_root)
    else:
        result = _validate(project_root, training_root, output_root)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
