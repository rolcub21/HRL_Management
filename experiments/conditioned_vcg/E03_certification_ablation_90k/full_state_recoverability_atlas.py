#!/usr/bin/env python3
"""Capture and render a full-state empirical recoverability atlas.

This is a diagnostic replay of E3, not a new performance experiment.  The
physical-only arm exposes every executable candidate and retains the exact
recoverability result as a shadow label.  We store the complete RecoveryState
for the current state and every candidate successor, then project the frozen
VCG graph representation for visualization.

The resulting picture is deliberately called an empirical atlas: PCA is a
lossy view of sampled states and is not an exhaustive viability kernel.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.conditioned_vcg.E03_certification_ablation_90k import run as e3
from experiments.conditioned_vcg.E13_operational_scalability_95k import (
    program as e13,
)
from PSLAP.viability import RecoveryState, ViabilityStatus
from PSLAP.viability_dataset import (
    canonical_sha256,
    recovery_state_from_dict,
    recovery_state_to_dict,
)
from PSLAP.yard_graph import (
    NODE_FEATURE_NAMES,
    encode_recovery_state,
    pad_yard_graphs,
)
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_nested_handling_pilot as pilot


SCHEMA_VERSION = 1
PROTOCOL = "vcg_full_state_empirical_recoverability_atlas_90k_v1"
LEDGER_PROTOCOL = "vcg_full_state_empirical_recoverability_atlas_ledger_90k_v1"
REPORT_PROTOCOL = "vcg_full_state_empirical_recoverability_atlas_report_90k_v1"
DEFAULT_OUTPUT = ROOT / "results/vcg-full-state-recoverability-atlas-90k"
DEFAULT_E3_OUTPUT = e3.DEFAULT_OUTPUT
CONTRACT_NAME = "atlas-contract.json"
FIGURE_STEM = "vcg-full-state-recoverability-atlas"
COMPANION_STEM = "vcg-full-state-recoverability-atlas-3d"

SAFE_COLOR = "#14866D"
UNSAFE_COLOR = "#CC3D4A"
UNKNOWN_COLOR = "#E2A72E"
CURRENT_EDGE = "#253B59"
SAFE_EDGE = "#98A5B3"


class AtlasError(RuntimeError):
    """Raised when a capture or rendering contract is inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Mapping[str, Any], hash_field: str | None = None) -> str:
    payload = dict(value)
    if hash_field is not None:
        payload.pop(hash_field, None)
    return canonical_sha256(payload)


def _with_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtlasError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise AtlasError(f"{label} must contain an object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    final90._atomic_json(path, value)


def _state_id(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(payload)


def _context(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    blocks = tuple(snapshot.get("blocks", ()))
    now = int(snapshot.get("time", 0))
    arrived = tuple(
        block for block in blocks if int(block.get("arrival_step", 0)) <= now
    )
    active = tuple(
        block
        for block in arrived
        if not bool(block.get("delivered", False))
    )
    stored = tuple(block for block in active if bool(block.get("stored", False)))
    remaining = tuple(
        float(block["storage_needed"]) - float(block["storage_elapsed"])
        for block in stored
        if block.get("storage_needed") is not None
        and block.get("storage_elapsed") is not None
    )
    return {
        "absolute_time": now,
        "total_workloads": len(blocks),
        "arrived_workloads": len(arrived),
        "active_workloads": len(active),
        "stored_workloads": len(stored),
        "delivered_workloads": sum(
            bool(block.get("delivered", False)) for block in blocks
        ),
        "carried_workloads": sum(
            bool(block.get("carrying", False)) for block in blocks
        ),
        "future_arrivals": len(blocks) - len(arrived),
        "minimum_remaining_time": min(remaining) if remaining else None,
        "mean_remaining_time": (
            float(sum(remaining) / len(remaining)) if remaining else None
        ),
    }


def _certificate_record(certificate) -> dict[str, Any]:
    witness = getattr(certificate, "witness", None)
    return {
        "status": certificate.status.value,
        "reason": str(certificate.reason),
        "exhaustive": bool(certificate.exhaustive),
        "witness_macro_count": None if witness is None else len(witness),
        "witness_primitive_steps": getattr(
            certificate, "witness_primitive_steps", None
        ),
    }


class AtlasCollector:
    """One replay ledger containing lossless states and frontier edges."""

    def __init__(self, *, model_seed: int, instance_seed: int) -> None:
        self.model_seed = int(model_seed)
        self.instance_seed = int(instance_seed)
        self.states: dict[str, dict[str, Any]] = {}
        self.state_labels: dict[str, str] = {}
        self.frontiers: list[dict[str, Any]] = []

    def store_state(self, state: RecoveryState, status: str) -> str:
        if not isinstance(state, RecoveryState):
            raise AtlasError("atlas state is not a RecoveryState")
        payload = recovery_state_to_dict(state)
        identifier = _state_id(payload)
        existing = self.states.setdefault(identifier, payload)
        if existing != payload:
            raise AtlasError("state digest collision")
        previous = self.state_labels.setdefault(identifier, str(status))
        if previous != str(status):
            raise AtlasError(
                f"identical state received conflicting labels: {previous}/{status}"
            )
        return identifier

    def record_frontier(self, env, physical) -> None:
        current_certificate = physical.current_certificate
        current_id = self.store_state(
            physical.recovery_state, current_certificate.status.value
        )
        candidates = []
        for candidate in physical.candidates:
            certificate = e3._shadow_certificate(candidate)
            successor_id = self.store_state(
                candidate.successor_state, certificate.status.value
            )
            recovery_action = getattr(candidate, "recovery_action", None)
            macro_steps = (
                None
                if recovery_action is None
                else int(recovery_action.steps)
            )
            if macro_steps is None and candidate.horizon_steps is not None:
                macro_steps = int(candidate.horizon_steps)
            candidates.append(
                {
                    "key": candidate.key,
                    "mode": candidate.mode.value,
                    "action_type": candidate.action_type.value,
                    "target_label": candidate.target_label,
                    "source": candidate.source,
                    "destination": candidate.destination,
                    "macro_steps": macro_steps,
                    "successor_state_id": successor_id,
                    "certificate": _certificate_record(certificate),
                }
            )
        snapshot = e3.behavior_view._snapshot(env)
        self.frontiers.append(
            {
                "frontier_index": len(self.frontiers),
                "decision_epoch": int(physical.decision_epoch),
                "current_state_id": current_id,
                "current_certificate": _certificate_record(current_certificate),
                "operational_context": _context(snapshot),
                "candidate_count": len(candidates),
                "candidates": candidates,
                "selected_key": None,
                "certified_counterfactual_key": None,
                "selection_diverges": None,
                "selected_execution_succeeded": None,
            }
        )

    def record_selection(self, decision_event: Mapping[str, Any]) -> None:
        if not self.frontiers:
            raise AtlasError("selection occurred before frontier capture")
        frontier = self.frontiers[-1]
        if frontier["decision_epoch"] != int(decision_event["decision_epoch"]):
            raise AtlasError("selection/frontier decision epoch mismatch")
        frontier["selected_key"] = decision_event["selected"]["key"]
        counterfactual = decision_event.get("certified_counterfactual")
        frontier["certified_counterfactual_key"] = (
            None if counterfactual is None else counterfactual["key"]
        )
        frontier["selection_diverges"] = bool(
            decision_event["selection_diverges"]
        )

    def record_execution(self, execution) -> None:
        if not self.frontiers:
            raise AtlasError("execution occurred before frontier capture")
        self.frontiers[-1]["selected_execution_succeeded"] = bool(
            execution.option_success
        )

    def payload(self, *, row: Mapping[str, Any], contract_sha256: str) -> dict:
        statuses = Counter(self.state_labels.values())
        candidate_statuses = Counter(
            candidate["certificate"]["status"]
            for frontier in self.frontiers
            for candidate in frontier["candidates"]
        )
        return _with_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": LEDGER_PROTOCOL,
                "status": "complete",
                "contract_sha256": contract_sha256,
                "model_seed": self.model_seed,
                "instance_seed": self.instance_seed,
                "training_runs": 0,
                "role": "diagnostic_replay_not_independent_performance_replication",
                "states": self.states,
                "state_status": self.state_labels,
                "frontiers": self.frontiers,
                "unique_state_count": len(self.states),
                "frontier_count": len(self.frontiers),
                "candidate_edge_count": sum(
                    len(frontier["candidates"]) for frontier in self.frontiers
                ),
                "state_status_counts": dict(statuses),
                "candidate_status_counts": dict(candidate_statuses),
                "replay_outcome": {
                    "strict_completion": bool(row["strict_completion"]),
                    "self_blocking": bool(row["self_blocking"]),
                    "failure_reason": row["method_failure_reason"],
                    "behavior_digest": row["behavior_digest"],
                },
            },
            "ledger_sha256",
        )


class CapturingTracker(e3.EpisodeTracker):
    def __init__(self, collector: AtlasCollector) -> None:
        super().__init__()
        self.collector = collector

    def record_frontier(self, env, physical) -> None:
        super().record_frontier(env, physical)
        self.collector.record_frontier(env, physical)

    def record_selection(self, snapshot, decision, within_temperatures) -> None:
        super().record_selection(snapshot, decision, within_temperatures)
        if self.pending_selection is None:
            raise AtlasError("E3 tracker did not retain its selection event")
        self.collector.record_selection(self.pending_selection)

    def record_execution(self, env, candidate, execution) -> None:
        self.collector.record_execution(execution)
        super().record_execution(env, candidate, execution)


def _contract(project_root: Path, output_dir: Path) -> dict[str, Any]:
    parent = _frozen_parents(project_root)
    e3_contract = parent["e3_contract"]
    e3_report = parent["e3_report"]
    conditioned = parent["conditioned"]
    return _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "prepared",
            "scientific_role": (
                "boundary_aware_projection_of_sampled_full_recovery_states"
            ),
            "not_claimed": [
                "exact_full_dimensional_viability_kernel",
                "independent_performance_replication",
                "future_arrival_robustness",
                "preference_dependent_safe_set",
            ],
            "candidate_source": e3.PHYSICAL,
            "shadow_labels": "exact_recoverability_results",
            "default_model_seed": 0,
            "default_instance_seed": 90016,
            "training_runs": 0,
            "projection": "deterministic_PCA_of_frozen_VCG_graph_embeddings",
            "graph_node_features": list(NODE_FEATURE_NAMES),
            "preference_lambda_excluded_from_state_embedding": True,
            "output_dir": str(output_dir.resolve()),
            "source": {
                "e3_contract_sha256": e3_contract["contract_sha256"],
                "e3_report_sha256": e3_report["report_sha256"],
                "conditioned_terminal_sha256": {
                    "0": conditioned["terminal_sha256"][0]
                },
                "script_sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "contract_sha256",
    )


def _self_hashed(path: Path, field: str, *, label: str) -> dict[str, Any]:
    value = _load_json(path, label=label)
    if value.get(field) != _digest(value, field):
        raise AtlasError(f"{label} self hash mismatch")
    return value


def _frozen_parents(project_root: Path) -> dict[str, Any]:
    """Authenticate saved identities without rebuilding superseded sources."""

    if project_root.resolve() != ROOT.resolve():
        raise AtlasError("unexpected project root")
    e3_contract = _self_hashed(
        DEFAULT_E3_OUTPUT / e3.CONTRACT_NAME,
        "contract_sha256",
        label="frozen E3 contract",
    )
    e3_report = _self_hashed(
        DEFAULT_E3_OUTPUT / e3.REPORT_NAME,
        "report_sha256",
        label="frozen E3 report",
    )
    e1_contract = _self_hashed(
        final90.DEFAULT_OUTPUT / final90.CONTRACT_NAME,
        "contract_sha256",
        label="frozen E1 contract",
    )
    e1_manifest = _self_hashed(
        final90.DEFAULT_OUTPUT / final90.INSTANCE_MANIFEST_NAME,
        "manifest_sha256",
        label="frozen E1 manifest",
    )
    if (
        e3_contract.get("protocol") != e3.PROTOCOL
        or e3_report.get("protocol") != e3.PROTOCOL
        or e3_report.get("contract_sha256") != e3_contract["contract_sha256"]
        or e1_contract.get("protocol") != final90.PROTOCOL
        or e1_manifest.get("protocol") != final90.PROTOCOL
        or e1_manifest.get("contract_sha256") != e1_contract["contract_sha256"]
        or e3_contract.get("e1_contract_sha256") != e1_contract["contract_sha256"]
        or e3_contract.get("e1_manifest_sha256") != e1_manifest["manifest_sha256"]
        or len(e1_manifest.get("instances", ())) != len(final90.INSTANCE_SEEDS)
    ):
        raise AtlasError("frozen E3/E1 parent binding changed")

    # E13 already established a byte-authenticated frozen-parent loader after
    # the historical training-source tree evolved. It resolves the same seed-0
    # base arm and conditioned terminal without reinterpreting old sources.
    conditioned = e13._parent_inputs()["conditioned"]
    observed_terminal = conditioned["terminal_sha256"][0]
    expected_terminal = e3_contract["conditioned_terminal_sha256"]["0"]
    if observed_terminal != expected_terminal:
        raise AtlasError("frozen conditioned checkpoint binding changed")
    return {
        "e3_contract": e3_contract,
        "e3_report": e3_report,
        "e1_contract": e1_contract,
        "e1_manifest": e1_manifest,
        "conditioned": conditioned,
    }


def prepare(project_root: Path, output_dir: Path) -> dict[str, Any]:
    expected = _contract(project_root, output_dir)
    path = output_dir / CONTRACT_NAME
    if path.is_file():
        observed = _load_json(path, label="atlas contract")
        if observed.get("contract_sha256") != _digest(
            observed, "contract_sha256"
        ):
            raise AtlasError("atlas contract self hash mismatch")
        if observed != expected:
            raise AtlasError("atlas contract, parent artifacts, or sources changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise AtlasError("nonempty atlas output has no contract")
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, expected)
    return {
        "status": "prepared",
        "contract": str(path.resolve()),
        "training_runs": 0,
    }


def authenticate_contract(project_root: Path, output_dir: Path) -> dict[str, Any]:
    observed = _load_json(output_dir / CONTRACT_NAME, label="atlas contract")
    if observed.get("contract_sha256") != _digest(observed, "contract_sha256"):
        raise AtlasError("atlas contract self hash mismatch")
    if observed != _contract(project_root, output_dir):
        raise AtlasError("atlas contract, parent artifacts, or sources changed")
    return observed


def _ledger_path(output_dir: Path, model_seed: int, instance_seed: int) -> Path:
    return (
        output_dir
        / "capture-ledger"
        / f"seed-{model_seed}"
        / f"instance-{instance_seed}.json"
    )


def capture(
    project_root: Path,
    output_dir: Path,
    *,
    model_seed: int,
    instance_seeds: Sequence[int],
) -> dict[str, Any]:
    contract = authenticate_contract(project_root, output_dir)
    if model_seed != 0:
        raise AtlasError("the frozen-parent atlas currently supports model seed 0")
    parent = _frozen_parents(project_root)
    manifest = parent["e1_manifest"]
    records = {int(record["seed"]): record for record in manifest["instances"]}
    unknown = sorted(set(map(int, instance_seeds)) - set(records))
    if unknown:
        raise AtlasError(f"unknown instance seeds: {unknown}")
    auth = {"conditioned": parent["conditioned"]}
    arm = parent["conditioned"]["inputs"]["arms"][model_seed]
    completed = []
    for position, instance_seed in enumerate(instance_seeds, start=1):
        instance_seed = int(instance_seed)
        path = _ledger_path(output_dir, model_seed, instance_seed)
        if path.is_file():
            ledger = _load_json(path, label="atlas capture ledger")
            if ledger.get("ledger_sha256") != _digest(ledger, "ledger_sha256"):
                raise AtlasError("atlas capture ledger self hash mismatch")
            if ledger.get("contract_sha256") != contract["contract_sha256"]:
                raise AtlasError("atlas ledger contract binding changed")
        else:
            collector = AtlasCollector(
                model_seed=model_seed, instance_seed=instance_seed
            )
            record = records[instance_seed]
            instance_path = final90.DEFAULT_OUTPUT / record["relative_path"]
            if _sha256(instance_path) != record["raw_sha256"]:
                raise AtlasError("frozen EpisodeInstance bytes changed")
            instance = final90._load_instance(final90.DEFAULT_OUTPUT, record)
            with patch.object(
                e3, "EpisodeTracker", lambda: CapturingTracker(collector)
            ):
                row = e3._run_physical_row(
                    project_root,
                    instance,
                    record,
                    model_seed=model_seed,
                    arm=arm,
                    auth=auth,
                    device=torch.device("cpu"),
                )
            ledger = collector.payload(
                row=row, contract_sha256=contract["contract_sha256"]
            )
            _atomic_json(path, ledger)
        completed.append(ledger)
        print(
            f"atlas {position}/{len(instance_seeds)} | seed={model_seed} "
            f"instance={instance_seed} | states={ledger['unique_state_count']} "
            f"edges={ledger['candidate_edge_count']} | "
            f"labels={ledger['candidate_status_counts']}",
            flush=True,
        )
    return {
        "status": "complete",
        "model_seed": model_seed,
        "instance_seeds": list(map(int, instance_seeds)),
        "capture_ledgers": len(completed),
        "training_runs": 0,
    }


def _load_ledgers(
    output_dir: Path,
    *,
    model_seed: int,
) -> list[dict[str, Any]]:
    directory = output_dir / "capture-ledger" / f"seed-{model_seed}"
    paths = sorted(directory.glob("instance-*.json")) if directory.is_dir() else []
    if not paths:
        raise AtlasError("no capture ledgers are available")
    ledgers = []
    for path in paths:
        ledger = _load_json(path, label="atlas capture ledger")
        if ledger.get("ledger_sha256") != _digest(ledger, "ledger_sha256"):
            raise AtlasError(f"capture ledger self hash mismatch: {path}")
        if ledger.get("protocol") != LEDGER_PROTOCOL:
            raise AtlasError("unexpected capture-ledger protocol")
        ledgers.append(ledger)
    return ledgers


def _frozen_encoder(project_root: Path, model_seed: int):
    if model_seed != 0:
        raise AtlasError("the frozen-parent atlas currently supports model seed 0")
    conditioned = _frozen_parents(project_root)["conditioned"]
    arm = conditioned["inputs"]["arms"][model_seed]
    base = pilot._fresh_base(arm, torch.device("cpu"))
    base.Q_local.requires_grad_(False).eval()
    return base.Q_local.graph_encoder, base.config.timing_scale, arm.checkpoint_sha256


def _embed_states(
    state_payloads: Sequence[Mapping[str, Any]],
    *,
    encoder,
    timing_scale: float,
    batch_size: int = 128,
) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for start in range(0, len(state_payloads), batch_size):
            states = tuple(
                recovery_state_from_dict(value)
                for value in state_payloads[start : start + batch_size]
            )
            graphs = tuple(
                encode_recovery_state(state, timing_scale=timing_scale)
                for state in states
            )
            batch = encoder(pad_yard_graphs(graphs)).detach().cpu().numpy()
            rows.append(batch)
    result = np.concatenate(rows, axis=0)
    if not np.isfinite(result).all():
        raise AtlasError("frozen graph encoder produced non-finite embeddings")
    return result


def deterministic_pca(
    values: np.ndarray, components: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise AtlasError("PCA requires at least two row vectors")
    count = min(int(components), values.shape[0] - 1, values.shape[1])
    if count < 1:
        raise AtlasError("PCA has no available component")
    centered = values - values.mean(axis=0, keepdims=True)
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:count].copy()
    # SVD signs are arbitrary. Anchor each component to its largest loading.
    for index in range(count):
        pivot = int(np.argmax(np.abs(axes[index])))
        if axes[index, pivot] < 0.0:
            axes[index] *= -1.0
    coordinates = centered @ axes.T
    denominator = float(np.square(singular).sum())
    explained = (
        np.zeros(count)
        if denominator == 0.0
        else np.square(singular[:count]) / denominator
    )
    return coordinates, axes, explained


def _state_descriptors(payload: Mapping[str, Any]) -> dict[str, float]:
    blocks = tuple(payload["blocks"])
    storage_count = max(len(payload["storage_cells"]), 1)
    remaining = np.asarray(
        [float(block["remaining_time"]) for block in blocks], dtype=float
    )
    return {
        "occupancy_fraction": len(blocks) / storage_count,
        "occupied_blocks": float(len(blocks)),
        "overdue_fraction": (
            float(np.mean(remaining <= 0.0)) if len(remaining) else 0.0
        ),
        "minimum_remaining_time": (
            float(np.min(remaining)) if len(remaining) else 0.0
        ),
        "mean_remaining_time": (
            float(np.mean(remaining)) if len(remaining) else 0.0
        ),
        "reserved_cells": float(len(payload["reserved_cells"])),
        "fixed_obstacles": float(len(payload["fixed_obstacles"])),
        "yard_cells": float(int(payload["rows"]) * int(payload["cols"])),
    }


def build_projection(
    project_root: Path,
    ledgers: Sequence[Mapping[str, Any]],
    *,
    model_seed: int,
) -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = {}
    labels: dict[str, str] = {}
    frontiers = []
    for ledger in ledgers:
        for identifier, payload in ledger["states"].items():
            if identifier != _state_id(payload):
                raise AtlasError("stored state identifier changed")
            previous = states.setdefault(identifier, payload)
            if previous != payload:
                raise AtlasError("state digest collision across ledgers")
        for identifier, status in ledger["state_status"].items():
            previous = labels.setdefault(identifier, status)
            if previous != status:
                raise AtlasError("state label changed across ledgers")
        frontiers.extend(ledger["frontiers"])
    identifiers = sorted(states)
    encoder, timing_scale, checkpoint_sha = _frozen_encoder(
        project_root, model_seed
    )
    embeddings = _embed_states(
        [states[identifier] for identifier in identifiers],
        encoder=encoder,
        timing_scale=timing_scale,
    )
    coordinates, axes, explained = deterministic_pca(embeddings, components=3)
    coordinate_map = {
        identifier: [float(item) for item in coordinates[index]]
        for index, identifier in enumerate(identifiers)
    }
    current_ids = {
        frontier["current_state_id"] for frontier in frontiers
    }
    selected_edges = set()
    unsafe_edges = set()
    edges = []
    for frontier_index, frontier in enumerate(frontiers):
        source = frontier["current_state_id"]
        for candidate in frontier["candidates"]:
            target = candidate["successor_state_id"]
            status = candidate["certificate"]["status"]
            edge = {
                "frontier_index": frontier_index,
                "source_state_id": source,
                "target_state_id": target,
                "candidate_key": candidate["key"],
                "action_type": candidate["action_type"],
                "status": status,
                "selected": candidate["key"] == frontier["selected_key"],
            }
            edges.append(edge)
            pair = (source, target)
            if edge["selected"]:
                selected_edges.add(pair)
            if status == ViabilityStatus.UNSAFE.value:
                unsafe_edges.add(pair)
    points = []
    for identifier in identifiers:
        points.append(
            {
                "state_id": identifier,
                "status": labels[identifier],
                "is_encountered_current_state": identifier in current_ids,
                "coordinates": coordinate_map[identifier],
                **_state_descriptors(states[identifier]),
            }
        )
    return {
        "points": points,
        "edges": edges,
        "coordinates": coordinate_map,
        "unique_state_count": len(points),
        "edge_count": len(edges),
        "state_status_counts": dict(Counter(labels.values())),
        "candidate_status_counts": dict(
            Counter(edge["status"] for edge in edges)
        ),
        "selected_edge_count": len(selected_edges),
        "unsafe_edge_count": len(unsafe_edges),
        "embedding": {
            "model_seed": model_seed,
            "graph_encoder_base_checkpoint_sha256": checkpoint_sha,
            "dimension": int(embeddings.shape[1]),
            "timing_scale": float(timing_scale),
            "node_features": list(NODE_FEATURE_NAMES),
            "projection": "centered_deterministic_PCA",
            "explained_variance_fraction": [
                float(item) for item in explained
            ],
            "component_loadings_sha256": canonical_sha256(
                [[float(item) for item in row] for row in axes]
            ),
        },
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _draw_edges(ax, projection: Mapping[str, Any], *, dimensions=(0, 1)) -> None:
    coordinates = projection["coordinates"]
    for edge in projection["edges"]:
        source = coordinates[edge["source_state_id"]]
        target = coordinates[edge["target_state_id"]]
        status = edge["status"]
        if status == ViabilityStatus.UNSAFE.value:
            color, alpha, width, zorder = UNSAFE_COLOR, 0.86, 1.35, 3
        elif edge["selected"]:
            color, alpha, width, zorder = CURRENT_EDGE, 0.48, 0.85, 2
        else:
            color, alpha, width, zorder = SAFE_EDGE, 0.075, 0.35, 1
        ax.plot(
            [source[dimensions[0]], target[dimensions[0]]],
            [source[dimensions[1]], target[dimensions[1]]],
            color=color,
            alpha=alpha,
            linewidth=width,
            zorder=zorder,
        )


def _status_scatter(ax, points: Sequence[Mapping[str, Any]]) -> None:
    styles = {
        ViabilityStatus.SAFE.value: ("o", SAFE_COLOR, 20),
        ViabilityStatus.UNSAFE.value: ("X", UNSAFE_COLOR, 86),
        ViabilityStatus.UNKNOWN.value: ("^", UNKNOWN_COLOR, 60),
    }
    for status, (marker, color, size) in styles.items():
        selected = [point for point in points if point["status"] == status]
        if not selected:
            continue
        ax.scatter(
            [point["coordinates"][0] for point in selected],
            [point["coordinates"][1] for point in selected],
            s=size,
            marker=marker,
            color=color,
            edgecolor="white" if marker == "o" else "#7E2630",
            linewidth=0.35 if marker == "o" else 0.75,
            alpha=0.83 if marker == "o" else 1.0,
            zorder=4,
            label=status,
        )
    current = [point for point in points if point["is_encountered_current_state"]]
    ax.scatter(
        [point["coordinates"][0] for point in current],
        [point["coordinates"][1] for point in current],
        s=52,
        marker="o",
        facecolors="none",
        edgecolors=CURRENT_EDGE,
        linewidth=1.0,
        zorder=5,
        label="Encountered decision state",
    )


def render_main(projection: Mapping[str, Any], output_dir: Path) -> list[str]:
    _style()
    points = projection["points"]
    explained = projection["embedding"]["explained_variance_fraction"]
    fig, axes = plt.subplots(1, 2, figsize=(13.7, 5.55))

    _draw_edges(axes[0], projection)
    _status_scatter(axes[0], points)
    axes[0].set_title("(a) Sampled recoverability region", loc="left")
    axes[0].set_xlabel(f"State component 1 ({100 * explained[0]:.1f}% variance)")
    axes[0].set_ylabel(f"State component 2 ({100 * explained[1]:.1f}% variance)")
    axes[0].grid(color="#D5DBE1", linewidth=0.65, alpha=0.55)
    axes[0].legend(loc="best", frameon=False)

    safe = [point for point in points if point["status"] == ViabilityStatus.SAFE.value]
    pressure = np.asarray(
        [-point["minimum_remaining_time"] for point in safe], dtype=float
    )
    sizes = np.asarray(
        [18.0 + 95.0 * point["occupancy_fraction"] ** 2 for point in safe]
    )
    scatter = axes[1].scatter(
        [point["coordinates"][0] for point in safe],
        [point["coordinates"][1] for point in safe],
        c=pressure,
        s=sizes,
        cmap="viridis",
        alpha=0.77,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    unsafe = [point for point in points if point["status"] == ViabilityStatus.UNSAFE.value]
    if unsafe:
        axes[1].scatter(
            [point["coordinates"][0] for point in unsafe],
            [point["coordinates"][1] for point in unsafe],
            s=100,
            marker="X",
            color=UNSAFE_COLOR,
            edgecolor="#7E2630",
            linewidth=0.8,
            zorder=5,
            label="UNSAFE",
        )
        axes[1].legend(loc="best", frameon=False)
    axes[1].set_title("(b) Timing pressure and storage load", loc="left")
    axes[1].set_xlabel(f"State component 1 ({100 * explained[0]:.1f}% variance)")
    axes[1].set_ylabel(f"State component 2 ({100 * explained[1]:.1f}% variance)")
    axes[1].grid(color="#D5DBE1", linewidth=0.65, alpha=0.55)
    colorbar = fig.colorbar(scatter, ax=axes[1], fraction=0.045, pad=0.025)
    colorbar.set_label(
        "Urgency pressure = -minimum remaining time\n(>0 means overdue)"
    )
    size_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="None", color="#607D8B",
            markerfacecolor="#607D8B", alpha=0.7,
            markersize=math.sqrt(18 + 95 * fraction**2) / 1.45,
            label=f"{int(100 * fraction)}% occupied",
        )
        for fraction in (0.4, 0.7, 1.0)
    ]
    axes[1].legend(
        handles=(
            *size_handles,
            Line2D(
                [0], [0], marker="X", linestyle="None", color=UNSAFE_COLOR,
                markersize=8, label="UNSAFE",
            ),
        ),
        loc="best",
        frameon=False,
    )

    fig.suptitle(
        "Empirical full recovery-state atlas",
        fontsize=15.5,
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        (
            f"{projection['unique_state_count']:,} unique current/successor states; "
            f"{projection['edge_count']:,} physically feasible candidate transitions. "
            "Each point encodes every cell, block occupancy and remaining time, "
            "transporter position, obstacle, reservation, entrance and exit. "
            "Closed admission; future arrivals and lambda are excluded."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#55616E",
    )
    fig.tight_layout(rect=(0.025, 0.07, 0.985, 0.94), w_pad=2.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"{FIGURE_STEM}.{suffix}"
        fig.savefig(path, dpi=280, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def render_3d(projection: Mapping[str, Any], output_dir: Path) -> list[str]:
    _style()
    points = projection["points"]
    explained = projection["embedding"]["explained_variance_fraction"]
    if len(explained) < 3:
        raise AtlasError("3D rendering requires three PCA components")
    fig = plt.figure(figsize=(8.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    styles = {
        ViabilityStatus.SAFE.value: ("o", SAFE_COLOR, 17),
        ViabilityStatus.UNSAFE.value: ("X", UNSAFE_COLOR, 88),
        ViabilityStatus.UNKNOWN.value: ("^", UNKNOWN_COLOR, 58),
    }
    for status, (marker, color, size) in styles.items():
        rows = [point for point in points if point["status"] == status]
        if not rows:
            continue
        ax.scatter(
            [row["coordinates"][0] for row in rows],
            [row["coordinates"][1] for row in rows],
            [row["coordinates"][2] for row in rows],
            marker=marker,
            color=color,
            s=size,
            alpha=0.78 if status == ViabilityStatus.SAFE.value else 1.0,
            depthshade=True,
            label=status,
        )
    ax.set_xlabel(f"Component 1 ({100 * explained[0]:.1f}%)", labelpad=8)
    ax.set_ylabel(f"Component 2 ({100 * explained[1]:.1f}%)", labelpad=8)
    ax.set_zlabel(f"Component 3 ({100 * explained[2]:.1f}%)", labelpad=8)
    ax.set_title("Full recovery-state atlas — 3D companion", pad=18)
    ax.legend(frameon=False, loc="upper right")
    ax.view_init(elev=25, azim=-56)
    ax.set_box_aspect((1.2, 1.0, 0.82))
    fig.text(
        0.5,
        0.02,
        "Diagnostic PCA of sampled frozen-controller graph states; not an exhaustive kernel.",
        ha="center",
        fontsize=8.5,
        color="#55616E",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"{COMPANION_STEM}.{suffix}"
        fig.savefig(path, dpi=280, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def render(
    project_root: Path,
    output_dir: Path,
    *,
    model_seed: int,
) -> dict[str, Any]:
    contract = authenticate_contract(project_root, output_dir)
    ledgers = _load_ledgers(output_dir, model_seed=model_seed)
    if any(
        ledger["contract_sha256"] != contract["contract_sha256"]
        for ledger in ledgers
    ):
        raise AtlasError("capture ledger contract binding changed")
    projection = build_projection(
        project_root, ledgers, model_seed=model_seed
    )
    main_paths = render_main(projection, output_dir)
    companion_paths = render_3d(projection, output_dir)
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": REPORT_PROTOCOL,
            "status": "complete",
            "contract_sha256": contract["contract_sha256"],
            "training_runs": 0,
            "capture_ledgers": len(ledgers),
            "instance_seeds": sorted(
                int(ledger["instance_seed"]) for ledger in ledgers
            ),
            **projection,
            "claim_scope": (
                "empirical_projection_of_sampled_closed_admission_full_"
                "recovery_states_and_physically_feasible_successors"
            ),
            "not_claimed": contract["not_claimed"],
            "figure_pdf": main_paths[0],
            "figure_png": main_paths[1],
            "figure_3d_pdf": companion_paths[0],
            "figure_3d_png": companion_paths[1],
            "caption": (
                "Empirical full-state recoverability atlas. Points are complete "
                "RecoveryStates encoded by the frozen VCG relational encoder and "
                "projected by deterministic PCA. Thin segments connect encountered "
                "decision states to physically feasible candidate successors; exact "
                "shadow certificates supply SAFE, UNSAFE, and UNKNOWN labels. The "
                "projection is sampled diagnostic evidence, not an exhaustive kernel."
            ),
        },
        "report_sha256",
    )
    report_path = output_dir / "atlas-report.json"
    # Keep the report compact: raw coordinates and edges already live in ledgers.
    _atomic_json(report_path, report)
    return {
        "status": "complete",
        "report": str(report_path.resolve()),
        "figure_pdf": main_paths[0],
        "figure_3d_pdf": companion_paths[0],
        "unique_states": projection["unique_state_count"],
        "candidate_edges": projection["edge_count"],
        "state_status_counts": projection["state_status_counts"],
        "training_runs": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "capture", "render", "run-pilot", "run-seed0")
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--instance-seed", type=int, action="append")
    return parser


def main() -> None:
    args = _parser().parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(project_root, output_dir)
    elif args.command == "capture":
        prepare(project_root, output_dir)
        seeds = args.instance_seed or [90016]
        result = capture(
            project_root,
            output_dir,
            model_seed=args.model_seed,
            instance_seeds=seeds,
        )
    elif args.command == "render":
        result = render(
            project_root, output_dir, model_seed=args.model_seed
        )
    elif args.command == "run-pilot":
        prepare(project_root, output_dir)
        capture(
            project_root,
            output_dir,
            model_seed=0,
            instance_seeds=(90016,),
        )
        result = render(project_root, output_dir, model_seed=0)
    else:
        prepare(project_root, output_dir)
        capture(
            project_root,
            output_dir,
            model_seed=0,
            instance_seeds=final90.INSTANCE_SEEDS,
        )
        result = render(project_root, output_dir, model_seed=0)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
