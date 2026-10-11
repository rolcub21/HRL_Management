#!/usr/bin/env python3
"""Prospective two-stage checkpoint selector for VCG-Dense V1.2.

This runner is deliberately a *development selection* protocol, not a final
test.  It authenticates the twenty immutable snapshots produced for each of
fresh model seeds 3/4/5, evaluates every snapshot on panel A before panel B is
created, and applies one frozen A-ranked/B-gated rule:

* the same-trajectory V1.1 lexicographic checkpoint is the reference;
* panel-A candidates must be strict/full and have paired mean MAE no more
  than 2.0 steps above their seed's reference;
* candidates are ranked by fewer physical storage relocations/100 deliveries,
  then greater dense return, lower MAE, and earlier episode;
* panel B accepts the first A-ranked candidate that is strict/full, remains
  within +2.0 MAE, and has strictly fewer physical storage relocations than
  the reference;
* the authenticated V1.1 reference is the mandatory final fallback.

All twenty snapshots are evaluated on both panels so the audit ledger is
complete.  Later panel-B outcomes never reorder the frozen panel-A ranking.
Transactional per-run ledgers make interruption/resume exact.  This module
does not instantiate an environment or open either panel merely by import.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Callable, Mapping, Optional, Sequence
import uuid

import torch

import run_vcg_objective_gamma_audit as objective_audit
import train_vcg_dense_proper as base_training
from benchmark_viability_critic_priority import (
    EXACT_FULL,
    _liveness_rule,
    _load_controller_checkpoint,
    _make_env,
    _search_config,
    _sha256_file,
    run_arm,
)
from compare_vcg_dense_pareto import (
    RUN_FIELDS,
    _compact_deployment_payload,
    _digest_json,
    _policy_digest,
    _shared_execution_contract,
)
from compare_viability_graph_baselines import (
    _csv_value,
    _json_safe,
    _normalize_vcg,
)
from contention_metrics import (
    CANONICAL_CONTENTION_FIELDS,
    CONTENTION_METRIC_SCHEMA_VERSION,
    validate_contention_metric_record,
)
from example.episode_instance import EpisodeInstance
from track_b_urgency_evaluate import resolve_device
from train_vcg_dense_proper import (
    FROZEN_GAMMA,
    FROZEN_MAX_STEPS,
    FROZEN_OBJECTIVE_SPEC,
    FROZEN_TOTAL_EPISODES,
    FROZEN_VALIDATION_SEEDS,
    OPENED_DEVELOPMENT_SEEDS,
    PROTECTED_SEEDS,
)
from train_vcg_dense_v1_2_prospective import (
    BASE_STATE_DIRECTORY,
    CONTRACT_FILENAME as PROSPECTIVE_CONTRACT_FILENAME,
    FRESH_MODEL_SEEDS,
    PROSPECTIVE_METHOD_VERSION,
    PROSPECTIVE_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    PROSPECTIVE_TRAINING_PROTOCOL,
    SNAPSHOT_ROLE,
    SNAPSHOT_EPISODES,
    _prospective_contract,
)


PROTOCOL = "vcg_dense_v1_2_guarded_prospective_selection_development_v2"
PROTOCOL_SCHEMA_VERSION = 2
SELECTION_RULE = (
    "v1_1_reference_then_panel_A_strict_full_mae_plus_2_physical_storage_"
    "relocation_rank_then_panel_B_strict_full_mae_plus_2_positive_physical_"
    "storage_relocation_saving_"
    "then_mandatory_reference_fallback_v1"
)
MODEL_SEEDS = tuple(FRESH_MODEL_SEEDS)
PANEL_A_SEEDS = tuple(range(83_000, 83_030))
PANEL_B_SEEDS = tuple(range(83_030, 83_060))
PANELS = {"A": PANEL_A_SEEDS, "B": PANEL_B_SEEDS}
MAE_MARGIN = 2.0
MAX_STEPS = FROZEN_MAX_STEPS
EXPECTED_DELIVERIES_PER_EPISODE = 8
POSITIVE_TOLERANCE = 1e-12

# These ranges were exhaustively searched as literal numeric tokens in source,
# result JSON/CSV/log/text artifacts, and paths before this protocol was frozen.
# Hash substrings were explicitly excluded.  Runtime guards below additionally
# reject overlap with every seed namespace declared by the implementation.
NAMESPACE_AUDIT = {
    "audit_date": "2026-08-08",
    "literal_numeric_range": (83_000, 83_059),
    "repo_and_results_literal_matches": 0,
    "apparent_matches_excluded": "hexadecimal_digest_substrings_only",
    "csv_seed_or_instance_field_matches": 0,
    "checkpoint_metadata_scan": (
        "all_121_data_pickles_at_most_10MB_plus_first_and_last_1MB_of_"
        "32_large_replay_pickles_124187349_bytes_scanned_zero_BININT_matches"
    ),
    "checkpoint_metadata_caveat": (
        "interiors_of_32_large_replay_heavy_data_pickles_not_fully_opcode_scanned"
    ),
    "status": "unused_and_reserved_for_v1_2_development_selection",
}

SNAPSHOT_MANIFEST_FILENAME = "prospective-snapshot-manifest.json"
PROTOCOL_MANIFEST_FILENAME = "selection-protocol-manifest.json"
PANEL_INSTANCE_MANIFEST = {
    "A": "panel-a-instance-manifest.json",
    "B": "panel-b-instance-manifest.json",
}
PANEL_RESULTS_FILENAME = {"A": "panel-a-runs.csv", "B": "panel-b-runs.csv"}
PANEL_A_RANKING_FILENAME = "panel-a-frozen-ranking.json"
REPORT_FILENAME = "prospective-selection-report.json"
AUDIT_FILENAME = "prospective-selection-audit.json"

# ``compare_vcg_dense_pareto.RUN_FIELDS`` intentionally remains frozen for
# the already-opened 80000--80029 V1 development artifact.  Prospective V1.2
# owns this additive schema before either 83000--83059 panel is opened.
PROSPECTIVE_RUN_FIELDS = RUN_FIELDS + tuple(
    field for field in CANONICAL_CONTENTION_FIELDS if field not in RUN_FIELDS
) + ("physical_storage_relocations_per_100_deliveries",)


@dataclass(frozen=True)
class SnapshotArm:
    model_seed: int
    episode: int
    method_id: str
    path: Path
    sha256: str
    policy_digest: str
    payload: dict
    validation_record: dict


@dataclass(frozen=True)
class SnapshotBundle:
    training_dir: Path
    model_seed: int
    manifest_path: Path
    manifest_sha256: str
    manifest: dict
    snapshots: tuple[SnapshotArm, ...]
    reference_episode: int
    reference_sha256: str
    shared_execution_contract: dict


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value) -> None:
    _atomic_text(
        path,
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _require_equal(name: str, observed, expected) -> None:
    if _json_safe(observed) != _json_safe(expected):
        raise ValueError(
            f"prospective V1.2 {name} mismatch: "
            f"observed={observed!r}, expected={expected!r}"
        )


def _is_sha256(value) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validation_record_identity(record: Mapping) -> dict:
    if not isinstance(record, Mapping):
        raise ValueError("snapshot validation record must be a mapping")
    summary = record.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("snapshot validation record has no summary")
    return {
        "checkpoint_episode": int(record["checkpoint_episode"]),
        "selection_score": tuple(float(value) for value in record["selection_score"]),
        "selection_score_fields": tuple(str(value) for value in record["selection_score_fields"]),
        "deployment_eligible": bool(record["deployment_eligible"]),
        "strict_method_success_rate": float(summary["strict_method_success_rate"]),
        "completion_rate": float(summary["completion_rate"]),
        "mean_dense_rescored_return": float(summary["mean_dense_rescored_return"]),
        "mean_absolute_error": float(summary["mean_absolute_error"]),
        "relocations_per_100_deliveries": float(summary["relocations_per_100_deliveries"]),
    }


def _safe_snapshot_path(training_dir: Path, episode: int, relative_path: str) -> Path:
    expected_relative = Path("snapshots") / f"episode-{episode:04d}.pth"
    if Path(relative_path) != expected_relative:
        raise ValueError(
            f"snapshot {episode} path is not canonical: {relative_path!r}"
        )
    path = (training_dir / relative_path).resolve()
    snapshots_root = (training_dir / "snapshots").resolve()
    if path.parent != snapshots_root:
        raise ValueError("snapshot path escapes the immutable snapshots directory")
    return path


def _snapshot_entries(manifest: Mapping) -> tuple[Mapping, ...]:
    entries = manifest.get("snapshots", manifest.get("entries"))
    if not isinstance(entries, (list, tuple)):
        raise ValueError("prospective snapshot manifest has no snapshot entries")
    return tuple(entries)


def _require_exact_prospective_contract(
    training_dir: Path, model_seed: int, observed: Mapping
) -> dict:
    """Regenerate and compare the entire frozen prospective/base recipe."""

    runtime = SimpleNamespace(
        output_dir=training_dir,
        model_seed=int(model_seed),
        device="cpu",
        resume_existing=True,
        stop_after_episode=None,
        log_every=5,
    )
    expected, expected_base, _ = _prospective_contract(runtime)
    _require_equal("full prospective/base training contract", observed, expected)
    return expected_base


def _authenticate_snapshot_bundle(training_dir: Path) -> SnapshotBundle:
    """Authenticate one complete prospective trainer output fail-closed."""

    training_dir = training_dir.resolve()
    manifest_path = training_dir / SNAPSHOT_MANIFEST_FILENAME
    manifest = _load_json(manifest_path)
    contract_document = _load_json(training_dir / PROSPECTIVE_CONTRACT_FILENAME)
    prospective_contract = contract_document.get("contract")
    if not isinstance(prospective_contract, Mapping):
        raise ValueError("prospective training contract document has no contract")
    prospective_contract_sha = objective_audit._contract_hash(dict(prospective_contract))
    model_seed = int(manifest.get("model_seed", -1))
    if model_seed not in MODEL_SEEDS:
        raise ValueError(f"unexpected prospective model seed: {model_seed}")
    expected_base_contract = _require_exact_prospective_contract(
        training_dir, model_seed, prospective_contract
    )

    expected_manifest = {
        "manifest_schema_version": PROSPECTIVE_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "prospective_training_protocol": PROSPECTIVE_TRAINING_PROTOCOL,
        "prospective_method_version": PROSPECTIVE_METHOD_VERSION,
        "model_seed": model_seed,
    }
    for name in expected_manifest:
        _require_equal(f"snapshot manifest {name}", manifest.get(name), expected_manifest[name])
    for name, expected in {
        "prospective_training_protocol": PROSPECTIVE_TRAINING_PROTOCOL,
        "prospective_method_version": PROSPECTIVE_METHOD_VERSION,
    }.items():
        _require_equal(f"prospective contract document {name}", contract_document.get(name), expected)
        _require_equal(f"prospective contract {name}", prospective_contract.get(name), expected)
    _require_equal(
        "prospective contract document SHA-256",
        contract_document.get("prospective_training_contract_sha256"),
        prospective_contract_sha,
    )
    _require_equal("prospective contract model seed", prospective_contract.get("model_seed"), model_seed)
    _require_equal(
        "manifest prospective contract SHA-256",
        manifest.get("prospective_training_contract_sha256"),
        prospective_contract_sha,
    )
    _require_equal(
        "manifest base resume contract SHA-256",
        manifest.get("base_resume_contract_sha256"),
        prospective_contract.get("base_resume_contract_sha256"),
    )
    base_resume_contract = prospective_contract.get("base_resume_contract")
    if not isinstance(base_resume_contract, Mapping):
        raise ValueError("prospective contract has no base resume contract")
    _require_equal(
        "computed base resume contract SHA-256",
        prospective_contract.get("base_resume_contract_sha256"),
        objective_audit._contract_hash(dict(base_resume_contract)),
    )
    if manifest.get("candidate_pool_complete") is not True:
        raise ValueError("prospective snapshot manifest is not complete")
    _require_equal("manifest completed training episodes", manifest.get("completed_training_episodes"), FROZEN_TOTAL_EPISODES)
    for name in (
        "candidate_selection_performed",
        "deployment_checkpoint_eligible",
        "performance_claim_authorized",
        "future_a_b_selection_panels_opened",
        "sealed_test_panels_opened",
    ):
        _require_equal(f"selection-free manifest {name}", manifest.get(name), False)
    _require_equal("selection-free fresh-A reference episode", manifest.get("fresh_a_reference_checkpoint_episode"), None)
    _require_equal("selection-free fresh-A reference SHA-256", manifest.get("fresh_a_reference_checkpoint_sha256"), None)
    declared_reference_episode = int(
        manifest.get("v1_1_reference_checkpoint_episode", -1)
    )
    declared_reference_sha = manifest.get("v1_1_reference_checkpoint_sha256")
    if declared_reference_episode not in SNAPSHOT_EPISODES or not _is_sha256(
        declared_reference_sha
    ):
        raise ValueError("manifest has no authenticated frozen V1.1 reference")
    if not _is_sha256(manifest.get("v1_1_best_checkpoint_sha256")):
        raise ValueError("manifest has no authenticated underlying V1.1 best artifact")
    _require_equal(
        "snapshot episode schedule",
        tuple(int(value) for value in manifest.get("snapshot_episodes", ())),
        tuple(SNAPSHOT_EPISODES),
    )

    entries = _snapshot_entries(manifest)
    if len(entries) != len(SNAPSHOT_EPISODES):
        raise ValueError("prospective manifest does not contain all twenty snapshots")
    by_episode: dict[int, Mapping] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("snapshot manifest entry must be a mapping")
        episode = int(entry.get("episode", -1))
        if episode in by_episode:
            raise ValueError(f"duplicate snapshot manifest episode: {episode}")
        by_episode[episode] = entry
    _require_equal("snapshot entry episode set", tuple(sorted(by_episode)), tuple(SNAPSHOT_EPISODES))

    arms = []
    shared = None
    for episode in SNAPSHOT_EPISODES:
        entry = by_episode[episode]
        relative_path = entry.get("relative_path")
        if not isinstance(relative_path, str):
            raise ValueError(f"snapshot {episode} has no relative_path")
        path = _safe_snapshot_path(training_dir, episode, relative_path)
        sha256 = _sha256_file(path)
        if not _is_sha256(entry.get("sha256")):
            raise ValueError(f"snapshot {episode} manifest SHA-256 is malformed")
        _require_equal(f"snapshot {episode} file SHA-256", sha256, entry["sha256"])
        payload = _load_controller_checkpoint(path)

        expected_payload = {
            "training_protocol": base_training.TRAINING_PROTOCOL,
            "method_version": base_training.METHOD_VERSION,
            "checkpoint_role": SNAPSHOT_ROLE,
            "trainer_resumable": False,
            "deployment_checkpoint_eligible": False,
            "selection_finalized_after_total_episodes": False,
            "prospective_training_protocol": PROSPECTIVE_TRAINING_PROTOCOL,
            "prospective_method_version": PROSPECTIVE_METHOD_VERSION,
            "model_seed": model_seed,
            "completed_training_episodes": episode,
            "timing_objective": "dense_piecewise_v1",
            "timing_objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
            "gamma": FROZEN_GAMMA,
            "performance_claim_authorized": False,
            "candidate_selection_performed": False,
            "future_a_b_selection_panels_opened": False,
            "sealed_test_panels_opened": False,
        }
        for name, expected in expected_payload.items():
            _require_equal(f"snapshot {episode} payload {name}", payload.get(name), expected)
        _require_equal(
            f"snapshot {episode} network config",
            payload.get("config"),
            expected_base_contract.get("graph_config"),
        )
        _require_equal(
            f"snapshot {episode} prospective contract SHA-256",
            payload.get("prospective_training_contract_sha256"),
            prospective_contract_sha,
        )
        _require_equal(
            f"snapshot {episode} base resume contract",
            payload.get("resume_contract"),
            prospective_contract.get("base_resume_contract"),
        )
        _require_equal(
            f"snapshot {episode} base resume contract SHA-256",
            payload.get("base_resume_contract_sha256"),
            prospective_contract.get("base_resume_contract_sha256"),
        )
        source_latest_sha = entry.get("source_latest_sha256")
        if not _is_sha256(source_latest_sha):
            raise ValueError(f"snapshot {episode} source latest SHA-256 is malformed")
        _require_equal(
            f"snapshot {episode} source latest provenance",
            payload.get("source_latest_checkpoint_sha256"),
            source_latest_sha,
        )
        _require_equal(
            f"snapshot {episode} source latest relative path",
            entry.get("source_latest_relative_path"),
            f"{BASE_STATE_DIRECTORY}/latest.pth",
        )
        _require_equal(
            f"snapshot {episode} entry base resume contract SHA-256",
            entry.get("base_resume_contract_sha256"),
            prospective_contract.get("base_resume_contract_sha256"),
        )

        state = payload.get("agent_state")
        if not isinstance(state, Mapping) or not isinstance(state.get("Q_local"), Mapping):
            raise ValueError(f"snapshot {episode} has no Q_local deployment state")
        if set(state) != {"Q_local", "Q_target"}:
            raise ValueError(
                f"snapshot {episode} is not model-only: agent_state keys "
                f"{sorted(state)}"
            )

        record = payload.get("snapshot_validation_record")
        record_identity = _validation_record_identity(record)
        _require_equal(
            f"snapshot {episode} validation episode",
            record_identity["checkpoint_episode"],
            episode,
        )
        _require_equal(
            f"snapshot {episode} validation record SHA-256",
            entry.get("validation_record_sha256"),
            objective_audit._contract_hash(dict(record)),
        )
        _require_equal(
            f"snapshot {episode} validation selection score",
            tuple(entry.get("validation_selection_score", ())),
            record_identity["selection_score"],
        )
        _require_equal(
            f"snapshot {episode} validation selection fields",
            tuple(entry.get("validation_selection_score_fields", ())),
            record_identity["selection_score_fields"],
        )
        current_shared = _shared_execution_contract(payload)
        if shared is None:
            shared = current_shared
        else:
            _require_equal(f"snapshot {episode} shared execution contract", current_shared, shared)

        arms.append(
            SnapshotArm(
                model_seed=model_seed,
                episode=episode,
                method_id=f"vcg_dense_v1_2_seed{model_seed}_ep{episode:04d}",
                path=path,
                sha256=sha256,
                policy_digest=_policy_digest(payload),
                payload=_compact_deployment_payload(payload),
                validation_record=dict(record),
            )
        )

    # The trainer remains selection-free.  Reconstruct the frozen V1.1
    # lexicographic selector from the authenticated per-snapshot validation
    # records; panel A and panel B play no role in establishing the reference.
    reference_arm = max(
        arms,
        key=lambda arm: tuple(
            float(value) for value in arm.validation_record["selection_score"]
        ),
    )
    reference_episode = reference_arm.episode
    reference_sha = reference_arm.sha256
    if reference_arm.validation_record.get("deployment_eligible") is not True:
        raise ValueError(
            "frozen V1.1 reference was not strict/full on its authenticated "
            "legacy validation panel"
        )
    _require_equal(
        "manifest frozen V1.1 reference episode",
        declared_reference_episode,
        reference_episode,
    )
    _require_equal(
        "manifest frozen V1.1 reference snapshot SHA-256",
        declared_reference_sha,
        reference_sha,
    )
    base_best_path = training_dir / BASE_STATE_DIRECTORY / "best.pth"
    base_best_sha = _sha256_file(base_best_path)
    _require_equal(
        "manifest frozen V1.1 finalized-best SHA-256",
        manifest.get("v1_1_best_checkpoint_sha256"),
        base_best_sha,
    )
    base_best = _load_controller_checkpoint(base_best_path)
    for name, expected in {
        "checkpoint_role": "best_deployment_finalized",
        "trainer_resumable": False,
        "completed_training_episodes": FROZEN_TOTAL_EPISODES,
        "protocol_training_complete": True,
        "selection_finalized_after_total_episodes": True,
        "deployment_checkpoint_eligible": True,
    }.items():
        _require_equal(f"frozen V1.1 finalized-best {name}", base_best.get(name), expected)
    _require_equal(
        "frozen V1.1 finalized-best selected episode",
        base_best.get("selected_checkpoint_episode"),
        reference_episode,
    )
    _require_equal(
        "frozen V1.1 reference deployment policy",
        _policy_digest(base_best),
        reference_arm.policy_digest,
    )
    _require_equal(
        "frozen V1.1 reference validation identity",
        _validation_record_identity(base_best.get("best_validation_record")),
        _validation_record_identity(reference_arm.validation_record),
    )
    final_episode = max(SNAPSHOT_EPISODES)
    final_arm = next(arm for arm in arms if arm.episode == final_episode)
    base_latest_path = training_dir / BASE_STATE_DIRECTORY / "latest.pth"
    final_entry = by_episode[final_episode]
    _require_equal(
        "episode-500 source latest SHA-256",
        _sha256_file(base_latest_path),
        final_entry.get("source_latest_sha256"),
    )
    base_latest = _load_controller_checkpoint(base_latest_path)
    raw_base_latest = torch.load(
        base_latest_path, map_location="cpu", weights_only=False
    )
    if not isinstance(raw_base_latest, Mapping):
        raise ValueError("prospective base latest checkpoint is not a mapping")
    base_training._validate_resume(raw_base_latest, expected_base_contract)
    _require_equal(
        "prospective base latest completed episode",
        raw_base_latest.get("completed_training_episodes"),
        FROZEN_TOTAL_EPISODES,
    )
    _require_equal(
        "prospective base latest protocol completion",
        raw_base_latest.get("protocol_training_complete"),
        True,
    )
    _require_equal(
        "episode-500 source latest deployment policy",
        _policy_digest(base_latest),
        final_arm.policy_digest,
    )
    if shared is None:  # pragma: no cover - twenty-entry invariant above
        raise RuntimeError("prospective bundle contains no snapshot")
    return SnapshotBundle(
        training_dir=training_dir,
        model_seed=model_seed,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        manifest=manifest,
        snapshots=tuple(arms),
        reference_episode=reference_episode,
        reference_sha256=reference_sha,
        shared_execution_contract=shared,
    )


def _load_bundles(training_dirs: Sequence[Path]) -> tuple[SnapshotBundle, ...]:
    if len(training_dirs) != len(MODEL_SEEDS):
        raise ValueError("exactly three --training-dirs are required")
    resolved = tuple(path.resolve() for path in training_dirs)
    if len(set(resolved)) != len(resolved):
        raise ValueError("prospective training directories must be distinct")
    bundles = tuple(_authenticate_snapshot_bundle(path) for path in resolved)
    by_seed = {bundle.model_seed: bundle for bundle in bundles}
    if set(by_seed) != set(MODEL_SEEDS) or len(by_seed) != len(bundles):
        raise ValueError("prospective bundles must contain model seeds 3, 4, and 5 once")
    ordered = tuple(by_seed[seed] for seed in MODEL_SEEDS)
    shared = ordered[0].shared_execution_contract
    for bundle in ordered[1:]:
        _require_equal(
            f"seed {bundle.model_seed} shared execution contract",
            bundle.shared_execution_contract,
            shared,
        )
    return ordered


def _validate_protocol() -> None:
    all_panel_seeds = set(PANEL_A_SEEDS) | set(PANEL_B_SEEDS)
    if len(all_panel_seeds) != 60 or set(PANEL_A_SEEDS) & set(PANEL_B_SEEDS):
        raise RuntimeError("prospective A/B panels must be disjoint 30-instance panels")
    forbidden = set(PROTECTED_SEEDS) | set(OPENED_DEVELOPMENT_SEEDS) | set(FROZEN_VALIDATION_SEEDS)
    # Earlier proper-training seeds occupy disjoint million-scale namespaces;
    # prospective seeds 3/4/5 use the corresponding 53--55M namespaces.
    for model_seed in tuple(range(0, 6)):
        forbidden.update(
            range(50_000_000 + model_seed * 1_000_000, 50_000_000 + model_seed * 1_000_000 + FROZEN_TOTAL_EPISODES)
        )
    overlap = all_panel_seeds & forbidden
    if overlap:
        raise RuntimeError(f"prospective panels overlap a declared seed namespace: {sorted(overlap)}")


def _instance_contract(panel: str, payload: Mapping) -> dict:
    return {
        "protocol": PROTOCOL,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "panel": panel,
        "seeds": PANELS[panel],
        "environment": payload["environment"],
        "shared_execution_contract_sha256": _digest_json(_shared_execution_contract(payload)),
    }


def _load_or_create_panel_instances(
    output_dir: Path,
    panel: str,
    payload: Mapping,
) -> tuple[dict[int, EpisodeInstance], dict]:
    """Materialize one panel; only ``main`` calls this panel-opening function."""

    seeds = PANELS[panel]
    manifest_path = output_dir / PANEL_INSTANCE_MANIFEST[panel]
    instances_dir = output_dir / "instances" / f"panel-{panel.lower()}"
    environment = _make_env(payload)
    contract = _instance_contract(panel, payload)
    existing = _load_json(manifest_path) if manifest_path.is_file() else None
    if existing is not None:
        _require_equal(f"panel {panel} instance contract", existing.get("contract"), contract)
    instances_dir.mkdir(parents=True, exist_ok=True)
    instances = {}
    records = {}
    for seed in seeds:
        expected = _make_env(payload).sample_episode_instance(seed)
        expected_text = expected.to_json() + "\n"
        path = instances_dir / f"seed-{seed}.json"
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if text != expected_text:
                raise ValueError(f"immutable EpisodeInstance mismatch: {path}")
        else:
            _atomic_text(path, expected_text)
        instance = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
        instance.validate_for(environment)
        _require_equal(f"panel {panel} instance seed", instance.seed, seed)
        instances[seed] = instance
        records[str(seed)] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "instance_id": instance.instance_id,
            "schedule_id": instance.schedule_id,
        }
    manifest = {"contract": contract, "instances": records}
    if existing is not None:
        _require_equal(f"panel {panel} immutable instance manifest", existing, manifest)
    else:
        _atomic_json(manifest_path, manifest)
    return instances, manifest


def _arm_identity(arm: SnapshotArm) -> dict:
    return {
        "method_id": arm.method_id,
        "policy_group": "prospective_snapshot",
        "system_family": "learned_exact_viability_constrained_graph_smdp",
        "model_seed": arm.model_seed,
        "checkpoint_variant": f"episode_{arm.episode:04d}",
        "checkpoint_weight_episode": arm.episode,
        "checkpoint_path": str(arm.path.resolve()),
        "checkpoint_sha256": arm.sha256,
        "deployment_policy_digest": arm.policy_digest,
        "primary_analysis": False,
        "diagnostic_only": True,
        "deployment_checkpoint_eligible": False,
        "checkpoint_readiness_interpretation": "prospective_selection_snapshot_not_final_tested",
        "execution_reused": False,
        "derived_from_exact_policy_duplicate": False,
        "derived_from_method_id": None,
    }


def _finish_row(normalized: Mapping, arm: SnapshotArm, panel: str) -> dict:
    delivery_count = int(normalized["delivery_count"])
    contention = validate_contention_metric_record(
        normalized, require_legacy_alias=True
    )
    physical_relocations = contention["physical_storage_relocations"]
    identity = _arm_identity(arm)
    return {
        **normalized,
        **identity,
        "protocol": PROTOCOL,
        "panel": panel,
        "method": arm.method_id,
        "environment_legacy_return": normalized["legacy_rescored_return"],
        "dense_objective_return": normalized["dense_rescored_return"],
        "primary_objective_return": normalized["dense_rescored_return"],
        "physical_storage_relocations_per_100_deliveries": (
            100.0 * physical_relocations / delivery_count
            if delivery_count
            else None
        ),
        # Migration alias; never used as the source of the V2 rank/gate.
        "relocations_per_100_deliveries": (
            100.0 * physical_relocations / delivery_count
            if delivery_count
            else None
        ),
    }


def _run_snapshot_once(
    arm: SnapshotArm,
    instance: EpisodeInstance,
    seed: int,
    *,
    panel: str,
    block_count: int,
    device: torch.device,
) -> dict:
    raw = run_arm(
        arm=EXACT_FULL,
        controller_payload=arm.payload,
        instance=instance,
        instance_seed=seed,
        search_config=_search_config(arm.payload),
        liveness_rule=_liveness_rule(arm.payload),
        prioritizer=None,
        max_steps=MAX_STEPS,
        device=device,
    )
    normalized = _normalize_vcg(raw, instance, block_count, FROZEN_OBJECTIVE_SPEC)
    return _finish_row(normalized, arm, panel)


def _ledger_path(output_dir: Path, panel: str, method_id: str, seed: int) -> Path:
    return output_dir / "run-ledger" / f"panel-{panel.lower()}" / method_id / f"seed-{seed}.json"


def _input_contract(
    panel: str,
    arm: SnapshotArm,
    instance_record: Mapping,
    shared_contract_sha256: str,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "selection_rule": SELECTION_RULE,
        "panel": panel,
        "method_id": arm.method_id,
        "model_seed": arm.model_seed,
        "checkpoint_episode": arm.episode,
        "instance_seed": int(Path(instance_record["path"]).stem.split("-")[-1]),
        "instance_id": instance_record["instance_id"],
        "schedule_id": instance_record["schedule_id"],
        "instance_sha256": instance_record["sha256"],
        "checkpoint_sha256": arm.sha256,
        "deployment_policy_digest": arm.policy_digest,
        "shared_execution_contract_sha256": shared_contract_sha256,
        "max_steps": MAX_STEPS,
        "timing_objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "contention_metric_schema_version": CONTENTION_METRIC_SCHEMA_VERSION,
        "selection_metric": (
            "physical_storage_relocations_per_100_deliveries"
        ),
    }


def _load_or_execute_run(
    *,
    output_dir: Path,
    panel: str,
    arm: SnapshotArm,
    seed: int,
    input_contract: dict,
    execute: Callable[[], dict],
) -> tuple[dict, dict]:
    path = _ledger_path(output_dir, panel, arm.method_id, seed)
    fingerprint = _digest_json(input_contract)
    if path.is_file():
        record = _load_json(path)
        _require_equal("ledger protocol", record.get("protocol"), PROTOCOL)
        _require_equal("ledger panel", record.get("panel"), panel)
        _require_equal("ledger run key", record.get("run_key"), f"{panel}:{arm.method_id}:{seed}")
        _require_equal("ledger input contract", record.get("input_contract"), input_contract)
        _require_equal("ledger input fingerprint", record.get("input_fingerprint"), fingerprint)
        row = record.get("run")
        if not isinstance(row, dict):
            raise ValueError(f"ledger has no run row: {path}")
    else:
        row = execute()
        if row.get("method_id") != arm.method_id or int(row.get("instance_seed", -1)) != seed:
            raise RuntimeError("snapshot executor returned the wrong method/instance identity")
        if row.get("panel") != panel:
            raise RuntimeError("snapshot executor returned the wrong panel identity")
        record = {
            "protocol": PROTOCOL,
            "panel": panel,
            "run_key": f"{panel}:{arm.method_id}:{seed}",
            "input_contract": input_contract,
            "input_fingerprint": fingerprint,
            "run": {key: value for key, value in row.items() if key != "method_audit"},
            "method_audit": row.get("method_audit"),
        }
        _atomic_json(path, record)
    if row["instance_id"] != input_contract["instance_id"] or row["schedule_id"] != input_contract["schedule_id"]:
        raise RuntimeError("ledger run EpisodeInstance provenance mismatch")
    return dict(row), record


def _execution_plan(arms: Sequence[SnapshotArm], panel: str) -> tuple[tuple[str, int], ...]:
    plan = tuple((arm.method_id, seed) for arm in arms for seed in PANELS[panel])
    if len(plan) != len(set(plan)):
        raise RuntimeError(f"panel {panel} execution plan contains duplicates")
    return plan


def _expected_run_keys(arms: Sequence[SnapshotArm], panel: str) -> set[tuple[str, int]]:
    return set(_execution_plan(arms, panel))


def _assert_complete_panel_grid(rows: Sequence[Mapping], arms: Sequence[SnapshotArm], panel: str) -> None:
    observed = {(str(row["method_id"]), int(row["instance_seed"])) for row in rows}
    expected = _expected_run_keys(arms, panel)
    if len(rows) != len(expected) or observed != expected:
        raise RuntimeError(f"panel {panel} snapshot/instance grid is incomplete")
    if any(row.get("panel") != panel for row in rows):
        raise RuntimeError(f"panel {panel} row has wrong panel identity")


def _strict_full(rows: Sequence[Mapping], expected_episodes: int) -> bool:
    if len(rows) != expected_episodes:
        return False
    try:
        for row in rows:
            validate_contention_metric_record(
                row, require_legacy_alias=True
            )
            if not (
                float(row["strict_method_success"]) == 1.0
                and float(row["completion_rate"]) == 1.0
                and int(row["delivery_count"])
                == EXPECTED_DELIVERIES_PER_EPISODE
            ):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _finite_mean_or_none(rows: Sequence[Mapping], field: str) -> Optional[float]:
    values = []
    for row in rows:
        value = row.get(field)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        values.append(number)
    return fmean(values) if values else None


def _summarize_arm(rows: Sequence[Mapping], expected_episodes: int) -> dict:
    if len(rows) != expected_episodes:
        raise ValueError("snapshot panel summary has the wrong episode count")
    deliveries = sum(int(row["delivery_count"]) for row in rows)
    physical_relocations = sum(
        int(row["physical_storage_relocations"]) for row in rows
    )
    strict_full = _strict_full(rows, expected_episodes)
    numeric = {
        field: _finite_mean_or_none(rows, field)
        for field in (
            "strict_method_success",
            "completion_rate",
            "mean_absolute_error",
            "dense_rescored_return",
            "mean_tardiness",
            "within_target_window_rate",
            "steps",
        )
    }
    if strict_full and any(value is None for value in numeric.values()):
        raise ValueError("strict/full snapshot panel has undefined numeric metrics")
    return {
        "episodes": len(rows),
        "strict_full": strict_full,
        "strict_method_success_rate": numeric["strict_method_success"],
        "completion_rate": numeric["completion_rate"],
        "delivery_count": deliveries,
        "mean_absolute_error": numeric["mean_absolute_error"],
        "contention_metric_schema_version": CONTENTION_METRIC_SCHEMA_VERSION,
        "physical_storage_relocations": physical_relocations,
        "physical_storage_relocations_per_100_deliveries": (
            100.0 * physical_relocations / deliveries
            if deliveries
            else math.nan
        ),
        "target_bound_obstruction_clearances": sum(
            int(row["target_bound_obstruction_clearances"]) for row in rows
        ),
        "standalone_reconfigurations": sum(
            int(row["standalone_reconfigurations"]) for row in rows
        ),
        "standalone_with_direct_delivery_available": sum(
            int(row["standalone_with_direct_delivery_available"])
            for row in rows
        ),
        "standalone_without_direct_delivery_available": sum(
            int(row["standalone_without_direct_delivery_available"])
            for row in rows
        ),
        "directly_deliverable_self_reconfigurations": sum(
            int(row["directly_deliverable_self_reconfigurations"])
            for row in rows
        ),
        # Migration aliases.  Their equality to the canonical fields is
        # enforced at every row before this summary is claimable.
        "relocations": physical_relocations,
        "relocations_per_100_deliveries": (
            100.0 * physical_relocations / deliveries
            if deliveries
            else math.nan
        ),
        "mean_dense_rescored_return": numeric["dense_rescored_return"],
        "mean_tardiness": numeric["mean_tardiness"],
        "within_target_window_rate": numeric["within_target_window_rate"],
        "mean_steps": numeric["steps"],
    }


def _panel_summaries(
    rows: Sequence[Mapping], bundles: Sequence[SnapshotBundle], panel: str
) -> dict[int, dict[int, dict]]:
    count = len(PANELS[panel])
    output: dict[int, dict[int, dict]] = {}
    for bundle in bundles:
        output[bundle.model_seed] = {}
        for arm in bundle.snapshots:
            selected = [row for row in rows if row["method_id"] == arm.method_id]
            output[bundle.model_seed][arm.episode] = _summarize_arm(selected, count)
    return output


def _rank_panel_a(
    summaries: Mapping[int, Mapping], reference_episode: int, *, mae_margin: float = MAE_MARGIN
) -> dict:
    if reference_episode not in summaries:
        raise ValueError("panel A summaries omit the V1.1 reference")
    reference = summaries[reference_episode]
    if reference.get("strict_full") is not True:
        raise ValueError("V1.1 reference is not strict/full on panel A")
    reference_mae = float(reference["mean_absolute_error"])
    candidates = []
    table = []
    for episode in sorted(summaries):
        summary = summaries[episode]
        mae = summary.get("mean_absolute_error")
        physical_relocation = summary.get(
            "physical_storage_relocations_per_100_deliveries"
        )
        dense = summary.get("mean_dense_rescored_return")
        finite_metrics = bool(
            mae is not None
            and physical_relocation is not None
            and dense is not None
            and all(
                math.isfinite(float(value))
                for value in (mae, physical_relocation, dense)
            )
        )
        mae_cost = float(mae) - reference_mae if finite_metrics else None
        eligible = bool(
            summary.get("strict_full") is True
            and finite_metrics
            and mae_cost is not None
            and mae_cost <= mae_margin
        )
        row = {
            "checkpoint_episode": int(episode),
            "is_reference": int(episode) == int(reference_episode),
            "strict_full": bool(summary.get("strict_full")),
            "mean_absolute_error": None if mae is None else float(mae),
            "mae_cost_vs_reference": mae_cost,
            "mae_margin": float(mae_margin),
            "physical_storage_relocations_per_100_deliveries": (
                None
                if physical_relocation is None
                else float(physical_relocation)
            ),
            "relocations_per_100_deliveries": (
                None
                if physical_relocation is None
                else float(physical_relocation)
            ),
            "mean_dense_rescored_return": None if dense is None else float(dense),
            "panel_A_eligible": eligible,
        }
        table.append(row)
        if eligible and episode != reference_episode:
            candidates.append(row)
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["physical_storage_relocations_per_100_deliveries"],
            -row["mean_dense_rescored_return"],
            row["mean_absolute_error"],
            row["checkpoint_episode"],
        ),
    )
    return {
        "reference_episode": int(reference_episode),
        "reference_summary": dict(reference),
        "mae_margin": float(mae_margin),
        "candidate_table": tuple(table),
        "ranked_nonreference_candidate_episodes": tuple(
            row["checkpoint_episode"] for row in ranked
        ),
        "selection_order_with_mandatory_fallback": tuple(
            [row["checkpoint_episode"] for row in ranked] + [int(reference_episode)]
        ),
    }


def _gate_panel_b(
    rank: Mapping,
    summaries: Mapping[int, Mapping],
    *,
    mae_margin: float = MAE_MARGIN,
) -> dict:
    reference_episode = int(rank["reference_episode"])
    reference = summaries.get(reference_episode)
    if not isinstance(reference, Mapping) or reference.get("strict_full") is not True:
        raise ValueError("mandatory V1.1 reference is not strict/full on panel B")
    reference_mae = float(reference["mean_absolute_error"])
    reference_physical_relocation = float(
        reference["physical_storage_relocations_per_100_deliveries"]
    )
    trials = []
    chosen = None
    for position, episode in enumerate(rank["selection_order_with_mandatory_fallback"], start=1):
        episode = int(episode)
        summary = summaries[episode]
        is_reference = episode == reference_episode
        mae = summary.get("mean_absolute_error")
        physical_relocation = summary.get(
            "physical_storage_relocations_per_100_deliveries"
        )
        finite_metrics = bool(
            mae is not None
            and physical_relocation is not None
            and math.isfinite(float(mae))
            and math.isfinite(float(physical_relocation))
        )
        mae_cost = float(mae) - reference_mae if finite_metrics else None
        physical_relocation_saving = (
            reference_physical_relocation - float(physical_relocation)
            if finite_metrics
            else None
        )
        strict_full = summary.get("strict_full") is True
        if is_reference:
            accepted = bool(strict_full)
            reason = "mandatory_authenticated_v1_1_reference_fallback"
        else:
            accepted = bool(
                strict_full
                and finite_metrics
                and mae_cost is not None
                and mae_cost <= mae_margin
                and physical_relocation_saving is not None
                and physical_relocation_saving > POSITIVE_TOLERANCE
            )
            reason = (
                "first_panel_A_ranked_candidate_passing_frozen_panel_B_gate"
                if accepted
                else (
                    "panel_B_strict_full_mae_or_positive_physical_storage_"
                    "relocation_gate_failed"
                )
            )
        trials.append(
            {
                "selection_order": position,
                "checkpoint_episode": episode,
                "is_reference_fallback": is_reference,
                "strict_full": strict_full,
                "mean_absolute_error": None if mae is None else float(mae),
                "mae_cost_vs_reference": mae_cost,
                "mae_margin": float(mae_margin),
                "physical_storage_relocations_per_100_deliveries": (
                    None
                    if physical_relocation is None
                    else float(physical_relocation)
                ),
                "physical_storage_relocation_saving_vs_reference": (
                    physical_relocation_saving
                ),
                "positive_physical_storage_relocation_saving_required": (
                    not is_reference
                ),
                # Migration aliases for existing plotting/report consumers.
                "relocations_per_100_deliveries": (
                    None
                    if physical_relocation is None
                    else float(physical_relocation)
                ),
                "relocation_saving_vs_reference": (
                    physical_relocation_saving
                ),
                "positive_relocation_saving_required": not is_reference,
                "accepted": accepted,
                "reason": reason,
            }
        )
        if accepted:
            chosen = episode
            break
    if chosen is None:  # reference is strict/full and necessarily last
        raise RuntimeError("frozen panel-B gate did not reach mandatory reference fallback")
    return {
        "reference_episode": reference_episode,
        "chosen_episode": chosen,
        "selected_nonreference": chosen != reference_episode,
        "panel_B_trials": tuple(trials),
        "chosen_summary": dict(summaries[chosen]),
        "reference_summary": dict(reference),
    }


def _compute_panel_a_ranks(
    bundles: Sequence[SnapshotBundle], panel_a_rows: Sequence[Mapping]
) -> dict[int, dict]:
    """Freeze all A ranks, failing before panel B can be materialized."""

    a_summaries = _panel_summaries(panel_a_rows, bundles, "A")
    return {
        bundle.model_seed: _rank_panel_a(
            a_summaries[bundle.model_seed], bundle.reference_episode
        )
        for bundle in bundles
    }


def _selection_report(
    bundles: Sequence[SnapshotBundle],
    panel_a_rows: Sequence[Mapping],
    panel_b_rows: Sequence[Mapping],
    *,
    panel_a_ranks: Optional[Mapping[int, Mapping]] = None,
) -> dict:
    b_summaries = _panel_summaries(panel_b_rows, bundles, "B")
    frozen_ranks = (
        _compute_panel_a_ranks(bundles, panel_a_rows)
        if panel_a_ranks is None
        else {int(seed): dict(rank) for seed, rank in panel_a_ranks.items()}
    )
    _require_equal(
        "panel-A frozen rank model seeds",
        tuple(sorted(frozen_ranks)),
        tuple(MODEL_SEEDS),
    )
    selections = []
    for bundle in bundles:
        rank = frozen_ranks[bundle.model_seed]
        gate = _gate_panel_b(rank, b_summaries[bundle.model_seed])
        chosen_arm = next(arm for arm in bundle.snapshots if arm.episode == gate["chosen_episode"])
        selections.append(
            {
                "model_seed": bundle.model_seed,
                "reference_episode": bundle.reference_episode,
                "reference_checkpoint_sha256": bundle.reference_sha256,
                "panel_A": rank,
                "panel_B": gate,
                "panel_B_all_snapshot_summaries": tuple(
                    {
                        "checkpoint_episode": episode,
                        **dict(b_summaries[bundle.model_seed][episode]),
                    }
                    for episode in sorted(b_summaries[bundle.model_seed])
                ),
                "chosen_episode": chosen_arm.episode,
                "chosen_checkpoint_path": str(chosen_arm.path.resolve()),
                "chosen_checkpoint_sha256": chosen_arm.sha256,
                "chosen_deployment_policy_digest": chosen_arm.policy_digest,
                "selected_nonreference": gate["selected_nonreference"],
                "deployment_checkpoint_eligible": False,
                "eligibility_reason": "requires_untouched_final_panel_after_prospective_development_selection",
            }
        )
    nonreference_count = sum(bool(item["selected_nonreference"]) for item in selections)
    total_reference_physical_relocations = sum(
        item["panel_B"]["reference_summary"][
            "physical_storage_relocations"
        ]
        for item in selections
    )
    total_chosen_physical_relocations = sum(
        item["panel_B"]["chosen_summary"][
            "physical_storage_relocations"
        ]
        for item in selections
    )
    total_deliveries = sum(
        item["panel_B"]["chosen_summary"]["delivery_count"] for item in selections
    )
    equal_seed_physical_relocation_saving = (
        100.0
        * (
            total_reference_physical_relocations
            - total_chosen_physical_relocations
        )
        / total_deliveries
    )
    return {
        "protocol": PROTOCOL,
        "scope": "prospective_development_selection_panels_A_and_B_final_panel_unopened",
        "performance_claim_authorized": False,
        "selection_rule": SELECTION_RULE,
        "contention_metric_schema_version": CONTENTION_METRIC_SCHEMA_VERSION,
        "selection_metric": (
            "physical_storage_relocations_per_100_deliveries"
        ),
        "mae_margin": MAE_MARGIN,
        "selections": tuple(selections),
        "advancement_gate": {
            "requires_at_least_two_of_three_nonreference": True,
            "nonreference_selected_count": nonreference_count,
            "requires_positive_equal_seed_panel_B_physical_storage_relocation_saving": True,
            "equal_seed_panel_B_physical_storage_relocation_saving_per_100_deliveries": (
                equal_seed_physical_relocation_saving
            ),
            # Migration aliases.
            "requires_positive_equal_seed_panel_B_relocation_saving": True,
            "equal_seed_panel_B_relocation_saving_per_100_deliveries": (
                equal_seed_physical_relocation_saving
            ),
            "passed": bool(
                nonreference_count >= 2
                and equal_seed_physical_relocation_saving > POSITIVE_TOLERANCE
            ),
        },
        "final_untouched_panel_opened": False,
        "deployment_checkpoint_eligible": False,
    }


def _write_rows(path: Path, rows: Sequence[Mapping]) -> None:
    fields = ("panel",) + tuple(
        name for name in PROSPECTIVE_RUN_FIELDS if name != "panel"
    )
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {name: _csv_value(row.get(name)) for name in fields} for row in rows
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ledger_manifest(rows: Sequence[Mapping], output_dir: Path, panel: str) -> tuple[dict, ...]:
    records = []
    for row in rows:
        path = _ledger_path(output_dir, panel, row["method_id"], int(row["instance_seed"]))
        ledger = _load_json(path)
        records.append(
            {
                "run_key": ledger["run_key"],
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "input_fingerprint": ledger["input_fingerprint"],
            }
        )
    return tuple(records)


def _run_panel(
    output_dir: Path,
    panel: str,
    bundles: Sequence[SnapshotBundle],
    instances: Mapping[int, EpisodeInstance],
    instance_manifest: Mapping,
    *,
    device: torch.device,
    shared_contract_sha256: str,
) -> list[dict]:
    arms = tuple(arm for bundle in bundles for arm in bundle.snapshots)
    arms_by_id = {arm.method_id: arm for arm in arms}
    if len(arms_by_id) != len(arms):
        raise RuntimeError("snapshot method identifiers are not unique")
    block_count = int(bundles[0].snapshots[0].payload["environment"]["number_blocks"])
    rows = []
    for method_id, seed in _execution_plan(arms, panel):
        arm = arms_by_id[method_id]
        instance = instances[seed]
        record = instance_manifest["instances"][str(seed)]
        contract = _input_contract(panel, arm, record, shared_contract_sha256)
        row, _ = _load_or_execute_run(
            output_dir=output_dir,
            panel=panel,
            arm=arm,
            seed=seed,
            input_contract=contract,
            execute=lambda arm=arm, instance=instance, seed=seed: _run_snapshot_once(
                arm,
                instance,
                seed,
                panel=panel,
                block_count=block_count,
                device=device,
            ),
        )
        rows.append(row)
        print(
            f"[panel {panel}] {method_id} seed={seed} "
            f"DenseR={row['dense_rescored_return']:.2f} "
            f"MAE={row['mean_absolute_error']} "
            f"physical_reloc={row['physical_storage_relocations']}",
            flush=True,
        )
    _assert_complete_panel_grid(rows, arms, panel)
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen prospective A-ranked/B-gated VCG-Dense V1.2 selection"
    )
    parser.add_argument("--training-dirs", nargs=3, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume-existing", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = _build_parser().parse_args(argv)
    _validate_protocol()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_existing:
        raise FileExistsError(
            f"{output_dir} is nonempty; pass --resume-existing or use a new directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = _load_bundles(args.training_dirs)
    arms = tuple(arm for bundle in bundles for arm in bundle.snapshots)
    canonical_payload = arms[0].payload
    shared_contract = bundles[0].shared_execution_contract
    shared_contract_sha = _digest_json(shared_contract)
    device = resolve_device(args.device)
    protocol_manifest = {
        "protocol": PROTOCOL,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "scope": "prospective_development_selection_final_panel_unopened",
        "performance_claim_authorized": False,
        "selection_rule": SELECTION_RULE,
        "contention_metric_schema_version": CONTENTION_METRIC_SCHEMA_VERSION,
        "common_contention_estimand": (
            "completed_physical_storage_to_storage_moves_per_100_deliveries"
        ),
        "mechanism_decomposition": {
            "identity": (
                "physical_storage_relocations_equals_target_bound_obstruction_"
                "clearances_plus_standalone_reconfigurations"
            ),
            "standalone_partition": (
                "with_direct_delivery_available_plus_without_direct_delivery_"
                "available"
            ),
            "directly_deliverable_self_reconfigurations": (
                "diagnostic_subset_of_standalone_with_direct_delivery_available"
            ),
            "legacy_relocations": "exact_alias_of_physical_storage_relocations",
            "legacy_obstructive_moves": (
                "exact_alias_of_target_bound_obstruction_clearances"
            ),
        },
        "model_seeds": MODEL_SEEDS,
        "snapshot_episodes": SNAPSHOT_EPISODES,
        "panel_A_seeds": PANEL_A_SEEDS,
        "panel_B_seeds": PANEL_B_SEEDS,
        "namespace_audit": NAMESPACE_AUDIT,
        "mae_margin": MAE_MARGIN,
        "training_bundles": tuple(
            {
                "model_seed": bundle.model_seed,
                "training_dir": str(bundle.training_dir),
                "snapshot_manifest_path": str(bundle.manifest_path),
                "snapshot_manifest_sha256": bundle.manifest_sha256,
                "reference_episode": bundle.reference_episode,
                "reference_sha256": bundle.reference_sha256,
                "snapshots": tuple(
                    {
                        "episode": arm.episode,
                        "path": str(arm.path),
                        "sha256": arm.sha256,
                        "policy_digest": arm.policy_digest,
                    }
                    for arm in bundle.snapshots
                ),
            }
            for bundle in bundles
        ),
        "shared_execution_contract": shared_contract,
        "shared_execution_contract_sha256": shared_contract_sha,
        "execution_device": str(device),
        "ordering_contract": "panel_A_complete_before_panel_B_instance_materialization",
        "all_snapshots_evaluated_on_both_panels": True,
    }
    protocol_path = output_dir / PROTOCOL_MANIFEST_FILENAME
    if protocol_path.is_file():
        _require_equal("selection protocol manifest", _load_json(protocol_path), protocol_manifest)
    else:
        if args.resume_existing and (output_dir / "run-ledger").exists():
            raise ValueError("cannot resume run ledgers without a protocol manifest")
        _atomic_json(protocol_path, protocol_manifest)

    # Panel A is fully materialized and executed before any panel-B instance is
    # sampled or written.  This sequencing is a protocol invariant, not merely
    # a reporting convention.
    a_instances, a_manifest = _load_or_create_panel_instances(output_dir, "A", canonical_payload)
    a_rows = _run_panel(
        output_dir,
        "A",
        bundles,
        a_instances,
        a_manifest,
        device=device,
        shared_contract_sha256=shared_contract_sha,
    )
    _assert_complete_panel_grid(a_rows, arms, "A")
    _write_rows(output_dir / PANEL_RESULTS_FILENAME["A"], a_rows)

    # This is deliberately before the first panel-B environment or
    # EpisodeInstance exists.  Any unsafe/missing V1.1 reference or malformed
    # A rank terminates the protocol without opening B.
    panel_a_ranks = _compute_panel_a_ranks(bundles, a_rows)
    panel_a_ranking_document = {
        "protocol": PROTOCOL,
        "selection_rule": SELECTION_RULE,
        "panel": "A",
        "panel_A_instance_manifest_sha256": _sha256_file(
            output_dir / PANEL_INSTANCE_MANIFEST["A"]
        ),
        "panel_A_runs_sha256": _sha256_file(
            output_dir / PANEL_RESULTS_FILENAME["A"]
        ),
        "ranks": panel_a_ranks,
        "panel_B_opened_when_written": False,
    }
    panel_a_ranking_path = output_dir / PANEL_A_RANKING_FILENAME
    if panel_a_ranking_path.is_file():
        _require_equal(
            "frozen panel-A ranking",
            _load_json(panel_a_ranking_path),
            panel_a_ranking_document,
        )
    else:
        if (
            (output_dir / PANEL_INSTANCE_MANIFEST["B"]).exists()
            or (output_dir / "run-ledger" / "panel-b").exists()
        ):
            raise ValueError(
                "panel B exists without the authenticated frozen panel-A "
                "ranking; temporal protocol provenance cannot be recovered"
            )
        _atomic_json(panel_a_ranking_path, panel_a_ranking_document)

    b_instances, b_manifest = _load_or_create_panel_instances(output_dir, "B", canonical_payload)
    b_rows = _run_panel(
        output_dir,
        "B",
        bundles,
        b_instances,
        b_manifest,
        device=device,
        shared_contract_sha256=shared_contract_sha,
    )
    _assert_complete_panel_grid(b_rows, arms, "B")
    _write_rows(output_dir / PANEL_RESULTS_FILENAME["B"], b_rows)

    report = _selection_report(
        bundles,
        a_rows,
        b_rows,
        panel_a_ranks=panel_a_ranks,
    )
    audit = {
        "protocol": PROTOCOL,
        "protocol_manifest": protocol_manifest,
        "panel_A_instance_manifest": a_manifest,
        "panel_B_instance_manifest": b_manifest,
        "panel_A_frozen_ranking": {
            "path": str(panel_a_ranking_path.resolve()),
            "sha256": _sha256_file(panel_a_ranking_path),
        },
        "panel_A_run_ledger_manifest": _ledger_manifest(a_rows, output_dir, "A"),
        "panel_B_run_ledger_manifest": _ledger_manifest(b_rows, output_dir, "B"),
        "panel_A_execution_count": len(a_rows),
        "panel_B_execution_count": len(b_rows),
        "physical_execution_count": len(a_rows) + len(b_rows),
        "expected_physical_execution_count": len(MODEL_SEEDS) * len(SNAPSHOT_EPISODES) * 60,
    }
    _atomic_json(output_dir / REPORT_FILENAME, report)
    _atomic_json(output_dir / AUDIT_FILENAME, audit)
    print(f"Panel A: {output_dir / PANEL_RESULTS_FILENAME['A']}", flush=True)
    print(f"Panel B: {output_dir / PANEL_RESULTS_FILENAME['B']}", flush=True)
    print(f"Report: {output_dir / REPORT_FILENAME}", flush=True)
    print(f"Audit: {output_dir / AUDIT_FILENAME}", flush=True)
    return report


if __name__ == "__main__":
    main()
