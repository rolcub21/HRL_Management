#!/usr/bin/env python3
"""Frozen VCG-Dense V1.1 control on the opened V2.2 development panel.

This is a development-only diagnostic.  It authenticates the three finalized
VCG-Dense V1.1 selected-best bundles, regenerates only the already-observed
85000--85011 EpisodeInstances and hash-matches them to the V2.2 manifest, then
executes one deterministic (epsilon-zero), frozen rollout per
model-seed x EpisodeInstance.  It never samples an 86xxx EpisodeInstance or
uses the reserved V2.2 final action-RNG namespace.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence
import uuid

import torch

import compare_vcg_dense_pareto as pareto
from example.episode_instance import EpisodeInstance
from track_b_urgency_evaluate import resolve_device
from train_vcg_dense_proper import (
    FROZEN_GAMMA,
    FROZEN_MAX_STEPS,
    FROZEN_OBJECTIVE_SPEC,
)


PROTOCOL = "vcg_dense_v1_1_on_v2_2_open_development_panel_control_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (0, 1, 2)
EVALUATION_SEEDS = tuple(range(85_000, 85_012))
V22_REFERENCE_EPISODE = 80
V22_METHOD_VERSION = "vcg_constrained_v2_2"
V22_TRAINING_PROTOCOL = (
    "vcg_constrained_vector_smdp_v2_2_policy_operator_consistent_"
    "development_calibration_v1"
)
FINAL_PANEL_SEEDS = frozenset(range(86_000, 86_030))
FINAL_POLICY_RNG_RANGE = range(622_000_000, 622_000_120)

REPORT_FILENAME = "control-report.json"
AUDIT_FILENAME = "control-audit.json"
RUNS_FILENAME = "control-runs.csv"
PROTOCOL_MANIFEST_FILENAME = "protocol-manifest.json"
INSTANCE_MANIFEST_FILENAME = "instance-manifest.json"


class V11PanelControlError(ValueError):
    """Raised when a frozen control or provenance condition is violated."""


def _json_safe(value):
    return pareto._json_safe(value)


def _canonical_json(value) -> str:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest_json(value) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise V11PanelControlError(f"{path} must contain a JSON object")
    return result


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


def _require_equal(name: str, observed, expected) -> None:
    if _json_safe(observed) != _json_safe(expected):
        raise V11PanelControlError(
            f"{name} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _verify_embedded_hash(payload: Mapping, field: str, *, name: str) -> str:
    claimed = payload.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise V11PanelControlError(f"{name} has no valid {field}")
    unhashed = dict(payload)
    unhashed.pop(field, None)
    observed = _digest_json(unhashed)
    _require_equal(f"{name}.{field}", claimed, observed)
    return claimed


def _validate_protocol_constants() -> None:
    if EVALUATION_SEEDS != tuple(range(85_000, 85_012)):
        raise RuntimeError("the V1.1 control panel drifted")
    if set(EVALUATION_SEEDS).intersection(FINAL_PANEL_SEEDS):
        raise RuntimeError("the V1.1 control intersects the unopened final panel")
    if any(seed in FINAL_POLICY_RNG_RANGE for seed in EVALUATION_SEEDS):
        raise RuntimeError("instance and final action-RNG namespaces overlap")
    if MODEL_SEEDS != (0, 1, 2):
        raise RuntimeError("the equal-training-seed estimand requires seeds 0, 1, 2")


def _source_paths(source_dir: Path) -> dict[str, Path]:
    return {
        "training-contract.json": source_dir / "training-contract.json",
        "training-summary.json": source_dir / "training-summary.json",
        "validation-instance-manifest.json": (
            source_dir / "validation-instance-manifest.json"
        ),
        "validation-ledger/episode-0080.json": (
            source_dir / "validation-ledger" / "episode-0080.json"
        ),
    }


def _source_file_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in sorted(paths.items())}


def _position_slice(values: Sequence[float], *, window: float = 20.0) -> dict:
    values = tuple(float(value) for value in values)
    if not values:
        return {
            "n": 0,
            "mean_signed_deviation": None,
            "mean_absolute_error": None,
            "within_target_window_rate": None,
            "mean_tardiness": None,
            "mean_earliness": None,
        }
    return {
        "n": len(values),
        "mean_signed_deviation": float(fmean(values)),
        "mean_absolute_error": float(fmean(abs(value) for value in values)),
        "within_target_window_rate": float(
            fmean(abs(value) <= float(window) for value in values)
        ),
        "mean_tardiness": float(fmean(max(value, 0.0) for value in values)),
        "mean_earliness": float(fmean(max(-value, 0.0) for value in values)),
    }


def _delivery_position_summary(
    rows: Sequence[Mapping], *, window: float = 20.0
) -> dict:
    sequences = tuple(
        tuple(float(value) for value in row["delivery_deviations"])
        for row in rows
    )
    longest = max((len(sequence) for sequence in sequences), default=0)
    by_position = []
    for index in range(longest):
        values = tuple(
            sequence[index] for sequence in sequences if len(sequence) > index
        )
        by_position.append(
            {"delivery_position": index + 1, **_position_slice(values, window=window)}
        )
    return {
        "by_delivery_position": tuple(by_position),
        "first_two": _position_slice(
            tuple(value for sequence in sequences for value in sequence[:2]),
            window=window,
        ),
        "positions_three_plus": _position_slice(
            tuple(value for sequence in sequences for value in sequence[2:]),
            window=window,
        ),
    }


def _summarize_rows(rows: Sequence[Mapping]) -> dict:
    rows = tuple(rows)
    if not rows:
        raise V11PanelControlError("cannot summarize an empty rollout set")
    deviations = tuple(
        float(value)
        for row in rows
        for value in row.get("delivery_deviations", ())
    )
    deliveries = sum(int(row["delivery_count"]) for row in rows)
    if deliveries != len(deviations):
        raise V11PanelControlError("delivery count/deviation sequence mismatch")
    physical = sum(int(row["physical_storage_relocations"]) for row in rows)
    return {
        "episodes": len(rows),
        "total_deliveries": deliveries,
        "mean_dense_return": float(
            fmean(float(row["dense_objective_return"]) for row in rows)
        ),
        "mean_absolute_error": float(fmean(abs(value) for value in deviations)),
        "mean_signed_deviation": float(fmean(deviations)),
        "within_target_window_rate": float(
            fmean(abs(value) <= FROZEN_OBJECTIVE_SPEC.window for value in deviations)
        ),
        "mean_steps": float(fmean(float(row["steps"]) for row in rows)),
        "total_physical_rehandles": physical,
        "physical_rehandles_per_100_deliveries": float(
            100.0 * physical / deliveries
        ),
        "strict_method_success_rate": float(
            fmean(float(row["strict_method_success"]) for row in rows)
        ),
        "completion_rate": float(
            fmean(float(row["completion_rate"]) for row in rows)
        ),
        "all_complete_exact_safe": bool(
            all(
                bool(row["strict_method_success"])
                and math.isclose(float(row["completion_rate"]), 1.0)
                and bool(row["all_selected_candidates_exact_safe"])
                and int(row["illegal_drops"]) == 0
                and int(row["fallbacks"]) == 0
                and row["method_failure_reason"] is None
                and row["evaluation_learning"] is False
                and math.isclose(float(row["evaluation_epsilon"]), 0.0)
                for row in rows
            )
        ),
        "delivery_position_metrics": _delivery_position_summary(
            rows, window=FROZEN_OBJECTIVE_SPEC.window
        ),
    }


def _equal_seed_aggregate(seed_summaries: Mapping[int, Mapping]) -> dict:
    _require_equal("equal-seed summary keys", sorted(seed_summaries), list(MODEL_SEEDS))
    metrics = (
        "mean_dense_return",
        "mean_absolute_error",
        "mean_signed_deviation",
        "within_target_window_rate",
        "mean_steps",
        "physical_rehandles_per_100_deliveries",
        "strict_method_success_rate",
        "completion_rate",
    )
    aggregate = {
        "estimand": "equal_weight_over_three_fixed_model_seeds",
        "model_seed_count": len(MODEL_SEEDS),
        **{
            metric: float(fmean(float(seed_summaries[seed][metric]) for seed in MODEL_SEEDS))
            for metric in metrics
        },
        "all_complete_exact_safe": all(
            bool(seed_summaries[seed]["all_complete_exact_safe"])
            for seed in MODEL_SEEDS
        ),
    }
    # The crossed grid is balanced (12 instances and 8 deliveries per seed),
    # so the pooled position summaries implement the same equal-seed estimand.
    return aggregate


def _validate_v22_source(source_dir: Path) -> dict:
    paths = _source_paths(source_dir)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = _load_json(paths["training-contract.json"])
    summary = _load_json(paths["training-summary.json"])
    instance_manifest = _load_json(paths["validation-instance-manifest.json"])
    reference_ledger = _load_json(paths["validation-ledger/episode-0080.json"])

    contract_hash = _verify_embedded_hash(
        contract, "contract_sha256", name="V2.2 training contract"
    )
    instance_manifest_hash = _verify_embedded_hash(
        instance_manifest,
        "manifest_sha256",
        name="V2.2 validation instance manifest",
    )
    ledger_hash = _verify_embedded_hash(
        reference_ledger, "ledger_sha256", name="V2.2 episode-80 ledger"
    )

    expected_contract = {
        "training_protocol": V22_TRAINING_PROTOCOL,
        "method_version": V22_METHOD_VERSION,
        "development_only": True,
        "performance_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "episodes": 200,
        "validation_seeds": EVALUATION_SEEDS,
        "max_steps": FROZEN_MAX_STEPS,
        "gamma_operational": FROZEN_GAMMA,
        "dense_objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
    }
    for field, expected in expected_contract.items():
        _require_equal(f"V2.2 contract.{field}", contract.get(field), expected)
    _require_equal("V2.2 summary.status", summary.get("status"), "complete")
    _require_equal(
        "V2.2 summary.method_version",
        summary.get("method_version"),
        V22_METHOD_VERSION,
    )
    _require_equal(
        "V2.2 summary.completed_training_episodes",
        summary.get("completed_training_episodes"),
        200,
    )
    _require_equal(
        "V2.2 summary.final_86xxx_panel_opened",
        summary.get("final_86xxx_panel_opened"),
        False,
    )
    _require_equal(
        "V2.2 manifest.final_86xxx_panel_opened",
        instance_manifest.get("final_86xxx_panel_opened"),
        False,
    )
    _require_equal(
        "V2.2 manifest.common_grid_every_checkpoint",
        instance_manifest.get("common_grid_every_checkpoint"),
        True,
    )
    _require_equal(
        "V2.2 summary training contract path",
        Path(summary["training_contract"]).resolve(),
        paths["training-contract.json"].resolve(),
    )
    _require_equal(
        "V2.2 summary instance manifest path",
        Path(summary["validation_instance_manifest"]).resolve(),
        paths["validation-instance-manifest.json"].resolve(),
    )

    manifest_records = instance_manifest.get("instances")
    if not isinstance(manifest_records, list) or len(manifest_records) != 12:
        raise V11PanelControlError("V2.2 manifest must contain exactly 12 instances")
    manifest_by_seed = {int(record["instance_seed"]): record for record in manifest_records}
    _require_equal("V2.2 manifest seed grid", sorted(manifest_by_seed), list(EVALUATION_SEEDS))

    _require_equal(
        "V2.2 reference ledger checkpoint",
        reference_ledger.get("checkpoint_episode"),
        V22_REFERENCE_EPISODE,
    )
    rows = reference_ledger.get("rows")
    if not isinstance(rows, list) or len(rows) != 48:
        raise V11PanelControlError("V2.2 episode-80 ledger must contain 48 rows")
    expected_keys = {(seed, index) for seed in EVALUATION_SEEDS for index in range(4)}
    observed_keys = {
        (int(row["instance_seed"]), int(row["policy_rng_index"])) for row in rows
    }
    _require_equal("V2.2 episode-80 crossed grid", observed_keys, expected_keys)
    for row in rows:
        record = manifest_by_seed[int(row["instance_seed"])]
        for row_field, manifest_field in (
            ("episode_instance_id", "episode_instance_id"),
            ("schedule_id", "schedule_id"),
            ("episode_instance_sha256", "episode_instance_sha256"),
        ):
            _require_equal(
                f"V2.2 episode-80 {row_field}",
                row.get(row_field),
                record.get(manifest_field),
            )
        for field, expected in (
            ("strict_method_success", True),
            ("completion_rate", 1.0),
            ("all_selected_candidates_exact_safe", True),
            ("evaluation_learning", False),
            ("map_selection_used", False),
        ):
            _require_equal(f"V2.2 episode-80 row.{field}", row.get(field), expected)

    histories = summary.get("validation_history")
    if not isinstance(histories, list):
        raise V11PanelControlError("V2.2 summary has no validation history")
    history_by_episode = {int(item["checkpoint_episode"]): item for item in histories}
    if V22_REFERENCE_EPISODE not in history_by_episode:
        raise V11PanelControlError("V2.2 summary has no episode-80 validation")
    candidate_histories = tuple(
        item for item in histories if bool(item.get("candidate_look_gate"))
    )
    closest = min(
        candidate_histories,
        key=lambda item: (float(item["mean_absolute_error"]), int(item["checkpoint_episode"])),
    )
    _require_equal(
        "nearest V2.2 candidate look", closest["checkpoint_episode"], V22_REFERENCE_EPISODE
    )
    reference_history = history_by_episode[V22_REFERENCE_EPISODE]
    _require_equal(
        "V2.2 summary episode-80 ledger hash",
        reference_history["validation_ledger"]["ledger_sha256"],
        ledger_hash,
    )
    _require_equal(
        "V2.2 summary episode-80 row count",
        reference_history["validation_ledger"]["row_count"],
        48,
    )

    reference_rows = []
    for row in rows:
        reference_rows.append(
            {
                "dense_objective_return": float(row["dense_return"]),
                "delivery_deviations": tuple(row["delivery_deviations"]),
                "delivery_count": int(row["delivery_count"]),
                "steps": int(row["steps"]),
                "physical_storage_relocations": int(row["physical_rehandles"]),
                "strict_method_success": bool(row["strict_method_success"]),
                "completion_rate": float(row["completion_rate"]),
                "all_selected_candidates_exact_safe": bool(
                    row["all_selected_candidates_exact_safe"]
                ),
                "illegal_drops": int(row["illegal_drops"]),
                "fallbacks": int(row["fallbacks"]),
                "method_failure_reason": row["method_failure_reason"],
                "evaluation_learning": bool(row["evaluation_learning"]),
                "evaluation_epsilon": 0.0,
            }
        )
    reference_summary = _summarize_rows(reference_rows)
    _require_equal(
        "V2.2 episode-80 mean dense return",
        reference_summary["mean_dense_return"],
        float(reference_history["mean_dense_return"]),
    )
    _require_equal(
        "V2.2 episode-80 MAE",
        reference_summary["mean_absolute_error"],
        float(reference_history["mean_absolute_error"]),
    )
    _require_equal(
        "V2.2 episode-80 physical rehandle rate",
        reference_summary["physical_rehandles_per_100_deliveries"],
        float(reference_history["expected_physical_rehandles_per_100_required_deliveries"]),
    )

    return {
        "paths": paths,
        "artifact_sha256": _source_file_hashes(paths),
        "contract_sha256": contract_hash,
        "instance_manifest_sha256": instance_manifest_hash,
        "reference_ledger_sha256": ledger_hash,
        "contract": contract,
        "instance_records": manifest_by_seed,
        "reference_summary": reference_summary,
        "reference_history": reference_history,
    }


def _regenerate_instances(
    output_dir: Path,
    payload: Mapping,
    v22_source: Mapping,
    *,
    resume_existing: bool,
) -> tuple[dict[int, EpisodeInstance], dict]:
    environment = pareto._make_env(payload)
    instances = {}
    records = []
    instances_dir = output_dir / "instances"
    for seed in EVALUATION_SEEDS:
        instance = environment.sample_episode_instance(seed)
        canonical_text = instance.to_json()
        canonical_sha = sha256(canonical_text.encode("utf-8")).hexdigest()
        source = v22_source["instance_records"][seed]
        _require_equal(f"seed {seed} instance id", instance.instance_id, source["episode_instance_id"])
        _require_equal(f"seed {seed} schedule id", instance.schedule_id, source["schedule_id"])
        _require_equal(
            f"seed {seed} canonical EpisodeInstance SHA-256",
            canonical_sha,
            source["episode_instance_sha256"],
        )
        path = instances_dir / f"seed-{seed}.json"
        expected_text = canonical_text + "\n"
        if path.is_file():
            _require_equal(
                f"seed {seed} saved EpisodeInstance", path.read_text(encoding="utf-8"), expected_text
            )
        else:
            _atomic_text(path, expected_text)
        restored = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
        restored.validate_for(environment)
        _require_equal(f"seed {seed} JSON round trip", restored.to_json(), canonical_text)
        instances[seed] = restored
        records.append(
            {
                "instance_seed": seed,
                "episode_instance_id": instance.instance_id,
                "schedule_id": instance.schedule_id,
                "episode_instance_sha256": canonical_sha,
                "saved_path": str(path.resolve()),
                "saved_file_sha256": _sha256_file(path),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "development_only": True,
        "regenerated_from_seed_and_hash_matched_to_v2_2": True,
        "source_v2_2_manifest_sha256": v22_source["instance_manifest_sha256"],
        "seeds": EVALUATION_SEEDS,
        "instances": records,
        "final_86xxx_panel_opened": False,
    }
    manifest["manifest_sha256"] = _digest_json(manifest)
    path = output_dir / INSTANCE_MANIFEST_FILENAME
    if path.is_file():
        if not resume_existing:
            raise FileExistsError(path)
        _require_equal("control instance manifest", _load_json(path), manifest)
    else:
        _atomic_json(path, manifest)
    return instances, manifest


def _compact_method_audit(row: dict) -> dict:
    audit = dict(row.get("method_audit", {}))
    frontiers = tuple(audit.pop("frontiers", ()))
    audit.update(
        {
            "frontier_count": len(frontiers),
            "all_frontiers_complete_exact": bool(
                frontiers
                and all(
                    bool(frontier.get("complete_frontier_exactly_verified"))
                    for frontier in frontiers
                )
            ),
            "all_frontier_candidates_exact_safe": bool(
                frontiers
                and all(
                    candidate.get("certificate", {}).get("status") == "SAFE"
                    for frontier in frontiers
                    for candidate in frontier.get("candidates", ())
                )
            ),
        }
    )
    row["method_audit"] = audit
    return row


def _execute_one(arm, instance: EpisodeInstance, instance_seed: int, device) -> dict:
    raw = pareto.run_arm(
        arm=pareto.EXACT_FULL,
        controller_payload=arm.payload,
        instance=instance,
        instance_seed=instance_seed,
        search_config=pareto._search_config(arm.payload),
        liveness_rule=pareto._liveness_rule(arm.payload),
        prioritizer=None,
        max_steps=FROZEN_MAX_STEPS,
        device=device,
    )
    _require_equal("evaluation training flag", raw.get("training"), False)
    _require_equal("evaluation gradient steps", raw.get("gradient_steps"), 0)
    _require_equal(
        "complete exact frontier flag",
        raw.get("complete_frontier_exactly_verified"),
        True,
    )
    block_count = int(arm.payload["environment"]["number_blocks"])
    normalized = pareto._normalize_vcg(
        raw, instance, block_count, FROZEN_OBJECTIVE_SPEC
    )
    row = pareto._finish_row(normalized, pareto._identity_for_policy(arm))
    row.update(
        {
            "protocol": PROTOCOL,
            "method": f"vcg_dense_v1_1_seed{arm.model_seed}_selected_best",
            "model_seed": int(arm.model_seed),
            "evaluation_epsilon": 0.0,
            "evaluation_learning": False,
            "gradient_steps": 0,
            "gamma": FROZEN_GAMMA,
            "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
            "reported_return_discounted": False,
            "all_selected_candidates_exact_safe": bool(
                raw["complete_frontier_exactly_verified"]
                and all(
                    candidate["certificate"]["status"] == "SAFE"
                    for frontier in raw["frontiers"]
                    for candidate in frontier["candidates"]
                )
            ),
        }
    )
    return _compact_method_audit(row)


def _ledger_path(output_dir: Path, model_seed: int, instance_seed: int) -> Path:
    return (
        output_dir
        / "run-ledger"
        / f"model-seed-{model_seed}"
        / f"instance-seed-{instance_seed}.json"
    )


def _load_or_execute(
    *,
    output_dir: Path,
    arm,
    instance: EpisodeInstance,
    instance_record: Mapping,
    device,
) -> tuple[dict, dict]:
    seed = int(instance_record["instance_seed"])
    path = _ledger_path(output_dir, arm.model_seed, seed)
    input_contract = {
        "protocol": PROTOCOL,
        "model_seed": int(arm.model_seed),
        "checkpoint_role": "best_deployment_finalized",
        "checkpoint_sha256": arm.checkpoint_sha256,
        "deployment_policy_digest": arm.deployment_policy_digest,
        "instance_seed": seed,
        "episode_instance_id": instance_record["episode_instance_id"],
        "schedule_id": instance_record["schedule_id"],
        "episode_instance_sha256": instance_record["episode_instance_sha256"],
        "evaluation_policy": "deterministic_greedy_epsilon_zero",
        "evaluation_epsilon": 0.0,
        "evaluation_learning": False,
        "gamma": FROZEN_GAMMA,
        "max_steps": FROZEN_MAX_STEPS,
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "development_only": True,
        "final_86xxx_panel_opened": False,
    }
    fingerprint = _digest_json(input_contract)
    if path.is_file():
        ledger = _load_json(path)
        _verify_embedded_hash(ledger, "ledger_sha256", name=str(path))
        _require_equal("ledger input contract", ledger.get("input_contract"), input_contract)
        _require_equal("ledger input fingerprint", ledger.get("input_fingerprint"), fingerprint)
        row = ledger.get("run")
        if not isinstance(row, dict):
            raise V11PanelControlError(f"{path} has no run")
    else:
        row = _execute_one(arm, instance, seed, device)
        ledger = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "run_key": f"model-seed-{arm.model_seed}:instance-seed-{seed}",
            "input_contract": input_contract,
            "input_fingerprint": fingerprint,
            "run": row,
        }
        ledger["ledger_sha256"] = _digest_json(ledger)
        _atomic_json(path, ledger)
    _require_equal("ledger model seed", row.get("model_seed"), arm.model_seed)
    _require_equal("ledger instance seed", row.get("instance_seed"), seed)
    _require_equal("ledger instance id", row.get("instance_id"), instance.instance_id)
    _require_equal("ledger schedule id", row.get("schedule_id"), instance.schedule_id)
    return row, ledger


def _descriptive_differences(control: Mapping, reference: Mapping) -> dict:
    metrics = (
        "mean_dense_return",
        "mean_absolute_error",
        "mean_steps",
        "physical_rehandles_per_100_deliveries",
        "mean_signed_deviation",
        "within_target_window_rate",
    )
    return {
        "orientation": "control_minus_v2_2_episode80",
        "inferential_claim": False,
        "reason": (
            "common EpisodeInstances but different learned-method replication and "
            "V2.2 stochastic action-realization design"
        ),
        "differences": {
            metric: float(control[metric]) - float(reference[metric])
            for metric in metrics
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    fields = (
        "model_seed",
        "instance_seed",
        "instance_id",
        "schedule_id",
        "checkpoint_weight_episode",
        "checkpoint_sha256",
        "dense_objective_return",
        "mean_absolute_error",
        "mean_signed_deviation",
        "within_target_window_rate",
        "steps",
        "delivery_count",
        "physical_storage_relocations",
        "physical_storage_relocations_per_100_deliveries",
        "strict_method_success",
        "completion_rate",
        "all_selected_candidates_exact_safe",
        "illegal_drops",
        "fallbacks",
        "evaluation_epsilon",
        "evaluation_learning",
        "method_failure_reason",
        "delivery_deviations",
    )
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                values = {field: row.get(field) for field in fields}
                values["delivery_deviations"] = json.dumps(
                    values["delivery_deviations"], separators=(",", ":")
                )
                writer.writerow(values)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen three-seed V1.1 control on V2.2's opened 85xxx panel"
    )
    parser.add_argument("--training-dirs", nargs=3, type=Path, required=True)
    parser.add_argument("--v2-2-source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume-existing", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = _build_parser().parse_args(argv)
    _validate_protocol_constants()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_existing:
        raise FileExistsError(
            f"{output_dir} is nonempty; pass --resume-existing only for this exact run"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    v22_source = _validate_v22_source(args.v2_2_source_dir.resolve())
    source_paths = dict(v22_source["paths"])
    for training_dir in args.training_dirs:
        directory = training_dir.resolve()
        for filename in ("training-contract.json", "training-summary.json", "best.pth", "latest.pth"):
            source_paths[f"v1.1/{directory.name}/{filename}"] = directory / filename
    source_hashes_before = _source_file_hashes(source_paths)

    bundles = pareto._load_training_bundles(args.training_dirs)
    _require_equal("authenticated model-seed order", [bundle.model_seed for bundle in bundles], list(MODEL_SEEDS))
    arms = tuple(bundle.selected for bundle in bundles)
    for arm in arms:
        _require_equal("selected checkpoint group", arm.policy_group, pareto.SELECTED_BEST)
        _require_equal("selected checkpoint eligibility", arm.readiness["deployment_checkpoint_eligible"], True)
        _require_equal("selected checkpoint diagnostic flag", arm.diagnostic_only, False)

    canonical_payload = arms[0].payload
    _require_equal("V1.1/V2.2 environment", canonical_payload["environment"], {
        **v22_source["contract"]["environment"], "max_steps": FROZEN_MAX_STEPS
    })
    instances, instance_manifest = _regenerate_instances(
        output_dir,
        canonical_payload,
        v22_source,
        resume_existing=args.resume_existing,
    )
    device = resolve_device(args.device)

    protocol_manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "development_only_already_opened_85000_85011",
        "performance_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "training_dirs": tuple(str(bundle.training_dir) for bundle in bundles),
        "model_seeds": MODEL_SEEDS,
        "evaluation_seeds": EVALUATION_SEEDS,
        "rollout_count": len(MODEL_SEEDS) * len(EVALUATION_SEEDS),
        "evaluation_policy": "deterministic_greedy_epsilon_zero",
        "evaluation_epsilon": 0.0,
        "evaluation_learning": False,
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "gamma": FROZEN_GAMMA,
        "reported_return": "undiscounted_dense_operational_return",
        "max_steps": FROZEN_MAX_STEPS,
        "device": str(device),
        "source_v2_2": {
            "directory": str(args.v2_2_source_dir.resolve()),
            "contract_sha256": v22_source["contract_sha256"],
            "instance_manifest_sha256": v22_source["instance_manifest_sha256"],
            "episode80_ledger_sha256": v22_source["reference_ledger_sha256"],
        },
        "checkpoints": tuple(
            {
                "model_seed": arm.model_seed,
                "path": str(arm.checkpoint_path.resolve()),
                "sha256": arm.checkpoint_sha256,
                "deployment_policy_digest": arm.deployment_policy_digest,
                "selected_checkpoint_episode": arm.checkpoint_weight_episode,
                "deployment_checkpoint_eligible": arm.readiness["deployment_checkpoint_eligible"],
            }
            for arm in arms
        ),
        "instance_manifest_sha256": instance_manifest["manifest_sha256"],
        "source_artifact_sha256_before": source_hashes_before,
    }
    manifest_path = output_dir / PROTOCOL_MANIFEST_FILENAME
    if manifest_path.is_file():
        if not args.resume_existing:
            raise FileExistsError(manifest_path)
        _require_equal("control protocol manifest", _load_json(manifest_path), protocol_manifest)
    else:
        _atomic_json(manifest_path, protocol_manifest)

    instance_records = {int(record["instance_seed"]): record for record in instance_manifest["instances"]}
    rows = []
    ledger_manifest = []
    for arm in arms:
        for seed in EVALUATION_SEEDS:
            row, ledger = _load_or_execute(
                output_dir=output_dir,
                arm=arm,
                instance=instances[seed],
                instance_record=instance_records[seed],
                device=device,
            )
            rows.append(row)
            ledger_path = _ledger_path(output_dir, arm.model_seed, seed)
            ledger_manifest.append(
                {
                    "run_key": ledger["run_key"],
                    "path": str(ledger_path.resolve()),
                    "sha256": _sha256_file(ledger_path),
                    "ledger_sha256": ledger["ledger_sha256"],
                }
            )
            print(
                f"[V1.1 seed {arm.model_seed}] instance={seed} "
                f"DenseR={float(row['dense_objective_return']):8.2f} "
                f"MAE={float(row['mean_absolute_error']):6.2f} "
                f"steps={int(row['steps']):4d} "
                f"rehandles={int(row['physical_storage_relocations']):2d}",
                flush=True,
            )

    observed_grid = {(int(row["model_seed"]), int(row["instance_seed"])) for row in rows}
    expected_grid = {(model_seed, seed) for model_seed in MODEL_SEEDS for seed in EVALUATION_SEEDS}
    _require_equal("completed control grid", observed_grid, expected_grid)

    rows_by_seed = defaultdict(list)
    for row in rows:
        rows_by_seed[int(row["model_seed"])].append(row)
    seed_summaries = {
        seed: _summarize_rows(rows_by_seed[seed]) for seed in MODEL_SEEDS
    }
    equal_seed = _equal_seed_aggregate(seed_summaries)
    equal_seed["delivery_position_metrics"] = _delivery_position_summary(
        rows, window=FROZEN_OBJECTIVE_SPEC.window
    )
    reference = v22_source["reference_summary"]
    comparisons = {
        "equal_seed_control": _descriptive_differences(equal_seed, reference),
        "by_model_seed": {
            str(seed): _descriptive_differences(seed_summaries[seed], reference)
            for seed in MODEL_SEEDS
        },
    }

    source_hashes_after = _source_file_hashes(source_paths)
    _require_equal("source artifacts unchanged", source_hashes_after, source_hashes_before)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "development_only": True,
        "performance_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "source_artifact_sha256_before": source_hashes_before,
        "source_artifact_sha256_after": source_hashes_after,
        "source_artifacts_unchanged": True,
        "authenticated_v1_1_bundle_count": len(bundles),
        "authenticated_selected_checkpoint_count": len(arms),
        "regenerated_instance_count": len(instances),
        "rollout_count": len(rows),
        "evaluation_epsilon": 0.0,
        "evaluation_learning": False,
        "all_complete_exact_safe": all(
            summary["all_complete_exact_safe"] for summary in seed_summaries.values()
        ),
        "ledger_manifest": ledger_manifest,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "development_only_already_opened_85000_85011",
        "performance_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "reported_return": "undiscounted_dense_operational_return",
        "policy_gamma": FROZEN_GAMMA,
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "evaluation_policy": "deterministic_greedy_epsilon_zero_no_learning",
        "model_seed_summaries": {str(seed): seed_summaries[seed] for seed in MODEL_SEEDS},
        "equal_seed_aggregate": equal_seed,
        "v2_2_nearest_candidate_look_reference": {
            "checkpoint_episode": V22_REFERENCE_EPISODE,
            "method_version": V22_METHOD_VERSION,
            "policy": "stochastic_induced_nested_regularized_lagrangian",
            "rollouts": 48,
            **reference,
        },
        "descriptive_comparisons": comparisons,
        "comparison_interpretation": (
            "Development control only. Common EpisodeInstances support descriptive "
            "alignment, but this is not a final-panel result or an inferential claim."
        ),
        "run_count": len(rows),
        "runs": tuple(rows),
    }
    _write_csv(output_dir / RUNS_FILENAME, rows)
    _atomic_json(output_dir / REPORT_FILENAME, report)
    _atomic_json(output_dir / AUDIT_FILENAME, audit)
    print(json.dumps({
        "status": "complete",
        "run_count": len(rows),
        "all_complete_exact_safe": audit["all_complete_exact_safe"],
        "equal_seed_aggregate": equal_seed,
        "v2_2_episode80_reference": reference,
        "report": str((output_dir / REPORT_FILENAME).resolve()),
        "audit": str((output_dir / AUDIT_FILENAME).resolve()),
        "final_86xxx_panel_opened": False,
    }, indent=2, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main()
