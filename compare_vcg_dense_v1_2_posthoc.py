#!/usr/bin/env python3
"""Post-hoc VCG-Dense V1.2 checkpoint-selection development diagnostic.

This experiment changes no trained parameters.  It applies one frozen
relocation-aware checkpoint-selection rule to each V1.1 validation history,
authenticates the preserved parameter artifacts, and evaluates the chosen
policies on the already-opened 80000--80029 EpisodeInstances.  Existing
selected-best and enhanced-GA rows are reused; no baseline is executed.

Every output is post-hoc, development-only, diagnostic, and deployment-
ineligible.  Opening or promoting a sealed panel is outside this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence
import uuid

import numpy as np

from benchmark_viability_critic_priority import _make_env, _sha256_file
from compare_vcg_dense_pareto import (
    EVALUATION_SEEDS,
    FROZEN_OBJECTIVE_SPEC,
    MAX_STEPS,
    PolicyArm,
    PROTOCOL as SOURCE_PROTOCOL,
    _compact_deployment_payload,
    _digest_json,
    _json_safe,
    _ledger_manifest_entry,
    _load_json,
    _load_or_execute_run,
    _load_training_bundles,
    _policy_digest,
    _run_policy_once,
    _shared_execution_contract,
)
from benchmark_viability_critic_priority import _load_controller_checkpoint
from compare_viability_graph_baselines import ENHANCED_GA_METHOD
from example.episode_instance import EpisodeInstance
from track_b_urgency_evaluate import resolve_device
from train_vcg_dense_proper import (
    FROZEN_TOTAL_EPISODES,
    FROZEN_VALIDATION_SEEDS,
)


PROTOCOL = "vcg_dense_v1_2_posthoc_relocation_selection_development_v1"
PROTOCOL_SCHEMA_VERSION = 1
SELECTION_RULE = (
    "strict_full_validation_and_mae_within_reference_plus_2_then_"
    "minimum_relocations_then_max_dense_return_then_lower_mae_then_earlier_v1"
)
MAE_MARGIN = 2.0
EXPECTED_CHOSEN_EPISODES = {0: 475, 1: 425, 2: 500}
EXPECTED_ARTIFACTS = {
    0: "best-candidate-slot-a.pth",
    1: "best-candidate-slot-a.pth",
    2: "latest.pth",
}
EXPECTED_ARTIFACT_SHA256 = {
    0: "0a357ef92c19dbdec8f8eb193d549e002658760e17b0155b43a85bfcd96dcbec",
    1: "d7920a78314ab68d7238f0004978e150ce2ca431aeec7cf0187314f964a2ac73",
    2: "da6e7828f3d2f1862483d2899c29691413fd5e539d51fbd3a03fc7940e2b201b",
}
SOURCE_SELECTED_METHODS = {
    seed: f"vcg_dense_seed{seed}_selected_best" for seed in (0, 1, 2)
}
SOURCE_SEED2_FINAL_METHOD = "vcg_dense_seed2_episode500_final"
POSTHOC_GROUP = "posthoc_v1_2_selection_diagnostic"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_RNG_SEED = 20_260_809

PROTOCOL_MANIFEST_FILENAME = "posthoc-protocol-manifest.json"
RESULTS_FILENAME = "posthoc-runs.csv"
REPORT_FILENAME = "posthoc-report.json"
AUDIT_FILENAME = "posthoc-audit.json"

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
        raise ValueError(
            f"V1.2 post-hoc {name} mismatch: "
            f"observed={observed!r}, expected={expected!r}"
        )


def _finite(record: Mapping, field: str) -> float:
    value = float(record[field])
    if not math.isfinite(value):
        raise ValueError(f"validation record has nonfinite {field}")
    return value


def _full_validation_eligible(record: Mapping) -> bool:
    if record.get("deployment_eligible") is not True:
        return False
    summary = record.get("summary")
    if not isinstance(summary, Mapping):
        return False
    try:
        return bool(
            int(summary["episodes"]) == len(FROZEN_VALIDATION_SEEDS)
            and tuple(int(seed) for seed in summary["instance_seeds"])
            == FROZEN_VALIDATION_SEEDS
            and float(summary["strict_method_success_rate"]) == 1.0
            and float(summary["completion_rate"]) == 1.0
            and int(summary["delivery_count"])
            == len(FROZEN_VALIDATION_SEEDS) * 8
            and not tuple(summary.get("method_failures", ()))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _select_posthoc_record(summary: Mapping, model_seed: int) -> dict:
    """Apply the frozen V1.2 rule to one complete validation history."""

    history = summary.get("validation_history")
    reference = summary.get("best_validation_record")
    if not isinstance(history, list) or not isinstance(reference, Mapping):
        raise ValueError("training summary has no complete validation history")
    expected_episodes = tuple(range(25, FROZEN_TOTAL_EPISODES + 1, 25))
    observed_episodes = tuple(int(item["checkpoint_episode"]) for item in history)
    if observed_episodes != expected_episodes:
        raise ValueError("validation history does not cover the frozen selection horizon")
    matches = [
        item for item in history if _json_safe(item) == _json_safe(reference)
    ]
    if len(matches) != 1:
        raise ValueError("current finalized reference is not unique in validation history")
    if not _full_validation_eligible(reference):
        raise ValueError("current finalized reference is not strict/full/eligible")
    reference_mae = _finite(reference["summary"], "mean_absolute_error")
    threshold = reference_mae + MAE_MARGIN

    table = []
    feasible = []
    for record in history:
        record_summary = record.get("summary", {})
        mae = _finite(record_summary, "mean_absolute_error")
        relocations = _finite(record_summary, "relocations_per_100_deliveries")
        dense_return = _finite(record_summary, "mean_dense_rescored_return")
        full = _full_validation_eligible(record)
        within_margin = mae <= threshold
        accepted = bool(full and within_margin)
        row = {
            "checkpoint_episode": int(record["checkpoint_episode"]),
            "deployment_eligible_strict_full_validation": full,
            "mean_absolute_error": mae,
            "mae_threshold": threshold,
            "within_mae_margin": within_margin,
            "relocations_per_100_deliveries": relocations,
            "mean_dense_rescored_return": dense_return,
            "feasible": accepted,
        }
        table.append(row)
        if accepted:
            feasible.append((record, row))
    if not feasible:
        raise ValueError("V1.2 selection has no feasible checkpoint")
    chosen_record, chosen_row = min(
        feasible,
        key=lambda item: (
            item[1]["relocations_per_100_deliveries"],
            -item[1]["mean_dense_rescored_return"],
            item[1]["mean_absolute_error"],
            item[1]["checkpoint_episode"],
        ),
    )
    expected = EXPECTED_CHOSEN_EPISODES[int(model_seed)]
    if int(chosen_record["checkpoint_episode"]) != expected:
        raise ValueError(
            f"frozen V1.2 choice drifted for seed {model_seed}: "
            f"{chosen_record['checkpoint_episode']} != {expected}"
        )
    return {
        "model_seed": int(model_seed),
        "selection_rule": SELECTION_RULE,
        "mae_margin": MAE_MARGIN,
        "reference_record": reference,
        "reference_episode": int(reference["checkpoint_episode"]),
        "reference_mae": reference_mae,
        "mae_threshold": threshold,
        "chosen_record": chosen_record,
        "chosen_episode": int(chosen_record["checkpoint_episode"]),
        "selection_table": tuple(table),
    }


def _history_record(payload: Mapping, episode: int) -> Mapping:
    matches = [
        record
        for record in payload.get("validation_history", ())
        if int(record.get("checkpoint_episode", -1)) == int(episode)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"checkpoint validation history has {len(matches)} records for episode {episode}"
        )
    return matches[0]


def _authenticate_chosen_artifact(bundle, selection: Mapping) -> tuple[PolicyArm, dict]:
    model_seed = bundle.model_seed
    chosen_episode = int(selection["chosen_episode"])
    artifact_name = EXPECTED_ARTIFACTS[model_seed]
    artifact_path = bundle.training_dir / artifact_name
    payload = _load_controller_checkpoint(artifact_path)
    sha256 = _sha256_file(artifact_path)
    _require_equal(
        "pre-frozen artifact SHA-256",
        sha256,
        EXPECTED_ARTIFACT_SHA256[model_seed],
    )
    _require_equal("artifact model seed", payload.get("model_seed"), model_seed)
    _require_equal(
        "artifact resume contract",
        payload.get("resume_contract"),
        bundle.selected.payload.get("resume_contract"),
    )
    _require_equal(
        "artifact execution contract",
        _shared_execution_contract(payload),
        bundle.shared_execution_contract,
    )
    _require_equal("artifact completed episode", payload.get("completed_training_episodes"), chosen_episode)
    _require_equal("artifact deployment eligibility", payload.get("deployment_checkpoint_eligible"), False)
    _require_equal("artifact test-panel state", payload.get("test_panels_opened"), False)
    chosen_record = selection["chosen_record"]
    history_record = _history_record(payload, chosen_episode)
    _require_equal("artifact validation-history record", history_record, chosen_record)

    if model_seed in (0, 1):
        _require_equal("candidate artifact filename", artifact_path.name, "best-candidate-slot-a.pth")
        _require_equal("candidate checkpoint role", payload.get("checkpoint_role"), "best_candidate_unfinalized")
        _require_equal("candidate trainer resumable", payload.get("trainer_resumable"), False)
        _require_equal("candidate protocol complete", payload.get("protocol_training_complete"), False)
        _require_equal("candidate selection finalized", payload.get("selection_finalized_after_total_episodes"), False)
        _require_equal("candidate selected episode", payload.get("selected_checkpoint_episode"), chosen_episode)
        _require_equal("candidate best record", payload.get("best_validation_record"), chosen_record)
        _require_equal("candidate slot metadata", payload.get("best_candidate_checkpoint_filename"), artifact_name)
    else:
        _require_equal("latest artifact filename", artifact_path.name, "latest.pth")
        _require_equal("latest checkpoint role", payload.get("checkpoint_role"), "latest_resumable")
        _require_equal("latest trainer resumable", payload.get("trainer_resumable"), True)
        _require_equal("latest protocol complete", payload.get("protocol_training_complete"), True)
        _require_equal("latest selection finalized", payload.get("selection_finalized_after_total_episodes"), True)
        _require_equal("latest final validation record", history_record, chosen_record)

    digest = _policy_digest(payload)
    arm = PolicyArm(
        method_id=f"vcg_dense_seed{model_seed}_posthoc_v1_2_ep{chosen_episode}",
        policy_group=POSTHOC_GROUP,
        model_seed=model_seed,
        checkpoint_variant="posthoc_preserved_artifact",
        checkpoint_weight_episode=chosen_episode,
        checkpoint_path=artifact_path,
        checkpoint_sha256=sha256,
        deployment_policy_digest=digest,
        primary_analysis=False,
        diagnostic_only=True,
        readiness={
            "deployment_checkpoint_eligible": False,
            "interpretation": "posthoc_development_selection_diagnostic_only",
        },
        payload=_compact_deployment_payload(payload),
    )
    record = {
        "model_seed": model_seed,
        "checkpoint_episode": chosen_episode,
        "checkpoint_path": str(artifact_path.resolve()),
        "checkpoint_sha256": sha256,
        "deployment_policy_digest": digest,
        "checkpoint_role": payload.get("checkpoint_role"),
        "trainer_resumable": payload.get("trainer_resumable"),
        "deployment_checkpoint_eligible": False,
        "posthoc": True,
        "development_only": True,
        "diagnostic_only": True,
        "selection_record_sha256": _digest_json(chosen_record),
    }
    return arm, record


def _parse_source_row(row: Mapping[str, str]) -> dict:
    integer_fields = (
        "model_seed",
        "checkpoint_weight_episode",
        "instance_seed",
        "steps",
        "delivery_count",
        "relocations",
        "obstructive_moves",
        "illegal_drops",
        "invalid_assignments",
        "infeasible_epochs",
        "fallbacks",
    )
    float_fields = (
        "return",
        "legacy_rescored_return",
        "dense_rescored_return",
        "success",
        "strict_method_success",
        "completion_rate",
        "first_two_mean_absolute_error",
        "positions_three_plus_mean_absolute_error",
        "mean_signed_deviation",
        "mean_absolute_error",
        "mean_tardiness",
        "mean_earliness",
        "within_target_window_rate",
        "p90_tardiness",
        "p90_absolute_error",
    )
    output = dict(row)
    for name in integer_fields:
        output[name] = int(row[name]) if row.get(name, "") != "" else None
    for name in float_fields:
        output[name] = float(row[name]) if row.get(name, "") != "" else None
    return output


def _load_source_rows(source_dir: Path, bundles) -> tuple[dict, dict]:
    source_dir = source_dir.resolve()
    protocol_manifest_path = source_dir / "protocol-manifest.json"
    instance_manifest_path = source_dir / "instance-manifest.json"
    runs_path = source_dir / "pareto-runs.csv"
    audit_path = source_dir / "pareto-audit.json"
    protocol_manifest = _load_json(protocol_manifest_path)
    instance_manifest = _load_json(instance_manifest_path)
    source_audit = _load_json(audit_path)
    _require_equal("source protocol", protocol_manifest.get("protocol"), SOURCE_PROTOCOL)
    _require_equal("source evaluation seeds", protocol_manifest.get("evaluation_seeds"), EVALUATION_SEEDS)
    _require_equal("source audit protocol", source_audit.get("protocol"), SOURCE_PROTOCOL)
    source_hashes = {
        "protocol_manifest": _sha256_file(protocol_manifest_path),
        "instance_manifest": _sha256_file(instance_manifest_path),
        "pareto_runs": _sha256_file(runs_path),
        "pareto_audit": _sha256_file(audit_path),
    }

    by_seed = {bundle.model_seed: bundle for bundle in bundles}
    checkpoint_records = protocol_manifest.get("checkpoint_records", ())
    for seed, method in SOURCE_SELECTED_METHODS.items():
        matches = [
            record
            for record in checkpoint_records
            if record.get("method_id") == method and record.get("executed") is True
        ]
        if len(matches) != 1:
            raise ValueError(f"source manifest lacks selected-best seed {seed}")
        _require_equal(
            f"source selected-best seed-{seed} SHA-256",
            matches[0].get("checkpoint_sha256"),
            by_seed[seed].selected.checkpoint_sha256,
        )
        _require_equal(
            f"source selected-best seed-{seed} policy digest",
            matches[0].get("deployment_policy_digest"),
            by_seed[seed].selected.deployment_policy_digest,
        )

    rows = [_parse_source_row(row) for row in csv.DictReader(runs_path.open(encoding="utf-8"))]
    wanted_methods = set(SOURCE_SELECTED_METHODS.values()) | {
        SOURCE_SEED2_FINAL_METHOD,
        ENHANCED_GA_METHOD,
    }
    selected = [row for row in rows if row["method_id"] in wanted_methods]
    expected_counts = {
        **{method: len(EVALUATION_SEEDS) for method in SOURCE_SELECTED_METHODS.values()},
        SOURCE_SEED2_FINAL_METHOD: len(EVALUATION_SEEDS),
        ENHANCED_GA_METHOD: len(EVALUATION_SEEDS),
    }
    for method, expected in expected_counts.items():
        method_rows = [row for row in selected if row["method_id"] == method]
        if len(method_rows) != expected:
            raise ValueError(f"source row grid for {method} is incomplete")
        if len({row["instance_id"] for row in method_rows}) != expected:
            raise ValueError(f"source row grid for {method} has duplicate instances")
        for row in method_rows:
            if row["strict_method_success"] != 1.0 or row["completion_rate"] != 1.0:
                raise ValueError(f"source method {method} is not strict/full eligible")
            if row["relocations"] != row["obstructive_moves"]:
                raise ValueError("source relocation semantics diverged")
            if method == ENHANCED_GA_METHOD and (
                int(row["invalid_assignments"]) != 0
                or int(row["fallbacks"]) != 0
                or int(row["illegal_drops"]) != 0
                or bool(row.get("method_failure_reason"))
            ):
                raise ValueError("source enhanced GA is not method-eligible")

    ledger_manifest = source_audit.get("run_ledger_manifest", ())
    ledger_by_key = {record["run_key"]: record for record in ledger_manifest}
    reused_keys = []
    for method in wanted_methods:
        for seed in EVALUATION_SEEDS:
            key = f"{method}:{seed}"
            record = ledger_by_key.get(key)
            if not isinstance(record, Mapping):
                raise ValueError(f"source ledger manifest lacks {key}")
            ledger_path = Path(record["path"])
            if not ledger_path.is_file() or _sha256_file(ledger_path) != record["sha256"]:
                raise ValueError(f"source ledger hash mismatch for {key}")
            reused_keys.append(record)
    source = {
        "source_dir": str(source_dir),
        "source_hashes": source_hashes,
        "source_execution_device": protocol_manifest.get("execution_device"),
        "instance_manifest": instance_manifest,
        "reused_ledger_records": tuple(reused_keys),
    }
    return {row["method_id"]: [item for item in selected if item["method_id"] == row["method_id"]] for row in selected}, source


def _load_source_instances(source: Mapping, payload: Mapping) -> dict[int, EpisodeInstance]:
    manifest = source["instance_manifest"]
    _require_equal("source instance seeds", manifest["contract"]["seeds"], EVALUATION_SEEDS)
    env = _make_env(payload)
    instances = {}
    for seed in EVALUATION_SEEDS:
        record = manifest["instances"][str(seed)]
        path = Path(record["path"])
        if not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise ValueError(f"source EpisodeInstance hash mismatch for seed {seed}")
        instance = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
        instance.validate_for(env)
        _require_equal("source instance id", instance.instance_id, record["instance_id"])
        _require_equal("source schedule id", instance.schedule_id, record["schedule_id"])
        instances[seed] = instance
    return instances


def _posthoc_row(row: Mapping, arm: PolicyArm, *, reused_from: Optional[str]) -> dict:
    output = dict(row)
    output.update(
        {
            "protocol": PROTOCOL,
            "method": arm.method_id,
            "method_id": arm.method_id,
            "policy_group": POSTHOC_GROUP,
            "model_seed": arm.model_seed,
            "checkpoint_variant": arm.checkpoint_variant,
            "checkpoint_weight_episode": arm.checkpoint_weight_episode,
            "checkpoint_path": str(arm.checkpoint_path.resolve()),
            "checkpoint_sha256": arm.checkpoint_sha256,
            "deployment_policy_digest": arm.deployment_policy_digest,
            "primary_analysis": False,
            "diagnostic_only": True,
            "posthoc": True,
            "development_only": True,
            "deployment_checkpoint_eligible": False,
            "checkpoint_readiness_interpretation": (
                "posthoc_development_selection_diagnostic_only"
            ),
            "execution_reused": reused_from is not None,
            "derived_from_exact_policy_duplicate": reused_from is not None,
            "derived_from_method_id": reused_from,
        }
    )
    output.pop("method_audit", None)
    return output


def _bootstrap_ci(values: Sequence[float], rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    estimates = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, (0.025, 0.975))]


def _summary(rows: Sequence[Mapping]) -> dict:
    deliveries = sum(int(row["delivery_count"]) for row in rows)
    return {
        "episodes": len(rows),
        "strict_method_success_rate": fmean(float(row["strict_method_success"]) for row in rows),
        "completion_rate": fmean(float(row["completion_rate"]) for row in rows),
        "mean_dense_rescored_return": fmean(float(row["dense_rescored_return"]) for row in rows),
        "mean_absolute_error": fmean(float(row["mean_absolute_error"]) for row in rows),
        "first_two_mean_absolute_error": fmean(float(row["first_two_mean_absolute_error"]) for row in rows),
        "positions_three_plus_mean_absolute_error": fmean(
            float(row["positions_three_plus_mean_absolute_error"])
            for row in rows
        ),
        "mean_tardiness": fmean(float(row["mean_tardiness"]) for row in rows),
        "mean_earliness": fmean(float(row["mean_earliness"]) for row in rows),
        "within_target_window_rate": fmean(float(row["within_target_window_rate"]) for row in rows),
        "mean_steps": fmean(float(row["steps"]) for row in rows),
        "relocations_per_100_deliveries": (
            100.0 * sum(int(row["relocations"]) for row in rows) / deliveries
        ),
    }


def _paired(chosen: Sequence[Mapping], comparison: Sequence[Mapping], label: str) -> dict:
    left = {row["instance_id"]: row for row in chosen}
    right = {row["instance_id"]: row for row in comparison}
    if set(left) != set(right) or len(left) != len(EVALUATION_SEEDS):
        raise ValueError(f"paired grid mismatch against {label}")
    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    definitions = {
        "dense_return_advantage": ("dense_rescored_return", 1.0),
        "absolute_error_reduction": ("mean_absolute_error", -1.0),
        "first_two_absolute_error_reduction": (
            "first_two_mean_absolute_error",
            -1.0,
        ),
        "tardiness_reduction": ("mean_tardiness", -1.0),
        "within_window_advantage": ("within_target_window_rate", 1.0),
        "step_reduction": ("steps", -1.0),
        "relocation_reduction": ("relocations", -1.0),
    }
    metrics = {}
    for name, (field, chosen_orientation) in definitions.items():
        deltas = []
        for instance_id in sorted(left):
            chosen_value = float(left[instance_id][field])
            comparison_value = float(right[instance_id][field])
            delta = (
                chosen_value - comparison_value
                if chosen_orientation > 0
                else comparison_value - chosen_value
            )
            deltas.append(delta)
        metrics[name] = {
            "n": len(deltas),
            "mean": float(fmean(deltas)),
            "bootstrap_95_ci": _bootstrap_ci(deltas, rng),
            "per_instance": tuple(deltas),
        }
    return {
        "comparison": label,
        "metric_orientation": "positive_favors_posthoc_v1_2",
        "instance_ids": tuple(sorted(left)),
        "metrics": metrics,
    }


AGGREGATE_DELTA_FIELDS = {
    "dense_return_change": ("dense_rescored_return", 1.0),
    "mae_cost": ("mean_absolute_error", 1.0),
    "relocations_saving_per_100_deliveries": ("relocations", -12.5),
    "first_two_mae_cost": ("first_two_mean_absolute_error", 1.0),
    "positions_three_plus_mae_cost": (
        "positions_three_plus_mean_absolute_error",
        1.0,
    ),
    "tardiness_cost": ("mean_tardiness", 1.0),
    "earliness_cost": ("mean_earliness", 1.0),
    "within_window_change": ("within_target_window_rate", 1.0),
    "steps_change": ("steps", 1.0),
}


def _aggregate_contrast(
    rows: Sequence[Mapping],
    source_rows: Mapping[str, Sequence[Mapping]],
    *,
    comparator: str,
) -> dict:
    """Training-seed-aware paired contrasts on the shared 30-instance panel."""

    chosen_by_seed = {
        seed: {row["instance_id"]: row for row in rows if row["model_seed"] == seed}
        for seed in (0, 1, 2)
    }
    comparator_by_seed = {}
    for seed in (0, 1, 2):
        method = (
            SOURCE_SELECTED_METHODS[seed]
            if comparator == "selected_best"
            else ENHANCED_GA_METHOD
        )
        comparator_by_seed[seed] = {
            row["instance_id"]: row for row in source_rows[method]
        }
        if set(chosen_by_seed[seed]) != set(comparator_by_seed[seed]):
            raise ValueError(
                f"aggregate paired grid mismatch for seed {seed} vs {comparator}"
            )

    rng = np.random.default_rng(BOOTSTRAP_RNG_SEED)
    metrics = {}
    for metric, (field, multiplier) in AGGREGATE_DELTA_FIELDS.items():
        matrices = []
        instance_ids = tuple(sorted(chosen_by_seed[0]))
        for seed in (0, 1, 2):
            if tuple(sorted(chosen_by_seed[seed])) != instance_ids:
                raise ValueError("training seeds do not share the same instance grid")
            deltas = []
            for instance_id in instance_ids:
                chosen_value = float(chosen_by_seed[seed][instance_id][field])
                comparator_value = float(
                    comparator_by_seed[seed][instance_id][field]
                )
                if metric == "relocations_saving_per_100_deliveries":
                    # Eight deliveries are required by the strict/full gate;
                    # 100/8 = 12.5 converts paired count saving to the rate.
                    delta = (comparator_value - chosen_value) * 12.5
                else:
                    delta = (chosen_value - comparator_value) * multiplier
                deltas.append(delta)
            matrices.append(deltas)
        matrix = np.asarray(matrices, dtype=float)
        per_seed = matrix.mean(axis=1)
        estimate = float(per_seed.mean())
        sample_sd = float(per_seed.std(ddof=1))
        t_half = 4.302652729911275 * sample_sd / math.sqrt(3.0)

        per_instance_average = matrix.mean(axis=0)
        instance_indices = rng.integers(
            0,
            len(instance_ids),
            size=(BOOTSTRAP_SAMPLES, len(instance_ids)),
        )
        conditional = per_instance_average[instance_indices].mean(axis=1)

        seed_indices = rng.integers(0, 3, size=(BOOTSTRAP_SAMPLES, 3))
        crossed_instance_indices = rng.integers(
            0,
            len(instance_ids),
            size=(BOOTSTRAP_SAMPLES, len(instance_ids)),
        )
        crossed = matrix[
            seed_indices[:, :, None],
            crossed_instance_indices[:, None, :],
        ].mean(axis=(1, 2))
        metrics[metric] = {
            "n_training_seeds": 3,
            "n_paired_instances_per_seed": len(instance_ids),
            "per_training_seed": tuple(
                {
                    "model_seed": seed,
                    "panel_mean_delta": float(per_seed[seed]),
                }
                for seed in (0, 1, 2)
            ),
            "mean_across_training_seeds": estimate,
            "sample_sd_across_training_seeds": sample_sd,
            "t_degrees_of_freedom": 2,
            "training_seed_t95_ci": [estimate - t_half, estimate + t_half],
            "conditional_30_instance_bootstrap_95_ci": [
                float(value) for value in np.quantile(conditional, (0.025, 0.975))
            ],
            "crossed_seed_instance_bootstrap_sensitivity_95_ci": [
                float(value) for value in np.quantile(crossed, (0.025, 0.975))
            ],
        }
    return {
        "comparator": comparator,
        "estimand": (
            "mean paired panel delta per independent training seed, then "
            "mean across three training seeds"
        ),
        "sign_convention": {
            "dense_return_change": "positive favors posthoc",
            "mae_cost": "negative favors posthoc",
            "relocations_saving_per_100_deliveries": "positive favors posthoc",
            "first_two_mae_cost": "negative favors posthoc",
            "positions_three_plus_mae_cost": "negative favors posthoc",
            "tardiness_cost": "negative favors posthoc",
            "earliness_cost": "negative favors posthoc",
            "within_window_change": "positive favors posthoc",
            "steps_change": "negative favors posthoc",
        },
        "metrics": metrics,
    }


def _build_report(
    rows: Sequence[dict],
    source_rows: Mapping[str, Sequence[dict]],
    selections: Sequence[Mapping],
    artifacts: Sequence[Mapping],
) -> dict:
    by_seed = {seed: [row for row in rows if row["model_seed"] == seed] for seed in (0, 1, 2)}
    per_seed = []
    for seed in (0, 1, 2):
        chosen = by_seed[seed]
        selected = source_rows[SOURCE_SELECTED_METHODS[seed]]
        ga = source_rows[ENHANCED_GA_METHOD]
        per_seed.append(
            {
                "model_seed": seed,
                "chosen_episode": EXPECTED_CHOSEN_EPISODES[seed],
                "posthoc_summary": _summary(chosen),
                "selected_best_summary": _summary(selected),
                "enhanced_ga_summary": _summary(ga),
                "versus_selected_best": _paired(chosen, selected, "selected_best"),
                "versus_enhanced_ga": _paired(chosen, ga, ENHANCED_GA_METHOD),
            }
        )
    seed_summaries = [item["posthoc_summary"] for item in per_seed]
    across = {}
    for field in (
        "mean_dense_rescored_return",
        "mean_absolute_error",
        "first_two_mean_absolute_error",
        "positions_three_plus_mean_absolute_error",
        "mean_tardiness",
        "mean_earliness",
        "within_target_window_rate",
        "mean_steps",
        "relocations_per_100_deliveries",
    ):
        values = tuple(float(item[field]) for item in seed_summaries)
        across[field] = {
            "n_training_seeds": len(values),
            "per_training_seed": values,
            "mean": float(fmean(values)),
            "population_std": float(np.std(values)),
        }
    return {
        "protocol": PROTOCOL,
        "scope": "posthoc_development_diagnostic_only",
        "performance_claim_authorized": False,
        "deployment_checkpoint_eligible": False,
        "selection_rule": SELECTION_RULE,
        "mae_margin": MAE_MARGIN,
        "selection_decisions": tuple(selections),
        "artifact_manifest": tuple(artifacts),
        "per_training_seed": tuple(per_seed),
        "across_training_seeds": across,
        "aggregate_contrasts": {
            "candidate_vs_selected_best": _aggregate_contrast(
                rows, source_rows, comparator="selected_best"
            ),
            "candidate_vs_enhanced_ga": _aggregate_contrast(
                rows, source_rows, comparator="enhanced_ga"
            ),
        },
        "guardrails": {
            "no_new_baseline_runs": True,
            "source_selected_best_rows_reused": True,
            "source_enhanced_ga_rows_reused": True,
            "seed2_episode500_rows_reused": True,
            "physical_new_rollouts": 2 * len(EVALUATION_SEEDS),
            "analysis_rows": 3 * len(EVALUATION_SEEDS),
            "sealed_panels_opened": False,
            "posthoc_not_deployment_eligible": True,
        },
        "interpretation": (
            "This diagnoses whether a prespecified relocation-aware selection "
            "rule exposes useful capacity in already-trained V1.1 checkpoints. "
            "It is post-hoc development evidence, not a V1.2 deployment claim."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dirs", nargs=3, type=Path, required=True)
    parser.add_argument("--source-pareto-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-existing", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    source_dir = args.source_pareto_dir.resolve()
    if output_dir == source_dir:
        raise ValueError("post-hoc output must not modify the source Pareto directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_existing:
        raise FileExistsError(
            f"{output_dir} is nonempty; pass --resume-existing or choose a new directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = _load_training_bundles(args.training_dirs)
    selections = []
    arms = []
    artifacts = []
    for bundle in bundles:
        summary = _load_json(bundle.training_dir / "training-summary.json")
        selection = _select_posthoc_record(summary, bundle.model_seed)
        arm, artifact = _authenticate_chosen_artifact(bundle, selection)
        selections.append(selection)
        arms.append(arm)
        artifacts.append(artifact)

    source_rows, source = _load_source_rows(source_dir, bundles)
    instances = _load_source_instances(source, bundles[0].selected.payload)
    device = resolve_device(args.device)
    _require_equal("execution device", str(device), source["source_execution_device"])
    source_seed2_rows = source_rows[SOURCE_SEED2_FINAL_METHOD]
    seed2_arm = arms[2]
    source_seed2_manifest = [
        record
        for record in source["reused_ledger_records"]
        if record["run_key"].startswith(f"{SOURCE_SEED2_FINAL_METHOD}:")
    ]
    if len(source_seed2_manifest) != len(EVALUATION_SEEDS):
        raise ValueError("source seed-2 final ledger grid is incomplete")
    for row in source_seed2_rows:
        _require_equal("source seed-2 final model seed", row["model_seed"], 2)
        _require_equal("source seed-2 final policy digest", row.get("deployment_policy_digest"), seed2_arm.deployment_policy_digest)
        _require_equal("source seed-2 final checkpoint SHA-256", row.get("checkpoint_sha256"), seed2_arm.checkpoint_sha256)

    protocol_manifest = {
        "protocol": PROTOCOL,
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "scope": "posthoc_development_diagnostic_only",
        "performance_claim_authorized": False,
        "deployment_checkpoint_eligible": False,
        "selection_rule": SELECTION_RULE,
        "mae_margin": MAE_MARGIN,
        "expected_chosen_episodes": EXPECTED_CHOSEN_EPISODES,
        "artifact_manifest": artifacts,
        "source": {
            key: value for key, value in source.items() if key != "instance_manifest"
        },
        "source_instance_manifest_sha256": _digest_json(source["instance_manifest"]),
        "execution_device": str(device),
        "evaluation_seeds": EVALUATION_SEEDS,
        "physical_execution_seeds": {0: EVALUATION_SEEDS, 1: EVALUATION_SEEDS, 2: ()},
        "seed2_reuse_source_method": SOURCE_SEED2_FINAL_METHOD,
    }
    protocol_manifest_path = output_dir / PROTOCOL_MANIFEST_FILENAME
    if protocol_manifest_path.is_file():
        _require_equal("protocol manifest", _load_json(protocol_manifest_path), protocol_manifest)
    else:
        if args.resume_existing and (output_dir / "run-ledger").exists():
            raise ValueError("cannot resume run ledger without post-hoc protocol manifest")
        _atomic_json(protocol_manifest_path, protocol_manifest)

    block_count = int(bundles[0].selected.payload["environment"]["number_blocks"])
    shared_sha = _digest_json(bundles[0].shared_execution_contract)
    rows = []
    ledger_manifest = []
    for arm in arms[:2]:
        for seed in EVALUATION_SEEDS:
            instance = instances[seed]
            instance_record = source["instance_manifest"]["instances"][str(seed)]
            input_contract = {
                "protocol": PROTOCOL,
                "selection_rule": SELECTION_RULE,
                "mae_margin": MAE_MARGIN,
                "method_id": arm.method_id,
                "model_seed": arm.model_seed,
                "checkpoint_sha256": arm.checkpoint_sha256,
                "deployment_policy_digest": arm.deployment_policy_digest,
                "instance_seed": seed,
                "instance_id": instance.instance_id,
                "schedule_id": instance.schedule_id,
                "instance_sha256": instance_record["sha256"],
                "source_instance_manifest_sha256": _digest_json(source["instance_manifest"]),
                "shared_execution_contract_sha256": shared_sha,
                "execution_device": str(device),
                "max_steps": MAX_STEPS,
                "timing_objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
            }
            row, ledger = _load_or_execute_run(
                output_dir=output_dir,
                method_id=arm.method_id,
                seed=seed,
                input_contract=input_contract,
                execute=lambda arm=arm, instance=instance, seed=seed: _run_policy_once(
                    arm,
                    instance,
                    seed,
                    block_count=block_count,
                    device=device,
                ),
            )
            row = _posthoc_row(row, arm, reused_from=None)
            rows.append(row)
            ledger_manifest.append(
                _ledger_manifest_entry(output_dir, arm.method_id, seed, ledger)
            )
            print(
                f"[{arm.method_id}] seed={seed} "
                f"DenseR={row['dense_rescored_return']:.2f} "
                f"MAE={row['mean_absolute_error']:.3f} "
                f"reloc={row['relocations']}",
                flush=True,
            )

    for source_row in source_seed2_rows:
        rows.append(
            _posthoc_row(
                source_row,
                seed2_arm,
                reused_from=SOURCE_SEED2_FINAL_METHOD,
            )
        )
    expected_rows = 3 * len(EVALUATION_SEEDS)
    if len(rows) != expected_rows or len(
        {(row["method_id"], row["instance_id"]) for row in rows}
    ) != expected_rows:
        raise RuntimeError("post-hoc analysis grid is incomplete")
    for row in rows:
        if row["strict_method_success"] != 1.0 or row["completion_rate"] != 1.0:
            raise RuntimeError("post-hoc chosen policy failed strict/full evaluation")
        if int(row["relocations"]) != int(row["obstructive_moves"]):
            raise RuntimeError("post-hoc relocation semantics diverged")

    report = _build_report(rows, source_rows, selections, artifacts)
    report.update(
        {
            "source_provenance": {
                key: value for key, value in source.items() if key != "instance_manifest"
            },
            "execution_device": str(device),
            "new_execution_run_count": 2 * len(EVALUATION_SEEDS),
            "reused_seed2_run_count": len(EVALUATION_SEEDS),
            "rows": tuple(rows),
        }
    )
    audit = {
        "protocol": PROTOCOL,
        "protocol_manifest": protocol_manifest,
        "new_run_ledger_manifest": tuple(ledger_manifest),
        "reused_seed2_ledger_manifest": tuple(source_seed2_manifest),
        "selection_decisions": tuple(selections),
    }

    fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    results_path = output_dir / RESULTS_FILENAME
    temporary = results_path.parent / f".{results_path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        name: (
                            json.dumps(_json_safe(row.get(name)), separators=(",", ":"))
                            if isinstance(row.get(name), (dict, tuple, list))
                            else row.get(name)
                        )
                        for name in fieldnames
                    }
                )
        temporary.replace(results_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    _atomic_json(output_dir / REPORT_FILENAME, report)
    _atomic_json(output_dir / AUDIT_FILENAME, audit)
    print(f"Runs: {results_path}", flush=True)
    print(f"Report: {output_dir / REPORT_FILENAME}", flush=True)
    print(f"Audit: {output_dir / AUDIT_FILENAME}", flush=True)
    return report


if __name__ == "__main__":
    main()
