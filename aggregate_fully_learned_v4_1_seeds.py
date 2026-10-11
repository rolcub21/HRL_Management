#!/usr/bin/env python3
"""Audit and aggregate the three-seed v4.1 confirmation experiment.

The utility is intentionally fail-closed.  A training seed is admitted only
when ``latest.pth`` proves completion of the complete 500-episode v4.1
curriculum and ``best.pth`` is an authenticated deployment artifact from the
same run.  Ordinary and contention evaluations must use an identical ordered
paired-instance panel across all three training seeds.  The baseline input is
the complete hierarchy factorial, not a hand-selected summary.

Example::

    python aggregate_fully_learned_v4_1_seeds.py \
      --seed-run results/v4-1-seed0 results/eval-seed0-standard.json \
        results/eval-seed0-contention.json \
      --seed-run results/v4-1-seed1 results/eval-seed1-standard.json \
        results/eval-seed1-contention.json \
      --seed-run results/v4-1-seed2 results/eval-seed2-standard.json \
        results/eval-seed2-contention.json \
      --baseline-factorial results/hierarchy/hierarchy-full.json \
      --output results/v4-1-three-seed-confirmation.json
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping, Sequence

import torch

from fully_learned_hierarchy import (
    FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION,
    validate_fully_learned_checkpoint_metadata,
)
from PSLAP.checkpoint_identity import selector_deployment_digest
from track_b_fully_learned_evaluate import validate_deployment_payload
from train_fully_learned_track_b import (
    CURRICULUM_CONTRACT,
    FULLY_LEARNED_TRAINER_VERSION,
    METHOD,
)


PROTOCOL_VERSION = "fully_learned_v4_1_three_training_seed_confirmation_v1"
EXPECTED_PHASE_COUNTS = {
    "imitation": 50,
    "temporal": 150,
    "spatial": 50,
    "joint": 250,
}
EXPECTED_TOTAL_EPISODES = sum(EXPECTED_PHASE_COUNTS.values())
CONDITIONS = ("standard", "contention")
BASELINE_SOURCES = ("reg_selector_v5", "nearest_free", "dynamic_pslap")
GEOMETRY_BY_CONDITION = {
    "standard": "ordinary",
    "contention": "egress_constrained",
}

# Fields that bind best and latest to one run and bind the three runs to one
# experimental design.  Ephemeral progress and model-selection fields are
# deliberately excluded.
PAIR_IDENTITY_FIELDS = (
    "fully_learned_checkpoint_schema_version",
    "fully_learned_trainer_version",
    "curriculum_contract",
    "controller_architecture",
    "controller_action_interface",
    "network_architecture",
    "replay_version",
    "backup_version",
    "candidate_feature_version",
    "macro_return_contract",
    "truncation_contract",
    "failure_contract",
    "failure_penalty_contract",
    "model_selection_contract",
    "quality_gate_contract",
    "training_seed",
    "python_hash_seed",
    "training_instance_seed_base",
    "training_lambda",
    "training_mu",
    "geometry",
    "phase_episode_counts",
    "fully_learned_config",
    "selector_deployment_digest",
    "max_steps",
    "max_defer_steps",
    "lookahead_margin_steps",
    "learning_rate",
    "spatial_learning_rate",
    "batch_size",
    "buffer_size",
    "update_every",
    "updates_per_macro",
    "joint_updates_per_macro",
    "joint_td_warmup_decisions",
    "gamma",
    "target_tau",
    "grad_clip",
    "reward_scale",
    "failure_penalty",
    "replay_sampling",
    "validation_seeds",
    "validation_steps",
    "eval_every",
    "target_window",
)
DESIGN_IDENTITY_FIELDS = tuple(
    field
    for field in PAIR_IDENTITY_FIELDS
    if field not in {"training_seed", "python_hash_seed"}
)


@dataclass(frozen=True)
class SeedInput:
    training_dir: Path
    standard_evaluation: Path
    contention_evaluation: Path


def _load_json_mapping(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain a mapping: {path}")
    return value


def _load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, found {actual!r}")


def _embedded_selector_digest(payload: Mapping, *, label: str) -> str:
    selector = payload.get("selector_checkpoint")
    if not isinstance(selector, dict):
        raise ValueError(f"{label} does not embed its REG teacher checkpoint")
    observed = selector_deployment_digest(selector)
    _require_equal(
        payload.get("selector_deployment_digest"),
        observed,
        f"{label} selector deployment digest",
    )
    return observed


def _validate_latest(payload: dict, *, label: str) -> None:
    _require_equal(
        payload.get("fully_learned_checkpoint_schema_version"),
        FULLY_LEARNED_CHECKPOINT_SCHEMA_VERSION,
        f"{label} schema",
    )
    _require_equal(
        payload.get("fully_learned_trainer_version"),
        FULLY_LEARNED_TRAINER_VERSION,
        f"{label} trainer version",
    )
    _require_equal(
        payload.get("curriculum_contract"),
        CURRICULUM_CONTRACT,
        f"{label} curriculum",
    )
    validate_fully_learned_checkpoint_metadata(payload)
    _require_equal(payload.get("checkpoint_kind"), "resumable_latest", f"{label} kind")
    _require_equal(payload.get("resumable"), True, f"{label} resumable flag")
    _require_equal(payload.get("training_phase"), "joint", f"{label} phase")
    _require_equal(
        payload.get("phase_episode_counts"),
        EXPECTED_PHASE_COUNTS,
        f"{label} phase episode counts",
    )
    _require_equal(
        payload.get("completed_training_episodes"),
        EXPECTED_TOTAL_EPISODES,
        f"{label} completed episodes",
    )
    for phase, expected in EXPECTED_PHASE_COUNTS.items():
        _require_equal(
            payload.get(f"completed_{phase}_episodes"),
            expected,
            f"{label} completed {phase} episodes",
        )
    if payload.get("quality_gate_blocked") is not False:
        raise ValueError(f"{label} is quality-gate blocked or lacks gate state")
    state = payload.get("agent_checkpoint_state")
    if not isinstance(state, dict):
        raise ValueError(f"{label} lacks resumable agent state")
    missing_state = {"optimizer", "replay", "rng_state"} - set(state)
    if missing_state:
        raise ValueError(f"{label} lacks resumable state: {sorted(missing_state)}")
    if "rng_state" not in payload:
        raise ValueError(f"{label} lacks top-level RNG state")

    history = payload.get("training_history")
    if not isinstance(history, list) or len(history) != EXPECTED_TOTAL_EPISODES:
        raise ValueError(
            f"{label} must contain all {EXPECTED_TOTAL_EPISODES} training episodes"
        )
    episodes = [item.get("episode") if isinstance(item, dict) else None for item in history]
    _require_equal(
        episodes,
        list(range(1, EXPECTED_TOTAL_EPISODES + 1)),
        f"{label} training history episode order",
    )
    expected_phases = []
    for phase, count in EXPECTED_PHASE_COUNTS.items():
        expected_phases.extend([phase] * count)
    _require_equal(
        [item.get("phase") for item in history],
        expected_phases,
        f"{label} training history phase order",
    )
    validations = payload.get("validation_history")
    if not isinstance(validations, list) or not validations:
        raise ValueError(f"{label} has no validation history")
    final_validation = validations[-1]
    if not isinstance(final_validation, dict):
        raise ValueError(f"{label} final validation is malformed")
    _require_equal(
        final_validation.get("episode"),
        EXPECTED_TOTAL_EPISODES,
        f"{label} final validation episode",
    )
    gate = final_validation.get("quality_gate")
    enforcement = final_validation.get("quality_gate_enforcement")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError(f"{label} final validation did not pass its quality gate")
    if not isinstance(enforcement, dict) or enforcement.get("blocked") is not False:
        raise ValueError(f"{label} final validation was blocked")


def _validate_best(payload: dict, *, label: str) -> None:
    validate_deployment_payload(payload)
    if int(payload.get("completed_training_episodes", 0)) > EXPECTED_TOTAL_EPISODES:
        raise ValueError(f"{label} was selected outside the 500-episode run")
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"{label} lacks deployment validation")
    safety = {
        "strict_method_success_rate": 1.0,
        "reservation_integrity_rate": 1.0,
        "selector_independent_replay_isolation_rate": 1.0,
        "total_invalid_assignments": 0,
        "total_fallbacks": 0,
        "total_reservation_invalidations": 0,
    }
    for field, expected in safety.items():
        _require_equal(validation.get(field), expected, f"{label} validation {field}")
    if validation.get("method_failures") not in ([], ()):
        raise ValueError(f"{label} deployment validation contains method failures")


def _identity(payload: Mapping, fields: Sequence[str]) -> dict:
    return {field: payload.get(field) for field in fields}


def authenticate_training_run(training_dir: Path) -> dict:
    best_path = training_dir / "best.pth"
    latest_path = training_dir / "latest.pth"
    best = _load_checkpoint(best_path)
    latest = _load_checkpoint(latest_path)
    _validate_best(best, label=str(best_path))
    _validate_latest(latest, label=str(latest_path))
    _embedded_selector_digest(best, label=str(best_path))
    _embedded_selector_digest(latest, label=str(latest_path))
    _require_equal(
        _identity(best, PAIR_IDENTITY_FIELDS),
        _identity(latest, PAIR_IDENTITY_FIELDS),
        f"{training_dir} best/latest run identity",
    )
    _require_equal(
        tuple(best.get("best_score", ())),
        tuple(latest.get("best_score", ())),
        f"{training_dir} best score",
    )
    if int(best.get("completed_training_episodes", 0)) > int(
        latest["completed_training_episodes"]
    ):
        raise ValueError(f"{training_dir} best checkpoint is later than latest")
    seed = latest.get("training_seed")
    if not isinstance(seed, int):
        raise ValueError(f"{training_dir} training_seed must be an integer")
    _require_equal(latest.get("python_hash_seed"), seed, f"{training_dir} hash seed")
    return {
        "training_seed": seed,
        "best_path": best_path.resolve(),
        "latest_path": latest_path.resolve(),
        "best_sha256": _sha256(best_path),
        "latest_sha256": _sha256(latest_path),
        "best_episode": int(best["completed_training_episodes"]),
        "best_score": list(best["best_score"]),
        "design_identity": _identity(latest, DESIGN_IDENTITY_FIELDS),
    }


def _ordered_signature(runs: Sequence[Mapping]) -> tuple[tuple, ...]:
    return tuple(
        (
            int(run["eval_seed"]),
            str(run["instance_id"]),
            str(run["schedule_id"]),
        )
        for run in runs
    )


def _selector_count(run: Mapping, field: str) -> int:
    audit = run.get("selector_audit", {})
    return int(audit.get(field, 0)) if isinstance(audit, dict) else 0


def _scheduler_count(run: Mapping, field: str) -> int:
    audit = run.get("scheduler_audit", {})
    return int(audit.get(field, 0)) if isinstance(audit, dict) else 0


def summarize_episode_runs(runs: Sequence[Mapping]) -> dict:
    if not runs:
        raise ValueError("cannot summarize an empty run panel")
    values = lambda key: [float(run[key]) for run in runs]
    relocations = [int(run.get("obstructive_moves", 0)) for run in runs]
    failures = [
        {"eval_seed": run.get("eval_seed"), "reason": run.get("method_failure_reason")}
        for run in runs
        if run.get("method_failure_reason") is not None
    ]
    return {
        "episodes": len(runs),
        "mean_return": statistics.fmean(values("return")),
        "return_std": statistics.stdev(values("return")) if len(runs) > 1 else 0.0,
        "success_rate": statistics.fmean(values("success")),
        "strict_method_success_rate": statistics.fmean(values("strict_method_success")),
        "reservation_integrity_rate": statistics.fmean(values("reservation_integrity")),
        "selector_independent_replay_isolation_rate": statistics.fmean(
            float(run.get("selector_independent_replay_isolated", 1.0))
            for run in runs
        ),
        "minimum_episode_success": min(values("success")),
        "minimum_episode_strict_method_success": min(
            values("strict_method_success")
        ),
        "minimum_episode_reservation_integrity": min(
            values("reservation_integrity")
        ),
        "minimum_episode_selector_independent_replay_isolation": min(
            float(run.get("selector_independent_replay_isolated", 1.0))
            for run in runs
        ),
        "mean_absolute_error": statistics.fmean(values("mean_absolute_error")),
        "mean_tardiness": statistics.fmean(values("mean_tardiness")),
        "within_target_window_rate": statistics.fmean(values("within_target_window_rate")),
        "mean_steps": statistics.fmean(values("steps")),
        "total_relocations": sum(relocations),
        "episodes_with_relocation": sum(value > 0 for value in relocations),
        "max_episode_relocations": max(relocations),
        "total_invalid_assignments": sum(
            _selector_count(run, "invalid_assignment_count") for run in runs
        ),
        "total_fallbacks": sum(_selector_count(run, "fallback_count") for run in runs),
        "total_reservation_invalidations": sum(
            _scheduler_count(run, "reservation_invalidation_count") for run in runs
        ),
        "total_illegal_drops": sum(int(run.get("illegal_drops", 0)) for run in runs),
        "total_truncated": sum(float(run.get("truncated", 0.0)) != 0.0 for run in runs),
        "method_failures": failures,
    }


def _validate_claimed_summary(claimed: Mapping, observed: Mapping, *, label: str) -> None:
    fields = (
        "episodes",
        "mean_return",
        "return_std",
        "success_rate",
        "strict_method_success_rate",
        "reservation_integrity_rate",
        "selector_independent_replay_isolation_rate",
        "mean_absolute_error",
        "mean_steps",
        "total_invalid_assignments",
        "total_fallbacks",
        "total_reservation_invalidations",
    )
    for field in fields:
        if field not in claimed:
            raise ValueError(f"{label} claimed summary lacks {field}")
        left, right = claimed[field], observed[field]
        if isinstance(right, float):
            if not math.isclose(float(left), right, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"{label} claimed summary {field} is inconsistent")
        elif left != right:
            raise ValueError(f"{label} claimed summary {field} is inconsistent")


def authenticate_evaluation(
    path: Path,
    *,
    condition: str,
    training: Mapping,
) -> dict:
    payload = _load_json_mapping(path)
    protocol = payload.get("protocol")
    runs = payload.get("runs")
    if not isinstance(protocol, dict) or not isinstance(runs, list) or not runs:
        raise ValueError(f"malformed evaluation artifact: {path}")
    _require_equal(protocol.get("track"), "B_complete_system", f"{path} track")
    _require_equal(protocol.get("method"), METHOD, f"{path} method")
    _require_equal(protocol.get("checkpoint_kind"), "deployment_best", f"{path} kind")
    _require_equal(protocol.get("strict_no_fallback"), True, f"{path} strict protocol")
    checkpoint_path = Path(str(protocol.get("checkpoint", ""))).resolve()
    _require_equal(checkpoint_path, training["best_path"], f"{path} checkpoint path")
    protocol_seeds = tuple(int(seed) for seed in protocol.get("eval_seeds", ()))
    run_seeds = tuple(int(run.get("eval_seed")) for run in runs)
    _require_equal(run_seeds, protocol_seeds, f"{path} ordered evaluation seeds")
    if len(set(run_seeds)) != len(run_seeds):
        raise ValueError(f"{path} contains duplicate evaluation seeds")
    signature = _ordered_signature(runs)
    if any(not instance_id or not schedule_id for _, instance_id, schedule_id in signature):
        raise ValueError(f"{path} contains empty pairing identifiers")
    for run in runs:
        _require_equal(run.get("method"), METHOD, f"{path} run method")
        _require_equal(run.get("track"), "B", f"{path} run track")
        _require_equal(
            Path(str(run.get("checkpoint", ""))).resolve(),
            training["best_path"],
            f"{path} run checkpoint",
        )
        _require_equal(
            int(run.get("checkpoint_episode", -1)),
            training["best_episode"],
            f"{path} checkpoint episode",
        )
    observed = summarize_episode_runs(runs)
    claimed = payload.get("summary")
    if not isinstance(claimed, dict):
        raise ValueError(f"{path} lacks its evaluator summary")
    _validate_claimed_summary(claimed, observed, label=str(path))
    if condition == "standard":
        axes = tuple(protocol.get("observed_generalization_axes", ()))
        if axes:
            raise ValueError(f"{path} standard evaluation is not exact-domain")
    elif condition == "contention":
        axes = tuple(protocol.get("observed_generalization_axes", ()))
        if "exit_width" not in axes:
            raise ValueError(f"{path} contention evaluation does not shift exit width")
    else:
        raise ValueError(f"unknown evaluation condition: {condition}")
    return {
        "path": path.resolve(),
        "sha256": _sha256(path),
        "signature": signature,
        "runs": runs,
        "summary": observed,
        "protocol": protocol,
    }


def _factorial_cell_key(run: Mapping) -> tuple:
    return (
        run.get("geometry_condition"),
        run.get("scheduler_variant"),
        run.get("assignment_commitment"),
        run.get("assignment_source"),
    )


def authenticate_baseline_factorial(path: Path, expected_signatures: Mapping[str, tuple]) -> dict:
    payload = _load_json_mapping(path)
    protocol = payload.get("protocol")
    runs = payload.get("runs")
    if not isinstance(protocol, dict) or not isinstance(runs, list):
        raise ValueError(f"malformed baseline factorial: {path}")
    _require_equal(
        protocol.get("protocol_version"),
        "track_b_reserved_hierarchy_factorial_v1",
        f"{path} protocol",
    )
    expected_seeds = tuple(seed for seed, _, _ in expected_signatures["standard"])
    _require_equal(
        tuple(int(seed) for seed in protocol.get("seeds", ())),
        expected_seeds,
        f"{path} ordered seeds",
    )
    expected_cells = set()
    for geometry in GEOMETRY_BY_CONDITION.values():
        for commitment in ("post_pickup_recompute", "decision_epoch_reserved"):
            for source in BASELINE_SOURCES:
                expected_cells.add((geometry, "duration_aware", commitment, source))
            expected_cells.add((geometry, "due_only", commitment, "reg_selector_v5"))
    grouped: dict[tuple, dict[int, Mapping]] = defaultdict(dict)
    for run in runs:
        key = _factorial_cell_key(run)
        if key not in expected_cells:
            raise ValueError(f"{path} contains unexpected factorial cell {key}")
        seed = int(run.get("eval_seed"))
        if seed in grouped[key]:
            raise ValueError(f"{path} duplicates factorial cell {key}, seed {seed}")
        grouped[key][seed] = run
    _require_equal(set(grouped), expected_cells, f"{path} factorial cells")
    for key, by_seed in grouped.items():
        _require_equal(tuple(by_seed), expected_seeds, f"{path} seed order for {key}")

    selected = {}
    cell_summaries = []
    for condition, geometry in GEOMETRY_BY_CONDITION.items():
        expected = expected_signatures[condition]
        for source in BASELINE_SOURCES:
            key = (geometry, "duration_aware", "decision_epoch_reserved", source)
            cell_runs = [grouped[key][seed] for seed in expected_seeds]
            signature = _ordered_signature(cell_runs)
            _require_equal(signature, expected, f"{path} pairing for {condition}/{source}")
            selected[(condition, source)] = cell_runs
            cell_summaries.append(
                {
                    "condition": condition,
                    "assignment_source": source,
                    **summarize_episode_runs(cell_runs),
                }
            )

    # The complete factorial must be paired within geometry and must preserve
    # the stochastic schedule across geometries.
    for index, seed in enumerate(expected_seeds):
        standard_instance = expected_signatures["standard"][index][1]
        contention_instance = expected_signatures["contention"][index][1]
        schedule = expected_signatures["standard"][index][2]
        if standard_instance == contention_instance:
            raise ValueError(f"{path} geometry instances coincide for seed {seed}")
        _require_equal(
            expected_signatures["contention"][index][2],
            schedule,
            f"{path} cross-geometry schedule for seed {seed}",
        )
        for key, by_seed in grouped.items():
            run = by_seed[seed]
            geometry = key[0]
            expected_instance = (
                standard_instance if geometry == "ordinary" else contention_instance
            )
            _require_equal(run.get("instance_id"), expected_instance, f"{path} instance")
            _require_equal(run.get("schedule_id"), schedule, f"{path} schedule")
    return {
        "path": path.resolve(),
        "sha256": _sha256(path),
        "protocol_version": protocol["protocol_version"],
        "selected": selected,
        "cell_summaries": cell_summaries,
        "factorial_cell_count": len(grouped),
        "factorial_run_count": len(runs),
    }


def _distribution(values: Iterable[float]) -> dict:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "n_training_seeds": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }


def _across_training_seeds(per_seed: Sequence[Mapping], condition: str) -> dict:
    summaries = [item[condition] for item in per_seed]
    metric_fields = (
        "mean_return",
        "mean_absolute_error",
        "mean_tardiness",
        "within_target_window_rate",
        "mean_steps",
        "success_rate",
        "strict_method_success_rate",
        "reservation_integrity_rate",
        "selector_independent_replay_isolation_rate",
        "total_relocations",
    )
    distributions = {
        field: _distribution(summary[field] for summary in summaries)
        for field in metric_fields
    }
    additive_safety = (
        "total_invalid_assignments",
        "total_fallbacks",
        "total_reservation_invalidations",
        "total_illegal_drops",
        "total_truncated",
    )
    return {
        "training_seed_distribution": distributions,
        "relocation_total_across_training_seeds": sum(
            summary["total_relocations"] for summary in summaries
        ),
        "episodes_with_relocation_across_training_seeds": sum(
            summary["episodes_with_relocation"] for summary in summaries
        ),
        "worst_safety": {
            "minimum_strict_method_success_rate": min(
                summary["strict_method_success_rate"] for summary in summaries
            ),
            "minimum_reservation_integrity_rate": min(
                summary["reservation_integrity_rate"] for summary in summaries
            ),
            "minimum_success_rate": min(summary["success_rate"] for summary in summaries),
            "minimum_selector_independent_replay_isolation_rate": min(
                summary["selector_independent_replay_isolation_rate"]
                for summary in summaries
            ),
            "minimum_episode_success": min(
                summary["minimum_episode_success"] for summary in summaries
            ),
            "minimum_episode_strict_method_success": min(
                summary["minimum_episode_strict_method_success"]
                for summary in summaries
            ),
            "minimum_episode_reservation_integrity": min(
                summary["minimum_episode_reservation_integrity"]
                for summary in summaries
            ),
            "minimum_episode_selector_independent_replay_isolation": min(
                summary["minimum_episode_selector_independent_replay_isolation"]
                for summary in summaries
            ),
            **{
                f"maximum_seed_{field}": max(summary[field] for summary in summaries)
                for field in additive_safety
            },
            **{
                f"{field}_across_training_seeds": sum(
                    summary[field] for summary in summaries
                )
                for field in additive_safety
            },
            "method_failures": [
                failure
                for summary in summaries
                for failure in summary["method_failures"]
            ],
        },
    }


def _paired_comparison(learned: Sequence[Mapping], baseline: Sequence[Mapping]) -> dict:
    if _ordered_signature(learned) != _ordered_signature(baseline):
        raise ValueError("paired comparison received different instance panels")
    rows = []
    for left, right in zip(learned, baseline):
        rows.append(
            {
                "eval_seed": int(left["eval_seed"]),
                "instance_id": left["instance_id"],
                "schedule_id": left["schedule_id"],
                "return_advantage": float(left["return"]) - float(right["return"]),
                "absolute_error_reduction": (
                    float(right["mean_absolute_error"])
                    - float(left["mean_absolute_error"])
                ),
                "relocation_reduction": (
                    int(right.get("obstructive_moves", 0))
                    - int(left.get("obstructive_moves", 0))
                ),
                "strict_success_advantage": (
                    float(left["strict_method_success"])
                    - float(right["strict_method_success"])
                ),
            }
        )
    return {
        "episodes": len(rows),
        "mean_return_advantage": statistics.fmean(row["return_advantage"] for row in rows),
        "mean_absolute_error_reduction": statistics.fmean(
            row["absolute_error_reduction"] for row in rows
        ),
        "total_relocation_reduction": sum(row["relocation_reduction"] for row in rows),
        "mean_strict_success_advantage": statistics.fmean(
            row["strict_success_advantage"] for row in rows
        ),
        "per_instance": rows,
    }


def aggregate(seed_inputs: Sequence[SeedInput], baseline_path: Path) -> dict:
    if len(seed_inputs) != 3:
        raise ValueError("the confirmation protocol requires exactly three training runs")
    authenticated = []
    for item in seed_inputs:
        training = authenticate_training_run(item.training_dir)
        evaluations = {
            "standard": authenticate_evaluation(
                item.standard_evaluation, condition="standard", training=training
            ),
            "contention": authenticate_evaluation(
                item.contention_evaluation, condition="contention", training=training
            ),
        }
        authenticated.append({"input": item, "training": training, "evaluations": evaluations})

    training_seeds = [item["training"]["training_seed"] for item in authenticated]
    if len(set(training_seeds)) != 3:
        raise ValueError(f"training seeds must be distinct, found {training_seeds}")
    if training_seeds != sorted(training_seeds):
        raise ValueError("--seed-run entries must be ordered by training_seed")
    reference_design = authenticated[0]["training"]["design_identity"]
    for item in authenticated[1:]:
        _require_equal(
            item["training"]["design_identity"],
            reference_design,
            "cross-training-seed experiment design",
        )

    expected_signatures = {}
    for condition in CONDITIONS:
        reference = authenticated[0]["evaluations"][condition]["signature"]
        for item in authenticated[1:]:
            _require_equal(
                item["evaluations"][condition]["signature"],
                reference,
                f"ordered {condition} seed/instance/schedule panel",
            )
        expected_signatures[condition] = reference
    standard = expected_signatures["standard"]
    contention = expected_signatures["contention"]
    _require_equal(
        tuple(seed for seed, _, _ in standard),
        tuple(seed for seed, _, _ in contention),
        "cross-condition ordered evaluation seeds",
    )
    _require_equal(
        tuple(schedule for _, _, schedule in standard),
        tuple(schedule for _, _, schedule in contention),
        "cross-condition ordered schedules",
    )
    if any(left[1] == right[1] for left, right in zip(standard, contention)):
        raise ValueError("standard and contention instance IDs must differ by geometry")

    baseline = authenticate_baseline_factorial(baseline_path, expected_signatures)
    per_seed = []
    paired = []
    for item in authenticated:
        seed = item["training"]["training_seed"]
        condition_summaries = {
            condition: item["evaluations"][condition]["summary"]
            for condition in CONDITIONS
        }
        per_seed.append(
            {
                "training_seed": seed,
                "best_episode": item["training"]["best_episode"],
                "best_score": item["training"]["best_score"],
                **condition_summaries,
            }
        )
        for condition in CONDITIONS:
            learned = item["evaluations"][condition]["runs"]
            for source in BASELINE_SOURCES:
                paired.append(
                    {
                        "training_seed": seed,
                        "condition": condition,
                        "baseline_assignment_source": source,
                        **_paired_comparison(learned, baseline["selected"][(condition, source)]),
                    }
                )

    across = {
        condition: _across_training_seeds(per_seed, condition)
        for condition in CONDITIONS
    }
    comparison_distributions = []
    for condition in CONDITIONS:
        for source in BASELINE_SOURCES:
            selected = [
                item
                for item in paired
                if item["condition"] == condition
                and item["baseline_assignment_source"] == source
            ]
            comparison_distributions.append(
                {
                    "condition": condition,
                    "baseline_assignment_source": source,
                    "return_advantage": _distribution(
                        item["mean_return_advantage"] for item in selected
                    ),
                    "absolute_error_reduction": _distribution(
                        item["mean_absolute_error_reduction"] for item in selected
                    ),
                    "relocation_reduction": _distribution(
                        item["total_relocation_reduction"] for item in selected
                    ),
                    "strict_success_advantage": _distribution(
                        item["mean_strict_success_advantage"] for item in selected
                    ),
                }
            )

    training_records = []
    evaluation_records = []
    for item in authenticated:
        training = item["training"]
        training_records.append(
            {
                key: training[key]
                for key in (
                    "training_seed",
                    "best_path",
                    "latest_path",
                    "best_sha256",
                    "latest_sha256",
                    "best_episode",
                    "best_score",
                )
            }
        )
        for condition in CONDITIONS:
            evaluation = item["evaluations"][condition]
            evaluation_records.append(
                {
                    "training_seed": training["training_seed"],
                    "condition": condition,
                    "path": evaluation["path"],
                    "sha256": evaluation["sha256"],
                }
            )
    return {
        "protocol": {
            "protocol_version": PROTOCOL_VERSION,
            "expected_training_seeds": 3,
            "expected_training_episodes_per_seed": EXPECTED_TOTAL_EPISODES,
            "expected_phase_episode_counts": EXPECTED_PHASE_COUNTS,
            "training_seed_is_unit_of_replication": True,
            "evaluation_instance_pairing_key": "ordered(eval_seed, instance_id, schedule_id)",
            "cross_geometry_pairing_key": "ordered(eval_seed, schedule_id)",
            "baseline_scope": "complete_reserved_hierarchy_factorial",
            "sample_standard_deviation": True,
        },
        "authenticated_training_runs": training_records,
        "authenticated_evaluations": evaluation_records,
        "pairing_audit": {
            "ordered_standard_signature": [list(item) for item in standard],
            "ordered_contention_signature": [list(item) for item in contention],
            "identical_across_training_seeds": True,
            "cross_geometry_eval_seed_and_schedule_match": True,
            "cross_geometry_instance_ids_differ": True,
        },
        "per_training_seed": per_seed,
        "across_training_seeds": across,
        "baseline_factorial": {
            key: baseline[key]
            for key in (
                "path",
                "sha256",
                "protocol_version",
                "factorial_cell_count",
                "factorial_run_count",
                "cell_summaries",
            )
        },
        "paired_baseline_comparisons_per_training_seed": paired,
        "paired_baseline_comparisons_across_training_seeds": comparison_distributions,
    }


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-run",
        action="append",
        nargs=3,
        metavar=("TRAINING_DIR", "STANDARD_EVAL", "CONTENTION_EVAL"),
        required=True,
        help="Repeat exactly three times, ordered by training seed.",
    )
    parser.add_argument("--baseline-factorial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.seed_run) != 3:
        parser.error("--seed-run must be supplied exactly three times")
    args.seed_run = tuple(
        SeedInput(*(Path(value) for value in values)) for values in args.seed_run
    )
    return args


def main(argv=None):
    args = parse_args(argv)
    result = aggregate(args.seed_run, args.baseline_factorial)
    safe = _json_safe(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(safe, indent=2, allow_nan=False) + "\n")
    console = {
        "training_seeds": [
            item["training_seed"] for item in safe["authenticated_training_runs"]
        ],
        "across_training_seeds": safe["across_training_seeds"],
        "paired_baseline_comparisons": safe[
            "paired_baseline_comparisons_across_training_seeds"
        ],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(console, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
