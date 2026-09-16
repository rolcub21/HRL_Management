#!/usr/bin/env python3
"""Development architecture screen for preference-conditioned vector VCG.

The screen compares three already-defined controller families on the exact
serialized 85000--85011 EpisodeInstances:

``A``
    The authenticated, historical VCG 1.1 + detached Monte-Carlo handling
    head.  Its completed report is reused; it is neither retrained nor rerun.

``B``
    A jointly trained vector critic whose consequence heads receive lambda.

``C``
    The matched joint vector critic with the lambda input masked.  Lambda
    still scalarizes its two consequences at action selection.

B and C use fixed terminal checkpoints, deterministic deployment, the same
exact fail-closed verifier, and the same liveness rule.  Any unsafe or
incomplete B/C row suppresses that architecture's entire aggregate rather
than permitting complete-case averaging.  This is an opened-panel
development screen, not a confirmation experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

import torch

import run_vcg_v11_nested_handling_pilot as pilot
import run_vcg_v11_nested_lambda_frontier_85k as historical
import train_vcg_preference_conditioned as trainer
from viability_graph_preference_conditioned import (
    PREFERENCE_CONDITIONED_CONTROLLER_ARCHITECTURE,
    PreferenceConditionedVectorAgent,
)


PROTOCOL = "vcg_preference_conditioned_architecture_screen_85k_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (0, 1, 2)
INSTANCE_SEEDS = tuple(range(85_000, 85_012))
LAMBDA_GRID = (0.0, 0.025, 0.05, 0.1, 0.2)
EXPECTED_BLOCKS = 8
EXPECTED_ROWS_PER_ARCHITECTURE = (
    len(MODEL_SEEDS) * len(INSTANCE_SEEDS) * len(LAMBDA_GRID)
)
ARCHITECTURES = ("conditioned", "unconditioned")
ARM_BY_ARCHITECTURE = {"conditioned": "B", "unconditioned": "C"}
HISTORICAL_ARM = "A"
TRAINING_EPISODES = 500
PREFERENCE_RELABELS = 1
CONTRACT_NAME = "architecture-screen-contract.json"
REPORT_NAME = "architecture-screen-report.json"
HISTORICAL_ROOT_NAME = (
    "vcg-v1-1-nested-lambda-frontier-85k-development"
)

_ANCHOR_CACHE: dict[str, tuple[object, object, dict, str]] = {}


class ArchitectureScreenError(RuntimeError):
    pass


def _lambda_key(value: float) -> str:
    return str(float(value))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ArchitectureScreenError(f"missing required file: {path}")
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Mapping) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _atomic_json(path: Path, value: Mapping) -> None:
    pilot._atomic_json(path, value)


def _historical_root(project_root: Path) -> Path:
    return project_root / "results" / HISTORICAL_ROOT_NAME


def _load_historical_anchor(project_root: Path):
    """Authenticate A without binding its artifacts to this new source tree."""

    key = str(project_root.resolve())
    cached = _ANCHOR_CACHE.get(key)
    if cached is not None:
        return cached
    anchor_root = _historical_root(project_root)
    # _authenticate() verifies A's completed parent chain.  Bind the already
    # written frontier report by raw SHA rather than regenerating it: older
    # reports can differ in harmless JSON representation after source-level
    # compatibility fixes, and this new screen must not rewrite A.
    report_path = anchor_root / historical.REPORT_NAME
    if not report_path.is_file():
        raise ArchitectureScreenError("historical A report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ArchitectureScreenError("historical A report is not complete")
    if tuple(float(x) for x in report.get("lambda_grid", ())) != LAMBDA_GRID:
        raise ArchitectureScreenError("historical A lambda grid changed")
    if int(report.get("row_count", -1)) != EXPECTED_ROWS_PER_ARCHITECTURE:
        raise ArchitectureScreenError("historical A row grid is incomplete")
    aggregate = report.get("aggregate_metrics")
    if not isinstance(aggregate, Mapping) or set(aggregate) != {
        _lambda_key(value) for value in LAMBDA_GRID
    }:
        raise ArchitectureScreenError("historical A aggregate is incomplete")
    arms, sources, _parent, _stability, _paths = historical._authenticate(
        project_root
    )
    if tuple(sorted(arms)) != MODEL_SEEDS:
        raise ArchitectureScreenError("historical A model seeds changed")
    if tuple(sorted(sources.instances)) != INSTANCE_SEEDS:
        raise ArchitectureScreenError("historical 85k instances changed")
    raw_sha256 = _sha256_file(report_path)
    result = (arms[0], sources, report, raw_sha256)
    _ANCHOR_CACHE[key] = result
    return result


def _source_hashes(project_root: Path) -> dict[str, str]:
    paths = {
        "screen_runner": Path(__file__).resolve(),
        "vector_controller": project_root
        / "viability_graph_preference_conditioned.py",
        "vector_trainer": project_root / "train_vcg_preference_conditioned.py",
    }
    return {name: _sha256_file(path) for name, path in sorted(paths.items())}


def _instance_identities(sources) -> list[dict]:
    identities = []
    for seed in INSTANCE_SEEDS:
        instance = sources.instances[seed]
        source = sources.identities[seed]
        canonical_sha = _sha256_bytes(instance.to_json().encode("utf-8"))
        expected_sha = source.get("episode_instance_sha256")
        if canonical_sha != expected_sha:
            raise ArchitectureScreenError(
                f"serialized EpisodeInstance identity changed for {seed}"
            )
        identities.append(
            {
                "instance_seed": seed,
                "episode_instance_id": instance.instance_id,
                "schedule_id": instance.schedule_id,
                "canonical_sha256": canonical_sha,
            }
        )
    return identities


def _training_dir(
    output_root: Path, architecture: str, model_seed: int
) -> Path:
    return output_root / "training" / architecture / f"seed-{model_seed}"


def _terminal_path(
    output_root: Path, architecture: str, model_seed: int
) -> Path:
    return _training_dir(output_root, architecture, model_seed) / "terminal.pth"


def _contract(project_root: Path, output_root: Path) -> dict:
    _arm, sources, anchor_report, anchor_sha = _load_historical_anchor(
        project_root
    )
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "development_only_already_opened_85000_85011",
        "architecture_arms": {
            "A": {
                "method": "historical_vcg_1_1_plus_detached_mc_handling_head",
                "execution": "authenticated_completed_report_reuse",
                "retraining": False,
                "reevaluation": False,
                "report": str(
                    (_historical_root(project_root) / historical.REPORT_NAME).resolve()
                ),
                "report_sha256": anchor_sha,
            },
            "B": {
                "architecture": "conditioned",
                "joint_vector_critic": True,
                "condition_on_preference": True,
            },
            "C": {
                "architecture": "unconditioned",
                "joint_vector_critic": True,
                "condition_on_preference": False,
            },
        },
        "training": {
            "model_seeds": list(MODEL_SEEDS),
            "episodes_per_seed": TRAINING_EPISODES,
            "paired_training_instances_and_preference_schedules": True,
            "lambda_stratified_behavior": True,
            "preference_fixed_within_episode": True,
            "preference_relabels": PREFERENCE_RELABELS,
            "additional_replay_relabeling_enabled": False,
            "checkpoint_rule": "fixed_terminal_episode_500_only",
            "checkpoint_selection": False,
            "teacher": False,
        },
        "evaluation": {
            "model_seeds": list(MODEL_SEEDS),
            "instance_seeds": list(INSTANCE_SEEDS),
            "lambda_grid": list(LAMBDA_GRID),
            "rows_per_new_architecture": EXPECTED_ROWS_PER_ARCHITECTURE,
            "fixed_terminal_checkpoint": True,
            "deterministic_fixed_lambda_policy": True,
            "fresh_agent_per_episode_instance": True,
            "same_serialized_episode_instances": True,
            "same_real_simulator_execution_path": True,
            "same_exact_fail_closed_verifier": True,
            "same_recovery_witness_liveness_rule": True,
            "complete_case_filtering_allowed": False,
            "whole_architecture_suppression_on_any_unsafe_or_incomplete_row": True,
            "aggregation": (
                "12 EpisodeInstances within seed, then equal weighting of "
                "the three model seeds"
            ),
            "metrics": [
                "dense_return",
                "mean_absolute_error",
                "steps",
                "physical_rehandles",
                "physical_rehandles_per_100",
            ],
            "primary_tradeoff_plane": [
                "mean_absolute_error",
                "physical_rehandles_per_100",
            ],
        },
        "advancement_gate": {
            "all_B_rows_strict_safe_complete": True,
            "B_lambda0_mean_dense_return_at_least_A": True,
            "B_lambda0_mean_absolute_error_no_worse_than_A": True,
            "B_lambda0p2_rehandles_per_100_no_worse_than_A": True,
            "B_at_least_three_sampled_nondominated_points": True,
            "B_aggregate_rehandles_nonincreasing_in_lambda": True,
        },
        "conditioning_claim_scope": (
            "descriptive paired architecture evidence only; bounded prediction "
            "calibration is deferred, so no calibrated Q_N(lambda) claim"
        ),
        "historical_anchor_aggregate_sha256": _canonical_sha256(
            dict(anchor_report["aggregate_metrics"])
        ),
        "instance_identities": _instance_identities(sources),
        "planned_terminal_checkpoints": {
            architecture: {
                str(seed): str(
                    _terminal_path(output_root, architecture, seed).resolve()
                )
                for seed in MODEL_SEEDS
            }
            for architecture in ARCHITECTURES
        },
        "source_sha256": _source_hashes(project_root),
    }
    return {**semantic, "contract_sha256": _canonical_sha256(semantic)}


def prepare(project_root: Path, output_root: Path) -> dict:
    contract = _contract(project_root, output_root)
    path = output_root / CONTRACT_NAME
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ArchitectureScreenError(
                "existing screen contract, A anchor, instances, or sources changed"
            )
    else:
        if output_root.exists() and any(output_root.iterdir()):
            raise ArchitectureScreenError(
                "nonempty output root has no architecture screen contract"
            )
        _atomic_json(path, contract)
    return {
        "status": "prepared",
        "scope": contract["scope"],
        "historical_A_reused": True,
        "new_training_runs": len(ARCHITECTURES) * len(MODEL_SEEDS),
        "new_evaluation_rows": len(ARCHITECTURES)
        * EXPECTED_ROWS_PER_ARCHITECTURE,
        "contract": str(path.resolve()),
    }


def _require_contract(project_root: Path, output_root: Path) -> dict:
    path = output_root / CONTRACT_NAME
    if not path.is_file():
        raise ArchitectureScreenError("run prepare first")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = _contract(project_root, output_root)
    if observed != expected:
        raise ArchitectureScreenError(
            "screen contract, A anchor, instances, or sources changed"
        )
    return observed


def _load_terminal(
    output_root: Path,
    *,
    architecture: str,
    model_seed: int,
) -> tuple[dict, str]:
    path = _terminal_path(output_root, architecture, model_seed)
    if not path.is_file():
        raise ArchitectureScreenError(f"missing terminal checkpoint: {path}")
    raw_sha = _sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ArchitectureScreenError(f"terminal checkpoint is not a mapping: {path}")
    expected = {
        "training_protocol": trainer.TRAINING_PROTOCOL,
        "trainer_schema_version": trainer.TRAINER_SCHEMA_VERSION,
        "checkpoint_role": trainer.TERMINAL_CHECKPOINT_ROLE,
        "trainer_resumable": False,
        "fixed_terminal_checkpoint": True,
        "development_only": True,
        "evaluation_panels_opened": False,
        "completed_training_episodes": TRAINING_EPISODES,
    }
    mismatch = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatch:
        raise ArchitectureScreenError(
            f"terminal checkpoint metadata mismatch for {architecture}/"
            f"seed-{model_seed}: {mismatch}"
        )
    resume = payload.get("resume_contract")
    agent_checkpoint = payload.get("agent_checkpoint")
    if not isinstance(resume, Mapping) or not isinstance(
        agent_checkpoint, Mapping
    ):
        raise ArchitectureScreenError("terminal checkpoint lacks bound contracts")
    if resume.get("architecture") != architecture:
        raise ArchitectureScreenError("terminal architecture mismatch")
    if resume.get("model_seed") != model_seed:
        raise ArchitectureScreenError("terminal model seed mismatch")
    if resume.get("episodes") != TRAINING_EPISODES:
        raise ArchitectureScreenError("terminal episode budget mismatch")
    if tuple(float(x) for x in resume.get("preference_grid", ())) != LAMBDA_GRID:
        raise ArchitectureScreenError("terminal preference grid mismatch")
    config = agent_checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise ArchitectureScreenError("terminal agent config is missing")
    expected_conditioned = architecture == "conditioned"
    if bool(config.get("condition_on_preference")) != expected_conditioned:
        raise ArchitectureScreenError("terminal conditioning flag mismatch")
    if int(config.get("preference_relabels", -1)) != PREFERENCE_RELABELS:
        raise ArchitectureScreenError("terminal relabel count mismatch")
    if agent_checkpoint.get("controller_architecture") != (
        PREFERENCE_CONDITIONED_CONTROLLER_ARCHITECTURE
    ):
        raise ArchitectureScreenError("terminal controller architecture mismatch")
    if agent_checkpoint.get("exact_safe_mask_authoritative") is not True:
        raise ArchitectureScreenError("terminal lost exact verifier authority")
    if agent_checkpoint.get("baseline_teacher") is not False:
        raise ArchitectureScreenError("terminal unexpectedly used a teacher")
    if "replay" in agent_checkpoint.get("agent_state", {}):
        raise ArchitectureScreenError("terminal checkpoint is not replay-free")
    return dict(payload), raw_sha


class _FixedLambdaAgent:
    """Adapt the vector agent to the established deterministic runner API."""

    def __init__(
        self, agent: PreferenceConditionedVectorAgent, preference_lambda: float
    ) -> None:
        self.agent = agent
        self.preference_lambda = float(preference_lambda)
        self.config = agent.config

    def reset_episode_state(self) -> None:
        self.agent.reset_episode_state()

    def select(self, snapshot, *, training=False, epsilon=0.0):
        return self.agent.select(
            snapshot,
            preference_lambda=self.preference_lambda,
            training=training,
            epsilon=epsilon,
        )

    def observe_outcome(self, decision, *, next_snapshot, done):
        return self.agent.observe_outcome(
            decision, next_snapshot=next_snapshot, done=done
        )


def _fresh_fixed_lambda_agent(
    agent_checkpoint: Mapping,
    *,
    model_seed: int,
    preference_lambda: float,
    device: torch.device,
) -> _FixedLambdaAgent:
    agent = PreferenceConditionedVectorAgent.from_checkpoint(
        dict(agent_checkpoint),
        device=device,
        resumable=False,
        seed=model_seed,
    )
    for network in (agent.Q_local, agent.Q_target):
        network.requires_grad_(False)
        network.eval()
    agent.set_epsilon(0.0)
    return _FixedLambdaAgent(agent, preference_lambda)


def _ledger_path(
    output_root: Path,
    architecture: str,
    model_seed: int,
    preference_lambda: float,
) -> Path:
    token = f"{preference_lambda:.3f}".replace(".", "p")
    return (
        output_root
        / "evaluation-ledger"
        / architecture
        / f"seed-{model_seed}-lambda-{token}.json"
    )


def _validate_rows(rows: Sequence[Mapping]) -> None:
    if len(rows) != len(INSTANCE_SEEDS):
        raise ArchitectureScreenError("evaluation ledger row count changed")
    if tuple(int(row.get("instance_seed", -1)) for row in rows) != INSTANCE_SEEDS:
        raise ArchitectureScreenError("evaluation ledger instance order changed")
    required = {
        "instance_seed",
        "episode_instance_id",
        "schedule_id",
        "behavior_digest",
        "strict_safe_complete",
        "dense_return",
        "mean_absolute_error",
        "steps",
        "physical_rehandles",
        "physical_rehandles_per_100",
    }
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ArchitectureScreenError(
                f"evaluation row is missing fields: {sorted(missing)}"
            )


def _validate_ledger(
    ledger: Mapping,
    *,
    architecture: str,
    model_seed: int,
    preference_lambda: float,
    terminal_sha256: str,
) -> dict:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "architecture": architecture,
        "architecture_screen_arm": ARM_BY_ARCHITECTURE[architecture],
        "model_seed": model_seed,
        "fixed_lambda": preference_lambda,
        "terminal_checkpoint_sha256": terminal_sha256,
        "row_count": len(INSTANCE_SEEDS),
    }
    mismatch = {
        name: (ledger.get(name), value)
        for name, value in expected.items()
        if ledger.get(name) != value
    }
    if mismatch:
        raise ArchitectureScreenError(f"evaluation ledger changed: {mismatch}")
    rows = ledger.get("rows")
    if not isinstance(rows, list):
        raise ArchitectureScreenError("evaluation ledger rows are missing")
    _validate_rows(rows)
    return dict(ledger)


def evaluate_cell(
    project_root: Path,
    output_root: Path,
    *,
    architecture: str,
    model_seed: int,
    preference_lambda: float,
    device_name: str,
) -> dict:
    _require_contract(project_root, output_root)
    if architecture not in ARCHITECTURES:
        raise ArchitectureScreenError(f"unknown architecture: {architecture}")
    if model_seed not in MODEL_SEEDS:
        raise ArchitectureScreenError(f"unknown model seed: {model_seed}")
    if preference_lambda not in LAMBDA_GRID:
        raise ArchitectureScreenError(f"unknown lambda: {preference_lambda}")
    terminal, terminal_sha = _load_terminal(
        output_root, architecture=architecture, model_seed=model_seed
    )
    path = _ledger_path(
        output_root, architecture, model_seed, preference_lambda
    )
    if path.is_file():
        return _validate_ledger(
            json.loads(path.read_text(encoding="utf-8")),
            architecture=architecture,
            model_seed=model_seed,
            preference_lambda=preference_lambda,
            terminal_sha256=terminal_sha,
        )

    historical_arm, sources, _report, _sha = _load_historical_anchor(
        project_root
    )
    device = pilot._device(device_name)
    agent_checkpoint = terminal["agent_checkpoint"]
    rows = []
    for instance_seed in INSTANCE_SEEDS:
        instance = sources.instances[instance_seed]

        def wrapper_factory(_unused_base):
            return _fresh_fixed_lambda_agent(
                agent_checkpoint,
                model_seed=model_seed,
                preference_lambda=preference_lambda,
                device=device,
            )

        raw = pilot._run_raw(
            historical_arm,
            instance,
            device=device,
            wrapper_factory=wrapper_factory,
        )
        row = pilot._compact_row(raw, instance)
        row.update(
            {
                "protocol": PROTOCOL,
                "architecture": architecture,
                "architecture_screen_arm": ARM_BY_ARCHITECTURE[architecture],
                "model_seed": model_seed,
                "fixed_lambda": preference_lambda,
                "deterministic_policy": True,
                "fresh_agent": True,
                "exact_verifier_authoritative": True,
                "unsafe_unknown_fail_closed": True,
            }
        )
        rows.append(row)
    _validate_rows(rows)
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete",
        "architecture": architecture,
        "architecture_screen_arm": ARM_BY_ARCHITECTURE[architecture],
        "condition_on_preference": architecture == "conditioned",
        "model_seed": model_seed,
        "fixed_lambda": preference_lambda,
        "terminal_checkpoint_sha256": terminal_sha,
        "row_count": len(rows),
        "strict_safe_complete_rows": sum(
            row["strict_safe_complete"] is True for row in rows
        ),
        "rows": rows,
    }
    _atomic_json(path, ledger)
    return ledger


def evaluate(
    project_root: Path, output_root: Path, *, device_name: str
) -> dict:
    _require_contract(project_root, output_root)
    jobs = []
    for architecture in ARCHITECTURES:
        for model_seed in MODEL_SEEDS:
            for value in LAMBDA_GRID:
                ledger = evaluate_cell(
                    project_root,
                    output_root,
                    architecture=architecture,
                    model_seed=model_seed,
                    preference_lambda=value,
                    device_name=device_name,
                )
                jobs.append(
                    {
                        "architecture": architecture,
                        "model_seed": model_seed,
                        "lambda": value,
                        "strict_safe_complete_rows": ledger[
                            "strict_safe_complete_rows"
                        ],
                    }
                )
    return {
        "status": "complete",
        "job_count": len(jobs),
        "row_count": len(jobs) * len(INSTANCE_SEEDS),
        "jobs": jobs,
    }


def _cell_metrics(rows: Sequence[Mapping]) -> Optional[dict]:
    _validate_rows(rows)
    if not all(row.get("strict_safe_complete") is True for row in rows):
        return None
    total_rehandles = sum(int(row["physical_rehandles"]) for row in rows)
    return {
        "mean_dense_return": float(
            fmean(float(row["dense_return"]) for row in rows)
        ),
        "mean_absolute_error": float(
            fmean(float(row["mean_absolute_error"]) for row in rows)
        ),
        "mean_steps": float(fmean(float(row["steps"]) for row in rows)),
        "mean_physical_rehandles": float(
            fmean(float(row["physical_rehandles"]) for row in rows)
        ),
        "total_physical_rehandles": int(total_rehandles),
        "physical_rehandles_per_100": float(
            100.0 * total_rehandles / (len(rows) * EXPECTED_BLOCKS)
        ),
    }


def _aggregate_seed_metrics(seed_metrics: Mapping[int, Mapping]) -> dict:
    """Aggregate only after every seed cell is eligible (equal seed weights)."""

    if set(seed_metrics) != set(MODEL_SEEDS):
        raise ArchitectureScreenError("seed metric grid is incomplete")
    result = {
        name: float(fmean(float(seed_metrics[seed][name]) for seed in MODEL_SEEDS))
        for name in (
            "mean_dense_return",
            "mean_absolute_error",
            "mean_steps",
            "mean_physical_rehandles",
            "physical_rehandles_per_100",
        )
    }
    result["total_physical_rehandles"] = int(
        sum(
            int(seed_metrics[seed]["total_physical_rehandles"])
            for seed in MODEL_SEEDS
        )
    )
    return result


def _load_architecture_rows(
    output_root: Path, architecture: str
) -> dict[int, dict[float, list[dict]]]:
    terminal_hashes = {
        seed: _load_terminal(
            output_root, architecture=architecture, model_seed=seed
        )[1]
        for seed in MODEL_SEEDS
    }
    result = {seed: {} for seed in MODEL_SEEDS}
    for seed in MODEL_SEEDS:
        for value in LAMBDA_GRID:
            path = _ledger_path(output_root, architecture, seed, value)
            if not path.is_file():
                raise ArchitectureScreenError(f"missing evaluation ledger: {path}")
            ledger = _validate_ledger(
                json.loads(path.read_text(encoding="utf-8")),
                architecture=architecture,
                model_seed=seed,
                preference_lambda=value,
                terminal_sha256=terminal_hashes[seed],
            )
            result[seed][value] = list(ledger["rows"])
    return result


def _nondominated(metrics: Mapping[float, Mapping]) -> list[float]:
    result = []
    for value, point in metrics.items():
        dominated = False
        for other_value, other in metrics.items():
            if other_value == value:
                continue
            no_worse = (
                float(other["mean_absolute_error"])
                <= float(point["mean_absolute_error"])
                and float(other["physical_rehandles_per_100"])
                <= float(point["physical_rehandles_per_100"])
            )
            strict = (
                float(other["mean_absolute_error"])
                < float(point["mean_absolute_error"])
                or float(other["physical_rehandles_per_100"])
                < float(point["physical_rehandles_per_100"])
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            result.append(value)
    return sorted(result)


def _monotonicity(metrics: Mapping[float, Mapping]) -> dict:
    inversions = []
    for lower, upper in zip(LAMBDA_GRID, LAMBDA_GRID[1:]):
        increase = float(metrics[upper]["physical_rehandles_per_100"]) - float(
            metrics[lower]["physical_rehandles_per_100"]
        )
        if increase > 1e-12:
            inversions.append(
                {"from_lambda": lower, "to_lambda": upper, "increase": increase}
            )
    return {
        "nonincreasing": not inversions,
        "inversion_count": len(inversions),
        "total_inversion_magnitude": float(
            sum(item["increase"] for item in inversions)
        ),
        "inversions": inversions,
    }


def _architecture_summary(rows_by_seed: Mapping) -> dict:
    eligibility = {}
    internal_metrics: dict[int, dict[float, dict]] = {
        seed: {} for seed in MODEL_SEEDS
    }
    failures = []
    all_rows_safe = True
    for seed in MODEL_SEEDS:
        for value in LAMBDA_GRID:
            rows = rows_by_seed[seed][value]
            metrics = _cell_metrics(rows)
            key = f"seed-{seed}/lambda-{_lambda_key(value)}"
            safe_count = sum(
                row.get("strict_safe_complete") is True for row in rows
            )
            eligibility[key] = {
                "row_count": len(rows),
                "strict_safe_complete_rows": safe_count,
                "eligible": metrics is not None,
            }
            if metrics is None:
                all_rows_safe = False
                failures.extend(
                    {
                        "model_seed": seed,
                        "lambda": value,
                        "instance_seed": row["instance_seed"],
                    }
                    for row in rows
                    if row.get("strict_safe_complete") is not True
                )
            else:
                internal_metrics[seed][value] = metrics
    if not all_rows_safe:
        return {
            "whole_method_eligible": False,
            "whole_method_metrics_suppressed": True,
            "expected_rows": EXPECTED_ROWS_PER_ARCHITECTURE,
            "unsafe_or_incomplete_rows": failures,
            "cell_eligibility": eligibility,
            "seed_metrics": None,
            "aggregate_metrics": None,
            "aggregate_nondominated_lambdas": None,
            "handling_monotonicity": None,
        }
    aggregate = {
        value: _aggregate_seed_metrics(
            {seed: internal_metrics[seed][value] for seed in MODEL_SEEDS}
        )
        for value in LAMBDA_GRID
    }
    return {
        "whole_method_eligible": True,
        "whole_method_metrics_suppressed": False,
        "expected_rows": EXPECTED_ROWS_PER_ARCHITECTURE,
        "unsafe_or_incomplete_rows": [],
        "cell_eligibility": eligibility,
        "seed_metrics": {
            str(seed): {
                _lambda_key(value): internal_metrics[seed][value]
                for value in LAMBDA_GRID
            }
            for seed in MODEL_SEEDS
        },
        "aggregate_metrics": {
            _lambda_key(value): aggregate[value] for value in LAMBDA_GRID
        },
        "aggregate_nondominated_lambdas": _nondominated(aggregate),
        "handling_monotonicity": _monotonicity(aggregate),
    }


def _historical_summary(report: Mapping) -> dict:
    aggregate = {
        float(key): dict(value)
        for key, value in report["aggregate_metrics"].items()
    }
    return {
        "whole_method_eligible": True,
        "whole_method_metrics_suppressed": False,
        "historical_report_reused": True,
        "historical_rows_reexecuted": False,
        "expected_rows": EXPECTED_ROWS_PER_ARCHITECTURE,
        "aggregate_metrics": {
            _lambda_key(value): aggregate[value] for value in LAMBDA_GRID
        },
        "aggregate_nondominated_lambdas": _nondominated(aggregate),
        "handling_monotonicity": _monotonicity(aggregate),
    }


def _conditioning_comparison(
    conditioned: Mapping,
    unconditioned: Mapping,
    conditioned_rows: Mapping,
    unconditioned_rows: Mapping,
) -> dict:
    if not (
        conditioned["whole_method_eligible"]
        and unconditioned["whole_method_eligible"]
    ):
        return {
            "eligible": False,
            "reason": "B or C was suppressed after an unsafe/incomplete row",
            "fixed_lambda_deltas_B_minus_C": None,
        }
    deltas = {}
    b_wins = {"dense_return": 0, "mean_absolute_error": 0, "steps": 0, "rehandles": 0}
    behavior_equal = 0
    behavior_total = 0
    for value in LAMBDA_GRID:
        key = _lambda_key(value)
        b = conditioned["aggregate_metrics"][key]
        c = unconditioned["aggregate_metrics"][key]
        delta = {
            "mean_dense_return": b["mean_dense_return"] - c["mean_dense_return"],
            "mean_absolute_error": b["mean_absolute_error"]
            - c["mean_absolute_error"],
            "mean_steps": b["mean_steps"] - c["mean_steps"],
            "physical_rehandles_per_100": b["physical_rehandles_per_100"]
            - c["physical_rehandles_per_100"],
        }
        deltas[key] = delta
        b_wins["dense_return"] += int(delta["mean_dense_return"] > 0.0)
        b_wins["mean_absolute_error"] += int(delta["mean_absolute_error"] < 0.0)
        b_wins["steps"] += int(delta["mean_steps"] < 0.0)
        b_wins["rehandles"] += int(delta["physical_rehandles_per_100"] < 0.0)
        for seed in MODEL_SEEDS:
            b_rows = conditioned_rows[seed][value]
            c_rows = unconditioned_rows[seed][value]
            for b_row, c_row in zip(b_rows, c_rows):
                behavior_total += 1
                behavior_equal += int(
                    b_row["behavior_digest"] == c_row["behavior_digest"]
                )
    return {
        "eligible": True,
        "fixed_lambda_deltas_B_minus_C": deltas,
        "lambda_levels_where_B_is_better": b_wins,
        "matched_behavior_digest_pairs": behavior_total,
        "identical_behavior_digest_pairs": behavior_equal,
        "different_behavior_digest_pairs": behavior_total - behavior_equal,
        "interpretation": (
            "paired descriptive conditioning diagnostic; because no bounded "
            "Q-vector calibration branch is included, this is provisional "
            "architecture evidence rather than a calibration claim"
        ),
    }


def _combined_nondominated(arms: Mapping[str, Mapping]) -> list[dict]:
    points = []
    for arm, summary in arms.items():
        metrics = summary.get("aggregate_metrics")
        if not isinstance(metrics, Mapping):
            continue
        for value in LAMBDA_GRID:
            point = metrics[_lambda_key(value)]
            points.append(
                {
                    "arm": arm,
                    "lambda": value,
                    "mean_absolute_error": point["mean_absolute_error"],
                    "physical_rehandles_per_100": point[
                        "physical_rehandles_per_100"
                    ],
                }
            )
    output = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            no_worse = (
                other["mean_absolute_error"] <= point["mean_absolute_error"]
                and other["physical_rehandles_per_100"]
                <= point["physical_rehandles_per_100"]
            )
            strict = (
                other["mean_absolute_error"] < point["mean_absolute_error"]
                or other["physical_rehandles_per_100"]
                < point["physical_rehandles_per_100"]
            )
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            output.append(point)
    return output


def _advancement_gate(a: Mapping, b: Mapping) -> dict:
    if not b["whole_method_eligible"]:
        criteria = {
            "all_B_rows_strict_safe_complete": False,
            "B_lambda0_mean_dense_return_at_least_A": False,
            "B_lambda0_mean_absolute_error_no_worse_than_A": False,
            "B_lambda0p2_rehandles_per_100_no_worse_than_A": False,
            "B_at_least_three_sampled_nondominated_points": False,
            "B_aggregate_rehandles_nonincreasing_in_lambda": False,
        }
    else:
        a0 = a["aggregate_metrics"]["0.0"]
        b0 = b["aggregate_metrics"]["0.0"]
        a2 = a["aggregate_metrics"]["0.2"]
        b2 = b["aggregate_metrics"]["0.2"]
        criteria = {
            "all_B_rows_strict_safe_complete": True,
            "B_lambda0_mean_dense_return_at_least_A": (
                b0["mean_dense_return"] >= a0["mean_dense_return"]
            ),
            "B_lambda0_mean_absolute_error_no_worse_than_A": (
                b0["mean_absolute_error"] <= a0["mean_absolute_error"]
            ),
            "B_lambda0p2_rehandles_per_100_no_worse_than_A": (
                b2["physical_rehandles_per_100"]
                <= a2["physical_rehandles_per_100"]
            ),
            "B_at_least_three_sampled_nondominated_points": (
                len(b["aggregate_nondominated_lambdas"]) >= 3
            ),
            "B_aggregate_rehandles_nonincreasing_in_lambda": b[
                "handling_monotonicity"
            ]["nonincreasing"],
        }
    return {
        "criteria": criteria,
        "passed": all(criteria.values()),
        "decision": (
            "advance_B_to_a_new_unseen_confirmation_panel"
            if all(criteria.values())
            else "do_not_advance_B_without_redesign_or_more_development"
        ),
        "C_role": "matched_masked_preference_architecture_diagnostic_only",
    }


def analyze(project_root: Path, output_root: Path) -> dict:
    contract = _require_contract(project_root, output_root)
    _arm, _sources, anchor_report, anchor_sha = _load_historical_anchor(
        project_root
    )
    rows = {
        architecture: _load_architecture_rows(output_root, architecture)
        for architecture in ARCHITECTURES
    }
    arms = {
        "A": _historical_summary(anchor_report),
        "B": _architecture_summary(rows["conditioned"]),
        "C": _architecture_summary(rows["unconditioned"]),
    }
    conditioning = _conditioning_comparison(
        arms["B"], arms["C"], rows["conditioned"], rows["unconditioned"]
    )
    gate = _advancement_gate(arms["A"], arms["B"])
    suppressed = [
        arm for arm in ("B", "C") if not arms[arm]["whole_method_eligible"]
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": (
            "complete" if not suppressed else "complete_with_method_suppression"
        ),
        "scope": "opened_85k_development_architecture_screen",
        "contract_sha256": contract["contract_sha256"],
        "lambda_grid": list(LAMBDA_GRID),
        "model_seeds": list(MODEL_SEEDS),
        "instance_seeds": list(INSTANCE_SEEDS),
        "historical_A_report_sha256": anchor_sha,
        "new_evaluation_row_count": len(ARCHITECTURES)
        * EXPECTED_ROWS_PER_ARCHITECTURE,
        "suppressed_new_architectures": suppressed,
        "arms": arms,
        "combined_nondominated_points": _combined_nondominated(arms),
        "conditioning_comparison": conditioning,
        "advancement_gate": gate,
        "calibration": {
            "included": False,
            "conditioning_evidence": "provisional",
            "required_next_if_B_advances": (
                "bounded held-out Q_op/Q_N prediction calibration under "
                "fixed-lambda rollouts"
            ),
        },
        "interpretation": (
            "A is an authenticated historical development anchor. B and C "
            "were trained from scratch on paired lambda-stratified behavior "
            "and evaluated with fixed terminal checkpoints. This screen can "
            "select an architecture for confirmation; it cannot itself serve "
            "as unseen-panel confirmation."
        ),
    }
    path = output_root / REPORT_NAME
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != report:
            raise ArchitectureScreenError("existing analysis report changed")
    else:
        _atomic_json(path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "evaluate", "analyze", "run-all")
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root
        / "results/vcg-preference-conditioned-architecture-screen-85k-development"
    )
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(project_root, output_root)
    elif args.command == "evaluate":
        result = evaluate(project_root, output_root, device_name=args.device)
    elif args.command == "analyze":
        result = analyze(project_root, output_root)
    else:
        prepare(project_root, output_root)
        evaluate(project_root, output_root, device_name=args.device)
        result = analyze(project_root, output_root)
    display = result
    if args.command == "analyze" or (
        args.command == "run-all" and "advancement_gate" in result
    ):
        display = {
            "status": result["status"],
            "suppressed_new_architectures": result[
                "suppressed_new_architectures"
            ],
            "advancement_gate": result["advancement_gate"],
            "report": str((output_root / REPORT_NAME).resolve()),
        }
    print(json.dumps(display, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
