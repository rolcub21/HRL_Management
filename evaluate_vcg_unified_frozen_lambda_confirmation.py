#!/usr/bin/env python3
"""Confirm frozen unified VCG lambda 0 versus .05 on a fresh 87k panel."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Mapping, Optional, Sequence

import evaluate_vcg_unified_frozen_lambda_seed_stability as development
import evaluate_vcg_unified_frozen_lambda_sweep as base
from example.episode_instance import EpisodeInstance
import train_vcg_constrained_v2_1 as atomic_io
import train_vcg_constrained_v2_3 as v23
import train_vcg_unified as unified
import train_vcg_unified_seed_stability as training_seeds


PROTOCOL = "vcg_unified_frozen_lambda_005_confirmation_87k_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = (15, 16, 17)
LAMBDAS = (0.0, 0.05)
INSTANCE_SEEDS = tuple(range(87_000, 87_030))
RNG_INDICES = (0, 1, 2, 3)
POLICY_RNG_BASE = 624_000_000
EPISODE = 200
MAX_STEPS = 2_000
MAE_NONINFERIORITY_MARGIN = 2.0
# One-sided familywise 95% over the two primary endpoints:
# t_{1-.05/2, df=29} = t_{.975,29}.
PRIMARY_T_DF29 = 2.045229642132703
NOMINAL_TWO_SIDED_T_DF29 = 2.045229642132703
FIELDS = base.FIELDS

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "vcg-unified-frozen-lambda-confirmation-87k"
DEFAULT_PARENT = HERE / "results" / "vcg-unified-lambda-pair-seed-stability-85k"
DEVELOPMENT_REPORT = (
    HERE
    / "results"
    / "vcg-unified-frozen-lambda-seed-stability-85k-development"
    / "frozen-lambda-seed-stability.json"
)


class FrozenLambdaConfirmationError(RuntimeError):
    pass


def _read_mapping(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenLambdaConfirmationError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise FrozenLambdaConfirmationError(f"{path} must contain an object")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha(value: Mapping) -> str:
    return v23.contract_hash(value)


def _key(value: float) -> str:
    return f"lambda-{float(value):.2f}"


def _policy_rng(instance_index: int, rng_index: int) -> int:
    if instance_index not in range(30) or rng_index not in RNG_INDICES:
        raise FrozenLambdaConfirmationError("policy RNG index is outside the grid")
    return POLICY_RNG_BASE + 4 * instance_index + rng_index


def _authenticate_development() -> dict:
    report = _read_mapping(DEVELOPMENT_REPORT)
    criterion = report.get("continuation_criterion")
    if (
        report.get("status") != "passed"
        or report.get("candidate_lambda") != 0.05
        or report.get("model_seeds") != list(MODEL_SEEDS)
        or not isinstance(criterion, dict)
        or criterion.get("passed") is not True
        or criterion.get("all_288_rows_strict_safe_complete") is not True
        or criterion.get("strong_three_of_three") is not True
        or report.get("training_or_learning") is not False
        or report.get("final_86xxx_panel_opened") is not False
    ):
        raise FrozenLambdaConfirmationError("development selection report mismatch")
    return {
        "path": str(DEVELOPMENT_REPORT.resolve()),
        "raw_sha256": _sha(DEVELOPMENT_REPORT),
        "contract_sha256": report.get("contract_sha256"),
    }


def _parents(parent_root: Path) -> dict[int, dict]:
    return {
        seed: development._parent(Path(parent_root), seed)
        for seed in MODEL_SEEDS
    }


def _build_contract(parent_root: Path, *, device: str) -> dict:
    parents = _parents(parent_root)
    if any(parent["base_contract"].get("device") != device for parent in parents.values()):
        raise FrozenLambdaConfirmationError("device must match all frozen parents")
    source = Path(__file__).resolve()
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scientific_role": "prospective_frozen_policy_scalarization_confirmation",
        "hypothesis_frozen_before_panel_open": True,
        "model_seeds": MODEL_SEEDS,
        "lambda_values": LAMBDAS,
        "candidate_lambda": 0.05,
        "checkpoint_episode": EPISODE,
        "parent_root": str(Path(parent_root).resolve()),
        "parents": tuple(
            {
                "model_seed": seed,
                "checkpoint": str(parents[seed]["checkpoint_path"]),
                "checkpoint_raw_sha256": parents[seed]["checkpoint_sha256"],
                "training_summary_raw_sha256": parents[seed]["summary_sha256"],
                "training_contract_raw_sha256": parents[seed]["contract_sha256_raw"],
                "training_contract_canonical_sha256": parents[seed]["contract"]["contract_sha256"],
            }
            for seed in MODEL_SEEDS
        ),
        "development_selection": _authenticate_development(),
        "episode_instance_seeds": INSTANCE_SEEDS,
        "episode_instance_panel_generated_at_prepare": False,
        "policy_rng_formula": "624000000 + 4*instance_index + rng_index",
        "policy_rng_range": (POLICY_RNG_BASE, POLICY_RNG_BASE + 119),
        "rng_indices": RNG_INDICES,
        "rows_per_model_lambda": 120,
        "expected_total_rows": 720,
        "aggregation": (
            "four action RNGs within model-seed/EpisodeInstance, then equal "
            "model seeds within EpisodeInstance, then 30 paired EpisodeInstances"
        ),
        "primary_endpoints": (
            "physical_rehandles_per_100",
            "mean_absolute_error",
        ),
        "predeclared_success": {
            "all_720_rows_strict_safe_complete": True,
            "rehandle_difference_orientation": "lambda005_minus_lambda0",
            "rehandle_simultaneous_one_sided_95_upper_below": 0.0,
            "mae_difference_orientation": "lambda005_minus_lambda0",
            "mae_noninferiority_margin": MAE_NONINFERIORITY_MARGIN,
            "mae_simultaneous_one_sided_95_upper_below": MAE_NONINFERIORITY_MARGIN,
            "critical_t_df29": PRIMARY_T_DF29,
            "familywise_alpha": 0.05,
            "endpoint_count": 2,
        },
        "environment_semantics": {
            "grid": "5x5",
            "blocks": 8,
            "arrival_rate_lambda": 10.0,
            "exponential_interarrival_scale": 0.1,
            "processing_duration_poisson_mean": 80.0,
            "max_steps": MAX_STEPS,
        },
        "device": device,
        "source_path": str(source),
        "source_sha256": _sha(source),
        "development_evaluator_source_sha256": _sha(Path(development.__file__)),
        "base_evaluator_source_sha256": _sha(Path(base.__file__)),
        "training_or_learning": False,
        "panel_opened": False,
        "reuses_85xxx": False,
        "reuses_86xxx": False,
    }
    result["contract_sha256"] = _canonical_sha(result)
    return result


def prepare(output_dir: Path, parent_root: Path, *, device: str) -> dict:
    output = Path(output_dir).resolve()
    contract = _build_contract(parent_root, device=device)
    contract_path = output / "confirmation-contract.json"
    if contract_path.exists():
        if v23._json_safe(_read_mapping(contract_path)) != v23._json_safe(contract):
            raise FrozenLambdaConfirmationError("existing confirmation contract drifted")
    else:
        if output.exists() and any(output.iterdir()):
            raise FrozenLambdaConfirmationError("new confirmation root is not empty")
        atomic_io._atomic_json(contract, contract_path)
    return contract


@contextmanager
def _confirmation_rng_profile():
    old_v23 = v23.VALIDATION_POLICY_RNG_BASE
    old_unified = unified.VALIDATION_POLICY_RNG_BASE
    try:
        v23.VALIDATION_POLICY_RNG_BASE = POLICY_RNG_BASE
        unified.VALIDATION_POLICY_RNG_BASE = POLICY_RNG_BASE
        yield
    finally:
        v23.VALIDATION_POLICY_RNG_BASE = old_v23
        unified.VALIDATION_POLICY_RNG_BASE = old_unified


def _runtime_context(seed: int):
    stack = ExitStack()
    stack.enter_context(training_seeds.activated_paired_profile(training_seeds.profile_for_seed(seed)))
    stack.enter_context(unified.activated_unified_seed_profile())
    stack.enter_context(_confirmation_rng_profile())
    return stack


class _BoundEnvironment:
    def __init__(self, environment, instance: EpisodeInstance) -> None:
        self._environment = environment
        self._instance = instance

    def sample_episode_instance(self, seed: int):
        if int(seed) != int(self._instance.seed):
            raise FrozenLambdaConfirmationError("bound EpisodeInstance seed mismatch")
        return self._instance

    def __getattr__(self, name):
        return getattr(self._environment, name)


def _activation(output: Path, contract: Mapping) -> dict:
    path = output / "panel-activation.json"
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "panel_opened": True,
        "episode_instance_seeds": INSTANCE_SEEDS,
        "reuses_85xxx": False,
        "reuses_86xxx": False,
    }
    value["activation_sha256"] = _canonical_sha(value)
    if path.exists():
        if v23._json_safe(_read_mapping(path)) != v23._json_safe(value):
            raise FrozenLambdaConfirmationError("panel activation marker drifted")
    else:
        atomic_io._atomic_json(value, path)
    return value


def _load_instance(path: Path, entry: Mapping, environment=None) -> EpisodeInstance:
    if not path.is_file() or _sha(path) != entry.get("raw_sha256"):
        raise FrozenLambdaConfirmationError(f"instance file mismatch: {path}")
    instance = EpisodeInstance.from_dict(_read_mapping(path))
    canonical = instance.to_json().encode("utf-8")
    if (
        instance.seed != entry.get("instance_seed")
        or instance.instance_id != entry.get("episode_instance_id")
        or instance.schedule_id != entry.get("schedule_id")
        or hashlib.sha256(canonical).hexdigest() != entry.get("canonical_sha256")
    ):
        raise FrozenLambdaConfirmationError(f"instance identity mismatch: {path}")
    if environment is not None:
        instance.validate_for(environment)
    return instance


def _validate_manifest(output: Path, contract: Mapping) -> dict:
    path = output / "episode-instance-manifest.json"
    manifest = _read_mapping(path)
    canonical = dict(manifest)
    received = canonical.pop("manifest_sha256", None)
    entries = manifest.get("instances")
    if (
        manifest.get("protocol") != PROTOCOL
        or manifest.get("contract_sha256") != contract["contract_sha256"]
        or manifest.get("panel_opened") is not True
        or manifest.get("instance_count") != 30
        or not isinstance(received, str)
        or _canonical_sha(canonical) != received
        or not isinstance(entries, list)
        or len(entries) != 30
        or [entry.get("instance_seed") for entry in entries] != list(INSTANCE_SEEDS)
    ):
        raise FrozenLambdaConfirmationError("EpisodeInstance manifest mismatch")
    for entry in entries:
        _load_instance(output / entry["relative_path"], entry)
    return manifest


def materialize_panel(output_dir: Path, parent_root: Path, *, device: str) -> dict:
    contract = prepare(output_dir, parent_root, device=device)
    output = Path(output_dir).resolve()
    _activation(output, contract)
    manifest_path = output / "episode-instance-manifest.json"
    if manifest_path.exists():
        return _validate_manifest(output, contract)
    parents = _parents(parent_root)
    entries = []
    with _runtime_context(MODEL_SEEDS[0]):
        runtime = base._runtime(parents[MODEL_SEEDS[0]], device=device)
        for instance_seed in INSTANCE_SEEDS:
            instance = runtime.env.sample_episode_instance(instance_seed)
            instance.validate_for(runtime.env)
            relative = Path("episode-instances") / f"seed-{instance_seed}.json"
            path = output / relative
            if path.exists():
                observed = EpisodeInstance.from_dict(_read_mapping(path))
                if observed.to_dict() != instance.to_dict():
                    raise FrozenLambdaConfirmationError("partial panel instance drifted")
            else:
                atomic_io._atomic_json(instance.to_dict(), path)
            entries.append(
                {
                    "instance_seed": instance_seed,
                    "relative_path": str(relative),
                    "raw_sha256": _sha(path),
                    "canonical_sha256": hashlib.sha256(instance.to_json().encode("utf-8")).hexdigest(),
                    "episode_instance_id": instance.instance_id,
                    "schedule_id": instance.schedule_id,
                }
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "panel_opened": True,
        "instance_count": 30,
        "instances": entries,
        "reuses_85xxx": False,
        "reuses_86xxx": False,
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    atomic_io._atomic_json(manifest, manifest_path)
    return _validate_manifest(output, contract)


def _row_grid() -> set[tuple[int, int, int]]:
    return {
        (instance_seed, rng_index, _policy_rng(instance_index, rng_index))
        for instance_index, instance_seed in enumerate(INSTANCE_SEEDS)
        for rng_index in RNG_INDICES
    }


def _validate_ledger(
    ledger: Mapping,
    *,
    contract: Mapping,
    manifest: Mapping,
    model_seed: int,
    fixed_lambda: float,
    parent_sha256: str,
) -> list[dict]:
    canonical = dict(ledger)
    received = canonical.pop("ledger_sha256", None)
    rows = ledger.get("rows")
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("protocol") != PROTOCOL
        or ledger.get("contract_sha256") != contract["contract_sha256"]
        or ledger.get("instance_manifest_sha256") != manifest["manifest_sha256"]
        or ledger.get("model_seed") != model_seed
        or ledger.get("parent_checkpoint_raw_sha256") != parent_sha256
        or not math.isclose(float(ledger.get("fixed_lambda", -1.0)), fixed_lambda, rel_tol=0.0, abs_tol=1e-12)
        or ledger.get("row_count") != 120
        or ledger.get("strict_integrity_gate") is not True
        or not isinstance(received, str)
        or _canonical_sha(canonical) != received
        or not isinstance(rows, list)
        or len(rows) != 120
    ):
        raise FrozenLambdaConfirmationError(f"seed {model_seed} {_key(fixed_lambda)} ledger mismatch")
    observed = {
        (int(row["instance_seed"]), int(row["policy_rng_index"]), int(row["policy_rng_seed"]))
        for row in rows
    }
    identities = {entry["instance_seed"]: entry for entry in manifest["instances"]}
    if observed != _row_grid():
        raise FrozenLambdaConfirmationError("confirmation row grid drifted")
    for row in rows:
        entry = identities[int(row["instance_seed"])]
        if (
            not math.isclose(float(row.get("dual_lambda", -1.0)), fixed_lambda, rel_tol=0.0, abs_tol=1e-12)
            or row.get("episode_instance_id") != entry["episode_instance_id"]
            or row.get("schedule_id") != entry["schedule_id"]
            or row.get("episode_instance_sha256") != entry["canonical_sha256"]
            or row.get("strict_method_success") is not True
            or row.get("delivery_count") != row.get("required_deliveries")
            or row.get("method_failure_reason") is not None
            or row.get("all_selected_candidates_exact_safe") is not True
            or row.get("evaluation_learning") is not False
            or row.get("fresh_evaluation_clone") is not True
            or row.get("training_agent_unchanged") is not True
            or any(int(row.get(field, -1)) != 0 for field in ("illegal_drops", "fallbacks", "witness_mismatches"))
        ):
            raise FrozenLambdaConfirmationError("confirmation contains an unsafe or rebound row")
    _instance_points(rows)
    return rows


def _instance_points(rows: Sequence[Mapping]) -> dict[int, dict]:
    groups: dict[int, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[int(row["instance_seed"])].append(row)
    if set(groups) != set(INSTANCE_SEEDS) or any(len(group) != 4 for group in groups.values()):
        raise FrozenLambdaConfirmationError("expected 30 instances crossed with four RNGs")
    points = {}
    for seed, group in groups.items():
        delivered = sum(int(row["required_deliveries"]) for row in group)
        deviations = []
        for row in group:
            values = row.get("delivery_deviations")
            if (
                not isinstance(values, (list, tuple))
                or len(values) != int(row["required_deliveries"])
                or any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values)
            ):
                raise FrozenLambdaConfirmationError("invalid delivery deviations")
            if not math.isclose(
                fmean(abs(float(value)) for value in values),
                float(row["mean_absolute_error"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise FrozenLambdaConfirmationError("stored MAE mismatch")
            deviations.extend(float(value) for value in values)
        points[seed] = {
            "dense_return": float(fmean(float(row["dense_return"]) for row in group)),
            "mean_absolute_error": float(fmean(float(row["mean_absolute_error"]) for row in group)),
            "physical_rehandles_per_100": 100.0 * sum(int(row["physical_rehandles"]) for row in group) / delivered,
            "steps": float(fmean(float(row["steps"]) for row in group)),
            "within_window_percentage": 100.0 * sum(abs(value) <= 20.0 for value in deviations) / delivered,
            "mean_earliness": float(fmean(max(-value, 0.0) for value in deviations)),
            "mean_tardiness": float(fmean(max(value, 0.0) for value in deviations)),
        }
    return points


def evaluate(output_dir: Path, parent_root: Path, *, device: str) -> dict:
    contract = prepare(output_dir, parent_root, device=device)
    output = Path(output_dir).resolve()
    manifest = materialize_panel(output, parent_root, device=device)
    parents = _parents(parent_root)
    completed = {}
    for model_seed in MODEL_SEEDS:
        with _runtime_context(model_seed):
            runtime = base._runtime(parents[model_seed], device=device)
            schedule = v23._install_schedule(runtime, v23.schedule_for_episode(EPISODE, validation=True))
            q_local = runtime._module_signature(runtime.agent.Q_local)
            q_target = runtime._module_signature(runtime.agent.Q_target)
            for fixed_lambda in LAMBDAS:
                path = output / "validation-ledger" / f"seed-{model_seed}" / f"{_key(fixed_lambda)}.json"
                if path.exists():
                    rows = _validate_ledger(
                        _read_mapping(path), contract=contract, manifest=manifest,
                        model_seed=model_seed, fixed_lambda=fixed_lambda,
                        parent_sha256=parents[model_seed]["checkpoint_sha256"],
                    )
                else:
                    runtime.set_dual_lambda(fixed_lambda)
                    raw_rows = []
                    runtime.begin_validation_batch()
                    try:
                        for instance_index, entry in enumerate(manifest["instances"]):
                            instance = _load_instance(output / entry["relative_path"], entry, runtime.env)
                            for rng_index in RNG_INDICES:
                                original_environment = runtime.env
                                runtime.env = _BoundEnvironment(original_environment, instance)
                                try:
                                    raw = dict(runtime.run_episode(
                                        instance_seed=int(instance.seed), training=False,
                                        max_steps=MAX_STEPS, policy_rng_index=rng_index,
                                        policy_rng_seed=_policy_rng(instance_index, rng_index),
                                    ))
                                finally:
                                    runtime.env = original_environment
                                raw["instance_index"] = instance_index
                                raw_rows.append(raw)
                    finally:
                        batch = dict(runtime.end_validation_batch())
                    if batch.get("training_agent_unchanged") is not True:
                        raise FrozenLambdaConfirmationError("validation batch mutated frozen agent")
                    rows = []
                    for index, raw in enumerate(raw_rows):
                        raw["validation_batch_state_unchanged"] = True
                        rows.append(v23._normalize_authenticated_run(
                            raw, index=index, schedule=schedule,
                            dual_lambda=fixed_lambda, training=False,
                        ))
                    ledger = {
                        "schema_version": SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "contract_sha256": contract["contract_sha256"],
                        "instance_manifest_sha256": manifest["manifest_sha256"],
                        "model_seed": model_seed,
                        "parent_checkpoint_raw_sha256": parents[model_seed]["checkpoint_sha256"],
                        "fixed_lambda": fixed_lambda,
                        "row_count": len(rows),
                        "strict_integrity_gate": len(rows) == 120,
                        "batch_audit": batch,
                        "training_or_learning": False,
                        "rows": rows,
                    }
                    ledger["ledger_sha256"] = _canonical_sha(ledger)
                    atomic_io._atomic_json(ledger, path)
                    rows = _validate_ledger(
                        _read_mapping(path), contract=contract, manifest=manifest,
                        model_seed=model_seed, fixed_lambda=fixed_lambda,
                        parent_sha256=parents[model_seed]["checkpoint_sha256"],
                    )
                completed[f"seed-{model_seed}/{_key(fixed_lambda)}"] = len(rows)
            if (
                runtime._module_signature(runtime.agent.Q_local) != q_local
                or runtime._module_signature(runtime.agent.Q_target) != q_target
            ):
                raise FrozenLambdaConfirmationError("frozen checkpoint weights changed")
    return {
        "status": "evaluation_complete",
        "completed_rows": completed,
        "total_rows": sum(completed.values()),
        "training_or_learning": False,
    }


def _means(points: Mapping[int, Mapping]) -> dict:
    return {
        field: float(fmean(float(point[field]) for point in points.values()))
        for field in FIELDS
    }


def _difference(candidate: Mapping[int, Mapping], control: Mapping[int, Mapping], field: str) -> dict:
    values = [float(candidate[seed][field]) - float(control[seed][field]) for seed in INSTANCE_SEEDS]
    mean = float(fmean(values))
    se = float(stdev(values) / math.sqrt(len(values)))
    return {
        "orientation": "lambda005_minus_lambda0",
        "mean_difference": mean,
        "standard_error": se,
        "paired_episode_instances": 30,
        "nominal_two_sided_95_ci": (
            mean - NOMINAL_TWO_SIDED_T_DF29 * se,
            mean + NOMINAL_TWO_SIDED_T_DF29 * se,
        ),
        "simultaneous_one_sided_95_upper": mean + PRIMARY_T_DF29 * se,
    }


def analyze(output_dir: Path, parent_root: Path, *, device: str) -> dict:
    contract = prepare(output_dir, parent_root, device=device)
    output = Path(output_dir).resolve()
    manifest = _validate_manifest(output, contract)
    parents = _parents(parent_root)
    points = {seed: {} for seed in MODEL_SEEDS}
    for seed in MODEL_SEEDS:
        for fixed_lambda in LAMBDAS:
            ledger = _read_mapping(
                output / "validation-ledger" / f"seed-{seed}" / f"{_key(fixed_lambda)}.json"
            )
            points[seed][fixed_lambda] = _instance_points(_validate_ledger(
                ledger, contract=contract, manifest=manifest, model_seed=seed,
                fixed_lambda=fixed_lambda,
                parent_sha256=parents[seed]["checkpoint_sha256"],
            ))
    aggregate = {
        fixed_lambda: {
            instance_seed: {
                field: float(fmean(points[seed][fixed_lambda][instance_seed][field] for seed in MODEL_SEEDS))
                for field in FIELDS
            }
            for instance_seed in INSTANCE_SEEDS
        }
        for fixed_lambda in LAMBDAS
    }
    differences = {
        field: _difference(aggregate[0.05], aggregate[0.0], field)
        for field in FIELDS
    }
    rehandle_pass = differences["physical_rehandles_per_100"]["simultaneous_one_sided_95_upper"] < 0.0
    mae_pass = differences["mean_absolute_error"]["simultaneous_one_sided_95_upper"] < MAE_NONINFERIORITY_MARGIN
    passed = bool(rehandle_pass and mae_pass)
    per_seed = {}
    for seed in MODEL_SEEDS:
        per_seed[str(seed)] = {
            "lambda0": _means(points[seed][0.0]),
            "lambda005": _means(points[seed][0.05]),
            "point_difference_lambda005_minus_lambda0": {
                field: float(fmean(
                    points[seed][0.05][instance_seed][field]
                    - points[seed][0.0][instance_seed][field]
                    for instance_seed in INSTANCE_SEEDS
                ))
                for field in FIELDS
            },
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "passed" if passed else "did_not_pass",
        "contract_sha256": contract["contract_sha256"],
        "instance_manifest_sha256": manifest["manifest_sha256"],
        "all_720_rows_strict_safe_complete": True,
        "aggregate": {
            "lambda0": _means(aggregate[0.0]),
            "lambda005": _means(aggregate[0.05]),
            "paired_difference_lambda005_minus_lambda0": differences,
        },
        "per_seed": per_seed,
        "predeclared_success": {
            "rehandle_simultaneous_upper_below_zero": rehandle_pass,
            "mae_simultaneous_upper_below_margin_2": mae_pass,
            "passed": passed,
        },
        "statistical_unit": "30 paired EpisodeInstances after RNG and equal-model averaging",
        "training_or_learning": False,
        "panel": "87000..87029",
        "reuses_85xxx": False,
        "reuses_86xxx": False,
    }
    report["report_sha256"] = _canonical_sha(report)
    atomic_io._atomic_json(report, output / "confirmation-report.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.output_dir, args.parent_root, device=args.device)
    elif args.action == "run":
        result = {
            "evaluation": evaluate(args.output_dir, args.parent_root, device=args.device),
            "analysis": analyze(args.output_dir, args.parent_root, device=args.device),
        }
    else:
        result = analyze(args.output_dir, args.parent_root, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
