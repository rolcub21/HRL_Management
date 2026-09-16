#!/usr/bin/env python3
"""Frozen protocol and artifact authentication for VCG V2.3 seed stability.

This module contains no environment runner.  Importing it cannot train or
evaluate a policy.  It defines the predeclared seed/RNG grid and authenticates
the completed capacity-aware repair V2 result that gates the three fresh
training replicates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence


STABILITY_PROTOCOL = "vcg_constrained_v2_3_gamma1_seed_stability_85xxx_v1"
STABILITY_SCHEMA_VERSION = 1
STABILITY_CHECKPOINT_FAMILY = (
    "vcg_constrained_vector_smdp_v2_3_gamma1_seed_stability_v1"
)
REPAIR_V2_PROTOCOL = (
    "vcg_v2_3_capacity_aware_partial_ga_repair_85xxx_development_v2"
)
REPAIR_V2_SCHEMA_VERSION = 2
PANEL_SEEDS = tuple(range(85_000, 85_012))
VALIDATION_POLICY_RNGS = tuple(range(620_000_000, 620_000_048))
MODEL_SEEDS = (11, 12, 13)
EPISODES = 200
CANDIDATE_LOOKS = (80, 100, 120, 140, 160, 180, 200)
FINAL_INSTANCE_SEEDS = frozenset(range(86_000, 86_030))
FINAL_POLICY_RNGS = frozenset(range(622_000_000, 622_000_120))
V23_DEVELOPMENT_METHOD = "vcg_constrained_v2_3_gamma1_episode160"
V11_METHOD = "vcg_dense_v1_1_selected_three_seed"
NEAREST_METHOD = "duration_aware_nearest_free"
DYNAMIC_METHOD = "duration_aware_dynamic_pslap"
REPAIRED_METHODS = (
    "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial",
    "duration_aware_pslap_ga_duration_aware_rolling_capacity_aware_partial",
    "duration_aware_pslap_ga_operational_rolling_capacity_aware_partial",
    "duration_aware_enhanced_complete_rolling_ga_capacity_aware_partial",
)
BASELINE_METHODS = (V11_METHOD, NEAREST_METHOD, DYNAMIC_METHOD, *REPAIRED_METHODS)
LEGACY_UNSAFE_METHODS = (
    "duration_aware_pslap_ga_2009_rolling",
    "duration_aware_pslap_ga_duration_aware_rolling",
    "duration_aware_pslap_ga_operational_rolling",
    "duration_aware_enhanced_complete_rolling_ga",
)


class StabilityProtocolError(ValueError):
    """Raised when a frozen seed-stability contract is violated."""


@dataclass(frozen=True)
class FrozenSeedProfile:
    model_seed: int
    train_seed_base: int
    behavior_rng_base: int
    replay_rng_seed: int

    @property
    def train_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.train_seed_base, self.train_seed_base + EPISODES))

    @property
    def behavior_rng_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.behavior_rng_base, self.behavior_rng_base + EPISODES))

    def to_manifest(self) -> dict:
        return {
            **asdict(self),
            "train_seed_range": (self.train_seed_base, self.train_seed_base + EPISODES - 1),
            "behavior_rng_range": (
                self.behavior_rng_base,
                self.behavior_rng_base + EPISODES - 1,
            ),
        }


SEED_PROFILES = {
    11: FrozenSeedProfile(11, 61_001_000, 610_001_000, 610_101_010),
    12: FrozenSeedProfile(12, 61_002_000, 610_002_000, 610_102_010),
    13: FrozenSeedProfile(13, 61_003_000, 610_003_000, 610_103_010),
}


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def digest_json(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise StabilityProtocolError(f"cannot read artifact: {path}") from error
    return digest.hexdigest()


def load_json(path: Path, *, name: str) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StabilityProtocolError(f"cannot read {name}: {path}") from error
    if not isinstance(payload, Mapping):
        raise StabilityProtocolError(f"{name} must contain a JSON object")
    return dict(payload)


def verify_self_hash(payload: Mapping, field: str, *, name: str) -> str:
    received = payload.get(field)
    if (
        not isinstance(received, str)
        or len(received) != 64
        or any(character not in "0123456789abcdef" for character in received)
    ):
        raise StabilityProtocolError(f"{name} has an invalid {field}")
    canonical = dict(payload)
    canonical.pop(field, None)
    if digest_json(canonical) != received:
        raise StabilityProtocolError(f"{name} {field} mismatch")
    return received


def atomic_json(path: Path, payload) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_equivalent(value):
    if isinstance(value, Mapping):
        return {str(key): _json_equivalent(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_equivalent(item) for item in value]
    return value


def _require(observed, expected, *, name: str) -> None:
    if _json_equivalent(observed) != _json_equivalent(expected):
        raise StabilityProtocolError(
            f"{name} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _finite_metric(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StabilityProtocolError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise StabilityProtocolError(f"{name} must be finite")
    return result


def _assert_profile_namespaces() -> None:
    if tuple(sorted(SEED_PROFILES)) != MODEL_SEEDS:
        raise StabilityProtocolError("seed profile grid drifted")
    namespaces: list[set[int]] = []
    # Seed 10 remains development-only, but its historical streams are part of
    # the disjointness proof for the fresh stability replicates.
    namespaces.extend(
        (
            set(range(61_000_000, 61_000_200)),
            set(range(610_000_000, 610_000_200)),
            {610_100_010},
        )
    )
    for seed in MODEL_SEEDS:
        profile = SEED_PROFILES[seed]
        if profile.model_seed != seed:
            raise StabilityProtocolError("model seed/profile mismatch")
        namespaces.extend(
            (
                set(profile.train_seeds),
                set(profile.behavior_rng_seeds),
                {profile.replay_rng_seed},
            )
        )
        if profile.replay_rng_seed in FINAL_POLICY_RNGS:
            raise StabilityProtocolError("replay RNG overlaps the sealed final panel")
        if set(profile.train_seeds) & FINAL_INSTANCE_SEEDS:
            raise StabilityProtocolError("training instances overlap the sealed final panel")
        if set(profile.behavior_rng_seeds) & FINAL_POLICY_RNGS:
            raise StabilityProtocolError("behavior RNG overlaps the sealed final panel")
    namespaces.append(set(PANEL_SEEDS))
    namespaces.append(set(VALIDATION_POLICY_RNGS))
    for index, left in enumerate(namespaces):
        for right in namespaces[index + 1 :]:
            if left & right:
                raise StabilityProtocolError("frozen seed/RNG namespaces overlap")


def expected_freeze_spec() -> dict:
    _assert_profile_namespaces()
    payload = {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "protocol": STABILITY_PROTOCOL,
        "status": "reviewed_and_frozen_before_seed_11_13_outcomes",
        "scientific_role": (
            "development_seed_stability_screen_on_already_opened_85xxx_panel"
        ),
        "method": {
            "method_version": "vcg_constrained_v2_3_gamma1",
            "algorithmic_change_from_seed10": False,
            "retraining_for_ga_repair": False,
            "episodes": EPISODES,
            "candidate_looks": CANDIDATE_LOOKS,
            "validation_every": 20,
            "gamma_operational": 1.0,
            "gamma_rehandle": 1.0,
            "reward_scale": 0.01,
            "rehandle_budget_per_100_required_deliveries": 20.0,
            "max_steps": 2000,
            "grid_rows": 5,
            "grid_cols": 5,
            "number_blocks": 8,
            "arrival_rate": 10.0,
            "proc_mean": 80,
            "max_hold_steps": 10,
            "max_idle_steps": 20,
            "tuning_after_seed10": False,
        },
        "development_seed10": {
            "model_seed": 10,
            "label": "development_only_not_part_of_stability_aggregate",
            "selection_episode": 160,
        },
        "fresh_training_seed_profiles": tuple(
            SEED_PROFILES[seed].to_manifest() for seed in MODEL_SEEDS
        ),
        "validation": {
            "instance_seeds": PANEL_SEEDS,
            "action_rng_range": (
                VALIDATION_POLICY_RNGS[0], VALIDATION_POLICY_RNGS[-1]
            ),
            "action_rng_count_per_instance": 4,
            "rows_per_training_seed": 48,
            "total_selected_checkpoint_rows": 144,
            "same_opened_panel_for_every_seed": True,
            "fresh_frozen_weight_only_clone_per_row": True,
            "policy": "induced_nested_regularized_lagrangian_sample",
        },
        "selection": {
            "per_seed_fail_closed_selected_best_loader_required": True,
            "same_seven_candidate_look_gate": True,
            "same_eligibility_and_lexicographic_selection_as_v2_3_seed10": True,
            "no_cross_seed_checkpoint_selection": True,
        },
        "aggregation": {
            "order": (
                "average_4_action_rngs_within_each_episode_instance",
                "average_12_episode_instances_within_each_training_seed",
                "average_3_training_seeds_with_equal_seed_weight",
            ),
            "complete_case_filtering_permitted": False,
            "baselines_reused_once_not_multiplied_by_training_seed": True,
            "dominance_coordinates": (
                "mean_absolute_error",
                "physical_rehandles_per_100_required_deliveries",
            ),
            "lower_is_better_for_both_dominance_coordinates": True,
            "scalarization_used": False,
        },
        "predeclared_pass_criterion": {
            "all_three_training_seeds_produce_eligible_checkpoint": True,
            "all_144_selected_checkpoint_rows_strict_safe_complete": True,
            "equal_seed_aggregate_non_dominated_by_every_whole_safe_baseline": True,
            "minimum_individual_non_dominated_seed_count": 2,
            "strong_result_individual_non_dominated_seed_count": 3,
            "weak_dominance_requires_one_strict_coordinate": True,
            "performance_or_confirmatory_claim_authorized": False,
        },
        "repair_v2_gate": {
            "protocol": REPAIR_V2_PROTOCOL,
            "schema_version": REPAIR_V2_SCHEMA_VERSION,
            "exact_repaired_row_count": 48,
            "development_method_capacity_or_pareto_screen_must_pass": True,
            "baseline_method_ids": BASELINE_METHODS,
            "whole_method_safety_required": True,
        },
        "sealed_final_panel": {
            "instance_seed_range": (86_000, 86_029),
            "action_rng_range": (622_000_000, 622_000_119),
            "opened": False,
            "hard_reject": True,
            "may_open_only_after_seed_selection_is_frozen": True,
        },
    }
    payload["freeze_spec_sha256"] = digest_json(payload)
    return payload


def authenticate_freeze_spec(path: Path) -> dict:
    payload = load_json(path, name="reviewed seed-stability freeze spec")
    verify_self_hash(payload, "freeze_spec_sha256", name="freeze spec")
    _require(payload, expected_freeze_spec(), name="freeze spec")
    return payload


def _method_summary(report: Mapping, method_id: str) -> dict:
    summaries = report.get("method_summaries")
    if not isinstance(summaries, list):
        raise StabilityProtocolError("repair report method summaries are unavailable")
    matches = [item for item in summaries if item.get("method_id") == method_id]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise StabilityProtocolError(f"repair report method identity invalid: {method_id}")
    return dict(matches[0])


def authenticate_repair_v2(repair_dir: Path) -> dict:
    """Authenticate the completed V2 repair output without executing a policy."""

    requested_root = Path(repair_dir).absolute()
    if requested_root.is_symlink():
        raise StabilityProtocolError("capacity-aware repair V2 root must not be a symlink")
    root = requested_root.resolve()
    if requested_root != root or not root.is_dir():
        raise StabilityProtocolError("capacity-aware repair V2 root is non-canonical")
    names = {
        "contract": "repair-contract.json",
        "preflight": "preflight.json",
        "report": "expanded-report.json",
        "runs": "expanded-runs.csv",
        "audit": "expanded-audit.json",
    }
    expected_root_names = set(names.values()) | {"run-ledger"}
    root_entries = {path.name: path for path in root.iterdir()}
    _require(set(root_entries), expected_root_names, name="repair V2 root file set")
    for name, path in root_entries.items():
        if path.is_symlink():
            raise StabilityProtocolError(f"repair V2 root entry is a symlink: {name}")
        if name == "run-ledger":
            if not path.is_dir():
                raise StabilityProtocolError("repair V2 run-ledger is not a directory")
        elif not path.is_file():
            raise StabilityProtocolError(f"repair V2 top artifact is not a file: {name}")
    paths = {label: (root / name).resolve() for label, name in names.items()}
    if any(
        not path.is_relative_to(root)
        or not path.is_file()
        or (root / names[label]).is_symlink()
        for label, path in paths.items()
    ):
        raise StabilityProtocolError("completed capacity-aware repair V2 artifacts are missing")
    top_raw_before = {label: sha256_file(path) for label, path in paths.items()}
    contract = load_json(paths["contract"], name="repair V2 contract")
    preflight = load_json(paths["preflight"], name="repair V2 preflight")
    report = load_json(paths["report"], name="repair V2 report")
    audit = load_json(paths["audit"], name="repair V2 audit")
    contract_sha = verify_self_hash(contract, "contract_sha256", name="repair V2 contract")
    report_sha = verify_self_hash(report, "report_sha256", name="repair V2 report")
    audit_sha = verify_self_hash(audit, "audit_sha256", name="repair V2 audit")
    for label, payload in (
        ("contract", contract), ("preflight", preflight),
        ("report", report), ("audit", audit),
    ):
        _require(payload.get("protocol"), REPAIR_V2_PROTOCOL, name=f"repair {label} protocol")
        _require(payload.get("schema_version"), REPAIR_V2_SCHEMA_VERSION, name=f"repair {label} schema")
        _require(payload.get("final_86xxx_panel_opened"), False, name=f"repair {label} final flag")
        if "performance_claim_authorized" in payload:
            _require(payload.get("performance_claim_authorized"), False, name=f"repair {label} claim flag")
    _require(contract.get("panel_seeds"), list(PANEL_SEEDS), name="repair panel")
    _require(preflight.get("status"), "complete", name="repair completion status")
    _require(preflight.get("repair_contract_sha256"), contract_sha, name="preflight contract binding")
    _require(preflight.get("loaded_or_executed_repaired_rows"), 48, name="repair loaded row count")
    _require(preflight.get("expected_repaired_rows"), 48, name="repair expected row count")
    _require(preflight.get("missing_repaired_rows"), [], name="repair missing rows")
    _require(report.get("repair_contract_sha256"), contract_sha, name="report contract binding")
    _require(report.get("run_count"), 204, name="repair report row count")
    _require(report.get("new_repaired_run_count"), 48, name="repair report new row count")
    screen = report.get("development_method_capacity_or_pareto_screen")
    if not isinstance(screen, Mapping):
        screen = report.get("seed_stability_screen")
    if not isinstance(screen, Mapping):
        raise StabilityProtocolError("repair V2 screen is unavailable")
    for field in (
        "passed", "authenticated_selected_v2_3_source",
        "all_48_v2_3_rows_strict_safe_complete",
    ):
        _require(screen.get(field), True, name=f"repair screen {field}")
    _require(
        screen.get("dominance_coordinates"),
        ["mean_absolute_error", "physical_rehandles_per_100_required_deliveries"],
        name="repair dominance coordinates",
    )
    _require(screen.get("safety_eligible_weak_dominators_with_one_strict"), [], name="repair V2.3 dominators")

    v23 = _method_summary(report, V23_DEVELOPMENT_METHOD)
    _require(v23.get("whole_method_numeric_eligible"), True, name="repair V2.3 eligibility")
    _require(v23.get("source_row_count"), 48, name="repair V2.3 row count")
    _require(v23.get("expected_source_row_count"), 48, name="repair V2.3 expected rows")
    _require(v23.get("grid_issues"), [], name="repair V2.3 grid issues")
    _require(v23.get("row_safety_issues"), [], name="repair V2.3 row safety")

    baseline_summaries = []
    for method in BASELINE_METHODS:
        summary = _method_summary(report, method)
        _require(summary.get("whole_method_numeric_eligible"), True, name=f"baseline eligibility {method}")
        _require(summary.get("grid_issues"), [], name=f"baseline grid {method}")
        _require(summary.get("row_safety_issues"), [], name=f"baseline row safety {method}")
        metrics = summary.get("metrics")
        if not isinstance(metrics, Mapping):
            raise StabilityProtocolError(f"baseline metrics unavailable: {method}")
        for metric in (
            "mean_absolute_error", "physical_rehandles_per_100_required_deliveries",
            "mean_dense_objective_return", "mean_steps",
        ):
            _finite_metric(metrics.get(metric), name=f"{method}:{metric}")
        baseline_summaries.append(summary)
    _require(
        report.get("whole_method_safety_exclusions"),
        list(LEGACY_UNSAFE_METHODS),
        name="historical whole-method exclusions",
    )

    _require(audit.get("repair_contract_sha256"), contract_sha, name="audit contract binding")
    _require(audit.get("new_repaired_grid_row_count"), 48, name="audit repaired row count")
    for field in (
        "repair_sources_unchanged", "authenticated_learned_sources_unchanged",
        "historical_baseline_ledgers_unchanged",
    ):
        _require(audit.get(field), True, name=f"repair audit {field}")
    _require(
        audit.get("expanded_report_raw_sha256"), sha256_file(paths["report"]),
        name="repair report raw SHA binding",
    )
    _require(
        audit.get("expanded_runs_raw_sha256"), sha256_file(paths["runs"]),
        name="repair CSV raw SHA binding",
    )

    manifest = audit.get("repair_ledger_manifest")
    expected_keys = {(method, seed) for method in REPAIRED_METHODS for seed in PANEL_SEEDS}
    if not isinstance(manifest, list) or len(manifest) != len(expected_keys):
        raise StabilityProtocolError("repair ledger manifest is not the exact 4x12 grid")
    observed_keys = set()
    requested_ledger_root = root / "run-ledger"
    ledger_root = requested_ledger_root.resolve()
    if (
        requested_ledger_root.is_symlink()
        or ledger_root.parent != root
        or not ledger_root.is_dir()
    ):
        raise StabilityProtocolError("repair run-ledger root is non-canonical")
    method_entries = {path.name: path for path in ledger_root.iterdir()}
    _require(
        set(method_entries), set(REPAIRED_METHODS), name="repair ledger method directory set"
    )
    expected_ledger_names = {f"seed-{seed}.json" for seed in PANEL_SEEDS}
    for method, method_dir in method_entries.items():
        if method_dir.is_symlink() or not method_dir.is_dir():
            raise StabilityProtocolError(
                f"repair ledger method directory is invalid: {method}"
            )
        ledger_entries = {path.name: path for path in method_dir.iterdir()}
        _require(
            set(ledger_entries),
            expected_ledger_names,
            name=f"repair ledger file set {method}",
        )
        for name, path in ledger_entries.items():
            if path.is_symlink() or not path.is_file():
                raise StabilityProtocolError(
                    f"repair ledger artifact is invalid: {method}/{name}"
                )
    ledger_paths = {}
    for item in manifest:
        if not isinstance(item, Mapping):
            raise StabilityProtocolError("repair ledger manifest record is invalid")
        try:
            method, seed_text = str(item.get("run_key")).rsplit(":", 1)
            seed = int(seed_text)
        except (TypeError, ValueError) as error:
            raise StabilityProtocolError("repair ledger run key is invalid") from error
        key = (method, seed)
        if key not in expected_keys or key in observed_keys:
            raise StabilityProtocolError(f"repair ledger grid key is invalid: {key!r}")
        observed_keys.add(key)
        method_dir = ledger_root / method
        requested_ledger = method_dir / f"seed-{seed}.json"
        ledger_path = requested_ledger.resolve()
        if (
            method_dir.is_symlink()
            or not method_dir.is_dir()
            or requested_ledger.is_symlink()
            or ledger_path.parent != method_dir
            or not ledger_path.is_file()
        ):
            raise StabilityProtocolError(f"repair ledger is missing: {key!r}")
        ledger_paths[f"{method}:{seed}"] = ledger_path
        _require(sha256_file(ledger_path), item.get("raw_sha256"), name=f"repair ledger raw SHA {key}")
        ledger = load_json(ledger_path, name=f"repair ledger {key}")
        _require(verify_self_hash(ledger, "ledger_sha256", name=f"repair ledger {key}"), item.get("ledger_sha256"), name=f"repair ledger self SHA {key}")
        _require(ledger.get("protocol"), REPAIR_V2_PROTOCOL, name=f"repair ledger protocol {key}")
        _require(ledger.get("schema_version"), REPAIR_V2_SCHEMA_VERSION, name=f"repair ledger schema {key}")
        _require(ledger.get("run_key"), f"{method}:{seed}", name=f"repair ledger key {key}")
        run = ledger.get("run")
        if not isinstance(run, Mapping):
            raise StabilityProtocolError(f"repair ledger run is unavailable: {key!r}")
        _require(run.get("method_id"), method, name=f"repair run method {key}")
        _require(run.get("instance_seed"), seed, name=f"repair run seed {key}")
        _require(run.get("strict_method_success"), True, name=f"repair run safety {key}")
        _require(run.get("method_failure_reason"), None, name=f"repair run failure {key}")
    _require(observed_keys, expected_keys, name="repair ledger exact grid")
    # Close the read/verify window before returning a trust root to prepare or
    # analysis.  A later phase repeats the complete authentication as well.
    for run_key, ledger_path in ledger_paths.items():
        record = next(item for item in manifest if item.get("run_key") == run_key)
        _require(
            sha256_file(ledger_path),
            record.get("raw_sha256"),
            name=f"repair ledger post-auth raw SHA {run_key}",
        )
    for label, path in paths.items():
        _require(
            sha256_file(path),
            top_raw_before[label],
            name=f"repair V2 top artifact changed during authentication: {label}",
        )

    return {
        "root": root,
        "paths": paths,
        "raw_sha256": top_raw_before,
        "contract_sha256": contract_sha,
        "report_sha256": report_sha,
        "audit_sha256": audit_sha,
        "contract": contract,
        "report": report,
        "audit": audit,
        "repair_ledger_raw_sha256": {
            run_key: sha256_file(path)
            for run_key, path in sorted(ledger_paths.items())
        },
        "baseline_summaries": tuple(baseline_summaries),
    }


def validate_no_final_panel_values(*, instance_seeds: Sequence[int], action_rngs: Sequence[int]) -> None:
    if set(map(int, instance_seeds)) & FINAL_INSTANCE_SEEDS:
        raise StabilityProtocolError("86xxx final instances remain sealed")
    if set(map(int, action_rngs)) & FINAL_POLICY_RNGS:
        raise StabilityProtocolError("622000xxx final action RNGs remain sealed")


__all__ = [
    "BASELINE_METHODS", "CANDIDATE_LOOKS", "EPISODES", "FINAL_INSTANCE_SEEDS",
    "FINAL_POLICY_RNGS", "FrozenSeedProfile", "MODEL_SEEDS", "PANEL_SEEDS",
    "REPAIRED_METHODS", "REPAIR_V2_PROTOCOL", "SEED_PROFILES",
    "STABILITY_CHECKPOINT_FAMILY", "STABILITY_PROTOCOL",
    "STABILITY_SCHEMA_VERSION", "StabilityProtocolError",
    "VALIDATION_POLICY_RNGS", "atomic_json", "authenticate_freeze_spec",
    "authenticate_repair_v2", "canonical_json_bytes", "digest_json",
    "expected_freeze_spec", "load_json", "sha256_file",
    "validate_no_final_panel_values", "verify_self_hash",
]
