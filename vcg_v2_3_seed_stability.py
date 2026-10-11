#!/usr/bin/env python3
"""Prepare, execute, and analyze the frozen VCG V2.3 seed-stability screen."""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Mapping, Optional, Sequence

import train_vcg_constrained_v2_3 as v23_base
from train_vcg_constrained_v2_3_stability import (
    StabilityRunBinding,
    authenticate_seed_training_contract,
    load_seed_best_candidate,
    load_seed_candidate_look,
    run_frozen_stability_seed,
)
from vcg_v2_3_seed_stability_protocol import (
    BASELINE_METHODS,
    CANDIDATE_LOOKS,
    MODEL_SEEDS,
    PANEL_SEEDS,
    SEED_PROFILES,
    STABILITY_PROTOCOL,
    STABILITY_SCHEMA_VERSION,
    VALIDATION_POLICY_RNGS,
    StabilityProtocolError,
    atomic_json,
    authenticate_freeze_spec,
    authenticate_repair_v2,
    digest_json,
    load_json,
    sha256_file,
    validate_no_final_panel_values,
    verify_self_hash,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPAIR_DIR = PROJECT_ROOT / "results/vcg-v2-3-capacity-aware-ga-repair-v2-85k"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/vcg-v2-3-seed-stability-85k"
DEFAULT_FREEZE_SPEC = (
    PROJECT_ROOT / "experiments/vcg_v2_3_seed_stability_85k/freeze-spec.json"
)
RUN_MANIFEST_NAME = "stability-run-manifest.json"
COMPLETION_NAME = "seed-completion.json"
REPORT_NAME = "stability-report.json"
AUDIT_NAME = "stability-audit.json"
_SEED_ROOT_REQUIRED_FILES = frozenset(
    {
        "training-contract.json",
        "training-summary.json",
        "candidate-look-checkpoint-manifest.json",
        "validation-instance-manifest.json",
        "latest.pth",
    }
)
_SEED_ROOT_OPTIONAL_FILES = frozenset(
    {"best-development-candidate.pth", COMPLETION_NAME}
)
_SEED_ROOT_DIRECTORIES = frozenset(
    {"candidate-look-checkpoints", "validation-ledger"}
)
_TRAINING_SUMMARY_KEYS = frozenset(
    {
        "status", "method_version", "completed_training_episodes",
        "completed_blocks", "dual_state", "dual_update_history",
        "terminal_dual_proposal", "development_candidate_eligible",
        "deployment_checkpoint_eligible", "best_validation",
        "validation_history", "candidate_look_diagnostic_checkpoints",
        "candidate_look_checkpoint_manifest", "best_development_candidate",
        "latest_checkpoint", "training_contract",
        "validation_instance_manifest", "prospective_83xxx_panel_opened",
        "prior_83xxx_usage", "83xxx_globally_untouched_claimed",
        "final_86xxx_panel_opened",
    }
)


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


def _canonical_output_root(output_dir: Path, *, require_exists: bool) -> Path:
    requested = Path(output_dir).absolute()
    if requested.is_symlink():
        raise StabilityProtocolError("seed-stability output root must not be a symlink")
    resolved = requested.resolve()
    if requested != resolved:
        raise StabilityProtocolError("seed-stability output root must be canonical")
    if require_exists and not resolved.is_dir():
        raise StabilityProtocolError("seed-stability output root is missing")
    if resolved.exists() and not resolved.is_dir():
        raise StabilityProtocolError("seed-stability output root is not a directory")
    return resolved


def _canonical_seed_dir(
    output_dir: Path, model_seed: int, *, require_exists: bool,
) -> Path:
    if type(model_seed) is not int or model_seed not in MODEL_SEEDS:
        raise StabilityProtocolError(f"model seed must be one of {MODEL_SEEDS}")
    root = _canonical_output_root(output_dir, require_exists=True)
    requested = root / f"seed-{model_seed}"
    if requested.is_symlink():
        raise StabilityProtocolError(f"seed-{model_seed} directory must not be a symlink")
    resolved = requested.resolve()
    if resolved.parent != root or resolved != requested:
        raise StabilityProtocolError(f"seed-{model_seed} is not a canonical direct child")
    if require_exists and not resolved.is_dir():
        raise StabilityProtocolError(f"seed-{model_seed} directory is missing")
    if resolved.exists() and not resolved.is_dir():
        raise StabilityProtocolError(f"seed-{model_seed} path is not a directory")
    return resolved


def _require_regular_file(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise StabilityProtocolError(f"{name} must be a regular non-symlink file")


def _validate_output_tree(output_dir: Path) -> None:
    """Reject unbound top-level artifacts and all non-regular entries."""

    root = _canonical_output_root(output_dir, require_exists=True)
    expected_seed_names = {f"seed-{seed}" for seed in MODEL_SEEDS}
    allowed_file_names = {RUN_MANIFEST_NAME, REPORT_NAME, AUDIT_NAME}
    allowed_names = allowed_file_names | expected_seed_names
    entries = {path.name: path for path in root.iterdir()}
    unexpected = set(entries) - allowed_names
    if unexpected:
        raise StabilityProtocolError(
            f"stability output contains unexpected entries: {sorted(unexpected)!r}"
        )
    if RUN_MANIFEST_NAME not in entries:
        raise StabilityProtocolError("seed-stability run manifest is missing")
    for name, path in entries.items():
        if path.is_symlink():
            raise StabilityProtocolError(
                f"stability output contains a forbidden symlink: {name}"
            )
        if name in expected_seed_names:
            if not path.is_dir() or path.resolve().parent != root:
                raise StabilityProtocolError(
                    f"stability seed entry is not a direct regular directory: {name}"
                )
        elif not path.is_file():
            raise StabilityProtocolError(
                f"stability output entry is not a regular file: {name}"
            )


def _seed_tree_raw_sha256(
    seed_dir: Path,
    *,
    allow_completion: bool,
) -> dict[str, str]:
    """Validate the exact seed tree and hash every allowed regular file."""

    seed_dir = Path(seed_dir).absolute()
    if seed_dir.is_symlink() or not seed_dir.is_dir() or seed_dir.resolve() != seed_dir:
        raise StabilityProtocolError("seed artifact root is non-canonical")
    entries = {path.name: path for path in seed_dir.iterdir()}
    allowed_files = set(_SEED_ROOT_REQUIRED_FILES) | {
        "best-development-candidate.pth"
    }
    if allow_completion:
        allowed_files.add(COMPLETION_NAME)
    allowed = allowed_files | set(_SEED_ROOT_DIRECTORIES)
    if set(entries) - allowed:
        raise StabilityProtocolError(
            f"seed artifact root contains unexpected entries: {sorted(set(entries) - allowed)!r}"
        )
    missing = set(_SEED_ROOT_REQUIRED_FILES) - set(entries)
    if missing or set(_SEED_ROOT_DIRECTORIES) - set(entries):
        raise StabilityProtocolError(
            f"seed artifact root is incomplete: {sorted(missing)!r}"
        )
    hashes = {}
    for name in sorted(set(entries) & allowed_files):
        path = entries[name]
        _require_regular_file(path, name=f"seed artifact {name}")
        if name != COMPLETION_NAME:
            hashes[name] = sha256_file(path)

    candidate_dir = entries["candidate-look-checkpoints"]
    validation_dir = entries["validation-ledger"]
    for label, directory in (
        ("candidate-look-checkpoints", candidate_dir),
        ("validation-ledger", validation_dir),
    ):
        if directory.is_symlink() or not directory.is_dir():
            raise StabilityProtocolError(f"{label} must be a regular directory")
    expected_candidates = {
        f"episode-{episode:04d}-model-only.pth" for episode in CANDIDATE_LOOKS
    }
    candidate_entries = {path.name: path for path in candidate_dir.iterdir()}
    _require(set(candidate_entries), expected_candidates, name="candidate checkpoint file set")
    for name, path in sorted(candidate_entries.items()):
        _require_regular_file(path, name=f"candidate checkpoint {name}")
        hashes[f"candidate-look-checkpoints/{name}"] = sha256_file(path)

    expected_ledgers = {
        f"episode-{episode:04d}.json" for episode in range(20, 201, 20)
    }
    validation_entries = {path.name: path for path in validation_dir.iterdir()}
    _require(set(validation_entries), expected_ledgers, name="validation ledger file set")
    for name, path in sorted(validation_entries.items()):
        _require_regular_file(path, name=f"validation ledger {name}")
        hashes[f"validation-ledger/{name}"] = sha256_file(path)
    return hashes


def _source_paths() -> dict[str, Path]:
    # Frozen local import closure for the trainer, controller, environment,
    # exact-safety verifier, Hold rule, objective audit, and stability facade.
    # Keeping the explicit list in the hashed workflow makes additions and
    # removals reviewable; it avoids an over-broad repository snapshot while
    # ensuring a transitive algorithm dependency cannot drift after prepare.
    names = (
        "GA/helper_functions.py",
        "PSLAP/dynamic_yard.py",
        "PSLAP/neutral_protocol.py",
        "PSLAP/viability.py",
        "PSLAP/viability_candidates.py",
        "PSLAP/viability_candidates_hold_v2.py",
        "PSLAP/viability_critic.py",
        "PSLAP/viability_dataset.py",
        "PSLAP/viability_filter.py",
        "PSLAP/viability_prioritizer.py",
        "PSLAP/yard_graph.py",
        "environment.py",
        "example/Options/DirectDeliverOption.py",
        "example/Options/ExplicitAcceptOption.py",
        "example/Options/ExplicitDeferOption.py",
        "example/Options/ExplicitHoldOption.py",
        "example/Options/ReconfigureOption.py",
        "example/__init__.py",
        "example/block_instance.py",
        "example/episode_instance.py",
        "example/helper/timing_metrics.py",
        "example/small_rooms_env.py",
        "example/yard_geometry.py",
        "option.py",
        "train_vcg_constrained_v2.py",
        "train_vcg_constrained_v2_1.py",
        "train_vcg_constrained_v2_3.py",
        "train_vcg_constrained_v2_3_stability.py",
        "vcg_objective_audit.py",
        "vcg_v2_3_seed_stability.py",
        "vcg_v2_3_seed_stability_protocol.py",
        "viability_graph_constrained_v2.py",
        "viability_graph_constrained_v2_1.py",
        "viability_graph_constrained_v2_3.py",
        "viability_graph_episodic_audit.py",
        "viability_graph_hierarchy.py",
    )
    return {name: PROJECT_ROOT / name for name in names}


def _source_hashes() -> dict[str, str]:
    result = {}
    for name, path in _source_paths().items():
        if path.is_symlink() or not path.is_file():
            raise StabilityProtocolError(f"seed-stability source is missing: {path}")
        result[name] = sha256_file(path)
    return result


def _repair_roots(repair: Mapping) -> dict:
    return {
        "root": str(Path(repair["root"]).resolve()),
        "raw_sha256": dict(sorted(repair["raw_sha256"].items())),
        "contract_sha256": repair["contract_sha256"],
        "report_sha256": repair["report_sha256"],
        "audit_sha256": repair["audit_sha256"],
        "repair_ledger_raw_sha256": dict(
            sorted(repair["repair_ledger_raw_sha256"].items())
        ),
        "screen_passed": True,
        "whole_safe_baseline_method_ids": BASELINE_METHODS,
    }


def build_run_manifest(
    *, output_dir: Path, freeze_spec_path: Path, freeze_spec: Mapping,
    repair: Mapping,
) -> dict:
    output_dir = _canonical_output_root(output_dir, require_exists=False)
    payload = {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "protocol": STABILITY_PROTOCOL,
        "status": "frozen_ready_for_seed_runs",
        "development_only": True,
        "performance_claim_authorized": False,
        "confirmatory_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "freeze_spec": {
            "path": str(Path(freeze_spec_path).resolve()),
            "raw_sha256": sha256_file(Path(freeze_spec_path)),
            "freeze_spec_sha256": freeze_spec["freeze_spec_sha256"],
        },
        "repair_v2": _repair_roots(repair),
        "implementation_source_sha256": _source_hashes(),
        "implementation_source_scope": (
            "explicit_frozen_transitive_local_import_closure_for_v2_3_training_"
            "controller_environment_objective_and_exact_safety"
        ),
        "implementation_source_count": len(_source_paths()),
        "output_root": str(output_dir),
        "seed_output_directories": {
            str(seed): f"seed-{seed}" for seed in MODEL_SEEDS
        },
        "model_seeds": MODEL_SEEDS,
        "selection_outcomes_observed_when_frozen": False,
        "seed10_included_in_stability_aggregate": False,
        "baselines_reused_once": True,
    }
    payload["manifest_sha256"] = digest_json(payload)
    return payload


def _binding_from_manifest(manifest: Mapping) -> StabilityRunBinding:
    repair = manifest.get("repair_v2")
    freeze = manifest.get("freeze_spec")
    if not isinstance(repair, Mapping) or not isinstance(freeze, Mapping):
        raise StabilityProtocolError("run manifest trust roots are unavailable")
    return StabilityRunBinding(
        freeze_spec_sha256=str(freeze.get("freeze_spec_sha256")),
        run_manifest_sha256=str(manifest.get("manifest_sha256")),
        repair_v2_contract_sha256=str(repair.get("contract_sha256")),
        repair_v2_report_sha256=str(repair.get("report_sha256")),
        implementation_source_sha256=dict(
            manifest.get("implementation_source_sha256", {})
        ),
    )


def prepare_protocol(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR,
    freeze_spec_path: Path = DEFAULT_FREEZE_SPEC,
    repair_dir: Path = DEFAULT_REPAIR_DIR,
) -> dict:
    """Freeze the exact seed run after the repaired comparison passes."""

    freeze_spec = authenticate_freeze_spec(freeze_spec_path)
    repair = authenticate_repair_v2(repair_dir)
    output_dir = _canonical_output_root(output_dir, require_exists=False)
    manifest_path = output_dir / RUN_MANIFEST_NAME
    expected = build_run_manifest(
        output_dir=output_dir,
        freeze_spec_path=freeze_spec_path,
        freeze_spec=freeze_spec,
        repair=repair,
    )
    if output_dir.exists():
        for path in output_dir.iterdir():
            if path.is_symlink():
                raise StabilityProtocolError(
                    f"stability output contains a forbidden symlink: {path.name}"
                )
        unexpected = [
            path for path in output_dir.iterdir()
            if path.name not in {
                RUN_MANIFEST_NAME, REPORT_NAME, AUDIT_NAME,
                *(f"seed-{seed}" for seed in MODEL_SEEDS),
            }
        ]
        if unexpected:
            raise StabilityProtocolError(
                f"stability output contains unexpected paths: {unexpected!r}"
            )
        for seed in MODEL_SEEDS:
            seed_path = output_dir / f"seed-{seed}"
            if seed_path.exists() and not seed_path.is_dir():
                raise StabilityProtocolError(f"seed-{seed} must be a directory")
        for name in (RUN_MANIFEST_NAME, REPORT_NAME, AUDIT_NAME):
            path = output_dir / name
            if path.exists() and not path.is_file():
                raise StabilityProtocolError(f"{name} must be a regular file")
    if manifest_path.is_file():
        observed = load_json(manifest_path, name="seed-stability run manifest")
        verify_self_hash(observed, "manifest_sha256", name="seed-stability run manifest")
        _require(observed, expected, name="existing seed-stability run manifest")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise StabilityProtocolError(
                "cannot adopt stability artifacts without their frozen run manifest"
            )
        atomic_json(manifest_path, expected)
    return expected


def authenticate_run_manifest(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR,
    freeze_spec_path: Path = DEFAULT_FREEZE_SPEC,
    repair_dir: Path = DEFAULT_REPAIR_DIR,
) -> tuple[dict, StabilityRunBinding, dict]:
    freeze_spec = authenticate_freeze_spec(freeze_spec_path)
    repair = authenticate_repair_v2(repair_dir)
    output_dir = _canonical_output_root(output_dir, require_exists=True)
    _validate_output_tree(output_dir)
    manifest_path = output_dir / RUN_MANIFEST_NAME
    _require_regular_file(manifest_path, name="seed-stability run manifest")
    observed = load_json(manifest_path, name="seed-stability run manifest")
    verify_self_hash(observed, "manifest_sha256", name="seed-stability run manifest")
    expected = build_run_manifest(
        output_dir=output_dir,
        freeze_spec_path=freeze_spec_path,
        freeze_spec=freeze_spec,
        repair=repair,
    )
    _require(observed, expected, name="seed-stability run manifest")
    return observed, _binding_from_manifest(observed), repair


def _reauthenticate_run_trust(
    *,
    output_dir: Path,
    freeze_spec_path: Path,
    repair_dir: Path,
    expected_manifest: Mapping,
    expected_binding: StabilityRunBinding,
    expected_repair: Mapping,
) -> tuple[dict, StabilityRunBinding, dict]:
    """Repeat every immutable trust-root check after a long/read phase."""

    observed_manifest, observed_binding, observed_repair = authenticate_run_manifest(
        output_dir=output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
    )
    _require(observed_manifest, expected_manifest, name="post-phase run manifest")
    if observed_binding != expected_binding:
        raise StabilityProtocolError("post-phase stability binding changed")
    _require(
        _repair_roots(observed_repair),
        _repair_roots(expected_repair),
        name="post-phase repair V2 trust roots",
    )
    return observed_manifest, observed_binding, observed_repair


def _validation_ledger_manifest(seed_dir: Path) -> tuple[dict, ...]:
    requested_root = seed_dir / "validation-ledger"
    if requested_root.is_symlink():
        raise StabilityProtocolError("validation ledger directory must not be a symlink")
    root = requested_root.resolve()
    expected_names = tuple(f"episode-{episode:04d}.json" for episode in range(20, 201, 20))
    if not root.is_dir() or root.parent != seed_dir:
        raise StabilityProtocolError("validation ledger directory is missing")
    observed_names = tuple(sorted(path.name for path in root.iterdir()))
    _require(observed_names, expected_names, name="validation ledger file set")
    exact_keys = {
        "schema_version", "validation_protocol", "checkpoint_episode",
        "validated_lambda", "schedule_state", "row_count",
        "complete_case_filtering_used", "validation_batch_audit", "rows",
        "ledger_sha256",
    }
    records = []
    for episode, name in zip(range(20, 201, 20), expected_names):
        path = root / name
        _require_regular_file(path, name=f"validation ledger {name}")
        ledger = load_json(path, name=f"validation ledger {name}")
        _require(set(ledger), exact_keys, name=f"validation ledger schema {name}")
        if type(ledger.get("schema_version")) is not int or ledger["schema_version"] != 1:
            raise StabilityProtocolError(f"validation ledger {name} schema version is invalid")
        if (
            type(ledger.get("checkpoint_episode")) is not int
            or ledger["checkpoint_episode"] != episode
        ):
            raise StabilityProtocolError(f"validation ledger {name} episode is invalid")
        if type(ledger.get("row_count")) is not int or ledger["row_count"] != 48:
            raise StabilityProtocolError(f"validation ledger {name} row count is invalid")
        rows = ledger.get("rows")
        if (
            not isinstance(rows, list)
            or len(rows) != 48
            or any(not isinstance(row, Mapping) for row in rows)
        ):
            raise StabilityProtocolError(f"validation ledger {name} rows are invalid")
        if ledger.get("validation_protocol") != v23_base.VALIDATION_PROTOCOL:
            raise StabilityProtocolError(f"validation ledger {name} protocol is invalid")
        if ledger.get("complete_case_filtering_used") is not False:
            raise StabilityProtocolError(f"validation ledger {name} permits filtering")
        if not isinstance(ledger.get("schedule_state"), Mapping):
            raise StabilityProtocolError(f"validation ledger {name} schedule is invalid")
        if not isinstance(ledger.get("validation_batch_audit"), Mapping):
            raise StabilityProtocolError(f"validation ledger {name} batch audit is invalid")
        validated_lambda = ledger.get("validated_lambda")
        if (
            isinstance(validated_lambda, bool)
            or not isinstance(validated_lambda, (int, float))
            or not math.isfinite(float(validated_lambda))
        ):
            raise StabilityProtocolError(f"validation ledger {name} lambda is invalid")
        ledger_sha = verify_self_hash(ledger, "ledger_sha256", name=f"validation ledger {name}")
        records.append(
            {
                "relative_path": f"validation-ledger/{name}",
                "raw_sha256": sha256_file(path),
                "ledger_sha256": ledger_sha,
                "checkpoint_episode": ledger["checkpoint_episode"],
                "row_count": ledger["row_count"],
            }
        )
    return tuple(records)


def reconstruct_seed_completion(
    model_seed: int,
    *, seed_dir: Path,
    run_manifest: Mapping,
    binding: StabilityRunBinding,
    pinned_completion: Optional[Mapping] = None,
) -> tuple[dict, Optional[dict]]:
    """Authenticate all seed artifacts and reconstruct its completion record."""

    if type(model_seed) is not int or model_seed not in MODEL_SEEDS:
        raise StabilityProtocolError(f"model seed must be one of {MODEL_SEEDS}")
    output_root = _canonical_output_root(
        Path(run_manifest["output_root"]), require_exists=True,
    )
    expected_seed_dir = _canonical_seed_dir(
        output_root, model_seed, require_exists=True,
    )
    requested_seed_dir = Path(seed_dir).absolute()
    if requested_seed_dir.is_symlink() or requested_seed_dir != expected_seed_dir:
        raise StabilityProtocolError("seed output directory is non-canonical")
    seed_dir = expected_seed_dir
    tree_before = _seed_tree_raw_sha256(
        seed_dir,
        allow_completion=pinned_completion is not None,
    )
    if pinned_completion is not None:
        pinned_tree = pinned_completion.get("seed_tree_raw_sha256")
        if not isinstance(pinned_tree, Mapping):
            raise StabilityProtocolError(
                f"seed {model_seed} completion lacks a seed-tree trust root"
            )
        _require(
            tree_before,
            pinned_tree,
            name=f"seed {model_seed} pinned seed artifact tree",
        )
    paths = {
        "contract": seed_dir / "training-contract.json",
        "summary": seed_dir / "training-summary.json",
        "candidate_manifest": seed_dir / "candidate-look-checkpoint-manifest.json",
        "validation_manifest": seed_dir / "validation-instance-manifest.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise StabilityProtocolError(f"seed {model_seed} training artifacts are incomplete")
    contract, resolved_contract, contract_sha = authenticate_seed_training_contract(
        model_seed, paths["contract"], binding=binding,
    )
    _require(resolved_contract, paths["contract"].resolve(), name="seed contract path")
    if pinned_completion is not None:
        _require(
            contract_sha,
            pinned_completion.get("training_contract_sha256"),
            name=f"seed {model_seed} pinned contract self SHA",
        )
        _require(
            sha256_file(paths["contract"]),
            pinned_completion.get("training_contract_raw_sha256"),
            name=f"seed {model_seed} pinned contract raw SHA",
        )
    summary = load_json(paths["summary"], name=f"seed {model_seed} training summary")
    if pinned_completion is not None:
        _require(
            sha256_file(paths["summary"]),
            pinned_completion.get("training_summary_raw_sha256"),
            name=f"seed {model_seed} pinned training summary raw SHA",
        )
    _require(set(summary), _TRAINING_SUMMARY_KEYS, name=f"seed {model_seed} summary schema")
    _require(summary.get("status"), "complete", name=f"seed {model_seed} training status")
    _require(
        summary.get("method_version"),
        "vcg_constrained_v2_3_gamma1",
        name=f"seed {model_seed} method version",
    )
    _require(summary.get("completed_training_episodes"), 200, name=f"seed {model_seed} episodes")
    _require(summary.get("completed_blocks"), 20, name=f"seed {model_seed} blocks")
    _require(
        summary.get("deployment_checkpoint_eligible"),
        False,
        name=f"seed {model_seed} deployment flag",
    )
    _require(
        summary.get("prospective_83xxx_panel_opened"),
        False,
        name=f"seed {model_seed} prospective flag",
    )
    _require(
        summary.get("83xxx_globally_untouched_claimed"),
        False,
        name=f"seed {model_seed} historical claim flag",
    )
    _require(summary.get("final_86xxx_panel_opened"), False, name=f"seed {model_seed} final flag")
    summary_paths = {
        "training_contract": paths["contract"],
        "validation_instance_manifest": paths["validation_manifest"],
        "candidate_look_checkpoint_manifest": paths["candidate_manifest"],
        "latest_checkpoint": seed_dir / "latest.pth",
    }
    for field, expected_path in summary_paths.items():
        observed_path = summary.get(field)
        if (
            not isinstance(observed_path, str)
            or Path(observed_path).absolute() != expected_path
            or Path(observed_path).is_symlink()
        ):
            raise StabilityProtocolError(
                f"seed {model_seed} summary {field} path is inconsistent"
            )
    _require_regular_file(
        seed_dir / "latest.pth", name=f"seed {model_seed} latest audit checkpoint"
    )
    validation_history = summary.get("validation_history")
    if (
        not isinstance(validation_history, list)
        or len(validation_history) != 10
        or [item.get("checkpoint_episode") for item in validation_history]
        != list(range(20, 201, 20))
    ):
        raise StabilityProtocolError(f"seed {model_seed} validation history is invalid")
    if not isinstance(summary.get("dual_state"), Mapping):
        raise StabilityProtocolError(f"seed {model_seed} summary dual state is invalid")
    if not isinstance(summary.get("dual_update_history"), list):
        raise StabilityProtocolError(f"seed {model_seed} dual history is invalid")
    if not isinstance(summary.get("terminal_dual_proposal"), Mapping):
        raise StabilityProtocolError(f"seed {model_seed} terminal dual proposal is invalid")
    candidate_manifest = load_json(
        paths["candidate_manifest"], name=f"seed {model_seed} candidate manifest"
    )
    candidate_manifest_sha = verify_self_hash(
        candidate_manifest, "manifest_sha256", name=f"seed {model_seed} candidate manifest"
    )
    if pinned_completion is not None:
        _require(
            candidate_manifest_sha,
            pinned_completion.get("candidate_manifest_sha256"),
            name=f"seed {model_seed} pinned candidate manifest self SHA",
        )
        _require(
            sha256_file(paths["candidate_manifest"]),
            pinned_completion.get("candidate_manifest_raw_sha256"),
            name=f"seed {model_seed} pinned candidate manifest raw SHA",
        )
    _require(candidate_manifest.get("complete"), True, name=f"seed {model_seed} candidate completion")
    _require(candidate_manifest.get("saved_candidate_look_episodes"), list(CANDIDATE_LOOKS), name=f"seed {model_seed} candidate looks")
    _require(candidate_manifest.get("final_86xxx_panel_opened"), False, name=f"seed {model_seed} candidate final flag")
    artifacts = candidate_manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(CANDIDATE_LOOKS):
        raise StabilityProtocolError(f"seed {model_seed} candidate grid is incomplete")
    _require(
        summary.get("candidate_look_diagnostic_checkpoints"),
        artifacts,
        name=f"seed {model_seed} summary/candidate manifest records",
    )
    candidate_records = []
    authenticated_candidates = []
    for artifact in artifacts:
        path = (seed_dir / str(artifact.get("relative_path", ""))).resolve()
        loaded = load_seed_candidate_look(
            model_seed,
            path,
            binding=binding,
            manifest_path=paths["candidate_manifest"],
            validation_instance_manifest_path=paths["validation_manifest"],
            contract=paths["contract"],
            device="cpu",
        )
        authenticated_candidates.append(loaded)
        candidate_records.append(
            {
                "checkpoint_episode": int(loaded["checkpoint"]["completed_episodes"]),
                "relative_path": str(path.relative_to(seed_dir)),
                "raw_sha256": sha256_file(path),
                "manifest_artifact_sha256": loaded["checkpoint_sha256"],
                "validation_development_candidate_eligible": bool(
                    loaded["validation_summary"]["development_candidate_eligible"]
                ),
                "validation_ledger_sha256": loaded["validation_ledger_sha256"],
                "validation_summary": loaded["validation_summary"],
            }
        )
    derived_best_summary = None
    for loaded in authenticated_candidates:
        derived_best_summary = v23_base.select_better_validation(
            derived_best_summary,
            loaded["validation_summary"],
        )
    eligible = derived_best_summary is not None
    _require(
        summary.get("development_candidate_eligible"),
        eligible,
        name=f"seed {model_seed} derived/training-summary eligibility",
    )
    _require(
        summary.get("best_validation"),
        derived_best_summary,
        name=f"seed {model_seed} derived/training-summary best validation",
    )
    best_path = seed_dir / "best-development-candidate.pth"
    selected = None
    selected_record = None
    if eligible:
        if not best_path.is_file():
            raise StabilityProtocolError(f"seed {model_seed} eligible best is missing")
        summary_best_path = summary.get("best_development_candidate")
        if (
            not isinstance(summary_best_path, str)
            or Path(summary_best_path).absolute() != best_path
            or Path(summary_best_path).is_symlink()
        ):
            raise StabilityProtocolError(
                f"seed {model_seed} training summary best path is inconsistent"
            )
        if pinned_completion is not None and not isinstance(
            pinned_completion.get("selected_best"), Mapping
        ):
            raise StabilityProtocolError(
                f"seed {model_seed} completion lacks a selected-best trust root"
            )
        selected = load_seed_best_candidate(
            model_seed,
            best_path,
            binding=binding,
            expected_best_sha256=(
                pinned_completion["selected_best"]["best_checkpoint_raw_sha256"]
                if pinned_completion is not None
                else sha256_file(best_path)
            ),
            manifest_path=paths["candidate_manifest"],
            expected_manifest_sha256=(
                pinned_completion["candidate_manifest_sha256"]
                if pinned_completion is not None
                else candidate_manifest_sha
            ),
            validation_instance_manifest_path=paths["validation_manifest"],
            contract=paths["contract"],
            device="cpu",
        )
        _require(
            selected["validation_summary"],
            derived_best_summary,
            name=f"seed {model_seed} derived/loaded selected validation",
        )
        selected_record = {
            "checkpoint_episode": int(selected["selected_episode"]),
            "best_checkpoint_raw_sha256": selected["checkpoint_sha256"],
            "validation_summary": selected["validation_summary"],
            "selected_validation_ledger_relative_path": selected[
                "validation_summary"
            ]["validation_ledger"]["relative_path"],
            "selected_validation_ledger_raw_sha256": sha256_file(
                seed_dir
                / selected["validation_summary"]["validation_ledger"]["relative_path"]
            ),
        }
    else:
        if any(
            bool(loaded["validation_summary"]["development_candidate_eligible"])
            for loaded in authenticated_candidates
        ):
            raise StabilityProtocolError(
                f"seed {model_seed} ineligible derivation contains an eligible look"
            )
        if (
            best_path.exists()
            or summary.get("best_development_candidate") is not None
            or summary.get("best_validation") is not None
        ):
            raise StabilityProtocolError(f"seed {model_seed} ineligible run has a best artifact")

    validation_manifest = load_json(
        paths["validation_manifest"], name=f"seed {model_seed} validation manifest"
    )
    validation_manifest_sha = verify_self_hash(
        validation_manifest,
        "manifest_sha256",
        name=f"seed {model_seed} validation manifest",
    )
    if pinned_completion is not None:
        _require(
            validation_manifest_sha,
            pinned_completion.get("validation_instance_manifest_sha256"),
            name=f"seed {model_seed} pinned validation manifest self SHA",
        )
        _require(
            sha256_file(paths["validation_manifest"]),
            pinned_completion.get("validation_instance_manifest_raw_sha256"),
            name=f"seed {model_seed} pinned validation manifest raw SHA",
        )
    _require(validation_manifest.get("final_86xxx_panel_opened"), False, name=f"seed {model_seed} validation final flag")
    _require(
        [item.get("instance_seed") for item in validation_manifest.get("instances", [])],
        list(PANEL_SEEDS),
        name=f"seed {model_seed} validation instances",
    )
    validate_no_final_panel_values(
        instance_seeds=(
            *SEED_PROFILES[model_seed].train_seeds,
            *PANEL_SEEDS,
        ),
        action_rngs=(
            *SEED_PROFILES[model_seed].behavior_rng_seeds,
            *VALIDATION_POLICY_RNGS,
            SEED_PROFILES[model_seed].replay_rng_seed,
        ),
    )
    payload = {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "protocol": STABILITY_PROTOCOL,
        "status": "complete",
        "model_seed": model_seed,
        "profile": SEED_PROFILES[model_seed].to_manifest(),
        "run_manifest_sha256": run_manifest["manifest_sha256"],
        "training_contract_sha256": contract_sha,
        "training_contract_raw_sha256": sha256_file(paths["contract"]),
        "training_summary_raw_sha256": sha256_file(paths["summary"]),
        "candidate_manifest_sha256": candidate_manifest_sha,
        "candidate_manifest_raw_sha256": sha256_file(paths["candidate_manifest"]),
        "validation_instance_manifest_sha256": validation_manifest_sha,
        "validation_instance_manifest_raw_sha256": sha256_file(paths["validation_manifest"]),
        "candidate_look_artifacts": tuple(candidate_records),
        "validation_ledgers": _validation_ledger_manifest(seed_dir),
        "seed_tree_raw_sha256": tree_before,
        "latest_checkpoint": {
            "relative_path": "latest.pth",
            "raw_sha256": tree_before["latest.pth"],
            "authoritative_for_selection_or_analysis": False,
            "deserialization_performed": False,
            "role": "non_authoritative_audit_only_training_state",
            "bound_training_contract_sha256": contract_sha,
            "completed_episodes_claim_from_authenticated_summary": 200,
        },
        "development_candidate_eligible": eligible,
        "selected_best": selected_record,
        "completed_training_episodes": 200,
        "tuning_performed": False,
        "final_86xxx_panel_opened": False,
        "deployment_checkpoint_eligible": False,
    }
    payload["completion_sha256"] = digest_json(payload)
    tree_after = _seed_tree_raw_sha256(
        seed_dir,
        allow_completion=pinned_completion is not None,
    )
    _require(
        tree_after,
        tree_before,
        name=f"seed {model_seed} artifact tree changed during authentication",
    )
    return payload, selected


def authenticate_seed_completion(
    model_seed: int,
    *, output_dir: Path,
    run_manifest: Mapping,
    binding: StabilityRunBinding,
) -> tuple[dict, Optional[dict]]:
    seed_dir = _canonical_seed_dir(output_dir, model_seed, require_exists=True)
    completion_path = seed_dir / COMPLETION_NAME
    _require_regular_file(
        completion_path, name=f"seed {model_seed} completion"
    )
    observed = load_json(completion_path, name=f"seed {model_seed} completion")
    verify_self_hash(observed, "completion_sha256", name=f"seed {model_seed} completion")
    expected, selected = reconstruct_seed_completion(
        model_seed,
        seed_dir=seed_dir,
        run_manifest=run_manifest,
        binding=binding,
        pinned_completion=observed,
    )
    _require(observed, expected, name=f"seed {model_seed} completion")
    return observed, selected


def run_seed(
    model_seed: int,
    *, output_dir: Path = DEFAULT_OUTPUT_DIR,
    freeze_spec_path: Path = DEFAULT_FREEZE_SPEC,
    repair_dir: Path = DEFAULT_REPAIR_DIR,
) -> dict:
    """Execute one long job.  Prepare and analysis never call this function."""

    if type(model_seed) is not int or model_seed not in MODEL_SEEDS:
        raise StabilityProtocolError(f"model seed must be one of {MODEL_SEEDS}")
    run_manifest, binding, repair = authenticate_run_manifest(
        output_dir=output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
    )
    seed_dir = _canonical_seed_dir(output_dir, model_seed, require_exists=False)
    if seed_dir.exists() and any(seed_dir.iterdir()):
        if (seed_dir / COMPLETION_NAME).is_file():
            completion, _ = authenticate_seed_completion(
                model_seed,
                output_dir=output_dir,
                run_manifest=run_manifest,
                binding=binding,
            )
            _reauthenticate_run_trust(
                output_dir=output_dir,
                freeze_spec_path=freeze_spec_path,
                repair_dir=repair_dir,
                expected_manifest=run_manifest,
                expected_binding=binding,
                expected_repair=repair,
            )
            return completion
        raise StabilityProtocolError(
            f"seed {model_seed} output is partial; V2.3 is non-resumable and will not be overwritten"
        )
    run_frozen_stability_seed(
        model_seed,
        binding=binding,
        output_dir=seed_dir,
        device="cuda",
    )
    # A long training job is an explicit trust-window boundary.  Fail before
    # creating a completion record if code, freeze inputs, or repaired
    # baselines changed while it ran.
    _reauthenticate_run_trust(
        output_dir=output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
        expected_manifest=run_manifest,
        expected_binding=binding,
        expected_repair=repair,
    )
    completion, _ = reconstruct_seed_completion(
        model_seed,
        seed_dir=seed_dir,
        run_manifest=run_manifest,
        binding=binding,
    )
    atomic_json(seed_dir / COMPLETION_NAME, completion)
    # Reload from disk and authenticate against the hashes pinned in the
    # completion itself; never return an in-memory record that was not read
    # through the fail-closed loader.
    observed_completion, _ = authenticate_seed_completion(
        model_seed,
        output_dir=output_dir,
        run_manifest=run_manifest,
        binding=binding,
    )
    _reauthenticate_run_trust(
        output_dir=output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
        expected_manifest=run_manifest,
        expected_binding=binding,
        expected_repair=repair,
    )
    return observed_completion


_ROW_METRICS = {
    "mean_dense_objective_return": "dense_return",
    "mean_absolute_error": "mean_absolute_error",
    "mean_tardiness": "mean_tardiness",
    "mean_earliness": "mean_earliness",
    "mean_steps": "steps",
    "within_target_window_rate": "within_target_window_rate",
}


def summarize_selected_rows(model_seed: int, rows: Sequence[Mapping]) -> dict:
    """Apply the frozen RNG -> instance aggregation for one training seed."""

    if type(model_seed) is not int or model_seed not in MODEL_SEEDS:
        raise StabilityProtocolError(f"model seed must be one of {MODEL_SEEDS}")
    if len(rows) != 48:
        raise StabilityProtocolError(f"seed {model_seed} requires exactly 48 selected rows")
    by_key = {}
    safety_issues = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise StabilityProtocolError(f"seed {model_seed} selected row is invalid")
        instance_seed = row.get("instance_seed")
        rng_index = row.get("policy_rng_index")
        if type(instance_seed) is not int or type(rng_index) is not int:
            raise StabilityProtocolError(
                f"seed {model_seed} selected row identity types are invalid"
            )
        key = (instance_seed, rng_index)
        if key in by_key:
            raise StabilityProtocolError(f"seed {model_seed} selected grid has a duplicate")
        expected_instance_index = PANEL_SEEDS.index(instance_seed) if instance_seed in PANEL_SEEDS else -1
        expected_rng = 620_000_000 + 4 * expected_instance_index + rng_index
        if (
            expected_instance_index < 0
            or rng_index not in range(4)
            or type(row.get("instance_index")) is not int
            or row.get("instance_index") != expected_instance_index
            or type(row.get("policy_rng_seed")) is not int
            or row.get("policy_rng_seed") != expected_rng
        ):
            raise StabilityProtocolError(f"seed {model_seed} selected row identity is invalid")
        by_key[key] = row
        issues = []
        required = row.get("required_deliveries")
        delivered = row.get("delivery_count")
        if row.get("strict_method_success") is not True:
            issues.append("strict_method_failure")
        if type(required) is not int or type(delivered) is not int:
            raise StabilityProtocolError(
                f"seed {model_seed} delivery-count types are invalid"
            )
        if required <= 0 or delivered != required:
            issues.append("incomplete_delivery")
        if row.get("method_failure_reason") is not None:
            issues.append("method_failure_reason")
        if row.get("all_selected_candidates_exact_safe") is not True:
            issues.append("not_exact_safe")
        for field in ("illegal_drops", "fallbacks", "witness_mismatches"):
            if type(row.get(field)) is not int:
                raise StabilityProtocolError(
                    f"seed {model_seed} safety-count type is invalid: {field}"
                )
            if row.get(field) != 0:
                issues.append(f"nonzero_{field}")
        if (
            type(row.get("physical_rehandles")) is not int
            or row.get("physical_rehandles") < 0
        ):
            raise StabilityProtocolError(
                f"seed {model_seed} physical-rehandle count is invalid"
            )
        for field in _ROW_METRICS.values():
            value = row.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise StabilityProtocolError(
                    f"seed {model_seed} selected metric is invalid: {field}"
                )
        if row.get("evaluation_learning") is not False:
            issues.append("evaluation_learning")
        if row.get("training_agent_unchanged") is not True:
            issues.append("training_agent_changed")
        if row.get("validation_batch_state_unchanged") is not True:
            issues.append("validation_batch_changed")
        if row.get("fresh_evaluation_clone") is not True:
            issues.append("clone_not_fresh")
        if row.get("stochastic_selection_only") is not True:
            issues.append("not_stochastic")
        if row.get("map_selection_used") is not False:
            issues.append("map_selection")
        if issues:
            safety_issues.append({"row_index": index, "issues": issues})
    expected_grid = {(seed, rng) for seed in PANEL_SEEDS for rng in range(4)}
    _require(set(by_key), expected_grid, name=f"seed {model_seed} selected grid")

    instance_points = []
    for instance_seed in PANEL_SEEDS:
        cluster = [by_key[(instance_seed, rng)] for rng in range(4)]
        metrics = {
            output: float(fmean(float(row[source]) for row in cluster))
            for output, source in _ROW_METRICS.items()
        }
        physical = sum(row["physical_rehandles"] for row in cluster)
        required = sum(row["required_deliveries"] for row in cluster)
        metrics["physical_rehandles_per_100_required_deliveries"] = (
            100.0 * physical / required
        )
        instance_points.append(
            {
                "instance_seed": instance_seed,
                "action_rng_count": 4,
                **metrics,
            }
        )
    metric_names = (*_ROW_METRICS, "physical_rehandles_per_100_required_deliveries")
    metrics = {
        name: float(fmean(point[name] for point in instance_points))
        for name in metric_names
    }
    return {
        "model_seed": model_seed,
        "row_count": len(rows),
        "instance_count": len(instance_points),
        "whole_seed_strict_safe_complete": not safety_issues,
        "safety_issues": safety_issues,
        "aggregation_order": "equal_4_action_rng_then_equal_12_instances",
        "metrics": metrics,
        "instance_points": tuple(instance_points),
    }


def _dominators(point: Mapping, baseline_summaries: Sequence[Mapping]) -> tuple[str, ...]:
    mae = float(point["mean_absolute_error"])
    rehandles = float(point["physical_rehandles_per_100_required_deliveries"])
    dominators = []
    for summary in baseline_summaries:
        metrics = summary["metrics"]
        baseline_mae = float(metrics["mean_absolute_error"])
        baseline_rehandles = float(
            metrics["physical_rehandles_per_100_required_deliveries"]
        )
        if (
            baseline_mae <= mae
            and baseline_rehandles <= rehandles
            and (baseline_mae < mae or baseline_rehandles < rehandles)
        ):
            dominators.append(str(summary["method_id"]))
    return tuple(sorted(dominators))


def _dense_metric_ranking(
    methods: Sequence[Mapping], *, metric: str, higher_is_better: bool,
) -> tuple[dict, ...]:
    """Rank exact metric ties together using deterministic dense ranks."""

    ordered = sorted(
        methods,
        key=lambda item: (
            -float(item["metrics"][metric])
            if higher_is_better
            else float(item["metrics"][metric]),
            str(item["method_id"]),
        ),
    )
    result = []
    previous_value = None
    rank = 0
    for item in ordered:
        value = float(item["metrics"][metric])
        if previous_value is None or value != previous_value:
            rank += 1
            previous_value = value
        result.append(
            {"rank": rank, "method_id": item["method_id"], "value": value}
        )
    return tuple(result)


def analyze(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR,
    freeze_spec_path: Path = DEFAULT_FREEZE_SPEC,
    repair_dir: Path = DEFAULT_REPAIR_DIR,
) -> dict:
    """Authenticate selected artifacts and apply the predeclared screen."""

    output_dir = _canonical_output_root(output_dir, require_exists=True)
    run_manifest, binding, repair = authenticate_run_manifest(
        output_dir=output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
    )
    completion_records = []
    seed_summaries_by_seed = {}
    selected_episodes = {}
    completed_but_ineligible_seeds = []
    for seed in MODEL_SEEDS:
        completion, selected = authenticate_seed_completion(
            seed,
            output_dir=output_dir,
            run_manifest=run_manifest,
            binding=binding,
        )
        completion_records.append(completion)
        if selected is None:
            if completion.get("development_candidate_eligible") is not False:
                raise StabilityProtocolError(
                    f"seed {seed} completion/best eligibility is inconsistent"
                )
            completed_but_ineligible_seeds.append(seed)
            selected_episodes[str(seed)] = None
            continue
        rows = selected["selected_diagnostic"]["validation_ledger"]["rows"]
        summary = summarize_selected_rows(seed, rows)
        selected_summary = selected["validation_summary"]
        for field, expected in (
            ("mean_absolute_error", selected_summary["mean_absolute_error"]),
            (
                "physical_rehandles_per_100_required_deliveries",
                selected_summary["expected_physical_rehandles_per_100_required_deliveries"],
            ),
            ("mean_dense_objective_return", selected_summary["mean_dense_return"]),
        ):
            if abs(float(summary["metrics"][field]) - float(expected)) > 1e-10:
                raise StabilityProtocolError(
                    f"seed {seed} selected-summary aggregation mismatch: {field}"
                )
        selected_episodes[str(seed)] = int(selected["selected_episode"])
        seed_summaries_by_seed[seed] = summary

    all_three_eligible = not completed_but_ineligible_seeds
    baseline_summaries = tuple(repair["baseline_summaries"])
    aggregate_metrics = None
    variability = None
    aggregate_dominators = None
    rankings = None
    individual_screen = []
    non_dominated_seed_count = None
    if all_three_eligible:
        ordered_seed_summaries = tuple(
            seed_summaries_by_seed[seed] for seed in MODEL_SEEDS
        )
        metric_names = tuple(ordered_seed_summaries[0]["metrics"])
        aggregate_metrics = {
            metric: float(
                fmean(item["metrics"][metric] for item in ordered_seed_summaries)
            )
            for metric in metric_names
        }
        variability = {
            metric: {
                "n_training_seeds": 3,
                "sample_standard_deviation": float(
                    stdev(item["metrics"][metric] for item in ordered_seed_summaries)
                ),
                "minimum": min(
                    item["metrics"][metric] for item in ordered_seed_summaries
                ),
                "maximum": max(
                    item["metrics"][metric] for item in ordered_seed_summaries
                ),
                "inferential_claim_authorized": False,
            }
            for metric in metric_names
        }
        aggregate_dominators = _dominators(aggregate_metrics, baseline_summaries)
        for summary in ordered_seed_summaries:
            dominators = _dominators(summary["metrics"], baseline_summaries)
            individual_screen.append(
                {
                    "model_seed": summary["model_seed"],
                    "development_candidate_eligible": True,
                    "evaluation_status": "evaluated",
                    "non_dominated": not dominators,
                    "whole_seed_strict_safe_complete": summary[
                        "whole_seed_strict_safe_complete"
                    ],
                    "dominators": dominators,
                    "dominance_coordinates": {
                        "mean_absolute_error": summary["metrics"][
                            "mean_absolute_error"
                        ],
                        "physical_rehandles_per_100_required_deliveries": summary[
                            "metrics"
                        ]["physical_rehandles_per_100_required_deliveries"],
                    },
                }
            )
        non_dominated_seed_count = sum(
            item["non_dominated"] for item in individual_screen
        )

        rankings = {}
        combined = [
            {
                "method_id": "vcg_constrained_v2_3_gamma1_equal_seed_11_13",
                "metrics": aggregate_metrics,
            },
            *(
                {"method_id": item["method_id"], "metrics": item["metrics"]}
                for item in baseline_summaries
            ),
        ]
        directions = {
            "mean_dense_objective_return": True,
            "mean_absolute_error": False,
            "physical_rehandles_per_100_required_deliveries": False,
            "mean_steps": False,
        }
        for metric, higher in directions.items():
            rankings[metric] = _dense_metric_ranking(
                combined,
                metric=metric,
                higher_is_better=higher,
            )
    else:
        for seed in MODEL_SEEDS:
            summary = seed_summaries_by_seed.get(seed)
            individual_screen.append(
                {
                    "model_seed": seed,
                    "development_candidate_eligible": summary is not None,
                    "evaluation_status": (
                        "suppressed_incomplete_three_seed_estimand"
                        if summary is not None
                        else "completed_without_eligible_best"
                    ),
                    "non_dominated": None,
                    "whole_seed_strict_safe_complete": (
                        summary["whole_seed_strict_safe_complete"]
                        if summary is not None else None
                    ),
                    "dominators": None,
                    "dominance_coordinates": None,
                }
            )
    eligible_count = sum(
        bool(item["development_candidate_eligible"]) for item in completion_records
    )
    strict_row_count = sum(
        item["row_count"] if item["whole_seed_strict_safe_complete"] else 0
        for item in seed_summaries_by_seed.values()
    )
    passed = bool(
        all_three_eligible
        and eligible_count == 3
        and strict_row_count == 144
        and aggregate_dominators == ()
        and non_dominated_seed_count is not None
        and non_dominated_seed_count >= 2
    )
    strong = bool(passed and non_dominated_seed_count == 3)

    individual_seed_summaries = []
    for seed in MODEL_SEEDS:
        summary = seed_summaries_by_seed.get(seed)
        if summary is not None:
            individual_seed_summaries.append(
                {**summary, "development_candidate_eligible": True}
            )
        else:
            individual_seed_summaries.append(
                {
                    "model_seed": seed,
                    "development_candidate_eligible": False,
                    "selected_checkpoint_available": False,
                    "row_count": 0,
                    "instance_count": 0,
                    "whole_seed_strict_safe_complete": None,
                    "safety_issues": None,
                    "aggregation_order": None,
                    "metrics": None,
                    "instance_points": None,
                    "reason": "completed_without_eligible_best",
                }
            )

    report = {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "protocol": STABILITY_PROTOCOL,
        "status": "passed" if passed else "failed",
        "strong_result": strong,
        "development_only": True,
        "performance_claim_authorized": False,
        "confirmatory_claim_authorized": False,
        "deployment_checkpoint_eligible": False,
        "final_86xxx_panel_opened": False,
        "run_manifest_sha256": run_manifest["manifest_sha256"],
        "repair_v2_report_sha256": repair["report_sha256"],
        "selected_episodes": selected_episodes,
        "completed_but_ineligible_model_seeds": tuple(
            completed_but_ineligible_seeds
        ),
        "eligible_training_seed_count": eligible_count,
        "strict_safe_complete_selected_row_count": strict_row_count,
        "expected_selected_row_count": 144,
        "individual_seed_summaries": tuple(individual_seed_summaries),
        "equal_seed_aggregate": {
            "available": all_three_eligible,
            "unavailable_reason": (
                None
                if all_three_eligible
                else "one_or_more_completed_seeds_have_no_eligible_best"
            ),
            "model_seeds": MODEL_SEEDS,
            "aggregation_order": "equal_RNG_then_equal_instance_then_equal_training_seed",
            "metrics": aggregate_metrics,
            "descriptive_training_seed_variability": variability,
            "dominators_on_MAE_and_total_physical_rehandles": aggregate_dominators,
            "non_dominated": (
                aggregate_dominators == () if all_three_eligible else None
            ),
        },
        "individual_seed_screen": tuple(individual_screen),
        "non_dominated_individual_seed_count": non_dominated_seed_count,
        "minimum_required_non_dominated_individual_seed_count": 2,
        "baseline_summaries_reused_once": tuple(
            {
                "method_id": item["method_id"],
                "whole_method_numeric_eligible": item["whole_method_numeric_eligible"],
                "metrics": item["metrics"],
            }
            for item in baseline_summaries
        ),
        "metric_rankings": rankings,
        "criterion": {
            "three_of_three_eligible": all_three_eligible and eligible_count == 3,
            "all_144_rows_strict_safe_complete": strict_row_count == 144,
            "aggregate_non_dominated_evaluable": all_three_eligible,
            "aggregate_non_dominated": (
                aggregate_dominators == () if all_three_eligible else False
            ),
            "at_least_two_of_three_individual_seeds_non_dominated": (
                non_dominated_seed_count is not None
                and non_dominated_seed_count >= 2
            ),
            "three_of_three_individual_seeds_non_dominated_is_strong": strong,
            "passed": passed,
        },
        "seed10_label": "development_only_not_part_of_stability_aggregate",
        "baselines_reused_once_not_multiplied": True,
        "complete_case_filtering_used": False,
        "incomplete_three_seed_estimand_is_never_aggregated": True,
        "scalarization_used_for_gate": False,
    }
    # Re-read every completion through its pinned-hash loader after all
    # aggregation work.  This rejects seed-tree drift and closes the interval
    # between the first read and report publication.  Then repeat the complete
    # source/freeze/repair authentication.  Any failure occurs before a report
    # or audit is written.
    final_completion_raw_sha256 = {}
    for seed, expected_completion in zip(MODEL_SEEDS, completion_records):
        observed_completion, _ = authenticate_seed_completion(
            seed,
            output_dir=output_dir,
            run_manifest=run_manifest,
            binding=binding,
        )
        _require(
            observed_completion,
            expected_completion,
            name=f"seed {seed} completion changed during analysis",
        )
        final_completion_raw_sha256[str(seed)] = sha256_file(
            output_dir / f"seed-{seed}" / COMPLETION_NAME
        )
    _reauthenticate_run_trust(
        output_dir=output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
        expected_manifest=run_manifest,
        expected_binding=binding,
        expected_repair=repair,
    )
    report["report_sha256"] = digest_json(report)
    atomic_json(output_dir / REPORT_NAME, report)
    # Close the report-write window before certifying the audit.  If anything
    # drifts here, the report remains non-authoritative because no matching
    # self-hashed audit is emitted.
    for seed, expected_completion in zip(MODEL_SEEDS, completion_records):
        _require(
            _seed_tree_raw_sha256(
                output_dir / f"seed-{seed}", allow_completion=True,
            ),
            expected_completion["seed_tree_raw_sha256"],
            name=f"seed {seed} tree changed after report write",
        )
        _require(
            sha256_file(output_dir / f"seed-{seed}" / COMPLETION_NAME),
            final_completion_raw_sha256[str(seed)],
            name=f"seed {seed} completion changed after report write",
        )
    _reauthenticate_run_trust(
        output_dir=output_dir,
        freeze_spec_path=freeze_spec_path,
        repair_dir=repair_dir,
        expected_manifest=run_manifest,
        expected_binding=binding,
        expected_repair=repair,
    )
    audit = {
        "schema_version": STABILITY_SCHEMA_VERSION,
        "protocol": STABILITY_PROTOCOL,
        "status": "complete",
        "run_manifest_sha256": run_manifest["manifest_sha256"],
        "seed_completion_sha256": {
            str(record["model_seed"]): record["completion_sha256"]
            for record in completion_records
        },
        "seed_completion_raw_sha256": final_completion_raw_sha256,
        "repair_v2_raw_sha256": repair["raw_sha256"],
        "repair_v2_ledger_raw_sha256": repair["repair_ledger_raw_sha256"],
        "stability_report_raw_sha256": sha256_file(output_dir / REPORT_NAME),
        "all_144_rows_authenticated": strict_row_count == 144,
        "completed_but_ineligible_model_seeds": tuple(
            completed_but_ineligible_seeds
        ),
        "equal_seed_estimand_available": all_three_eligible,
        "all_sources_unchanged": True,
        "all_repair_v2_artifacts_reauthenticated_before_report": True,
        "all_seed_trees_reauthenticated_before_report": True,
        "trust_roots_and_seed_trees_rehashed_after_report_before_audit": True,
        "environment_rollouts_executed_during_analysis": 0,
        "final_86xxx_panel_opened": False,
        "performance_claim_authorized": False,
    }
    audit["audit_sha256"] = digest_json(audit)
    atomic_json(output_dir / AUDIT_NAME, audit)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen VCG V2.3 seed-stability workflow (85xxx development panel)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="authenticate repair V2 and freeze the run")
    run = subparsers.add_parser("run_seed", help="run one long frozen training seed")
    run.add_argument("model_seed", type=int, choices=MODEL_SEEDS)
    subparsers.add_parser("analyze", help="authenticate and aggregate completed seeds")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_protocol()
    elif args.command == "run_seed":
        result = run_seed(args.model_seed)
    elif args.command == "analyze":
        result = analyze()
    else:  # pragma: no cover
        raise StabilityProtocolError(f"unknown command: {args.command}")
    print(__import__("json").dumps(result, indent=2, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()


__all__ = [
    "analyze", "authenticate_run_manifest", "authenticate_seed_completion",
    "build_run_manifest", "build_parser", "main", "prepare_protocol",
    "reconstruct_seed_completion", "run_seed", "summarize_selected_rows",
]
