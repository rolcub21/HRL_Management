#!/usr/bin/env python3
"""Development-only matched 85xxx comparison for VCG V2.3.

The primary online comparison contains exactly two learned procedures and six
deterministic online baselines:

* the selected VCG V2.3 gamma=1 candidate (episode 160), reusing its 12 x 4
  stochastic validation ledger;
* the selected three-training-seed VCG-Dense V1.1 control, reusing its 3 x 12
  deterministic rollout ledger; and
* nearest-free, dynamic PSLAP, the 2009 rolling GA, duration-aware rolling GA,
  operational rolling GA, and enhanced-complete rolling GA, each executed once
  on each of the same 12 immutable EpisodeInstances.

This command is deliberately dry by default.  It authenticates all frozen
learned-policy sources and writes an execution contract, but runs no baseline
unless ``--execute-baselines`` is supplied.  A partially completed baseline
grid is resumable from independently authenticated atomic run ledgers.

The panel (85000--85011) was already opened for development.  Results are
therefore descriptive development evidence only.  The implementation refuses
the reserved 86xxx EpisodeInstances and the 622xxx action-RNG namespace, never
uses complete-case filtering, and excludes an entire method from numeric ranks
if any expected row is missing or unsafe.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Callable, Iterable, Mapping, Optional, Sequence
import uuid

from compare_vcg_dense_pareto import (
    FROZEN_OBJECTIVE_SPEC,
    GA_SEED_BASE,
    LOOKAHEAD_MARGIN_STEPS,
    MAX_DEFER_STEPS,
    MAX_STEPS,
    ROLLING_GA_EGRESS_WEIGHT,
    ROLLING_GENERATIONS,
    ROLLING_POPULATION,
    _baseline_args,
)
from compare_viability_graph_baselines import (
    _dual_rescore_from_legacy_return,
    _normalize_baseline,
)
from contention_metrics import (
    CONTENTION_METRIC_SCHEMA_VERSION,
    validate_contention_metric_record,
)
from evaluate_vcg_dense_v1_1_v2_2_panel_control import (
    PROTOCOL as V11_CONTROL_PROTOCOL,
    _validate_v22_source,
)
from example.episode_instance import EpisodeInstance
from PSLAP.track_a import (
    TRACK_A_DYNAMIC,
    TRACK_A_GA_OFFLINE,
    TRACK_A_GA_ROLLING,
    TRACK_A_GA_ROLLING_COMPLETE,
    TRACK_A_GA_ROLLING_DURATION_AWARE,
    TRACK_A_GA_ROLLING_OPERATIONAL,
    TRACK_A_KIM2020_A3C_SPATIAL,
    TRACK_A_NEAREST_FREE,
    TRACK_A_REG_SELECTOR_V5,
)
from track_b_urgency_evaluate import (
    evaluate_assignment_ablation_one,
    resolve_device,
)
from train_vcg_constrained_v2_3 import (
    CANDIDATE_EPISODES,
    CHECKPOINT_FAMILY as V23_CHECKPOINT_FAMILY,
    DEFAULT_VALIDATION_SEEDS,
    FROZEN_OBJECTIVE_SPEC as V23_OBJECTIVE_SPEC,
    METHOD_VERSION as V23_METHOD_VERSION,
    POLICY_REALIZATION as V23_POLICY_REALIZATION,
    TRAINING_PROTOCOL as V23_TRAINING_PROTOCOL,
    VALIDATION_POLICY_RNG_COUNT,
    V23DualConfig,
    load_best_development_candidate,
    select_better_validation,
    summarize_validation,
    validation_policy_rng_seed,
)


PROTOCOL = "vcg_v2_3_matched_85xxx_online_baselines_development_v1"
SCHEMA_VERSION = 1
EXPECTED_DELIVERIES = 8
PANEL_SEEDS = tuple(range(85_000, 85_012))
MODEL_SEEDS = (0, 1, 2)
V23_SELECTED_EPISODE = 160
V22_DIAGNOSTIC_EPISODE = 80
FINAL_PANEL_SEEDS = frozenset(range(86_000, 86_030))
FINAL_POLICY_RNG_START = 622_000_000
FINAL_POLICY_RNG_STOP = FINAL_POLICY_RNG_START + 30 * 4

# One-sided Student-t .95 quantile, df=11.  This is a descriptive, nominal
# bound over the 12 common EpisodeInstance clusters.  It is deliberately kept
# distinct from V2.3's authoritative seven-look Bonferroni gate (2.906203...).
DESCRIPTIVE_ONE_SIDED_T_95_DF11 = 1.7958848187036691

V23_METHOD = "vcg_constrained_v2_3_gamma1_episode160"
V11_METHOD = "vcg_dense_v1_1_selected_three_seed"
V22_DIAGNOSTIC_METHOD = "vcg_constrained_v2_2_episode80_superseded_diagnostic"
NEAREST_METHOD = "duration_aware_nearest_free"
DYNAMIC_METHOD = "duration_aware_dynamic_pslap"
ROLLING_2009_METHOD = "duration_aware_pslap_ga_2009_rolling"
DURATION_AWARE_GA_METHOD = "duration_aware_pslap_ga_duration_aware_rolling"
OPERATIONAL_GA_METHOD = "duration_aware_pslap_ga_operational_rolling"
ENHANCED_GA_METHOD = "duration_aware_enhanced_complete_rolling_ga"

DETERMINISTIC_METHOD_TO_SOURCE = {
    NEAREST_METHOD: TRACK_A_NEAREST_FREE,
    DYNAMIC_METHOD: TRACK_A_DYNAMIC,
    ROLLING_2009_METHOD: TRACK_A_GA_ROLLING,
    DURATION_AWARE_GA_METHOD: TRACK_A_GA_ROLLING_DURATION_AWARE,
    OPERATIONAL_GA_METHOD: TRACK_A_GA_ROLLING_OPERATIONAL,
    ENHANCED_GA_METHOD: TRACK_A_GA_ROLLING_COMPLETE,
}
DETERMINISTIC_METHODS = tuple(DETERMINISTIC_METHOD_TO_SOURCE)
PRIMARY_METHODS = (V23_METHOD, V11_METHOD, *DETERMINISTIC_METHODS)

CONTRACT_FILENAME = "comparison-contract.json"
PREFLIGHT_FILENAME = "preflight.json"
RUNS_FILENAME = "matched-runs.csv"
REPORT_FILENAME = "matched-report.json"
AUDIT_FILENAME = "matched-audit.json"

# External pins for the exact completed development artifacts.  Embedded
# self-hashes are checked independently below; these raw hashes prevent a
# rewritten self-consistent artifact from silently entering this comparison.
V23_RAW_SHA256 = {
    "training-contract.json": "e4d8c12bc7b7ed8f645392079b55eea2b83616a0ea63d03b29012c3e1f5d33ba",
    "candidate-look-checkpoint-manifest.json": "89e8906411768f6606dea7e045a50b420adf04be6d29c21de5ede428e647d580",
    "validation-instance-manifest.json": "b390230f1155e607486027d9075f84d4f2fe388acaaa81ddea885eefbca5f8ad",
    "validation-ledger/episode-0160.json": "f6e6b7f564187262e132b24603173eb87dff929d96b05a24f0f514f7cf9cf1c2",
    "best-development-candidate.pth": "5eedcf8f4f82dfb62f0917e1a71ef4c95e688b7b0ec03a3eafe85484e71c949a",
}
V23_CONTRACT_CANONICAL_SHA256 = (
    "781f5fa9f959d2803a35c08f86e74f45a4b07b3fa6f902c6b10cb680ea1e21d4"
)
V23_CANDIDATE_MANIFEST_CANONICAL_SHA256 = (
    "005e80a61e914cecc0fbe0a7eb1655b6df7946f50a49616d28bc4a02e992dc50"
)
V23_INSTANCE_MANIFEST_CANONICAL_SHA256 = (
    "68f488a236a3599bff6eb87e8e18b692e267755e68c72f14096d92be3c1418b6"
)
V23_EP160_DIAGNOSTIC_RAW_SHA256 = (
    "5bee304497d4ac48791c6b4b5c07e3a2055e711e15277ff7e39ce64f90b5849a"
)
V23_BEST_RAW_SHA256 = (
    "5eedcf8f4f82dfb62f0917e1a71ef4c95e688b7b0ec03a3eafe85484e71c949a"
)
V23_EP160_LEDGER_CANONICAL_SHA256 = (
    "37ae32cbf5722aba106cd3a3c47c667c4db8153303c207ff3924395a4fd893dd"
)

V11_CONTROL_RAW_SHA256 = {
    "protocol-manifest.json": "a5c8cefa053775d974bbd8d6043e587b19c2e56caea753d6b1776ff37ef14cc1",
    "instance-manifest.json": "23a835d58da883be8761ad4fe7d8177f7b7a0db4bb135e7acbd183f2470f4f8e",
    "control-report.json": "4393ddda0af35616d8a0de847f37dd3b1093c08a99f3e95aa2375a554d71155c",
    "control-audit.json": "78164e8a6317dd2241b4660cb946ac9cb1ea95a7117aa62e5a05d22e28c07606",
    "control-runs.csv": "0b825e9262753ca869a3691d3ee76bf266a19a3c0ced74e9cab875f23eda2c38",
}
V11_INSTANCE_MANIFEST_CANONICAL_SHA256 = (
    "0ead7930ea4be1eeea36efcc3422f88f4b55100d3eb7c2325eba7294db0f9559"
)

V22_RAW_SHA256 = {
    "training-contract.json": "c48ff6e56c1078d19c0129f9411cb35dcb00af0d6c3d1b344363ad4229efcfd1",
    "training-summary.json": "1c17c9d44e11f5ea8b05d5971f205a068a93d90e6e9b6ec7a2896995d7b23eb6",
    "validation-instance-manifest.json": "64f7bdeb8cd2887de232a81008cf1cf19ab6d539d4cec65414b5b16b4fbc8c83",
    "validation-ledger/episode-0080.json": "77980ce765d3cd01828723f206fc117873c04bc2badbb2380eada7af0facf076",
}


class MatchedComparisonError(ValueError):
    """Raised when provenance, grid, or safety invariants fail."""


@dataclass(frozen=True)
class AuthenticatedSources:
    v23_contract: dict
    identities: Mapping[int, dict]
    instances: Mapping[int, EpisodeInstance]
    v23_rows: tuple[dict, ...]
    v23_selection: dict
    v11_rows: tuple[dict, ...]
    v11_protocol: dict
    v22_diagnostic: dict
    source_paths: Mapping[str, Path]
    source_sha256: Mapping[str, str]


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
        raise MatchedComparisonError(f"{path} must contain one JSON object")
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
        raise MatchedComparisonError(
            f"{name} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _verify_self_hash(payload: Mapping, field: str, *, name: str) -> str:
    claimed = payload.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise MatchedComparisonError(f"{name} has no valid {field}")
    unhashed = dict(payload)
    unhashed.pop(field, None)
    observed = _digest_json(unhashed)
    _require_equal(f"{name}.{field}", claimed, observed)
    return claimed


def _verify_raw_pins(root: Path, pins: Mapping[str, str], *, label: str) -> dict[str, Path]:
    paths = {}
    for relative, expected in pins.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise MatchedComparisonError(f"{label} path escapes its root")
        if not path.is_file():
            raise FileNotFoundError(path)
        _require_equal(f"{label} raw SHA {relative}", _sha256_file(path), expected)
        paths[f"{label}/{relative}"] = path
    return paths


def _validate_panel_constants() -> None:
    _require_equal("V2.3 validation seeds", tuple(DEFAULT_VALIDATION_SEEDS), PANEL_SEEDS)
    if set(PANEL_SEEDS).intersection(FINAL_PANEL_SEEDS):
        raise RuntimeError("development and final EpisodeInstance panels overlap")
    validation_rngs = {
        validation_policy_rng_seed(i, j)
        for i in range(len(PANEL_SEEDS))
        for j in range(VALIDATION_POLICY_RNG_COUNT)
    }
    if any(FINAL_POLICY_RNG_START <= seed < FINAL_POLICY_RNG_STOP for seed in validation_rngs):
        raise RuntimeError("development and final action-RNG panels overlap")
    _require_equal("V2.3 objective", V23_OBJECTIVE_SPEC.to_dict(), FROZEN_OBJECTIVE_SPEC.to_dict())


def _canonical_instance_sha(instance: EpisodeInstance) -> str:
    return sha256(instance.to_json().encode("utf-8")).hexdigest()


def _authenticate_v23(source_dir: Path) -> tuple[dict, dict[int, dict], tuple[dict, ...], dict, dict[str, Path]]:
    """Authenticate all seven candidate looks and independently select ep160."""

    source_dir = source_dir.resolve()
    paths = _verify_raw_pins(source_dir, V23_RAW_SHA256, label="v2.3")
    contract_path = source_dir / "training-contract.json"
    candidate_manifest_path = source_dir / "candidate-look-checkpoint-manifest.json"
    instance_manifest_path = source_dir / "validation-instance-manifest.json"

    contract = _load_json(contract_path)
    _require_equal("V2.3 contract canonical SHA", contract.get("contract_sha256"), V23_CONTRACT_CANONICAL_SHA256)
    _verify_self_hash(contract, "contract_sha256", name="V2.3 training contract")
    for field, expected in {
        "training_protocol": V23_TRAINING_PROTOCOL,
        "method_version": V23_METHOD_VERSION,
        "checkpoint_family": V23_CHECKPOINT_FAMILY,
        "episodes": 200,
        "model_seed": 10,
        "validation_seeds": PANEL_SEEDS,
        "validation_rows_per_checkpoint": 48,
        "validation_every": 20,
        "gamma_operational": 1.0,
        "max_steps": MAX_STEPS,
        "development_only": True,
        "performance_claim_authorized": False,
        "final_86xxx_panel_opened": False,
    }.items():
        _require_equal(f"V2.3 contract.{field}", contract.get(field), expected)
    expected_rng_grid = tuple(
        tuple(validation_policy_rng_seed(i, j) for j in range(4))
        for i in range(len(PANEL_SEEDS))
    )
    _require_equal("V2.3 validation RNG grid", contract.get("validation_policy_rng_grid"), expected_rng_grid)
    _require_equal("V2.3 dense objective", contract.get("dense_objective_spec"), FROZEN_OBJECTIVE_SPEC.to_dict())
    _require_equal("V2.3 environment", contract.get("environment"), {
        "arrival_rate": 10.0, "proc_mean": 80, "grid_rows": 5,
        "grid_cols": 5, "number_blocks": EXPECTED_DELIVERIES,
    })

    candidate_manifest = _load_json(candidate_manifest_path)
    _require_equal(
        "V2.3 candidate manifest canonical SHA",
        _verify_self_hash(candidate_manifest, "manifest_sha256", name="V2.3 candidate manifest"),
        V23_CANDIDATE_MANIFEST_CANONICAL_SHA256,
    )
    _require_equal("V2.3 candidate manifest complete", candidate_manifest.get("complete"), True)
    _require_equal("V2.3 candidate episodes", candidate_manifest.get("saved_candidate_look_episodes"), CANDIDATE_EPISODES)
    _require_equal("V2.3 candidate manifest final panel", candidate_manifest.get("final_86xxx_panel_opened"), False)

    instance_manifest = _load_json(instance_manifest_path)
    _require_equal(
        "V2.3 instance manifest canonical SHA",
        _verify_self_hash(instance_manifest, "manifest_sha256", name="V2.3 instance manifest"),
        V23_INSTANCE_MANIFEST_CANONICAL_SHA256,
    )
    _require_equal("V2.3 manifest final panel", instance_manifest.get("final_86xxx_panel_opened"), False)
    records = instance_manifest.get("instances")
    if not isinstance(records, list) or len(records) != len(PANEL_SEEDS):
        raise MatchedComparisonError("V2.3 instance manifest is not the exact 12-row panel")
    identities = {int(record["instance_seed"]): dict(record) for record in records}
    _require_equal("V2.3 instance seeds", sorted(identities), PANEL_SEEDS)

    # The official selected-best loader reconstructs the frozen contract,
    # authenticates/recomputes all seven ledgers, applies the frozen ranking,
    # and proves that the standalone best artifact is byte-identical in policy,
    # dual, lambda, and schedule to the winning ep160 diagnostic.
    authenticated = load_best_development_candidate(
        source_dir / "best-development-candidate.pth",
        expected_best_sha256=V23_BEST_RAW_SHA256,
        manifest_path=candidate_manifest_path,
        expected_manifest_sha256=V23_CANDIDATE_MANIFEST_CANONICAL_SHA256,
        validation_instance_manifest_path=instance_manifest_path,
        contract=contract_path,
        device="cpu",
    )
    _require_equal("V2.3 authenticated contract SHA", authenticated["contract_sha256"], V23_CONTRACT_CANONICAL_SHA256)
    _require_equal("V2.3 authenticated manifest SHA", authenticated["manifest_sha256"], V23_CANDIDATE_MANIFEST_CANONICAL_SHA256)
    _require_equal("V2.3 selected-best raw SHA", authenticated["checkpoint_sha256"], V23_BEST_RAW_SHA256)
    _require_equal("V2.3 loader selected episode", authenticated["selected_episode"], V23_SELECTED_EPISODE)
    _require_equal(
        "V2.3 loader validation-instance manifest SHA",
        authenticated["validation_instance_manifest_sha256"],
        V23_INSTANCE_MANIFEST_CANONICAL_SHA256,
    )

    artifacts = candidate_manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(CANDIDATE_EPISODES):
        raise MatchedComparisonError("V2.3 candidate manifest must contain all seven looks")
    dual_config = V23DualConfig(**contract["dual"])
    authenticated_records = {
        int(record["checkpoint"]["completed_episodes"]): record
        for record in authenticated["candidate_records"]
    }
    _require_equal("V2.3 authenticated candidate record grid", sorted(authenticated_records), CANDIDATE_EPISODES)
    recomputed = []
    selected = None
    ep160_rows: Optional[tuple[dict, ...]] = None
    for artifact in artifacts:
        episode = int(artifact["checkpoint_episode"])
        _require_equal("candidate look membership", episode in CANDIDATE_EPISODES, True)
        diagnostic_path = (source_dir / artifact["relative_path"]).resolve()
        if not diagnostic_path.is_relative_to(source_dir) or not diagnostic_path.is_file():
            raise MatchedComparisonError("candidate diagnostic path is missing or escapes source root")
        _require_equal(f"V2.3 ep{episode} diagnostic raw SHA", _sha256_file(diagnostic_path), artifact["sha256"])
        paths[f"v2.3/{artifact['relative_path']}"] = diagnostic_path
        authenticated_record = authenticated_records[episode]
        payload = authenticated_record["checkpoint"]
        _require_equal(f"V2.3 ep{episode} envelope episode", payload.get("completed_episodes"), episode)
        _require_equal(f"V2.3 ep{episode} envelope contract", payload.get("training_contract_sha256"), V23_CONTRACT_CANONICAL_SHA256)
        embedded = payload.get("validation_summary")
        if not isinstance(embedded, Mapping):
            raise MatchedComparisonError(f"V2.3 ep{episode} has no validation summary")
        ledger_ref = embedded.get("validation_ledger")
        if not isinstance(ledger_ref, Mapping):
            raise MatchedComparisonError(f"V2.3 ep{episode} has no validation ledger binding")
        expected_relative = f"validation-ledger/episode-{episode:04d}.json"
        _require_equal(f"V2.3 ep{episode} ledger path", ledger_ref.get("relative_path"), expected_relative)
        ledger_path = (source_dir / expected_relative).resolve()
        if not ledger_path.is_relative_to(source_dir):
            raise MatchedComparisonError("V2.3 ledger path escapes source root")
        ledger = _load_json(ledger_path)
        ledger_sha = _verify_self_hash(ledger, "ledger_sha256", name=f"V2.3 ep{episode} ledger")
        _require_equal(f"V2.3 ep{episode} official ledger", authenticated_record["validation_ledger"], ledger)
        _require_equal(f"V2.3 ep{episode} official ledger SHA", authenticated_record["validation_ledger_sha256"], ledger_sha)
        _require_equal(f"V2.3 ep{episode} embedded ledger SHA", ledger_ref.get("ledger_sha256"), ledger_sha)
        _require_equal(f"V2.3 ep{episode} ledger rows", ledger_ref.get("row_count"), 48)
        _require_equal(f"V2.3 ep{episode} ledger checkpoint", ledger.get("checkpoint_episode"), episode)
        _require_equal(f"V2.3 ep{episode} complete-case flag", ledger.get("complete_case_filtering_used"), False)
        batch = ledger.get("validation_batch_audit")
        if not isinstance(batch, Mapping) or batch.get("training_agent_unchanged") is not True:
            raise MatchedComparisonError(f"V2.3 ep{episode} validation batch was not isolated")
        rows = ledger.get("rows")
        if not isinstance(rows, list):
            raise MatchedComparisonError(f"V2.3 ep{episode} ledger has no rows")
        summary = summarize_validation(
            rows,
            dual_config,
            checkpoint_episode=episode,
            validation_lambda=float(ledger["validated_lambda"]),
            schedule_state=ledger["schedule_state"],
            validation_seeds=PANEL_SEEDS,
            positive_residual_saturation_count=int(
                embedded.get("positive_residual_dual_saturation_count_before_validation", 0)
            ),
        )
        for key, value in summary.items():
            _require_equal(f"V2.3 ep{episode} recomputed validation.{key}", embedded.get(key), value)
            _require_equal(f"V2.3 ep{episode} official validation.{key}", authenticated_record["validation_summary"].get(key), value)
        _require_equal(
            f"V2.3 ep{episode} artifact eligibility",
            artifact.get("validation_development_candidate_eligible"),
            summary["development_candidate_eligible"],
        )
        selected = select_better_validation(selected, summary)
        recomputed.append(summary)
        paths[f"v2.3/{expected_relative}"] = ledger_path
        if episode == V23_SELECTED_EPISODE:
            ep160_rows = tuple(dict(row) for row in rows)
            _require_equal("V2.3 ep160 ledger canonical SHA", ledger_sha, V23_EP160_LEDGER_CANONICAL_SHA256)
            _require_equal(
                "V2.3 ep160 validation diagnostic raw SHA",
                artifact["sha256"],
                V23_EP160_DIAGNOSTIC_RAW_SHA256,
            )

    if selected is None:
        raise MatchedComparisonError("V2.3 has no eligible candidate look")
    _require_equal("independently selected V2.3 episode", selected["checkpoint_episode"], V23_SELECTED_EPISODE)
    for key, value in selected.items():
        _require_equal(
            f"official/recomputed selected validation.{key}",
            authenticated["validation_summary"].get(key),
            value,
        )
    if ep160_rows is None:
        raise MatchedComparisonError("V2.3 ep160 ledger was not found")
    selection = {
        "selection_recomputed_without_training_summary": True,
        "candidate_episodes": CANDIDATE_EPISODES,
        "eligible_episodes": tuple(
            int(item["checkpoint_episode"])
            for item in recomputed
            if item["development_candidate_eligible"]
        ),
        "selected_episode": V23_SELECTED_EPISODE,
        "selected_validation": selected,
        "authoritative_multiple_look_rehandle_ucb": selected[
            "physical_rehandle_rate_one_sided_95_ucb"
        ],
        "authoritative_multiple_look_t_quantile": 2.906203359932373,
    }
    return contract, identities, ep160_rows, selection, paths


def _resolve_audit_source_path(
    key: str,
    *,
    control_protocol: Mapping,
    v22_source_dir: Path,
) -> Path:
    if key.startswith("v1.1/"):
        parts = Path(key).parts
        if len(parts) != 3:
            raise MatchedComparisonError(f"invalid V1.1 source-audit key: {key}")
        directory_name, filename = parts[1], parts[2]
        matches = [
            Path(value).resolve()
            for value in control_protocol["training_dirs"]
            if Path(value).name == directory_name
        ]
        if len(matches) != 1:
            raise MatchedComparisonError(f"cannot resolve V1.1 source-audit key: {key}")
        return matches[0] / filename
    if key in {
        "training-contract.json",
        "training-summary.json",
        "validation-instance-manifest.json",
        "validation-ledger/episode-0080.json",
    }:
        return v22_source_dir / key
    raise MatchedComparisonError(f"unknown V1.1 source-audit key: {key}")


def _authenticate_v11_control(
    control_dir: Path,
    *,
    v22_source_dir: Path,
    v23_identities: Mapping[int, Mapping],
) -> tuple[tuple[dict, ...], dict, dict[int, EpisodeInstance], dict[str, Path]]:
    control_dir = control_dir.resolve()
    paths = _verify_raw_pins(control_dir, V11_CONTROL_RAW_SHA256, label="v1.1-control")
    protocol = _load_json(control_dir / "protocol-manifest.json")
    instance_manifest = _load_json(control_dir / "instance-manifest.json")
    report = _load_json(control_dir / "control-report.json")
    audit = _load_json(control_dir / "control-audit.json")

    expected_protocol = {
        "schema_version": 1,
        "protocol": V11_CONTROL_PROTOCOL,
        "scope": "development_only_already_opened_85000_85011",
        "performance_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "model_seeds": MODEL_SEEDS,
        "evaluation_seeds": PANEL_SEEDS,
        "rollout_count": len(MODEL_SEEDS) * len(PANEL_SEEDS),
        "evaluation_policy": "deterministic_greedy_epsilon_zero",
        "evaluation_epsilon": 0.0,
        "evaluation_learning": False,
        "gamma": 0.99,
        "max_steps": MAX_STEPS,
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "reported_return": "undiscounted_dense_operational_return",
        "instance_manifest_sha256": V11_INSTANCE_MANIFEST_CANONICAL_SHA256,
    }
    for field, expected in expected_protocol.items():
        _require_equal(f"V1.1 control protocol.{field}", protocol.get(field), expected)
    checkpoints = protocol.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(MODEL_SEEDS):
        raise MatchedComparisonError("V1.1 control must bind three selected checkpoints")
    _require_equal("V1.1 checkpoint model seeds", [item.get("model_seed") for item in checkpoints], MODEL_SEEDS)
    for checkpoint in checkpoints:
        _require_equal("V1.1 checkpoint eligibility", checkpoint.get("deployment_checkpoint_eligible"), True)
        checkpoint_path = Path(checkpoint["path"]).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        _require_equal("V1.1 checkpoint raw SHA", _sha256_file(checkpoint_path), checkpoint["sha256"])
        paths[f"v1.1-checkpoint/model-seed-{checkpoint['model_seed']}"] = checkpoint_path

    _require_equal(
        "V1.1 instance manifest canonical SHA",
        _verify_self_hash(instance_manifest, "manifest_sha256", name="V1.1 instance manifest"),
        V11_INSTANCE_MANIFEST_CANONICAL_SHA256,
    )
    _require_equal("V1.1 instance seeds", instance_manifest.get("seeds"), PANEL_SEEDS)
    _require_equal("V1.1 instance manifest final panel", instance_manifest.get("final_86xxx_panel_opened"), False)
    instance_records = instance_manifest.get("instances")
    if not isinstance(instance_records, list) or len(instance_records) != len(PANEL_SEEDS):
        raise MatchedComparisonError("V1.1 instance manifest is not the exact 12-row panel")
    records_by_seed = {int(item["instance_seed"]): item for item in instance_records}
    _require_equal("V1.1 instance manifest seed grid", sorted(records_by_seed), PANEL_SEEDS)
    instances = {}
    for seed in PANEL_SEEDS:
        record = records_by_seed[seed]
        v23 = v23_identities[seed]
        for left, right in (
            ("episode_instance_id", "episode_instance_id"),
            ("schedule_id", "schedule_id"),
            ("episode_instance_sha256", "episode_instance_sha256"),
        ):
            _require_equal(f"seed {seed} V1.1/V2.3 {left}", record.get(left), v23.get(right))
        saved_path = control_dir / "instances" / f"seed-{seed}.json"
        if not saved_path.is_file():
            raise FileNotFoundError(saved_path)
        _require_equal(f"seed {seed} saved instance raw SHA", _sha256_file(saved_path), record["saved_file_sha256"])
        instance = EpisodeInstance.from_json(saved_path.read_text(encoding="utf-8"))
        _require_equal(f"seed {seed} instance seed", instance.seed, seed)
        _require_equal(f"seed {seed} instance id", instance.instance_id, record["episode_instance_id"])
        _require_equal(f"seed {seed} schedule id", instance.schedule_id, record["schedule_id"])
        _require_equal(f"seed {seed} canonical instance SHA", _canonical_instance_sha(instance), record["episode_instance_sha256"])
        instances[seed] = instance
        paths[f"v1.1-control/instances/seed-{seed}.json"] = saved_path

    for field, expected in {
        "schema_version": 1,
        "protocol": V11_CONTROL_PROTOCOL,
        "development_only": True,
        "performance_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "source_artifacts_unchanged": True,
        "authenticated_v1_1_bundle_count": 3,
        "authenticated_selected_checkpoint_count": 3,
        "regenerated_instance_count": 12,
        "rollout_count": 36,
        "evaluation_epsilon": 0.0,
        "evaluation_learning": False,
        "all_complete_exact_safe": True,
    }.items():
        _require_equal(f"V1.1 control audit.{field}", audit.get(field), expected)
    _require_equal(
        "V1.1 control source hashes unchanged",
        audit.get("source_artifact_sha256_after"),
        audit.get("source_artifact_sha256_before"),
    )
    for key, expected_sha in audit["source_artifact_sha256_before"].items():
        source_path = _resolve_audit_source_path(
            key, control_protocol=protocol, v22_source_dir=v22_source_dir.resolve()
        )
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        _require_equal(f"V1.1 current source SHA {key}", _sha256_file(source_path), expected_sha)
        paths[f"v1.1-source/{key}"] = source_path

    report_rows = report.get("runs")
    if not isinstance(report_rows, list) or len(report_rows) != 36:
        raise MatchedComparisonError("V1.1 report must preserve all 36 source rows")
    report_by_key = {
        (int(row["model_seed"]), int(row["instance_seed"])): row
        for row in report_rows
    }
    expected_grid = {(model, seed) for model in MODEL_SEEDS for seed in PANEL_SEEDS}
    _require_equal("V1.1 report crossed grid", set(report_by_key), expected_grid)

    ledger_manifest = audit.get("ledger_manifest")
    if not isinstance(ledger_manifest, list) or len(ledger_manifest) != 36:
        raise MatchedComparisonError("V1.1 audit must bind all 36 atomic ledgers")
    ledger_rows = {}
    for record in ledger_manifest:
        ledger_path = Path(record["path"]).resolve()
        if not ledger_path.is_relative_to(control_dir) or not ledger_path.is_file():
            raise MatchedComparisonError("V1.1 ledger path is missing or outside control root")
        _require_equal("V1.1 ledger raw SHA", _sha256_file(ledger_path), record["sha256"])
        ledger = _load_json(ledger_path)
        _require_equal(
            "V1.1 ledger canonical SHA",
            _verify_self_hash(ledger, "ledger_sha256", name="V1.1 run ledger"),
            record["ledger_sha256"],
        )
        _require_equal("V1.1 ledger protocol", ledger.get("protocol"), V11_CONTROL_PROTOCOL)
        _require_equal("V1.1 ledger input fingerprint", ledger.get("input_fingerprint"), _digest_json(ledger["input_contract"]))
        run = ledger.get("run")
        if not isinstance(run, Mapping):
            raise MatchedComparisonError("V1.1 ledger has no run row")
        key = (int(run["model_seed"]), int(run["instance_seed"]))
        if key in ledger_rows:
            raise MatchedComparisonError(f"duplicate V1.1 ledger key: {key}")
        _require_equal("V1.1 ledger run key", ledger.get("run_key"), f"model-seed-{key[0]}:instance-seed-{key[1]}")
        _require_equal("V1.1 report/ledger row", report_by_key.get(key), run)
        input_contract = ledger["input_contract"]
        _require_equal("V1.1 input final panel", input_contract.get("final_86xxx_panel_opened"), False)
        _require_equal("V1.1 input evaluation epsilon", input_contract.get("evaluation_epsilon"), 0.0)
        _require_equal("V1.1 input evaluation learning", input_contract.get("evaluation_learning"), False)
        reused = dict(run)
        reused["_comparison_source_ledger_sha256"] = record["ledger_sha256"]
        ledger_rows[key] = reused
        paths[f"v1.1-control/run-ledger/{key[0]}-{key[1]}"] = ledger_path
    _require_equal("V1.1 audit ledger grid", set(ledger_rows), expected_grid)
    return tuple(ledger_rows[key] for key in sorted(ledger_rows)), protocol, instances, paths


def _authenticate_v22_diagnostic(
    source_dir: Path,
    *,
    v23_identities: Mapping[int, Mapping],
) -> tuple[dict, dict[str, Path]]:
    source_dir = source_dir.resolve()
    paths = _verify_raw_pins(source_dir, V22_RAW_SHA256, label="v2.2-diagnostic")
    source = _validate_v22_source(source_dir)
    _require_equal("V2.2 diagnostic canonical ledger SHA", source["reference_ledger_sha256"], "9bff145716f073181f8f725181d639e9286392f0d32d5def76d385fae90b4a7d")
    for seed in PANEL_SEEDS:
        record = source["instance_records"][seed]
        expected = v23_identities[seed]
        for field in ("episode_instance_id", "schedule_id", "episode_instance_sha256"):
            _require_equal(f"V2.2/V2.3 seed {seed} {field}", record[field], expected[field])
    summary = dict(source["reference_summary"])
    summary.update(
        {
            "method_id": V22_DIAGNOSTIC_METHOD,
            "checkpoint_episode": V22_DIAGNOSTIC_EPISODE,
            "status": "superseded_diagnostic_only",
            "included_in_primary_ranking": False,
            "included_in_pareto_fronts": False,
            "development_only": True,
            "performance_claim_authorized": False,
            "policy": "induced_nested_regularized_lagrangian_sample",
            "authoritative_original_rehandle_ucb": source["reference_history"].get(
                "physical_rehandle_rate_one_sided_95_ucb"
            ),
        }
    )
    return summary, paths


def authenticate_sources(
    *,
    v23_source_dir: Path,
    v11_control_dir: Path,
    v22_source_dir: Path,
) -> AuthenticatedSources:
    _validate_panel_constants()
    v23_contract, identities, v23_rows, selection, v23_paths = _authenticate_v23(v23_source_dir)
    v11_rows, v11_protocol, instances, v11_paths = _authenticate_v11_control(
        v11_control_dir,
        v22_source_dir=v22_source_dir,
        v23_identities=identities,
    )
    v22_diagnostic, v22_paths = _authenticate_v22_diagnostic(
        v22_source_dir, v23_identities=identities
    )
    all_paths = {**v23_paths, **v11_paths, **v22_paths}
    path_values = [path.resolve() for path in all_paths.values()]
    # Aliases are harmless across source manifests but must hash identically.
    by_path: dict[Path, set[str]] = {}
    for key, path in all_paths.items():
        by_path.setdefault(path.resolve(), set()).add(key)
    source_hashes = {
        key: _sha256_file(path) for key, path in sorted(all_paths.items())
    }
    if any(seed in FINAL_PANEL_SEEDS for seed in identities):
        raise MatchedComparisonError("source authentication attempted to open final panel")
    return AuthenticatedSources(
        v23_contract=dict(v23_contract),
        identities=identities,
        instances=instances,
        v23_rows=v23_rows,
        v23_selection=selection,
        v11_rows=v11_rows,
        v11_protocol=v11_protocol,
        v22_diagnostic=v22_diagnostic,
        source_paths=all_paths,
        source_sha256=source_hashes,
    )


def _timing_metrics(deviations: Sequence[float], *, window: float = 20.0) -> dict:
    values = tuple(float(value) for value in deviations)
    if not values:
        return {
            "mean_signed_deviation": None,
            "mean_absolute_error": None,
            "mean_tardiness": None,
            "mean_earliness": None,
            "within_target_window_rate": None,
        }
    return {
        "mean_signed_deviation": float(fmean(values)),
        "mean_absolute_error": float(fmean(abs(value) for value in values)),
        "mean_tardiness": float(fmean(max(value, 0.0) for value in values)),
        "mean_earliness": float(fmean(max(-value, 0.0) for value in values)),
        "within_target_window_rate": float(
            fmean(abs(value) <= float(window) for value in values)
        ),
    }


def _identity_fields(seed: int, identities: Mapping[int, Mapping]) -> dict:
    identity = identities[int(seed)]
    return {
        "instance_seed": int(seed),
        "instance_index": PANEL_SEEDS.index(int(seed)),
        "episode_instance_id": identity["episode_instance_id"],
        "schedule_id": identity["schedule_id"],
        "episode_instance_sha256": identity["episode_instance_sha256"],
    }


def _normalize_v23_rows(
    rows: Sequence[Mapping], identities: Mapping[int, Mapping]
) -> tuple[dict, ...]:
    normalized = []
    expected_grid = {
        (seed, index) for seed in PANEL_SEEDS for index in range(VALIDATION_POLICY_RNG_COUNT)
    }
    observed = {
        (int(row["instance_seed"]), int(row["policy_rng_index"])) for row in rows
    }
    _require_equal("V2.3 comparison source grid", observed, expected_grid)
    if len(rows) != len(expected_grid):
        raise MatchedComparisonError("V2.3 comparison source contains duplicate rows")
    for source in rows:
        seed = int(source["instance_seed"])
        rng_index = int(source["policy_rng_index"])
        identity = _identity_fields(seed, identities)
        for field in ("instance_seed", "instance_index", "schedule_id", "episode_instance_sha256"):
            _require_equal(f"V2.3 row {field}", source.get(field), identity[field])
        _require_equal("V2.3 row episode instance id", source.get("episode_instance_id"), identity["episode_instance_id"])
        _require_equal("V2.3 action RNG", source.get("policy_rng_seed"), validation_policy_rng_seed(identity["instance_index"], rng_index))
        deviations = tuple(float(value) for value in source.get("delivery_deviations", ()))
        timing = _timing_metrics(deviations, window=FROZEN_OBJECTIVE_SPEC.window)
        _require_equal("V2.3 row MAE", source.get("mean_absolute_error"), timing["mean_absolute_error"])
        physical = int(source["physical_storage_relocations"])
        _require_equal("V2.3 physical alias", source.get("physical_rehandles"), physical)
        selected_reconfigurations = int(source.get("selected_action_counts", {}).get("reconfigure", 0))
        _require_equal("V2.3 explicit reconfiguration/physical count", selected_reconfigurations, physical)
        normalized.append(
            {
                "protocol": PROTOCOL,
                "method_id": V23_METHOD,
                "method_category": "learned_online_primary",
                "information_regime": "online_arrived_only_exact_closed_admission_certificate",
                "replication_design": "four_fixed_action_rng_realizations_within_episode_instance",
                "model_seed": 10,
                "policy_rng_index": rng_index,
                "policy_rng_seed": int(source["policy_rng_seed"]),
                **identity,
                "dense_objective_return": float(source["dense_return"]),
                "delivery_deviations": deviations,
                **timing,
                "steps": int(source["steps"]),
                "required_deliveries": int(source["required_deliveries"]),
                "delivery_count": int(source["delivery_count"]),
                "physical_storage_relocations": physical,
                "physical_rehandles_per_100_required_deliveries": (
                    100.0 * physical / int(source["required_deliveries"])
                ),
                "target_bound_obstruction_clearances": None,
                "standalone_reconfigurations": None,
                "standalone_with_direct_delivery_available": None,
                "standalone_without_direct_delivery_available": None,
                "directly_deliverable_self_reconfigurations": None,
                "selected_reconfigure_action_count": selected_reconfigurations,
                "rehandle_decomposition_status": (
                    "decomposition_not_authenticated_in_source;selected_reconfigure_"
                    "action_count_equals_total_physical"
                ),
                "strict_method_success": bool(source["strict_method_success"]),
                "completion_rate": float(source["completion_rate"]),
                "all_selected_candidates_exact_safe": bool(source["all_selected_candidates_exact_safe"]),
                "illegal_drops": int(source["illegal_drops"]),
                "invalid_assignments": 0,
                "fallbacks": int(source["fallbacks"]),
                "witness_mismatches": int(source["witness_mismatches"]),
                "method_failure_reason": source["method_failure_reason"],
                "evaluation_learning": bool(source["evaluation_learning"]),
                "evaluation_epsilon": 0.0,
                "evaluation_policy": V23_POLICY_REALIZATION,
                "map_selection_used": bool(source["map_selection_used"]),
                "stochastic_selection_only": bool(source["stochastic_selection_only"]),
                "fresh_evaluation_clone": bool(source["fresh_evaluation_clone"]),
                "training_agent_unchanged": bool(source["training_agent_unchanged"]),
                "validation_batch_state_unchanged": bool(source["validation_batch_state_unchanged"]),
                "source_origin": "authenticated_v2_3_episode160_validation_ledger",
                "source_execution_reused": True,
                "source_checkpoint_sha256": V23_BEST_RAW_SHA256,
                "source_validation_checkpoint_sha256": V23_EP160_DIAGNOSTIC_RAW_SHA256,
                "source_ledger_sha256": V23_EP160_LEDGER_CANONICAL_SHA256,
                "source_manifest_sha256": V23_CANDIDATE_MANIFEST_CANONICAL_SHA256,
            }
        )
    for row in normalized:
        _validate_normalized_row_schema(row, V23_METHOD)
    return tuple(normalized)


def _normalize_v11_rows(
    rows: Sequence[Mapping], identities: Mapping[int, Mapping]
) -> tuple[dict, ...]:
    normalized = []
    expected = {(model, seed) for model in MODEL_SEEDS for seed in PANEL_SEEDS}
    observed = {(int(row["model_seed"]), int(row["instance_seed"])) for row in rows}
    _require_equal("V1.1 comparison source grid", observed, expected)
    if len(rows) != len(expected):
        raise MatchedComparisonError("V1.1 comparison source contains duplicate rows")
    for source in rows:
        seed = int(source["instance_seed"])
        identity = _identity_fields(seed, identities)
        _require_equal("V1.1 row instance id", source.get("instance_id"), identity["episode_instance_id"])
        _require_equal("V1.1 row schedule", source.get("schedule_id"), identity["schedule_id"])
        deviations = tuple(float(value) for value in source.get("delivery_deviations", ()))
        timing = _timing_metrics(deviations, window=FROZEN_OBJECTIVE_SPEC.window)
        for field in ("mean_signed_deviation", "mean_absolute_error", "mean_tardiness", "mean_earliness", "within_target_window_rate"):
            _require_equal(f"V1.1 row {field}", source.get(field), timing[field])
        contention = validate_contention_metric_record(source)
        physical = int(contention["physical_storage_relocations"])
        required = EXPECTED_DELIVERIES
        normalized.append(
            {
                "protocol": PROTOCOL,
                "method_id": V11_METHOD,
                "method_category": "learned_online_primary",
                "information_regime": str(source["information_regime"]),
                "replication_design": "three_fixed_training_seeds_crossed_with_episode_instances",
                "model_seed": int(source["model_seed"]),
                "policy_rng_index": None,
                "policy_rng_seed": None,
                **identity,
                "dense_objective_return": float(source["dense_objective_return"]),
                "delivery_deviations": deviations,
                **timing,
                "steps": int(source["steps"]),
                "required_deliveries": required,
                "delivery_count": int(source["delivery_count"]),
                "physical_storage_relocations": physical,
                "physical_rehandles_per_100_required_deliveries": 100.0 * physical / required,
                "target_bound_obstruction_clearances": int(contention["target_bound_obstruction_clearances"]),
                "standalone_reconfigurations": int(contention["standalone_reconfigurations"]),
                "standalone_with_direct_delivery_available": int(contention["standalone_with_direct_delivery_available"]),
                "standalone_without_direct_delivery_available": int(contention["standalone_without_direct_delivery_available"]),
                "directly_deliverable_self_reconfigurations": int(contention["directly_deliverable_self_reconfigurations"]),
                "rehandle_decomposition_status": "authenticated_canonical_event_time_schema_v1",
                "strict_method_success": bool(float(source["strict_method_success"])),
                "completion_rate": float(source["completion_rate"]),
                "all_selected_candidates_exact_safe": bool(source["all_selected_candidates_exact_safe"]),
                "illegal_drops": int(source["illegal_drops"]),
                "invalid_assignments": int(source["invalid_assignments"]),
                "fallbacks": int(source["fallbacks"]),
                "witness_mismatches": 0,
                "method_failure_reason": source["method_failure_reason"],
                "evaluation_learning": bool(source["evaluation_learning"]),
                "evaluation_epsilon": float(source["evaluation_epsilon"]),
                "evaluation_policy": "deterministic_greedy_epsilon_zero",
                "map_selection_used": None,
                "stochastic_selection_only": False,
                "fresh_evaluation_clone": None,
                "training_agent_unchanged": None,
                "validation_batch_state_unchanged": None,
                "source_origin": "authenticated_v1_1_three_seed_control_ledger",
                "source_execution_reused": True,
                "source_checkpoint_sha256": str(source["checkpoint_sha256"]),
                "source_validation_checkpoint_sha256": str(source["checkpoint_sha256"]),
                "source_ledger_sha256": str(
                    source["_comparison_source_ledger_sha256"]
                ),
                "source_manifest_sha256": V11_INSTANCE_MANIFEST_CANONICAL_SHA256,
            }
        )
    for row in normalized:
        _validate_normalized_row_schema(row, V11_METHOD)
    return tuple(normalized)


def _normalize_deterministic_baseline(
    method: str,
    raw: Mapping,
    instance: EpisodeInstance,
    identities: Mapping[int, Mapping],
) -> dict:
    seed = int(raw["eval_seed"])
    normalized = _normalize_baseline(
        method,
        dict(raw),
        instance,
        EXPECTED_DELIVERIES,
        FROZEN_OBJECTIVE_SPEC,
    )
    identity = _identity_fields(seed, identities)
    _require_equal("baseline normalized instance id", normalized.get("instance_id"), identity["episode_instance_id"])
    _require_equal("baseline normalized schedule", normalized.get("schedule_id"), identity["schedule_id"])
    deviations = tuple(float(value) for value in normalized.get("delivery_deviations", ()))
    timing = _timing_metrics(deviations, window=FROZEN_OBJECTIVE_SPEC.window)
    for field, value in timing.items():
        _require_equal(f"baseline normalized {field}", normalized.get(field), value)
    contention = validate_contention_metric_record(normalized)
    physical = int(contention["physical_storage_relocations"])
    return {
        "protocol": PROTOCOL,
        "method_id": method,
        "method_category": "deterministic_online_primary",
        "information_regime": "online_arrived_only",
        "replication_design": "one_deterministic_run_per_episode_instance",
        "model_seed": None,
        "policy_rng_index": None,
        "policy_rng_seed": None,
        **identity,
        # This is the exact dense rescore.  The environment legacy return is
        # retained only in method_audit and never enters the primary ranking.
        "dense_objective_return": float(normalized["dense_rescored_return"]),
        "delivery_deviations": deviations,
        **timing,
        "steps": int(normalized["steps"]),
        "required_deliveries": EXPECTED_DELIVERIES,
        "delivery_count": int(normalized["delivery_count"]),
        "physical_storage_relocations": physical,
        "physical_rehandles_per_100_required_deliveries": 100.0 * physical / EXPECTED_DELIVERIES,
        "target_bound_obstruction_clearances": int(contention["target_bound_obstruction_clearances"]),
        "standalone_reconfigurations": int(contention["standalone_reconfigurations"]),
        "standalone_with_direct_delivery_available": int(contention["standalone_with_direct_delivery_available"]),
        "standalone_without_direct_delivery_available": int(contention["standalone_without_direct_delivery_available"]),
        "directly_deliverable_self_reconfigurations": int(contention["directly_deliverable_self_reconfigurations"]),
        "rehandle_decomposition_status": "authenticated_canonical_event_time_schema_v1",
        "strict_method_success": bool(float(normalized["strict_method_success"])),
        "completion_rate": float(normalized["completion_rate"]),
        "all_selected_candidates_exact_safe": None,
        "illegal_drops": int(normalized["illegal_drops"]),
        "invalid_assignments": int(normalized["invalid_assignments"]),
        "fallbacks": int(normalized["fallbacks"]),
        "witness_mismatches": 0,
        "method_failure_reason": normalized["method_failure_reason"],
        "evaluation_learning": False,
        "evaluation_epsilon": 0.0,
        "evaluation_policy": "deterministic_online_duration_aware_scheduler",
        "map_selection_used": None,
        "stochastic_selection_only": None,
        "fresh_evaluation_clone": True,
        "training_agent_unchanged": True,
        "validation_batch_state_unchanged": True,
        "assignment_source": DETERMINISTIC_METHOD_TO_SOURCE[method],
        "legacy_environment_return": float(normalized["return"]),
        "planning_seconds": float(normalized["planning_seconds"]),
        "episode_wall_seconds": float(normalized["episode_wall_seconds"]),
        "method_audit": normalized["method_audit"],
        "source_origin": "new_deterministic_online_baseline_execution",
        "source_execution_reused": False,
        "source_checkpoint_sha256": None,
        "source_validation_checkpoint_sha256": None,
        "source_ledger_sha256": None,
        "source_manifest_sha256": V11_INSTANCE_MANIFEST_CANONICAL_SHA256,
    }


def _failed_baseline_row(
    method: str,
    seed: int,
    identities: Mapping[int, Mapping],
    error: Exception,
) -> dict:
    return {
        "protocol": PROTOCOL,
        "method_id": method,
        "method_category": "deterministic_online_primary",
        "information_regime": "online_arrived_only",
        "replication_design": "one_deterministic_run_per_episode_instance",
        "model_seed": None,
        "policy_rng_index": None,
        "policy_rng_seed": None,
        **_identity_fields(seed, identities),
        "dense_objective_return": None,
        "delivery_deviations": (),
        **_timing_metrics(()),
        "steps": None,
        "required_deliveries": EXPECTED_DELIVERIES,
        "delivery_count": 0,
        "physical_storage_relocations": None,
        "physical_rehandles_per_100_required_deliveries": None,
        "target_bound_obstruction_clearances": None,
        "standalone_reconfigurations": None,
        "standalone_with_direct_delivery_available": None,
        "standalone_without_direct_delivery_available": None,
        "directly_deliverable_self_reconfigurations": None,
        "rehandle_decomposition_status": "execution_failed",
        "strict_method_success": False,
        "completion_rate": 0.0,
        "all_selected_candidates_exact_safe": None,
        "illegal_drops": None,
        "invalid_assignments": None,
        "fallbacks": None,
        "witness_mismatches": None,
        "method_failure_reason": f"execution_exception:{type(error).__name__}:{error}",
        "evaluation_learning": False,
        "evaluation_epsilon": 0.0,
        "evaluation_policy": "deterministic_online_duration_aware_scheduler",
        "map_selection_used": None,
        "stochastic_selection_only": None,
        "fresh_evaluation_clone": True,
        "training_agent_unchanged": True,
        "validation_batch_state_unchanged": True,
        "assignment_source": DETERMINISTIC_METHOD_TO_SOURCE[method],
        "legacy_environment_return": None,
        "planning_seconds": None,
        "episode_wall_seconds": None,
        "method_audit": None,
        "source_origin": "new_deterministic_online_baseline_execution_failure",
        "source_execution_reused": False,
        "source_checkpoint_sha256": None,
        "source_validation_checkpoint_sha256": None,
        "source_ledger_sha256": None,
        "source_manifest_sha256": V11_INSTANCE_MANIFEST_CANONICAL_SHA256,
    }


def _baseline_source_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    relatives = (
        "compare_vcg_v2_3_matched_baselines.py",
        "compare_vcg_dense_pareto.py",
        "compare_viability_graph_baselines.py",
        "evaluate_vcg_dense_v1_1_v2_2_panel_control.py",
        "train_vcg_constrained_v2_3.py",
        "vcg_objective_audit.py",
        "track_b_urgency_evaluate.py",
        "contention_metrics.py",
        "example/episode_instance.py",
        "example/block_instance.py",
        "example/small_rooms_env.py",
        "example/controller_options.py",
        "example/urgency_scheduler.py",
        "example/yard_geometry.py",
        "example/helper/timing_metrics.py",
        "example/helper/occupancy_pressure.py",
        "example/helper/tools.py",
        "example/Options/AcceptStoreOption.py",
        "example/Options/DeliverOption.py",
        "example/Options/PickupRipeOption.py",
        "example/Options/RetrieveDeliverOption.py",
        "example/Options/StrategicDeferOption.py",
        "example/Options/pickupOption.py",
        "example/Options/selector_v5.py",
        "example/Options/storeOption.py",
        "environment.py",
        "option.py",
        "options_agent.py",
        "gated_agent.py",
        "GA/helper_functions.py",
        "PSLAP/track_a.py",
        "PSLAP/dynamic_policy.py",
        "PSLAP/dynamic_yard.py",
        "PSLAP/ga_policy.py",
        "PSLAP/ga_optimizer.py",
        "PSLAP/online_policy.py",
        "PSLAP/neutral_protocol.py",
        "PSLAP/retrieval_context.py",
        "PSLAP/retrieval_dispatch.py",
        "PSLAP/retrieval_executor.py",
        "PSLAP/checkpoint_identity.py",
    )
    result = {}
    for relative in relatives:
        path = (root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = path
    return result


def build_comparison_contract(
    sources: AuthenticatedSources, *, device: str
) -> dict:
    source_files = _baseline_source_paths()
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "development_only_already_opened_85000_85011",
        "performance_claim_authorized": False,
        "confirmatory_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "final_622xxx_action_rng_panel_opened": False,
        "complete_case_filtering_permitted": False,
        "panel_seeds": PANEL_SEEDS,
        "episode_instance_count": len(PANEL_SEEDS),
        "expected_deliveries_per_episode": EXPECTED_DELIVERIES,
        "primary_online_methods": PRIMARY_METHODS,
        "learned_source_rows_reused": {
            V23_METHOD: 48,
            V11_METHOD: 36,
        },
        "learned_policies_reexecuted": False,
        "deterministic_methods": DETERMINISTIC_METHODS,
        "deterministic_assignment_sources": DETERMINISTIC_METHOD_TO_SOURCE,
        "deterministic_runs_per_method": len(PANEL_SEEDS),
        "expected_new_deterministic_run_count": len(DETERMINISTIC_METHODS)
        * len(PANEL_SEEDS),
        "aggregation": {
            V23_METHOD: (
                "average_four_fixed_action_rng_realizations_within_each_"
                "EpisodeInstance_then_equal_weight_12_EpisodeInstances"
            ),
            V11_METHOD: (
                "equal_weight_three_selected_training_seeds_and_equal_weight_"
                "12_EpisodeInstances;retain_three_per_seed_summaries"
            ),
            "deterministic_online": "one_run_per_EpisodeInstance_then_equal_weight_12_EpisodeInstances",
            "statistical_unit": "EpisodeInstance_cluster",
            "descriptive_one_sided_t_95_df11": DESCRIPTIVE_ONE_SIDED_T_95_DF11,
            "no_pooled_48_or_36_row_inference": True,
        },
        "safety_protocol": {
            "whole_method_exclusion": True,
            "all_expected_rows_required": True,
            "no_partial_instance_filtering": True,
            "strict_success_and_full_completion_required": True,
            "zero_illegal_drop_invalid_assignment_fallback_required": True,
        },
        "seed_stability_screen_definition_frozen_before_baseline_outcomes": {
            "role": "development_triage_not_confirmatory_superiority_gate",
            "scientifically_precise_role": (
                "single-training-seed_method-capacity_or_pareto_screen"
            ),
            "training_seed_stability_established_by_this_screen": False,
            "independent_v2_3_training_seed_replication_still_required": True,
            "requires_authenticated_selected_v2_3_episode": V23_SELECTED_EPISODE,
            "requires_all_48_v2_3_rows_strict_safe_complete": True,
            "dominance_coordinates": {
                "mean_absolute_error": "lower_is_better",
                "physical_rehandles_per_100_required_deliveries": "lower_is_better",
            },
            "pass_condition": (
                "no_whole_method_safety_eligible_matched_online_comparator_"
                "weakly_dominates_v2_3_on_both_coordinates_with_at_least_one_strict"
            ),
            "return_steps_and_timing_direction_are_reported_guardrails_not_gates": True,
            "scalarization_used": False,
            "requires_beating_all_baselines": False,
        },
        "common_primary_rehandle_metric": (
            "physical_storage_relocations_per_100_required_deliveries"
        ),
        "target_bound_clearance_is_mechanism_only_not_common_total": True,
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "reported_return": "undiscounted_dense_operational_return",
        "environment": sources.v23_contract["environment"],
        "execution_device": str(device),
        "fixed_baseline_configuration": {
            "max_steps": MAX_STEPS,
            "max_defer_steps": MAX_DEFER_STEPS,
            "lookahead_margin_steps": LOOKAHEAD_MARGIN_STEPS,
            "rolling_population": ROLLING_POPULATION,
            "rolling_generations": ROLLING_GENERATIONS,
            "rolling_ga_egress_weight": ROLLING_GA_EGRESS_WEIGHT,
            "ga_seed_base": GA_SEED_BASE,
        },
        "v2_3_selection": {
            "selected_episode": sources.v23_selection["selected_episode"],
            "selection_recomputed_without_training_summary": True,
            "contract_sha256": V23_CONTRACT_CANONICAL_SHA256,
            "candidate_manifest_sha256": V23_CANDIDATE_MANIFEST_CANONICAL_SHA256,
            "selected_best_raw_sha256": V23_BEST_RAW_SHA256,
            "episode160_ledger_sha256": V23_EP160_LEDGER_CANONICAL_SHA256,
        },
        "v1_1_control": {
            "protocol": V11_CONTROL_PROTOCOL,
            "model_seeds": MODEL_SEEDS,
            "instance_manifest_sha256": V11_INSTANCE_MANIFEST_CANONICAL_SHA256,
        },
        "source_artifact_sha256": dict(sorted(sources.source_sha256.items())),
        "baseline_source_sha256": {
            name: _sha256_file(path) for name, path in sorted(source_files.items())
        },
        "offline_and_pending_methods_ranked": False,
    }
    contract["contract_sha256"] = _digest_json(contract)
    return contract


def _baseline_input_contract(
    method: str,
    seed: int,
    *,
    sources: AuthenticatedSources,
    comparison_contract: Mapping,
) -> dict:
    if method not in DETERMINISTIC_METHOD_TO_SOURCE:
        raise MatchedComparisonError(f"unknown deterministic method: {method}")
    if seed not in PANEL_SEEDS or seed in FINAL_PANEL_SEEDS:
        raise MatchedComparisonError(f"baseline seed is outside exact development panel: {seed}")
    identity = _identity_fields(seed, sources.identities)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "comparison_contract_sha256": comparison_contract["contract_sha256"],
        "method_id": method,
        "assignment_source": DETERMINISTIC_METHOD_TO_SOURCE[method],
        **identity,
        "information_regime": "online_arrived_only",
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "reported_return": "undiscounted_dense_operational_return",
        "max_steps": MAX_STEPS,
        "max_defer_steps": MAX_DEFER_STEPS,
        "lookahead_margin_steps": LOOKAHEAD_MARGIN_STEPS,
        "rolling_population": ROLLING_POPULATION,
        "rolling_generations": ROLLING_GENERATIONS,
        "rolling_ga_egress_weight": ROLLING_GA_EGRESS_WEIGHT,
        "ga_seed": GA_SEED_BASE + seed,
        "final_86xxx_panel_opened": False,
    }


def _baseline_ledger_path(output_dir: Path, method: str, seed: int) -> Path:
    return output_dir / "run-ledger" / method / f"seed-{seed}.json"


_COMMON_NORMALIZED_ROW_KEYS = frozenset(
    {
        "protocol", "method_id", "method_category", "information_regime",
        "replication_design",
        "model_seed", "policy_rng_index", "policy_rng_seed", "instance_seed",
        "instance_index", "episode_instance_id", "schedule_id",
        "episode_instance_sha256", "dense_objective_return",
        "delivery_deviations", "mean_signed_deviation", "mean_absolute_error",
        "mean_tardiness", "mean_earliness", "within_target_window_rate",
        "steps", "required_deliveries", "delivery_count",
        "physical_storage_relocations",
        "physical_rehandles_per_100_required_deliveries",
        "target_bound_obstruction_clearances", "standalone_reconfigurations",
        "standalone_with_direct_delivery_available",
        "standalone_without_direct_delivery_available",
        "directly_deliverable_self_reconfigurations",
        "rehandle_decomposition_status", "strict_method_success",
        "completion_rate", "all_selected_candidates_exact_safe",
        "illegal_drops", "invalid_assignments", "fallbacks",
        "witness_mismatches", "method_failure_reason", "evaluation_learning",
        "evaluation_epsilon", "evaluation_policy", "map_selection_used",
        "stochastic_selection_only", "fresh_evaluation_clone",
        "training_agent_unchanged", "validation_batch_state_unchanged",
        "source_origin", "source_execution_reused",
        "source_checkpoint_sha256", "source_validation_checkpoint_sha256",
        "source_ledger_sha256",
        "source_manifest_sha256",
    }
)
_V23_NORMALIZED_ROW_KEYS = frozenset(
    {*_COMMON_NORMALIZED_ROW_KEYS, "selected_reconfigure_action_count"}
)
_V11_NORMALIZED_ROW_KEYS = _COMMON_NORMALIZED_ROW_KEYS
_BASELINE_EXTRA_ROW_KEYS = frozenset(
    {
        "assignment_source", "legacy_environment_return", "planning_seconds",
        "episode_wall_seconds", "method_audit",
    }
)
_BASELINE_NORMALIZED_ROW_KEYS = frozenset(
    {*_COMMON_NORMALIZED_ROW_KEYS, *_BASELINE_EXTRA_ROW_KEYS}
)


def _validate_normalized_row_schema(row: Mapping, method: str) -> None:
    expected = (
        _V23_NORMALIZED_ROW_KEYS
        if method == V23_METHOD
        else _V11_NORMALIZED_ROW_KEYS
        if method == V11_METHOD
        else _BASELINE_NORMALIZED_ROW_KEYS
        if method in DETERMINISTIC_METHODS
        else None
    )
    if expected is None:
        raise MatchedComparisonError(f"unknown normalized row method: {method}")
    observed = frozenset(row)
    if observed != expected:
        raise MatchedComparisonError(
            f"normalized row schema mismatch for {method}: "
            f"missing={sorted(expected - observed)!r}, extra={sorted(observed - expected)!r}"
        )


def _load_or_execute_baseline(
    *,
    output_dir: Path,
    method: str,
    seed: int,
    input_contract: Mapping,
    execute: bool,
    executor: Callable[[], dict],
) -> Optional[dict]:
    path = _baseline_ledger_path(output_dir, method, seed)
    fingerprint = _digest_json(input_contract)
    if path.is_file():
        ledger = _load_json(path)
        _require_equal(
            "baseline ledger schema",
            set(ledger),
            {
                "schema_version", "protocol", "run_key", "input_contract",
                "input_fingerprint", "run", "ledger_sha256",
            },
        )
        _require_equal("baseline ledger schema version", ledger.get("schema_version"), SCHEMA_VERSION)
        _require_equal("baseline ledger canonical SHA", _verify_self_hash(ledger, "ledger_sha256", name="baseline run ledger"), ledger.get("ledger_sha256"))
        _require_equal("baseline ledger protocol", ledger.get("protocol"), PROTOCOL)
        _require_equal("baseline ledger run key", ledger.get("run_key"), f"{method}:{seed}")
        _require_equal("baseline ledger input", ledger.get("input_contract"), input_contract)
        _require_equal("baseline ledger input fingerprint", ledger.get("input_fingerprint"), fingerprint)
        row = ledger.get("run")
        if not isinstance(row, Mapping):
            raise MatchedComparisonError(f"baseline ledger has no run row: {path}")
    elif not execute:
        return None
    else:
        row = executor()
        _require_equal("baseline executor method", row.get("method_id"), method)
        _require_equal("baseline executor seed", row.get("instance_seed"), seed)
        _validate_normalized_row_schema(row, method)
        ledger = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "run_key": f"{method}:{seed}",
            "input_contract": dict(input_contract),
            "input_fingerprint": fingerprint,
            "run": dict(row),
        }
        ledger["ledger_sha256"] = _digest_json(ledger)
        _atomic_json(path, ledger)
    _require_equal("baseline ledger instance id", row.get("episode_instance_id"), input_contract["episode_instance_id"])
    _require_equal("baseline ledger schedule id", row.get("schedule_id"), input_contract["schedule_id"])
    _require_equal("baseline ledger instance SHA", row.get("episode_instance_sha256"), input_contract["episode_instance_sha256"])
    _validate_normalized_row_schema(row, method)
    for field, expected in {
        "protocol": PROTOCOL,
        "method_id": method,
        "method_category": "deterministic_online_primary",
        "information_regime": "online_arrived_only",
        "assignment_source": input_contract["assignment_source"],
        "evaluation_learning": False,
        "evaluation_epsilon": 0.0,
        "source_execution_reused": False,
        "source_checkpoint_sha256": None,
        "source_validation_checkpoint_sha256": None,
        "source_ledger_sha256": None,
    }.items():
        _require_equal(f"baseline ledger row.{field}", row.get(field), expected)
    if row.get("strict_method_success") is True:
        legacy = row.get("legacy_environment_return")
        deviations = row.get("delivery_deviations")
        if legacy is None or not math.isfinite(float(legacy)):
            raise MatchedComparisonError(
                "successful baseline ledger lacks a finite legacy return for dense authentication"
            )
        if not isinstance(deviations, (tuple, list)):
            raise MatchedComparisonError(
                "successful baseline ledger lacks delivery deviations"
            )
        _, expected_dense = _dual_rescore_from_legacy_return(
            float(legacy), deviations, FROZEN_OBJECTIVE_SPEC
        )
        if not math.isclose(
            float(row.get("dense_objective_return")),
            float(expected_dense),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise MatchedComparisonError(
                "successful baseline ledger dense objective fails exact rescore"
            )
    return dict(row)


def execute_or_load_baseline_grid(
    *,
    output_dir: Path,
    sources: AuthenticatedSources,
    comparison_contract: Mapping,
    device,
    execute: bool,
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    runtime_args = _baseline_args({"environment": sources.v23_contract["environment"]}, device)
    rows = []
    missing = []
    for method in DETERMINISTIC_METHODS:
        for seed in PANEL_SEEDS:
            instance = sources.instances[seed]
            input_contract = _baseline_input_contract(
                method, seed, sources=sources, comparison_contract=comparison_contract
            )

            def run_one(method=method, seed=seed, instance=instance):
                try:
                    raw = evaluate_assignment_ablation_one(
                        runtime_args,
                        seed,
                        None,
                        assignment_source=DETERMINISTIC_METHOD_TO_SOURCE[method],
                        episode_instance=instance,
                    )
                    return _normalize_deterministic_baseline(
                        method, raw, instance, sources.identities
                    )
                except Exception as error:  # preserve a full, auditable failed grid
                    return _failed_baseline_row(method, seed, sources.identities, error)

            row = _load_or_execute_baseline(
                output_dir=output_dir,
                method=method,
                seed=seed,
                input_contract=input_contract,
                execute=execute,
                executor=run_one,
            )
            if row is None:
                missing.append(
                    {
                        "method_id": method,
                        "instance_seed": seed,
                        "ledger_path": str(
                            _baseline_ledger_path(output_dir, method, seed)
                        ),
                    }
                )
                continue
            rows.append(row)
            if execute:
                dense = row.get("dense_objective_return")
                dense_text = "NA" if dense is None else f"{float(dense):8.2f}"
                print(
                    f"[{method}] instance={seed} DenseR={dense_text} "
                    f"strict={int(bool(row['strict_method_success']))} "
                    f"failure={row['method_failure_reason']}",
                    flush=True,
                )
    return tuple(rows), tuple(missing)


COMMON_ADDITIVE_METRICS = (
    "mean_dense_objective_return",
    "mean_signed_deviation",
    "mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "mean_steps",
    "physical_rehandles_per_100_required_deliveries",
    "mean_physical_rehandles_per_episode",
)
DECOMPOSITION_COMPONENTS = (
    "target_bound_obstruction_clearances",
    "standalone_reconfigurations",
    "standalone_with_direct_delivery_available",
    "standalone_without_direct_delivery_available",
    "directly_deliverable_self_reconfigurations",
)


def _row_safety_issues(row: Mapping, identities: Mapping[int, Mapping]) -> list[str]:
    issues = []
    seed = row.get("instance_seed")
    if seed not in PANEL_SEEDS:
        return ["instance_seed_outside_exact_panel"]
    identity = _identity_fields(int(seed), identities)
    for field in (
        "instance_seed",
        "instance_index",
        "episode_instance_id",
        "schedule_id",
        "episode_instance_sha256",
    ):
        if _json_safe(row.get(field)) != _json_safe(identity[field]):
            issues.append(f"identity_mismatch:{field}")
    finite_fields = (
        "dense_objective_return",
        "mean_signed_deviation",
        "mean_absolute_error",
        "mean_tardiness",
        "mean_earliness",
        "within_target_window_rate",
        "steps",
        "physical_storage_relocations",
        "physical_rehandles_per_100_required_deliveries",
    )
    for field in finite_fields:
        try:
            if row.get(field) is None or not math.isfinite(float(row[field])):
                issues.append(f"missing_or_nonfinite:{field}")
        except (TypeError, ValueError):
            issues.append(f"missing_or_nonfinite:{field}")
    if row.get("strict_method_success") is not True:
        issues.append("strict_method_failure")
    try:
        if not math.isclose(float(row.get("completion_rate")), 1.0):
            issues.append("incomplete_episode")
    except (TypeError, ValueError):
        issues.append("invalid_completion_rate")
    if row.get("method_failure_reason") is not None:
        issues.append("method_failure_reason_present")
    if row.get("evaluation_learning") is not False:
        issues.append("evaluation_learning_enabled")
    try:
        if int(row.get("required_deliveries")) != EXPECTED_DELIVERIES:
            issues.append("required_delivery_count_mismatch")
        if int(row.get("delivery_count")) != EXPECTED_DELIVERIES:
            issues.append("delivery_count_mismatch")
    except (TypeError, ValueError):
        issues.append("invalid_delivery_count")
    deviations = row.get("delivery_deviations")
    if not isinstance(deviations, (tuple, list)) or len(deviations) != EXPECTED_DELIVERIES:
        issues.append("delivery_deviation_sequence_incomplete")
    else:
        timing = _timing_metrics(deviations, window=FROZEN_OBJECTIVE_SPEC.window)
        for field, expected in timing.items():
            observed = row.get(field)
            if observed is None or not math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=1e-10
            ):
                issues.append(f"timing_metric_mismatch:{field}")
    try:
        steps = int(row.get("steps"))
        if not 0 < steps <= MAX_STEPS:
            issues.append("steps_outside_horizon")
    except (TypeError, ValueError):
        issues.append("invalid_steps")
    for field in (
        "illegal_drops",
        "invalid_assignments",
        "fallbacks",
        "witness_mismatches",
    ):
        try:
            if int(row.get(field)) != 0:
                issues.append(f"nonzero:{field}")
        except (TypeError, ValueError):
            issues.append(f"invalid:{field}")
    try:
        physical = int(row.get("physical_storage_relocations"))
        required = int(row.get("required_deliveries"))
        if physical < 0:
            issues.append("negative_physical_rehandles")
        expected_rate = 100.0 * physical / required
        if not math.isclose(
            float(row.get("physical_rehandles_per_100_required_deliveries")),
            expected_rate,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            issues.append("physical_rehandle_rate_mismatch")
    except (TypeError, ValueError, ZeroDivisionError):
        issues.append("invalid_physical_rehandle_count")
    available_components = [row.get(field) for field in DECOMPOSITION_COMPONENTS]
    if all(value is not None for value in available_components):
        try:
            validate_contention_metric_record(
                {
                    "contention_metric_schema_version": CONTENTION_METRIC_SCHEMA_VERSION,
                    "physical_storage_relocations": row["physical_storage_relocations"],
                    **{field: row[field] for field in DECOMPOSITION_COMPONENTS},
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"invalid_rehandle_decomposition:{error}")
    elif any(value is not None for value in available_components):
        issues.append("partially_recorded_rehandle_decomposition")

    method = row.get("method_id")
    if method == V23_METHOD:
        for field in (
            "all_selected_candidates_exact_safe",
            "fresh_evaluation_clone",
            "training_agent_unchanged",
            "validation_batch_state_unchanged",
            "stochastic_selection_only",
        ):
            if row.get(field) is not True:
                issues.append(f"v2_3_false:{field}")
        if row.get("map_selection_used") is not False:
            issues.append("v2_3_map_selection_used")
        if row.get("evaluation_policy") != V23_POLICY_REALIZATION:
            issues.append("v2_3_policy_mismatch")
        if row.get("target_bound_obstruction_clearances") is not None:
            issues.append("v2_3_unauthenticated_decomposition_fabricated")
    elif method == V11_METHOD:
        if row.get("all_selected_candidates_exact_safe") is not True:
            issues.append("v1_1_not_exact_safe")
        if not math.isclose(float(row.get("evaluation_epsilon", -1.0)), 0.0):
            issues.append("v1_1_nonzero_epsilon")
        if row.get("evaluation_policy") != "deterministic_greedy_epsilon_zero":
            issues.append("v1_1_policy_mismatch")
    elif method in DETERMINISTIC_METHODS:
        if row.get("assignment_source") != DETERMINISTIC_METHOD_TO_SOURCE[method]:
            issues.append("baseline_assignment_source_mismatch")
        legacy = row.get("legacy_environment_return")
        if legacy is None:
            issues.append("baseline_legacy_return_missing_for_dense_authentication")
        elif isinstance(deviations, (tuple, list)):
            try:
                _, expected_dense = _dual_rescore_from_legacy_return(
                    float(legacy), deviations, FROZEN_OBJECTIVE_SPEC
                )
                if not math.isclose(
                    float(row.get("dense_objective_return")),
                    float(expected_dense),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    issues.append("baseline_dense_rescore_mismatch")
            except (TypeError, ValueError):
                issues.append("baseline_dense_rescore_invalid")
    else:
        issues.append("unknown_primary_method")
    return sorted(set(issues))


def _grid_issues(method: str, rows: Sequence[Mapping]) -> list[str]:
    issues = []
    if method == V23_METHOD:
        expected = {
            (seed, index)
            for seed in PANEL_SEEDS
            for index in range(VALIDATION_POLICY_RNG_COUNT)
        }
        observed = {
            (int(row["instance_seed"]), int(row["policy_rng_index"]))
            for row in rows
        }
        if observed != expected:
            issues.append("v2_3_crossed_instance_rng_grid_incomplete")
        if len(rows) != len(expected):
            issues.append("v2_3_row_count_or_duplicates")
    elif method == V11_METHOD:
        expected = {(model, seed) for model in MODEL_SEEDS for seed in PANEL_SEEDS}
        observed = {
            (int(row["model_seed"]), int(row["instance_seed"])) for row in rows
        }
        if observed != expected:
            issues.append("v1_1_crossed_model_instance_grid_incomplete")
        if len(rows) != len(expected):
            issues.append("v1_1_row_count_or_duplicates")
    else:
        expected = set(PANEL_SEEDS)
        observed = {int(row["instance_seed"]) for row in rows}
        if observed != expected:
            issues.append("deterministic_instance_grid_incomplete")
        if len(rows) != len(expected):
            issues.append("deterministic_row_count_or_duplicates")
    return issues


def _metrics_for_rows(rows: Sequence[Mapping]) -> dict:
    rows = tuple(rows)
    if not rows:
        raise MatchedComparisonError("cannot aggregate an empty row set")
    deviations = tuple(
        float(value) for row in rows for value in row["delivery_deviations"]
    )
    timing = _timing_metrics(deviations, window=FROZEN_OBJECTIVE_SPEC.window)
    required = sum(int(row["required_deliveries"]) for row in rows)
    physical = sum(int(row["physical_storage_relocations"]) for row in rows)
    metrics = {
        "mean_dense_objective_return": float(
            fmean(float(row["dense_objective_return"]) for row in rows)
        ),
        **timing,
        "absolute_mean_signed_deviation": abs(float(timing["mean_signed_deviation"])),
        "mean_steps": float(fmean(float(row["steps"]) for row in rows)),
        "physical_rehandles_per_100_required_deliveries": float(
            100.0 * physical / required
        ),
        "mean_physical_rehandles_per_episode": float(
            fmean(float(row["physical_storage_relocations"]) for row in rows)
        ),
    }
    for component in DECOMPOSITION_COMPONENTS:
        values = [row.get(component) for row in rows]
        metrics[f"{component}_per_100_required_deliveries"] = (
            float(100.0 * sum(int(value) for value in values) / required)
            if all(value is not None for value in values)
            else None
        )
    planning = [row.get("planning_seconds") for row in rows]
    metrics["mean_planning_seconds"] = (
        float(fmean(float(value) for value in planning))
        if all(value is not None for value in planning)
        else None
    )
    return metrics


def _instance_cluster_points(rows: Sequence[Mapping]) -> tuple[dict, ...]:
    points = []
    for seed in PANEL_SEEDS:
        selected = [row for row in rows if int(row["instance_seed"]) == seed]
        if not selected:
            raise MatchedComparisonError(f"instance {seed} has no replication rows")
        points.append(
            {
                "instance_seed": seed,
                "episode_instance_id": selected[0]["episode_instance_id"],
                "replication_count": len(selected),
                **_metrics_for_rows(selected),
            }
        )
    return tuple(points)


def _cluster_stat(points: Sequence[Mapping], field: str) -> Optional[dict]:
    values = [point.get(field) for point in points]
    if any(value is None for value in values):
        return None
    values = [float(value) for value in values]
    if len(values) != len(PANEL_SEEDS):
        raise MatchedComparisonError("cluster statistic requires all 12 instances")
    mean = float(fmean(values))
    standard_deviation = float(stdev(values))
    standard_error = standard_deviation / math.sqrt(len(values))
    return {
        "n_episode_instance_clusters": len(values),
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "nominal_one_sided_95_upper_bound": float(
            mean + DESCRIPTIVE_ONE_SIDED_T_95_DF11 * standard_error
        ),
        "one_sided_t_quantile_df11": DESCRIPTIVE_ONE_SIDED_T_95_DF11,
        "multiplicity_adjusted": False,
        "inferential_claim_authorized": False,
        "statistical_unit": "EpisodeInstance_cluster",
    }


def _mean_cluster_metrics(points: Sequence[Mapping]) -> dict:
    fields = (
        *COMMON_ADDITIVE_METRICS,
        *(f"{field}_per_100_required_deliveries" for field in DECOMPOSITION_COMPONENTS),
        "mean_planning_seconds",
    )
    result = {}
    for field in fields:
        values = [point.get(field) for point in points]
        result[field] = (
            float(fmean(float(value) for value in values))
            if all(value is not None for value in values)
            else None
        )
    result["absolute_mean_signed_deviation"] = abs(
        float(result["mean_signed_deviation"])
    )
    return result


def _delivery_position_metrics(rows: Sequence[Mapping]) -> dict:
    sequences = [tuple(float(value) for value in row["delivery_deviations"]) for row in rows]
    if any(len(values) != EXPECTED_DELIVERIES for values in sequences):
        raise MatchedComparisonError("position metrics require eight deliveries in every row")

    def summarize(values: Iterable[float]) -> dict:
        values = tuple(float(value) for value in values)
        return {"n": len(values), **_timing_metrics(values, window=FROZEN_OBJECTIVE_SPEC.window)}

    return {
        "by_delivery_position": tuple(
            {
                "delivery_position": position + 1,
                **summarize(sequence[position] for sequence in sequences),
            }
            for position in range(EXPECTED_DELIVERIES)
        ),
        "first_two": summarize(
            value for sequence in sequences for value in sequence[:2]
        ),
        "positions_three_plus": summarize(
            value for sequence in sequences for value in sequence[2:]
        ),
        "aggregation_note": (
            "balanced source grid: positions are averaged within fixed replication "
            "design, then represent the same equal-instance/equal-seed estimand"
        ),
    }


def _summarize_method(
    method: str,
    all_rows: Sequence[Mapping],
    identities: Mapping[int, Mapping],
) -> dict:
    rows = tuple(row for row in all_rows if row.get("method_id") == method)
    grid_issues = _grid_issues(method, rows)
    row_issues = []
    for row in rows:
        issues = _row_safety_issues(row, identities)
        if issues:
            row_issues.append(
                {
                    "instance_seed": row.get("instance_seed"),
                    "model_seed": row.get("model_seed"),
                    "policy_rng_index": row.get("policy_rng_index"),
                    "issues": issues,
                }
            )
    eligible = not grid_issues and not row_issues
    summary = {
        "method_id": method,
        "category": (
            "learned_online_primary"
            if method in (V23_METHOD, V11_METHOD)
            else "deterministic_online_primary"
        ),
        "source_row_count": len(rows),
        "expected_source_row_count": (
            48 if method == V23_METHOD else 36 if method == V11_METHOD else 12
        ),
        "instance_count": len({row.get("instance_seed") for row in rows}),
        "grid_issues": grid_issues,
        "row_safety_issues": row_issues,
        "whole_method_numeric_eligible": eligible,
        "complete_case_filtering_used": False,
        "performance_metrics_suppressed_if_any_row_unsafe": True,
        "estimand": (
            "equal_12_instances_after_averaging_4_action_rngs_within_instance"
            if method == V23_METHOD
            else "equal_3_training_seeds_and_equal_12_instances"
            if method == V11_METHOD
            else "equal_12_instances_one_deterministic_run_each"
        ),
        "observed_strict_success_rate": (
            float(fmean(bool(row.get("strict_method_success")) for row in rows))
            if rows
            else None
        ),
        "training_seed_replication": (
            {"n": 1, "variability_estimable": False, "status": "V2.3_single_training_seed"}
            if method == V23_METHOD
            else {"n": 3, "variability_estimable": True}
            if method == V11_METHOD
            else {"n": 0, "variability_estimable": False, "status": "deterministic_method_not_applicable"}
        ),
    }
    if not eligible:
        summary.update(
            {
                "metrics": None,
                "instance_cluster_points": (),
                "instance_cluster_statistics": None,
                "model_seed_summaries": None,
                "delivery_position_metrics": None,
            }
        )
        return summary
    points = _instance_cluster_points(rows)
    metrics = _mean_cluster_metrics(points)
    stats = {
        field: _cluster_stat(points, field)
        for field in (
            "mean_dense_objective_return",
            "mean_signed_deviation",
            "mean_absolute_error",
            "mean_tardiness",
            "mean_earliness",
            "within_target_window_rate",
            "mean_steps",
            "physical_rehandles_per_100_required_deliveries",
        )
    }
    seed_summaries = None
    if method == V11_METHOD:
        seed_summaries = {}
        for model_seed in MODEL_SEEDS:
            seed_rows = [
                row for row in rows if int(row["model_seed"]) == model_seed
            ]
            seed_summaries[str(model_seed)] = {
                "model_seed": model_seed,
                "episode_instance_count": len(PANEL_SEEDS),
                "strict_success_rate": float(
                    fmean(bool(row["strict_method_success"]) for row in seed_rows)
                ),
                "whole_seed_safe_complete": all(
                    not _row_safety_issues(row, identities) for row in seed_rows
                ),
                "delivery_position_metrics": _delivery_position_metrics(seed_rows),
                **_metrics_for_rows(seed_rows),
            }
        for field in COMMON_ADDITIVE_METRICS:
            equal_seed_value = fmean(
                float(seed_summaries[str(seed)][field]) for seed in MODEL_SEEDS
            )
            if not math.isclose(
                float(equal_seed_value), float(metrics[field]), rel_tol=0.0, abs_tol=1e-10
            ):
                raise MatchedComparisonError(
                    f"V1.1 equal-seed/equal-instance aggregation mismatch: {field}"
                )
        variability = {}
        for field in COMMON_ADDITIVE_METRICS:
            values = [
                float(seed_summaries[str(seed)][field]) for seed in MODEL_SEEDS
            ]
            variability[field] = {
                "n_training_seeds": len(values),
                "sample_standard_deviation": float(stdev(values)),
                "minimum": min(values),
                "maximum": max(values),
                "range": max(values) - min(values),
                "inferential_claim_authorized": False,
            }
        summary["training_seed_replication"]["descriptive_variability"] = variability
    summary.update(
        {
            "metrics": metrics,
            "instance_cluster_points": points,
            "instance_cluster_statistics": stats,
            "model_seed_summaries": seed_summaries,
            "delivery_position_metrics": _delivery_position_metrics(rows),
            "descriptive_rehandle_nominal_one_sided_95_ucb": stats[
                "physical_rehandles_per_100_required_deliveries"
            ]["nominal_one_sided_95_upper_bound"],
        }
    )
    return summary


RANKING_SPECS = {
    "dense_return_higher_is_better": ("mean_dense_objective_return", True),
    "absolute_timing_error_lower_is_better": ("mean_absolute_error", False),
    "tardiness_lower_is_better": ("mean_tardiness", False),
    "earliness_lower_is_better": ("mean_earliness", False),
    "absolute_signed_bias_lower_is_better": (
        "absolute_mean_signed_deviation",
        False,
    ),
    "within_target_window_rate_higher_is_better": (
        "within_target_window_rate",
        True,
    ),
    "physical_rehandles_per_100_lower_is_better": (
        "physical_rehandles_per_100_required_deliveries",
        False,
    ),
    "steps_lower_is_better": ("mean_steps", False),
}


def _rank_methods(summaries: Sequence[Mapping]) -> dict:
    eligible = [item for item in summaries if item["whole_method_numeric_eligible"]]
    result = {}
    for label, (field, higher) in RANKING_SPECS.items():
        ordered = sorted(
            eligible,
            key=lambda item: (
                -float(item["metrics"][field])
                if higher
                else float(item["metrics"][field]),
                item["method_id"],
            ),
        )
        ranked = []
        previous = None
        previous_rank = None
        for index, item in enumerate(ordered):
            value = float(item["metrics"][field])
            if previous is not None and math.isclose(
                value, previous, rel_tol=0.0, abs_tol=1e-12
            ):
                rank = previous_rank
            else:
                rank = index + 1
            ranked.append(
                {"rank": rank, "method_id": item["method_id"], "value": value}
            )
            previous, previous_rank = value, rank
        result[label] = ranked
    return result


PARETO_SPECS = {
    "timing_vs_physical_rehandling": {
        "mean_absolute_error": "min",
        "physical_rehandles_per_100_required_deliveries": "min",
    },
    "return_vs_physical_rehandling": {
        "mean_dense_objective_return": "max",
        "physical_rehandles_per_100_required_deliveries": "min",
    },
    "timing_vs_execution_steps": {
        "mean_absolute_error": "min",
        "mean_steps": "min",
    },
    "four_metric_operational": {
        "mean_dense_objective_return": "max",
        "mean_absolute_error": "min",
        "physical_rehandles_per_100_required_deliveries": "min",
        "mean_steps": "min",
    },
}


def _dominates(left: Mapping, right: Mapping, objectives: Mapping[str, str]) -> bool:
    weak = []
    strict = []
    for field, direction in objectives.items():
        a = float(left["metrics"][field])
        b = float(right["metrics"][field])
        if direction == "min":
            weak.append(a <= b)
            strict.append(a < b)
        elif direction == "max":
            weak.append(a >= b)
            strict.append(a > b)
        else:
            raise MatchedComparisonError(f"invalid Pareto direction: {direction}")
    return all(weak) and any(strict)


def _pareto_fronts(summaries: Sequence[Mapping]) -> dict:
    eligible = [item for item in summaries if item["whole_method_numeric_eligible"]]
    result = {}
    for label, objectives in PARETO_SPECS.items():
        front = []
        dominated = {}
        for candidate in eligible:
            dominators = [
                other["method_id"]
                for other in eligible
                if other is not candidate and _dominates(other, candidate, objectives)
            ]
            if dominators:
                dominated[candidate["method_id"]] = sorted(dominators)
            else:
                front.append(candidate["method_id"])
        result[label] = {
            "objectives": dict(objectives),
            "front": sorted(front),
            "dominated_by": dominated,
        }
    return result


def _paired_descriptive_contrasts(summaries: Sequence[Mapping]) -> tuple[dict, ...]:
    by_method = {item["method_id"]: item for item in summaries}
    reference = by_method[V23_METHOD]
    if not reference["whole_method_numeric_eligible"]:
        return ()
    reference_points = {
        int(point["instance_seed"]): point
        for point in reference["instance_cluster_points"]
    }
    fields = (
        "mean_dense_objective_return",
        "mean_absolute_error",
        "mean_tardiness",
        "mean_earliness",
        "within_target_window_rate",
        "physical_rehandles_per_100_required_deliveries",
        "mean_steps",
    )
    output = []
    for comparator in summaries:
        if comparator["method_id"] == V23_METHOD:
            continue
        if not comparator["whole_method_numeric_eligible"]:
            output.append(
                {
                    "reference_method": V23_METHOD,
                    "comparator_method": comparator["method_id"],
                    "available": False,
                    "reason": "comparator_whole_method_safety_ineligible",
                }
            )
            continue
        comparator_points = {
            int(point["instance_seed"]): point
            for point in comparator["instance_cluster_points"]
        }
        _require_equal(
            "paired contrast instance grid",
            set(comparator_points),
            set(reference_points),
        )
        metrics = {}
        for field in fields:
            deltas = tuple(
                float(reference_points[seed][field])
                - float(comparator_points[seed][field])
                for seed in PANEL_SEEDS
            )
            metrics[field] = {
                "orientation": "V2.3_minus_comparator",
                "n_episode_instance_clusters": len(deltas),
                "mean_difference": float(fmean(deltas)),
                "per_instance_differences": deltas,
                "inferential_claim_authorized": False,
            }
        output.append(
            {
                "reference_method": V23_METHOD,
                "comparator_method": comparator["method_id"],
                "available": True,
                "metrics": metrics,
            }
        )
    return tuple(output)


def _deterministic_outcome_equivalence(
    rows: Sequence[Mapping], summaries: Sequence[Mapping]
) -> dict:
    eligible = {
        item["method_id"]
        for item in summaries
        if item["method_id"] in DETERMINISTIC_METHODS
        and item["whole_method_numeric_eligible"]
    }
    by_method = {
        method: {
            int(row["instance_seed"]): row
            for row in rows
            if row.get("method_id") == method
        }
        for method in eligible
    }
    fields = (
        "dense_objective_return",
        "delivery_deviations",
        "steps",
        "physical_storage_relocations",
        "target_bound_obstruction_clearances",
        "standalone_reconfigurations",
    )
    pairs = []
    for left, right in combinations(sorted(eligible), 2):
        differences = []
        for seed in PANEL_SEEDS:
            differing = [
                field
                for field in fields
                if _json_safe(by_method[left][seed].get(field))
                != _json_safe(by_method[right][seed].get(field))
            ]
            if differing:
                differences.append(
                    {"instance_seed": seed, "differing_fields": differing}
                )
        pairs.append(
            {
                "left": left,
                "right": right,
                "exactly_outcome_equivalent_on_all_12_instances": not differences,
                "per_instance_differences": differences,
            }
        )
    return {
        "compared_fields": fields,
        "pairs": pairs,
        "exact_equivalent_pairs": tuple(
            {"left": item["left"], "right": item["right"]}
            for item in pairs
            if item["exactly_outcome_equivalent_on_all_12_instances"]
        ),
        "interpretation": (
            "Outcome-equivalent deterministic variants are retained as named "
            "procedures but must not be counted as independent evidence."
        ),
    }


def _seed_stability_screen(
    summaries: Sequence[Mapping], sources: AuthenticatedSources
) -> dict:
    by_method = {item["method_id"]: item for item in summaries}
    v23 = by_method[V23_METHOD]
    selected = sources.v23_selection["selected_validation"]
    source_authenticated = bool(
        sources.v23_selection.get("selection_recomputed_without_training_summary")
        and sources.v23_selection.get("selected_episode") == V23_SELECTED_EPISODE
        and selected.get("strict_integrity_gate") is True
        and selected.get("development_candidate_eligible") is True
    )
    dominators = []
    if v23["whole_method_numeric_eligible"]:
        v_mae = float(v23["metrics"]["mean_absolute_error"])
        v_rehandle = float(
            v23["metrics"]["physical_rehandles_per_100_required_deliveries"]
        )
        for comparator in summaries:
            if comparator["method_id"] == V23_METHOD:
                continue
            if not comparator["whole_method_numeric_eligible"]:
                continue
            c_mae = float(comparator["metrics"]["mean_absolute_error"])
            c_rehandle = float(
                comparator["metrics"][
                    "physical_rehandles_per_100_required_deliveries"
                ]
            )
            if (
                c_mae <= v_mae
                and c_rehandle <= v_rehandle
                and (c_mae < v_mae or c_rehandle < v_rehandle)
            ):
                dominators.append(
                    {
                        "method_id": comparator["method_id"],
                        "mean_absolute_error": c_mae,
                        "physical_rehandles_per_100_required_deliveries": c_rehandle,
                    }
                )
    passed = bool(
        source_authenticated
        and v23["whole_method_numeric_eligible"]
        and not dominators
    )
    return {
        "role": "development_triage_not_confirmatory_superiority_gate",
        "frozen_before_new_baseline_outcomes": True,
        "passed": passed,
        "authenticated_selected_v2_3_source": source_authenticated,
        "all_48_v2_3_rows_strict_safe_complete": v23[
            "whole_method_numeric_eligible"
        ],
        "dominance_coordinates": (
            "mean_absolute_error",
            "physical_rehandles_per_100_required_deliveries",
        ),
        "safety_eligible_weak_dominators_with_one_strict": tuple(dominators),
        "return_steps_and_timing_direction_reported_not_gated": True,
        "scalarization_used": False,
        "requires_beating_all_baselines": False,
        "training_seed_stability_established": False,
        "independent_v2_3_training_seed_replication_still_required": True,
        "scientific_interpretation": (
            "development method-capacity/Pareto triage on one V2.3 training seed; "
            "the protocol label does not itself establish seed stability"
        ),
        "performance_claim_authorized": False,
    }


def build_report(
    *,
    rows: Sequence[Mapping],
    sources: AuthenticatedSources,
    comparison_contract: Mapping,
) -> dict:
    summaries = tuple(
        _summarize_method(method, rows, sources.identities)
        for method in PRIMARY_METHODS
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scope": "development_only_already_opened_85000_85011",
        "performance_claim_authorized": False,
        "confirmatory_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "complete_case_filtering_used": False,
        "comparison_contract_sha256": comparison_contract["contract_sha256"],
        "reported_return": "undiscounted_dense_operational_return",
        "objective_spec": FROZEN_OBJECTIVE_SPEC.to_dict(),
        "common_metric_definitions": {
            "mean_signed_deviation": (
                "delivery_time_minus_target; direction bias, not accuracy"
            ),
            "mean_absolute_error": "absolute delivery timing error; lower is better",
            "mean_tardiness": "max(0,delivery_time_minus_target); lower is better",
            "mean_earliness": "max(0,target_minus_delivery_time); lower is better",
            "within_target_window_rate": "proportion with absolute error <=20; higher is better",
            "physical_storage_relocations": (
                "all completed storage-to-storage block moves; common physical rehandle burden"
            ),
            "target_bound_obstruction_clearances": (
                "mechanism-specific subset; never substituted for total physical rehandles"
            ),
            "steps": "environment transitions to completion; lower is better",
        },
        "primary_online_methods": PRIMARY_METHODS,
        "method_summaries": summaries,
        "whole_method_safety_exclusions": tuple(
            item["method_id"]
            for item in summaries
            if not item["whole_method_numeric_eligible"]
        ),
        "rankings_are_metric_specific_no_composite_winner": True,
        "metric_rankings": _rank_methods(summaries),
        "pareto_fronts": _pareto_fronts(summaries),
        "paired_descriptive_v2_3_contrasts": _paired_descriptive_contrasts(
            summaries
        ),
        "deterministic_outcome_equivalence": _deterministic_outcome_equivalence(
            rows, summaries
        ),
        "seed_stability_screen": _seed_stability_screen(summaries, sources),
        "v2_3_authoritative_development_selection": sources.v23_selection,
        "controller_specific_diagnostics_not_cross_family_metrics": {
            V23_METHOD: {
                "selected_action_counts": sources.v23_selection[
                    "selected_validation"
                ]["selected_action_counts"],
                "selected_action_shares": sources.v23_selection[
                    "selected_validation"
                ]["selected_action_shares"],
                "hold_outcome_counts": sources.v23_selection[
                    "selected_validation"
                ]["hold_outcome_counts"],
                "cross_controller_ranking_permitted": False,
                "reason": "macro/action alphabets differ across controller families",
            }
        },
        "v2_2_superseded_diagnostic": sources.v22_diagnostic,
        "method_registry": {
            "ranked_primary_online": PRIMARY_METHODS,
            "superseded_diagnostic_unranked": (V22_DIAGNOSTIC_METHOD,),
            "pending_geometry_matched_online": (
                {
                    "method": TRACK_A_REG_SELECTOR_V5,
                    "status": "pending_geometry_matched_5x5_8block_checkpoint",
                    "ranked": False,
                },
                {
                    "method": f"{TRACK_A_KIM2020_A3C_SPATIAL}__stochastic",
                    "status": "pending_geometry_matched_checkpoint_and_policy_rng_protocol",
                    "ranked": False,
                },
                {
                    "method": f"{TRACK_A_KIM2020_A3C_SPATIAL}__greedy_diagnostic",
                    "status": "pending_geometry_matched_checkpoint;policy_mismatch_diagnostic_only",
                    "ranked": False,
                },
            ),
            "offline_information_advantaged_unranked": (
                {
                    "method": TRACK_A_GA_OFFLINE,
                    "information_regime": "offline_full_future_schedule",
                    "executed": False,
                    "ranked": False,
                    "reason": (
                        "future-schedule access differs from the online arrived-only regime"
                    ),
                },
            ),
            "incompatible_or_different_panel_unranked": (
                {
                    "method": "fully_learned_v4_1",
                    "status": "existing_result_is_10x10_40block_standard_track_b",
                    "reason": "geometry_load_and_panel_do_not_match_5x5_8block_85xxx",
                    "ranked": False,
                },
                {
                    "method": TRACK_A_REG_SELECTOR_V5,
                    "status": "existing_training_regime_not_geometry_matched",
                    "reason": "requires_new_5x5_8block_contention_checkpoint",
                    "ranked": False,
                },
                {
                    "method": TRACK_A_KIM2020_A3C_SPATIAL,
                    "status": "existing_training_regime_not_geometry_matched",
                    "reason": "requires_matched_checkpoint_and_policy_rng_protocol",
                    "ranked": False,
                },
                {
                    "method": "legacy_pslap_and_literature_results",
                    "status": "different_instances_geometry_or_information_regime",
                    "reason": "published_or_prior-panel aggregates_cannot_be_pooled",
                    "ranked": False,
                },
            ),
        },
        "statistical_interpretation": {
            "primary_unit": "12 EpisodeInstance clusters",
            "v2_3_nuisance_rngs_averaged_within_instance": 4,
            "v1_1_training_seeds_equal_weighted": MODEL_SEEDS,
            "deterministic_repetitions_per_instance": 1,
            "nominal_cluster_ucbs_are_descriptive": True,
            "v2_3_original_seven_look_budget_ucb_preserved_separately": True,
            "no_final_or_confirmatory_claim": True,
        },
        "run_count": len(rows),
        "runs": tuple(rows),
    }


CSV_FIELDS = (
    "method_id",
    "method_category",
    "information_regime",
    "model_seed",
    "policy_rng_index",
    "policy_rng_seed",
    "instance_seed",
    "instance_index",
    "episode_instance_id",
    "schedule_id",
    "episode_instance_sha256",
    "dense_objective_return",
    "mean_signed_deviation",
    "mean_absolute_error",
    "mean_tardiness",
    "mean_earliness",
    "within_target_window_rate",
    "steps",
    "required_deliveries",
    "delivery_count",
    "physical_storage_relocations",
    "physical_rehandles_per_100_required_deliveries",
    "target_bound_obstruction_clearances",
    "standalone_reconfigurations",
    "standalone_with_direct_delivery_available",
    "standalone_without_direct_delivery_available",
    "directly_deliverable_self_reconfigurations",
    "rehandle_decomposition_status",
    "strict_method_success",
    "completion_rate",
    "all_selected_candidates_exact_safe",
    "illegal_drops",
    "invalid_assignments",
    "fallbacks",
    "witness_mismatches",
    "method_failure_reason",
    "evaluation_policy",
    "evaluation_epsilon",
    "evaluation_learning",
    "assignment_source",
    "planning_seconds",
    "episode_wall_seconds",
    "source_origin",
    "source_execution_reused",
    "source_checkpoint_sha256",
    "source_validation_checkpoint_sha256",
    "source_ledger_sha256",
    "source_manifest_sha256",
    "delivery_deviations",
)


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                record = {field: row.get(field) for field in CSV_FIELDS}
                record["delivery_deviations"] = json.dumps(
                    _json_safe(record["delivery_deviations"]),
                    separators=(",", ":"),
                )
                writer.writerow(record)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _current_source_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {key: _sha256_file(path) for key, path in sorted(paths.items())}


def _baseline_ledger_manifest(output_dir: Path) -> tuple[dict, ...]:
    records = []
    for method in DETERMINISTIC_METHODS:
        for seed in PANEL_SEEDS:
            path = _baseline_ledger_path(output_dir, method, seed)
            if not path.is_file():
                raise MatchedComparisonError(
                    f"completed baseline grid is missing ledger: {path}"
                )
            ledger = _load_json(path)
            records.append(
                {
                    "run_key": ledger["run_key"],
                    "path": str(path.resolve()),
                    "raw_sha256": _sha256_file(path),
                    "ledger_sha256": _verify_self_hash(
                        ledger, "ledger_sha256", name="baseline audit ledger"
                    ),
                    "input_fingerprint": ledger["input_fingerprint"],
                }
            )
    return tuple(records)


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate the exact 85xxx learned ledgers and optionally run "
            "the six matched deterministic online baselines"
        )
    )
    parser.add_argument(
        "--v2-3-source-dir",
        type=Path,
        default=root
        / "results/vcg-constrained-v2-3-gamma1-ablation-seed10-200ep",
    )
    parser.add_argument(
        "--v1-1-control-dir",
        type=Path,
        default=root / "results/vcg-dense-v1-1-v2-2-panel-control-12instance",
    )
    parser.add_argument(
        "--v2-2-source-dir",
        type=Path,
        default=root / "results/vcg-constrained-v2-2-development-seed10-200ep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/vcg-v2-3-matched-baselines-85k",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda", "auto"), default="cpu"
    )
    parser.add_argument(
        "--execute-baselines",
        action="store_true",
        help="execute missing deterministic rows; absent means authenticate/plan only",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="resume only ledgers matching the exact saved comparison contract",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    source_dirs = tuple(
        path.resolve()
        for path in (
            args.v2_3_source_dir,
            args.v1_1_control_dir,
            args.v2_2_source_dir,
        )
    )
    for source_dir in source_dirs:
        if output_dir == source_dir or output_dir.is_relative_to(
            source_dir
        ) or source_dir.is_relative_to(output_dir):
            raise MatchedComparisonError(
                "output directory must be disjoint from every immutable source directory"
            )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume_existing:
        raise FileExistsError(
            f"{output_dir} is nonempty; pass --resume-existing only for this exact contract"
        )

    # Authenticate every reused source before creating or trusting output.
    sources = authenticate_sources(
        v23_source_dir=args.v2_3_source_dir,
        v11_control_dir=args.v1_1_control_dir,
        v22_source_dir=args.v2_2_source_dir,
    )
    device = resolve_device(args.device)
    comparison_contract = build_comparison_contract(sources, device=str(device))
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / CONTRACT_FILENAME
    if contract_path.is_file():
        _require_equal(
            "resumed comparison contract",
            _load_json(contract_path),
            comparison_contract,
        )
    else:
        if (output_dir / "run-ledger").exists():
            raise MatchedComparisonError(
                "cannot resume baseline ledgers without their exact comparison contract"
            )
        _atomic_json(contract_path, comparison_contract)

    v23_rows = _normalize_v23_rows(sources.v23_rows, sources.identities)
    v11_rows = _normalize_v11_rows(sources.v11_rows, sources.identities)
    baseline_rows, missing = execute_or_load_baseline_grid(
        output_dir=output_dir,
        sources=sources,
        comparison_contract=comparison_contract,
        device=device,
        execute=bool(args.execute_baselines),
    )

    # Source immutability is checked after all potentially long baseline work.
    source_after = _current_source_hashes(sources.source_paths)
    _require_equal("reused source artifacts unchanged", source_after, sources.source_sha256)
    baseline_source_after = {
        name: _sha256_file(path)
        for name, path in sorted(_baseline_source_paths().items())
    }
    _require_equal(
        "baseline implementation sources unchanged",
        baseline_source_after,
        comparison_contract["baseline_source_sha256"],
    )

    preflight = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "complete" if not missing else "prepared_missing_baseline_ledgers",
        "execute_baselines_requested": bool(args.execute_baselines),
        "comparison_contract": str(contract_path),
        "comparison_contract_sha256": comparison_contract["contract_sha256"],
        "authenticated_v2_3_source_rows": len(v23_rows),
        "authenticated_v1_1_source_rows": len(v11_rows),
        "loaded_or_executed_deterministic_rows": len(baseline_rows),
        "missing_deterministic_rows": missing,
        "final_86xxx_panel_opened": False,
        "performance_claim_authorized": False,
    }
    _atomic_json(output_dir / PREFLIGHT_FILENAME, preflight)
    if missing:
        print(
            f"Prepared/authenticated contract; {len(missing)} deterministic rows "
            "remain. Re-run with --execute-baselines --resume-existing.",
            flush=True,
        )
        return preflight

    all_rows = tuple((*v23_rows, *v11_rows, *baseline_rows))
    expected_total = 48 + 36 + len(DETERMINISTIC_METHODS) * len(PANEL_SEEDS)
    _require_equal("complete comparison row count", len(all_rows), expected_total)
    report = build_report(
        rows=all_rows,
        sources=sources,
        comparison_contract=comparison_contract,
    )
    report["report_sha256"] = _digest_json(report)
    _write_csv(output_dir / RUNS_FILENAME, all_rows)
    _atomic_json(output_dir / REPORT_FILENAME, report)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "development_only": True,
        "performance_claim_authorized": False,
        "final_86xxx_panel_opened": False,
        "comparison_contract_sha256": comparison_contract["contract_sha256"],
        "source_artifact_sha256_before": sources.source_sha256,
        "source_artifact_sha256_after": source_after,
        "source_artifacts_unchanged": True,
        "baseline_source_sha256_before": comparison_contract[
            "baseline_source_sha256"
        ],
        "baseline_source_sha256_after": baseline_source_after,
        "baseline_sources_unchanged": True,
        "learned_source_rows_reused": {V23_METHOD: 48, V11_METHOD: 36},
        "learned_policy_executions": 0,
        "deterministic_grid_row_count": len(baseline_rows),
        "baseline_ledger_manifest": _baseline_ledger_manifest(output_dir),
        "matched_runs_raw_sha256": _sha256_file(output_dir / RUNS_FILENAME),
        "matched_report_raw_sha256": _sha256_file(output_dir / REPORT_FILENAME),
        "whole_method_safety_exclusions": report[
            "whole_method_safety_exclusions"
        ],
    }
    audit["audit_sha256"] = _digest_json(audit)
    _atomic_json(output_dir / AUDIT_FILENAME, audit)
    print(f"Runs: {output_dir / RUNS_FILENAME}", flush=True)
    print(f"Report: {output_dir / REPORT_FILENAME}", flush=True)
    print(f"Audit: {output_dir / AUDIT_FILENAME}", flush=True)
    return report


if __name__ == "__main__":
    main()
