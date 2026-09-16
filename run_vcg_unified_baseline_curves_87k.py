#!/usr/bin/env python3
"""Evaluate matched baselines on the frozen unified-VCG 87k panel.

The unified lambda=0 and lambda=.05 rows already exist.  This additive runner
loads the same 30 serialized EpisodeInstances and evaluates only the missing
comparators: legacy VCG 1.1, Dynamic PSLAP, the capacity-aware rolling GA, and
the Kim2020-inspired spatial policy.  No policy is trained here.

The implementation deliberately reuses the corrected final86 method adapters;
only the panel identity and output protocol are rebound for the existing 87k
EpisodeInstances.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

import evaluate_vcg_unified_frozen_lambda_confirmation as confirmation
import run_kim2020_final86_comparison as kim_final
import run_vcg_final86_four_method as final86
import vcg_v2_3_kim2020_supplement_evaluation as kim85


PROTOCOL = "vcg_unified_matched_baselines_87k_curves_v1"
KIM_PROTOCOL = PROTOCOL + "_kim2020"
SCHEMA_VERSION = 1

PROJECT_ROOT = Path(__file__).resolve().parent
PANEL_ROOT = PROJECT_ROOT / "results/vcg-unified-frozen-lambda-confirmation-87k"
OUTPUT_ROOT = PROJECT_ROOT / "results/vcg-unified-matched-baselines-87k"
CONFIG_NAME = "curve-evaluation-config.json"

INSTANCE_SEEDS = tuple(range(87_000, 87_030))
V11_ROWS = 3 * 30
DYNAMIC_ROWS = 30
GA_ROWS = 4 * 30
KIM_ROWS = 3 * 5 * 30
EXPECTED_NEW_ROWS = V11_ROWS + DYNAMIC_ROWS + GA_ROWS + KIM_ROWS


class CurveEvaluationError(RuntimeError):
    pass


def _panel_inputs() -> tuple[dict, dict[int, dict]]:
    """Authenticate the existing 87k manifest and adapt its records."""

    path = PANEL_ROOT / "episode-instance-manifest.json"
    manifest = final86._load_json(path, name="unified 87k instance manifest")
    final86._self_hash(
        manifest, "manifest_sha256", name="unified 87k instance manifest"
    )
    records = manifest.get("instances")
    if manifest.get("panel_opened") is not True or not isinstance(records, list):
        raise CurveEvaluationError("unified 87k panel manifest is incomplete")
    if len(records) != len(INSTANCE_SEEDS):
        raise CurveEvaluationError("unified 87k panel must contain 30 instances")

    adapted = []
    by_seed: dict[int, dict] = {}
    for expected_index, (expected_seed, entry) in enumerate(zip(INSTANCE_SEEDS, records)):
        if entry.get("instance_seed") != expected_seed:
            raise CurveEvaluationError("unified 87k panel seed order mismatch")
        instance_path = (PANEL_ROOT / str(entry["relative_path"])).resolve()
        instance = confirmation._load_instance(instance_path, entry)
        if instance.seed != expected_seed:
            raise CurveEvaluationError("serialized 87k EpisodeInstance seed mismatch")
        record = {
            "seed": expected_seed,
            "instance_index": expected_index,
            # final86's loader accepts an absolute path here; the source files
            # remain read-only and are not copied into this additive result.
            "relative_path": str(instance_path),
            "raw_sha256": entry["raw_sha256"],
            "canonical_sha256": entry["canonical_sha256"],
            "episode_instance_id": entry["episode_instance_id"],
            "schedule_id": entry["schedule_id"],
        }
        adapted.append(record)
        by_seed[expected_seed] = record

    adapted_manifest = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "panel_opened": True,
        "instance_count": len(adapted),
        "source_manifest_sha256": manifest["manifest_sha256"],
        # Downstream ledger fingerprints bind directly to the immutable source
        # manifest hash rather than inventing a second EpisodeInstance identity.
        "manifest_sha256": manifest["manifest_sha256"],
        "instances": adapted,
    }
    return adapted_manifest, by_seed


def _kim_inputs() -> tuple[dict, dict[int, dict], dict[int, dict]]:
    manifest, records = _panel_inputs()
    completions: dict[int, dict] = {}
    for model_seed in kim_final.MODEL_SEEDS:
        completion_path = (
            kim_final.KIM85_ROOT
            / "training-completion"
            / f"seed-{model_seed}.json"
        )
        completion = kim85._load_json(completion_path, expected_type=dict)
        kim85._verify_self_hash(
            completion,
            "completion_sha256",
            label=f"Kim seed {model_seed} training completion",
        )
        checkpoint = (
            kim_final.TRAINING_ROOT
            / "training"
            / f"seed-{model_seed}"
            / "best.pth"
        )
        if kim85._sha256(checkpoint) != completion.get(
            "selected_checkpoint_raw_sha256"
        ):
            raise CurveEvaluationError(
                f"Kim seed {model_seed} checkpoint hash mismatch"
            )
        completions[model_seed] = completion
    return manifest, completions, records


def _config(manifest: Mapping[str, Any]) -> dict:
    payload = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_no_new_policy_rows_executed",
        "source_panel_root": str(PANEL_ROOT),
        "source_instance_manifest_sha256": manifest["manifest_sha256"],
        "instance_seeds": list(INSTANCE_SEEDS),
        "existing_unified_rows_reused": 720,
        "new_rows": {
            final86.V11_METHOD: V11_ROWS,
            final86.DYNAMIC_METHOD: DYNAMIC_ROWS,
            final86.GA_METHOD: GA_ROWS,
            kim85.KIM_STOCHASTIC: KIM_ROWS,
        },
        "expected_new_rows": EXPECTED_NEW_ROWS,
        "training_or_learning": False,
        "complete_case_filtering_allowed": False,
        "aggregation_unit": "EpisodeInstance",
        "curve_order": list(INSTANCE_SEEDS),
    }
    payload["config_sha256"] = final86._digest(payload)
    return payload


def prepare(output_root: Path = OUTPUT_ROOT) -> dict:
    manifest, _ = _panel_inputs()
    # These are the only learned-artifact checks needed before execution.
    final86._authenticate_v11()
    _kim_inputs()
    config = _config(manifest)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / CONFIG_NAME
    if path.exists():
        final86._require_equal(
            "87k curve evaluation config",
            final86._load_json(path, name="87k curve evaluation config"),
            config,
        )
    else:
        final86._atomic_json(path, config)
    return config


@contextmanager
def _final86_panel_context():
    """Rebind the already-corrected adapters to the frozen 87k identities."""

    old_protocol = final86.PROTOCOL
    old_seeds = final86.INSTANCE_SEEDS
    old_matched_seeds = final86.matched.PANEL_SEEDS
    try:
        final86.PROTOCOL = PROTOCOL
        final86.INSTANCE_SEEDS = INSTANCE_SEEDS
        # This is the identity-adapter correction already used by the final86
        # correction runner; without it, normalization searches the old 85k grid.
        final86.matched.PANEL_SEEDS = INSTANCE_SEEDS
        yield
    finally:
        final86.matched.PANEL_SEEDS = old_matched_seeds
        final86.INSTANCE_SEEDS = old_seeds
        final86.PROTOCOL = old_protocol


def _base_inputs(output_root: Path) -> tuple[Path, dict, dict, dict]:
    config = prepare(output_root)
    manifest, _ = _panel_inputs()
    contract = {"contract_sha256": config["config_sha256"]}
    return Path(output_root).resolve(), config, manifest, contract


def run_v11(output_root: Path = OUTPUT_ROOT) -> dict:
    root, config, manifest, contract = _base_inputs(output_root)
    arms, records = final86._authenticate_v11()
    with _final86_panel_context():
        rows, missing = final86._v11_rows(
            root=root,
            contract=contract,
            manifest=manifest,
            auth={"v11_arms": arms},
            execute=True,
        )
    return {
        "method": final86.V11_METHOD,
        "config_sha256": config["config_sha256"],
        "selected_checkpoints": list(records),
        "rows": len(rows),
        "strict_safe_complete_rows": sum(
            row["strict_safe_complete"] is True for row in rows
        ),
        "missing": len(missing),
    }


def run_baselines(output_root: Path = OUTPUT_ROOT) -> dict:
    root, config, manifest, contract = _base_inputs(output_root)
    with _final86_panel_context():
        dynamic, missing_dynamic = final86._dynamic_rows(
            root=root,
            contract=contract,
            manifest=manifest,
            auth={},
            execute=True,
        )
        ga, missing_ga = final86._ga_rows(
            root=root,
            contract=contract,
            manifest=manifest,
            auth={},
            execute=True,
        )
    return {
        "config_sha256": config["config_sha256"],
        final86.DYNAMIC_METHOD: {
            "rows": len(dynamic),
            "strict_safe_complete_rows": sum(
                row["strict_safe_complete"] is True for row in dynamic
            ),
            "missing": len(missing_dynamic),
        },
        final86.GA_METHOD: {
            "rows": len(ga),
            "strict_safe_complete_rows": sum(
                row["strict_safe_complete"] is True for row in ga
            ),
            "missing": len(missing_ga),
        },
    }


def _kim_config(manifest: Mapping, completions: Mapping[int, Mapping]) -> dict:
    payload = {
        "protocol": KIM_PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_no_kim_87k_rows_executed",
        "scientific_role": "matched_87k_frozen_policy_baseline",
        "model_seeds": list(kim_final.MODEL_SEEDS),
        "instance_seeds": list(INSTANCE_SEEDS),
        "rollouts_per_model_instance": len(kim_final.ROLLOUTS),
        "expected_rows": KIM_ROWS,
        "policy_rng_formula": (
            "633000000 + 1000*model_seed + 10*panel_index + rollout"
        ),
        "source_instance_manifest_sha256": manifest["manifest_sha256"],
        "checkpoints": {
            str(seed): {
                "selected_episode": completions[seed]["selected_episode"],
                "raw_sha256": completions[seed]["selected_checkpoint_raw_sha256"],
                "deployment_sha256": completions[seed][
                    "selected_checkpoint_deployment_sha256"
                ],
            }
            for seed in kim_final.MODEL_SEEDS
        },
        "aggregation": (
            "five stochastic rolls within model-instance, equal three models "
            "within instance, then EpisodeInstances"
        ),
        "complete_case_filtering_allowed": False,
        "kim_retraining": False,
    }
    payload["config_sha256"] = final86._digest(payload)
    return payload


@contextmanager
def _kim87_context():
    old = {
        "PROTOCOL": kim_final.PROTOCOL,
        "PANEL_SEEDS": kim_final.PANEL_SEEDS,
        "FINAL86_ROOT": kim_final.FINAL86_ROOT,
        "OUTPUT_ROOT": kim_final.OUTPUT_ROOT,
        "EXPECTED_ROWS": kim_final.EXPECTED_ROWS,
        "inputs": kim_final._inputs,
        "config": kim_final._config,
    }
    try:
        kim_final.PROTOCOL = KIM_PROTOCOL
        kim_final.PANEL_SEEDS = INSTANCE_SEEDS
        kim_final.FINAL86_ROOT = PANEL_ROOT
        kim_final.OUTPUT_ROOT = OUTPUT_ROOT
        kim_final.EXPECTED_ROWS = KIM_ROWS
        kim_final._inputs = _kim_inputs
        kim_final._config = _kim_config
        yield
    finally:
        kim_final._config = old["config"]
        kim_final._inputs = old["inputs"]
        kim_final.EXPECTED_ROWS = old["EXPECTED_ROWS"]
        kim_final.OUTPUT_ROOT = old["OUTPUT_ROOT"]
        kim_final.FINAL86_ROOT = old["FINAL86_ROOT"]
        kim_final.PANEL_SEEDS = old["PANEL_SEEDS"]
        kim_final.PROTOCOL = old["PROTOCOL"]


def run_kim(output_root: Path = OUTPUT_ROOT) -> dict:
    prepare(output_root)
    with _kim87_context():
        return kim_final.run(Path(output_root).resolve())


def inspect(output_root: Path = OUTPUT_ROOT) -> dict:
    """Report exact on-disk row counts without executing a policy."""

    root = Path(output_root).resolve()
    expected = {
        final86.V11_METHOD: V11_ROWS,
        final86.DYNAMIC_METHOD: DYNAMIC_ROWS,
        final86.GA_METHOD: GA_ROWS,
        kim85.KIM_STOCHASTIC: KIM_ROWS,
    }
    patterns = {
        final86.V11_METHOD: root / "run-ledger" / final86.V11_METHOD,
        final86.DYNAMIC_METHOD: root / "run-ledger" / final86.DYNAMIC_METHOD,
        final86.GA_METHOD: root / "run-ledger" / final86.GA_METHOD,
        kim85.KIM_STOCHASTIC: root / "run-ledger",
    }
    counts = {}
    for method, directory in patterns.items():
        if method == kim85.KIM_STOCHASTIC:
            observed = sum(
                1
                for seed in kim_final.MODEL_SEEDS
                for _ in (directory / f"seed-{seed}").glob("instance-*-roll-*.json")
            )
        else:
            observed = len(list(directory.glob("*.json"))) if directory.is_dir() else 0
        counts[method] = {
            "expected": expected[method],
            "observed": observed,
            "complete": observed == expected[method],
        }
    return {
        "protocol": PROTOCOL,
        "output_root": str(root),
        "methods": counts,
        "all_complete": all(value["complete"] for value in counts.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "run-v11", "run-baselines", "run-kim", "inspect"),
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output_root)
    elif args.command == "run-v11":
        result = run_v11(args.output_root)
    elif args.command == "run-baselines":
        result = run_baselines(args.output_root)
    elif args.command == "run-kim":
        result = run_kim(args.output_root)
    else:
        result = inspect(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
