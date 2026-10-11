#!/usr/bin/env python3
"""Evaluate all frozen comparators on the nested-VCG 88k panel."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

import final86_v23_runtime as v23_adapter
import run_kim2020_final86_comparison as kim_final
import run_vcg_final86_four_method as final86
import vcg_v2_3_kim2020_supplement_evaluation as kim85


PROTOCOL = "vcg_v11_nested_all_matched_baselines_88k_v1"
KIM_PROTOCOL = PROTOCOL + "_kim2020"
SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parent
PANEL_ROOT = PROJECT_ROOT / "results/vcg-v1-1-nested-handling-confirmation-88k"
OUTPUT_ROOT = PROJECT_ROOT / "results/vcg-v1-1-nested-handling-matched-baselines-88k"
CONFIG_NAME = "matched-evaluation-config.json"

INSTANCE_SEEDS = tuple(range(88_000, 88_030))
V23_ROWS = 3 * 4 * 30
DYNAMIC_ROWS = 30
GA_ROWS = 4 * 30
KIM_ROWS = 3 * 5 * 30
EXPECTED_NEW_ROWS = V23_ROWS + DYNAMIC_ROWS + GA_ROWS + KIM_ROWS


class Matched88Error(RuntimeError):
    pass


def _panel_inputs() -> tuple[dict, dict[int, dict]]:
    path = PANEL_ROOT / "episode-instance-manifest.json"
    manifest = final86._load_json(path, name="nested VCG 88k instance manifest")
    final86._self_hash(
        manifest, "manifest_sha256", name="nested VCG 88k instance manifest"
    )
    records = manifest.get("instances")
    if (
        manifest.get("panel_opened") is not True
        or not isinstance(records, list)
        or len(records) != len(INSTANCE_SEEDS)
    ):
        raise Matched88Error("nested VCG 88k panel manifest is incomplete")
    env = final86._new_environment()
    adapted = []
    by_seed = {}
    for index, (expected_seed, entry) in enumerate(zip(INSTANCE_SEEDS, records)):
        if entry.get("instance_seed") != expected_seed:
            raise Matched88Error("88k panel seed order changed")
        instance_path = (PANEL_ROOT / str(entry.get("relative_path"))).absolute()
        raw = final86._read_bytes(instance_path, name="88k EpisodeInstance")
        final86._require_equal(
            "88k EpisodeInstance raw SHA",
            final86._sha256_bytes(raw),
            entry.get("raw_sha256"),
        )
        instance = final86.EpisodeInstance.from_json(raw.decode("utf-8"))
        instance.validate_for(env)
        final86._require_equal("88k instance seed", instance.seed, expected_seed)
        final86._require_equal(
            "88k instance ID", instance.instance_id, entry.get("episode_instance_id")
        )
        final86._require_equal(
            "88k schedule ID", instance.schedule_id, entry.get("schedule_id")
        )
        final86._require_equal(
            "88k canonical SHA",
            final86._sha256_bytes(instance.to_json().encode("utf-8")),
            entry.get("canonical_sha256"),
        )
        record = {
            "seed": expected_seed,
            "instance_index": index,
            "relative_path": str(instance_path),
            "raw_sha256": entry["raw_sha256"],
            "canonical_sha256": entry["canonical_sha256"],
            "episode_instance_id": entry["episode_instance_id"],
            "schedule_id": entry["schedule_id"],
        }
        adapted.append(record)
        by_seed[expected_seed] = record
    return (
        {
            "protocol": PROTOCOL,
            "schema_version": SCHEMA_VERSION,
            "panel_opened": True,
            "instance_count": len(adapted),
            "source_manifest_sha256": manifest["manifest_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "instances": adapted,
        },
        by_seed,
    )


def _kim_inputs() -> tuple[dict, dict[int, dict], dict[int, dict]]:
    manifest, records = _panel_inputs()
    completions = {}
    for model_seed in kim_final.MODEL_SEEDS:
        path = (
            kim_final.KIM85_ROOT
            / "training-completion"
            / f"seed-{model_seed}.json"
        )
        completion = kim85._load_json(path, expected_type=dict)
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
            raise Matched88Error(f"Kim seed-{model_seed} checkpoint changed")
        completions[model_seed] = completion
    return manifest, completions, records


def _config(manifest: Mapping[str, Any]) -> dict:
    payload = {
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_on_already_opened_88k_panel_no_rows_executed",
        "scientific_role": "post_confirmation_matched_secondary_comparison",
        "source_panel_root": str(PANEL_ROOT),
        "source_instance_manifest_sha256": manifest["manifest_sha256"],
        "instance_seeds": list(INSTANCE_SEEDS),
        "existing_nested_vcg_rows_reused": 180,
        "new_rows": {
            final86.V23_METHOD: V23_ROWS,
            final86.DYNAMIC_METHOD: DYNAMIC_ROWS,
            final86.GA_METHOD: GA_ROWS,
            kim85.KIM_STOCHASTIC: KIM_ROWS,
        },
        "expected_new_rows": EXPECTED_NEW_ROWS,
        "vcg_1_1_not_rerun_because_lambda0_is_exactly_vcg_1_1": True,
        "training_or_learning": False,
        "complete_case_filtering_allowed": False,
        "aggregation_unit": "EpisodeInstance",
        "curve_order": list(INSTANCE_SEEDS),
        "comparison_was_added_after_nested_confirmation_outcomes": True,
    }
    payload["config_sha256"] = final86._digest(payload)
    return payload


def prepare(output_root: Path = OUTPUT_ROOT) -> dict:
    manifest, _ = _panel_inputs()
    final86._authenticate_all()
    _kim_inputs()
    config = _config(manifest)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / CONFIG_NAME
    if path.exists():
        final86._require_equal(
            "88k matched evaluation config",
            final86._load_json(path, name="88k matched evaluation config"),
            config,
        )
    else:
        final86._atomic_json(path, config)
    return config


@contextmanager
def _final88_context():
    old = {
        "protocol": final86.PROTOCOL,
        "seeds": final86.INSTANCE_SEEDS,
        "matched": final86.matched.PANEL_SEEDS,
        "adapter": v23_adapter.FINAL_INSTANCE_SEEDS,
    }
    try:
        final86.PROTOCOL = PROTOCOL
        final86.INSTANCE_SEEDS = INSTANCE_SEEDS
        final86.matched.PANEL_SEEDS = INSTANCE_SEEDS
        v23_adapter.FINAL_INSTANCE_SEEDS = INSTANCE_SEEDS
        yield
    finally:
        v23_adapter.FINAL_INSTANCE_SEEDS = old["adapter"]
        final86.matched.PANEL_SEEDS = old["matched"]
        final86.INSTANCE_SEEDS = old["seeds"]
        final86.PROTOCOL = old["protocol"]


def _base_inputs(output_root: Path) -> tuple[Path, dict, dict, dict]:
    config = prepare(output_root)
    manifest, _ = _panel_inputs()
    return (
        Path(output_root).resolve(),
        config,
        manifest,
        {"contract_sha256": config["config_sha256"]},
    )


def run_v23(output_root: Path = OUTPUT_ROOT) -> dict:
    root, config, manifest, contract = _base_inputs(output_root)
    auth = final86._authenticate_all()
    with _final88_context():
        rows, missing = final86._v23_rows(
            root=root,
            contract=contract,
            manifest=manifest,
            auth=auth,
            execute=True,
        )
    return {
        "method": final86.V23_METHOD,
        "config_sha256": config["config_sha256"],
        "rows": len(rows),
        "strict_safe_complete_rows": sum(
            row.get("strict_safe_complete") is True for row in rows
        ),
        "missing": len(missing),
    }


def run_baselines(output_root: Path = OUTPUT_ROOT) -> dict:
    root, config, manifest, contract = _base_inputs(output_root)
    with _final88_context():
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
                row.get("strict_safe_complete") is True for row in dynamic
            ),
            "missing": len(missing_dynamic),
        },
        final86.GA_METHOD: {
            "rows": len(ga),
            "strict_safe_complete_rows": sum(
                row.get("strict_safe_complete") is True for row in ga
            ),
            "missing": len(missing_ga),
        },
    }


def _kim_config(manifest: Mapping, completions: Mapping[int, Mapping]) -> dict:
    payload = {
        "protocol": KIM_PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_no_kim_88k_rows_executed",
        "scientific_role": "post_confirmation_matched_secondary_baseline",
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
def _kim88_context(output_root: Path):
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
        kim_final.OUTPUT_ROOT = Path(output_root).resolve()
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
    with _kim88_context(output_root):
        return kim_final.run(Path(output_root).resolve())


def inspect(output_root: Path = OUTPUT_ROOT) -> dict:
    root = Path(output_root).resolve()
    expected = {
        final86.V23_METHOD: V23_ROWS,
        final86.DYNAMIC_METHOD: DYNAMIC_ROWS,
        final86.GA_METHOD: GA_ROWS,
        kim85.KIM_STOCHASTIC: KIM_ROWS,
    }
    directories = {
        final86.V23_METHOD: root / "run-ledger" / final86.V23_METHOD,
        final86.DYNAMIC_METHOD: root / "run-ledger" / final86.DYNAMIC_METHOD,
        final86.GA_METHOD: root / "run-ledger" / final86.GA_METHOD,
    }
    counts = {}
    for method, directory in directories.items():
        observed = len(list(directory.glob("*.json"))) if directory.is_dir() else 0
        counts[method] = {
            "expected": expected[method],
            "observed": observed,
            "complete": observed == expected[method],
        }
    kim_count = sum(
        len(list((root / "run-ledger" / f"seed-{seed}").glob("instance-*-roll-*.json")))
        for seed in kim_final.MODEL_SEEDS
    )
    counts[kim85.KIM_STOCHASTIC] = {
        "expected": KIM_ROWS,
        "observed": kim_count,
        "complete": kim_count == KIM_ROWS,
    }
    return {
        "protocol": PROTOCOL,
        "output_root": str(root),
        "methods": counts,
        "all_complete": all(value["complete"] for value in counts.values()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "run-v23", "run-baselines", "run-kim", "inspect"),
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.output_root)
    elif args.command == "run-v23":
        result = run_v23(args.output_root)
    elif args.command == "run-baselines":
        result = run_baselines(args.output_root)
    elif args.command == "run-kim":
        result = run_kim(args.output_root)
    else:
        result = inspect(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
