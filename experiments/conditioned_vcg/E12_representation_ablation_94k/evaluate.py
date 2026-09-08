#!/usr/bin/env python3
"""Frozen matched evaluation and shift-interaction analysis for E12."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
import sys
from typing import Mapping, Optional, Sequence
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import benchmark_viability_critic_priority as benchmark
from example.episode_instance import EpisodeInstance
from experiments.conditioned_vcg.E11_distribution_shift_93k import run as e11
from experiments.conditioned_vcg.E12_representation_ablation_94k import program
from experiments.conditioned_vcg.E12_representation_ablation_94k import train_handling
from experiments.conditioned_vcg.E12_representation_ablation_94k import train_operational
from methods.conditioned_vcg.representation_ablation import (
    FULL_RELATIONAL_SUCCESSOR,
    REPRESENTATION_VARIANTS,
    RepresentationConditionedHandlingAgent,
    agent_class_for_variant,
)
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_nested_handling_pilot as pilot


PILOT_PANEL = "development_94k"
CONFIRMATION_PANEL = "e11_93k"
PANELS = (PILOT_PANEL, CONFIRMATION_PANEL)


class FixedPreferenceAgent:
    def __init__(self, agent, value: float) -> None:
        self.agent = agent
        self.value = float(value)
        # The executor needs operational gamma, not handling-head metadata.
        self.config = agent.base_agent.config

    def reset_episode_state(self):
        return self.agent.reset_episode_state()

    def select(self, snapshot, *, training=False, epsilon=0.0):
        return self.agent.select(
            snapshot,
            preference_lambda=self.value,
            training=training,
            epsilon=epsilon,
        )

    def observe_outcome(self, decision, *, next_snapshot, done):
        return self.agent.observe_outcome(
            decision, next_snapshot=next_snapshot, done=done
        )


def _checkpoint_factory(
    output_dir: Path,
    *,
    variant: str,
    model_seed: int,
    value: float,
    device: torch.device,
):
    operational_dir = train_operational.output_path(
        output_dir, variant, model_seed
    )
    handling_dir = train_handling.output_path(output_dir, variant, model_seed)
    op_summary = program.load_json(
        operational_dir / "training-summary.json",
        label="E12 operational summary",
    )
    handling_summary = program.load_json(
        handling_dir / "training-summary.json",
        label="E12 handling summary",
    )
    op_path = operational_dir / "best.pth"
    handling_path = handling_dir / "terminal.pth"
    op_sha = program.sha256(op_path)
    handling_sha = program.sha256(handling_path)
    if (
        op_summary.get("status") != "complete"
        or op_summary.get("best_checkpoint_sha256") != op_sha
        or handling_summary.get("status") != "complete"
        or handling_summary.get("terminal_checkpoint_sha256") != handling_sha
    ):
        raise program.E12Error(
            f"frozen training artifacts are incomplete for {variant}/seed-{model_seed}"
        )
    op_payload = torch.load(op_path, map_location="cpu", weights_only=False)
    handling_payload = torch.load(
        handling_path, map_location="cpu", weights_only=False
    )
    policy_digest = handling_summary["base_policy_digest"]

    def factory():
        base = agent_class_for_variant(variant).from_checkpoint(
            op_payload, device=device, resumable=False, seed=model_seed
        )
        for network in (base.Q_local, base.Q_target):
            network.requires_grad_(False).eval()
        base.set_epsilon(0.0)
        agent = RepresentationConditionedHandlingAgent.from_checkpoint(
            handling_payload["agent_checkpoint"],
            base_agent=base,
            expected_base_checkpoint_sha256=op_sha,
            expected_base_policy_digest=policy_digest,
            seed=model_seed,
        )
        return FixedPreferenceAgent(agent, value)

    return op_payload, factory, {
        "operational_checkpoint_sha256": op_sha,
        "handling_checkpoint_sha256": handling_sha,
    }


@contextmanager
def _runtime(factory, regime):
    with patch.object(
        benchmark, "_freeze_agent", lambda _payload, **_kwargs: factory()
    ), patch.object(benchmark, "_make_env", lambda _payload: regime.make_env()):
        yield


def _strict(raw: Mapping) -> bool:
    return bool(
        raw.get("strict_method_success")
        and raw.get("terminal")
        and raw.get("method_failure_reason") is None
        and raw.get("complete_frontier_exactly_verified")
        and int(raw.get("illegal_drops", 0)) == 0
        and int(raw.get("macro_failures", 0)) == 0
        and len(raw.get("delivery_deviations", ())) == e11.EXPECTED_BLOCKS
    )


def _row(
    raw: Mapping,
    instance: EpisodeInstance,
    record: Mapping,
    spec: Mapping,
    panel: str,
) -> dict:
    strict = _strict(raw)
    timing = (
        final90._timing(raw["delivery_deviations"])
        if strict
        else {
            "mean_signed_deviation": None,
            "mean_absolute_error": None,
            "mean_tardiness": None,
            "mean_earliness": None,
            "within_target_window_rate": None,
        }
    )
    rehandles = int(raw.get("physical_storage_relocations", raw.get("relocations", 0)))
    return {
        "schema_version": program.SCHEMA_VERSION,
        "protocol": program.PROTOCOL,
        "panel": panel,
        "regime_id": record["regime_id"],
        "instance_seed": int(record["seed"]),
        "instance_index": int(record["instance_index"]),
        "episode_instance_id": instance.instance_id,
        "schedule_id": instance.schedule_id,
        **dict(spec),
        "strict_safe_complete": strict,
        "method_failure_reason": raw.get("method_failure_reason"),
        "all_selected_candidates_exact_safe": bool(
            raw.get("complete_frontier_exactly_verified", False)
        ),
        "dense_return": float(raw["return"]) if strict else None,
        **timing,
        "steps": int(raw["steps"]) if strict else None,
        "physical_storage_relocations": rehandles if strict else None,
        "physical_rehandles_per_100_required_deliveries": (
            100.0 * rehandles / e11.EXPECTED_BLOCKS if strict else None
        ),
        "observed_steps_to_stop": int(raw.get("steps", 0)),
        "observed_rehandles_to_stop": rehandles,
        "behavior_digest": raw.get("behavior_digest"),
        "macro_decisions": int(raw.get("macro_decisions", 0)),
        "evaluation_learning": False,
        "exact_verifier_authoritative": bool(
            raw.get("exact_verifier_authoritative", False)
        ),
    }


def _failed_row(error: Exception, record: Mapping, spec: Mapping, panel: str) -> dict:
    return {
        "schema_version": program.SCHEMA_VERSION,
        "protocol": program.PROTOCOL,
        "panel": panel,
        "regime_id": record["regime_id"],
        "instance_seed": int(record["seed"]),
        "instance_index": int(record["instance_index"]),
        "episode_instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        **dict(spec),
        "strict_safe_complete": False,
        "method_failure_reason": f"{type(error).__name__}: {error}",
        "all_selected_candidates_exact_safe": None,
        "dense_return": None,
        "mean_signed_deviation": None,
        "mean_absolute_error": None,
        "mean_tardiness": None,
        "mean_earliness": None,
        "within_target_window_rate": None,
        "steps": None,
        "physical_storage_relocations": None,
        "physical_rehandles_per_100_required_deliveries": None,
        "observed_steps_to_stop": None,
        "observed_rehandles_to_stop": None,
        "behavior_digest": None,
        "macro_decisions": None,
        "evaluation_learning": False,
        "exact_verifier_authoritative": None,
    }


def _records(output_dir: Path, panel: str):
    if panel == PILOT_PANEL:
        _contract, manifest = program.authenticate(output_dir)
        for record in manifest["records"]:
            path = output_dir / record["relative_path"]
            if program.sha256(path) != record["raw_sha256"]:
                raise program.E12Error("development instance bytes changed")
            instance = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
            yield record, instance
        return
    _e11_contract, manifest = e11.authenticate(PROJECT_ROOT, e11.DEFAULT_OUTPUT)
    for record in manifest["records"]:
        instance = e11._load_instance(e11.DEFAULT_OUTPUT, record)
        yield record, instance


def _ledger_path(output_dir: Path, panel: str, spec: Mapping, record: Mapping) -> Path:
    value = str(float(spec["preference_lambda"])).replace(".", "p")
    return (
        output_dir
        / "evaluation"
        / panel
        / str(record["regime_id"])
        / str(spec["representation_variant"])
        / f"seed-{spec['model_seed']}"
        / f"lambda-{value}"
        / f"instance-{record['seed']}.json"
    )


def evaluate(
    output_dir: Path,
    *,
    panel: str,
    model_seeds: Sequence[int],
    variants: Sequence[str],
    regimes: Optional[Sequence[str]],
    device_name: str,
) -> dict:
    contract, manifest = program.authenticate(output_dir)
    if panel not in PANELS:
        raise ValueError(f"unknown panel: {panel}")
    if panel == PILOT_PANEL and tuple(model_seeds) != program.PILOT_MODEL_SEEDS:
        raise program.E12Error("the development pilot is frozen to model seed 0")
    if any(seed not in program.MODEL_SEEDS for seed in model_seeds):
        raise program.E12Error("model seeds must lie in {0,1,2}")
    if any(variant not in REPRESENTATION_VARIANTS for variant in variants):
        raise program.E12Error("unknown representation variant")
    device = pilot._device(device_name)
    selected_records = [
        item
        for item in _records(output_dir, panel)
        if regimes is None or item[0]["regime_id"] in regimes
    ]
    specs = tuple(
        {
            "representation_variant": variant,
            "model_seed": int(model_seed),
            "preference_lambda": float(value),
        }
        for variant in variants
        for model_seed in model_seeds
        for value in program.DEPLOYMENT_LAMBDAS
    )
    checkpoint_cache = {}
    complete = safe = 0
    for spec in specs:
        key = (
            spec["representation_variant"],
            spec["model_seed"],
            spec["preference_lambda"],
        )
        checkpoint_cache[key] = _checkpoint_factory(
            output_dir,
            variant=key[0],
            model_seed=key[1],
            value=key[2],
            device=device,
        )
    for record, instance in selected_records:
        regime = e11.REGIME_BY_ID[record["regime_id"]]
        for spec in specs:
            path = _ledger_path(output_dir, panel, spec, record)
            if path.is_file():
                ledger = program.load_json(path, label="E12 evaluation ledger")
                if (
                    ledger.get("ledger_sha256")
                    != program.digest(ledger, hash_field="ledger_sha256")
                    or ledger.get("contract_sha256") != contract["contract_sha256"]
                    or ledger.get("spec") != spec
                    or ledger.get("episode_instance_id") != instance.instance_id
                ):
                    raise program.E12Error("E12 evaluation ledger binding changed")
                row = ledger["row"]
            else:
                op_payload, factory, checkpoint_hashes = checkpoint_cache[
                    (
                        spec["representation_variant"],
                        spec["model_seed"],
                        spec["preference_lambda"],
                    )
                ]
                try:
                    with _runtime(factory, regime):
                        raw = benchmark.run_arm(
                            arm=benchmark.EXACT_FULL,
                            controller_payload=op_payload,
                            instance=instance,
                            instance_seed=int(instance.seed),
                            search_config=benchmark._search_config(op_payload),
                            liveness_rule=benchmark._liveness_rule(op_payload),
                            prioritizer=None,
                            max_steps=program.MAX_STEPS,
                            device=device,
                        )
                    row = _row(raw, instance, record, spec, panel)
                except Exception as error:
                    row = _failed_row(error, record, spec, panel)
                ledger = program.with_hash(
                    {
                        "schema_version": program.SCHEMA_VERSION,
                        "protocol": program.PROTOCOL,
                        "contract_sha256": contract["contract_sha256"],
                        "development_manifest_sha256": manifest["manifest_sha256"],
                        "panel": panel,
                        "spec": spec,
                        "episode_instance_id": instance.instance_id,
                        "checkpoints": checkpoint_hashes,
                        "row": row,
                    },
                    "ledger_sha256",
                )
                program.atomic_json(path, ledger)
            complete += 1
            safe += int(row["strict_safe_complete"])
            print(
                f"E12 {panel} {complete}/{len(selected_records) * len(specs)} | "
                f"{record['regime_id']} {record['seed']} | "
                f"{spec['representation_variant']} seed={spec['model_seed']} "
                f"lambda={spec['preference_lambda']:.2f} | "
                f"strict={int(row['strict_safe_complete'])}",
                flush=True,
            )
    return {
        "status": "complete",
        "panel": panel,
        "rows": complete,
        "strict_safe_complete_rows": safe,
    }


def _all_rows(output_dir: Path, panel: str) -> list[dict]:
    root = output_dir / "evaluation" / panel
    rows = []
    for path in sorted(root.glob("**/instance-*.json")):
        ledger = program.load_json(path, label="E12 evaluation ledger")
        if ledger.get("ledger_sha256") != program.digest(
            ledger, hash_field="ledger_sha256"
        ):
            raise program.E12Error("E12 ledger self-hash mismatch")
        rows.append(ledger["row"])
    return rows


def _mean_sd(values) -> dict:
    values = [float(value) for value in values]
    return {
        "n": len(values),
        "mean": fmean(values) if values else None,
        "sd": pstdev(values) if len(values) > 1 else (0.0 if values else None),
    }


def analyze(output_dir: Path, *, panel: str, allow_partial: bool) -> dict:
    contract, _manifest = program.authenticate(output_dir)
    rows = _all_rows(output_dir, panel)
    seeds = program.PILOT_MODEL_SEEDS if panel == PILOT_PANEL else program.MODEL_SEEDS
    regimes = (
        program.DEVELOPMENT_REGIMES
        if panel == PILOT_PANEL
        else tuple(regime.regime_id for regime in e11.REGIMES)
    )
    instances = (
        program.DEVELOPMENT_INSTANCE_SEEDS
        if panel == PILOT_PANEL
        else e11.INSTANCE_SEEDS
    )
    expected = (
        len(REPRESENTATION_VARIANTS)
        * len(seeds)
        * len(program.DEPLOYMENT_LAMBDAS)
        * len(regimes)
        * len(instances)
    )
    if not allow_partial and len(rows) != expected:
        raise program.E12Error(f"expected {expected} rows, found {len(rows)}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["representation_variant"],
                int(row["model_seed"]),
                float(row["preference_lambda"]),
                row["regime_id"],
            )
        ].append(row)
    metrics = (
        "mean_absolute_error",
        "physical_rehandles_per_100_required_deliveries",
        "steps",
        "within_target_window_rate",
        "dense_return",
    )
    aggregates = []
    for key, group in sorted(grouped.items()):
        strict = [row for row in group if row["strict_safe_complete"]]
        complete_group = len(strict) == len(group)
        aggregates.append(
            {
                "representation_variant": key[0],
                "model_seed": key[1],
                "preference_lambda": key[2],
                "regime_id": key[3],
                "rows": len(group),
                "strict_safe_complete": len(strict),
                "strict_completion_rate": len(strict) / len(group),
                "metrics_suppressed_for_incomplete_group": not complete_group,
                **{
                    metric: (
                        _mean_sd(row[metric] for row in strict)
                        if complete_group
                        else {"n": 0, "mean": None, "sd": None}
                    )
                    for metric in metrics
                },
            }
        )

    index = {
        (
            row["representation_variant"],
            int(row["model_seed"]),
            float(row["preference_lambda"]),
            row["regime_id"],
            int(row["instance_seed"]),
        ): row
        for row in rows
    }
    interactions = []
    lower_metrics = (
        "mean_absolute_error",
        "physical_rehandles_per_100_required_deliveries",
        "steps",
    )
    for ablation in REPRESENTATION_VARIANTS[1:]:
        for model_seed in seeds:
            for value in program.DEPLOYMENT_LAMBDAS:
                for regime in regimes:
                    if regime == "reference":
                        continue
                    for metric in lower_metrics:
                        paired = []
                        for instance_seed in instances:
                            keys = {
                                "ar": (ablation, model_seed, value, "reference", instance_seed),
                                "as": (ablation, model_seed, value, regime, instance_seed),
                                "fr": (FULL_RELATIONAL_SUCCESSOR, model_seed, value, "reference", instance_seed),
                                "fs": (FULL_RELATIONAL_SUCCESSOR, model_seed, value, regime, instance_seed),
                            }
                            selected = {name: index.get(key) for name, key in keys.items()}
                            if all(
                                row is not None
                                and row["strict_safe_complete"]
                                and row[metric] is not None
                                for row in selected.values()
                            ):
                                paired.append(
                                    (selected["as"][metric] - selected["ar"][metric])
                                    - (selected["fs"][metric] - selected["fr"][metric])
                                )
                        interactions.append(
                            {
                                "ablation": ablation,
                                "model_seed": model_seed,
                                "preference_lambda": value,
                                "regime_id": regime,
                                "metric": metric,
                                "positive_means_ablation_degraded_more": True,
                                **_mean_sd(paired),
                            }
                        )
    report = program.with_hash(
        {
            "schema_version": program.SCHEMA_VERSION,
            "protocol": program.PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "panel": panel,
            "partial": len(rows) != expected,
            "expected_rows": expected,
            "observed_rows": len(rows),
            "strict_safe_complete_rows": sum(
                int(row["strict_safe_complete"]) for row in rows
            ),
            "aggregates": aggregates,
            "shift_degradation_interactions": interactions,
            "interpretation_rule": (
                "positive lower-is-better interaction means the ablated arm "
                "deteriorated more from reference than the full arm"
            ),
        },
        "report_sha256",
    )
    destination = output_dir / "analysis" / f"{panel}-report.json"
    program.atomic_json(destination, report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("run-pilot", "analyze-pilot", "run-confirmation", "analyze-confirmation"),
    )
    parser.add_argument("--output-dir", type=Path, default=program.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--model-seed", type=int, action="append")
    parser.add_argument("--variant", choices=REPRESENTATION_VARIANTS, action="append")
    parser.add_argument("--regime", action="append")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    panel = PILOT_PANEL if "pilot" in args.command else CONFIRMATION_PANEL
    if args.command.startswith("run"):
        default_seeds = (
            program.PILOT_MODEL_SEEDS if panel == PILOT_PANEL else program.MODEL_SEEDS
        )
        result = evaluate(
            args.output_dir.resolve(),
            panel=panel,
            model_seeds=tuple(args.model_seed or default_seeds),
            variants=tuple(args.variant or REPRESENTATION_VARIANTS),
            regimes=tuple(args.regime) if args.regime else None,
            device_name=args.device,
        )
    else:
        result = analyze(
            args.output_dir.resolve(), panel=panel, allow_partial=args.allow_partial
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
