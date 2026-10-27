#!/usr/bin/env python3
"""Evaluate frozen E80 heads on fresh, paired 8x8 preference sweeps."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import torch

import benchmark_viability_critic_priority as benchmark
from example.helper.timing_metrics import summarize_block_storage_flow
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E28_empty_start_scale_baselines_100k import adapter, run as e28
from experiments.conditioned_vcg.E80_original_vcg_base80_stage_b_8x8_145k_v1 import run as e80
from experiments.conditioned_vcg.target_size_vcg_training import run as prior
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
from train_vcg_preference_conditioned import _atomic_json_save
from vcg_v11_conditioned_handling import ConditionedHandlingAgent

from . import protocol as p


class E82Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E82Error(message)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing E82 file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E82 JSON: {path}")
    return value


def authenticate(output: Path) -> dict:
    observed = read_json(output / "contract.json")
    expected = p.contract()
    require(observed == expected, "E82 source, checkpoint, or protocol changed")
    return observed


def prepare(output: Path) -> dict:
    expected = p.contract()
    file = output / "contract.json"
    if file.is_file():
        authenticate(output)
    else:
        require(not output.exists() or not any(output.iterdir()),
                "E82 refuses a nonempty, uncontracted output")
        _atomic_json_save(expected, file)
    return {
        "status": "prepared", "contract_sha256": expected["contract_sha256"],
        "model_seeds": list(p.MODEL_SEEDS), "lambdas": list(p.LAMBDAS),
        "pilot_rows": len(p.LAMBDAS),
        "confirmation_rows": len(p.MODEL_SEEDS) * len(p.PANEL_SEEDS["confirmation"]) * len(p.LAMBDAS),
        "episode_runs": 0, "training_runs": 0,
    }


def lambda_name(value: float) -> str:
    require(value in p.LAMBDAS, "E82 lambda is outside frozen sweep")
    return f"lambda-{value:.2f}".replace(".", "p")


def coordinate_path(output: Path, panel: str, model_seed: int,
                    instance_seed: int, preference_lambda: float) -> Path:
    return (output / "rows" / panel / f"model-{model_seed}"
            / f"instance-{instance_seed}-{lambda_name(preference_lambda)}.json")


def coordinates(panel: str):
    require(panel in p.PANEL_SEEDS, "E82 unknown panel")
    seeds = (0,) if panel == "pilot" else p.MODEL_SEEDS
    for instance_seed in p.PANEL_SEEDS[panel]:
        for model_seed in seeds:
            for value in p.LAMBDAS:
                yield model_seed, instance_seed, value


def load_model(contract: dict, seed: int):
    require(seed in p.MODEL_SEEDS, "E82 unknown model seed")
    device = torch.device("cpu")
    base, search, guard, base_digest = e80.load_base(contract, seed, device)
    entry = contract["terminal_heads"][seed]
    path = p.ROOT / entry["checkpoint"]
    require(p.sha256(path) == entry["checkpoint_sha256"],
            "E82 E80 terminal handling head changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    require(saved.get("training_protocol") == p.e80.PROTOCOL
            and saved.get("role") == "terminal"
            and saved.get("contract_sha256") == contract["e80_contract_sha256"]
            and saved.get("scenario") == p.SCENARIO_ID
            and saved.get("model_seed") == seed
            and saved.get("completed_rounds") == p.e80.ROUNDS
            and saved.get("base_checkpoint_sha256") == entry["base_checkpoint_sha256"]
            and saved.get("base_policy_digest") == base_digest,
            "E82 E80 terminal checkpoint identity differs")
    agent = ConditionedHandlingAgent.from_checkpoint(
        saved["agent_checkpoint"], base_agent=base,
        expected_base_checkpoint_sha256=entry["base_checkpoint_sha256"],
        expected_base_policy_digest=base_digest,
        expected_source_cost_sha256=prior.NO_SOURCE_COST,
        seed=seed,
    )
    return agent, search, guard, base_digest


class FixedLambdaAgent:
    """Only adapts the conditioned select signature to the shared executor."""

    def __init__(self, agent: ConditionedHandlingAgent, value: float):
        self.agent = agent
        self.value = float(value)
        self.config = agent.base_agent.config

    def reset_episode_state(self):
        self.agent.reset_episode_state()

    def select(self, snapshot, *, training=False, epsilon=0.0):
        return self.agent.select(snapshot, preference_lambda=self.value,
                                 training=training, epsilon=epsilon)

    def observe_outcome(self, decision, *, next_snapshot, done):
        return self.agent.observe_outcome(decision, next_snapshot=next_snapshot,
                                          done=done)


def verify_models(output: Path) -> dict:
    contract = authenticate(output)
    controls = []
    for seed in p.MODEL_SEEDS:
        agent, search, guard, base_digest = load_model(contract, seed)
        marker = object()
        with patch.object(agent.base_agent, "select", return_value=marker) as select:
            require(agent.select(None, preference_lambda=0.0,
                                 training=False, epsilon=0.0) is marker,
                    "E82 lambda zero does not delegate exactly to Stage A")
            select.assert_called_once_with(None, training=False, epsilon=0.0)
        with torch.no_grad():
            values = agent.handling_network(
                torch.zeros((2, agent.config.feature_dim)),
                torch.tensor([0.05, 0.20]),
            )
        require(values.shape == (2,) and bool(torch.isfinite(values).all()),
                "E82 conditioned head is not finite")
        controls.append((search, guard, base_digest))
    require(all(item[:2] == controls[0][:2] for item in controls),
            "E82 model seeds have different search or guard settings")
    instance = e28._instance(p.SCENARIO, p.PANEL_SEEDS["pilot"][0])
    env = adapter.AuditedEmptyStartEnv(p.SCENARIO)
    env.reset(instance=instance)
    require(env.warm_start_audit is not None
            and env.warm_start_audit["passed"],
            "E82 pilot does not start empty")
    return {
        "status": "passed", "model_seeds": list(p.MODEL_SEEDS),
        "lambda_zero_delegation_checked": True,
        "base_policy_digests": [item[2] for item in controls],
        "pilot_instance_id": instance.instance_id,
        "episode_runs": 0, "training_runs": 0,
    }


def timing(values: list[float]) -> dict:
    require(values and all(math.isfinite(value) for value in values),
            "E82 timing records are incomplete")
    return {
        "mae": mean(abs(value) for value in values),
        "mean_signed_error": mean(values),
        "within_20_rate": mean(abs(value) <= 20.0 for value in values),
    }


def one_row(output: Path, contract: dict, panel: str, model_seed: int,
            instance_seed: int, value: float) -> dict:
    require((model_seed, instance_seed, value) in set(coordinates(panel)),
            "E82 coordinate is outside frozen panel")
    path = coordinate_path(output, panel, model_seed, instance_seed, value)
    identity = {
        "protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
        "panel": panel, "scenario": p.SCENARIO_ID,
        "model_seed": model_seed, "instance_seed": instance_seed,
        "preference_lambda": value,
    }
    if path.is_file():
        row = read_json(path)
        require(all(row.get(key) == expected for key, expected in identity.items()),
                f"E82 saved coordinate differs: {path}")
        return {**row, "reused": True}

    print(f"E82 starting {panel} model={model_seed} instance={instance_seed} "
          f"lambda={value:.2f}", flush=True)
    instance = e28._instance(p.SCENARIO, instance_seed)
    agent, search, guard, base_digest = load_model(contract, model_seed)
    base_entry = contract["stage_a"][model_seed]
    payload = torch.load(p.ROOT / base_entry["terminal_checkpoint"],
                         map_location="cpu", weights_only=False, mmap=True)
    envs = []

    def make_env(_payload):
        env = adapter.AuditedEmptyStartEnv(p.SCENARIO)
        envs.append(env)
        return env

    def freeze_agent(_payload, *, device, seed):
        require(device.type == "cpu" and seed == model_seed,
                "E82 runtime changed model seed or device")
        return FixedLambdaAgent(agent, value)

    with ExitStack() as stack:
        stack.enter_context(reuse.path_cleanup_active())
        stack.enter_context(patch.object(benchmark, "_make_env", make_env))
        stack.enter_context(patch.object(benchmark, "_freeze_agent", freeze_agent))
        stack.enter_context(patch.object(
            benchmark, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache,
        ))
        raw = benchmark.run_arm(
            arm=benchmark.EXACT_FULL, controller_payload=payload,
            instance=instance, instance_seed=instance_seed,
            search_config=search, liveness_rule=guard, prioritizer=None,
            max_steps=p.SCENARIO.max_steps, device=torch.device("cpu"),
            recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION,
        )
    require(len(envs) == 1 and envs[0].warm_start_audit is not None,
            "E82 empty-start reset was not authenticated")
    env = envs[0]
    require(env.current_episode_instance.instance_id == instance.instance_id,
            "E82 paired instance changed during execution")
    require(e80.state_digest(agent.base_agent.Q_local.state_dict()) == base_digest,
            "E82 evaluation mutated frozen Stage A")
    blocks = tuple(env.blocks)
    strict = bool(raw["strict_method_success"] and raw["terminal"]
                  and raw["method_failure_reason"] is None
                  and raw["complete_frontier_exactly_verified"]
                  and raw["illegal_drops"] == 0 and raw["macro_failures"] == 0
                  and sum(block.delivered for block in blocks) == p.SCENARIO.total_jobs)
    flow = summarize_block_storage_flow(blocks, int(env.time_steps))
    require(flow["storage_flow_manifest_count"] == p.SCENARIO.total_jobs,
            "E82 storage-flow manifest differs")
    storage_errors = []
    release_errors = []
    if strict:
        require(all(block.stored_time_step is not None
                    and block.delivered_time_step is not None for block in blocks),
                "E82 strict episode lacks a block clock")
        storage_errors = [
            float(block.delivered_time_step - block.stored_time_step
                  - block.storage_steps_needed) for block in blocks
        ]
        release_errors = [
            float(block.delivered_time_step - block.arrival_step
                  - block.storage_steps_needed) for block in blocks
        ]
        require(all(abs(a - b) < 1e-9 for a, b in zip(
            sorted(storage_errors), sorted(raw["delivery_deviations"]))),
            "E82 storage clock disagrees with executor")
    rehandles = int(raw.get("physical_storage_relocations", raw["relocations"]))
    row = {
        **identity,
        "instance_id": instance.instance_id,
        "strict_safe_complete": strict,
        "failure_class": ("completed" if strict else
                          "certificate_budget_limited" if any(
                              item.get("unknown_candidates_rejected", 0) > 0
                              or item.get("safe_candidate_count", 0) == 0
                              for item in raw["frontiers"]
                          ) else "valid_but_operationally_incomplete"),
        "method_failure_reason": raw["method_failure_reason"],
        "required_deliveries": p.SCENARIO.total_jobs,
        "delivered_at_stop": sum(block.delivered for block in blocks),
        "steps_at_stop": int(raw["steps"]),
        "max_steps": p.SCENARIO.max_steps,
        "rehandles_at_stop": rehandles,
        "macro_decisions": int(raw["macro_decisions"]),
        "guard_forced_decisions": sum(bool(item["liveness_forced"])
                                      for item in raw["decisions"]),
        "storage_flow_completed": flow["storage_flow_completed_count"],
        "storage_flow_censored": flow["storage_flow_right_censored_count"],
        "mean_wait_before_storage": flow["mean_storage_flow_time"] if strict else None,
        "storage_clock": timing(storage_errors) if strict else None,
        "release_clock": timing(release_errors) if strict else None,
        "steps_per_delivery": raw["steps"] / p.SCENARIO.total_jobs if strict else None,
        "rehandles_per_100": 100.0 * rehandles / p.SCENARIO.total_jobs if strict else None,
        "occupancy": env.occupancy_summary(),
        "episode_wall_seconds": float(raw["episode_wall_seconds"]),
        "behavior_digest": raw["behavior_digest"],
        "base_checkpoint_sha256": base_entry["terminal_checkpoint_sha256"],
        "handling_checkpoint_sha256": contract["terminal_heads"][model_seed]["checkpoint_sha256"],
    }
    _atomic_json_save(row, path)
    return row


def run_panel(output: Path, panel: str) -> dict:
    contract = authenticate(output)
    new_runs = 0
    for index, (seed, instance_seed, value) in enumerate(coordinates(panel), 1):
        row = one_row(output, contract, panel, seed, instance_seed, value)
        new_runs += int(not row.get("reused", False))
        print(f"E82 {panel} {index} | seed={seed} instance={instance_seed} "
              f"lambda={value:.2f} strict={int(row['strict_safe_complete'])} "
              f"steps={row['steps_at_stop']}", flush=True)
    return {
        "status": "complete", "panel": panel,
        "rows": len(list(coordinates(panel))),
        "new_episode_runs": new_runs, "training_runs": 0,
    }


def analyze(output: Path, panel: str) -> dict:
    contract = authenticate(output)
    rows = []
    missing = []
    for seed, instance_seed, value in coordinates(panel):
        path = coordinate_path(output, panel, seed, instance_seed, value)
        if not path.is_file():
            missing.append(str(path))
            continue
        row = read_json(path)
        require(row.get("contract_sha256") == contract["contract_sha256"]
                and row.get("panel") == panel and row.get("model_seed") == seed
                and row.get("instance_seed") == instance_seed
                and row.get("preference_lambda") == value,
                f"E82 saved row identity differs: {path}")
        rows.append(row)
    summaries = []
    for value in p.LAMBDAS:
        subset = [row for row in rows if row["preference_lambda"] == value]
        complete = [row for row in subset if row["strict_safe_complete"]]
        summaries.append({
            "preference_lambda": value,
            "observed_rows": len(subset),
            "strict_complete_rows": len(complete),
            "mean_wait_completed": (mean(row["mean_wait_before_storage"]
                                         for row in complete) if complete else None),
            "mean_release_mae_completed": (mean(row["release_clock"]["mae"]
                                                for row in complete) if complete else None),
            "mean_storage_mae_completed": (mean(row["storage_clock"]["mae"]
                                                for row in complete) if complete else None),
            "mean_steps_per_delivery_completed": (mean(row["steps_per_delivery"]
                                                        for row in complete) if complete else None),
            "mean_rehandles_per_100_completed": (mean(row["rehandles_per_100"]
                                                       for row in complete) if complete else None),
        })
    by_key = {(row["model_seed"], row["instance_seed"], row["preference_lambda"]): row
              for row in rows}
    pairs = []
    for seed, instance_seed, value in coordinates(panel):
        if value == 0.0:
            continue
        base = by_key.get((seed, instance_seed, 0.0))
        treatment = by_key.get((seed, instance_seed, value))
        if base is None or treatment is None:
            continue
        both = base["strict_safe_complete"] and treatment["strict_safe_complete"]
        pairs.append({
            "model_seed": seed, "instance_seed": instance_seed,
            "preference_lambda": value,
            "lambda_zero_strict": base["strict_safe_complete"],
            "lambda_positive_strict": treatment["strict_safe_complete"],
            "both_strict": both,
            "positive_minus_zero_wait": (treatment["mean_wait_before_storage"]
                                         - base["mean_wait_before_storage"] if both else None),
            "positive_minus_zero_release_mae": (treatment["release_clock"]["mae"]
                                                - base["release_clock"]["mae"] if both else None),
            "positive_minus_zero_storage_mae": (treatment["storage_clock"]["mae"]
                                                - base["storage_clock"]["mae"] if both else None),
            "positive_minus_zero_steps_per_delivery": (treatment["steps_per_delivery"]
                                                       - base["steps_per_delivery"] if both else None),
            "positive_minus_zero_rehandles_per_100": (treatment["rehandles_per_100"]
                                                      - base["rehandles_per_100"] if both else None),
        })
    result = {
        "protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
        "panel": panel, "status": "complete" if not missing else "partial",
        "observed_rows": len(rows),
        "expected_rows": len(list(coordinates(panel))),
        "missing_rows": missing, "summaries": summaries, "pairs": pairs,
        "interpretation_caveat": (
            "Pilot is exploratory. The three-seed confirmation compares frozen terminal "
            "Stage-B heads on paired fresh 8x8 instances. Lambda zero is direct Stage-A "
            "delegation. Completion is unconditional; quality means condition on strict "
            "completion. No best lambda or model checkpoint is selected here."
        ),
        "training_runs": 0,
    }
    report = output / f"e82-{panel}-report.json"
    _atomic_json_save(result, report)
    return {
        "status": result["status"], "panel": panel,
        "observed_rows": len(rows), "expected_rows": result["expected_rows"],
        "report": str(report), "training_runs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "prepare", "verify-models", "run-one", "run-panel", "analyze",
    ))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--panel", choices=tuple(p.PANEL_SEEDS), default="pilot")
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--instance-seed", type=int)
    parser.add_argument("--lambda", dest="preference_lambda", type=float,
                        choices=p.LAMBDAS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "verify-models":
        result = verify_models(output)
    elif args.command == "run-one":
        require(args.model_seed is not None and args.instance_seed is not None
                and args.preference_lambda is not None,
                "run-one needs model-seed, instance-seed, and lambda")
        result = one_row(output, authenticate(output), args.panel,
                         args.model_seed, args.instance_seed, args.preference_lambda)
    elif args.command == "run-panel":
        result = run_panel(output, args.panel)
    else:
        result = analyze(output, args.panel)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
