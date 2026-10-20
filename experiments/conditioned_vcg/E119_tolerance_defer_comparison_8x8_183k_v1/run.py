"""Replay a frozen E118 policy under two waiting-deadline rules."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
from PSLAP.viability_candidates import BoundedEventDeferRule
from PSLAP.viability_filter import ViabilitySearchConfig
from experiments.conditioned_vcg.E13_operational_scalability_95k import program as e13
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1 import (
    instrumentation, model, rewards, run as e115run,
)
from . import protocol as p
from .policy import ToleranceDeferRule


class E119Error(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise E119Error(message)


def read(path):
    require(path.is_file(), f"missing E119 artifact: {path}")
    return json.loads(path.read_text())


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def parameter_digest(agent):
    digest = hashlib.sha256()
    for network in (agent.Q_local, agent.Q_target):
        for name, value in sorted(network.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def parity_fields(raw):
    return json_safe({key: raw[key] for key in (
        "episode_instance_id", "steps", "macro_decisions", "delivery_count",
        "delivery_deviations", "selected_action_counts", "selection_source_counts",
        "operator_cost", "strict_method_success",
    )})


def parity_differences(observed, expected):
    observed, expected = json_safe(observed), json_safe(expected)
    differences = {}
    for key, wanted in expected.items():
        actual = observed.get(key)
        equal = (math.isclose(float(actual), float(wanted), abs_tol=1.0e-6, rel_tol=0.)
                 if key == "operator_cost" else actual == wanted)
        if not equal:
            differences[key] = {"observed": actual, "expected": wanted}
    return differences


def prepare(output):
    output = output.resolve()
    contract = p.contract()
    path = output / "contract.json"
    if path.is_file():
        require(read(path) == contract, "E119 source or contract changed")
    else:
        require(not output.exists() or not any(output.iterdir()),
                "E119 refuses nonempty uncontracted output")
        write(path, contract)
    frozen = output / "frozen-model.pth"
    manifest_path = output / "frozen-source.json"
    if manifest_path.is_file():
        manifest = read(manifest_path)
        require(manifest["contract_sha256"] == contract["contract_sha256"], "E119 frozen source differs")
        require(frozen.is_file() and p.e118.base.sha256(frozen) == manifest["model_sha256"],
                "E119 frozen model changed")
    else:
        print("E119 extracting episode-120 weights; replay and optimizer are excluded", flush=True)
        payload = torch.load(p.SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
        require(payload.get("completed_training_episodes") == 120
                and payload.get("e118_arm") == "schedule_only"
                and payload.get("e118_contract_sha256") == contract["source_arm_contract_sha256"],
                "E119 source checkpoint identity differs")
        thin = {key: value for key, value in payload.items() if key not in (
            "agent_state", "training_history", "validation_history", "best_validation_record", "global_rng_state")}
        thin["agent_state"] = {key: payload["agent_state"][key] for key in (
            "Q_local", "Q_target", "epsilon", "decision_count", "transition_count",
            "gradient_steps", "target_updates")}
        temporary = frozen.with_suffix(".pth.tmp")
        torch.save(thin, temporary)
        os.replace(temporary, frozen)
        summary = read(p.SOURCE_SUMMARY)
        record = next(row for row in summary["validation_history"] if row["checkpoint_episode"] == 120)
        instances = []
        env = rewards.ReleaseETCostEnv(p.SCENARIO, p.e118.base.e107.e105.e78.SPEC)
        for seed in p.INSTANCE_SEEDS:
            instance = env.sample_episode_instance(seed)
            reference = next(run for run in record["runs"] if run["instance_seed"] == seed)
            require(instance.instance_id == reference["episode_instance_id"], "E119 instance generator drifted")
            instances.append({"seed": seed, "instance_id": instance.instance_id,
                              "instance_json": instance.to_json(), "parity_reference": parity_fields(reference)})
        training = read(p.SOURCE_DIRECTORY / "training-contract.json")["contract"]
        manifest = {"contract_sha256": contract["contract_sha256"],
                    "model_sha256": p.e118.base.sha256(frozen),
                    "search_config": training["search_config"],
                    "liveness_rule": training["liveness_rule"], "instances": instances}
        write(manifest_path, manifest)
    return {"status": "prepared", "contract_sha256": contract["contract_sha256"],
            "episode_runs": 0, "training_runs": 0, "planned_episodes": 10}


class Trace:
    def __init__(self):
        self.decisions = []
        self.deliveries = []
        self.current = None

    def decision(self, env, snapshot, selected):
        prepared = selected.prepared_snapshot
        available = Counter(record.action_type for record in prepared.records)
        by_type = {}
        for record, merit in zip(prepared.records, selected.q_values):
            by_type[record.action_type] = min(by_type.get(record.action_type, float("inf")),
                                            -float(merit) / .01)
        kwargs = dict(inbound_present=snapshot.inbound_label is not None,
                      current_certificate=snapshot.current_certificate,
                      consecutive_defer_decisions=snapshot.audit.consecutive_defer_decisions,
                      terminal=False)
        yard = SimpleNamespace(blocks=snapshot.recovery_state.blocks)
        native = BoundedEventDeferRule().evaluate(env, yard, **kwargs)
        shifted = ToleranceDeferRule().evaluate(env, yard, **kwargs)
        target = next((block for block in env.blocks
                       if str(block.label) == str(selected.candidate.target_label)), None)
        due = None if target is None else int(target.arrival_step + target.storage_steps_needed)
        selected_position = prepared.records.index(selected.record)
        row = {
            "decision": len(self.decisions) + 1, "time": int(env.time_steps),
            "selected_key": selected.candidate.key,
            "action": selected.candidate.action_type.value,
            "target": selected.candidate.target_label,
            "target_due": due, "target_due_slack": None if due is None else due - int(env.time_steps),
            "inbound_at_pickup": snapshot.inbound_label is not None,
            "minimum_stored_due_slack": min((float(b.remaining_time) for b in yard.blocks), default=None),
            "defer_allowed_by_rule": snapshot.audit.defer_allowed,
            "defer_rule_reason": snapshot.audit.defer_reason,
            "defer_admissible": bool(available.get("defer", 0)),
            "nominal_rule_allows_defer": native.allowed,
            "nominal_rule_reason": native.reason,
            "tolerance_rule_allows_defer": shifted.allowed,
            "tolerance_rule_reason": shifted.reason,
            "guard_forced": bool(selected.liveness_forced),
            "selection_source": selected.selection_source,
            "admissible_counts": dict(available),
            "best_predicted_cost_by_action": by_type,
            "selected_predicted_cost": -float(selected.q_values[selected_position]) / .01,
        }
        self.decisions.append(row)
        self.current = row


class TracedEnv(rewards.ReleaseETCostEnv):
    def step(self, action):
        result = super().step(action)
        label = result[3].get("delivered_block")
        if label:
            block = next(item for item in self.blocks if str(item.label) == str(label))
            self.trace.deliveries.append({
                **self.trace.current, "delivered_label": str(label),
                "delivered_time": int(block.delivered_time_step),
                "release_error": rewards.release_error(block),
                "macro_elapsed_to_delivery": int(self.time_steps) - self.trace.current["time"],
            })
        return result


class TracedAgent(model.FactorizedConservativeAgent):
    def select(self, snapshot, **kwargs):
        require(not kwargs.get("training"), "E119 disallows training selection")
        decision = super().select(snapshot, **kwargs)
        self.context_environment.trace.decision(self.context_environment, snapshot, decision)
        return decision


def attribution(trace):
    early = [row for row in trace.deliveries if row["release_error"] < -20]
    reasons = Counter(row["defer_rule_reason"] for row in early if not row["defer_admissible"])
    return {"deliveries": len(trace.deliveries), "early_beyond_tolerance": len(early),
            "early_with_defer_available": sum(row["defer_admissible"] for row in early),
            "early_without_defer": sum(not row["defer_admissible"] for row in early),
            "early_guard_forced": sum(row["guard_forced"] for row in early),
            "early_nominal_block_removed_by_tolerance": sum(
                not row["nominal_rule_allows_defer"] and row["tolerance_rule_allows_defer"] for row in early),
            "early_defer_unavailable_reasons": dict(reasons),
            "decisions_newly_eligible_under_tolerance": sum(
                not row["nominal_rule_allows_defer"] and row["tolerance_rule_allows_defer"] for row in trace.decisions)}


def row_path(output, arm, seed):
    return output / "rows" / arm / f"instance-{seed}.json"


def run_one(output, arm, seed, device="cpu"):
    prepare(output)
    require(arm in p.ARMS and seed in p.INSTANCE_SEEDS, "E119 coordinate outside frozen panel")
    contract = p.contract()
    path = row_path(output, arm, seed)
    if path.is_file():
        saved = read(path)
        require(saved["contract_sha256"] == contract["contract_sha256"], "E119 saved row changed")
        require(arm != "nominal_due" or saved.get("native_parity_passed") is True,
                "E119 saved native parity failed; intervention remains blocked")
        return {"status": "reused", "arm": arm, "seed": seed, "episode_runs": 0, "training_runs": 0}
    if arm == "due_plus_20":
        for instance in p.INSTANCE_SEEDS:
            native_path = row_path(output, "nominal_due", instance)
            require(native_path.is_file() and read(native_path).get("native_parity_passed") is True,
                    "E119 requires all five current-rule parity rows before the intervention")
    manifest = read(output / "frozen-source.json")
    payload = torch.load(output / "frozen-model.pth", map_location="cpu", weights_only=False)
    agent = TracedAgent.from_checkpoint(payload, device=device, resumable=False, seed=0)
    agent.Q_local.eval().requires_grad_(False)
    agent.Q_target.eval().requires_grad_(False)
    agent.set_epsilon(0.)
    before = parameter_digest(agent)
    steps_before = agent.gradient_steps
    trace = Trace()
    env = TracedEnv(p.SCENARIO, p.e118.base.e107.e105.e78.SPEC)
    env.trace = trace
    env.current_episode = f"E119-{arm}-{seed}"
    options = {key: manifest["liveness_rule"][key] for key in
               ("max_option_steps", "max_consecutive_defer_decisions")}
    rule = BoundedEventDeferRule(**options) if arm == "nominal_due" else ToleranceDeferRule(**options)
    started = perf_counter()
    print(f"E119 starting {arm} instance={seed} checkpoint=120 device={device}", flush=True)
    with ExitStack() as stack:
        stack.enter_context(patch.object(audit, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache))
        stack.enter_context(patch.object(audit, "enumerate_viability_candidates",
            e115run.v2run._accelerated_enumerator(audit.enumerate_viability_candidates)))
        stack.enter_context(reuse.path_cleanup_active())
        stack.enter_context(e115run.fast_inputs.enabled())
        stack.enter_context(e13._WallClock(300))
        with torch.no_grad():
            raw = instrumentation.run_episode(
                agent, env, objective_spec=p.e118.base.e107.e105.e78.SPEC,
                instance_seed=seed, training=False, max_steps=p.MAX_STEPS,
                search_config=ViabilitySearchConfig(**manifest["search_config"]),
                liveness_rule=rule, epsilon_start=0., epsilon_end=0.,
                epsilon_decay_decisions=1, updates_per_macro=0,
                epsilon_warmup_decisions=0, run_protocol=p.PROTOCOL)
    require(parameter_digest(agent) == before and agent.gradient_steps == steps_before and len(agent.replay) == 0,
            "E119 evaluation mutated weights, optimizer count, or replay")
    instance = next(item for item in manifest["instances"] if item["seed"] == seed)
    require(raw["episode_instance_id"] == instance["instance_id"], "E119 executed another instance")
    parity = parity_differences(parity_fields(raw), instance["parity_reference"]) if arm == "nominal_due" else None
    strict = bool(raw["strict_method_success"] and raw["delivery_count"] == 36
                  and raw["service_metrics"]["terminal_costs_match_dense_accrual"])
    require(math.isfinite(float(raw["operator_cost"])), "E119 objective cost is nonfinite")
    require(len(trace.deliveries) == raw["delivery_count"], "E119 missed delivery attribution events")
    result = {"protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
              "arm": arm, "instance_seed": seed, "status": "complete" if strict else "failed_completion",
              "strict_complete": strict, "raw": raw,
              "native_parity_passed": not parity if arm == "nominal_due" else None,
              "parity_differences": parity, "decisions": trace.decisions,
              "deliveries": trace.deliveries, "attribution": attribution(trace),
              "parameters_unchanged": True, "wall_seconds": perf_counter() - started,
              "episode_runs": 1, "training_runs": 0}
    write(path, result)
    e115run.fast_inputs._REGISTRY.clear()
    print(f"E119 {arm} instance={seed} strict={int(strict)} J={raw['operator_cost']:.2f} "
          f"early={result['attribution']['early_beyond_tolerance']}/36 "
          f"parity={result['native_parity_passed']} seconds={result['wall_seconds']:.1f}", flush=True)
    if parity:
        raise E119Error(f"E119 current-rule replay failed parity: {parity}; intervention blocked")
    return {"status": result["status"], "arm": arm, "seed": seed,
            "episode_runs": 1, "training_runs": 0}


def analyze(output):
    prepare(output)
    arms = {}
    for arm in p.ARMS:
        paths = [row_path(output, arm, seed) for seed in p.INSTANCE_SEEDS]
        rows = [read(path) for path in paths if path.is_file()]
        valid = [row for row in rows if row["strict_complete"]]
        means = None
        if len(valid) == len(p.INSTANCE_SEEDS):
            means = {key: fmean(row["raw"]["service_metrics"][key] for row in valid) for key in (
                "objective_cost", "release_clock_mae", "mean_wait_before_storage",
                "mean_release_earliness_beyond_tolerance", "mean_release_tardiness_beyond_tolerance",
                "within_tolerance_rate", "equipment_work", "physical_rehandles")}
            means["steps"] = fmean(row["raw"]["steps"] for row in valid)
        reasons = Counter()
        counts = Counter()
        for row in rows:
            for key, value in row["attribution"].items():
                if isinstance(value, dict):
                    reasons.update(value)
                else:
                    counts[key] += value
        arms[arm] = {"rows": len(rows), "strict_complete_rows": len(valid), "means": means,
                     "attribution": dict(counts), "early_defer_unavailable_reasons": dict(reasons)}
    paired = []
    for seed in p.INSTANCE_SEEDS:
        paths = [row_path(output, arm, seed) for arm in p.ARMS]
        if not all(path.is_file() for path in paths):
            continue
        left, right = [read(path) for path in paths]
        require(left["raw"]["episode_instance_id"] == right["raw"]["episode_instance_id"], "E119 pairing changed")
        paired.append({"seed": seed, "both_strict_complete": left["strict_complete"] and right["strict_complete"],
                       "J_difference": right["raw"]["operator_cost"] - left["raw"]["operator_cost"]
                       if left["strict_complete"] and right["strict_complete"] else None})
    report = {"protocol": p.PROTOCOL, "contract_sha256": p.contract()["contract_sha256"],
              "status": "complete" if all(arm["rows"] == 5 for arm in arms.values()) else "partial",
              "arms": arms, "paired": paired,
              "mean_J_difference_due_plus_20_minus_nominal": fmean(row["J_difference"] for row in paired)
                  if len(paired) == 5 and all(row["both_strict_complete"] for row in paired) else None,
              "paper_evidence": False, "training_runs": 0, "episode_runs": 0}
    write(output / "e119-report.json", report)
    lines = ["# E119 frozen Defer-cutoff comparison", "",
             "Same E118 episode-120 model and five reused development instances. Lower J is better.", "",
             "| Rule | Rows | Strict | J | Early excess/job | Late excess/job | Within ±20 | Work |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, value in arms.items():
        m = value["means"]
        if m is None:
            lines.append(f"| {arm} | {value['rows']}/5 | {value['strict_complete_rows']}/5 | — | — | — | — | — |")
        else:
            lines.append(f"| {arm} | 5/5 | 5/5 | {m['objective_cost']:.2f} | "
                         f"{m['mean_release_earliness_beyond_tolerance']:.3f} | "
                         f"{m['mean_release_tardiness_beyond_tolerance']:.3f} | "
                         f"{m['within_tolerance_rate']:.3f} | {m['equipment_work']:.1f} |")
    lines.extend(["", "## Early-delivery attribution", ""])
    for arm, value in arms.items():
        lines.append(f"{arm}: {json.dumps(value['attribution'], sort_keys=True)}")
        lines.extend(["", f"Defer rejection reasons: {json.dumps(value['early_defer_unavailable_reasons'], sort_keys=True)}", ""])
    lines.extend(["The cutoff changes runtime choices on a frozen policy. "
                  "No conclusion about retrained performance or a formal liveness extension follows automatically.", ""])
    (output / "e119-table.md").write_text("\n".join(lines))
    return {"status": report["status"], "rows": {arm: value["rows"] for arm, value in arms.items()},
            "report": str(output / "e119-report.json"), "training_runs": 0, "episode_runs": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "self-check", "run-one", "run-panel", "analyze", "status"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--arm", choices=p.ARMS)
    parser.add_argument("--seed", type=int, choices=p.INSTANCE_SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = args.output.resolve()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "self-check":
        prepare(output)
        payload = torch.load(output / "frozen-model.pth", map_location="cpu", weights_only=False)
        agent = TracedAgent.from_checkpoint(payload, device="cpu", resumable=False, seed=0)
        require(len(agent.replay) == 0 and payload["completed_training_episodes"] == 120,
                "E119 frozen model self-check failed")
        result = {"status": "passed", "source_episode": 120, "replay_size": 0,
                  "training_runs": 0, "episode_runs": 0}
    elif args.command == "run-one":
        require(args.arm is not None and args.seed is not None, "run-one needs --arm and --seed")
        result = run_one(output, args.arm, args.seed, args.device)
    elif args.command == "run-panel":
        prepare(output)
        for arm in p.ARMS:
            for seed in p.INSTANCE_SEEDS:
                run_one(output, arm, seed, args.device)
        result = analyze(output)
    else:
        result = analyze(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
