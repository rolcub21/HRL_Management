"""Observe the original admission pipeline without changing its decisions."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import fmean
from time import perf_counter
from unittest.mock import patch

import torch

import PSLAP.viability_candidates as candidates
from PSLAP.dynamic_yard import shortest_clear_path
from example.Options.ExplicitAcceptOption import ExplicitAcceptOption
from experiments.conditioned_vcg.E119_tolerance_defer_comparison_8x8_183k_v1 import run as replay
from . import protocol as p

read, write, require = replay.read, replay.write, replay.require
STAGES = ("physical_accept_count", "executable_accept_count", "accept_executor_rejections",
          "safe_accept_count", "unsafe_accept_count", "unknown_accept_count", "accept_candidate_count")


def missing_stage(counts, admissible_accepts):
    """Identify the first stage with no remaining options; never infer it from occupancy."""
    if admissible_accepts:
        return "accept_available"
    if not counts["physical_accept_count"]:
        return "physical_mask"
    if not counts["executable_accept_count"]:
        return "option_executability"
    if not counts["safe_accept_count"]:
        return "certification"
    return "post_certification_admissibility"


def executor_reason(option):
    """Read-only decomposition of the original option's failed initiation."""
    reason = option._binding_failure()
    if reason:
        return reason
    env, block = option.env, option._block()
    approach = env.plan_path_heuristic(env.current_state, block.position, ignore_block=block)
    if env.current_state != block.position and not approach:
        return "transporter_to_pickup_unreachable"
    transport = env.plan_path_heuristic(block.position, option.destination, ignore_block=block)
    if block.position != option.destination and not transport:
        return "pickup_to_storage_unreachable"
    return "unattributed_initiation_rejection"


class AuditedAcceptOption(ExplicitAcceptOption):
    def initiation(self, state):
        allowed = super().initiation(state)
        trace = getattr(self.env, "trace", None)
        if isinstance(trace, Trace):
            entry = trace.pending.setdefault(int(self.env.time_steps), {})
            entry.setdefault("executor", {})[self.destination] = {
                "cell": self.destination, "executable": bool(allowed),
                "reason": "executable" if allowed else executor_reason(self),
            }
        return allowed


def audited_certificate(original):
    def wrapped(env, *args, **kwargs):
        report = original(env, *args, **kwargs)
        trace = getattr(env, "trace", None)
        if isinstance(trace, Trace):
            entry = trace.pending.setdefault(int(env.time_steps), {})
            entry["certificates"] = [
                {"cell": item.cell, "status": item.status.value,
                 "reason": item.certificate.reason, "cache_hit": item.cache_hit,
                 "explored_nodes": item.certificate.explored_nodes,
                 "exhaustive": item.certificate.exhaustive,
                 "fixed_obstacles": sorted(item.recovery_state.fixed_obstacles)}
                for item in report.candidates]
        return report
    return wrapped


class Trace(replay.Trace):
    def __init__(self):
        super().__init__()
        self.pending = {}

    def decision(self, env, snapshot, selected):
        super().decision(env, snapshot, selected)
        row = self.current
        pipeline = {name: getattr(snapshot.audit, name) for name in STAGES}
        detail = self.pending.pop(int(env.time_steps), {})
        executor = list(detail.get("executor", {}).values())
        certificates = detail.get("certificates", [])
        require(len(executor) == pipeline["physical_accept_count"], "E120 missed physical Accept records")
        require(sum(item["executable"] for item in executor) == pipeline["executable_accept_count"],
                "E120 executor attribution disagrees with original enumeration")
        require(len(certificates) == pipeline["executable_accept_count"], "E120 missed certificates")
        # Diagnose the mask only; never add a candidate to the controller.
        physical_cells = {tuple(item["cell"]) for item in executor}
        inbound = candidates._strict_inbound(env)
        mask_reasons = Counter()
        for cell in snapshot.yard.storage_cells:
            if cell in physical_cells:
                mask_reasons["passed"] += 1
            elif cell in snapshot.yard.occupancy():
                mask_reasons["occupied_storage"] += 1
            elif cell in snapshot.recovery_state.reserved_cells:
                mask_reasons["reserved"] += 1
            elif inbound is not None and shortest_clear_path(
                    snapshot.yard, tuple(inbound.position), cell, ignore_labels=(inbound.label,)) is None:
                mask_reasons["no_stored_obstacle_clear_route"] += 1
            else:
                mask_reasons["no_inbound_or_other_mask_reason"] += 1
        stored = [block for block in env.blocks if block.stored and not block.delivered]
        live = [block for block in env.blocks if block.position is not None
                and not block.delivered and not block.carrying]
        row["accept_audit"] = {
            **pipeline, "missing_stage": missing_stage(pipeline, row["admissible_counts"].get("accept", 0)),
            "mask_reasons": dict(mask_reasons), "executor_candidates": executor,
            "executor_rejection_reasons": dict(Counter(item["reason"] for item in executor if not item["executable"])),
            "certificates": certificates,
            "certificate_status_reasons": dict(Counter(f"{item['status']}:{item['reason']}" for item in certificates)),
            "stored_occupancy": len(stored), "storage_capacity": len(snapshot.yard.storage_cells),
            "free_storage_cells": len(snapshot.yard.storage_cells) - len(stored),
            "stored_blocks": [{"label": block.label, "cell": block.position,
                               "due_slack": block.arrival_step + block.storage_steps_needed - env.time_steps}
                              for block in stored],
            "live_occupied_cells": [{"label": block.label, "cell": block.position, "stored": block.stored}
                                    for block in live],
            "transporter_cell": env.current_state, "pickup_cell": env.pickup_cell,
            "waiting_cells": [env.waiting_cell],
            "carried_blocks": [block.label for block in env.blocks if block.carrying],
            "certified_accept_keys": [item.key for item in snapshot.candidates
                                      if item.action_type.value == "accept"],
        }


def prepare(output):
    contract = p.contract()
    path = output / "contract.json"
    if path.exists():
        require(read(path) == contract, "E120 contract changed; use a new output version")
    else:
        require(not output.exists() or not any(output.iterdir()), "E120 refuses nonempty uncontracted output")
        write(path, contract)
    return contract


def behavior_signature(rows):
    return [{key: row[key] for key in ("decision", "time", "selected_key")} for row in rows]


def run_one(output, seed, device):
    contract = prepare(output)
    path = output / "rows" / f"instance-{seed}.json"
    if path.exists():
        saved = read(path)
        require(saved["contract_sha256"] == contract["contract_sha256"] and saved["exact_parity_passed"],
                "E120 cannot reuse failed or changed row")
        return
    source = read(p.SOURCE / f"rows/nominal_due/instance-{seed}.json")
    manifest = read(p.SOURCE / "frozen-source.json")
    payload = torch.load(p.SOURCE / "frozen-model.pth", map_location="cpu", weights_only=False)
    agent = replay.TracedAgent.from_checkpoint(payload, device=device, resumable=False, seed=0)
    agent.Q_local.eval().requires_grad_(False)
    agent.Q_target.eval().requires_grad_(False)
    agent.set_epsilon(0.)
    before, gradients = replay.parameter_digest(agent), agent.gradient_steps
    env = replay.TracedEnv(replay.p.SCENARIO, replay.p.e118.base.e107.e105.e78.SPEC)
    env.trace = Trace()
    env.current_episode = f"E120-attribution-{seed}"
    options = {key: manifest["liveness_rule"][key] for key in
               ("max_option_steps", "max_consecutive_defer_decisions")}
    started = perf_counter()
    print(f"E120 starting instance={seed} frozen episode=120 nominal rule", flush=True)
    with ExitStack() as stack:
        stack.enter_context(patch.object(candidates, "ExplicitAcceptOption", AuditedAcceptOption))
        stack.enter_context(patch.object(candidates, "certify_post_accept_candidates",
                                        audited_certificate(candidates.certify_post_accept_candidates)))
        stack.enter_context(patch.object(replay.audit, "ViabilityCertificateCache", replay.reuse.TimingInvariantCertificateCache))
        stack.enter_context(patch.object(replay.audit, "enumerate_viability_candidates",
            replay.e115run.v2run._accelerated_enumerator(replay.audit.enumerate_viability_candidates)))
        stack.enter_context(replay.reuse.path_cleanup_active())
        stack.enter_context(replay.e115run.fast_inputs.enabled())
        stack.enter_context(replay.e13._WallClock(300))
        with torch.no_grad():
            raw = replay.instrumentation.run_episode(
                agent, env, objective_spec=replay.p.e118.base.e107.e105.e78.SPEC,
                instance_seed=seed, training=False, max_steps=replay.p.MAX_STEPS,
                search_config=replay.ViabilitySearchConfig(**manifest["search_config"]),
                liveness_rule=replay.BoundedEventDeferRule(**options), epsilon_start=0., epsilon_end=0.,
                epsilon_decay_decisions=1, updates_per_macro=0, epsilon_warmup_decisions=0, run_protocol=p.PROTOCOL)
    require(replay.parameter_digest(agent) == before and agent.gradient_steps == gradients and not len(agent.replay),
            "E120 frozen evaluation mutated learner")
    differences = replay.parity_differences(replay.parity_fields(raw), replay.parity_fields(source["raw"]))
    signature_equal = behavior_signature(env.trace.decisions) == behavior_signature(source["decisions"])
    deliveries_equal = [(d["delivered_label"], d["delivered_time"], d["release_error"]) for d in env.trace.deliveries] == [
        (d["delivered_label"], d["delivered_time"], d["release_error"]) for d in source["deliveries"]]
    parity = not differences and signature_equal and deliveries_equal
    result = {"protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
              "instance_seed": seed, "raw": raw, "decisions": env.trace.decisions, "deliveries": env.trace.deliveries,
              "exact_parity_passed": parity, "aggregate_parity_differences": differences,
              "parameters_unchanged": True, "wall_seconds": perf_counter() - started,
              "training_runs": 0, "episode_runs": 1}
    write(path, result)
    replay.e115run.fast_inputs._REGISTRY.clear()
    require(parity, f"E120 behavior changed at instance {seed}; attribution invalid")
    early = [row for row in env.trace.deliveries if row["release_error"] < -20]
    stages = dict(Counter(row["accept_audit"]["missing_stage"] for row in early))
    print(f"E120 instance={seed} strict={int(raw['strict_method_success'])} early={len(early)} "
          f"stages={stages} parity=yes elapsed={result['wall_seconds']:.1f}s", flush=True)


def analyze(output):
    contract = prepare(output)
    rows = [read(path) for seed in p.INSTANCE_SEEDS
            if (path := output / "rows" / f"instance-{seed}.json").is_file()]
    require(all(row["exact_parity_passed"] for row in rows), "E120 parity failure")
    early = [{"instance_seed": row["instance_seed"], **event} for row in rows
             for event in row["deliveries"] if event["release_error"] < -20]
    stage_counts = dict(Counter(event["accept_audit"]["missing_stage"] for event in early))
    occupancy = Counter(event["accept_audit"]["stored_occupancy"] for event in early)
    pipeline = {name: sorted(set(event["accept_audit"][name] for event in early)) for name in STAGES}
    reasons = Counter()
    for event in early:
        reasons.update(event["accept_audit"]["executor_rejection_reasons"])
        reasons.update(event["accept_audit"]["certificate_status_reasons"])
    report = {"protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
              "status": "complete" if len(rows) == len(p.INSTANCE_SEEDS) else "partial",
              "rows": len(rows), "exact_parity_rows": len(rows), "early_deliveries": len(early),
              "missing_accept_stages": stage_counts, "stored_occupancy_counts": dict(occupancy),
              "pipeline_count_values_at_early_deliveries": pipeline, "cell_reason_counts": dict(reasons),
              "mean_J": fmean(row["raw"]["operator_cost"] for row in rows) if rows else None,
              "early_events": early, "training_runs": 0, "episode_runs": 0,
              "controller_changed": False, "paper_evidence": False, "test_panel_opened": False}
    write(output / "e120-report.json", report)
    lines = ["# E120 missing-Accept attribution", "",
             "Frozen E118 episode 120, nominal rules; exact E119 decision and delivery parity required.", "",
             "| Instance | Decision | Time | Occupancy | Physical | Executable | SAFE | UNSAFE | UNKNOWN | Missing stage |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for event in early:
        a = event["accept_audit"]
        lines.append(f"| {event['instance_seed']} | {event['decision']} | {event['time']} | "
                     f"{a['stored_occupancy']}/{a['storage_capacity']} | {a['physical_accept_count']} | "
                     f"{a['executable_accept_count']} | {a['safe_accept_count']} | {a['unsafe_accept_count']} | "
                     f"{a['unknown_accept_count']} | {a['missing_stage']} |")
    lines.extend(["", f"Early events: {len(early)}. Stage counts: {stage_counts}.", "",
                  f"Per-cell reasons (not independent episodes): {dict(reasons)}.", "",
                  "This observes candidate removal; it does not establish that changing the rule improves guarded returns.", ""])
    (output / "e120-table.md").write_text("\n".join(lines))
    return {key: value for key, value in report.items() if key != "early_events"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run-one", "run-panel", "analyze"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, choices=p.INSTANCE_SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "run-one":
        require(args.seed is not None, "E120 run-one requires --seed")
        run_one(args.output.resolve(), args.seed, args.device)
    elif args.command == "run-panel":
        for seed in p.INSTANCE_SEEDS:
            run_one(args.output.resolve(), seed, args.device)
    print(json.dumps(analyze(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
