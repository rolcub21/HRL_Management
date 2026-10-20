"""Frozen test of waiting as an option, not a forced-wait policy or retraining."""

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

from experiments.conditioned_vcg.E120_accept_frontier_attribution_8x8_184k_v1 import run as attribution
from . import protocol as p
from .policy import BlockedPickupWaitRule

replay = attribution.replay
read, write, require = replay.read, replay.write, replay.require


def prepare(output):
    contract = p.contract()
    path = output / "contract.json"
    if path.is_file():
        require(read(path) == contract, "E121 contract changed; use a separately versioned output")
    else:
        require(not output.exists() or not any(output.iterdir()), "E121 refuses nonempty uncontracted output")
        write(path, contract)
    return contract


def run_one(output, seed, device):
    contract = prepare(output)
    path = output / "rows" / f"instance-{seed}.json"
    if path.exists():
        require(read(path)["contract_sha256"] == contract["contract_sha256"], "E121 saved row differs")
        return
    nominal = read(p.SOURCE / f"rows/nominal_due/instance-{seed}.json")
    manifest = read(p.SOURCE / "frozen-source.json")
    payload = torch.load(p.SOURCE / "frozen-model.pth", map_location="cpu", weights_only=False)
    agent = replay.TracedAgent.from_checkpoint(payload, device=device, resumable=False, seed=0)
    agent.Q_local.eval().requires_grad_(False)
    agent.Q_target.eval().requires_grad_(False)
    agent.set_epsilon(0.)
    before, gradients = replay.parameter_digest(agent), agent.gradient_steps
    env = replay.TracedEnv(replay.p.SCENARIO, replay.p.e118.base.e107.e105.e78.SPEC)
    env.trace = attribution.Trace()
    env.current_episode = f"E121-blocked-pickup-wait-{seed}"
    options = {key: manifest["liveness_rule"][key] for key in
               ("max_option_steps", "max_consecutive_defer_decisions")}
    print(f"E121 starting instance={seed} episode=120 bounded blocked-pickup wait", flush=True)
    started = perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(attribution.candidates, "ExplicitAcceptOption", attribution.AuditedAcceptOption))
        stack.enter_context(patch.object(attribution.candidates, "certify_post_accept_candidates",
            attribution.audited_certificate(attribution.candidates.certify_post_accept_candidates)))
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
                liveness_rule=BlockedPickupWaitRule(**options), epsilon_start=0., epsilon_end=0.,
                epsilon_decay_decisions=1, updates_per_macro=0, epsilon_warmup_decisions=0, run_protocol=p.PROTOCOL)
    require(replay.parameter_digest(agent) == before and agent.gradient_steps == gradients and not len(agent.replay),
            "E121 evaluation mutated learner")
    require(raw["episode_instance_id"] == nominal["raw"]["episode_instance_id"], "E121 instance mismatch")
    eligible = [row for row in env.trace.decisions
                if row["defer_rule_reason"] == "e121_bounded_inbound_wait_at_physically_blocked_pickup"]
    chosen = [row for row in eligible if row["action"] == "defer"]
    left, right = attribution.behavior_signature(env.trace.decisions), attribution.behavior_signature(nominal["decisions"])
    first_difference = next((i + 1 for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
    if first_difference is None and len(left) != len(right):
        first_difference = min(len(left), len(right)) + 1
    result = {"protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
              "instance_seed": seed, "raw": raw, "decisions": env.trace.decisions, "deliveries": env.trace.deliveries,
              "parameters_unchanged": True, "new_wait_eligible_decisions": len(eligible),
              "new_wait_selected_decisions": len(chosen), "first_choice_or_time_difference": first_difference,
              "same_trajectory_as_nominal": first_difference is None,
              "J_difference_from_nominal": raw["operator_cost"] - nominal["raw"]["operator_cost"],
              "wall_seconds": perf_counter() - started, "training_runs": 0, "episode_runs": 1}
    write(path, result)
    replay.e115run.fast_inputs._REGISTRY.clear()
    print(f"E121 instance={seed} strict={int(raw['strict_method_success'])} J={raw['operator_cost']:.2f} "
          f"deltaJ={result['J_difference_from_nominal']:+.2f} new_wait={len(chosen)}/{len(eligible)} "
          f"elapsed={result['wall_seconds']:.1f}s", flush=True)


def metrics(rows):
    names = ("objective_cost", "mean_release_earliness_beyond_tolerance", "mean_release_tardiness_beyond_tolerance",
             "mean_wait_before_storage", "within_tolerance_rate", "equipment_work", "physical_rehandles",
             "release_clock_mae")
    early = [event for row in rows for event in row["deliveries"] if event["release_error"] < -20]
    return {**{name: fmean(row["raw"]["service_metrics"][name] for row in rows) for name in names},
            "steps": fmean(row["raw"]["steps"] for row in rows), "early_deliveries": len(early),
            "guard_forced_early_deliveries": sum(event["guard_forced"] for event in early),
            "guard_forced_decisions": sum(event["guard_forced"] for row in rows for event in row["decisions"])}


def analyze(output):
    contract = prepare(output)
    rows = [read(path) for seed in p.INSTANCE_SEEDS if (path := output / "rows" / f"instance-{seed}.json").exists()]
    nominal = [read(p.SOURCE / f"rows/nominal_due/instance-{row['instance_seed']}.json") for row in rows]
    strict = sum(row["raw"]["strict_method_success"] and row["raw"]["delivery_count"] == 36 for row in rows)
    comparable = bool(rows) and strict == len(rows)
    report = {"protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
              "status": "complete" if len(rows) == 5 else "partial", "rows": len(rows), "strict_complete_rows": strict,
              "nominal": metrics(nominal) if comparable else None,
              "bounded_pickup_wait": metrics(rows) if comparable else None,
              "eligible_decisions": sum(row["new_wait_eligible_decisions"] for row in rows),
              "new_wait_selected_decisions": sum(row["new_wait_selected_decisions"] for row in rows),
              "unchanged_trajectory_rows": sum(row["same_trajectory_as_nominal"] for row in rows),
              "paired_J_differences": [row["J_difference_from_nominal"] for row in rows] if comparable else None,
              "incomplete_episodes": "counted as failures; quality means suppressed",
              "training_runs": 0, "episode_runs": 0, "paper_evidence": False, "test_panel_opened": False}
    write(output / "e121-report.json", report)
    lines = ["# E121 frozen blocked-pickup waiting", "",
             "Same E118 episode-120 critic, objective, instances, nominal due cutoff, certificates and guard.", "",
             "| Controller | Strict | J | Early excess/job | Late excess/job | Wait before storage | Within ±20 | Work | Early count |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    if comparable:
        for name in ("nominal", "bounded_pickup_wait"):
            m = report[name]
            lines.append(f"| {name} | {strict}/{len(rows)} | {m['objective_cost']:.2f} | "
                         f"{m['mean_release_earliness_beyond_tolerance']:.3f} | "
                         f"{m['mean_release_tardiness_beyond_tolerance']:.3f} | {m['mean_wait_before_storage']:.2f} | "
                         f"{m['within_tolerance_rate']:.3f} | {m['equipment_work']:.1f} | {m['early_deliveries']} |")
    lines.extend(["", f"New waiting selected: {report['new_wait_selected_decisions']}/{report['eligible_decisions']} eligible decisions.",
                  "", "This tests a newly available action with a frozen critic. No retrained performance or new liveness theorem is established.", ""])
    (output / "e121-table.md").write_text("\n".join(lines))
    return report


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
        require(args.seed is not None, "E121 run-one needs --seed")
        run_one(args.output.resolve(), args.seed, args.device)
    elif args.command == "run-panel":
        for seed in p.INSTANCE_SEEDS:
            run_one(args.output.resolve(), seed, args.device)
    print(json.dumps(analyze(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
