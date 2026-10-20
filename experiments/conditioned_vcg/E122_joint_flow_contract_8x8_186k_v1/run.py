"""Gate first, then bounded frozen four-arm replays. No training commands."""

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

from experiments.conditioned_vcg.E121_blocked_pickup_wait_8x8_185k_v1 import run as reference
from . import protocol as p
from .flow import JointVerifier, JointCertificateCache, access, physical_key
from .policy import ProgressOnlyDeferRule
from .preflight import checks

replay = reference.replay
read, write, require = replay.read, replay.write, replay.require


def prepare(output):
    contract = p.contract()
    path = output / "contract.json"
    if path.exists():
        require(read(path) == contract, "E122 source/contract changed; use a new versioned output")
    else:
        require(not output.exists() or not any(output.iterdir()), "E122 refuses nonempty uncontracted output")
        write(path, contract)
    manifest = output / "instances.json"
    if not manifest.exists():
        payload = torch.load(p.SOURCE / "frozen-model.pth", map_location="cpu", weights_only=False)
        p.verify_freshness(payload)
        require(payload["config"]["force_recovery_witness_when_due"] is False
                and payload["config"]["max_nonprogress_recovery_decisions"] == 2,
                "E122 guard configuration differs")
        env = replay.rewards.ReleaseETCostEnv(replay.p.SCENARIO, replay.p.e118.base.e107.e105.e78.SPEC)
        records = [{"panel": panel, "seed": seed, "instance_id": (instance := env.sample_episode_instance(seed)).instance_id,
                    "instance_json": instance.to_json()}
                   for panel, seeds in p.PANELS.items() for seed in seeds]
        write(manifest, {"contract_sha256": contract["contract_sha256"], "freshness_passed": True,
                         "records": records})
    else:
        require(read(manifest)["contract_sha256"] == contract["contract_sha256"], "E122 manifest changed")
    return contract


class Trace(replay.Trace):
    def __init__(self, joint, label):
        super().__init__()
        self.joint, self.label = joint, label
        self.occupancy_sum, self.peak_occupancy = 0, 0

    def decision(self, env, snapshot, selected):
        super().decision(env, snapshot, selected)
        current = access(physical_key(snapshot.recovery_state))
        successor = access(physical_key(selected.candidate.successor_state))
        if self.joint:
            require(current.allowed and successor.allowed, "E122 selected action leaves joint access set")
        count = len(snapshot.recovery_state.blocks)
        self.current.update({
            "stored_occupancy": count, "free_storage": current.free_cells,
            "inflow_access": current.allowed, "inflow_reason": current.reason,
            "successor_inflow_access": successor.allowed,
            "physical_accept_count": snapshot.audit.physical_accept_count,
            "safe_accept_count": snapshot.audit.safe_accept_count,
            "unsafe_accept_count": snapshot.audit.unsafe_accept_count,
            "unknown_accept_count": snapshot.audit.unknown_accept_count,
            "certificate_contract": selected.candidate.certificate.contract,
        })
        if len(self.decisions) % 25 == 0:
            print(f"E122 {self.label} | decision={len(self.decisions)} step={env.time_steps} "
                  f"delivered={sum(b.delivered for b in env.blocks)}/36 stored={count} "
                  f"last={selected.candidate.action_type.value}", flush=True)


class Env(replay.TracedEnv):
    def step(self, action):
        result = super().step(action)
        count = sum(block.stored and not block.delivered for block in self.blocks)
        self.trace.occupancy_sum += count
        self.trace.peak_occupancy = max(self.trace.peak_occupancy, count)
        return result


def row_path(output, panel, arm, seed):
    return output / "rows" / panel / arm / f"instance-{seed}.json"


def run_one(output, panel, arm, seed, device, label="ep=1/1"):
    contract = prepare(output)
    target = row_path(output, panel, arm, seed)
    if target.exists():
        row = read(target)
        require(row["contract_sha256"] == contract["contract_sha256"], "E122 saved row changed")
        return row
    if panel != "parity":
        gate = read(output / "preflight.json")
        require(gate["passed"] and gate["contract_sha256"] == contract["contract_sha256"], "E122 preflight gate closed")
        require(seed in p.PANELS[panel], "E122 coordinate outside predeclared panel")
    config = p.ARMS[arm]
    manifest = read(p.SOURCE / "frozen-source.json")
    payload = torch.load(p.SOURCE / "frozen-model.pth", map_location="cpu", weights_only=False)
    agent = replay.TracedAgent.from_checkpoint(payload, device=device, resumable=False, seed=0)
    agent.Q_local.eval().requires_grad_(False)
    agent.Q_target.eval().requires_grad_(False)
    agent.set_epsilon(0.)
    before, gradients = replay.parameter_digest(agent), agent.gradient_steps
    env = Env(replay.p.SCENARIO, replay.p.e118.base.e107.e105.e78.SPEC)
    env.trace = Trace(config["joint_flow"], f"{label} {arm} instance={seed}")
    env.current_episode = f"E122-{panel}-{arm}-{seed}"
    options = {key: manifest["liveness_rule"][key] for key in ("max_option_steps", "max_consecutive_defer_decisions")}
    rule = ProgressOnlyDeferRule(**options) if config["progress_only"] else reference.BlockedPickupWaitRule(**options)
    verifier = JointVerifier() if config["joint_flow"] else None
    started = perf_counter()
    print(f"E122 starting {label} {panel} {arm} instance={seed} | frozen episode=120 device={device}", flush=True)
    raw, error = None, None
    try:
        with ExitStack() as stack:
            cache_type = JointCertificateCache if verifier else replay.reuse.TimingInvariantCertificateCache
            stack.enter_context(patch.object(replay.audit, "ViabilityCertificateCache", cache_type))
            if verifier:
                stack.enter_context(verifier.enabled())
                stack.enter_context(patch.object(replay.audit, "enumerate_viability_candidates", verifier.enumerate))
            # Native enumeration also uses exact-only certificates, deliberately
            # avoiding a proof shortcut supported only by the unaugmented model.
            stack.enter_context(replay.reuse.path_cleanup_active())
            stack.enter_context(replay.e115run.fast_inputs.enabled())
            stack.enter_context(replay.e13._WallClock(300))
            with torch.no_grad():
                raw = replay.instrumentation.run_episode(
                    agent, env, objective_spec=replay.p.e118.base.e107.e105.e78.SPEC,
                    instance_seed=seed, training=False, max_steps=3600,
                    search_config=replay.ViabilitySearchConfig(**manifest["search_config"]),
                    liveness_rule=rule, epsilon_start=0., epsilon_end=0., epsilon_decay_decisions=1,
                    updates_per_macro=0, epsilon_warmup_decisions=0, run_protocol=p.PROTOCOL)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        replay.e115run.fast_inputs._REGISTRY.clear()
    require(replay.parameter_digest(agent) == before and agent.gradient_steps == gradients and len(agent.replay) == 0,
            "E122 frozen evaluation mutated learner")
    wall = perf_counter() - started
    strict = bool(raw and raw["strict_method_success"] and raw["delivery_count"] == 36
                  and raw["service_metrics"]["terminal_costs_match_dense_accrual"])
    if panel != "parity" and raw:
        planned = next(row for row in read(output / "instances.json")["records"]
                       if row["panel"] == panel and row["seed"] == seed)
        require(raw["episode_instance_id"] == planned["instance_id"], "E122 executed different instance")
    row = {"protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
           "panel": panel, "arm": arm, "instance_seed": seed,
           "status": "complete" if strict else "failed_or_bounded", "strict_complete": strict,
           "raw": raw, "error": error, "decisions": env.trace.decisions, "deliveries": env.trace.deliveries,
           "parameters_unchanged": True, "wall_seconds": wall,
           "seconds_per_decision": wall / len(env.trace.decisions) if env.trace.decisions else None,
           "peak_stored_occupancy": env.trace.peak_occupancy,
           "time_weighted_stored_occupancy": env.trace.occupancy_sum / env.time_steps if env.time_steps else None,
           "inbound_no_physical_accept_decisions": sum(d["inbound_at_pickup"] and not d["physical_accept_count"]
                                                        for d in env.trace.decisions),
           "access_false_boundaries_with_free_capacity": sum(not d["inflow_access"] and d["free_storage"] > 0
                                                             for d in env.trace.decisions),
           "verifier": verifier.stats() if verifier else None, "training_runs": 0, "episode_runs": 1}
    write(target, row)
    shown = f"J={raw['operator_cost']:.2f} wait={raw['service_metrics']['mean_wait_before_storage']:.1f}" if strict else f"failure={error or (raw or {}).get('method_failure_reason')}"
    print(f"E122 {label} {arm} instance={seed} | strict={int(strict)} {shown} "
          f"decisions={len(env.trace.decisions)} elapsed={wall:.1f}s", flush=True)
    return row


def preflight(output, device):
    contract = prepare(output)
    result = checks()
    require(result["passed"], "E122 deterministic contract/liveness fixtures failed")
    row = run_one(output, "parity", "A_outflow_bounded", 112800, device, "parity=1/1")
    expected = read(p.parent.DEFAULT_OUTPUT / "rows/instance-112800.json")
    parity = bool(row["strict_complete"] and
        not replay.parity_differences(replay.parity_fields(row["raw"]), replay.parity_fields(expected["raw"])) and
        reference.attribution.behavior_signature(row["decisions"]) == reference.attribution.behavior_signature(expected["decisions"]))
    result.update({"contract_sha256": contract["contract_sha256"], "reference_E121_exact_parity": parity,
                   "passed": result["passed"] and parity, "episode_runs": 1})
    write(output / "preflight.json", result)
    require(result["passed"], "E122 native reference parity failed; fresh panel remains closed")
    return result


def aggregate(rows):
    if not rows or not all(row["strict_complete"] for row in rows):
        return None
    names = ("objective_cost", "mean_release_earliness_beyond_tolerance", "mean_release_tardiness_beyond_tolerance",
             "mean_wait_before_storage", "within_tolerance_rate", "equipment_work", "equipment_work_per_delivery",
             "physical_rehandles", "release_clock_mae", "storage_clock_mae")
    return {**{name: fmean(row["raw"]["service_metrics"][name] for row in rows) for name in names},
            "steps": fmean(row["raw"]["steps"] for row in rows),
            "early_deliveries": sum(event["release_error"] < -20 for row in rows for event in row["deliveries"]),
            "guard_forced_decisions": sum(event["guard_forced"] for row in rows for event in row["decisions"]),
            "mean_peak_occupancy": fmean(row["peak_stored_occupancy"] for row in rows),
            "time_weighted_occupancy": fmean(row["time_weighted_stored_occupancy"] for row in rows),
            "seconds_per_decision": fmean(row["seconds_per_decision"] for row in rows)}


def analyze(output, panel):
    contract = prepare(output)
    arms, all_rows = {}, {}
    for arm in p.ARMS:
        rows = [read(path) for seed in p.PANELS[panel] if (path := row_path(output, panel, arm, seed)).exists()]
        for row in rows:
            require(row["contract_sha256"] == contract["contract_sha256"], "E122 analyzed row from another contract")
        all_rows[arm] = {row["instance_seed"]: row for row in rows}
        arms[arm] = {"rows": len(rows), "strict_complete_rows": sum(row["strict_complete"] for row in rows),
                     "means": aggregate(rows),
                     "blocked_admission_decisions": sum(row["inbound_no_physical_accept_decisions"] for row in rows),
                     "access_false_boundaries_with_free_capacity": sum(row["access_false_boundaries_with_free_capacity"] for row in rows),
                     "failures": [{"seed": row["instance_seed"], "error": row["error"],
                                   "reason": (row["raw"] or {}).get("method_failure_reason")}
                                  for row in rows if not row["strict_complete"]]}
    paired = []
    reference_arm = all_rows["A_outflow_bounded"]
    for arm, rows in all_rows.items():
        if arm == "A_outflow_bounded":
            continue
        for seed, row in rows.items():
            if seed not in reference_arm:
                continue
            base = reference_arm[seed]
            complete = row["strict_complete"] and base["strict_complete"]
            if complete:
                require(row["raw"]["episode_instance_id"] == base["raw"]["episode_instance_id"], "E122 paired instance mismatch")
            paired.append({"arm": arm, "seed": seed, "both_complete": complete,
                           "delta_J_from_A": row["raw"]["operator_cost"] - base["raw"]["operator_cost"] if complete else None})
    total = sum(item["rows"] for item in arms.values())
    report = {"protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"], "panel": panel,
              "status": "complete" if total == len(p.PANELS[panel]) * 4 else "partial",
              "rows": total, "expected_rows": len(p.PANELS[panel]) * 4, "arms": arms, "paired": paired,
              "all_planned_rows_completed": all(a["strict_complete_rows"] == len(p.PANELS[panel]) for a in arms.values()),
              "training_runs": 0, "episode_runs": 0, "paper_evidence": False, "test_panel_opened": False,
              "quality_limit": "frozen critic was not retrained under changed feasible/waiting frontiers"}
    write(output / f"e122-{panel}-report.json", report)
    lines = [f"# E122 Joint-Flow VCG — {panel}", "",
             "One frozen episode-120 model; fresh matched development instances; no training.", "",
             "| Arm | Complete | J | Early excess/job | Late excess/job | Within ±20 | Outside wait | Work | Steps |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, values in arms.items():
        m = values["means"]
        start = f"| {arm} | {values['strict_complete_rows']}/{len(p.PANELS[panel])} |"
        if m:
            lines.append(start + f" {m['objective_cost']:.2f} | {m['mean_release_earliness_beyond_tolerance']:.3f} | "
                         f"{m['mean_release_tardiness_beyond_tolerance']:.3f} | {m['within_tolerance_rate']:.3f} | "
                         f"{m['mean_wait_before_storage']:.2f} | {m['equipment_work']:.1f} | {m['steps']:.1f} |")
        else:
            lines.append(start + " — | — | — | — | — | — | — |")
    lines.extend(["", "Failures remain failures; quality means are suppressed if an arm has an incomplete trajectory.",
                  "Controller invariance is checked at macro boundaries, not throughout primitive motion.", ""])
    (output / f"e122-{panel}-table.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run-panel", "run-one", "analyze", "status"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--panel", choices=p.PANELS, default="pilot")
    parser.add_argument("--arm", choices=p.ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    output = args.output.resolve()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "prepare":
        result = {"status": "prepared", "contract_sha256": prepare(output)["contract_sha256"]}
    elif args.command == "preflight":
        result = preflight(output, args.device)
    elif args.command == "run-one":
        require(args.arm is not None and args.seed is not None, "E122 run-one requires --arm and --seed")
        run_one(output, args.panel, args.arm, args.seed, args.device)
        result = analyze(output, args.panel)
    elif args.command == "run-panel":
        prepare(output)
        if not (output / "preflight.json").exists():
            preflight(output, args.device)
        total, index = len(p.PANELS[args.panel]) * 4, 0
        for seed in p.PANELS[args.panel]:
            for arm in p.ARMS:
                index += 1
                run_one(output, args.panel, arm, seed, args.device, f"ep={index}/{total}")
        result = analyze(output, args.panel)
    else:
        result = analyze(output, args.panel)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
