"""Frozen three-day pilot: wait-priced static batch with no admission preference mask."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from PSLAP.viability import ViabilityStatus
from experiments.conditioned_vcg.E147_wait_aware_working_day_206k_v1 import model as wait_model
from experiments.conditioned_vcg.E147_wait_aware_working_day_206k_v1 import run as e147
from experiments.conditioned_vcg.E159_lazy_joint_training_218k_v1 import run as e159
from experiments.conditioned_vcg.E164_fixed_admission_storage_clock_223k_v1 import run as e164
from experiments.conditioned_vcg.E168_immediate_admission_objective_pilot_227k_v1 import run as e168
from experiments.conditioned_vcg.E172_capacity_preserving_access_pilot_231k_v1 import run as e172
from experiments.conditioned_vcg.E176_event_progress_guard_235k_v1 import choice, guard, run as e176
from experiments.conditioned_vcg.E177_event_guard_matched_panel_236k_v1 import dynamic

from .objective import WaitPricedStorageEnv, nominal_components


ROOT = e164.ROOT
OUTPUT = ROOT / "results/vcg-conditioned-e180-wait-cost-unmasked-pilot-239k-v1-repair2"
LEVELS = (0.20, 0.50, 0.80)
SEEDS = (24168000, 24168001, 24168002)  # matched E168/E172 development days
CASE_WALL_SECONDS = 900
REFERENCE_WALL_SECONDS = 3600
HORIZON = e168.HORIZON
REFERENCE_MARGIN_STEPS = HORIZON // 5  # 20% headroom, declared before reference


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def implementation():
    """Replace only admission preference and immediate-cost accounting.

    E176 supplies the sole progress guard. E172 supplies outflow + strengthened
    inflow certification. E161/E158 lazy certification and batching remain.
    """
    with ExitStack() as stack:
        stack.enter_context(e176.implementation())
        stack.enter_context(patch.object(e159.proposals, "build", guard.build))
        stack.enter_context(patch.object(e159, "choose", choice.choose))
        stack.enter_context(patch.object(wait_model, "nominal_components", nominal_components))
        yield


def prepare(output=OUTPUT):
    output = Path(output)
    if not e147.SOURCE_PATH.is_file():
        raise RuntimeError("E180 frozen source checkpoint is missing")
    cases = []
    for rho, seed in zip(LEVELS, SEEDS):
        value, path = e164.stored_case(
            output, seed=seed, rho=rho,
            name=f"development-rho-{rho:g}-seed-{seed}",
        )
        cases.append({
            "rho": rho, "seed": seed, "incoming_count": value["incoming_count"],
            "path": str(path.resolve()), "sha256": sha(path),
        })
    sources = (
        Path(__file__), Path(__file__).with_name("objective.py"),
        Path(__file__).with_name("README.txt"), Path(__file__).with_name("run.sh"),
        ROOT / "tests/test_vcg_e180_wait_cost_unmasked.py",
    )
    value = {
        "protocol": "e180_wait_priced_unmasked_static_batch_frozen_pilot_v3",
        "source_checkpoint": str(e147.SOURCE_PATH.resolve()),
        "source_checkpoint_sha256": sha(e147.SOURCE_PATH),
        "cases": cases,
        "availability": "all blocks released at zero; FIFO admission order",
        "due_rule": "stored_time_step + storage_steps_needed",
        "objective": "J=2T_storage+E_storage+0.5W_release_to_admission+0.1L+10R",
        "action_frontier": "all physical Accept/Deliver/Reconfigure/event-Defer actions; no Accept-only or inbound-Defer mask",
        "certification": e172.CONTRACT,
        "guard": "E176 event-progress bounded Reconfigure with one-step witness fallback",
        "ranking": "frozen E147 residual critic; no training updates",
        "per_case_wall_seconds": CASE_WALL_SECONDS,
        "reference_wall_seconds": REFERENCE_WALL_SECONDS,
        "primitive_horizon": HORIZON,
        "heavy_reference_required": "certified Dynamic PSLAP on the same 28-job day",
        "heavy_reference_minimum_headroom_steps": REFERENCE_MARGIN_STEPS,
        "access_invariant_caveat": (
            "all free cells must remain pickup-reachable; stronger than one-cell access, "
            "and may exclude capacity that this day's jobs would never use"
        ),
        "test_panel_opened": False,
        "paper_evidence": False,
        "sources": {str(path.relative_to(ROOT)): sha(path) for path in sources},
    }
    value["contract_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()
    destination = output / "contract.json"
    if destination.exists():
        if e147.read(destination) != value:
            raise RuntimeError("E180 contract changed; use a fresh output directory")
    else:
        e147.write(destination, value)
    return value


def preflight(output, contract):
    if sha(contract["source_checkpoint"]) != contract["source_checkpoint_sha256"]:
        raise RuntimeError("frozen checkpoint bytes changed")
    counts = []
    for case in contract["cases"]:
        if sha(case["path"]) != case["sha256"]:
            raise RuntimeError("development case bytes changed")
        value = e147.read(case["path"])
        if set(value["instance"]["arrival_steps"]) != {0}:
            raise RuntimeError("case has nonzero release time")
        counts.append(case["incoming_count"])
    if counts != [7, 18, 28]:
        raise RuntimeError(f"unexpected case counts: {counts}")
    with implementation():
        if e159.proposals.build is not guard.build or e159.choose is not choice.choose:
            raise RuntimeError("old admission or guard action path is still active")
        if wait_model.nominal_components is not nominal_components:
            raise RuntimeError("wait-priced nominal cost is not active")
    result = {"status": "passed", "cases": counts, "test_panel_opened": False}
    e147.write(Path(output) / "preflight.json", result)
    return result


def run_one(output, contract, case, device, *, method="frozen"):
    if method not in ("frozen", "certified_dynamic"):
        raise ValueError("unknown E180 method")
    destination = Path(output) / "rows" / method / f"rho-{case['rho']:g}-seed-{case['seed']}.json"
    if destination.exists():
        old = e147.read(destination)
        if old["contract_sha256"] != contract["contract_sha256"]:
            raise RuntimeError("E180 row belongs to a different contract")
        if old["status"] == "complete":
            return old
    value = e147.read(case["path"])
    value.pop("e164_case_sha256", None)
    env = WaitPricedStorageEnv(value)
    e147.seed_everything(0)
    agent = e164._clean_agent(device)
    before_updates = agent.gradient_steps
    print(f"E180 {method} rho={case['rho']:.0%} jobs={len(env.blocks)}", flush=True)

    def progress(row):
        if row["decisions"] % 5 == 0:
            print(
                f"E180 rho={case['rho']:.0%} decision={row['decisions']} "
                f"step={row['steps']} delivered={row['delivered']}/{row['required']} "
                f"peak={env.peak_stored}", flush=True,
            )

    try:
        wall_seconds = (
            REFERENCE_WALL_SECONDS if method == "certified_dynamic"
            else CASE_WALL_SECONDS
        )
        with implementation(), (
            patch.object(e159, "choose", dynamic.choose)
            if method == "certified_dynamic" else ExitStack()
        ), e147.cache_session(), torch.no_grad(), e147.wall_limit(wall_seconds):
            episode = e159.episode(agent, env, training=False, emit=progress)
        if not episode["strict_complete"]:
            raise RuntimeError("E180 did not complete every delivery")
        if agent.gradient_steps != before_updates:
            raise RuntimeError("E180 frozen pilot performed learning")
        metrics = episode["metrics"]
        unknown_cached = sum(
            certificate.status is ViabilityStatus.UNKNOWN
            for certificate in agent.lazy_certificate_cache.values()
        )
        result = {
            "status": "complete", "method": method,
            "rho": case["rho"], "seed": case["seed"],
            "contract_sha256": contract["contract_sha256"],
            "metrics": metrics,
            "J_without_wait": metrics["J"] - 0.5 * metrics["admission_wait"],
            "day_statistics": episode["day_statistics"],
            "decisions": episode["decisions"],
            "wall_seconds": episode["wall_seconds"],
            "certification": episode["certification"],
            "action_counts": dict(agent.action_decisions),
            "guard_witness_steps": getattr(agent, "e176_witness_steps", 0),
            "unknown_selector_queries": int(getattr(agent, "lazy_unknowns", 0)),
            "unknown_cached_successor_states": int(unknown_cached),
            "horizon_headroom_steps": HORIZON - int(env.time_steps),
            "access_invariant_caveat": contract["access_invariant_caveat"],
            "test_panel_opened": False,
        }
    except (Exception, KeyboardInterrupt) as error:
        result = {
            "status": "failed", "method": method,
            "rho": case["rho"], "seed": case["seed"],
            "contract_sha256": contract["contract_sha256"],
            "error": f"{type(error).__name__}: {error}",
            "steps": int(env.time_steps),
            "delivered": sum(block.delivered for block in env.blocks),
            "day_statistics": env.day_metrics(),
            "test_panel_opened": False,
        }
        e147.write(destination, result)
        raise
    e147.write(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "reference", "pilot"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rho", type=float, choices=LEVELS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract = prepare(args.output)
    if args.command == "prepare":
        result = {"status": "prepared", "contract": str(args.output / "contract.json")}
    elif args.command == "preflight":
        result = preflight(args.output, contract)
    elif args.command == "reference":
        if not (Path(args.output) / "preflight.json").is_file():
            raise RuntimeError("E180 reference requires preflight")
        heavy = contract["cases"][-1]
        result = run_one(args.output, contract, heavy, args.device, method="certified_dynamic")
        if result["horizon_headroom_steps"] < REFERENCE_MARGIN_STEPS:
            raise RuntimeError(
                "certified Dynamic lacks declared 20% horizon headroom; "
                "do not run the frozen pilot with this horizon"
            )
    else:
        if not (Path(args.output) / "preflight.json").is_file():
            raise RuntimeError("E180 pilot requires preflight")
        reference_path = Path(args.output) / "rows/certified_dynamic" / (
            f"rho-0.8-seed-{SEEDS[-1]}.json"
        )
        if not reference_path.is_file():
            raise RuntimeError("E180 pilot requires the certified-Dynamic heavy reference")
        reference = e147.read(reference_path)
        if (reference["contract_sha256"] != contract["contract_sha256"]
                or reference["status"] != "complete"
                or reference["horizon_headroom_steps"] < REFERENCE_MARGIN_STEPS):
            raise RuntimeError("E180 heavy reference failed the declared horizon gate")
        selected = [
            case for case in contract["cases"]
            if args.rho is None or case["rho"] == args.rho
        ]
        rows = [run_one(args.output, contract, case, args.device) for case in selected]
        result = {
            "status": "complete" if all(row["status"] == "complete" for row in rows) else "partial",
            "rows": rows, "test_panel_opened": False,
        }
        e147.write(Path(args.output) / "pilot-summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
