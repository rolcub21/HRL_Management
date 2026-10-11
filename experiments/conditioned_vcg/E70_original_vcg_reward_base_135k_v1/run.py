"""Base-80 reward pilot with the original target-size VCG architecture."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import train_vcg_dense_proper as dense
from experiments.conditioned_vcg.target_size_dense_stage_a_v2 import run as v2run
from experiments.conditioned_vcg.target_size_dense_stage_a_v4 import run as v4run

from . import protocol as p


class E70Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E70Error(message)


def _file(output: Path, name: str) -> Path:
    return output.resolve() / name


def _read(path: Path) -> dict:
    require(path.is_file(), f"missing E70 file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E70 JSON: {path}")
    return value


def authenticate(output: Path) -> dict:
    observed = _read(_file(output, "contract.json"))
    expected = p.contract()
    require(observed == expected, "E70 contract, source, or frozen control changed")
    return expected


def prepare(output: Path) -> dict:
    output = output.resolve()
    contract = p.contract()
    file = _file(output, "contract.json")
    if file.is_file():
        authenticate(output)
    else:
        require(not output.exists() or not any(output.iterdir()),
                "E70 refuses a nonempty uncontracted output")
        v2run.e13._atomic_json(file, contract)
    return {"status": "prepared", "contract_sha256": contract["contract_sha256"],
            "scenario": p.SCENARIO, "model_seed": 0,
            "base40_control_checkpoints": [6, 12, 18, 24, 30],
            "training_runs": 0, "episode_runs": 0}


def self_check(output: Path) -> dict:
    contract = authenticate(output)
    require(p.SPEC.to_dict()["dense_b"] == 80.0, "E70 base-80 spec changed")
    require(dense.FROZEN_OBJECTIVE_SPEC.to_dict()["dense_b"] == 40.0,
            "E70 source objective is no longer base 40")
    original = dense.FROZEN_OBJECTIVE_SPEC
    amended = p.SPEC
    for error in (0, 20, 25, 45, 100, 500):
        old = original.dense_b - original.lambda_abs * error - original.lambda_outside * max(0, error - original.window)
        new = amended.dense_b - amended.lambda_abs * error - amended.lambda_outside * max(0, error - amended.window)
        require(abs((new - old) - 40.0) < 1e-9,
                "E70 changed more than the per-delivery constant")
    env = p.TargetDenseEnv(p.SCENARIO, p.SPEC)
    instance = env.sample_episode_instance(p.VALIDATION_SEEDS[p.SCENARIO][0])
    env.reset(instance=instance)
    require(env.time_steps == 0 and len(env.blocks) == 36
            and all(not block.stored for block in env.blocks),
            "E70 empty-start environment changed")
    return {"status": "passed", "contract_sha256": contract["contract_sha256"],
            "objective": p.SPEC.to_dict(), "gamma": 0.99,
            "same_workloads_as_base40": True, "episode_runs": 0, "training_runs": 0}


def train(output: Path, *, through_episode: int, device: str) -> dict:
    contract = authenticate(output)
    require(1 <= through_episode <= 112, "through-episode must be 1..112")
    require(device in ("cpu", "cuda"), "device must be cpu or cuda")
    args = SimpleNamespace(output=output.resolve(), scenario=p.SCENARIO,
                           model_seed=0, device=device,
                           stop_after_episode=through_episode,
                           allow_cpu_pilot=False)
    with ExitStack() as stack:
        stack.enter_context(patch.object(v4run, "p", p))
        stack.enter_context(patch.object(dense, "FROZEN_OBJECTIVE_SPEC", p.SPEC))
        if device == "cpu" and through_episode > 2:
            # The v2 wrapper's >2-episode CPU guard is operational only. Pass
            # its auto hardware gate without changing dense.main's real Torch
            # module; dense.main resolves auto to the actual CPU device.
            cpu_gate = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
            stack.enter_context(patch.object(v2run, "torch", cpu_gate))
            args.device = "auto"
        result = v4run.train(args)
    return {**result, "protocol": p.PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "objective": p.SPEC.to_dict(), "requested_device": device,
            "through_episode": through_episode,
            "paper_evidence": False}


def _row(record: dict, *, delivery_base: float) -> dict:
    rows = p._row(record)["runs"]
    require([run["instance_seed"] for run in rows]
            == list(p.VALIDATION_SEEDS[p.SCENARIO]),
            "E70 validation instances are not paired")
    strict = all(run["strict_method_success"] and run["delivery_count"] == 36
                 for run in rows)
    return {
        "checkpoint_episode": int(record["checkpoint_episode"]),
        "strict_complete": sum(run["strict_method_success"] for run in rows),
        "episodes": len(rows),
        "base40_equivalent_return": (sum(run["dense_rescored_return"]
                                           - 36 * (delivery_base - 40.0)
                                           for run in rows) / len(rows)
                                     if strict else None),
        "placement_mae": (sum(run["mean_absolute_error"] for run in rows) / len(rows)
                          if strict else None),
        "within_20_rate": (sum(run["within_target_window_rate"] for run in rows) / len(rows)
                           if strict else None),
        "steps": sum(run["steps"] for run in rows) / len(rows),
        "rehandles_per_100": (100 * sum(run["relocations"] for run in rows)
                              / sum(run["delivery_count"] for run in rows)
                              if strict else None),
        "instances": [run["episode_instance_id"] for run in rows],
    }


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    file = (output.resolve() / "training/operational" / p.SCENARIO
            / "seed-0/training-summary.json")
    summary = _read(file)
    require(summary.get("model_seed") == 0
            and summary.get("scenario") == p.SCENARIO
            and summary.get("training_protocol") == p.PROTOCOL
            and summary.get("objective_spec") == p.SPEC.to_dict()
            and summary.get("target_contract_sha256") == contract["contract_sha256"],
            "E70 training summary identity differs")
    baseline = {row["checkpoint_episode"]: row
                for row in contract["base40_control"]["rows"]}
    matched = []
    for record in summary["validation_history"]:
        episode = int(record["checkpoint_episode"])
        if episode not in baseline:
            continue
        control = _row(baseline[episode], delivery_base=40.0)
        pilot = _row(record, delivery_base=80.0)
        require(control["instances"] == pilot["instances"],
                f"E70 episode {episode} workload mismatch")
        comparable = (control["strict_complete"] == pilot["strict_complete"] == 5)
        matched.append({"checkpoint_episode": episode,
                        "control_base40": control, "pilot_base80": pilot,
                        "base80_minus_base40": ({key: pilot[key] - control[key]
                                                  for key in ("base40_equivalent_return",
                                                              "placement_mae", "within_20_rate",
                                                              "steps", "rehandles_per_100")}
                                                 if comparable else None)})
    require(matched, "E70 has no matched validation checkpoint")
    report = {
        "status": "complete" if summary["completed_training_episodes"] >= 30 else "partial",
        "protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"],
        "scope": "development-only matched reward-base test, original VCG architecture",
        "control_source": contract["base40_control"]["path"],
        "control_snapshot_sha256": contract["base40_control"]["first_30_episode_snapshot_sha256"],
        "completed_training_episodes": summary["completed_training_episodes"],
        "matched_checkpoints": matched,
        "comparison_basis": "base-80 undiscounted return minus 40 per completed delivery; strict-complete rows only",
        "caveat": "single training seed and reused development instances; not a sealed test or paper result",
        "paper_evidence": False,
    }
    target = _file(output, "e70-report.json")
    v2run.e13._atomic_json(target, report)
    return {"status": report["status"], "report": str(target),
            "matched_checkpoints": [row["checkpoint_episode"] for row in matched],
            "training_runs": 0, "episode_runs": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "self-check", "train", "analyze"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--through-episode", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "train":
        result = train(args.output, through_episode=args.through_episode,
                       device=args.device)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
