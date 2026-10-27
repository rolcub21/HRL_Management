"""Fast static-batch training with fixed admission and storage-clock timing."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import torch

from experiments.conditioned_vcg.E147_wait_aware_working_day_206k_v1 import model as wait_model
from experiments.conditioned_vcg.E147_wait_aware_working_day_206k_v1 import run as e147
from experiments.conditioned_vcg.E152_event_defer_working_day_211k_v1 import option as e152_option
from experiments.conditioned_vcg.E152_event_defer_working_day_211k_v1 import policy as e152_policy
from experiments.conditioned_vcg.E153_reconfigure_mask_event_defer_212k_v1 import guard as e153_guard
from experiments.conditioned_vcg.E159_lazy_joint_training_218k_v1 import agent as e159_agent
from experiments.conditioned_vcg.E159_lazy_joint_training_218k_v1 import proposals as e159_proposals
from experiments.conditioned_vcg.E159_lazy_joint_training_218k_v1 import run as e159
from experiments.conditioned_vcg.E161_batched_lazy_targets_220k_v1.agent import BatchedLazyJointAgent
from experiments.conditioned_vcg.E161_batched_lazy_targets_220k_v1 import run as e161
from experiments.conditioned_vcg.E162_certified_guard_mask_221k_v1 import frontier as e162_frontier
from experiments.conditioned_vcg.E162_certified_guard_mask_221k_v1 import run as e162

from . import frontier
from .objective import FixedAdmissionStorageEnv, nominal_components, storage_due, stored_window_open


ROOT = e147.ROOT
OUTPUT = ROOT / "results/vcg-conditioned-e164-fixed-admission-storage-clock-223k-v1-repair3"
CAPACITY = 35
MEAN_DURATION = 300
HORIZON = 1200
RHO_LOW = 0.10
RHO_HIGH = 0.80
MAXIMUM_EPISODE = 12
VALIDATION_ENDPOINTS = (3, 6, 12)
DEV_LEVELS = (0.20, 0.50, 0.80)
DEV_SEEDS = (24164000, 24164001, 24164002)
TRAIN_BASE = 24165000
BATCH_SIZE = 32
UPDATE_EVERY = 2
TRAINING_WALL_SECONDS = 1800
VALIDATION_WALL_SECONDS = 600


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def jobs_for_workload(rho):
    value = float(rho)
    if not RHO_LOW <= value <= RHO_HIGH:
        raise ValueError("static workload is outside [0.10,0.80]")
    return max(1, int(value * CAPACITY + 0.5))


def workload_rho(episode):
    number = int(episode)
    if not 1 <= number <= MAXIMUM_EPISODE:
        raise ValueError("episode is outside the E164 budget")
    band = (number - 1) % 3
    width = (RHO_HIGH - RHO_LOW) / 3.0
    low = RHO_LOW + band * width
    high = RHO_LOW + (band + 1) * width
    return random.Random(16400000 + number).uniform(low, high)


def static_case(seed, rho):
    count = jobs_for_workload(rho)
    template = e147.source.discovery.source.original.new_env("A_outflow_bounded")
    native = template.sample_episode_instance(int(seed))
    durations = np.maximum(
        1,
        np.random.default_rng(int(seed) + 1641).poisson(MEAN_DURATION, count),
    ).tolist()
    instance = replace(
        native,
        arrival_rate=1.0,
        proc_mean=float(MEAN_DURATION),
        arrival_steps=tuple(0 for _ in range(count)),
        storage_steps_needed=tuple(int(value) for value in durations),
    )
    value = {
        "instance": instance.to_dict(),
        "initial_positions": [],
        # GridEnv authenticates this field, but E164 never uses it as a due
        # date.  Dynamic storage-clock dates are created on admission.
        "external_due_dates": {
            f"B{index + 1}": int(duration)
            for index, duration in enumerate(durations)
        },
        "seed": int(seed),
        "sampled_workload": float(rho),
        "incoming_count": count,
        "capacity": CAPACITY,
        "jobs_per_capacity": count / CAPACITY,
        "mean_storage_duration": MEAN_DURATION,
        "admission_order": [f"B{index + 1}" for index in range(count)],
        "release_rule": "all jobs available at primitive step zero",
        "due_rule": "stored_time_step + storage_steps_needed",
    }
    return json.loads(json.dumps(value))


def stored_case(output, *, seed, rho, name):
    path = Path(output) / "instances" / f"{name}.json"
    expected = static_case(seed, rho)
    if path.exists():
        observed = e147.read(path)
        digest = observed.pop("e164_case_sha256", None)
        if digest != case_digest(observed) or observed != expected:
            raise RuntimeError("stored E164 static batch changed")
        observed["e164_case_sha256"] = digest
        return observed, path
    value = {**expected, "e164_case_sha256": case_digest(expected)}
    e147.write(path, value)
    return value, path


def _event_signature(env):
    now = int(env.time_steps)
    stored = frozenset(
        str(block.label)
        for block in env.blocks
        if block.stored_time_step is not None and not block.delivered
    )
    opened = frozenset(
        str(block.label)
        for block in env.blocks
        if block.stored_time_step is not None
        and not block.delivered
        and storage_due(block) - 20 <= now
    )
    return stored, opened


@contextmanager
def implementation():
    # E162 retains E161's accepted batched targets and certification-ordered
    # liveness repair.  E164 changes task time, cost, event semantics and the
    # physical frontier's admission priority only.
    with ExitStack() as stack:
        stack.enter_context(e162.implementation())
        stack.enter_context(patch.object(e159.proposals, "build", frontier.build))
        stack.enter_context(patch.object(wait_model, "nominal_components", nominal_components))
        stack.enter_context(patch.object(wait_model, "observed_arrivals", lambda _env: (0.0, 0.0, 0.0)))
        stack.enter_context(patch.object(e152_policy, "next_window_open", stored_window_open))
        stack.enter_context(patch.object(e152_option, "next_window_open", stored_window_open))
        stack.enter_context(patch.object(e153_guard, "next_window_open", stored_window_open))
        stack.enter_context(patch.object(e153_guard, "observable_event_signature", _event_signature))
        stack.enter_context(patch.object(e147, "HORIZON", HORIZON))
        stack.enter_context(patch.object(e152_policy, "HORIZON", HORIZON))
        stack.enter_context(patch.object(e159_proposals, "EPISODE_HORIZON", HORIZON))
        yield


def _initialize_counters(agent):
    e162._initialize_counters(agent)
    for name, value in (
        ("e164_accept_priority_epochs", 0),
        ("e164_accept_certificate_queries", 0),
        ("e164_accept_certificate_hits", 0),
        ("e164_accept_certificate_seconds", 0.0),
    ):
        if not hasattr(agent, name):
            setattr(agent, name, value)


def _clean_agent(device="cpu", *, batch_size=BATCH_SIZE):
    payload = deepcopy(torch.load(e147.SOURCE_PATH, map_location="cpu", weights_only=False))
    payload["config"] = dict(payload["config"])
    payload["config"]["batch_size"] = int(batch_size)
    payload["config"]["update_every"] = UPDATE_EVERY
    base = e159_agent.initialize(payload, device=device, horizon=HORIZON)
    seed_payload = base.checkpoint(
        include_replay=True,
        completed_adaptation_episodes=0,
        training_history=[],
        validation_history=[],
        global_rng_state=e147.audit._capture_global_rng_state(),
        model_seed=0,
        e159_lazy_counters=base.lazy_counters(),
        test_panel_opened=False,
    )
    agent = BatchedLazyJointAgent.from_checkpoint(seed_payload, device=device, resumable=True)
    e147.audit._optimizer_to(agent.optimizer, torch.device(device))
    _initialize_counters(agent)
    if agent.transition_count or agent.gradient_steps or len(agent.replay):
        raise RuntimeError("E164 must start with fresh learning state")
    return agent


def prepare(output):
    output = Path(output)
    if not e147.SOURCE_PATH.is_file():
        raise RuntimeError("E164 residual warm-start checkpoint is missing")
    acceptance = e147.read(e161.OUTPUT / "acceptance.json")
    if acceptance.get("status") != "passed":
        raise RuntimeError("E164 requires E161's passed batching acceptance")
    development = []
    for rho, seed in zip(DEV_LEVELS, DEV_SEEDS):
        value, path = stored_case(
            output, seed=seed, rho=rho, name=f"development-rho-{rho:g}-seed-{seed}"
        )
        development.append({
            "rho": rho,
            "seed": seed,
            "incoming_count": value["incoming_count"],
            "path": str(path.resolve()),
            "sha256": sha(path),
        })
    paths = (
        Path(__file__),
        Path(__file__).with_name("objective.py"),
        Path(__file__).with_name("frontier.py"),
        Path(__file__).with_name("README.txt"),
        ROOT / "tests/test_vcg_e164_fixed_admission.py",
    )
    value = {
        "protocol": "e164_fixed_order_immediate_admission_storage_clock_v1",
        "source_checkpoint": str(e147.SOURCE_PATH.resolve()),
        "source_checkpoint_sha256": sha(e147.SOURCE_PATH),
        "geometry": "original 8x8; empty yard; 35 storage cells",
        "workload": "N=round(rho*35), rho sampled continuously in balanced thirds",
        "training_workload_support": [RHO_LOW, RHO_HIGH],
        "release_rule": "all jobs available at primitive step zero",
        "admission_rule": "fixed B1..BN order; certified Accept mandatory whenever available",
        "controller_scope": "choose safe placement, reconfiguration and retrieval only",
        "due_rule": "d_i=stored_time_step_i+storage_steps_needed_i",
        "duration": "positive Poisson mean 300",
        "objective": "J=2T_storage+E_storage+0.1L+10R; admission wait excluded",
        "arrival_features": "constant zero; no arrival process or workload label",
        "primitive_horizon": HORIZON,
        "batch_size": BATCH_SIZE,
        "update_every": UPDATE_EVERY,
        "development_cases": development,
        "validation_endpoints": list(VALIDATION_ENDPOINTS),
        "validation_scope": "three static batches only; no working-day/full-day evaluation",
        "maximum_authorized_endpoint": MAXIMUM_EPISODE,
        "target_implementation": "E161 accepted exact batched lazy targets",
        "certification": "E162 joint-flow boundary check and certified guard ordering",
        "test_panel_opened": False,
        "deployment_checkpoint_eligible": False,
        "paper_evidence": False,
        "sources": {str(path.relative_to(ROOT)): sha(path) for path in paths},
    }
    value["contract_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()
    destination = output / "contract.json"
    if destination.exists() and e147.read(destination) != value:
        raise RuntimeError("E164 contract changed; use a fresh output")
    if not destination.exists():
        e147.write(destination, value)
    return value


def preflight(output, contract):
    cases = [e147.read(item["path"]) for item in contract["development_cases"]]
    counts = [int(item["incoming_count"]) for item in cases]
    if counts != [7, 18, 28]:
        raise RuntimeError(f"E164 development counts differ: {counts}")
    if any(set(item["instance"]["arrival_steps"]) != {0} for item in cases):
        raise RuntimeError("E164 contains a nonzero release")
    e147.seed_everything(0)
    agent = _clean_agent("cpu")
    value = {
        "status": "passed",
        "contract_sha256": contract["contract_sha256"],
        "development_counts": counts,
        "all_releases_at_zero": True,
        "fresh_replay": len(agent.replay) == 0,
        "batch_size": agent.config.batch_size,
        "update_every": agent.config.update_every,
        "horizon": HORIZON,
        "test_panel_opened": False,
    }
    e147.write(Path(output) / "preflight.json", value)
    return value


def smoke(output, contract):
    if e147.read(Path(output) / "preflight.json").get("status") != "passed":
        raise RuntimeError("E164 smoke requires passed preflight")
    e147.seed_everything(0)
    agent = _clean_agent("cpu", batch_size=2)
    value = static_case(24164999, RHO_LOW)
    env = FixedAdmissionStorageEnv(value)
    with implementation(), e147.cache_session(), e147.wall_limit(300):
        row = e159.episode(agent, env, training=True)
    if not row["strict_complete"] or agent.gradient_steps < 1:
        raise RuntimeError("E164 smoke did not complete learning")
    if tuple(env.observed_admission_labels) != env.expected_admission_labels:
        raise RuntimeError("E164 smoke violated fixed admission order")
    result = {
        "status": "passed",
        "jobs": len(env.blocks),
        "steps": int(env.time_steps),
        "decisions": int(row["decisions"]),
        "gradient_steps": int(agent.gradient_steps),
        "J": float(row["metrics"]["J"]),
        "peak_stored": int(env.peak_stored),
        "fixed_admission_order": True,
        "test_panel_opened": False,
    }
    e147.write(Path(output) / "smoke.json", result)
    return result


def _counters(agent):
    return {
        "guard": e162_frontier.counters(agent),
        "admission": frontier.counters(agent),
        "lazy_targets": agent.lazy_counters(),
        "batched_targets": agent.batch_counters(),
    }


def save(output, contract, agent, history, validations):
    directory = Path(output) / "seed-0"
    directory.mkdir(parents=True, exist_ok=True)
    payload = agent.checkpoint(
        include_replay=True,
        e147_contract_sha256=contract["contract_sha256"],
        completed_adaptation_episodes=len(history),
        training_history=history,
        validation_history=validations,
        global_rng_state=e147.audit._capture_global_rng_state(),
        model_seed=0,
        e159_lazy_counters=agent.lazy_counters(),
        e161_batch_counters=agent.batch_counters(),
        e162_guard_counters=e162_frontier.counters(agent),
        e164_counters=frontier.counters(agent),
        e164_fixed_admission_storage_clock=True,
        test_panel_opened=False,
    )
    temporary = directory / "latest.pth.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, directory / "latest.pth")
    e147.write(directory / "summary.json", {
        "completed_adaptation_episodes": len(history),
        "maximum_authorized_endpoint": MAXIMUM_EPISODE,
        "training_decisions": int(agent.decision_count),
        "gradient_steps": int(agent.gradient_steps),
        "training_jobs": [row["incoming_count"] for row in history],
        "validation_endpoints": [row["episode"] for row in validations],
        **_counters(agent),
        "test_panel_opened": False,
        "deployment_checkpoint_eligible": False,
        "paper_evidence": False,
    })


def _restore_counters(agent, payload):
    _initialize_counters(agent)
    for name, value in payload.get("e162_guard_counters", {}).items():
        attribute = f"e162_{name}"
        if hasattr(agent, attribute):
            setattr(agent, attribute, value)
    mapping = {
        "accept_priority_epochs": "e164_accept_priority_epochs",
        "accept_certificate_queries": "e164_accept_certificate_queries",
        "accept_certificate_hits": "e164_accept_certificate_hits",
        "accept_certificate_seconds": "e164_accept_certificate_seconds",
    }
    for key, attribute in mapping.items():
        setattr(agent, attribute, payload.get("e164_counters", {}).get(key, 0))


def checkpoint_copy(output, contract, agent, endpoint):
    destination = Path(output) / "seed-0/checkpoints" / f"episode-{endpoint}.pth"
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = agent.checkpoint(
        include_replay=False,
        e147_contract_sha256=contract["contract_sha256"],
        completed_adaptation_episodes=int(endpoint),
        model_seed=0,
        e159_lazy_counters=agent.lazy_counters(),
        e161_batch_counters=agent.batch_counters(),
        e162_guard_counters=e162_frontier.counters(agent),
        e164_counters=frontier.counters(agent),
        e164_fixed_admission_storage_clock=True,
        test_panel_opened=False,
    )
    temporary = destination.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def validate(output, contract, agent, endpoint):
    rng = e147.audit._capture_global_rng_state()
    try:
        payload = agent.checkpoint(
            include_replay=False,
            e159_lazy_counters=agent.lazy_counters(),
            e161_batch_counters=agent.batch_counters(),
        )
        evaluator = BatchedLazyJointAgent.from_checkpoint(
            payload, device=agent.device, resumable=False
        )
        _initialize_counters(evaluator)
        evaluator.Q_local.eval().requires_grad_(False)
        evaluator.Q_target.eval().requires_grad_(False)
        rows = []
        with torch.no_grad():
            for index, case in enumerate(contract["development_cases"]):
                destination = Path(output) / "validation" / f"episode-{endpoint}-rho-{case['rho']:g}.json"
                if destination.exists():
                    rows.append(e147.read(destination))
                    continue
                if sha(case["path"]) != case["sha256"]:
                    raise RuntimeError("E164 development instance changed")
                value = e147.read(case["path"])
                value.pop("e164_case_sha256", None)
                env = FixedAdmissionStorageEnv(value)
                print(
                    f"E164 greedy episode={endpoint} batch={index + 1}/3 "
                    f"rho={case['rho']:.0%} jobs={case['incoming_count']}",
                    flush=True,
                )
                with e147.wall_limit(VALIDATION_WALL_SECONDS):
                    row = e159.episode(evaluator, env, training=False)
                row.update(
                    episode=int(endpoint), rho=float(case["rho"]),
                    instance_seed=int(case["seed"]), incoming_count=int(case["incoming_count"]),
                )
                e147.write(destination, row)
                rows.append(row)
        return {"episode": int(endpoint), "rows": rows}
    finally:
        e147.audit._restore_global_rng_state(rng)


def train(output, contract, through, device, retry_failed):
    through = int(through)
    if not 1 <= through <= MAXIMUM_EPISODE:
        raise ValueError("E164 endpoint must be in 1..12")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E164 training requires CUDA")
    if e147.read(Path(output) / "preflight.json").get("status") != "passed" or e147.read(
        Path(output) / "smoke.json"
    ).get("status") != "passed":
        raise RuntimeError("E164 training requires passed preflight and smoke")
    output = Path(output)
    checkpoint = output / "seed-0/latest.pth"
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("e147_contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError("E164 resume contract differs")
        agent = BatchedLazyJointAgent.from_checkpoint(payload, device=device, resumable=True)
        e147.audit._optimizer_to(agent.optimizer, torch.device(device))
        history = list(payload["training_history"])
        validations = list(payload["validation_history"])
        e147.audit._restore_global_rng_state(payload["global_rng_state"])
        _restore_counters(agent, payload)
    else:
        e147.seed_everything(0)
        agent = _clean_agent(device)
        history, validations = [], []
        save(output, contract, agent, history, validations)
    if len(history) > through:
        raise ValueError("requested E164 endpoint precedes the checkpoint")
    stopped = output / "stopped.json"
    if stopped.exists():
        if not retry_failed:
            raise RuntimeError("E164 retained a previous stop; inspect it first")
        os.replace(stopped, output / f"acknowledged-stop-{time.time_ns()}.json")
    number = len(history)
    try:
        with implementation(), e147.cache_session():
            for number in range(len(history) + 1, through + 1):
                rho = workload_rho(number)
                seed = TRAIN_BASE + number
                value, _ = stored_case(
                    output, seed=seed, rho=rho,
                    name=f"training-episode-{number}-seed-{seed}",
                )
                value = dict(value)
                value.pop("e164_case_sha256", None)
                env = FixedAdmissionStorageEnv(value)
                print(
                    f"E164 train episode={number}/12 rho={rho:.3f} "
                    f"jobs={len(env.blocks)} decisions={agent.decision_count}", flush=True,
                )
                last = time.monotonic()

                def progress(row):
                    nonlocal last
                    if time.monotonic() - last >= 30:
                        print(
                            f"E164 episode={number} step={row['steps']} "
                            f"admitted={sum(b.stored_time_step is not None for b in env.blocks)}/{len(env.blocks)} "
                            f"delivered={row['delivered']}/{row['required']}", flush=True,
                        )
                        e147.write(output / "progress.json", {"episode": number, "rho": rho, **row})
                        last = time.monotonic()

                with e147.wall_limit(TRAINING_WALL_SECONDS):
                    row = e159.episode(agent, env, training=True, emit=progress)
                if not row["strict_complete"]:
                    raise RuntimeError("E164 training batch did not complete")
                if tuple(env.observed_admission_labels) != env.expected_admission_labels:
                    raise RuntimeError("E164 fixed admission order was not preserved")
                row.update(
                    episode=number, rho=rho, instance_seed=seed,
                    incoming_count=len(env.blocks), peak_stored=env.peak_stored,
                )
                history.append(row)
                save(output, contract, agent, history, validations)
                print(
                    f"E164 episode={number} complete J={row['metrics']['J']:.1f} "
                    f"steps={env.time_steps} peak={env.peak_stored} wall={row['wall_seconds']:.1f}s",
                    flush=True,
                )
                if number in VALIDATION_ENDPOINTS:
                    validations.append(validate(output, contract, agent, number))
                    save(output, contract, agent, history, validations)
                    checkpoint_copy(output, contract, agent, number)
    except (Exception, KeyboardInterrupt) as error:
        e147.write(stopped, {
            "episode": number,
            "error": f"{type(error).__name__}: {error}",
            "resume_checkpoint": str(checkpoint),
            "completed_adaptation_episodes": len(history),
            "unfinished_episode_updates_not_saved": True,
        })
        raise
    return {
        "status": "paused_at_requested_endpoint" if through < MAXIMUM_EPISODE else "complete",
        "completed_adaptation_episodes": len(history),
        "latest_checkpoint": str(checkpoint),
        "validation_endpoints": [row["episode"] for row in validations],
        **_counters(agent),
        "test_panel_opened": False,
    }


def status(output):
    summary = Path(output) / "seed-0/summary.json"
    if summary.exists():
        return e147.read(summary)
    return {"status": "not_started", "maximum_authorized_endpoint": MAXIMUM_EPISODE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "smoke", "train", "status"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--through-episode", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract = prepare(args.output)
    if args.command == "prepare":
        result = {"status": "prepared", "maximum_jobs": jobs_for_workload(RHO_HIGH)}
    elif args.command == "preflight":
        result = preflight(args.output, contract)
    elif args.command == "smoke":
        result = smoke(args.output, contract)
    elif args.command == "train":
        result = train(args.output, contract, args.through_episode, args.device, args.retry_failed)
    else:
        result = status(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
