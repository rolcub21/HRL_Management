"""Warm-started wait-aware training on continuously sampled working days."""

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from example.episode_instance import EpisodeInstance
from train_viability_graph_smdp import seed_everything
from PSLAP.relocation_family_certification import RELOCATION_FAMILY_CERTIFICATION
from PSLAP.viability_filter import ViabilitySearchConfig
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2 import run as shared
from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v2.acceleration import (
    cache_session,
)
from experiments.conditioned_vcg.E137_grid_rollout_readiness_201k_v1.rollout import (
    collect,
)
from experiments.conditioned_vcg.E141_original_frozen_workload_transfer_205k_v1 import (
    run as source,
)
from experiments.conditioned_vcg.E146_empty_start_working_day import run as prior
from .liveness import interface
from .model import ARRIVAL_FEATURES, NETWORK, WaitAgent, initialize
from .objective import (
    COMPONENT_NAMES,
    OBJECTIVE_WEIGHTS,
    WAIT_WEIGHT,
    WaitDayEnv,
)


ROOT = source.ROOT
OUTPUT = ROOT / "results/vcg-conditioned-e147-wait-aware-working-day-206k-v1-repair3"
DAY = 5400
MEAN_DURATION = 1200
CAPACITY = 35
HORIZON = 16200
RHO_LOW = 0.10
RHO_HIGH = 0.80
MAXIMUM_EPISODE = 120
EPSILON_DECISIONS = 24000
TRAIN_BASE = 24071000
DEV_LEVELS = (0.20, 0.50, 0.80)
DEV_SEEDS = (24070000, 24070001, 24070002)
VALIDATION_ENDPOINTS = (0, 12, 30, 60, 90, 120)
SOURCE_PATH = (
    source.OUTPUT / "models/original-seed-0-ep-60.pth"
)
audit = shared.original.replay.audit


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epsilon(decisions):
    fraction = min(max(int(decisions), 0) / float(EPSILON_DECISIONS), 1.0)
    return 0.90 + fraction * (0.05 - 0.90)


def jobs_for_overlap(rho):
    value = float(rho)
    if not RHO_LOW <= value <= RHO_HIGH:
        raise ValueError("working-day overlap is outside the declared training support")
    return max(1, int(value * CAPACITY * DAY / MEAN_DURATION + 0.5))


def workload_rho(episode):
    if not 1 <= int(episode) <= MAXIMUM_EPISODE:
        raise ValueError("episode is outside the adaptation budget")
    return random.Random(14700000 + int(episode)).uniform(RHO_LOW, RHO_HIGH)


def day_case(seed, rho):
    rho = float(rho)
    n = jobs_for_overlap(rho)
    template = source.discovery.source.original.new_env("A_outflow_bounded")
    native = template.sample_episode_instance(int(seed))
    times = [
        0,
        *np.random.default_rng(int(seed) + 1471)
        .integers(1, DAY, n - 1)
        .tolist(),
    ]
    durations = np.maximum(
        1, np.random.default_rng(int(seed) + 1472).poisson(MEAN_DURATION, n)
    ).tolist()
    jobs = sorted(zip(times, durations), key=lambda item: item[0])
    instance = replace(
        native,
        arrival_rate=n / DAY,
        proc_mean=float(MEAN_DURATION),
        arrival_steps=tuple(arrival for arrival, _ in jobs),
        storage_steps_needed=tuple(duration for _, duration in jobs),
    )
    value = {
        "instance": instance.to_dict(),
        "initial_positions": [],
        "external_due_dates": {
            f"B{index + 1}": int(arrival + duration)
            for index, (arrival, duration) in enumerate(jobs)
        },
        "seed": int(seed),
        "day_steps": DAY,
        "mean_duration": MEAN_DURATION,
        "sampled_overlap": rho,
        "incoming_count": n,
        "capacity": CAPACITY,
        "nominal_overlap": n * MEAN_DURATION / (DAY * CAPACITY),
        "arrivals": "one anchored release at0; remaining releases iid discrete Uniform[1,day), sorted",
    }
    return json.loads(json.dumps(value))


def case_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def stored_case(output, *, seed, rho, name):
    path = Path(output) / "instances" / f"{name}.json"
    expected = day_case(seed, rho)
    if path.exists():
        observed = read(path)
        digest = observed.pop("e147_case_sha256", None)
        if digest != case_digest(observed) or observed != expected:
            raise RuntimeError("stored E147 workload changed")
        observed["e147_case_sha256"] = digest
        return observed, path
    digest = case_digest(expected)
    value = {**expected, "e147_case_sha256": digest}
    write(path, value)
    return value, path


def prepare(output):
    output = Path(output)
    source.prepare(source.OUTPUT)
    info = read(source.OUTPUT / "models/seed-0.json")
    if info["episode"] != 60 or sha(SOURCE_PATH) != info["snapshot_sha256"]:
        raise RuntimeError("E147 warm-start source identity differs")
    development = []
    for level, seed in zip(DEV_LEVELS, DEV_SEEDS):
        value, path = stored_case(
            output, seed=seed, rho=level, name=f"development-rho-{level:g}-seed-{seed}"
        )
        development.append(
            {
                "rho": level,
                "seed": seed,
                "incoming_count": value["incoming_count"],
                "path": str(path.resolve()),
                "sha256": sha(path),
            }
        )
    paths = [
        Path(__file__),
        Path(__file__).with_name("model.py"),
        Path(__file__).with_name("objective.py"),
        Path(__file__).with_name("liveness.py"),
        Path(__file__).with_name("README.txt"),
        ROOT / "tests/test_vcg_e147_wait_aware_day.py",
    ]
    value = {
        "protocol": "e147_wait_aware_continuous_working_day_v1",
        "source_checkpoint": str(SOURCE_PATH.resolve()),
        "source_sha256": sha(SOURCE_PATH),
        "source_episode": 60,
        "source_model_seed": 0,
        "network": NETWORK,
        "objective": "J=2T+E+0.5W+0.1L+10R",
        "component_names": COMPONENT_NAMES,
        "objective_weights": OBJECTIVE_WEIGHTS,
        "wait_definition": "release through first completed storage admission, charged densely",
        "day_steps": DAY,
        "mean_duration": MEAN_DURATION,
        "empty_start": True,
        "geometry": "original 8x8 with 35 storage cells",
        "training_overlap_distribution": "independent Uniform[0.10,0.80] draw per episode",
        "workload_target_visible_to_policy": False,
        "future_release_schedule_visible": False,
        "arrival_features": ARRIVAL_FEATURES,
        "arrival_window_steps": 120,
        "development_cases": development,
        "sealed_test_levels": [0.20, 0.50, 0.80],
        "test_panel_opened": False,
        "initialization": (
            "source encoder/trunk/four heads copied; admission-wait head and "
            "arrival projection exactly zero; optimizer/replay/counters fresh"
        ),
        "liveness_repair": (
            "repeated bounded Defer allowed only in nonterminal states with no released work; "
            "ordinary cap/rules resume as soon as work is observable"
        ),
        "exploration": ".90 to .05 over 24000 adaptation decisions, then .05",
        "maximum_adaptation_episodes": MAXIMUM_EPISODE,
        "validation_endpoints": VALIDATION_ENDPOINTS,
        "execution": {
            "primitive_horizon": HORIZON,
            "macro_limit": 4000,
            "certificate_nodes": 2000,
            "episode_wall_seconds": 1800,
        },
        "first_decision_gate": (
            "at episode60 all three development days complete; rho20 J no more than110% "
            "of episode0; rho80 J at60 below episode30 and episode0"
        ),
        "no_regime_scaling_or_head_normalization": True,
        "same_selected_successor_for_all_heads": True,
        "deployment_checkpoint_eligible": False,
        "paper_evidence": False,
        "sources": {str(path.relative_to(ROOT)): sha(path) for path in paths},
    }
    # Normalize tuples before both hashing and equality authentication.
    value = json.loads(json.dumps(value))
    value["contract_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()
    ).hexdigest()
    path = output / "contract.json"
    if path.exists() and read(path) != value:
        raise RuntimeError("E147 contract changed; use a fresh output")
    if not path.exists():
        write(path, value)
    return value


@contextmanager
def wall_limit(seconds):
    def expired(*_):
        raise TimeoutError(f"episode exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def episode(agent, env, *, training, emit=lambda _: None, macro_limit=4000):
    agent.bind_context_environment(
        env, wait_cap=80, required_jobs=len(env.blocks), max_steps=3600
    )
    agent.reset_episode_state()
    started = time.perf_counter()
    decisions = 0
    streak = 0
    maximum_streak = 0
    reason = None
    before_updates = agent.gradient_steps
    with interface(env) as rule:
        cache = shared.original.replay.reuse.TimingInvariantCertificateCache()

        def snapshot():
            return audit.enumerate_viability_candidates(
                env,
                consecutive_defer_decisions=streak,
                search_config=ViabilitySearchConfig(max_nodes=2000),
                liveness_rule=rule,
                cache=cache,
                recovery_certification_strategy=RELOCATION_FAMILY_CERTIFICATION,
            )

        current = snapshot()
        while (
            decisions < macro_limit
            and env.time_steps < HORIZON
            and not all(block.delivered for block in env.blocks)
        ):
            eps = epsilon(agent.decision_count) if training else 0.0
            if training:
                agent.set_epsilon(eps)
            events_before = env.progress_clock.events
            decision = agent.select(current, training=training, epsilon=eps)
            execution = audit._execute_audit_macro(
                env,
                decision.candidate,
                gamma=1.0,
                remaining_steps=HORIZON - env.time_steps,
                evaluation=not training,
            )
            decisions += 1
            if not execution.option_success or execution.duration < 1:
                raise RuntimeError(f"macro failed: {execution.failure_reason}")
            if (
                decision.candidate.action_type.value == "defer"
                and env.progress_clock.events == events_before
            ):
                streak += 1
            else:
                streak = 0
            maximum_streak = max(maximum_streak, streak)
            complete = all(block.delivered for block in env.blocks)
            terminal = complete or env.time_steps >= HORIZON or decisions >= macro_limit
            following = None if terminal else snapshot()
            if training:
                agent.remember(
                    decision,
                    reward=execution.discounted_return,
                    duration=execution.duration,
                    next_snapshot=following,
                    done=terminal,
                )
                loss = agent.learn()
                if loss is not None and not math.isfinite(loss):
                    raise RuntimeError("nonfinite E147 loss")
            else:
                agent.observe_outcome(
                    decision, next_snapshot=following, done=terminal
                )
            emit(
                {
                    "decisions": decisions,
                    "steps": env.time_steps,
                    "delivered": sum(block.delivered for block in env.blocks),
                    "required": len(env.blocks),
                    "epsilon": eps,
                    "gradient_steps": agent.gradient_steps,
                    "defer_streak": streak,
                }
            )
            if terminal:
                if not complete:
                    reason = (
                        "primitive_horizon"
                        if env.time_steps >= HORIZON
                        else "decision_cap"
                    )
                break
            current = following
    strict = all(block.delivered for block in env.blocks) and reason is None
    return {
        "strict_complete": strict,
        "reason": reason,
        "metrics": collect(env, strict),
        "day_statistics": env.day_metrics(),
        "decisions": decisions,
        "maximum_defer_streak": maximum_streak,
        "gradient_updates": agent.gradient_steps - before_updates,
        "wall_seconds": time.perf_counter() - started,
        "epsilon": agent.epsilon if training else 0.0,
    }


def initialize_agent(contract, device):
    payload = torch.load(
        contract["source_checkpoint"], map_location="cpu", weights_only=False
    )
    return initialize(payload, device=device, horizon=HORIZON)


def save(output, contract, agent, history, validations):
    output = Path(output)
    directory = output / "seed-0"
    directory.mkdir(parents=True, exist_ok=True)
    payload = agent.checkpoint(
        include_replay=True,
        e147_contract_sha256=contract["contract_sha256"],
        completed_adaptation_episodes=len(history),
        training_history=history,
        validation_history=validations,
        global_rng_state=audit._capture_global_rng_state(),
        model_seed=0,
        test_panel_opened=False,
    )
    temporary = directory / "latest.pth.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, directory / "latest.pth")
    write(
        directory / "summary.json",
        {
            "completed_adaptation_episodes": len(history),
            "source_training_episodes": 60,
            "maximum_adaptation_episodes": MAXIMUM_EPISODE,
            "training_decisions": agent.decision_count,
            "gradient_steps": agent.gradient_steps,
            "epsilon": epsilon(agent.decision_count),
            "sampled_overlap_min": min((row["rho"] for row in history), default=None),
            "sampled_overlap_max": max((row["rho"] for row in history), default=None),
            "validation_endpoints": [row["episode"] for row in validations],
            "continue_decision": continue_decision(validations),
            "test_panel_opened": False,
            "deployment_checkpoint_eligible": False,
            "paper_evidence": False,
        },
    )


def validate(output, contract, agent, endpoint):
    rng = audit._capture_global_rng_state()
    try:
        evaluator = WaitAgent.from_checkpoint(
            agent.checkpoint(include_replay=False), device=agent.device
        )
        evaluator.Q_local.eval().requires_grad_(False)
        evaluator.Q_target.eval().requires_grad_(False)
        rows = []
        with torch.no_grad():
            for index, case in enumerate(contract["development_cases"]):
                destination = (
                    Path(output)
                    / "validation"
                    / f"episode-{endpoint}-rho-{case['rho']:g}.json"
                )
                if destination.exists():
                    rows.append(read(destination))
                    continue
                path = Path(case["path"])
                if sha(path) != case["sha256"]:
                    raise RuntimeError("E147 development instance changed")
                value = read(path)
                value.pop("e147_case_sha256", None)
                env = WaitDayEnv(value)
                env._e147_day_steps = DAY
                print(
                    f"E147 greedy episode={endpoint} day={index + 1}/3 "
                    f"rho={case['rho']:.0%} jobs={case['incoming_count']}",
                    flush=True,
                )
                with wall_limit(1800):
                    row = episode(evaluator, env, training=False)
                row.update(
                    episode=endpoint,
                    rho=case["rho"],
                    instance_seed=case["seed"],
                )
                write(destination, row)
                rows.append(row)
        return {"episode": endpoint, "rows": rows}
    finally:
        audit._restore_global_rng_state(rng)


def continue_decision(validations):
    endpoints = {row["episode"]: row for row in validations}
    if not all(number in endpoints for number in (0, 30, 60)):
        return {"status": "pending", "continue_authorized": False}
    selected = [endpoints[number] for number in (0, 30, 60)]
    if not all(all(row["strict_complete"] for row in item["rows"]) for item in selected):
        return {
            "status": "stop_reassess_validation_failure",
            "continue_authorized": False,
        }

    def cost(endpoint, rho):
        row = next(item for item in endpoint["rows"] if item["rho"] == rho)
        return float(row["metrics"]["J"])

    easy0 = cost(endpoints[0], 0.20)
    easy60 = cost(endpoints[60], 0.20)
    hard0 = cost(endpoints[0], 0.80)
    hard30 = cost(endpoints[30], 0.80)
    hard60 = cost(endpoints[60], 0.80)
    easy_retained = easy60 <= 1.10 * easy0
    hard_improving = hard60 < hard30 and hard60 < hard0
    eligible = easy_retained and hard_improving
    return {
        "status": "eligible_to_continue_to_120" if eligible else "stop_reassess",
        "eligible": eligible,
        "continue_authorized": False,
        "easy_retained": easy_retained,
        "hard_improving": hard_improving,
        "easy_J": {"episode0": easy0, "episode60": easy60},
        "hard_J": {"episode0": hard0, "episode30": hard30, "episode60": hard60},
        "note": "episode120 and other seeds require a separate user launch",
    }


def checkpoint_copy(output, agent, endpoint):
    destination = Path(output) / "seed-0" / "checkpoints" / f"episode-{endpoint}.pth"
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pth.tmp")
    torch.save(agent.checkpoint(include_replay=False), temporary)
    os.replace(temporary, destination)


def train(output, contract, through, device, retry_failed):
    if not 1 <= int(through) <= MAXIMUM_EPISODE:
        raise ValueError("E147 training endpoint must be in 1..120")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E147 long training requires available CUDA")
    output = Path(output)
    stopped = output / "stopped.json"
    if stopped.exists():
        if not retry_failed:
            raise RuntimeError("E147 retained a previous stop; inspect it before --retry-failed")
        os.replace(stopped, output / f"acknowledged-stop-{time.time_ns()}.json")
    path = output / "seed-0/latest.pth"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("e147_contract_sha256") != contract["contract_sha256"]:
            raise RuntimeError("E147 resume contract differs")
        agent = WaitAgent.from_checkpoint(payload, device=device, resumable=True)
        audit._optimizer_to(agent.optimizer, torch.device(device))
        history = list(payload["training_history"])
        validations = list(payload["validation_history"])
        audit._restore_global_rng_state(payload["global_rng_state"])
    else:
        seed_everything(0)
        agent = initialize_agent(contract, device)
        history, validations = [], []
        save(output, contract, agent, history, validations)
    if len(history) > through:
        raise ValueError("requested E147 endpoint precedes the checkpoint")
    number = len(history)
    try:
        with cache_session():
            def endpoint_validation(value):
                if value in VALIDATION_ENDPOINTS and not any(
                    row["episode"] == value for row in validations
                ):
                    validations.append(validate(output, contract, agent, value))
                    save(output, contract, agent, history, validations)
                    checkpoint_copy(output, agent, value)

            endpoint_validation(number)
            for number in range(len(history) + 1, through + 1):
                rho = workload_rho(number)
                seed = TRAIN_BASE + number
                value, _ = stored_case(
                    output,
                    seed=seed,
                    rho=rho,
                    name=f"training-episode-{number}-seed-{seed}",
                )
                value = dict(value)
                value.pop("e147_case_sha256", None)
                env = WaitDayEnv(value)
                env._e147_day_steps = DAY
                print(
                    f"E147 train episode={number}/120 rho={rho:.3f} "
                    f"jobs={len(env.blocks)} decisions={agent.decision_count} "
                    f"epsilon={epsilon(agent.decision_count):.3f}",
                    flush=True,
                )
                last_log = time.monotonic()

                def progress(row):
                    nonlocal last_log
                    if time.monotonic() - last_log >= 30:
                        print(
                            f"E147 episode={number} step={row['steps']} "
                            f"delivered={row['delivered']}/{row['required']} "
                            f"epsilon={row['epsilon']:.3f}",
                            flush=True,
                        )
                        write(
                            output / "progress.json",
                            {"episode": number, "rho": rho, **row},
                        )
                        last_log = time.monotonic()

                with wall_limit(1800):
                    row = episode(agent, env, training=True, emit=progress)
                if not row["strict_complete"]:
                    raise RuntimeError(f"training day incomplete: {row['reason']}")
                row.update(episode=number, rho=rho, instance_seed=seed)
                history.append(row)
                save(output, contract, agent, history, validations)
                print(
                    f"E147 episode={number} complete J={row['metrics']['J']:.1f} "
                    f"wait={row['metrics']['total_admission_wait']:.0f} "
                    f"wall={row['wall_seconds']:.1f}s",
                    flush=True,
                )
                endpoint_validation(number)
    except (Exception, KeyboardInterrupt) as error:
        write(
            stopped,
            {
                "episode": number,
                "error": f"{type(error).__name__}: {error}",
                "resume_checkpoint": str(path),
                "completed_adaptation_episodes": len(history),
                "unfinished_episode_updates_not_saved": True,
            },
        )
        raise
    return {
        "status": "paused_at_requested_endpoint" if through < MAXIMUM_EPISODE else "complete",
        "completed_adaptation_episodes": len(history),
        "latest_checkpoint": str(path),
        "continue_decision": continue_decision(validations),
        "test_panel_opened": False,
    }


def smoke(output, contract):
    payload = torch.load(
        contract["source_checkpoint"], map_location="cpu", weights_only=False
    )
    agent = initialize(payload, device="cpu", horizon=HORIZON)
    value = day_case(24079999, RHO_LOW)
    instance = EpisodeInstance.from_dict(value["instance"])
    instance = replace(
        instance,
        arrival_rate=2 / DAY,
        proc_mean=80.0,
        arrival_steps=(0, 1800),
        storage_steps_needed=(80, 80),
    )
    value.update(
        instance=instance.to_dict(),
        external_due_dates={"B1": 80, "B2": 1880},
        incoming_count=2,
        sampled_overlap=2 * 80 / (DAY * CAPACITY),
        nominal_overlap=2 * 80 / (DAY * CAPACITY),
    )
    env = WaitDayEnv(value)
    env._e147_day_steps = DAY
    with cache_session(), wall_limit(180):
        row = episode(agent, env, training=True)
    if not row["strict_complete"] or agent.transition_count < 128 or agent.gradient_steps < 1:
        raise RuntimeError("E147 smoke did not exercise completion, replay, and learning")
    result = {
        "status": "passed",
        "strict_complete": True,
        "steps": env.time_steps,
        "decisions": row["decisions"],
        "transitions": agent.transition_count,
        "gradient_steps": agent.gradient_steps,
        "J": row["metrics"]["J"],
        "wait": row["metrics"]["total_admission_wait"],
        "maximum_defer_streak": row["maximum_defer_streak"],
        "maximum_defer_streak_exceeded_legacy_cap": (
            row["maximum_defer_streak"] > 16
        ),
        "test_panel_opened": False,
    }
    write(Path(output) / "smoke.json", result)
    return result


def status(output):
    path = Path(output) / "seed-0/summary.json"
    result = read(path) if path.exists() else {"status": "prepared_not_started"}
    stopped = Path(output) / "stopped.json"
    if stopped.exists():
        result["stop"] = read(stopped)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "train", "status"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--through-episode", type=int, default=12)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)

    def terminate(*_):
        raise KeyboardInterrupt("termination requested")

    signal.signal(signal.SIGTERM, terminate)
    contract = prepare(args.output)
    if args.command == "prepare":
        result = {
            "status": "prepared",
            "maximum_episode": MAXIMUM_EPISODE,
            "first_long_endpoint": 12,
            "test_panel_opened": False,
        }
    elif args.command == "smoke":
        result = smoke(args.output, contract)
    elif args.command == "train":
        result = train(
            args.output,
            contract,
            args.through_episode,
            args.device,
            args.retry_failed,
        )
    else:
        result = status(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

