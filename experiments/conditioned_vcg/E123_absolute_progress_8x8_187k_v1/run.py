"""Gate absolute liveness before enabling separately labelled scratch training."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
from statistics import fmean
from time import perf_counter
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.E122_joint_flow_contract_8x8_186k_v1 import run as prior
from experiments.conditioned_vcg.E122_joint_flow_contract_8x8_186k_v1.flow import JointVerifier, JointCertificateCache
from experiments.conditioned_vcg.E121_blocked_pickup_wait_8x8_185k_v1.policy import BlockedPickupWaitRule
from . import protocol as p
from .liveness import ProgressEnv, AbsoluteProgressDeferRule, quantitative_contract
from .model import ProgressAgent

replay = prior.replay
read, write, require = replay.read, replay.write, replay.require


def prepare(output):
    expected = p.contract()
    path = output / "contract.json"
    if path.exists():
        require(read(path) == expected, "E123 code/contract changed; use a new output version")
    else:
        require(not output.exists() or not any(output.iterdir()), "E123 refuses nonempty uncontracted output")
        write(path, expected)
    return expected


def new_env(arm):
    env = ProgressEnv(replay.p.SCENARIO, replay.p.e118.base.e107.e105.e78.SPEC)
    env.enforce_absolute_bound = p.ARMS[arm][1]
    return env


@contextmanager
def controller(arm):
    joint, absolute = p.ARMS[arm]
    with ExitStack() as stack:
        verifier = JointVerifier() if joint else None
        stack.enter_context(patch.object(replay.audit, "ViabilityCertificateCache",
                           JointCertificateCache if joint else replay.reuse.TimingInvariantCertificateCache))
        if joint:
            stack.enter_context(verifier.enabled())
            stack.enter_context(patch.object(replay.audit, "enumerate_viability_candidates", verifier.enumerate))
        stack.enter_context(replay.reuse.path_cleanup_active())
        stack.enter_context(replay.e115run.fast_inputs.enabled())
        stack.enter_context(patch.object(replay.model, "learning_rate_at", p.e118.learning_rate_at))
        stack.enter_context(patch.object(replay.instrumentation, "learning_rate_at", p.e118.learning_rate_at))
        try:
            yield AbsoluteProgressDeferRule() if absolute else BlockedPickupWaitRule()
        finally:
            replay.e115run.fast_inputs._REGISTRY.clear()


def episode(agent, arm, seed, training, updates=1):
    env = new_env(arm)
    profile = env.sample_episode_instance(seed)
    bound = quantitative_contract(profile.arrival_steps)
    require(bound["fits_horizon"], "E123 release assumptions do not support the completion budget")
    env.current_episode = f"E123-{arm}-{'train' if training else 'greedy'}-{seed}"
    manifest = read(p.parent.SOURCE / "frozen-source.json")
    def readable(message, *args, **kwargs):
        if isinstance(message, str) and message.startswith("E115 "):
            message = message.replace("E115 ", f"E123 arm={arm} ", 1)
            message += f" flow_events={env.progress_clock.events}/72 max_progress_gap={env.progress_clock.maximum_gap}"
        print(message, *args, **kwargs)
    with controller(arm) as rule, patch.object(replay.instrumentation, "print", readable, create=True):
        raw = replay.instrumentation.run_episode(agent, env,
            objective_spec=replay.p.e118.base.e107.e105.e78.SPEC,
            instance_seed=seed, training=training, max_steps=3600,
            search_config=replay.ViabilitySearchConfig(**manifest["search_config"]),
            liveness_rule=rule, epsilon_start=.9 if training else 0., epsilon_end=.05 if training else 0.,
            epsilon_decay_decisions=8000, epsilon_warmup_decisions=0,
            updates_per_macro=updates if training else 0, run_protocol=p.PROTOCOL)
    raw.update(e123_arm=arm, progress_events=env.progress_clock.events,
               maximum_absolute_progress_gap=env.progress_clock.maximum_gap,
               final_progress_age=env.progress_clock.age(env.time_steps),
               absolute_budget_passed=(not p.ARMS[arm][1] or
                    (env.progress_clock.maximum_gap <= 50 and raw["strict_method_success"]
                     and env.progress_clock.events == 72)),
               quantitative_contract=bound)
    return raw


class AdversarialWaitAgent(ProgressAgent):
    """Stress the shield; this is not a learned or benchmark policy."""
    def select(self, snapshot, **kwargs):
        decision = super().select(snapshot, **kwargs)
        if self.decision_count % 50 == 0:
            clock = self.context_environment.progress_clock
            print(f"E123 bound stress | decision={self.decision_count} step={snapshot.decision_epoch} "
                  f"flow_events={clock.events}/72 current_gap={clock.age(snapshot.decision_epoch)}", flush=True)
        if decision.liveness_forced:
            return decision
        prepared = decision.prepared_snapshot
        priority = {"defer": 0, "deliver": 1, "accept": 2, "reconfigure": 3}
        index = min(range(len(prepared.records)), key=lambda i: (priority[prepared.records[i].action_type],
                                                                prepared.records[i].key))
        candidate = snapshot.candidates[prepared.source_indices[index]]
        return replace(decision, candidate=candidate, record=prepared.records[index],
                       explored=False, selection_source="e123_adversarial_wait_stress")


def preflight(output, device):
    contract = prepare(output)
    from .self_check import checks
    deterministic = checks()
    require(deterministic["passed"], "E123 deterministic liveness/model checks failed")
    config = replay.model.EpisodicViabilityGraphConfig(gamma=1., episode_horizon_steps=3600)
    agent = AdversarialWaitAgent(config=config, seed=0, device=device, epsilon=0.,
                                arm="C_joint_absolute", absolute_progress=True)
    agent.Q_local.eval().requires_grad_(False)
    agent.Q_target.eval().requires_grad_(False)
    before = replay.parameter_digest(agent)
    started = perf_counter()
    with torch.no_grad():
        # Also exercise real replay/TD-frontier construction; updates are off.
        raw = episode(agent, "C_joint_absolute", p.PILOT_SEED, True, updates=0)
    require(replay.parameter_digest(agent) == before and agent.gradient_steps == 0,
            "E123 no-optimizer preflight mutated weights")
    from .model import ProgressObservedState
    replay_ok = len(agent.replay) == raw["macro_decisions"] and all(
        isinstance(record.current_state, ProgressObservedState)
        for transition in agent.replay.memory
        for record in (transition.chosen, *transition.next_candidates))
    from viability_graph_hierarchy import ACTION_FEATURE_NAMES
    forced_index = ACTION_FEATURE_NAMES.index("guard_forced_frontier")
    deadline_frontiers = [transition.next_candidates for transition in agent.replay.memory
                         if transition.next_candidates
                         and transition.next_candidates[0].current_state.progress_context[1] == 0.]
    deadline_replay_ok = bool(deadline_frontiers) and all(
        record.action_type in ("accept", "deliver")
        or (len(frontier) == 1 and record.action_features[forced_index] == 1.)
        or (not record.current_state.blocks and record.action_type == "defer")
        for frontier in deadline_frontiers for record in frontier)
    passed = bool(raw["strict_method_success"] and raw["absolute_budget_passed"]
                  and replay_ok and deadline_replay_ok)
    result = {"contract_sha256": contract["contract_sha256"], "passed": deterministic["passed"] and passed,
              "deterministic": deterministic, "adversarial_wait_stress": raw,
              "wall_seconds": perf_counter() - started, "episode_runs": 1, "training_runs": 0,
              "real_replay_frontier_clock_roundtrip": replay_ok, "replay_transitions": len(agent.replay),
              "deadline_replay_frontiers": len(deadline_frontiers),
              "deadline_constraints_retained_in_td_frontiers": deadline_replay_ok,
              "stress_optimizer_updates": 0,
              "not_a_performance_comparison": True}
    write(output / "preflight.json", result)
    require(result["passed"], "E123 absolute-progress gate failed; training remains disabled")
    return {key: value for key, value in result.items() if key != "adversarial_wait_stress"}


def checkpoint(path, agent, arm, seed, history, validations, contract):
    payload = agent.checkpoint(include_replay=True, e123_protocol=p.PROTOCOL,
        e123_contract_sha256=contract["contract_sha256"], model_seed=seed,
        completed_training_episodes=len(history), total_training_episodes=p.MAX_EPISODE,
        training_history=history, validation_history=validations,
        global_rng_state=replay.audit._capture_global_rng_state(),
        deployment_checkpoint_eligible=False, test_panel_opened=False, paper_evidence=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def metrics(runs):
    """Per-instance means; failed episodes never masquerade as low costs."""
    complete = all(row["strict_method_success"] for row in runs)
    result = {"episodes": len(runs), "all_complete": complete,
              "strict_completion_rate": fmean(float(row["strict_method_success"]) for row in runs)}
    if not complete:
        return result
    result.update(mean_J=fmean(row["operator_cost"] for row in runs),
                  mean_steps=fmean(row["steps"] for row in runs),
                  mean_steps_per_delivery=fmean(row["steps"] / 36 for row in runs),
                  mean_max_progress_gap=fmean(row["maximum_absolute_progress_gap"] for row in runs))
    for name in ("mean_wait_before_storage", "release_clock_mae", "storage_clock_mae",
                 "mean_release_tardiness_beyond_tolerance", "mean_release_earliness_beyond_tolerance",
                 "within_tolerance_rate", "equipment_work", "physical_rehandles"):
        result[name] = fmean(row["service_metrics"][name] for row in runs)
    return result


def validate(agent, arm, seed, episode_number):
    before = replay.parameter_digest(agent)
    decisions, gradients, replay_count = agent.decision_count, agent.gradient_steps, len(agent.replay)
    global_rng = replay.audit._capture_global_rng_state()
    payload = agent.checkpoint(include_replay=False)
    evaluator = ProgressAgent.from_checkpoint(payload, device=agent.device, resumable=False, seed=seed)
    evaluator.set_epsilon(0.)
    evaluator.Q_local.eval().requires_grad_(False)
    evaluator.Q_target.eval().requires_grad_(False)
    try:
        with torch.no_grad():
            runs = [episode(evaluator, arm, instance, False) for instance in p.VALIDATION_SEEDS]
    finally:
        replay.audit._restore_global_rng_state(global_rng)
    require((decisions, gradients, replay_count) == (agent.decision_count, agent.gradient_steps, len(agent.replay))
            and replay.parameter_digest(agent) == before, "E123 validation advanced training state")
    return {"checkpoint_episode": episode_number, "runs": runs, "evaluation_epsilon": 0., "evaluation_learning": False,
            "all_complete": all(row["strict_method_success"] for row in runs),
            "metrics": metrics(runs),
            "mean_J": fmean(row["operator_cost"] for row in runs) if all(row["strict_method_success"] for row in runs) else None}


def train_arm(output, arm, seed, through, device, authorize_absolute, cpu_pilot):
    contract = prepare(output)
    gate = read(output / "preflight.json")
    require(gate["passed"] and gate["contract_sha256"] == contract["contract_sha256"], "E123 training gate closed")
    require(arm != "C_joint_absolute" or authorize_absolute, "E123 absolute arm requires explicit --authorize-absolute-arm")
    require(device == "cuda" or (cpu_pilot and through <= 1), "E123 long training requires CUDA; CPU pilot is capped at one episode")
    require(device != "cuda" or torch.cuda.is_available(), "CUDA is not available")
    directory = output / arm / f"seed-{seed}"
    latest = directory / "latest.pth"
    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        require(payload["e123_contract_sha256"] == contract["contract_sha256"] and payload["e123_arm"] == arm
                and payload["model_seed"] == seed, "E123 resume identity differs")
        agent = ProgressAgent.from_checkpoint(payload, device=device, resumable=True, seed=seed)
        replay.audit._optimizer_to(agent.optimizer, torch.device(device))
        history, validations = payload["training_history"], payload["validation_history"]
        require(all(row["strict_method_success"] for row in history),
                "E123 saved history contains a failed episode; investigate under a new protocol before continuing")
        replay.audit._restore_global_rng_state(payload["global_rng_state"])
    else:
        from train_viability_graph_smdp import seed_everything
        seed_everything(seed)
        agent = ProgressAgent(config=replay.model.EpisodicViabilityGraphConfig(gamma=1., episode_horizon_steps=3600),
                              seed=seed, device=device, epsilon=.9, arm=arm, absolute_progress=p.ARMS[arm][1])
        history, validations = [], []
        checkpoint(latest, agent, arm, seed, history, validations, contract)
    require(len(history) <= through <= p.MAX_EPISODE, "E123 invalid episode endpoint")
    for number in range(len(history) + 1, through + 1):
        print(f"E123 {arm} seed={seed} episode={number}/{p.MAX_EPISODE} | "
              f"training decisions={agent.decision_count} epsilon floor at 8000", flush=True)
        run = episode(agent, arm, p.train_seed(seed, number), True)
        history.append(run)
        if number % 6 == 0 and run["strict_method_success"]:
            validations.append(validate(agent, arm, seed, number))
            small = agent.checkpoint(include_replay=False, e123_contract_sha256=contract["contract_sha256"],
                                      completed_training_episodes=number, model_seed=seed,
                                      deployment_checkpoint_eligible=False)
            temp = directory / f"candidate-episode-{number}.pth.tmp"
            torch.save(small, temp)
            os.replace(temp, directory / f"candidate-episode-{number}.pth")
        checkpoint(latest, agent, arm, seed, history, validations, contract)
        write(directory / "summary.json", {"contract_sha256": contract["contract_sha256"], "arm": arm,
            "model_seed": seed, "completed_training_episodes": len(history), "training_history": history,
            "validation_history": validations, "training_decisions": agent.decision_count,
            "epsilon": agent.epsilon, "paper_evidence": False, "deployment_checkpoint_eligible": False})
        require(run["strict_method_success"], "E123 training episode failed; endpoint saved, investigate before continuing")
    return {"status": "complete" if len(history) == p.MAX_EPISODE else "paused", "arm": arm,
            "model_seed": seed, "completed_training_episodes": len(history), "latest_checkpoint": str(latest),
            "training_runs": 1, "paper_evidence": False}


def analyze(output):
    contract = prepare(output)
    arms = {}
    for arm in p.ARMS:
        path = output / arm / "seed-0/summary.json"
        if not path.exists():
            continue
        summary = read(path)
        require(summary["contract_sha256"] == contract["contract_sha256"], "E123 summary contract differs")
        late = [row for row in summary["validation_history"] if row["checkpoint_episode"] in (96, 102, 108, 114, 120)]
        arms[arm] = {"episodes": summary["completed_training_episodes"], "decisions": summary["training_decisions"],
                     "epsilon": summary["epsilon"], "validation": [{"episode": r["checkpoint_episode"],
                     "all_complete": r["all_complete"], "J": r["mean_J"],
                     "metrics": r["metrics"]} for r in summary["validation_history"]],
                     "fixed_late_window_metrics": metrics([run for row in late for run in row["runs"]])
                        if len(late) == 5 else None,
                     "fixed_late_window_J": fmean(row["mean_J"] for row in late)
                        if len(late) == 5 and all(row["all_complete"] for row in late) else None}
    report = {"protocol": p.PROTOCOL, "contract_sha256": contract["contract_sha256"], "arms": arms,
              "training_runs": 0, "paper_evidence": False, "test_panel_opened": False}
    write(output / "e123-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "train-arm", "train-matched", "analyze"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--arm", choices=p.ARMS)
    parser.add_argument("--model-seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--through-episode", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--authorize-absolute-arm", action="store_true")
    parser.add_argument("--allow-cpu-pilot", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "preflight":
        result = preflight(output, args.device)
    elif args.command == "train-arm":
        require(args.arm is not None, "E123 train-arm needs --arm")
        result = train_arm(output, args.arm, args.model_seed, args.through_episode, args.device,
                           args.authorize_absolute_arm, args.allow_cpu_pilot)
    elif args.command == "train-matched":
        result = {arm: train_arm(output, arm, args.model_seed, args.through_episode, args.device, False,
                                args.allow_cpu_pilot) for arm in ("A_outflow_bounded", "B_joint_bounded")}
    else:
        result = analyze(output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
