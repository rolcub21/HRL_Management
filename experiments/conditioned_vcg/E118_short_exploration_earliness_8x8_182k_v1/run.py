"""Prepare and run E118's matched 120-episode pilots."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from types import SimpleNamespace
from unittest.mock import patch

import torch

from train_viability_graph_smdp import epsilon_at
from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1 import (
    instrumentation as base_instrumentation,
    model as base_model,
    rewards as base_rewards,
    run as base_run,
)
from . import protocol as p


class E118Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E118Error(message)


def weighted_cost(components, weights) -> float:
    values = tuple(float(value) for value in components)
    require(len(values) == 4, "E118 requires four cost components")
    require(all(math.isfinite(value) and value >= 0 for value in values),
            "E118 costs must be finite and nonnegative")
    return float(sum(value * weight for value, weight in zip(values, weights)))


def rescore_run(run: dict, arm: str) -> dict:
    values = tuple(run["factorized_component_totals"][name]
                   for name in base_rewards.COMPONENT_NAMES)
    trained_cost = weighted_cost(values, p.cost_weights(arm))
    require(math.isclose(trained_cost, float(run["operator_cost"]), abs_tol=1.0e-6),
            "E118 training objective differs from primitive components")
    require(math.isclose(-trained_cost, float(run["return"]), abs_tol=1.0e-6),
            "E118 environment reward differs from training objective")
    run.update(
        e118_arm=arm,
        reference_operator_cost=weighted_cost(values, p.REFERENCE_WEIGHTS),
        early125_operator_cost=weighted_cost(values, p.ALTERNATIVE_WEIGHTS),
        e118_training_cost_weights=list(p.cost_weights(arm)),
    )
    return run


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = p.contract()
    path = output / "contract.json"
    if path.is_file():
        require(base_run.read_json(path) == expected, "E118 contract or source changed")
    else:
        require(not output.exists() or not any(output.iterdir()),
                "E118 refuses a nonempty uncontracted output")
        base_run.write_json(path, expected)
    for arm in p.ARMS:
        arm_path = output / arm / "contract.json"
        wanted = p.arm_contract(arm)
        if arm_path.is_file():
            require(base_run.read_json(arm_path) == wanted,
                    f"E118 contract differs for arm {arm}")
        else:
            directory = arm_path.parent
            require(not directory.exists() or not any(directory.iterdir()),
                    f"E118 refuses a nonempty uncontracted arm: {arm}")
            base_run.write_json(arm_path, wanted)
    return {"status": "prepared", "protocol": p.PROTOCOL,
            "contract_sha256": expected["contract_sha256"], "arms": list(p.ARMS),
            "maximum_episode_per_arm": p.MAX_EPISODE, "fresh_initialization": True,
            "episode_runs": 0, "training_runs": 0}


@contextmanager
def active_arm(output: Path, arm: str):
    """Scope every objective consumer and restore E115 after the arm finishes."""
    view = p.training_view(output, arm)
    weights = p.cost_weights(arm)
    scalar_weights = tuple(-0.01 * value for value in weights)
    original_checkpoint = base_model.FactorizedConservativeAgent.checkpoint

    def checkpoint(agent, *args, **kwargs):
        value = original_checkpoint(agent, *args, **kwargs)
        value.update(e118_arm=arm,
                     e118_contract_sha256=p.arm_contract(arm)["contract_sha256"],
                     e118_parent_contract_sha256=p.contract()["contract_sha256"])
        return value

    def episode(*args, **kwargs):
        env = args[1] if len(args) >= 2 else kwargs["env"]

        def readable_print(message, *extra, **options):
            if isinstance(message, str) and message.startswith("E115 "):
                totals = tuple(sum(row[index] for row in env.factorized_component_records)
                               for index in range(4))
                reference = weighted_cost(totals, p.REFERENCE_WEIGHTS)
                message = message.replace("E115 ", f"E118 arm={arm} ", 1)
                message += f" Jref={reference:.1f}"
            print(message, *extra, **options)

        with patch.object(base_instrumentation, "print", readable_print, create=True):
            run = base_instrumentation.run_episode(*args, **kwargs)
        return rescore_run(run, arm)

    def summarize(runs):
        runs = tuple(runs)
        summary = base_instrumentation.summarize_runs(runs)
        summary["mean_reference_operator_cost"] = fmean(
            run["reference_operator_cost"] for run in runs)
        summary["mean_early125_operator_cost"] = fmean(
            run["early125_operator_cost"] for run in runs)
        summary["e118_arm"] = arm
        return summary

    def trainer_print(message, *extra, **options):
        if isinstance(message, str) and message.startswith("VCG-Dense v1.1 | "):
            message = (
                f"E118 arm={arm} | seed=0 | max episodes={p.MAX_EPISODE} | gamma=1 | "
                f"J=2T+{p.ARMS[arm]:g}E+0.1L+10R | eps=0.90->0.05/8000 decisions | "
                "lr=5e-5->1e-5 at decision 8000"
            )
        elif isinstance(message, str) and message.startswith(("Ep ", "Validation Ep")):
            message = f"E118 arm={arm} | " + message.replace("DenseR", "Return")
        print(message, *extra, **options)

    with ExitStack() as stack:
        for module, name, value in (
            (base_rewards, "EARLINESS_WEIGHT", p.ARMS[arm]),
            (base_rewards, "SCALAR_WEIGHTS", scalar_weights),
            (base_model, "SCALAR_WEIGHTS", scalar_weights),
            (base_model, "COMPONENT_LOSS_WEIGHTS", weights),
            (base_run, "COMPONENT_LOSS_WEIGHTS", weights),
            (base_model, "LR_PHASES", p.LR_PHASES),
            (base_model, "learning_rate_at", p.learning_rate_at),
            (base_run, "learning_rate_at", p.learning_rate_at),
            (base_instrumentation, "learning_rate_at", p.learning_rate_at),
            (base_run, "p", view),
            (base_run, "instrumentation", SimpleNamespace(
                run_episode=episode, summarize_runs=summarize)),
            (base_model.FactorizedConservativeAgent, "checkpoint", checkpoint),
        ):
            stack.enter_context(patch.object(module, name, value))
        stack.enter_context(patch.object(base_run.dense, "print", trainer_print, create=True))
        yield view


def destination(output: Path, arm: str) -> Path:
    return output.resolve() / arm / "training/operational" / p.SCENARIO / "seed-0"


def train_arm(output: Path, arm: str, through_episode: int, device: str) -> dict:
    prepare(output)
    require(arm in p.ARMS, "unknown E118 arm")
    require(1 <= through_episode <= p.MAX_EPISODE, "E118 episode limit is 1..120")
    frozen = p.arm_contract(arm)
    print(f"E118 {arm}: from-scratch seed 0 or authenticated E118 resume; "
          f"through episode {through_episode}/{p.MAX_EPISODE}; "
          f"weights={p.cost_weights(arm)}; eps floor at decision 8000; "
          f"lr 5e-5 -> 1e-5", flush=True)
    try:
        with active_arm(output.resolve() / arm, arm):
            result = base_run.train_seed(output.resolve() / arm, through_episode, device)
    finally:
        # The input cache owns state objects. Release the finished arm's states
        # before constructing the next seed-0 model in a train-all process.
        base_run.fast_inputs._REGISTRY.clear()
    result.update(e118_protocol=p.PROTOCOL, arm=arm,
                  e118_contract_sha256=frozen["contract_sha256"],
                  development_only=True)
    return result


def validation_rows(saved: dict, arm: str) -> list[dict]:
    history = list(saved.get("training", {}).get("runs", ()))
    rows = []
    for record in saved.get("validation_history", ()):
        require(record.get("runs"), "E118 validation has no per-instance runs")
        summary = record.get("summary", {})
        require(summary.get("evaluation_epsilon") == 0.0
                and summary.get("evaluation_learning") is False,
                "E118 comparison requires greedy validation with learning disabled")
        runs = [rescore_run(dict(run), arm) for run in record["runs"]]
        expected = set(p.contract()["validation_seeds"])
        require(len(runs) == len(expected)
                and {int(run["instance_seed"]) for run in runs} == expected,
                "E118 validation instance panel changed")
        instances = []
        for run in runs:
            service = run["service_metrics"]
            instances.append({
                "instance_seed": int(run["instance_seed"]),
                "instance_id": run["episode_instance_id"],
                "strict_complete": bool(run["strict_method_success"]),
                "training_cost": float(run["operator_cost"]),
                "reference_cost": run["reference_operator_cost"],
                "early125_cost": run["early125_operator_cost"],
                "earliness": service["mean_release_earliness_beyond_tolerance"],
                "tardiness": service["mean_release_tardiness_beyond_tolerance"],
                "release_mae": service["release_clock_mae"],
                "storage_mae": service["storage_clock_mae"],
                "wait": service["mean_wait_before_storage"],
                "within_tolerance": service["within_tolerance_rate"],
                "work": service["equipment_work"],
                "steps": run["steps"],
                "rehandles": service["physical_rehandles"],
                "accounting_exact": bool(service["terminal_costs_match_dense_accrual"]),
            })
        episode = int(record["checkpoint_episode"])
        training = history[episode - 1] if len(history) >= episode else {}
        fields = tuple(key for key in instances[0]
                       if key not in ("instance_seed", "instance_id"))
        rows.append({"episode": episode,
                     "epsilon": training.get("scheduled_epsilon_after"),
                     "training_decisions": training.get("training_decision_count_after"),
                     "learning_rate": training.get("learning_rate_after"),
                     **{key: fmean(float(item[key]) for item in instances) for key in fields},
                     "instances": instances})
    return rows


def late_window(rows: list[dict]) -> dict:
    selected = [row for row in rows if row["episode"] in p.LATE_CHECKPOINTS]
    complete = [row["episode"] for row in selected] == list(p.LATE_CHECKPOINTS)
    if not complete:
        return {"status": "pending", "present_checkpoints": [row["episode"] for row in selected],
                "required_checkpoints": list(p.LATE_CHECKPOINTS)}
    seeds = sorted(item["instance_seed"] for item in selected[0]["instances"])
    fields = ("strict_complete", "training_cost", "reference_cost", "early125_cost",
              "earliness", "tardiness", "release_mae", "storage_mae", "wait",
              "within_tolerance", "work", "steps", "rehandles", "accounting_exact")
    by_instance = []
    for seed in seeds:
        matching = [item for row in selected for item in row["instances"]
                    if item["instance_seed"] == seed]
        require(len(matching) == len(p.LATE_CHECKPOINTS), "incomplete E118 paired window")
        require(len({item["instance_id"] for item in matching}) == 1,
                "E118 instance changed across checkpoints")
        by_instance.append({"instance_seed": seed, "instance_id": matching[0]["instance_id"],
                            **{key: fmean(float(item[key]) for item in matching) for key in fields}})
    return {"status": "complete", "checkpoints": list(p.LATE_CHECKPOINTS),
            "unique_instances": len(seeds), "by_instance": by_instance,
            "means": {key: fmean(item[key] for item in by_instance) for key in fields},
            "reference_cost_checkpoint_sd": pstdev(row["reference_cost"] for row in selected),
            "reference_cost_checkpoint_range": [min(row["reference_cost"] for row in selected),
                                                 max(row["reference_cost"] for row in selected)],
            "all_checkpoints_at_epsilon_floor": all(
                row["epsilon"] is not None
                and float(row["epsilon"]) <= p.EPSILON_END + 1.0e-12 for row in selected)}


def analyze(output: Path) -> dict:
    prepare(output)
    arms = {}
    for arm in p.ARMS:
        path = destination(output, arm) / "training-summary.json"
        if not path.is_file():
            arms[arm] = {"status": "not_started", "completed_training_episodes": 0,
                         "rows": [], "late_window": {"status": "pending"}}
            continue
        saved = base_run.read_json(path)
        require(saved.get("e115_contract_sha256") == p.arm_contract(arm)["contract_sha256"],
                f"E118 {arm} summary has another contract")
        rows = validation_rows(saved, arm)
        completed = int(saved["completed_training_episodes"])
        history = list(saved.get("training", {}).get("runs", ()))
        require(len(history) == completed, "E118 summary training history is incomplete")
        floor_episodes = [index + 1 for index, item in enumerate(history)
                          if item["scheduled_epsilon_after"] <= p.EPSILON_END + 1.0e-12]
        window = late_window(rows)
        arms[arm] = {
            "status": "complete" if completed == p.MAX_EPISODE and window["status"] == "complete" else "partial",
            "completed_training_episodes": completed,
            "current_epsilon": history[-1]["scheduled_epsilon_after"] if history else p.EPSILON_START,
            "floor_reached_episode": floor_episodes[0] if floor_episodes else None,
            "episodes_ending_at_floor": len(floor_episodes),
            "rows": rows, "late_window": window,
        }
    paired = None
    if all(value["late_window"]["status"] == "complete" for value in arms.values()):
        left = arms["schedule_only"]["late_window"]["by_instance"]
        right = arms["early125"]["late_window"]["by_instance"]
        pairs = []
        for reference, changed in zip(left, right):
            require(reference["instance_id"] == changed["instance_id"],
                    "E118 arms are not paired on the same instance")
            pairs.append({"instance_seed": reference["instance_seed"],
                          **{key: changed[key] - reference[key] for key in
                             ("reference_cost", "early125_cost", "earliness", "tardiness", "work")}})
        paired = {"direction": "early125 minus schedule_only; lower costs favor early125",
                  "by_instance": pairs,
                  "mean_difference": {key: fmean(item[key] for item in pairs)
                                      for key in pairs[0] if key != "instance_seed"},
                  "replication_unit": "five reused development instances, one training seed",
                  "completion_accounting_pass": all(
                      value["late_window"]["means"]["strict_complete"] == 1.0
                      and value["late_window"]["means"]["accounting_exact"] == 1.0
                      for value in arms.values()),
                  "caveat": "incomplete episodes cannot win through smaller accumulated costs"}
    report = {"protocol": p.PROTOCOL, "contract_sha256": p.contract()["contract_sha256"],
              "status": "complete" if all(value["status"] == "complete" for value in arms.values()) else "partial",
              "arms": arms, "paired_late_window": paired,
              "paper_evidence": False, "independent_test_opened": False,
              "episode_runs": 0, "training_runs": 0}
    path = output.resolve() / "e118-report.json"
    base_run.write_json(path, report)
    table = output.resolve() / "e118-table.md"
    table.write_text(render_table(report), encoding="utf-8")
    return {"status": report["status"], "report": str(path), "table": str(table),
            "completed_training_episodes": {arm: value["completed_training_episodes"]
                                            for arm, value in arms.items()},
            "late_window_complete": {arm: value["late_window"]["status"] == "complete"
                                     for arm, value in arms.items()},
            "episode_runs": 0, "training_runs": 0}


def render_table(report: dict) -> str:
    lines = ["# E118: short exploration and modest earliness penalty", "",
             "Costs are lower-is-better. Jref always uses 2T + E + 0.1L + 10R; "
             "J125 always uses 2T + 1.25E + 0.1L + 10R. E/T are per-job excess beyond ±20.", "",
             "| Arm | Ep | ε | Jref | J125 | E | T | Work | Wait | Within ±20 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, value in report["arms"].items():
        for row in value["rows"]:
            epsilon = "n/a" if row["epsilon"] is None else f"{row['epsilon']:.3f}"
            lines.append(f"| {arm} | {row['episode']} | {epsilon} | {row['reference_cost']:.2f} | "
                         f"{row['early125_cost']:.2f} | {row['earliness']:.3f} | {row['tardiness']:.3f} | "
                         f"{row['work']:.1f} | {row['wait']:.2f} | {row['within_tolerance']:.3f} |")
    lines.extend(["", "Final-five comparison: checkpoints 96, 102, 108, 114, 120, "
                  "averaged within each of the five instances first.", ""])
    for arm, value in report["arms"].items():
        window = value["late_window"]
        if window["status"] != "complete":
            lines.append(f"- {arm}: window pending; {value['completed_training_episodes']}/120 episodes complete.")
        else:
            means = window["means"]
            lines.append(f"- {arm}: Jref={means['reference_cost']:.2f}, J125={means['early125_cost']:.2f}, "
                         f"E={means['earliness']:.3f}, T={means['tardiness']:.3f}; "
                         f"checkpoint Jref SD={window['reference_cost_checkpoint_sd']:.2f}; "
                         f"floor reached at episode {value['floor_reached_episode']}.")
    lines.extend(["", "One training seed and a reused development panel. "
                  "This comparison tests the earliness coefficient under the same shorter schedule; "
                  "it does not isolate epsilon from learning-rate changes or establish replication.", ""])
    return "\n".join(lines)


def self_check(output: Path) -> dict:
    prepare(output)
    from PSLAP.viability import RecoveryState
    from viability_graph_hierarchy import ACTION_FEATURE_DIM, ViabilityGraphCandidateRecord
    from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1.representation import (
        QUEUE_CONTEXT_DIM, attach_queue_context,
    )

    state = RecoveryState(
        rows=2, cols=2,
        traversable=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
        storage_cells=frozenset({(1, 0), (1, 1)}), exits=((0, 1),),
        blocks=(), agent_position=(0, 0), pickup_cells=frozenset({(0, 0)}),
        wait_cells=frozenset({(0, 0)}),
    )
    record = ViabilityGraphCandidateRecord(
        key="defer", mode_id=2, action_type="defer",
        current_state=attach_queue_context(state, (0.0,) * QUEUE_CONTEXT_DIM),
        successor_state=state, action_features=(0.0,) * ACTION_FEATURE_DIM,
    )

    identities, networks = [], []
    observations = {}
    for arm in p.ARMS:
        with active_arm(output.resolve() / arm, arm):
            env = base_rewards.ReleaseETCostEnv(p.SCENARIO, p.base.e107.e105.e78.SPEC)
            instance = env.sample_episode_instance(p.contract()["validation_seeds"][0])
            identities.append(instance.instance_id)
            env.reset(instance=instance)
            env.step(env.ACTION_IDS["WAIT"])
            agent = base_model.FactorizedConservativeAgent(
                config=base_run.EpisodicViabilityGraphConfig(
                    gamma=1.0, episode_horizon_steps=3600, batch_size=1),
                seed=0, device="cpu", epsilon=p.EPSILON_START)
            networks.append({key: value.detach().clone()
                             for key, value in agent.Q_local.state_dict().items()})
            components = (2.0, 3.0, 4.0, 1.0)
            transition = base_model.FactorizedTransition(
                chosen=record, reward=-weighted_cost(components, p.cost_weights(arm)),
                duration=1, next_candidates=(), done=True, component_returns=components)
            agent.replay.add(transition)
            agent.transition_count = agent.config.update_every
            loss = agent.learn()
            require(loss is not None and math.isfinite(loss), "E118 synthetic learning failed")
            payload = agent.checkpoint(include_replay=True, future_schedule_visible_to_policy=False)
            restored = base_model.FactorizedConservativeAgent.from_checkpoint(
                payload, device="cpu", resumable=True, seed=0)
            require(restored.replay.memory[0].component_returns == components,
                    "E118 component replay roundtrip failed")
            observations[arm] = {"scalar_reward_probe": transition.reward,
                                 "replay_roundtrip": True, "synthetic_loss": loss,
                                 "scalar_weights": list(payload["factorized_scalar_weights"])}
    require(len(set(identities)) == 1, "E118 arms generated different instances")
    require(all(torch.equal(networks[0][key], networks[1][key]) for key in networks[0]),
            "E118 arms did not start from identical networks")
    return {"status": "passed", "arms": observations,
            "matched_initial_networks": True, "matched_validation_instance": identities[0],
            "epsilon_probe": {str(count): epsilon_at(count, start=p.EPSILON_START,
                end=p.EPSILON_END, warmup_decisions=0, decay_decisions=p.EPSILON_DECAY_DECISIONS)
                for count in (0, 4000, 8000, 12000)},
            "learning_rate_probe": {str(count): p.learning_rate_at(count)
                                    for count in (0, 7999, 8000, 12000)},
            "episode_runs": 0, "training_runs": 0, "synthetic_optimizer_updates": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "self-check", "train-arm", "train-all", "analyze", "status"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--arm", choices=tuple(p.ARMS))
    parser.add_argument("--through-episode", type=int, default=p.MAX_EPISODE)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "train-arm":
        require(args.arm is not None, "train-arm requires --arm")
        result = train_arm(args.output, args.arm, args.through_episode, args.device)
    elif args.command == "train-all":
        result = {"arms": [train_arm(args.output, arm, args.through_episode, args.device)
                           for arm in p.ARMS]}
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
