#!/usr/bin/env python3
"""Train and evaluate the objective-matched Kim-inspired placement baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

from example.Options.ReconfigureOption import ReconfigureOption
from example.Options.RetrieveDeliverOption import RetrieveDeliverOption
from example.Options.selector_v5 import StrictStorageAssignmentOption
from example.controller_options import build_controller_options, scheduler_episode_audit
from example.urgency_scheduler import RESERVED_CELL_ACTION_INTERFACE, SchedulerInfeasibleError
from experiments.conditioned_vcg.E25_v4_reconfiguration_baselines_95k.controller import ReconfigurationScheduler
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import scenario as e191
from experiments.conditioned_vcg.E195_e191_method_5x5_longstay_254k_v1 import scenario as e195
from experiments.conditioned_vcg.E201_e195_matched_5x5_test_panel_260k_v1 import scenario as panel5
from experiments.conditioned_vcg.E222_e215_e218_fresh_matched_8x8_panel_281k_v1 import scenario as panel8
from PSLAP.kim2020_a3c_spatial import DEPLOYMENT_STOCHASTIC, Kim2020Config
from track_b_urgency_evaluate import _failed_macro_outcome

from . import protocol as p
from .source import ARCHITECTURE_NAME, METHOD_NAME, ObjectiveMatchedKimSource


class ObjectiveMatchedAssignmentOption(StrictStorageAssignmentOption):
    """Atomic common-scheduler adapter, including learning-time previews.

    The common scheduler reserves a placement before pickup.  The inherited
    adapter normally permits that preview only for frozen sources.  Here the
    preview itself *is* the stochastic training decision; the source records
    it once and reuses it on retries.  No scheduler logic is changed.
    """

    @property
    def learning_enabled(self) -> bool:
        # Permit the inherited atomic preview path. The source independently
        # controls whether that preview records a training action.
        return False

    def audit(self) -> dict:
        value = super().audit()
        value["frozen"] = not bool(self.source.learning_enabled)
        value["training_preview_is_policy_decision"] = bool(self.source.learning_enabled)
        value["certification_used"] = False
        value["vcg_guard_used"] = False
        return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _seed(value: int) -> None:
    random.seed(int(value))
    np.random.seed(int(value) % (2**32))
    torch.manual_seed(int(value))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(value))


def make_env(size: str, *, panel: bool = False):
    if size == "5x5":
        cls = panel5.TestPanel5x5Env if panel else e195.LongStay5x5Env
        return cls(e195.SCENARIO, p.SIZE[size]["objective"])
    if size == "8x8":
        cls = panel8.TestPanel8x8Env if panel else e191.LongStayCrowdingEnv
        return cls(e191.SCENARIO, p.SIZE[size]["objective"])
    raise ValueError(f"unknown size: {size}")


def level_for_instance(size: str, seed: int, *, panel: bool) -> float:
    module = panel5 if size == "5x5" else panel8
    if panel:
        return float(module.level_for_seed(seed))
    module = e195 if size == "5x5" else e191
    return float(module.target_level(seed))


def _selector(env, source):
    selector = ObjectiveMatchedAssignmentOption(
        env,
        source,
        assignment_source=METHOD_NAME,
        assignment_source_version=ARCHITECTURE_NAME,
        preview_contract="current_epoch_stochastic_kim_objective_matched_preview_v1",
        assignment_contract="decision_epoch_bound_kim_objective_matched_assignment_v1",
    )
    selector.assignment_source_family = "learned_spatial_assignment_policy"
    selector.assignment_source_learned = True
    return selector


def execute_episode(env, source, instance, *, size: str, learning: bool, policy_seed: int) -> dict:
    source.env = env
    source.set_learning_enabled(bool(learning))
    if not learning:
        source.set_deployment_mode(DEPLOYMENT_STOCHASTIC, policy_seed=int(policy_seed))
    state = env.reset(instance=instance)
    source.on_episode_start()
    # Options are episode-bound because they hold live block references and
    # audit counters. Reset has already closed the previous episode; rebuild
    # the exact same interface against the newly reset blocks.
    env.options.clear()
    selector = _selector(env, source)
    build_controller_options(
        env, selector, controller_action_interface=RESERVED_CELL_ACTION_INTERFACE,
        max_defer_steps=p.MAX_DEFER_STEPS,
    )
    scheduler = ReconfigurationScheduler(env, margin_steps=p.LOOKAHEAD_MARGIN_STEPS)
    current_option = None
    total_return = 0.0
    physical_rehandles = 0
    target_rehandles = 0
    standalone_rehandles = 0
    illegal_drops = 0
    steps = 0
    done = False
    failure = None
    max_steps = int(p.SIZE[size]["max_steps"])
    while steps < max_steps and not done and failure is None:
        if current_option is None:
            try:
                current_option = scheduler.select_action(state)
            except SchedulerInfeasibleError as error:
                failure = f"scheduler_infeasible:{error}"
                break
        try:
            action = current_option.policy(state)
            next_state, reward, done, info = env.step(action)
            steps += 1
            selector.on_step(reward, info)
            total_return += float(reward)
            if info.get("relocated_block"):
                physical_rehandles += 1
                if isinstance(current_option, ReconfigureOption):
                    standalone_rehandles += 1
                elif isinstance(current_option, RetrieveDeliverOption):
                    target_rehandles += 1
                else:
                    failure = "relocation_outside_named_macro"
            illegal_drops += int(bool(info.get("illegal_drop")))
            terminated = current_option.termination(next_state)
        except RuntimeError as error:
            failure = f"macro_runtime:{error}"
            break
        state = next_state
        if terminated:
            outcome = _failed_macro_outcome(current_option)
            if outcome is not None:
                failure = f"macro_failure:{type(current_option).__name__}:{outcome.get('reason', 'unknown')}"
            current_option = None
    truncated = bool(steps >= max_steps and not done)
    selector.on_episode_end(success=done and failure is None, truncated=truncated)
    selector_audit = selector.audit()
    option_audit = scheduler_episode_audit(env)
    scheduler_audit = scheduler.audit()
    bound = option_audit["bound_proposal_ids"]
    committed = option_audit["committed_proposal_ids"]
    selector_committed = [
        decision.get("proposal_id") for decision in selector_audit["decisions"]
        if decision.get("valid") and decision.get("commitment_contract") == "decision_epoch_proposal_bound_once_v2"
    ]
    reservation_integrity = bool(
        option_audit["reservation_invalidation_count"] == 0
        and option_audit["reservation_bound_count"] == option_audit["inbound_successes"]
        and option_audit["reservation_commit_count"] == option_audit["inbound_successes"]
        and option_audit["reservation_execution_match_count"] == option_audit["inbound_successes"]
        and len(bound) == len(set(bound)) and bound == committed == selector_committed
    )
    strict = bool(
        done and failure is None and not truncated
        and selector_audit["invalid_assignment_count"] == 0
        and selector_audit.get("fallback_count", 0) == 0
        and option_audit["inbound_failures"] == 0
        and option_audit["retrieve_failures"] == 0
        and scheduler_audit["standalone_reconfiguration_failures"] == 0
        and illegal_drops == 0 and reservation_integrity
    )
    metrics = env.e187_metrics()
    if strict:
        require(metrics["objective_accounting_exact"], "objective accounting failed")
        require(metrics["reward_sum_equals_negative_J"], "reward/J accounting failed")
        require(math.isclose(total_return, -float(metrics["J"]), abs_tol=1e-7), "return differs from -J")
    return {
        "size": size,
        "instance_seed": int(instance.seed),
        "instance_id": instance.instance_id,
        "policy_seed": int(policy_seed),
        "learning": bool(learning),
        "strict_complete": strict,
        "failure_class": "completed" if strict else (failure or "step_limit" if truncated else "incomplete"),
        "steps": int(steps),
        "return": float(total_return),
        "J": float(metrics["J"]) if strict else None,
        "accrued_J": float(metrics["J"]),
        "components": metrics["components"],
        "physical_rehandles": int(physical_rehandles),
        "target_bound_rehandles": int(target_rehandles),
        "standalone_rehandles": int(standalone_rehandles),
        "peak_stored": metrics["peak_stored"],
        "mean_admission_wait": metrics["mean_admission_wait"],
        "storage_clock_mae": metrics["storage_clock_mae"],
        "release_clock_mae": metrics["release_clock_mae"],
        "certification_used": False,
        "vcg_guard_used": False,
        "scheduler_method": ReconfigurationScheduler.METHOD,
        "scheduler_version": ReconfigurationScheduler.VERSION,
        "selector_audit": selector_audit,
        "scheduler_audit": scheduler_audit,
        "option_audit": option_audit,
    }


def _new_source(env, model_seed: int, device: str):
    _seed(306_247_000 + int(model_seed))
    config = Kim2020Config(gamma=1.0, reward_scale=1.0)
    return ObjectiveMatchedKimSource(
        env, config, seed=306_247_000 + int(model_seed),
        policy_seed=306_247_000 + int(model_seed), learning_enabled=True,
        deployment_mode=DEPLOYMENT_STOCHASTIC, device=device,
    )


def _paths(output: Path, size: str, model_seed: int):
    root = output / "training" / size / f"seed-{model_seed}"
    return root, root / "latest.pth", root / "best.pth", root / "training-summary.json"


def validate(source, *, size: str, model_seed: int) -> dict:
    env = make_env(size)
    rows = []
    source.set_learning_enabled(False)
    for index, seed in enumerate(p.SIZE[size]["validation_seeds"]):
        instance = env.sample_episode_instance(seed=int(seed))
        rows.append(execute_episode(
            env, source, instance, size=size, learning=False,
            policy_seed=247_900_000 + model_seed * 10_000 + index,
        ))
    source.set_learning_enabled(True)
    strict_rate = sum(row["strict_complete"] for row in rows) / len(rows)
    completed = [row["J"] for row in rows if row["strict_complete"]]
    return {
        "strict_completion_rate": strict_rate,
        "mean_J": None if len(completed) != len(rows) else float(np.mean(completed)),
        "rows": rows,
    }


def _plateau(history: list[dict]) -> bool:
    eligible = [row for row in history if row["episode"] >= p.MIN_EPISODES and row["mean_J"] is not None]
    if len(eligible) <= p.PLATEAU_CHECKS:
        return False
    prior = eligible[:-p.PLATEAU_CHECKS]
    recent = eligible[-p.PLATEAU_CHECKS:]
    prior_best = min(row["mean_J"] for row in prior)
    recent_best = min(row["mean_J"] for row in recent)
    return recent_best > prior_best * (1.0 - p.PLATEAU_RELATIVE_IMPROVEMENT)


def train_seed(output: Path, size: str, model_seed: int, through_episode: int, device: str) -> dict:
    require(size in p.SIZES, "invalid size")
    require(model_seed in p.MODEL_SEEDS, "invalid model seed")
    require(1 <= through_episode <= p.MAX_EPISODES, "through episode outside protocol")
    contract = p.contract()
    root, latest_path, best_path, summary_path = _paths(output, size, model_seed)
    root.mkdir(parents=True, exist_ok=True)
    env = make_env(size)
    history = []
    completed = 0
    best_key = (-1.0, float("inf"))
    best_episode = None
    if latest_path.is_file():
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        require(payload.get("contract_sha256") == contract["contract_sha256"], "resume contract changed")
        source = ObjectiveMatchedKimSource.from_checkpoint(
            env, payload, learning_enabled=True, device=device,
            seed=306_247_000 + model_seed,
        )
        completed = int(payload["completed_training_episodes"])
        history = list(payload.get("validation_history", ()))
        best_episode = payload.get("best_episode")
        if history:
            valid = [row for row in history if row["mean_J"] is not None]
            if valid:
                chosen = max(valid, key=lambda row: (row["strict_completion_rate"], -row["mean_J"]))
                best_key = (chosen["strict_completion_rate"], chosen["mean_J"])
    else:
        source = _new_source(env, model_seed, device)
    stopped_for_plateau = False
    for episode in range(completed + 1, through_episode + 1):
        instance_seed = int(p.SIZE[size]["training_seed_base"]) + model_seed * 100_000 + episode
        instance = env.sample_episode_instance(seed=instance_seed)
        row = execute_episode(
            env, source, instance, size=size, learning=True,
            policy_seed=instance_seed,
        )
        print(
            f"E247 train size={size} seed={model_seed} ep={episode} "
            f"J={row['accrued_J']:.1f} strict={int(row['strict_complete'])} "
            f"R={row['physical_rehandles']} steps={row['steps']}", flush=True,
        )
        completed = episode
        if episode % p.VALIDATION_EVERY == 0:
            result = validate(source, size=size, model_seed=model_seed)
            record = {"episode": episode, **result}
            history.append(record)
            print(
                f"E247 validation size={size} seed={model_seed} ep={episode} "
                f"J={result['mean_J']} strict={result['strict_completion_rate']:.3f}", flush=True,
            )
            key = (result["strict_completion_rate"], float("inf") if result["mean_J"] is None else result["mean_J"])
            better = key[0] > best_key[0] or (key[0] == best_key[0] and key[1] < best_key[1])
            if better:
                best_key, best_episode = key, episode
                _torch_save(best_path, source.checkpoint(
                    protocol=p.PROTOCOL, contract_sha256=contract["contract_sha256"],
                    size=size, model_seed=model_seed, selected_episode=episode,
                    selection_validation=result, protocol_training_complete=False,
                ))
            stopped_for_plateau = episode >= p.MIN_EPISODES and _plateau(history)
        if episode % p.CHECKPOINT_EVERY == 0 or episode == through_episode or stopped_for_plateau:
            _torch_save(latest_path, source.checkpoint(
                protocol=p.PROTOCOL, contract_sha256=contract["contract_sha256"],
                size=size, model_seed=model_seed,
                completed_training_episodes=episode, validation_history=history,
                best_episode=best_episode, stopped_for_plateau=stopped_for_plateau,
            ))
        if stopped_for_plateau:
            break
    training_complete = bool(stopped_for_plateau or completed == p.MAX_EPISODES)
    if training_complete and best_path.is_file():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        best["protocol_training_complete"] = True
        best["final_training_episode"] = completed
        _torch_save(best_path, best)
    summary = {
        "status": "complete" if training_complete else "paused",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "size": size, "model_seed": model_seed,
        "completed_training_episodes": completed,
        "stopped_for_plateau": stopped_for_plateau,
        "best_episode": best_episode,
        "best_checkpoint": str(best_path.resolve()) if best_path.is_file() else None,
        "latest_checkpoint": str(latest_path.resolve()),
        "validation_history": history,
        "sealed_test_panel_opened": False,
    }
    _write(summary_path, summary)
    return summary


def _panel(size: str):
    module = panel5 if size == "5x5" else panel8
    return module, tuple(module.TEST_SEEDS)


def evaluate_seed(output: Path, size: str, model_seed: int, device: str) -> dict:
    contract = p.contract()
    _root, _latest, best_path, summary_path = _paths(output, size, model_seed)
    require(best_path.is_file() and summary_path.is_file(), "completed training artifacts are missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary["status"] == "complete", "training is not finalized")
    payload = torch.load(best_path, map_location="cpu", weights_only=False)
    require(payload.get("contract_sha256") == contract["contract_sha256"], "best checkpoint contract changed")
    require(payload.get("protocol_training_complete") is True, "best checkpoint is not finalized")
    env = make_env(size, panel=True)
    source = ObjectiveMatchedKimSource.from_checkpoint(
        env, payload, learning_enabled=False, device=device,
        seed=306_247_000 + model_seed,
    )
    module, seeds = _panel(size)
    rows = []
    for instance_index, instance_seed in enumerate(seeds):
        instance = env.sample_episode_instance(seed=int(instance_seed))
        for trial in range(p.STOCHASTIC_TRIALS):
            path = output / "evaluation" / size / f"seed-{model_seed}" / f"instance-{instance_seed}-trial-{trial}.json"
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            row = execute_episode(
                env, source, instance, size=size, learning=False,
                policy_seed=247_000_000 + model_seed * 100_000 + instance_index * 10 + trial,
            )
            row.update({
                "model_seed": model_seed, "trial": trial,
                "workload": module.level_for_seed(instance_seed),
                "checkpoint": str(best_path.resolve()),
                "checkpoint_selected_episode": payload["selected_episode"],
            })
            _write(path, row)
            rows.append(row)
            print(
                f"E247 eval size={size} seed={model_seed} instance={instance_seed} "
                f"trial={trial} strict={int(row['strict_complete'])} J={row['J']}", flush=True,
            )
    complete = [row for row in rows if row["strict_complete"]]
    result = {
        "status": "complete", "size": size, "model_seed": model_seed,
        "rows": len(rows), "expected_rows": len(seeds) * p.STOCHASTIC_TRIALS,
        "strict_completion_rate": len(complete) / len(rows),
        "mean_J_among_completers": None if not complete else float(np.mean([row["J"] for row in complete])),
        "failures": len(rows) - len(complete),
        "test_panel_opened": True,
    }
    _write(output / "evaluation" / size / f"seed-{model_seed}-summary.json", result)
    return result


def prepare(output: Path) -> dict:
    contract = p.contract()
    _write(output / "contract.json", contract)
    return {
        "status": "prepared", "contract": str((output / "contract.json").resolve()),
        "contract_sha256": contract["contract_sha256"],
        "certification_used": False, "vcg_guard_used": False,
        "scheduler": ReconfigurationScheduler.METHOD,
        "sealed_test_panel_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    train = sub.add_parser("train-seed")
    train.add_argument("--size", choices=p.SIZES, required=True)
    train.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS, required=True)
    train.add_argument("--through-episode", type=int, default=p.MAX_EPISODES)
    train.add_argument("--device", default="cuda")
    evaluate = sub.add_parser("evaluate-seed")
    evaluate.add_argument("--size", choices=p.SIZES, required=True)
    evaluate.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS, required=True)
    evaluate.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "train-seed":
        result = train_seed(output, args.size, args.model_seed, args.through_episode, args.device)
    else:
        result = evaluate_seed(output, args.size, args.model_seed, args.device)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
