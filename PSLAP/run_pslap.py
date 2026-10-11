"""Execution and logging shared by the explicit PSLAP baselines."""

from collections import deque

import numpy as np

from example.helper.timing_metrics import summarize_delivery_timing
from PSLAP.baselines import DYNAMIC_PSLAP, make_policy, normalize_baseline


def run_pslap_episode(
    env,
    max_steps=1700,
    baseline=DYNAMIC_PSLAP,
    episode_instance=None,
):
    """Run one baseline episode and retain paper- and environment-level metrics."""

    baseline = normalize_baseline(baseline)
    env.reset(instance=episode_instance)
    policy = make_policy(env, baseline)
    generator = policy.step()
    total_reward = 0.0
    errors = []
    obstructive_moves = 0
    illegal_drops = 0
    steps = 0

    while steps < max_steps and not env.is_state_terminal(env.current_state):
        try:
            action = next(generator)
        except StopIteration:
            generator = policy.step()
            action = next(generator)

        _, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if "delivery_error_time" in info:
            errors.append(float(info["delivery_error_time"]))
        if info.get("relocated_block"):
            obstructive_moves += 1
        if info.get("illegal_drop"):
            illegal_drops += 1
        if done:
            break

    ga_result = getattr(policy, "ga_result", None)
    ga_cost = getattr(ga_result, "best_cost", None)
    assigner = getattr(policy, "ga_assigner", None)
    return {
        "instance_id": env.current_episode_instance.instance_id,
        "return": total_reward,
        "steps": steps,
        "errors": errors,
        "success": float(env.is_state_terminal(env.current_state)),
        "obstructive_moves": obstructive_moves,
        "illegal_drops": illegal_drops,
        "assignment_fallbacks": int(
            getattr(assigner, "fallback_count", 0)
        ),
        "ga_predicted_infeasible_events": getattr(
            ga_cost, "infeasible_events", None
        ),
        "ga_predicted_obstructive_moves": getattr(
            ga_cost, "obstructive_moves", None
        ),
        "ga_predicted_route_steps": getattr(ga_cost, "route_steps", None),
    }


def run_pslap_experiment(
    env,
    n_episodes,
    max_steps,
    baseline=DYNAMIC_PSLAP,
    episode_instances=None,
):
    baseline = normalize_baseline(baseline)
    if episode_instances is not None:
        episode_instances = tuple(episode_instances)
        if len(episode_instances) != n_episodes:
            raise ValueError(
                "episode_instances length must match n_episodes"
            )
    logs = {
        "baseline": baseline,
        "episode_returns": [],
        "episode_instance_ids": [],
        "episode_avg_error": [],
        "episode_mean_signed_deviation": [],
        "episode_mean_absolute_error": [],
        "episode_mean_tardiness": [],
        "episode_mean_earliness": [],
        "episode_within_target_window_rate": [],
        "episode_tardy_delivery_rate": [],
        "episode_p90_tardiness": [],
        "episode_obstructive_moves": [],
        "episode_illegal_drops": [],
        "episode_assignment_fallbacks": [],
        "episode_success": [],
        "manager_losses": [],
        "worker_losses": [],
    }

    for episode in range(1, n_episodes + 1):
        instance = (
            None if episode_instances is None else episode_instances[episode - 1]
        )
        result = run_pslap_episode(
            env,
            max_steps=max_steps,
            baseline=baseline,
            episode_instance=instance,
        )
        timing = summarize_delivery_timing(
            result["errors"], getattr(env, "DELIVERY_TARGET_WINDOW", 20.0)
        )

        logs["episode_instance_ids"].append(result["instance_id"])
        logs["episode_returns"].append(result["return"])
        logs["episode_avg_error"].append(timing["mean_signed_deviation"])
        logs["episode_mean_signed_deviation"].append(
            timing["mean_signed_deviation"]
        )
        logs["episode_mean_absolute_error"].append(
            timing["mean_absolute_error"]
        )
        logs["episode_mean_tardiness"].append(timing["mean_tardiness"])
        logs["episode_mean_earliness"].append(timing["mean_earliness"])
        logs["episode_within_target_window_rate"].append(
            timing["within_target_window_rate"]
        )
        logs["episode_tardy_delivery_rate"].append(
            timing["tardy_delivery_rate"]
        )
        logs["episode_p90_tardiness"].append(timing["p90_tardiness"])
        logs["episode_obstructive_moves"].append(
            result["obstructive_moves"]
        )
        logs["episode_illegal_drops"].append(result["illegal_drops"])
        logs["episode_assignment_fallbacks"].append(
            result["assignment_fallbacks"]
        )
        logs["episode_success"].append(result["success"])

        mean_return = np.mean(deque(logs["episode_returns"], maxlen=100))
        recent_errors = np.asarray(
            deque(logs["episode_avg_error"], maxlen=100), dtype=float
        )
        mean_error = (
            np.nan
            if np.all(np.isnan(recent_errors))
            else np.nanmean(recent_errors)
        )
        mean_success = np.mean(deque(logs["episode_success"], maxlen=100))
        mean_moves = np.mean(
            deque(logs["episode_obstructive_moves"], maxlen=100)
        )

        print(
            f"[{baseline}] Ep {episode:4d} | "
            f"AvgR {mean_return:7.2f} | "
            f"Bias {mean_error:7.2f} | "
            f"Obs {mean_moves:5.2f} | "
            f"Succ {mean_success:.3f}"
        )

    return logs


__all__ = ["run_pslap_episode", "run_pslap_experiment"]
