#!/usr/bin/env python3
"""Decision-level diagnosis for a frozen exact-safe VCG checkpoint.

The diagnostic is development-only.  It replays explicitly named
EpisodeInstances with epsilon zero and no learning, records the complete exact
SAFE frontier, Q/mode margins, target-network disagreement, witness-guard
overrides, realized SMDP returns, and relocation context.  It refuses both
predeclared sealed test panels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Mapping, Optional, Sequence

import torch

from benchmark_viability_critic_priority import (
    EXACT_FULL,
    _enumerate_frontier,
    _freeze_agent,
    _liveness_rule,
    _load_controller_checkpoint,
    _make_env,
    _search_config,
)
from example.episode_instance import EpisodeInstance
from example.helper.timing_metrics import summarize_delivery_timing
from PSLAP.viability_candidates import (
    ViabilityActionType,
    ViabilityCertificateCache,
)
from train_viability_graph_smdp import execute_certified_macro, resolve_device
from train_viability_graph_smdp_proper import SEALED_IN_REGIME_TEST_SEEDS
from viability_graph_hierarchy import (
    ID_TO_MODE,
    MODE_TO_ID,
    ViabilityGraphHierarchyAgent,
    prepare_viability_snapshot,
    regularized_mode_values,
)


PROTOCOL = "vcg_behavior_diagnostic_development_v1"
SEALED_STRESS_SEEDS = frozenset(range(69_000, 69_010))


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _softmax(values: Sequence[float], temperature: float) -> tuple[float, ...]:
    values = tuple(float(value) for value in values)
    if not values:
        return ()
    scaled = tuple(value / float(temperature) for value in values)
    maximum = max(scaled)
    weights = tuple(math.exp(value - maximum) for value in scaled)
    total = sum(weights)
    return tuple(value / total for value in weights)


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


def _finite(values) -> list[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def _mean(values) -> Optional[float]:
    values = _finite(values)
    return float(fmean(values)) if values else None


def _std(values) -> Optional[float]:
    values = _finite(values)
    return float(pstdev(values)) if values else None


def _correlation(left, right) -> Optional[float]:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right)
        if a is not None
        and b is not None
        and math.isfinite(float(a))
        and math.isfinite(float(b))
    ]
    if len(pairs) < 2:
        return None
    x = [item[0] for item in pairs]
    y = [item[1] for item in pairs]
    mx, my = fmean(x), fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in pairs)
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0.0 or dy == 0.0:
        return None
    return float(numerator / (dx * dy))


def _candidate_step_count(candidate) -> Optional[int]:
    if candidate.recovery_action is not None:
        return int(candidate.recovery_action.steps)
    if candidate.horizon_steps is not None:
        return int(candidate.horizon_steps)
    return None


def _full_policy_diagnostic(
    agent: ViabilityGraphHierarchyAgent,
    snapshot,
    decision,
) -> dict:
    """Score the full safe frontier, including when the guard restricts it."""

    guard_context = agent.recovery_witness_guard.feature_context(
        snapshot,
        forced_frontier=bool(decision.liveness_forced),
    )
    prepared = prepare_viability_snapshot(
        snapshot,
        guard_context=guard_context,
    )
    online = agent._score_records(prepared.records, network=agent.Q_local)
    target = agent._score_records(prepared.records, network=agent.Q_target)
    online_values = tuple(float(value) for value in online.detach().cpu().tolist())
    target_values = tuple(float(value) for value in target.detach().cpu().tolist())
    modes = torch.as_tensor(
        prepared.mode_ids,
        dtype=torch.long,
        device=online.device,
    )
    mode_tensor, live_mode_ids = regularized_mode_values(
        online,
        modes,
        agent.within_temperatures,
    )
    mode_values = tuple(float(value) for value in mode_tensor.cpu().tolist())
    live_modes = tuple(
        ID_TO_MODE[int(value)] for value in live_mode_ids.cpu().tolist()
    )
    mode_probabilities = _softmax(mode_values, agent.config.tau_mode)

    candidate_rows = []
    by_mode: dict[str, list[int]] = defaultdict(list)
    for index, source_index in enumerate(prepared.source_indices):
        candidate = snapshot.candidates[source_index]
        mode = candidate.mode.value
        by_mode[mode].append(index)
        candidate_rows.append(
            {
                "key": candidate.key,
                "mode": mode,
                "action_type": candidate.action_type.value,
                "target_label": candidate.target_label,
                "source": candidate.source,
                "destination": candidate.destination,
                "macro_steps": _candidate_step_count(candidate),
                "recovery_rank_before": candidate.recovery_rank_before,
                "recovery_rank_after": candidate.recovery_rank_after,
                "rank_delta": candidate.rank_delta,
                "q_online": online_values[index],
                "q_target": target_values[index],
                "target_gap": online_values[index] - target_values[index],
            }
        )

    for mode, indices in by_mode.items():
        tau = agent.config.within_temperatures[MODE_TO_ID[mode]]
        probabilities = _softmax(
            [online_values[index] for index in indices], tau
        )
        for index, probability in zip(indices, probabilities):
            candidate_rows[index]["within_mode_probability"] = probability

    best_mode_position = max(
        range(len(live_modes)),
        key=lambda index: (mode_values[index], -MODE_TO_ID[live_modes[index]]),
    )
    preferred_mode = live_modes[best_mode_position]
    preferred_indices = by_mode[preferred_mode]
    preferred_index = min(
        preferred_indices,
        key=lambda index: (-online_values[index], candidate_rows[index]["key"]),
    )
    selected_index = next(
        index
        for index, row in enumerate(candidate_rows)
        if row["key"] == decision.candidate.key
    )
    selected_mode = candidate_rows[selected_index]["mode"]
    selected_mode_position = live_modes.index(selected_mode)
    selected_mode_indices = by_mode[selected_mode]
    selected_mode_sorted = sorted(
        (online_values[index] for index in selected_mode_indices), reverse=True
    )
    within_margin = (
        selected_mode_sorted[0] - selected_mode_sorted[1]
        if len(selected_mode_sorted) > 1
        else None
    )
    ordered_mode_values = sorted(mode_values, reverse=True)
    mode_margin = (
        ordered_mode_values[0] - ordered_mode_values[1]
        if len(ordered_mode_values) > 1
        else None
    )

    within_entropies = {}
    for mode, indices in by_mode.items():
        probabilities = [
            candidate_rows[index]["within_mode_probability"]
            for index in indices
        ]
        within_entropies[mode] = _entropy(probabilities)

    return {
        "candidate_count": len(candidate_rows),
        "candidate_counts_by_mode": {
            mode: len(indices) for mode, indices in by_mode.items()
        },
        "candidate_counts_by_action": dict(
            Counter(row["action_type"] for row in candidate_rows)
        ),
        "candidates": tuple(candidate_rows),
        "mode_values": dict(zip(live_modes, mode_values)),
        "mode_probabilities": dict(zip(live_modes, mode_probabilities)),
        "mode_entropy": _entropy(mode_probabilities),
        "within_mode_entropies": within_entropies,
        "preferred_mode_without_guard": preferred_mode,
        "preferred_key_without_guard": candidate_rows[preferred_index]["key"],
        "selected_q_online": online_values[selected_index],
        "selected_q_target": target_values[selected_index],
        "selected_target_gap": (
            online_values[selected_index] - target_values[selected_index]
        ),
        "selected_within_mode_probability": candidate_rows[selected_index][
            "within_mode_probability"
        ],
        "selected_mode_probability": mode_probabilities[
            selected_mode_position
        ],
        "within_mode_q_margin": within_margin,
        "mode_value_margin": mode_margin,
        "guard_override": bool(
            decision.liveness_forced
            and candidate_rows[preferred_index]["key"]
            != decision.candidate.key
        ),
        "guard_q_opportunity_cost": (
            online_values[preferred_index] - online_values[selected_index]
            if decision.liveness_forced
            else 0.0
        ),
        "guard_mode_opportunity_cost": (
            mode_values[best_mode_position]
            - mode_values[selected_mode_position]
            if decision.liveness_forced
            else 0.0
        ),
    }


def _load_instance(seed: int, payload: Mapping, directory: Optional[Path]):
    env = _make_env(payload)
    if directory is None:
        return env.sample_episode_instance(int(seed))
    path = directory / f"seed-{seed}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    instance = EpisodeInstance.from_json(path.read_text(encoding="utf-8"))
    instance.validate_for(env)
    return instance


def run_episode(
    payload: dict,
    instance,
    *,
    instance_seed: int,
    max_steps: int,
    device: torch.device,
) -> dict:
    env = _make_env(payload)
    env.current_episode = 1
    env.reset(instance=instance)
    search_config = _search_config(payload)
    liveness_rule = _liveness_rule(payload)
    agent = _freeze_agent(payload, device=device, seed=int(payload.get("model_seed", 0)))
    agent.reset_episode_state()
    cache = ViabilityCertificateCache()

    decisions = []
    total_return = 0.0
    steps = 0
    consecutive_defer = 0
    delivery_deviations = []
    method_failure_reason = None
    pending = None
    pending_record = None

    while steps < max_steps and not env.is_state_terminal(env.current_state):
        if pending is None:
            pending, pending_record = _enumerate_frontier(
                env,
                consecutive_defer=consecutive_defer,
                search_config=search_config,
                liveness_rule=liveness_rule,
                cache=cache,
                prioritizer=None,
            )
        snapshot = pending
        frontier_record = pending_record
        pending = None
        pending_record = None
        if not snapshot.candidates:
            method_failure_reason = "no_exact_safe_candidate"
            break

        decision = agent.select(snapshot, training=False, epsilon=0.0)
        policy = _full_policy_diagnostic(agent, snapshot, decision)
        execution = execute_certified_macro(
            env,
            decision.candidate,
            gamma=agent.config.gamma,
            remaining_steps=max_steps - steps,
            evaluation=True,
        )
        steps += execution.duration
        total_return += execution.raw_return
        delivery_deviations.extend(execution.delivery_deviations)

        if execution.action_type == ViabilityActionType.DEFER.value:
            outcome = getattr(decision.option, "last_outcome", None)
            observed_event = bool(
                isinstance(outcome, dict)
                and outcome.get("reason") == "observed_event"
            )
            consecutive_defer = 0 if observed_event else consecutive_defer + 1
        else:
            consecutive_defer = 0

        boundary_done = bool(
            execution.env_terminal
            or execution.truncated
            or not execution.option_success
            or execution.duration == 0
        )
        if not boundary_done:
            pending, pending_record = _enumerate_frontier(
                env,
                consecutive_defer=consecutive_defer,
                search_config=search_config,
                liveness_rule=liveness_rule,
                cache=cache,
                prioritizer=None,
            )
            if not pending.candidates:
                boundary_done = True
                pending = None
                pending_record = None
                method_failure_reason = "no_exact_safe_candidate"

        agent.observe_outcome(
            decision,
            next_snapshot=None if boundary_done else pending,
            done=boundary_done,
        )
        decisions.append(
            {
                "instance_seed": int(instance_seed),
                "decision_index": len(decisions),
                "decision_epoch": int(snapshot.decision_epoch),
                "selected_key": decision.candidate.key,
                "selected_mode": decision.candidate.mode.value,
                "selected_action_type": decision.candidate.action_type.value,
                "selected_target_label": decision.candidate.target_label,
                "selected_source": decision.candidate.source,
                "selected_destination": decision.candidate.destination,
                "selection_source": decision.selection_source,
                "liveness_forced": bool(decision.liveness_forced),
                "exact_rank_progress": bool(decision.exact_rank_progress),
                "duration": int(execution.duration),
                "discounted_macro_return": float(execution.discounted_return),
                "raw_macro_return": float(execution.raw_return),
                "delivery_deviations": tuple(execution.delivery_deviations),
                "relocations": int(execution.relocations),
                "option_success": bool(execution.option_success),
                "safe_candidates": int(frontier_record["safe_candidate_count"]),
                "unsafe_candidates_rejected": int(
                    frontier_record["unsafe_candidates_rejected"]
                ),
                "unknown_candidates_rejected": int(
                    frontier_record["unknown_candidates_rejected"]
                ),
                **policy,
            }
        )
        if not execution.option_success and method_failure_reason is None:
            method_failure_reason = (
                f"macro_failure:{execution.action_type}:"
                f"{execution.failure_reason or 'unknown'}"
            )
        if boundary_done:
            break

    future = 0.0
    for decision in reversed(decisions):
        future = (
            agent.config.reward_scale * decision["discounted_macro_return"]
            + (agent.config.gamma ** decision["duration"]) * future
        )
        decision["monte_carlo_scaled_return"] = future
        decision["q_minus_monte_carlo"] = (
            decision["selected_q_online"] - future
        )

    for index, decision in enumerate(decisions):
        if decision["selected_action_type"] != ViabilityActionType.RECONFIGURE.value:
            continue
        next_delivery = None
        steps_to_delivery = 0
        macros_to_delivery = 0
        for later in decisions[index + 1 :]:
            steps_to_delivery += int(later["duration"])
            macros_to_delivery += 1
            if later["selected_action_type"] == ViabilityActionType.DELIVER.value:
                next_delivery = later
                break
        decision["delivery_candidate_available"] = bool(
            decision["candidate_counts_by_action"].get("deliver", 0)
        )
        decision["defer_candidate_available"] = bool(
            decision["candidate_counts_by_action"].get("defer", 0)
        )
        decision["macros_until_next_delivery"] = (
            macros_to_delivery if next_delivery is not None else None
        )
        decision["steps_until_next_delivery"] = (
            steps_to_delivery if next_delivery is not None else None
        )
        decision["next_delivery_deviation"] = (
            next_delivery["delivery_deviations"][0]
            if next_delivery is not None
            and next_delivery["delivery_deviations"]
            else None
        )

    terminal = bool(env.is_state_terminal(env.current_state))
    timing = summarize_delivery_timing(
        delivery_deviations, env.DELIVERY_TARGET_WINDOW
    )
    return {
        "instance_seed": int(instance_seed),
        "instance_id": instance.instance_id,
        "return": float(total_return),
        "steps": int(steps),
        "macro_decisions": len(decisions),
        "terminal": terminal,
        "success": bool(terminal and method_failure_reason is None),
        "method_failure_reason": method_failure_reason,
        "delivery_deviations": tuple(delivery_deviations),
        **timing,
        "relocations": sum(item["relocations"] for item in decisions),
        "guard_audit": agent.recovery_witness_guard.audit_dict(),
        "decisions": tuple(decisions),
    }


def _group_summary(decisions: Sequence[dict]) -> dict:
    q_errors = [item["q_minus_monte_carlo"] for item in decisions]
    return {
        "count": len(decisions),
        "mean_duration": _mean(item["duration"] for item in decisions),
        "mean_raw_macro_return": _mean(
            item["raw_macro_return"] for item in decisions
        ),
        "mean_q_online": _mean(item["selected_q_online"] for item in decisions),
        "mean_monte_carlo_scaled_return": _mean(
            item["monte_carlo_scaled_return"] for item in decisions
        ),
        "q_bias": _mean(q_errors),
        "q_mae": _mean(abs(value) for value in q_errors),
        "q_rmse": (
            math.sqrt(fmean(value * value for value in q_errors))
            if q_errors
            else None
        ),
        "q_return_correlation": _correlation(
            [item["selected_q_online"] for item in decisions],
            [item["monte_carlo_scaled_return"] for item in decisions],
        ),
        "mean_target_gap": _mean(
            item["selected_target_gap"] for item in decisions
        ),
        "mean_mode_margin": _mean(
            item["mode_value_margin"] for item in decisions
        ),
        "mean_within_mode_margin": _mean(
            item["within_mode_q_margin"] for item in decisions
        ),
        "mean_selected_mode_probability": _mean(
            item["selected_mode_probability"] for item in decisions
        ),
        "mean_selected_within_mode_probability": _mean(
            item["selected_within_mode_probability"] for item in decisions
        ),
    }


def summarize(episodes: Sequence[dict]) -> dict:
    decisions = [item for episode in episodes for item in episode["decisions"]]
    grouped = {}
    for action_type in ("accept", "deliver", "reconfigure", "defer"):
        grouped[action_type] = _group_summary(
            [
                item
                for item in decisions
                if item["selected_action_type"] == action_type
            ]
        )
    forced = [item for item in decisions if item["liveness_forced"]]
    unforced = [item for item in decisions if not item["liveness_forced"]]
    relocations = [
        item
        for item in decisions
        if item["selected_action_type"] == "reconfigure"
    ]
    episode_relocations = [item["relocations"] for item in episodes]
    return {
        "episodes": len(episodes),
        "mean_return": _mean(item["return"] for item in episodes),
        "return_std": _std(item["return"] for item in episodes),
        "success_rate": _mean(float(item["success"]) for item in episodes),
        "mean_steps": _mean(item["steps"] for item in episodes),
        "mean_absolute_error": _mean(
            item["mean_absolute_error"] for item in episodes
        ),
        "mean_signed_deviation": _mean(
            item["mean_signed_deviation"] for item in episodes
        ),
        "mean_tardiness": _mean(item["mean_tardiness"] for item in episodes),
        "within_target_window_rate": _mean(
            item["within_target_window_rate"] for item in episodes
        ),
        "total_decisions": len(decisions),
        "action_counts": dict(
            Counter(item["selected_action_type"] for item in decisions)
        ),
        "mode_counts": dict(Counter(item["selected_mode"] for item in decisions)),
        "selection_source_counts": dict(
            Counter(item["selection_source"] for item in decisions)
        ),
        "mean_safe_candidates": _mean(
            item["safe_candidates"] for item in decisions
        ),
        "mean_candidate_counts_by_mode": {
            mode: _mean(
                item["candidate_counts_by_mode"].get(mode, 0)
                for item in decisions
            )
            for mode in ("accept", "recover", "defer")
        },
        "mean_mode_entropy": _mean(item["mode_entropy"] for item in decisions),
        "mean_target_network_gap": _mean(
            abs(item["selected_target_gap"]) for item in decisions
        ),
        "all_decisions": _group_summary(decisions),
        "unforced_decisions": _group_summary(unforced),
        "forced_decisions": {
            **_group_summary(forced),
            "fraction": len(forced) / len(decisions) if decisions else None,
            "override_count": sum(item["guard_override"] for item in forced),
            "override_rate": (
                sum(item["guard_override"] for item in forced) / len(forced)
                if forced
                else None
            ),
            "mean_q_opportunity_cost": _mean(
                item["guard_q_opportunity_cost"] for item in forced
            ),
            "mean_mode_opportunity_cost": _mean(
                item["guard_mode_opportunity_cost"] for item in forced
            ),
        },
        "by_action_type": grouped,
        "relocation_behavior": {
            "count": len(relocations),
            "per_100_deliveries": (
                100.0
                * len(relocations)
                / sum(item["delivery_count"] for item in episodes)
            ),
            "guard_forced_count": sum(
                item["liveness_forced"] for item in relocations
            ),
            "with_delivery_candidate_count": sum(
                item.get("delivery_candidate_available", False)
                for item in relocations
            ),
            "with_defer_candidate_count": sum(
                item.get("defer_candidate_available", False)
                for item in relocations
            ),
            "mean_steps_until_next_delivery": _mean(
                item.get("steps_until_next_delivery") for item in relocations
            ),
            "mean_next_delivery_deviation": _mean(
                item.get("next_delivery_deviation") for item in relocations
            ),
            "episode_relocation_return_correlation": _correlation(
                episode_relocations,
                [item["return"] for item in episodes],
            ),
            "episode_relocation_mae_correlation": _correlation(
                episode_relocations,
                [item["mean_absolute_error"] for item in episodes],
            ),
        },
        "total_unsafe_rejections": sum(
            item["unsafe_candidates_rejected"] for item in decisions
        ),
        "total_unknown_rejections": sum(
            item["unknown_candidates_rejected"] for item in decisions
        ),
        "method_failures": [
            {
                "instance_seed": item["instance_seed"],
                "reason": item["method_failure_reason"],
            }
            for item in episodes
            if item["method_failure_reason"] is not None
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--instances-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = build_parser().parse_args(argv)
    args.seeds = tuple(dict.fromkeys(args.seeds))
    sealed = set(args.seeds).intersection(
        SEALED_STRESS_SEEDS | SEALED_IN_REGIME_TEST_SEEDS
    )
    if sealed:
        raise ValueError(f"diagnostic refuses sealed test seeds: {sorted(sealed)}")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    result_path = args.output_dir / "diagnostic.json"
    if result_path.exists():
        raise FileExistsError(result_path)

    payload = _load_controller_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    episodes = []
    for seed in args.seeds:
        instance = _load_instance(seed, payload, args.instances_dir)
        episode = run_episode(
            payload,
            instance,
            instance_seed=seed,
            max_steps=args.max_steps,
            device=device,
        )
        episodes.append(episode)
        print(
            f"seed={seed} R={episode['return']:.2f} "
            f"MAE={episode['mean_absolute_error']:.3f} "
            f"steps={episode['steps']} reloc={episode['relocations']} "
            f"guard={episode['guard_audit']['forced_decisions']}",
            flush=True,
        )

    summary = summarize(episodes)
    result = {
        "protocol": PROTOCOL,
        "scope": "development_only_no_sealed_test_opened",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_training_protocol": payload.get("protocol"),
        "arm": EXACT_FULL,
        "epsilon": 0.0,
        "learning": False,
        "teacher": False,
        "critic": False,
        "exact_verifier_authoritative": True,
        "seeds": args.seeds,
        "summary": summary,
        "episodes": tuple(episodes),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    csv_path = args.output_dir / "decisions.csv"
    fields = (
        "instance_seed",
        "decision_index",
        "decision_epoch",
        "selected_mode",
        "selected_action_type",
        "selected_target_label",
        "selection_source",
        "liveness_forced",
        "guard_override",
        "duration",
        "raw_macro_return",
        "selected_q_online",
        "selected_q_target",
        "monte_carlo_scaled_return",
        "q_minus_monte_carlo",
        "mode_value_margin",
        "within_mode_q_margin",
        "selected_mode_probability",
        "selected_within_mode_probability",
        "safe_candidates",
        "delivery_candidate_available",
        "defer_candidate_available",
        "steps_until_next_delivery",
        "next_delivery_deviation",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            for decision in episode["decisions"]:
                writer.writerow({name: decision.get(name) for name in fields})

    print(json.dumps(_json_safe(summary), indent=2), flush=True)
    print(f"Diagnostic: {result_path}", flush=True)
    print(f"Decisions: {csv_path}", flush=True)
    return result


if __name__ == "__main__":
    main()
