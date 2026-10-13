"""Matched 5x5 pilot: full certification versus certify-ranked-proposals.

Both arms use the same graph network, masked rank/guard inputs, epsilon
schedule, Double-Q backup, instances, and held-out evaluation. Only training-
time frontier construction differs. Every *executed* action in both arms is
certified; the proposal arm never calls an unproved action SAFE. The proposal
arm reuses the replay record's numeric container, but its next-candidate
records are deliberately unverified legal proposals.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import torch

from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    ViabilityActionCandidate, ViabilityActionType, ViabilityMode,
    enumerate_viability_candidates,
)
from PSLAP import viability_candidates as vc
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E83_legal_proposal_5x5_pilot_148k_v1 import legal
from viability_graph_episodic_audit import EpisodicViabilityGraphHierarchyAgent
from viability_graph_hierarchy import (
    ViabilityGraphCandidateRecord, ViabilityGraphTransition,
    regularized_mode_values,
)
from train_vcg_preference_conditioned import _atomic_json_save
import run_vcg_objective_gamma_audit as audit
import train_vcg_dense_proper as dense


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e83-legal-proposal-5x5-148k-v1"
PROTOCOL = "vcg_conditioned_e83_legal_proposal_5x5_148k_v1"
ARMS = ("full_frontier", "rank_then_certify")
TRAIN_SEED_BASE = 148_000
EVAL_SEEDS = tuple(range(148_200, 148_205))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def runtime_args(output: Path):
    return audit.build_parser().parse_args([
        "--output-dir", str(output), "--device", "cpu",
        "--grid-rows", "5", "--grid-cols", "5",
        "--number-blocks", "8", "--max-steps", "2000",
        "--dense-b", "40", "--search-order", "goal_directed",
        "--batch-size", "32", "--update-every", "4",
    ])


def _rank_index(agent, records: tuple[ViabilityGraphCandidateRecord, ...],
                *, epsilon: float, training: bool) -> int:
    require(bool(records), "empty legal ranking frontier")
    scores = agent._score_records(records).detach()
    mode_ids = torch.as_tensor(
        [record.mode_id for record in records],
        dtype=torch.long, device=agent.device,
    )
    mode_values, modes = regularized_mode_values(
        scores, mode_ids, agent.within_temperatures,
    )
    if training and agent.rng.random() < epsilon:
        mode = agent.rng.choice(modes.tolist())
        indices = [i for i, item in enumerate(records) if item.mode_id == mode]
        return agent.rng.choice(indices)
    winning_mode = int(modes[int(torch.argmax(mode_values).item())].item())
    indices = [i for i, item in enumerate(records)
               if item.mode_id == winning_mode]
    return min(indices, key=lambda i: (-float(scores[i]), records[i].key))


def _certify_one(proposal: legal.LegalProposal, state, current_certificate,
                 search, cache):
    if proposal.action_type is ViabilityActionType.DEFER:
        certificate = current_certificate
        cache_hit, exact_seconds = True, 0.0
    else:
        certificate, cache_hit, exact_seconds = vc._analyze_cached(
            proposal.successor_state, search, cache,
        )
    if certificate.status is not ViabilityStatus.SAFE:
        return None, cache_hit, exact_seconds
    rank_before = (current_certificate.witness_macro_count
                   if current_certificate.recovery_rank_is_exact else None)
    rank_after = (certificate.witness_macro_count
                  if certificate.recovery_rank_is_exact else None)
    candidate = ViabilityActionCandidate(
        key=proposal.key, mode=proposal.mode,
        action_type=proposal.action_type,
        target_label=proposal.target_label, source=proposal.source,
        destination=proposal.destination,
        successor_state=proposal.successor_state, certificate=certificate,
        option=proposal.option, recovery_action=proposal.recovery_action,
        horizon_steps=proposal.horizon_steps,
        recovery_rank_before=rank_before, recovery_rank_after=rank_after,
    )
    return candidate, cache_hit, exact_seconds


def _choose(agent, env, *, arm: str, search, liveness, cache,
            consecutive_defer: int, nonprogress: int, epsilon: float,
            training: bool):
    started = perf_counter()
    if arm == "full_frontier":
        snapshot = enumerate_viability_candidates(
            env, consecutive_defer_decisions=consecutive_defer,
            search_config=search, liveness_rule=liveness, cache=cache,
        )
        state = snapshot.recovery_state
        current_certificate = snapshot.current_certificate
        candidates = tuple(snapshot.candidates)
        records = tuple(legal.masked_certified_record(item, state)
                        for item in candidates)
        misses = int(snapshot.audit.cache_misses)
        exact_seconds = float(snapshot.audit.exact_analysis_seconds)
        legal_count = (int(snapshot.audit.executable_accept_count)
                       + int(snapshot.audit.legal_recovery_count)
                       - int(snapshot.audit.recovery_executor_rejections)
                       + int(snapshot.audit.defer_candidate_count))
        require(legal_count >= len(candidates),
                "certified frontier exceeds executable legal frontier")
    else:
        state = legal.current_state_from_env(env, search)
        current_certificate, hit, exact_seconds = vc._analyze_cached(
            state, search, cache,
        )
        require(current_certificate.status is ViabilityStatus.SAFE,
                "proposal arm reached an uncertified current state")
        misses = int(not hit)
        proposals = legal.physical_frontier(
            env, current_state=state,
            current_certificate=current_certificate, search=search,
            liveness=liveness, consecutive_defer=consecutive_defer,
        )
        legal_count = len(proposals)
        candidates = proposals
        records = tuple(legal.legal_record(item, state) for item in proposals)
    require(bool(candidates), "no executable candidate")
    require(current_certificate.status is ViabilityStatus.SAFE,
            "current state has no recovery certificate")
    available = list(range(len(candidates)))
    attempts = 0
    forced_witness = bool(
        nonprogress >= agent.config.max_nonprogress_recovery_decisions
        and current_certificate.witness
    )
    if forced_witness:
        first = current_certificate.witness[0]
        matching = [i for i, item in enumerate(candidates)
                    if item.recovery_action == first]
        require(len(matching) == 1, "certified witness action is not executable")
        available = matching
    while available:
        if forced_witness:
            index = available[0]
        else:
            subset = tuple(records[i] for i in available)
            index = available[_rank_index(
                agent, subset, epsilon=epsilon, training=training,
            )]
        attempts += 1
        if arm == "full_frontier":
            candidate = candidates[index]
            record = records[index]
            break
        candidate, hit, elapsed = _certify_one(
            candidates[index], state, current_certificate, search, cache,
        )
        exact_seconds += elapsed
        misses += int(not hit)
        if candidate is not None:
            record = records[index]
            break
        available.remove(index)
    else:
        raise RuntimeError("no certified proposal survived filtering")
    agent.decision_count += 1
    agent.mode_decisions[candidate.mode.value] += 1
    agent.action_decisions[candidate.action_type.value] += 1
    backup_records = (record,) if forced_witness else records
    return candidate, record, backup_records, {
        "legal_candidates": legal_count,
        "safe_candidates": len(candidates) if arm == "full_frontier" else None,
        "attempts": attempts,
        "exact_searches": misses,
        "exact_search_seconds": exact_seconds,
        "frontier_seconds": perf_counter() - started,
        "forced_witness": forced_witness,
    }


def _append_transition(agent, previous, next_records, *, updates: int):
    started = perf_counter()
    record, reward, duration, done = previous
    transition = ViabilityGraphTransition(
        chosen=record, reward=reward, duration=duration,
        next_candidates=() if done else tuple(next_records), done=done,
    )
    agent.replay.add(transition)
    agent.transition_count += 1
    losses = []
    for _ in range(updates):
        loss = agent.learn()
        if loss is not None:
            losses.append(float(loss))
    return len(losses), perf_counter() - started


def run_episode(agent, env, *, arm: str, instance_seed: int,
                training: bool, args, initial_macro=None) -> dict:
    started = perf_counter()
    env.reset(instance=env.sample_episode_instance(instance_seed))
    env.current_episode = instance_seed
    agent.reset_episode_state()
    search = audit._search_config(args)
    liveness = audit._liveness_rule(args)
    cache = reuse.TimingInvariantCertificateCache()
    counters = Counter()
    steps = 0
    total_return = 0.0
    deviations = []
    consecutive_defer = 0
    nonprogress = 0
    previous = None
    failure = None
    action_counts = Counter()
    if initial_macro is not None:
        prelude = initial_macro(env, args, search, liveness, cache)
        require(not prelude["done"] and prelude["duration"] > 0,
                "initial macro must leave a live decision boundary")
        steps += prelude["duration"]
        total_return += prelude["raw_return"]
        action_counts[prelude["action_type"]] += 1
        counters["macros"] += 1
        counters["initial_macros"] += 1
        counters["legal_candidates"] += prelude["legal_candidates"]
        counters["attempts"] += 1
        if arm == "full_frontier":
            counters["certified_candidates"] += prelude["safe_candidates"]
        counters["exact_searches"] += prelude["exact_searches"]
        counters["exact_search_seconds"] += prelude["exact_search_seconds"]
        counters["frontier_seconds"] += prelude["frontier_seconds"]
        if training:
            previous = (prelude["record"], prelude["discounted_return"],
                        prelude["duration"], False)
    while steps < args.max_steps and not env.is_state_terminal(env.current_state):
        epsilon = (dense.epsilon_at(
            agent.decision_count, start=dense.FROZEN_EPSILON_START,
            end=dense.FROZEN_EPSILON_END,
            decay_decisions=dense.FROZEN_EPSILON_DECAY_DECISIONS,
            warmup_decisions=dense.FROZEN_EPSILON_WARMUP_DECISIONS,
        ) if training else 0.0)
        try:
            candidate, record, backup_records, diagnostics = _choose(
                agent, env, arm=arm, search=search, liveness=liveness,
                cache=cache, consecutive_defer=consecutive_defer,
                nonprogress=nonprogress, epsilon=epsilon, training=training,
            )
        except (RuntimeError, ValueError) as error:
            failure = f"selection:{type(error).__name__}:{error}"
            break
        for name in ("legal_candidates", "attempts", "exact_searches"):
            counters[name] += diagnostics[name]
        counters["exact_search_seconds"] += diagnostics["exact_search_seconds"]
        counters["frontier_seconds"] += diagnostics["frontier_seconds"]
        counters["forced_witness"] += int(diagnostics["forced_witness"])
        if diagnostics["safe_candidates"] is not None:
            counters["certified_candidates"] += diagnostics["safe_candidates"]
            counters["rejected_legal_candidates"] += (
                diagnostics["legal_candidates"] - diagnostics["safe_candidates"]
            )
        else:
            counters["rejected_proposals"] += diagnostics["attempts"] - 1
        if training and previous is not None:
            update_count, update_seconds = _append_transition(
                agent, previous, backup_records, updates=args.updates_per_macro,
            )
            counters["gradient_updates"] += update_count
            counters["learning_seconds"] += update_seconds
            counters["backup_candidates"] += len(backup_records)
            previous = None
        execution = audit._execute_audit_macro(
            env, candidate, gamma=agent.config.gamma,
            remaining_steps=args.max_steps - steps, evaluation=not training,
        )
        action_counts[candidate.action_type.value] += 1
        counters["macros"] += 1
        counters["relocations"] += execution.relocations
        counters["illegal_drops"] += execution.illegal_drops
        steps += execution.duration
        total_return += execution.raw_return
        deviations.extend(execution.delivery_deviations)
        if candidate.action_type is ViabilityActionType.DEFER:
            outcome = getattr(candidate.option, "last_outcome", None)
            observed = isinstance(outcome, dict) and outcome.get("reason") == "observed_event"
            consecutive_defer = 0 if observed else consecutive_defer + 1
        else:
            consecutive_defer = 0
        if candidate.mode is ViabilityMode.RECOVER:
            nonprogress = (0 if candidate.action_type is ViabilityActionType.DELIVER
                           else nonprogress + 1)
        elif candidate.mode is ViabilityMode.ACCEPT:
            nonprogress = 0
        done = bool(execution.env_terminal or execution.truncated
                    or not execution.option_success or steps >= args.max_steps
                    or execution.duration == 0)
        if not execution.option_success:
            failure = f"macro:{execution.failure_reason}"
        elif execution.truncated or steps >= args.max_steps and not execution.env_terminal:
            failure = "episode_step_limit"
        if training and execution.duration > 0:
            previous = (record, execution.discounted_return,
                        execution.duration, done)
        if done:
            break
    if training and previous is not None:
        update_count, update_seconds = _append_transition(
            agent, (previous[0], previous[1], previous[2], True),
            (), updates=args.updates_per_macro,
        )
        counters["gradient_updates"] += update_count
        counters["learning_seconds"] += update_seconds
    strict = bool(env.is_state_terminal(env.current_state)
                  and failure is None and counters["illegal_drops"] == 0
                  and len(deviations) == args.number_blocks)
    return {
        "arm": arm, "instance_seed": instance_seed, "training": training,
        "strict_complete": strict, "failure": failure,
        "return": total_return, "steps": steps,
        "mae": (sum(abs(x) for x in deviations) / len(deviations)
                if deviations else None),
        "within_20": (sum(abs(x) <= 20 for x in deviations) / len(deviations)
                      if deviations else None),
        "deliveries": len(deviations), "actions": dict(action_counts),
        "certificate_cache_entries": len(cache),
        "wall_seconds": perf_counter() - started, **dict(counters),
    }


def _agent(args):
    return EpisodicViabilityGraphHierarchyAgent(
        config=audit._graph_config(args, dense.FROZEN_GAMMA),
        seed=0, device="cpu", epsilon=dense.FROZEN_EPSILON_START,
    )


def train(output: Path, arm: str, episodes: int,
          through_episode: int | None) -> dict:
    require(arm in ARMS, "unknown E83 arm")
    require(0 < episodes <= 100, "pilot episode count must be in 1..100")
    stop_at = episodes if through_episode is None else through_episode
    require(0 < stop_at <= episodes, "through-episode must be in 1..episodes")
    path = output / arm
    args = runtime_args(output)
    torch.manual_seed(0)
    agent = _agent(args)
    env = audit._new_env(args, dense.FROZEN_OBJECTIVE_SPEC)
    history = []
    latest = path / "latest.pth"
    final = path / "model.pth"
    require(not final.exists(), f"E83 finalized arm cannot be overwritten: {final}")
    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        require(payload["protocol"] == PROTOCOL and payload["arm"] == arm,
                "E83 resumable checkpoint identity mismatch")
        require(payload["total_episodes"] == episodes,
                "E83 resume training budget differs")
        history = list(payload["history"])
        require(len(history) <= stop_at,
                "resume target is behind checkpoint")
        agent.load_checkpoint_state(payload["agent_state"], resumable=True)
        torch.set_rng_state(payload["torch_rng_state"])
    else:
        require(not path.exists() or not any(path.iterdir()),
                f"E83 refuses uncontracted output: {path}")
        path.mkdir(parents=True, exist_ok=True)
    with reuse.path_cleanup_active():
        for index in range(len(history), stop_at):
            row = run_episode(
                agent, env, arm=arm, instance_seed=TRAIN_SEED_BASE + index,
                training=True, args=args,
            )
            history.append(row)
            audit._atomic_torch_save({
                "protocol": PROTOCOL, "arm": arm,
                "total_episodes": episodes,
                "history": history,
                "agent_state": agent.checkpoint_state(include_replay=True),
                "torch_rng_state": torch.get_rng_state(),
            }, latest)
            print(f"E83 {arm} train {index + 1}/{episodes} | "
                  f"strict={int(row['strict_complete'])} "
                  f"steps={row['steps']} wall={row['wall_seconds']:.1f}s "
                  f"proof={row['exact_search_seconds']:.1f}s "
                  f"learn={row.get('learning_seconds', 0.0):.1f}s",
                  flush=True)
    if len(history) < episodes:
        return {"status": "paused", "arm": arm,
                "completed_training_episodes": len(history),
                "total_training_episodes": episodes,
                "latest": str(latest),
                "training_wall_seconds": sum(row["wall_seconds"] for row in history)}
    audit._atomic_torch_save({
        "protocol": PROTOCOL, "arm": arm, "episodes": episodes,
        "model_seed": 0, "network_state": agent.Q_local.state_dict(),
        "graph_config": asdict(agent.config),
        "paper_evidence": False,
    }, final)
    _atomic_json_save({"protocol": PROTOCOL, "arm": arm,
                       "episodes": episodes, "rows": history},
                      path / "training.json")
    return {"status": "trained", "arm": arm, "episodes": episodes,
            "strict_complete": sum(row["strict_complete"] for row in history),
            "training_wall_seconds": sum(row["wall_seconds"] for row in history),
            "checkpoint": str(path / "model.pth")}


def evaluate(output: Path) -> dict:
    require(not (output / "evaluation.json").exists(),
            "E83 refuses to overwrite a completed evaluation")
    args = runtime_args(output)
    rows = []
    for arm in ARMS:
        checkpoint = torch.load(output / arm / "model.pth", map_location="cpu",
                                weights_only=False)
        require(checkpoint["protocol"] == PROTOCOL and checkpoint["arm"] == arm,
                "E83 checkpoint identity mismatch")
        agent = _agent(args)
        require(checkpoint["graph_config"] == asdict(agent.config),
                "E83 frozen network configuration differs")
        agent.Q_local.load_state_dict(checkpoint["network_state"])
        agent.Q_target.load_state_dict(checkpoint["network_state"])
        agent.Q_local.eval()
        env = audit._new_env(args, dense.FROZEN_OBJECTIVE_SPEC)
        with reuse.path_cleanup_active():
            for seed in EVAL_SEEDS:
                row = run_episode(agent, env, arm="full_frontier",
                                  instance_seed=seed, training=False, args=args)
                row["trained_arm"] = arm
                rows.append(row)
                print(f"E83 eval {arm} seed={seed} "
                      f"strict={int(row['strict_complete'])} "
                      f"MAE={row['mae']}", flush=True)
    _atomic_json_save({"protocol": PROTOCOL,
                       "certified_deployment": True,
                       "rows": rows}, output / "evaluation.json")
    return {"status": "complete", "episode_runs": len(rows),
            "training_runs": 0, "evaluation": str(output / "evaluation.json")}


def analyze(output: Path) -> dict:
    evaluation = json.loads((output / "evaluation.json").read_text())
    rows = evaluation["rows"]
    require(len(rows) == len(ARMS) * len(EVAL_SEEDS), "incomplete E83 evaluation")
    summary = {}
    episode_counts = set()
    for arm in ARMS:
        training = json.loads((output / arm / "training.json").read_text())
        train_rows = training["rows"]
        episode_counts.add(len(train_rows))
        test_rows = [row for row in rows if row["trained_arm"] == arm]
        require({row["instance_seed"] for row in test_rows} == set(EVAL_SEEDS),
                "E83 held-out seed coverage differs")
        mae_rows = [row["mae"] for row in test_rows if row["mae"] is not None]
        deliveries = sum(row["deliveries"] for row in test_rows)
        summary[arm] = {
            "training_episodes": len(train_rows),
            "training_strict": sum(row["strict_complete"] for row in train_rows),
            "training_wall_seconds": sum(row["wall_seconds"] for row in train_rows),
            "training_exact_searches": sum(row["exact_searches"] for row in train_rows),
            "training_exact_search_seconds": sum(row["exact_search_seconds"] for row in train_rows),
            "training_learning_seconds": sum(row.get("learning_seconds", 0.0) for row in train_rows),
            "training_gradient_updates": sum(row.get("gradient_updates", 0) for row in train_rows),
            "training_legal_candidates": sum(row["legal_candidates"] for row in train_rows),
            "training_proposals_tested": sum(row["attempts"] for row in train_rows),
            "training_rejected_legal_candidates": sum(
                row.get("rejected_legal_candidates", 0) for row in train_rows),
            "training_rejected_proposals": sum(
                row.get("rejected_proposals", 0) for row in train_rows),
            "heldout_strict": sum(row["strict_complete"] for row in test_rows),
            "heldout_mean_mae": (sum(mae_rows) / len(mae_rows)
                                 if mae_rows else None),
            "heldout_mean_steps": sum(row["steps"] for row in test_rows) / len(test_rows),
            "heldout_rehandles_per_100": (
                100 * sum(row["relocations"] for row in test_rows) / deliveries
                if deliveries else None),
        }
    require(len(episode_counts) == 1, "E83 training arms used unequal budgets")
    report = {"protocol": PROTOCOL, "status": "complete",
              "scope": "single_seed_5x5_development_only_not_paper_evidence",
              "training_reward": "dense_base40_gamma0.99",
              "learning_target": "matched_nested_DoubleQ_full_safe_or_all_legal_next_frontier",
              "rank_and_guard_features_masked_in_both_arms": True,
              "evaluation_filter": "complete_exact_certified_frontier",
              "summary": summary,
              "caveat": "pilot compares frontier cost and certified deployment quality using matched Double-Q; it is not the original VCG training recipe",
              "training_runs": 2,
              "evaluation_episode_runs": len(rows)}
    report["unsafe_candidate_exposure"] = (
        "observed" if summary["full_frontier"]["training_rejected_legal_candidates"] > 0
        else "absent_in_control_trajectories_quality_comparison_not_decisive"
    )
    _atomic_json_save(report, output / "e83-report.json")
    return report


def self_check(output: Path) -> dict:
    args = runtime_args(output)
    env = audit._new_env(args, dense.FROZEN_OBJECTIVE_SPEC)
    env.reset(instance=env.sample_episode_instance(TRAIN_SEED_BASE))
    search = audit._search_config(args)
    liveness = audit._liveness_rule(args)
    def check_boundary():
        snapshot = enumerate_viability_candidates(
            env, consecutive_defer_decisions=0, search_config=search,
            liveness_rule=liveness, cache=reuse.TimingInvariantCertificateCache(),
        )
        state = legal.current_state_from_env(env, search)
        require(state == snapshot.recovery_state,
                "E83 legal and certified current states differ")
        proposals = legal.physical_frontier(
            env, current_state=state,
            current_certificate=snapshot.current_certificate,
            search=search, liveness=liveness, consecutive_defer=0,
        )
        by_key = {item.key: item for item in proposals}
        require(len(by_key) == len(proposals), "E83 physical keys are not unique")
        for item in snapshot.candidates:
            require(item.key in by_key,
                    f"certified candidate absent from physical frontier: {item.key}")
            expected = legal.masked_certified_record(item, state)
            actual = legal.legal_record(by_key[item.key], state)
            require(expected == actual,
                    f"E83 legal representation differs for {item.key}")
        return snapshot, len(proposals)

    with reuse.path_cleanup_active():
        first, first_legal = check_boundary()
        accepts = [item for item in first.candidates
                   if item.action_type is ViabilityActionType.ACCEPT]
        require(bool(accepts), "E83 self-check seed lacks an Accept frontier")
        macro = audit._execute_audit_macro(
            env, accepts[0], gamma=dense.FROZEN_GAMMA,
            remaining_steps=args.max_steps, evaluation=True,
        )
        require(macro.option_success and macro.duration > 0,
                "E83 self-check Accept macro did not execute")
        second, second_legal = check_boundary()
        require(any(item.mode is ViabilityMode.RECOVER for item in second.candidates),
                "E83 self-check did not reach a recovery frontier")
    return {"status": "passed", "protocol": PROTOCOL,
            "episode_runs": 0, "training_runs": 0,
            "physical_candidates": [first_legal, second_legal],
            "certified_candidates": [len(first.candidates), len(second.candidates)],
            "representation_parity": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-check", "train", "evaluate", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--through-episode", type=int)
    args = parser.parse_args()
    if args.command == "self-check":
        result = self_check(args.output)
    elif args.command == "train":
        require(args.arm is not None, "train requires --arm")
        result = train(args.output, args.arm, args.episodes, args.through_episode)
    elif args.command == "evaluate":
        result = evaluate(args.output)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
