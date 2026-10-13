"""E84: paired legal-frontier learning with a known UNSAFE 5x5 proposal.

The geometry is the recorded E03 companion counterexample. Both arms use
E83's matched learner and still certify every executed action. This experiment
changes only the instance distribution, so the TD frontier is genuinely
different: one physically executable Accept is UNSAFE at the first boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import torch

from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    ViabilityActionType, enumerate_viability_candidates,
)
from PSLAP import viability_candidates as vc
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E83_legal_proposal_5x5_pilot_148k_v1 import legal
from experiments.conditioned_vcg.E83_legal_proposal_5x5_pilot_148k_v1 import run as e83
from experiments.conditioned_vcg.E84_unsafe_legal_5x5_pilot_149k_v1.stress import DenseStartEnv
from train_vcg_preference_conditioned import _atomic_json_save


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e84-unsafe-legal-5x5-149k-v1"
PROTOCOL = "vcg_conditioned_e84_unsafe_legal_5x5_149k_v1"
PLACEMENTS = ((1, 1), (3, 1), (2, 1), (3, 2), (3, 3), (2, 2), (2, 3))
AGENT_CELL = (2, 3)
EXPECTED_UNSAFE = "accept:B8:1:2"
TRAIN_SEED_BASE = 149_000
DENSE_EVAL_SEEDS = tuple(range(149_200, 149_205))
EMPTY_EVAL_SEEDS = tuple(range(149_300, 149_305))
EPISODES = 30


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _args(output: Path):
    return e83.runtime_args(output)


def _dense_env(args):
    return DenseStartEnv(
        placement_cells=PLACEMENTS, agent_cell=AGENT_CELL,
        timing_objective=e83.dense.FROZEN_OBJECTIVE_SPEC,
        grid_rows=5, grid_cols=5, number_blocks=8,
        choose_storage=False, arrival_rate=args.arrival_rate,
        proc_mean=args.proc_mean,
    )


def _certified_prelude(env, args, search, liveness, cache) -> dict:
    """Make one common, certified Defer transition into the unsafe frontier."""
    require(vc._strict_inbound(env) is None,
            "E84 prelude must start before the final inbound release")
    started = perf_counter()
    snapshot = enumerate_viability_candidates(
        env, consecutive_defer_decisions=0,
        search_config=search, liveness_rule=liveness, cache=cache,
    )
    require(snapshot.current_certificate.status is ViabilityStatus.SAFE,
            "E84 prelude begins outside SAFE")
    defer = [candidate for candidate in snapshot.candidates
             if candidate.action_type is ViabilityActionType.DEFER]
    require(len(defer) == 1, "E84 prelude has no unique certified Defer")
    candidate = defer[0]
    record = legal.masked_certified_record(candidate, snapshot.recovery_state)
    frontier_seconds = perf_counter() - started
    execution = e83.audit._execute_audit_macro(
        env, candidate, gamma=e83.dense.FROZEN_GAMMA,
        remaining_steps=args.max_steps, evaluation=True,
    )
    require(execution.option_success and not execution.truncated
            and not execution.env_terminal and execution.duration == 1
            and execution.illegal_drops == 0 and execution.relocations == 0
            and not execution.delivery_deviations,
            "E84 certified Defer prelude did not reach the intended boundary")
    outcome = getattr(candidate.option, "last_outcome", None)
    require(isinstance(outcome, dict) and outcome.get("reason") == "observed_event"
            and vc._strict_inbound(env) is not None,
            "E84 prelude did not release the inbound job")
    return {
        "record": record, "action_type": candidate.action_type.value,
        "raw_return": execution.raw_return,
        "discounted_return": execution.discounted_return,
        "duration": execution.duration, "done": False,
        "legal_candidates": len(snapshot.candidates),
        "safe_candidates": len(snapshot.candidates),
        "exact_searches": int(snapshot.audit.cache_misses),
        "exact_search_seconds": float(snapshot.audit.exact_analysis_seconds),
        "frontier_seconds": frontier_seconds,
    }


def exposure(output: Path) -> dict:
    args = _args(output)
    env = _dense_env(args)
    env.reset(instance=env.sample_episode_instance(TRAIN_SEED_BASE))
    search = e83.audit._search_config(args)
    liveness = e83.audit._liveness_rule(args)
    cache = reuse.TimingInvariantCertificateCache()
    with reuse.path_cleanup_active():
        prelude = _certified_prelude(env, args, search, liveness, cache)
        snapshot = enumerate_viability_candidates(
            env, consecutive_defer_decisions=0,
            search_config=search, liveness_rule=liveness, cache=cache,
        )
        require(snapshot.current_certificate.status is ViabilityStatus.SAFE,
                "E84 initial yard has no SAFE certificate")
        state = legal.current_state_from_env(env, search)
        require(state == snapshot.recovery_state,
                "E84 physical and certified states differ")
        proposals = legal.physical_frontier(
            env, current_state=state,
            current_certificate=snapshot.current_certificate,
            search=search, liveness=liveness, consecutive_defer=0,
        )
        statuses = {}
        for proposal in proposals:
            if proposal.action_type is ViabilityActionType.DEFER:
                certificate = snapshot.current_certificate
            else:
                certificate, _, _ = vc._analyze_cached(
                    proposal.successor_state, search, cache,
                )
            statuses[proposal.key] = certificate.status.value
            if certificate.status is ViabilityStatus.UNSAFE:
                require(certificate.exhaustive,
                        "E84 UNSAFE status was not exhaustively proven")
        require(len(statuses) == len(proposals), "E84 duplicate proposal keys")
        require(statuses.get(EXPECTED_UNSAFE) == ViabilityStatus.UNSAFE.value,
                "E84 expected Accept is not proven UNSAFE")
        require(ViabilityStatus.UNKNOWN.value not in statuses.values(),
                "E84 initial frontier has UNKNOWN proposals")
        safe_keys = {key for key, status in statuses.items()
                     if status == ViabilityStatus.SAFE.value}
        require(safe_keys == {candidate.key for candidate in snapshot.candidates},
                "E84 certified frontier differs from proof-classified legal frontier")
        by_key = {proposal.key: proposal for proposal in proposals}
        for candidate in snapshot.candidates:
            require(legal.masked_certified_record(candidate, state)
                    == legal.legal_record(by_key[candidate.key], state),
                    f"E84 action features differ for {candidate.key}")
    report = {
        "status": "passed", "protocol": PROTOCOL,
        "scope": "fixed_E03_geometry_development_only",
        "seed": TRAIN_SEED_BASE, "placement_cells": PLACEMENTS,
        "agent_cell": AGENT_CELL,
        "prelude": {"action_type": prelude["action_type"],
                    "duration": prelude["duration"],
                    "raw_return": prelude["raw_return"]},
        "current_status": snapshot.current_certificate.status.value,
        "proposal_statuses": statuses,
        "legal_candidates": len(proposals),
        "certified_candidates": len(snapshot.candidates),
        "unsafe_candidates": sum(status == ViabilityStatus.UNSAFE.value
                                 for status in statuses.values()),
        "unknown_candidates": 0,
        "representation_parity": True,
        "training_runs": 0, "episode_runs": 0,
    }
    path = output / "exposure.json"
    if path.exists():
        require(json.loads(path.read_text()) == json.loads(json.dumps(report)),
                "E84 exposure preflight changed")
    else:
        _atomic_json_save(report, path)
    return report


def _require_exposure(output: Path) -> None:
    path = output / "exposure.json"
    require(path.is_file(), "run E84 exposure preflight first")
    observed = json.loads(path.read_text())
    require(observed["status"] == "passed"
            and observed["protocol"] == PROTOCOL
            and observed["proposal_statuses"][EXPECTED_UNSAFE] == "UNSAFE",
            "E84 exposure preflight is invalid")


def train(output: Path, arm: str, through_episode: int) -> dict:
    require(arm in e83.ARMS, "unsupported E84 arm")
    require(1 <= through_episode <= EPISODES, "invalid E84 episode target")
    _require_exposure(output)
    args = _args(output)
    path = output / arm
    latest = path / "latest.pth"
    final = path / "model.pth"
    require(not final.exists(), f"E84 finalized arm cannot be overwritten: {final}")
    torch.manual_seed(0)
    agent = e83._agent(args)
    env = _dense_env(args)
    history = []
    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        require(payload["protocol"] == PROTOCOL and payload["arm"] == arm
                and payload["total_episodes"] == EPISODES,
                "E84 checkpoint identity or budget differs")
        history = list(payload["history"])
        require(len(history) <= through_episode,
                "E84 resume target is behind checkpoint")
        agent.load_checkpoint_state(payload["agent_state"], resumable=True)
        torch.set_rng_state(payload["torch_rng_state"])
    else:
        require(not path.exists() or not any(path.iterdir()),
                f"E84 refuses uncontracted output: {path}")
        path.mkdir(parents=True, exist_ok=True)
    with reuse.path_cleanup_active():
        for index in range(len(history), through_episode):
            row = e83.run_episode(
                agent, env, arm=arm, instance_seed=TRAIN_SEED_BASE + index,
                training=True, args=args, initial_macro=_certified_prelude,
            )
            if index == 0:
                first = agent.replay.memory[0]
                first_keys = [record.key for record in first.next_candidates]
                require(first.chosen.action_type == ViabilityActionType.DEFER.value
                        and (EXPECTED_UNSAFE in first_keys)
                        == (arm == "rank_then_certify")
                        and len(first_keys) == (5 if arm == "rank_then_certify" else 4),
                        "E84 first TD backup did not expose the intended frontier")
                row["first_td_backup_keys"] = first_keys
            if not row["strict_complete"]:
                _atomic_json_save(row, path / f"failed-episode-{index + 1}.json")
                raise RuntimeError(
                    f"E84 training episode {index + 1} did not strictly complete; "
                    "diagnostic row was retained"
                )
            history.append(row)
            e83.audit._atomic_torch_save({
                "protocol": PROTOCOL, "arm": arm,
                "total_episodes": EPISODES, "history": history,
                "agent_state": agent.checkpoint_state(include_replay=True),
                "torch_rng_state": torch.get_rng_state(),
            }, latest)
            print(f"E84 {arm} train {index + 1}/{EPISODES} | "
                  f"unsafe={row.get('rejected_legal_candidates', 0)} "
                  f"rejected_proposals={row.get('rejected_proposals', 0)} "
                  f"strict=1 wall={row['wall_seconds']:.1f}s", flush=True)
    if len(history) < EPISODES:
        return {"status": "paused", "arm": arm,
                "completed_training_episodes": len(history),
                "training_runs": 1}
    e83.audit._atomic_torch_save({
        "protocol": PROTOCOL, "arm": arm, "episodes": EPISODES,
        "model_seed": 0, "network_state": agent.Q_local.state_dict(),
        "graph_config": asdict(agent.config), "paper_evidence": False,
    }, final)
    _atomic_json_save({"protocol": PROTOCOL, "arm": arm,
                       "episodes": EPISODES, "rows": history},
                      path / "training.json")
    return {"status": "trained", "arm": arm,
            "completed_training_episodes": EPISODES,
            "training_runs": 1, "checkpoint": str(final)}


def evaluate(output: Path) -> dict:
    _require_exposure(output)
    path = output / "evaluation.json"
    require(not path.exists(), "E84 refuses to overwrite evaluation")
    args = _args(output)
    rows = []
    for arm in e83.ARMS:
        payload = torch.load(output / arm / "model.pth", map_location="cpu",
                             weights_only=False)
        require(payload["protocol"] == PROTOCOL and payload["arm"] == arm,
                "E84 checkpoint identity differs")
        agent = e83._agent(args)
        require(payload["graph_config"] == asdict(agent.config),
                "E84 network configuration differs")
        agent.Q_local.load_state_dict(payload["network_state"])
        agent.Q_target.load_state_dict(payload["network_state"])
        agent.Q_local.eval()
        with reuse.path_cleanup_active():
            for panel, seeds in (("dense", DENSE_EVAL_SEEDS),
                                 ("empty_start", EMPTY_EVAL_SEEDS)):
                env = (_dense_env(args) if panel == "dense" else
                       e83.audit._new_env(args, e83.dense.FROZEN_OBJECTIVE_SPEC))
                for seed in seeds:
                    row = e83.run_episode(
                        agent, env, arm="full_frontier", instance_seed=seed,
                        training=False, args=args,
                        initial_macro=(_certified_prelude if panel == "dense" else None),
                    )
                    row.update(trained_arm=arm, panel=panel)
                    rows.append(row)
                    print(f"E84 eval {arm} {panel} seed={seed} "
                          f"strict={int(row['strict_complete'])} "
                          f"MAE={row['mae']}", flush=True)
    _atomic_json_save({"protocol": PROTOCOL, "certified_deployment": True,
                       "rows": rows}, path)
    return {"status": "complete", "episode_runs": len(rows),
            "training_runs": 0, "evaluation": str(path)}


def analyze(output: Path) -> dict:
    _require_exposure(output)
    evaluation = json.loads((output / "evaluation.json").read_text())
    require(evaluation["protocol"] == PROTOCOL
            and evaluation["certified_deployment"],
            "E84 evaluation identity differs")
    rows = evaluation["rows"]
    require(len(rows) == 2 * len(e83.ARMS) * len(DENSE_EVAL_SEEDS),
            "E84 evaluation is incomplete")
    summary = {}
    for arm in e83.ARMS:
        training = json.loads((output / arm / "training.json").read_text())
        train_rows = training["rows"]
        require(training["protocol"] == PROTOCOL and training["arm"] == arm
                and len(train_rows) == EPISODES,
                "E84 training identity or episode count differs")
        first_keys = train_rows[0].get("first_td_backup_keys", [])
        require((EXPECTED_UNSAFE in first_keys) == (arm == "rank_then_certify")
                and len(first_keys) == (5 if arm == "rank_then_certify" else 4),
                "E84 saved first TD backup lost unsafe-exposure parity")
        arm_summary = {
            "first_td_backup_keys": first_keys,
            "training_strict": sum(row["strict_complete"] for row in train_rows),
            "training_wall_seconds": sum(row["wall_seconds"] for row in train_rows),
            "training_exact_searches": sum(row["exact_searches"] for row in train_rows),
            "training_exact_search_seconds": sum(row["exact_search_seconds"] for row in train_rows),
            "training_learning_seconds": sum(row.get("learning_seconds", 0.0)
                                             for row in train_rows),
            "training_rejected_legal_candidates": sum(
                row.get("rejected_legal_candidates", 0) for row in train_rows),
            "training_rejected_proposals": sum(
                row.get("rejected_proposals", 0) for row in train_rows),
        }
        for panel, seeds in (("dense", DENSE_EVAL_SEEDS),
                             ("empty_start", EMPTY_EVAL_SEEDS)):
            panel_rows = [row for row in rows if row["trained_arm"] == arm
                          and row["panel"] == panel]
            require({row["instance_seed"] for row in panel_rows} == set(seeds),
                    f"E84 {arm} {panel} seed coverage differs")
            strict_rows = [row for row in panel_rows if row["strict_complete"]]
            strict_deliveries = sum(row["deliveries"] for row in strict_rows)
            arm_summary[panel] = {
                "strict_complete": len(strict_rows),
                "mean_mae_strict": (sum(row["mae"] for row in strict_rows) /
                                    len(strict_rows) if strict_rows else None),
                "mean_return_strict": (sum(row["return"] for row in strict_rows) /
                                       len(strict_rows) if strict_rows else None),
                "mean_steps_all": sum(row["steps"] for row in panel_rows) / len(panel_rows),
                "mean_steps_strict": (sum(row["steps"] for row in strict_rows) /
                                      len(strict_rows) if strict_rows else None),
                "rehandles_per_100_strict": (
                    100 * sum(row["relocations"] for row in strict_rows) /
                    strict_deliveries if strict_deliveries else None),
                "rejected_legal_candidates": sum(
                    row.get("rejected_legal_candidates", 0) for row in panel_rows),
            }
        summary[arm] = arm_summary
    report = {
        "status": "complete", "protocol": PROTOCOL,
        "scope": "single_seed_dense_5x5_development_only_not_paper_evidence",
        "training_reward": "dense_base40_gamma0.99",
        "learning_target": "matched_nested_DoubleQ_full_safe_or_all_legal_next_frontier",
        "evaluation_filter": "complete_exact_certified_frontier",
        "exposure_at_first_boundary": json.loads((output / "exposure.json").read_text()),
        "summary": summary, "training_runs": 2,
        "evaluation_episode_runs": len(rows),
        "caveat": "fixed E03 geometry and one model seed; quality is descriptive and cannot establish larger-yard equivalence",
    }
    _atomic_json_save(report, output / "e84-report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("exposure", "train", "evaluate", "analyze"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arm", choices=e83.ARMS)
    parser.add_argument("--through-episode", type=int, default=EPISODES)
    args = parser.parse_args()
    if args.command == "exposure":
        result = exposure(args.output)
    elif args.command == "train":
        require(args.arm is not None, "train requires --arm")
        result = train(args.output, args.arm, args.through_episode)
    elif args.command == "evaluate":
        result = evaluate(args.output)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
