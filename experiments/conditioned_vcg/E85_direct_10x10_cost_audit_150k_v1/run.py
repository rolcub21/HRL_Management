"""Bounded, read-only cost profile of E79-v2 on its actual 10x10 workload."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E79_original_vcg_base100_full_10x10_144k_v2 import protocol as e79
from experiments.conditioned_vcg.E83_legal_proposal_5x5_pilot_148k_v1 import legal
from experiments.conditioned_vcg.target_size_dense_stage_a_v2 import run as v2run
from experiments.conditioned_vcg.target_size_dense_stage_a_v4 import acceleration
from viability_graph_episodic_audit import EpisodicViabilityGraphHierarchyAgent
import run_vcg_objective_gamma_audit as audit
import train_vcg_dense_proper as dense


PROTOCOL = "vcg_conditioned_e85_direct_10x10_cost_150k_v1"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e85-direct-10x10-cost-150k-v1"
DEFAULT_SEEDS = (150100, 150101)


class E85Error(RuntimeError):
    pass


class _PrefixComplete(BaseException):
    """Stop before constructing a frontier beyond the declared boundary cap."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E85Error(message)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _read(path: Path) -> dict:
    require(path.is_file(), f"missing input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid JSON object: {path}")
    return value


def _write_new(path: Path, value: dict) -> None:
    require(not path.exists(), f"refusing to overwrite existing E85 result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _checkpoint_path(source_output: Path, model_seed: int) -> Path:
    return (source_output / "training/operational" / e79.SCENARIO
            / f"seed-{model_seed}/latest.pth")


def _load_agent(source_output: Path, model_seed: int, device_name: str):
    source_output = source_output.resolve()
    require(_read(source_output / "contract.json") == e79.contract(),
            "E79-v2 source contract changed; refusing unauthenticated checkpoint")
    checkpoint_path = _checkpoint_path(source_output, model_seed)
    require(checkpoint_path.is_file(), f"missing E79-v2 checkpoint: {checkpoint_path}")
    require(device_name == "cpu" or torch.cuda.is_available(),
            "CUDA requested but unavailable")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(payload.get("training_protocol") == e79.PROTOCOL
            and payload.get("method_version") == e79.METHOD_VERSION
            and payload.get("checkpoint_role") == "latest_resumable"
            and payload.get("td_backup_implementation") == e79.TD_BACKUP_IMPLEMENTATION,
            "checkpoint is not an E79-v2 latest resumable checkpoint")
    require(payload.get("resume_contract", {}).get("target_contract_sha256")
            == e79.contract()["contract_sha256"],
            "checkpoint does not authenticate the E79-v2 source contract")
    require(int(payload.get("completed_training_episodes", -1)) >= 1,
            "E85 requires a trained E79-v2 checkpoint")
    require(payload.get("config", {}).get("gamma") == 0.99,
            "checkpoint gamma differs from E79")
    agent = EpisodicViabilityGraphHierarchyAgent.from_checkpoint(
        payload, device=device_name, resumable=True, seed=model_seed,
    )
    return agent, checkpoint_path, payload


def _runtime_args(agent):
    scenario = e79.SCENARIOS[e79.SCENARIO]
    values = dict(dense.FROZEN_AUDIT_DEFAULTS)
    values.update(grid_rows=scenario.rows, grid_cols=scenario.cols,
                  number_blocks=scenario.total_jobs, arrival_rate=0.1,
                  proc_mean=80, max_steps=scenario.max_steps)
    arguments = ["--output-dir", str(DEFAULT_OUTPUT),
                 "--model-seed", "0", "--dense-b", "100"]
    for name, value in values.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
        elif value is not None:
            arguments.extend((flag, str(value)))
    args = audit.build_parser().parse_args(arguments)
    require(audit._graph_config(args, 0.99).to_dict() == agent.config.to_dict(),
            "reconstructed E79 model/search settings differ from checkpoint")
    return args


def _td_sample(agent) -> dict:
    if len(agent.replay) < agent.config.batch_size:
        return {"status": "insufficient_replay", "replay_size": len(agent.replay),
                "batch_size": agent.config.batch_size}
    timing = {"seconds": 0.0}

    def measured_batch(transitions):
        _sync(agent.device)
        started = perf_counter()
        result = acceleration.accelerated_cached_td_batch(agent, transitions)
        _sync(agent.device)
        timing["seconds"] += perf_counter() - started
        timing["candidate_records"] = sum(
            len(item.next_candidates) for item in transitions if not item.done
        )
        return result

    with patch.object(agent, "_td_batch", measured_batch):
        _sync(agent.device)
        started = perf_counter()
        loss = agent.learn()
        _sync(agent.device)
        elapsed = perf_counter() - started
    require(loss is not None, "E79 replay failed to produce a TD update")
    return {"status": "sampled_in_memory", "replay_size": len(agent.replay),
            "batch_size": agent.config.batch_size,
            "candidate_records": timing["candidate_records"],
            "td_seconds": timing["seconds"], "learn_seconds": elapsed,
            "loss": float(loss), "optimizer_or_backward_seconds": max(0.0, elapsed - timing["seconds"])}


def profile(source_output: Path, output: Path, model_seed: int,
            instance_seed: int, max_decisions: int, device_name: str) -> dict:
    require(model_seed in e79.MODEL_SEEDS, "model seed must be 0, 1, or 2")
    require(instance_seed in DEFAULT_SEEDS,
            f"development instance must be one of {DEFAULT_SEEDS}")
    require(1 <= max_decisions <= 32, "decision cap must be in 1..32")
    require(device_name in ("cpu", "cuda"), "device must be cpu or cuda")
    path = (output.resolve() / f"profile-seed-{model_seed}-instance-"
            f"{instance_seed}-decisions-{max_decisions}.json")
    require(not path.exists(), f"profile already exists: {path}")
    agent, checkpoint_path, payload = _load_agent(source_output, model_seed, device_name)
    checkpoint_sha = audit._sha256(checkpoint_path)
    args = _runtime_args(agent)
    search = audit._search_config(args)
    liveness = audit._liveness_rule(args)
    env = e79.TargetDenseEnv(e79.SCENARIO, e79.SPEC)
    env.current_episode = int(payload["completed_training_episodes"]) + 1
    rows: list[dict] = []
    selection: list[dict] = []
    executions: list[dict] = []
    original_enumerator = audit.enumerate_viability_candidates
    original_execute = audit._execute_audit_macro
    original_select = agent.select

    def enumerator(environment, **kwargs):
        if len(rows) >= max_decisions:
            raise _PrefixComplete
        _sync(agent.device)
        started = perf_counter()
        snapshot = v2run._accelerated_enumerator(original_enumerator)(
            environment, **kwargs,
        )
        _sync(agent.device)
        frontier_seconds = perf_counter() - started
        audit_row = snapshot.audit.audit_dict()
        legal_started = perf_counter()
        proposals = legal.physical_frontier(
            environment, current_state=snapshot.recovery_state,
            current_certificate=snapshot.current_certificate, search=search,
            liveness=liveness,
            consecutive_defer=kwargs["consecutive_defer_decisions"],
        )
        legal_seconds = perf_counter() - legal_started
        proposal_keys = {item.key for item in proposals}
        safe_keys = {item.key for item in snapshot.candidates}
        require(len(proposal_keys) == len(proposals)
                and len(safe_keys) == len(snapshot.candidates)
                and safe_keys <= proposal_keys,
                "physical candidate frontier failed key parity")
        by_key = {item.key: item for item in proposals}
        for candidate in snapshot.candidates:
            require(legal.masked_certified_record(candidate, snapshot.recovery_state)
                    == legal.legal_record(by_key[candidate.key], snapshot.recovery_state),
                    f"legal/certified masked features differ for {candidate.key}")
        require(len(proposals) - len(snapshot.candidates)
                == audit_row["fail_closed_rejection_count"],
                "legal minus SAFE count does not equal fail-closed rejections")
        rows.append({
            "decision": len(rows) + 1,
            "epoch": int(snapshot.decision_epoch),
            "legal_candidates": len(proposals),
            "safe_candidates": len(snapshot.candidates),
            "unsafe_candidates": audit_row["unsafe_accept_count"] + audit_row["unsafe_recovery_count"],
            "unknown_candidates": audit_row["unknown_accept_count"] + audit_row["unknown_recovery_count"],
            "frontier_seconds": frontier_seconds,
            "exact_search_seconds": audit_row["exact_analysis_seconds"],
            "relocation_family_seconds": (
                audit_row["relocation_family_setup_seconds"]
                + audit_row["relocation_family_connection_seconds"]
            ),
            "cache_misses_including_family_proofs": audit_row["cache_misses"],
            "native_recovery_searches": audit_row["native_recovery_search_count"],
            "relocation_family_proofs": audit_row["relocation_family_proof_count"],
            "cache_hits": audit_row["cache_hits"],
            "legal_shadow_seconds": legal_seconds,
            "complete_frontier_exactly_verified": audit_row["complete_frontier_exactly_verified"],
        })
        print(f"E85 boundary {len(rows)}/{max_decisions} | "
              f"legal={len(proposals)} SAFE={len(snapshot.candidates)} "
              f"proof={audit_row['exact_analysis_seconds'] + audit_row['relocation_family_setup_seconds'] + audit_row['relocation_family_connection_seconds']:.2f}s "
              f"frontier={frontier_seconds:.2f}s", flush=True)
        return snapshot

    def select(snapshot, **kwargs):
        _sync(agent.device)
        started = perf_counter()
        decision = original_select(snapshot, **kwargs)
        _sync(agent.device)
        selection.append({"decision": len(selection) + 1,
                          "seconds": perf_counter() - started,
                          "selected_key": decision.candidate.key,
                          "source": decision.selection_source})
        return decision

    def execute(environment, candidate, **kwargs):
        started = perf_counter()
        result = original_execute(environment, candidate, **kwargs)
        executions.append({"decision": len(executions) + 1,
                           "seconds": perf_counter() - started,
                           "duration": result.duration,
                           "success": result.option_success})
        return result

    started = perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(audit, "ViabilityCertificateCache",
                                         reuse.TimingInvariantCertificateCache))
        stack.enter_context(patch.object(audit, "enumerate_viability_candidates", enumerator))
        stack.enter_context(patch.object(audit, "_execute_audit_macro", execute))
        stack.enter_context(patch.object(agent, "select", select))
        stack.enter_context(reuse.path_cleanup_active())
        try:
            complete_run = audit.run_objective_audit_episode(
                agent, env, objective_spec=e79.SPEC,
                instance_seed=instance_seed, training=True,
                max_steps=e79.SCENARIOS[e79.SCENARIO].max_steps,
                search_config=search, liveness_rule=liveness,
                epsilon_start=dense.FROZEN_EPSILON_START,
                epsilon_end=dense.FROZEN_EPSILON_END,
                epsilon_decay_decisions=dense.FROZEN_EPSILON_DECAY_DECISIONS,
                epsilon_warmup_decisions=dense.FROZEN_EPSILON_WARMUP_DECISIONS,
                updates_per_macro=0,
                run_protocol=PROTOCOL,
            )
        except _PrefixComplete:
            complete_run = None
    prefix_wall = perf_counter() - started
    require(audit._sha256(checkpoint_path) == checkpoint_sha,
            "E79 checkpoint changed during profiling; timing sample is invalid")
    require(len(rows) == len(selection) == len(executions),
            "profile did not finish each selected boundary")
    require(rows, "profile captured no boundary")
    td = _td_sample(agent)
    report = {
        "protocol": PROTOCOL, "status": "complete",
        "scope": "read_only_bounded_10x10_training_like_profile_no_prefix_updates",
        "source_protocol": e79.PROTOCOL,
        "source_contract_sha256": e79.contract()["contract_sha256"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_episode": int(payload["completed_training_episodes"]),
        "model_seed": model_seed, "instance_seed": instance_seed,
        "device": device_name, "requested_max_decisions": max_decisions,
        "captured_decisions": len(rows),
        "ended_before_cap": complete_run is not None,
        "episode_strict_complete": (None if complete_run is None else complete_run["strict_method_success"]),
        "prefix_wall_seconds_including_shadow": prefix_wall,
        "frontiers": rows, "selection": selection, "executions": executions,
        "td_sample": td, "training_runs": 0,
        "episode_runs": 0 if complete_run is None else 1,
        "partial_episode_replays": 0 if complete_run is not None else 1,
        "caveat": "bounded E79-epsilon prefix with no gradient updates during the prefix, followed by one in-memory TD update; not a full E79 training episode or proposal-policy quality result",
    }
    _write_new(path, report)
    return {"status": "complete", "captured_decisions": len(rows),
            "profile": str(path), "training_runs": 0}


def _summary(profiles: list[dict]) -> dict:
    frontiers = [row for item in profiles for row in item["frontiers"]]
    selections = [row for item in profiles for row in item["selection"]]
    executions = [row for item in profiles for row in item["executions"]]
    exact = sum(row["exact_search_seconds"] for row in frontiers)
    family = sum(row["relocation_family_seconds"] for row in frontiers)
    proof = exact + family
    frontier = sum(row["frontier_seconds"] for row in frontiers)
    selection = sum(row["seconds"] for row in selections)
    execution = sum(row["seconds"] for row in executions)
    shadow = sum(row["legal_shadow_seconds"] for row in frontiers)
    wall = sum(item["prefix_wall_seconds_including_shadow"] for item in profiles)
    native = max(wall - shadow, 0.0)
    td_rows = [item["td_sample"] for item in profiles
               if item["td_sample"]["status"] == "sampled_in_memory"]
    td_mean = sum(row["learn_seconds"] for row in td_rows) / len(td_rows) if td_rows else None
    training_proxy = native + (td_mean or 0.0) * len(frontiers)
    share = proof / training_proxy if training_proxy else None
    require(proof <= frontier + 1e-6 and (share is None or share <= 1.0 + 1e-6),
            "exact-search time exceeds measured parent time")
    return {
        "profiles": len(profiles), "boundaries": len(frontiers),
        "frontier_seconds": frontier, "selection_seconds": selection,
        "primitive_execution_seconds": execution,
        "exact_search_seconds": exact,
        "relocation_family_seconds": family,
        "certification_work_seconds": proof,
        "cache_misses_including_family_proofs": sum(
            row["cache_misses_including_family_proofs"] for row in frontiers),
        "native_recovery_searches": sum(row["native_recovery_searches"] for row in frontiers),
        "relocation_family_proofs": sum(row["relocation_family_proofs"] for row in frontiers),
        "cache_hits": sum(row["cache_hits"] for row in frontiers),
        "legal_candidates": sum(row["legal_candidates"] for row in frontiers),
        "safe_candidates": sum(row["safe_candidates"] for row in frontiers),
        "unsafe_candidates": sum(row["unsafe_candidates"] for row in frontiers),
        "unknown_candidates": sum(row["unknown_candidates"] for row in frontiers),
        "legal_shadow_seconds_excluded_from_native": shadow,
        "native_online_prefix_seconds": native,
        "sampled_learn_seconds_per_decision": td_mean,
        "training_cost_proxy_seconds": training_proxy if td_mean is not None else None,
        "certification_work_fraction_of_training_proxy": share if td_mean is not None else None,
        "optimistic_max_speedup_eliminating_measured_certification_work": (
            None if share is None or td_mean is None or share >= 1.0
            else 1.0 / (1.0 - share)
        ),
    }


def analyze(output: Path) -> dict:
    output = output.resolve()
    available = [(path, _read(path)) for path in sorted(output.glob("profile-*.json"))]
    require(available, "no E85 profile files exist")
    require(all(item.get("protocol") == PROTOCOL for _, item in available),
            "mixed profile protocols")
    by_instance: dict[tuple[int, int], tuple[Path, dict]] = {}
    for path, item in available:
        key = (item["model_seed"], item["instance_seed"])
        previous = by_instance.get(key)
        if previous is None or item["captured_decisions"] > previous[1]["captured_decisions"]:
            by_instance[key] = (path, item)
    selected = list(by_instance.values())
    profiles = [item for _, item in selected]
    summary = _summary(profiles)
    expanded = all(item["captured_decisions"] >= 20 for item in profiles)
    independent = len({item["instance_seed"] for item in profiles}) >= 2
    same_checkpoint = len({item["checkpoint_sha256"] for item in profiles}) == 1
    if not expanded or not independent:
        gate = "insufficient_coverage_for_cost_decision"
    elif not same_checkpoint:
        gate = "checkpoint_mismatch_repeat_matched_profiles"
    elif summary["certification_work_fraction_of_training_proxy"] is None:
        gate = "td_replay_sample_missing"
    elif summary["certification_work_fraction_of_training_proxy"] < 0.30:
        gate = "proof_not_dominant_do_not_train_proposal_arm"
    else:
        gate = "cost_signal_only_prepare_matched_10x10_pilot"
    report = {"protocol": PROTOCOL, "status": "complete",
              "profile_files": [path.name for path, _ in selected],
              "excluded_shorter_profile_files": [
                  path.name for path, _ in available if path not in {p for p, _ in selected}
              ],
              "summary": summary, "gate": gate,
              "gate_rule": "two distinct >=20-decision development prefixes from the same checkpoint, TD sample available, measured exact-plus-relocation-family time fraction >=0.30; signal only, never automatic authorization to alter E79",
              "training_runs": 0,
              "caveat": "prefix distribution and one TD batch per checkpoint cannot establish full-episode cost or quality; validation/checkpoint I/O excluded"}
    path = output / f"e85-report-{len(available)}-profiles.json"
    _write_new(path, report)
    return {"status": "complete", "report": str(path),
            "gate": gate, "training_runs": 0}


def self_check() -> dict:
    require(e79.SCENARIO == "empty_10x10_n64"
            and e79.SPEC.dense_b == 100.0
            and e79.EPISODES_BY_SCENARIO[e79.SCENARIO] == 63,
            "E79-v2 workload/reward contract changed")
    require(legal.MASKED_NAMES[0] == "rank_before_known",
            "physical proposal feature mask changed")
    return {"status": "passed", "protocol": PROTOCOL,
            "scenario": e79.SCENARIO, "delivery_base": e79.SPEC.dense_b,
            "training_runs": 0, "episode_runs": 0}


def preflight(source_output: Path, model_seed: int, device_name: str) -> dict:
    require(model_seed in e79.MODEL_SEEDS, "model seed must be 0, 1, or 2")
    agent, checkpoint_path, payload = _load_agent(
        source_output, model_seed, device_name,
    )
    _runtime_args(agent)
    return {
        "status": "passed", "protocol": PROTOCOL,
        "checkpoint": str(checkpoint_path),
        "checkpoint_episode": int(payload["completed_training_episodes"]),
        "replay_size": len(agent.replay),
        "td_batch_available": len(agent.replay) >= agent.config.batch_size,
        "device": device_name, "training_runs": 0, "episode_runs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-check", "preflight", "profile", "analyze"))
    parser.add_argument("--source-output", type=Path, default=e79.DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--instance-seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--max-decisions", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.command == "self-check":
        result = self_check()
    elif args.command == "preflight":
        result = preflight(args.source_output, args.model_seed, args.device)
    elif args.command == "profile":
        result = profile(args.source_output, args.output, args.model_seed,
                         args.instance_seed, args.max_decisions, args.device)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
