#!/usr/bin/env python3
"""Train and gate the round-frozen 8x8 trace-return Stage-B repair."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from statistics import fmean
from time import perf_counter

import torch

import train_vcg_preference_conditioned as common
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1 import (
    run as e204,
)
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1.centered import (
    CenteredFutureHandlingNetwork,
    CenteredHandlingAgent,
    CenteredHandlingConfig,
)
from experiments.conditioned_vcg.E205_e204_paired_tradeoff_development_264k_v1 import (
    protocol as e205,
    run as e205_run,
)
from experiments.conditioned_vcg.E213_post_placement_continuation_audit_272k_v1 import (
    run as e213_run,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)
from train_viability_graph_smdp import resolve_device, seed_everything

from . import protocol as p
from .returns import diagnostics, examples_from_episode, fit_round


class E214Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E214Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E214 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E214 JSON: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _declaration(contract: dict) -> dict:
    return {
        key: contract[key]
        for key in (
            "method_reference", "method_revision", "claim_level",
            "declared_deviations", "oracle_inputs", "nonconforming_shortcuts",
            "conformance_audit",
        )
    }


def prepare(output: Path) -> dict:
    output = output.resolve()
    contract = p.contract()
    validate_protocol(contract, source=p.PROTOCOL)
    path = output / "contract.json"
    if path.is_file():
        require(_read(path) == contract, "E214 prepared contract changed")
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E214 refuses nonempty uncontracted output",
        )
        _write(path, contract)
        _write(output / "schedule.json", {
            "protocol": p.PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "training_schedule": contract["training_schedule"],
        })
    return {
        "status": "prepared",
        "contract_sha256": contract["contract_sha256"],
        "model_seed": p.MODEL_SEED,
        "rounds": p.ROUNDS,
        "training_episodes": p.TRAIN_EPISODES,
        "learned_evaluation_episodes": 9,
        "causal_replay_episodes": 2,
        "trace_eta": p.TRACE_ETA,
        "test_panel_opened": False,
        "paper_evidence": False,
    }


def authenticate(output: Path) -> dict:
    output = output.resolve()
    if not (output / "contract.json").is_file():
        prepare(output)
    contract = _read(output / "contract.json")
    require(contract.get("protocol") == p.PROTOCOL, "E214 protocol identity differs")
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    require(
        p.digest(payload) == contract.get("contract_sha256"),
        "E214 contract digest differs",
    )
    render_validated_conformance_audit(_declaration(contract))
    for relative, expected in contract["source_sha256"].items():
        require(p.sha256(p.ROOT / relative) == expected, f"E214 source changed: {relative}")
    frozen = contract["frozen_stage_a"][0]
    require(
        p.sha256(Path(frozen["checkpoint"])) == frozen["checkpoint_sha256"],
        "E214 frozen E191 checkpoint changed",
    )
    require(
        p.sha256(e205.DEFAULT_OUTPUT / "instances.json")
        == contract["endpoint_evaluation"]["instances_sha256"],
        "E214 E205 instance manifest changed",
    )
    for relative, expected in contract["endpoint_evaluation"]["immediate_only_rows"].items():
        require(
            p.sha256(p.ROOT / relative) == expected,
            f"E214 immediate-only source row changed: {relative}",
        )
    causal = contract["causal_replays"]
    require(
        p.sha256(p.ROOT / causal["source_report"])
        == causal["source_report_sha256"],
        "E214 E213 causal source report changed",
    )
    return contract


def new_agent(base) -> CenteredHandlingAgent:
    config = CenteredHandlingConfig(
        feature_dim=int(base.Q_local.q_head[0].in_features),
        hidden_dim=base.config.head_hidden_dim,
        lambda_min=0.0,
        lambda_center=10.0,
        lambda_max=40.0,
        gamma_op=base.config.gamma,
        reward_scale=base.config.reward_scale,
    )
    seed_everything(p.HEAD_SEED)
    network = CenteredFutureHandlingNetwork(config, seed=p.HEAD_SEED).to(base.device)
    network.initialize_near_zero(0.01)
    return CenteredHandlingAgent(
        base, network, config=config, seed=p.HEAD_SEED + 1, epsilon=0.0
    )


def _snapshot_network(agent):
    target = CenteredFutureHandlingNetwork(
        agent.config, seed=p.HEAD_SEED + 2
    ).to(agent.device)
    target.load_state_dict(agent.handling_network.state_dict())
    target.requires_grad_(False).eval()
    return target


def _training_dir(output: Path) -> Path:
    return output.resolve() / "training" / f"seed-{p.MODEL_SEED}"


def _checkpoint(agent, optimizer, contract, base_digest, records, role):
    frozen = contract["frozen_stage_a"][0]
    return {
        "training_protocol": p.PROTOCOL,
        "role": role,
        "contract_sha256": contract["contract_sha256"],
        "scenario": p.SCENARIO,
        "model_seed": p.MODEL_SEED,
        "completed_rounds": len(records),
        "completed_training_episodes": len(records) * p.EPISODES_PER_ROUND,
        "base_checkpoint_sha256": frozen["checkpoint_sha256"],
        "base_policy_digest": base_digest,
        "round_records": records,
        "agent_checkpoint": agent.checkpoint(
            base_checkpoint_sha256=frozen["checkpoint_sha256"],
            base_policy_digest=base_digest,
        ),
        "optimizer_state": optimizer.state_dict(),
        "test_panel_opened": False,
        "paper_evidence": False,
    }


def _load_training(output, contract, device_name):
    device = resolve_device(device_name)
    base, search, liveness, base_digest, frozen = e204.load_base(
        contract, p.MODEL_SEED, device
    )
    destination = _training_dir(output)
    latest = destination / "latest.pth"
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        expected = {
            "training_protocol": p.PROTOCOL,
            "role": "latest",
            "contract_sha256": contract["contract_sha256"],
            "model_seed": p.MODEL_SEED,
            "base_checkpoint_sha256": frozen["checkpoint_sha256"],
            "base_policy_digest": base_digest,
        }
        mismatch = {
            key: (saved.get(key), value)
            for key, value in expected.items()
            if saved.get(key) != value
        }
        require(not mismatch, f"E214 latest identity differs: {mismatch}")
        agent = CenteredHandlingAgent.from_checkpoint(
            saved["agent_checkpoint"],
            base_agent=base,
            expected_base_checkpoint_sha256=frozen["checkpoint_sha256"],
            expected_base_policy_digest=base_digest,
            seed=p.HEAD_SEED + 1,
        )
        optimizer = torch.optim.Adam(
            agent.handling_network.parameters(), lr=p.OPTIMIZER_LR
        )
        optimizer.load_state_dict(saved["optimizer_state"])
        records = list(saved["round_records"])
    else:
        require(
            not destination.exists() or not any(destination.iterdir()),
            "E214 nonempty training directory lacks latest",
        )
        agent = new_agent(base)
        optimizer = torch.optim.Adam(
            agent.handling_network.parameters(), lr=p.OPTIMIZER_LR
        )
        records = []
    return agent, optimizer, records, search, liveness, base_digest, destination


def _episode_path(destination: Path, round_number: int, offset: int) -> Path:
    return destination / "rounds" / f"round-{round_number}" / f"episode-{offset:02d}.pth"


def collect_episode(
    agent, env, coordinate, contract, destination, search, liveness, round_digest,
):
    path = _episode_path(
        destination, int(coordinate["round"]), int(coordinate["round_episode"])
    )
    identity = {
        "contract_sha256": contract["contract_sha256"],
        "round": int(coordinate["round"]),
        "round_episode": int(coordinate["round_episode"]),
        "episode": int(coordinate["episode"]),
        "instance_seed": int(coordinate["instance_seed"]),
        "workload": float(coordinate["workload"]),
        "alpha": float(coordinate["alpha"]),
        "round_start_head_sha256": round_digest,
    }
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        require(
            all(saved.get(key) == value for key, value in identity.items()),
            f"E214 episode artifact differs: {path}",
        )
        return saved["row"], tuple(saved["samples"]), False
    agent.base_agent.bind_context_environment(
        env,
        required_jobs=e191.scenario_for(p.SCENARIO).total_jobs,
        max_steps=e191.MAX_STEPS,
    )
    label = (
        f"round={coordinate['round']}/{p.ROUNDS} "
        f"episode={coordinate['round_episode']}/{p.EPISODES_PER_ROUND} "
        f"rho={100*coordinate['workload']:.0f}% alpha={coordinate['alpha']:.3f}"
    )
    print(f"E214 collect start {label}", flush=True)
    started = perf_counter()
    with e205_run.heartbeat(f"E214 {label}"), ExitStack() as stack:
        e204._runtime_stack(stack)
        result = common.run_preference_episode(
            agent,
            env,
            instance_seed=int(coordinate["instance_seed"]),
            preference_lambda=float(coordinate["alpha"]),
            preference_schedule_record={
                "behavior_alpha": float(coordinate["alpha"]),
                "round_start_head_sha256": round_digest,
                "fixed_head_for_round": True,
                "alpha_fixed_for_complete_episode": True,
                "round": int(coordinate["round"]),
                "workload": float(coordinate["workload"]),
            },
            max_steps=e191.MAX_STEPS,
            search_config=search,
            liveness_rule=liveness,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0,
            updates_per_macro=1,
        )
    samples = tuple(item.numeric_copy() for item in agent.pop_completed_samples())
    agent.base_agent.decision_log.clear()
    metrics = env.e187_metrics()
    row = {
        "round": int(coordinate["round"]),
        "round_episode": int(coordinate["round_episode"]),
        "episode": int(coordinate["episode"]),
        "instance_seed": int(coordinate["instance_seed"]),
        "workload": float(coordinate["workload"]),
        "alpha": float(coordinate["alpha"]),
        "strict_complete": bool(result["strict_method_success"]),
        "failure": result["method_failure_reason"],
        "J10": float(metrics["J"]),
        "rehandles": int(result["physical_rehandles"]),
        "steps": int(result["steps"]),
        "macro_decisions": int(result["macro_decisions"]),
        "peak_stored": int(metrics["peak_stored"]),
        "wall_seconds": perf_counter() - started,
    }
    require(
        row["strict_complete"] and len(samples) == row["macro_decisions"],
        "E214 collection did not complete",
    )
    common._atomic_torch_save({**identity, "row": row, "samples": samples}, path)
    print(
        f"E214 collect done {label} J10={row['J10']:.1f} "
        f"R={row['rehandles']} wall={row['wall_seconds']:.1f}s",
        flush=True,
    )
    return row, samples, True


def train(output: Path, through_round: int, device_name: str) -> dict:
    contract = authenticate(output)
    require(1 <= int(through_round) <= p.ROUNDS, "E214 through-round must be 1..4")
    (
        agent, optimizer, records, search, liveness, base_digest, destination,
    ) = _load_training(output, contract, device_name)
    require(len(records) <= through_round, "E214 cannot train backward")
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    new_runs = 0
    schedule = contract["training_schedule"]
    for round_index in range(len(records), through_round):
        round_number = round_index + 1
        coordinates = [row for row in schedule if row["round"] == round_number]
        round_digest = e204.state_digest(agent.handling_network.state_dict())
        target = _snapshot_network(agent)
        collection_rows = []
        all_examples = []
        for coordinate in coordinates:
            require(
                e204.state_digest(agent.handling_network.state_dict()) == round_digest,
                "E214 behavior head changed during round collection",
            )
            row, samples, executed = collect_episode(
                agent, env, coordinate, contract, destination, search,
                liveness, round_digest,
            )
            collection_rows.append(row)
            all_examples.extend(examples_from_episode(
                samples, target, trace_eta=p.TRACE_ETA, device=agent.device
            ))
            new_runs += int(executed)
        fit = fit_round(
            agent.handling_network,
            optimizer,
            all_examples,
            device=agent.device,
            epochs=p.FIT_EPOCHS_PER_ROUND,
            batch_size=p.FIT_BATCH_SIZE,
            seed=p.HEAD_SEED + round_number,
        )
        after_digest = e204.state_digest(agent.handling_network.state_dict())
        require(after_digest != round_digest, "E214 round fit did not change the head")
        require(
            e204.state_digest(agent.base_agent.Q_local.state_dict()) == base_digest,
            "E214 mutated frozen E191",
        )
        records.append({
            "round": round_number,
            "round_start_head_sha256": round_digest,
            "round_end_head_sha256": after_digest,
            "collection": collection_rows,
            "trace_fit": fit,
            "round_data_discarded_after_fit": True,
        })
        payload = _checkpoint(
            agent, optimizer, contract, base_digest, records, "latest"
        )
        common._atomic_torch_save(payload, destination / "latest.pth")
        common._atomic_torch_save(
            {**payload, "role": "endpoint", "endpoint_round": round_number},
            destination / "checkpoints" / f"round-{round_number}.pth",
        )
        _write(destination / "training-summary.json", {
            "status": "complete" if len(records) == p.ROUNDS else "paused",
            "training_protocol": p.PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "model_seed": p.MODEL_SEED,
            "completed_rounds": len(records),
            "completed_training_episodes": len(records) * p.EPISODES_PER_ROUND,
            "base_checkpoint_sha256": contract["frozen_stage_a"][0]["checkpoint_sha256"],
            "base_policy_digest": base_digest,
            "round_records": records,
            "test_panel_opened": False,
            "paper_evidence": False,
        })
        print(
            f"E214 fit round={round_number} "
            f"mcMAE={fit['before']['mc_mae']:.3f}->{fit['after']['mc_mae']:.3f} "
            f"zeroMAE={fit['after']['zero_mc_mae']:.3f} "
            f"traceMAE={fit['before']['trace_mae']:.3f}->{fit['after']['trace_mae']:.3f}",
            flush=True,
        )
    return {
        "status": "complete" if len(records) == p.ROUNDS else "paused",
        "model_seed": p.MODEL_SEED,
        "completed_rounds": len(records),
        "completed_training_episodes": len(records) * p.EPISODES_PER_ROUND,
        "new_episode_runs": new_runs,
        "latest": str(destination / "latest.pth"),
        "test_panel_opened": False,
    }


def _load_endpoint(output, contract, device_name):
    device = resolve_device(device_name)
    base, search, liveness, base_digest, frozen = e204.load_base(
        contract, p.MODEL_SEED, device
    )
    path = _training_dir(output) / "checkpoints" / f"round-{p.ROUNDS}.pth"
    require(path.is_file(), "E214 requires the round-4 endpoint")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    require(
        saved.get("role") == "endpoint"
        and saved.get("endpoint_round") == p.ROUNDS
        and saved.get("contract_sha256") == contract["contract_sha256"],
        "E214 endpoint identity differs",
    )
    agent = CenteredHandlingAgent.from_checkpoint(
        saved["agent_checkpoint"],
        base_agent=base,
        expected_base_checkpoint_sha256=frozen["checkpoint_sha256"],
        expected_base_policy_digest=base_digest,
        seed=p.HEAD_SEED + 1,
    )
    return agent, search, liveness, base_digest, path


def _row_path(output, instance_seed, value):
    return (
        output.resolve() / "evaluation" / f"instance-{int(instance_seed)}"
        / f"trace-lambda-{float(value):g}.json"
    )


def _run_evaluation(
    output, contract, agent, search, liveness, endpoint, instance_seed, value,
):
    path = _row_path(output, instance_seed, value)
    if path.is_file():
        return False
    manifest = _read(e205.DEFAULT_OUTPUT / "instances.json")
    instance = e205_run.frozen_instance(manifest, instance_seed)
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    agent.base_agent.bind_context_environment(
        env,
        required_jobs=e191.scenario_for(p.SCENARIO).total_jobs,
        max_steps=e191.MAX_STEPS,
    )
    label = f"rho={100*e205.workload_for_seed(instance_seed):.0f}% lambda={value:g}"
    print(f"E214 eval start {label}", flush=True)
    before = e204.state_digest(agent.handling_network.state_dict())
    started = perf_counter()
    with e205_run.heartbeat(f"E214 eval {label}"), ExitStack() as stack:
        e204._runtime_stack(stack)
        result = common.run_preference_episode(
            agent,
            env,
            instance_seed=int(instance_seed),
            preference_lambda=float(value),
            preference_schedule_record={
                "behavior_lambda": float(value),
                "endpoint_round": p.ROUNDS,
                "fixed_for_complete_episode": True,
            },
            max_steps=e191.MAX_STEPS,
            search_config=search,
            liveness_rule=liveness,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0,
            updates_per_macro=1,
        )
    samples = tuple(agent.pop_completed_samples())
    agent.base_agent.decision_log.clear()
    require(
        e204.state_digest(agent.handling_network.state_dict()) == before,
        "E214 evaluation mutated the head",
    )
    require(bool(result["strict_method_success"]), f"E214 evaluation failed: {result}")
    target = _snapshot_network(agent)
    examples = examples_from_episode(
        samples, target, trace_eta=p.TRACE_ETA, device=agent.device
    )
    calibration = diagnostics(agent.handling_network, examples, device=agent.device)
    metrics = env.e187_metrics()
    components = metrics["components"]
    rehandles = int(result["physical_rehandles"])
    j10 = float(metrics["J"])
    row = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "endpoint_sha256": p.sha256(endpoint),
        "model_seed": p.MODEL_SEED,
        "instance_seed": int(instance_seed),
        "instance_id": instance.instance_id,
        "workload": e205.workload_for_seed(instance_seed),
        "method": "on_policy_trace",
        "lambda": float(value),
        "strict_complete": True,
        "J10": j10,
        "J_lambda": j10 + (float(value) - 10.0) * rehandles,
        "physical_rehandles": rehandles,
        "timing_cost_2T_plus_E": (
            2.0 * float(components["storage_tardiness_accrual"])
            + float(components["storage_earliness_at_delivery"])
        ),
        "admission_wait_cost_0p5W": 0.5 * float(components["admission_wait_accrual"]),
        "equipment_cost_0p1L": 0.1 * float(components["equipment_work"]),
        "steps": int(result["steps"]),
        "peak_stored": int(metrics["peak_stored"]),
        "calibration": calibration,
        "wall_seconds": perf_counter() - started,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    require(
        env.current_episode_instance.instance_id == instance.instance_id,
        "E214 frozen instance identity differs",
    )
    _write(path, row)
    print(
        f"E214 eval done {label} Jlambda={row['J_lambda']:.1f} "
        f"R={rehandles} mcMAE={calibration['mc_mae']:.3f} "
        f"zeroMAE={calibration['zero_mc_mae']:.3f}",
        flush=True,
    )
    return True


def _causal_path(output: Path, name: str) -> Path:
    return output.resolve() / "causal" / f"{name}.json"


def _run_causal_episode(
    output, contract, *, name: str, forced_prefix: tuple[str, ...], device_name: str,
):
    path = _causal_path(output, name)
    if path.is_file():
        return False
    agent, search, liveness, base_digest, endpoint = _load_endpoint(
        output, contract, device_name
    )
    wrapped = e213_run.PrefixInterventionAgent(
        agent, forced_prefix=forced_prefix
    )
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    wrapped.base_agent.bind_context_environment(
        env,
        required_jobs=e191.scenario_for(p.SCENARIO).total_jobs,
        max_steps=e191.MAX_STEPS,
    )
    before_head = e204.state_digest(agent.handling_network.state_dict())
    started = perf_counter()
    print(f"E214 causal start {name}", flush=True)
    with e205_run.heartbeat(f"E214 causal {name}"), ExitStack() as stack:
        e204._runtime_stack(stack)
        result = common.run_preference_episode(
            wrapped,
            env,
            instance_seed=int(contract["causal_replays"]["heavy_instance_seed"]),
            preference_lambda=20.0,
            preference_schedule_record={
                "behavior_lambda": 20.0,
                "diagnostic": p.PROTOCOL,
                "forced_prefix_length": len(forced_prefix),
                "fixed_for_complete_episode": True,
            },
            max_steps=e191.MAX_STEPS,
            search_config=search,
            liveness_rule=liveness,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0,
            updates_per_macro=1,
        )
    samples = tuple(wrapped.pop_completed_samples())
    wrapped.base_agent.decision_log.clear()
    require(bool(result["strict_method_success"]), f"E214 causal episode failed: {result}")
    require(len(samples) == int(result["macro_decisions"]), "E214 causal sample count differs")
    require(
        e204.state_digest(agent.handling_network.state_dict()) == before_head
        and e204.state_digest(agent.base_agent.Q_local.state_dict()) == base_digest,
        "E214 causal replay mutated a frozen network",
    )
    metrics = env.e187_metrics()
    components = metrics["components"]
    rehandles = int(result["physical_rehandles"])
    j10 = float(metrics["J"])
    audit = wrapped.audit()
    row = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "endpoint_sha256": p.sha256(endpoint),
        "name": name,
        "strict_complete": True,
        "lambda": 20.0,
        "instance_seed": int(contract["causal_replays"]["heavy_instance_seed"]),
        "J10": j10,
        "J_lambda": j10 + 10.0 * rehandles,
        "physical_rehandles": rehandles,
        "timing_cost_2T_plus_E": (
            2.0 * float(components["storage_tardiness_accrual"])
            + float(components["storage_earliness_at_delivery"])
        ),
        "steps": int(result["steps"]),
        "macro_decisions": int(result["macro_decisions"]),
        "peak_stored": int(metrics["peak_stored"]),
        "intervention": audit,
        "decision_7_natural_key": wrapped.natural_keys[
            int(contract["causal_replays"]["divergence_index"])
        ],
        "wall_seconds": perf_counter() - started,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    if name == "forced-post-placement-continuation":
        require(
            audit["handoff_signature"]
            == contract["causal_replays"]["post_placement_handoff_signature"],
            "E214 forced post-placement state differs from E213",
        )
    _write(path, row)
    print(
        f"E214 causal done {name} key7={row['decision_7_natural_key']} "
        f"R={rehandles} J20={row['J_lambda']:.1f}",
        flush=True,
    )
    return True


def evaluate(output: Path, device_name: str) -> dict:
    contract = authenticate(output)
    agent, search, liveness, _base_digest, endpoint = _load_endpoint(
        output, contract, device_name
    )
    new_runs = 0
    for instance_seed in e205.INSTANCE_SEEDS:
        for value in p.EVAL_LAMBDAS:
            new_runs += int(_run_evaluation(
                output, contract, agent, search, liveness, endpoint,
                instance_seed, value,
            ))
    prefix = tuple(contract["causal_replays"]["forced_prefix"])
    divergence = int(contract["causal_replays"]["divergence_index"])
    new_runs += int(_run_causal_episode(
        output,
        contract,
        name="decision-7-replay",
        forced_prefix=prefix[:divergence],
        device_name=device_name,
    ))
    new_runs += int(_run_causal_episode(
        output,
        contract,
        name="forced-post-placement-continuation",
        forced_prefix=prefix,
        device_name=device_name,
    ))
    return {
        "status": "complete",
        "learned_rows": 9,
        "causal_rows": 2,
        "new_episode_runs": new_runs,
        "test_panel_opened": False,
    }


def _immediate_row(workload: float, value: float) -> dict:
    rho = int(round(100.0 * float(workload)))
    row = _read(
        p.ROOT
        / "results/vcg-conditioned-e207-e204-immediate-only-ablation-266k-v1"
        / "rows"
        / f"rho-{rho}"
        / f"lambda-{float(value):g}.json"
    )
    return {
        **row,
        "method": "immediate_only",
        "workload": float(workload),
        "lambda": float(value),
    }


def _weighted(rows, key: str) -> float:
    numerator = sum(
        row["calibration"][key] * row["calibration"]["sample_count"]
        for row in rows
    )
    denominator = sum(row["calibration"]["sample_count"] for row in rows)
    require(denominator > 0, "E214 calibration has no held-out samples")
    return float(numerator / denominator)


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    learned = [
        _read(_row_path(output, seed, value))
        for seed in e205.INSTANCE_SEEDS
        for value in p.EVAL_LAMBDAS
    ]
    immediate = [
        _immediate_row(e205.workload_for_seed(seed), value)
        for seed in e205.INSTANCE_SEEDS
        for value in (20.0, 40.0)
    ]
    comparisons = []
    for row in learned:
        if row["lambda"] == 10.0:
            continue
        baseline = next(
            item for item in immediate
            if item["workload"] == row["workload"]
            and item["lambda"] == row["lambda"]
        )
        comparisons.append({
            "instance_seed": row["instance_seed"],
            "workload": row["workload"],
            "lambda": row["lambda"],
            "trace_R": row["physical_rehandles"],
            "immediate_R": baseline["physical_rehandles"],
            "trace_timing": row["timing_cost_2T_plus_E"],
            "immediate_timing": baseline["timing_cost_2T_plus_E"],
            "trace_J_lambda": row["J_lambda"],
            "immediate_J_lambda": baseline["J_lambda"],
            "delta_trace_minus_immediate_J_lambda": (
                row["J_lambda"] - baseline["J_lambda"]
            ),
            "relative_trace_minus_immediate_J_lambda": (
                (row["J_lambda"] - baseline["J_lambda"])
                / max(abs(baseline["J_lambda"]), 1.0)
            ),
        })
    price_rows = []
    for workload in sorted({row["workload"] for row in learned}):
        selected = [row for row in learned if row["workload"] == workload]
        for price in p.EVAL_LAMBDAS:
            scores = {
                row["lambda"]: row["J10"]
                + (price - 10.0) * row["physical_rehandles"]
                for row in selected
            }
            best = min(scores.values())
            diagonal = scores[price]
            price_rows.append({
                "workload": workload,
                "price": price,
                "diagonal_regret": diagonal - best,
                "diagonal_relative_regret": (
                    (diagonal - best) / max(abs(best), 1.0)
                ),
            })
    decision = _read(_causal_path(output, "decision-7-replay"))
    forced = _read(_causal_path(output, "forced-post-placement-continuation"))
    mc_mae = _weighted(learned, "mc_mae")
    zero_mae = _weighted(learned, "zero_mc_mae")
    source = _read(p.E213_REPORT)
    forced_immediate = int(
        source["continuations"]["immediate_only_lambda20"]["physical_rehandles"]
    )
    collapse = int(source["natural_baselines"]["learned_lambda20_collapse_rehandles"])
    forced_r = int(forced["physical_rehandles"])
    gates = {
        "calibration_beats_zero": mc_mae < zero_mae,
        "decision_7_avoids_old_failed_placement": (
            decision["decision_7_natural_key"]
            != contract["causal_replays"]["old_failed_placement"]
        ),
        "forced_continuation_closer_to_immediate_than_collapse": (
            abs(forced_r - forced_immediate) < abs(forced_r - collapse)
        ),
        "frontier_adds_value_somewhere": any(
            row["delta_trace_minus_immediate_J_lambda"] < 0
            for row in comparisons
        ),
        "no_catastrophic_regression": all(
            row["relative_trace_minus_immediate_J_lambda"]
            <= p.MAX_RELATIVE_J_REGRESSION
            for row in comparisons
        ),
        "strict_completion": all(
            bool(row["strict_complete"])
            for row in learned + [decision, forced]
        ),
    }
    passed = all(gates.values())
    summary = {
        "gates": gates,
        "all_gates_passed": passed,
        "stage_b_candidate_eligible": passed,
        "sample_weighted_heldout_mc_mae": mc_mae,
        "sample_weighted_zero_head_mc_mae": zero_mae,
        "mc_mae_gain_over_zero": zero_mae - mc_mae,
        "lower_J_than_immediate_rate": fmean(
            float(row["delta_trace_minus_immediate_J_lambda"] < 0)
            for row in comparisons
        ),
        "maximum_relative_J_regression": max(
            row["relative_trace_minus_immediate_J_lambda"]
            for row in comparisons
        ),
        "price_exact_rate": fmean(
            float(math.isclose(row["diagonal_regret"], 0.0, abs_tol=1e-9))
            for row in price_rows
        ),
        "price_near_5pct_rate": fmean(
            float(row["diagonal_relative_regret"] <= 0.05)
            for row in price_rows
        ),
        "decision_7_natural_key": decision["decision_7_natural_key"],
        "forced_continuation_rehandles": forced_r,
        "forced_immediate_anchor_rehandles": forced_immediate,
        "collapse_anchor_rehandles": collapse,
    }
    report = {
        "status": "complete",
        "verdict": "candidate_ready_for_expansion" if passed else "stopped_at_failed_gate",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "summary": summary,
        "learned_rows": learned,
        "immediate_only_rows": immediate,
        "paired_comparisons": comparisons,
        "price_regret": price_rows,
        "causal_replays": {
            "decision_7": decision,
            "forced_post_placement": forced,
        },
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    _write(output.resolve() / "e214-report.json", report)
    audit = render_validated_conformance_audit(_declaration(contract))
    lines = [
        audit,
        "RESULTS",
        "=" * 72,
        "",
        f"Verdict: {report['verdict']}",
        f"Held-out MC MAE: {mc_mae:.3f} (zero head {zero_mae:.3f})",
        f"Decision-7 natural key: {decision['decision_7_natural_key']}",
        f"Forced post-placement continuation: R={forced_r} "
        f"(immediate anchor {forced_immediate}; collapse anchor {collapse})",
        f"Lower J_lambda than immediate-only: "
        f"{100*summary['lower_J_than_immediate_rate']:.1f}%",
        f"Maximum relative J_lambda regression: "
        f"{100*summary['maximum_relative_J_regression']:.1f}%",
        f"Exact lambda=w rate: {100*summary['price_exact_rate']:.1f}%",
        f"Within-5% lambda=w rate: {100*summary['price_near_5pct_rate']:.1f}%",
        "",
        "Predeclared gates:",
    ]
    lines.extend(f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in gates.items())
    lines += [
        "",
        "Development-only seed-0 repair pilot; sealed test panels remain closed.",
        "A failed gate stops expansion rather than being averaged away.",
        "",
    ]
    (output.resolve() / "REPORT.txt").write_text("\n".join(lines), encoding="utf-8")
    from .plot import create_plot
    create_plot(learned + immediate, output.resolve() / "e214-frontier.png")
    return {
        "status": "complete",
        "verdict": report["verdict"],
        **summary,
        "report": str(output.resolve() / "REPORT.txt"),
        "plot": str(output.resolve() / "e214-frontier.png"),
        "test_panel_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "train-seed", "evaluate", "analyze")
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--through-round", type=int, default=p.ROUNDS)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "train-seed":
        result = train(args.output, args.through_round, args.device)
    elif args.command == "evaluate":
        result = evaluate(args.output, args.device)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
