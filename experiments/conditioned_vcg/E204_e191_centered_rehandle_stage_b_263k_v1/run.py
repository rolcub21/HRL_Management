#!/usr/bin/env python3
"""Fit E191's development-only centered rehandle consequence head."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
from time import perf_counter
from unittest.mock import patch

import torch

import train_vcg_preference_conditioned as common
from PSLAP.viability import ViabilityStatus
from PSLAP.viability_candidates import (
    BoundedEventDeferRule,
    ViabilityCertificateCache,
)
from PSLAP.viability_filter import ViabilitySearchConfig
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    representation as e107_representation,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
    representation as e191_representation,
    stage as e191_stage,
)
from train_viability_graph_smdp import resolve_device, seed_everything

from . import protocol as p
from .centered import (
    CenteredFutureHandlingNetwork,
    CenteredHandlingAgent,
    CenteredHandlingConfig,
    centered_calibration,
    fit_centered_future_handling,
)


class E204Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E204Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E204 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E204 JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def prepare(output: Path) -> dict:
    output = output.resolve()
    expected = p.contract()
    path = output / "contract.json"
    if path.is_file():
        require(_read(path) == expected, "E204 contract or frozen sources changed")
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E204 refuses a nonempty uncontracted output",
        )
        _write(path, expected)
    return {
        "status": "prepared",
        "contract_sha256": expected["contract_sha256"],
        "model_seeds": list(p.MODEL_SEEDS),
        "lambda_values": list(p.LAMBDAS),
        "lambda_center": p.LAMBDA_CENTER,
        "rounds": p.ROUNDS,
        "episodes_per_seed": p.ROUNDS * p.EPISODES_PER_ROUND,
        "episode_runs": 0,
        "training_runs": 0,
        "test_panel_opened": False,
    }


def authenticate(output: Path) -> dict:
    observed = _read(output.resolve() / "contract.json")
    expected = p.contract()
    require(observed == expected, "E204 contract or frozen sources changed")
    return expected


def _search(payload: dict) -> ViabilitySearchConfig:
    value = payload["viability_search"]
    return ViabilitySearchConfig(
        max_depth=value["max_depth"],
        max_nodes=int(value["max_nodes"]),
        max_primitive_steps=value["max_primitive_steps"],
        reserve_queue_cells=bool(value["reserve_queue_cells"]),
        search_order=str(value["search_order"]),
    )


def _liveness(payload: dict) -> BoundedEventDeferRule:
    value = payload["liveness_rule"]
    return BoundedEventDeferRule(
        max_option_steps=int(value["max_option_steps"]),
        max_consecutive_defer_decisions=int(
            value["max_consecutive_defer_decisions"]
        ),
    )


def load_base(contract: dict, model_seed: int, device: torch.device):
    source = next(
        row for row in contract["frozen_stage_a"]
        if int(row["model_seed"]) == int(model_seed)
    )
    checkpoint = Path(source["checkpoint"])
    require(
        checkpoint.is_file() and p.sha256(checkpoint) == source["checkpoint_sha256"],
        f"E204 frozen E191 checkpoint changed for seed {model_seed}",
    )
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    expected = {
        "training_protocol": e191.PROTOCOL,
        "model_seed": model_seed,
        "checkpoint_role": "best_candidate_unfinalized",
        "completed_training_episodes": int(source["selected_episode"]),
        "trainer_resumable": False,
        "test_panels_opened": False,
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    require(not mismatch, f"E204 frozen E191 checkpoint identity differs: {mismatch}")
    search = _search(payload)
    liveness = _liveness(payload)
    base = e191_representation.FreshChoicePreservingQueueAgent.from_checkpoint(
        payload, device=device, resumable=False, seed=model_seed
    )
    del payload
    gc.collect()
    base.set_epsilon(0.0)
    for network in (base.Q_local, base.Q_target):
        network.requires_grad_(False).eval()
    return base, search, liveness, state_digest(base.Q_local.state_dict()), source


def new_agent(base, model_seed: int) -> CenteredHandlingAgent:
    config = CenteredHandlingConfig(
        feature_dim=int(base.Q_local.q_head[0].in_features),
        hidden_dim=base.config.head_hidden_dim,
        lambda_min=min(p.LAMBDAS),
        lambda_center=p.LAMBDA_CENTER,
        lambda_max=max(p.LAMBDAS),
        gamma_op=base.config.gamma,
        reward_scale=base.config.reward_scale,
    )
    seed_everything(263_204_700 + int(model_seed))
    network = CenteredFutureHandlingNetwork(
        config, seed=263_204_700 + int(model_seed)
    ).to(base.device)
    network.initialize_near_zero(0.01)
    return CenteredHandlingAgent(
        base, network, config=config, seed=263_204_800 + int(model_seed), epsilon=0.0
    )


def _execute_gamma_one(
    env,
    candidate,
    *,
    gamma: float,
    remaining_steps: int,
    evaluation: bool,
):
    """The historical exact macro executor, extended to the valid gamma=1 case."""

    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must satisfy 0 <= gamma <= 1")
    if (
        isinstance(remaining_steps, bool)
        or not isinstance(remaining_steps, int)
        or remaining_steps <= 0
    ):
        raise ValueError("remaining_steps must be a positive integer")
    if candidate.certificate.status is not ViabilityStatus.SAFE:
        raise ValueError("only exactly SAFE candidates may be executed")
    option = candidate.option
    if not bool(option.initiation(env.get_current_state())):
        return common.PreferenceMacroExecution(
            candidate_key=candidate.key,
            action_type=candidate.action_type.value,
            mode=candidate.mode.value,
            discounted_operational_return=0.0,
            raw_operational_return=0.0,
            discounted_rehandles=0.0,
            raw_rehandles=0,
            duration=0,
            env_terminal=False,
            truncated=False,
            option_terminated=False,
            option_success=False,
            failure_reason="bound_option_not_initiable",
            delivery_deviations=(),
            illegal_drops=0,
        )
    discounted_operational = 0.0
    raw_operational = 0.0
    discounted_rehandles = 0.0
    raw_rehandles = 0
    deviations = []
    illegal_drops = 0
    terminal = False
    terminated = False
    runtime_failure = None
    duration = 0
    while duration < remaining_steps and not terminal and not terminated:
        state = env.get_current_state()
        try:
            action = option.policy(state, test=evaluation)
            next_state, reward, terminal, info = env.step(action)
            duration += 1
            reward = float(reward)
            primitive_discount = gamma ** (duration - 1)
            discounted_operational += primitive_discount * reward
            raw_operational += reward
            relocated = int(bool(info.get("relocated_block")))
            discounted_rehandles += primitive_discount * relocated
            raw_rehandles += relocated
            if "delivery_error_time" in info:
                deviations.append(float(info["delivery_error_time"]))
            illegal_drops += int(bool(info.get("illegal_drop")))
            terminated = bool(option.termination(next_state))
        except RuntimeError as error:
            runtime_failure = f"macro_runtime:{type(error).__name__}:{error}"
            break
    truncated = bool(duration >= remaining_steps and not terminal and not terminated)
    failure = runtime_failure or common._option_failure(option)
    if terminal and not terminated and failure is None:
        failure = "environment_terminated_before_option_contract"
    if truncated and failure is None:
        failure = "step_limit_during_bound_macro"
    return common.PreferenceMacroExecution(
        candidate_key=candidate.key,
        action_type=candidate.action_type.value,
        mode=candidate.mode.value,
        discounted_operational_return=float(discounted_operational),
        raw_operational_return=float(raw_operational),
        discounted_rehandles=float(discounted_rehandles),
        raw_rehandles=int(raw_rehandles),
        duration=int(duration),
        env_terminal=bool(terminal),
        truncated=truncated,
        option_terminated=terminated,
        option_success=failure is None and (terminated or terminal),
        failure_reason=failure,
        delivery_deviations=tuple(deviations),
        illegal_drops=int(illegal_drops),
    )


def _runtime_stack(stack: ExitStack) -> None:
    stack.enter_context(patch.object(
        common,
        "enumerate_viability_candidates",
        e191_stage._accelerated_enumerator(common.enumerate_viability_candidates),
    ))
    stack.enter_context(patch.object(
        common, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache
    ))
    stack.enter_context(patch.object(common, "OBJECTIVE_SPEC", e191.SPEC))
    stack.enter_context(patch.object(
        common, "execute_preference_macro", _execute_gamma_one
    ))
    stack.enter_context(reuse.path_cleanup_active())
    stack.enter_context(fast_inputs.enabled())
    stack.enter_context(e107_representation.acceleration_enabled())


def verify_base(output: Path, model_seed: int, device_name: str) -> dict:
    contract = authenticate(output)
    device = resolve_device(device_name)
    base, search, liveness, digest, source = load_base(
        contract, model_seed, device
    )
    require(base.config.gamma == 1.0, "E204 requires E191 gamma=1")
    require(base.config.reward_scale == 0.01, "E204 requires E191 reward scale 0.01")
    agent = new_agent(base, model_seed)
    with torch.no_grad():
        features = torch.zeros((len(p.LAMBDAS), agent.config.feature_dim), device=device)
        values = agent.handling_network(
            features, torch.tensor(p.LAMBDAS, device=device)
        )
    require(
        values.shape == (len(p.LAMBDAS),) and bool(torch.isfinite(values).all()),
        "E204 handling head is incompatible with frozen E191",
    )
    return {
        "status": "passed",
        "model_seed": model_seed,
        "selected_episode": source["selected_episode"],
        "device": str(device),
        "base_policy_digest": digest,
        "base_checkpoint_sha256": source["checkpoint_sha256"],
        "gamma": base.config.gamma,
        "reward_scale": base.config.reward_scale,
        "lambda_center": agent.config.lambda_center,
        "search_order": str(search.search_order),
        "max_consecutive_defer": liveness.max_consecutive_defer_decisions,
        "episode_runs": 0,
        "training_runs": 0,
    }


def verify_center_parity(output: Path, model_seed: int, device_name: str) -> dict:
    """Compare the first real E191 decision at lambda=10 without an episode."""

    contract = authenticate(output)
    device = resolve_device(device_name)
    centered_base, search, liveness, _, _ = load_base(contract, model_seed, device)
    reference, _, _, _, _ = load_base(contract, model_seed, device)
    agent = new_agent(centered_base, model_seed)
    coordinate = next(
        item for item in p.collection_coordinates(model_seed, 0)
        if item["lambda"] == p.LAMBDA_CENTER and item["workload_index"] == 0
    )
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    instance = env.sample_episode_instance(coordinate["instance_seed"])
    env.reset(instance=instance)
    centered_base.bind_context_environment(
        env, required_jobs=len(env.blocks), max_steps=e191.MAX_STEPS
    )
    reference.bind_context_environment(
        env, required_jobs=len(env.blocks), max_steps=e191.MAX_STEPS
    )
    centered_base.reset_episode_state()
    reference.reset_episode_state()
    with ExitStack() as stack:
        _runtime_stack(stack)
        snapshot = common.enumerate_viability_candidates(
            env,
            consecutive_defer_decisions=0,
            search_config=search,
            liveness_rule=liveness,
            cache=ViabilityCertificateCache(),
            state_prioritizer=None,
        )
        reference_decision = reference.select(
            snapshot, training=False, epsilon=0.0
        )
        centered_decision = agent.select(
            snapshot,
            preference_lambda=p.LAMBDA_CENTER,
            training=False,
            epsilon=0.0,
        )
    require(
        centered_decision.candidate.key == reference_decision.candidate.key,
        "E204 lambda=10 selected another action",
    )
    require(
        centered_decision.selection_source == reference_decision.selection_source
        and centered_decision.base_decision.q_values == reference_decision.q_values,
        "E204 lambda=10 changed E191 selection inputs",
    )
    selected_index = next(
        index
        for index, record in enumerate(reference_decision.prepared_snapshot.records)
        if record.key == reference_decision.record.key
    )
    selected_q_delta = abs(
        centered_decision.operational_values[selected_index]
        - reference_decision.q_values[selected_index]
    )
    require(
        selected_q_delta <= 1.0e-5,
        "E204 full E191 feature extractor changed the selected Q value "
        f"by {selected_q_delta}",
    )
    off_center = agent.select(
        snapshot,
        preference_lambda=max(p.LAMBDAS),
        training=False,
        epsilon=0.0,
    )
    require(
        off_center.candidate.certificate.status is ViabilityStatus.SAFE,
        "E204 off-center selector escaped the certified frontier",
    )
    return {
        "status": "passed",
        "model_seed": model_seed,
        "instance_seed": coordinate["instance_seed"],
        "candidate_key": centered_decision.candidate.key,
        "selection_source": centered_decision.selection_source,
        "q_values_bitwise_equal": True,
        "full_feature_selected_q_absolute_delta": selected_q_delta,
        "full_feature_selected_q_within_roundoff": True,
        "off_center_smoke_lambda": max(p.LAMBDAS),
        "off_center_smoke_candidate_key": off_center.candidate.key,
        "off_center_candidate_exact_safe": True,
        "lambda_center_exact_e191_delegation": True,
        "episode_runs": 0,
        "training_runs": 0,
    }


def stage_b_dir(output: Path, model_seed: int) -> Path:
    return output.resolve() / "training/handling" / p.SCENARIO / f"seed-{model_seed}"


def episode_file(destination: Path, round_index: int, offset: int) -> Path:
    return (
        destination / "collection" / f"round-{round_index + 1:02d}"
        / f"episode-{offset + 1:02d}.pth"
    )


def _checkpoint(
    agent: CenteredHandlingAgent,
    contract: dict,
    model_seed: int,
    base_digest: str,
    rounds: list[dict],
    role: str,
) -> dict:
    source = next(
        row for row in contract["frozen_stage_a"]
        if int(row["model_seed"]) == int(model_seed)
    )
    return {
        "training_protocol": p.PROTOCOL,
        "role": role,
        "contract_sha256": contract["contract_sha256"],
        "scenario": p.SCENARIO,
        "model_seed": model_seed,
        "base_checkpoint_sha256": source["checkpoint_sha256"],
        "base_policy_digest": base_digest,
        "completed_rounds": len(rounds),
        "round_records": rounds,
        "agent_checkpoint": agent.checkpoint(
            base_checkpoint_sha256=source["checkpoint_sha256"],
            base_policy_digest=base_digest,
        ),
        "development_only": True,
        "test_panel_opened": False,
        "paper_evidence": False,
    }


def collect_episode(
    agent,
    env,
    *,
    contract,
    destination,
    model_seed,
    round_index,
    offset,
    coordinate,
    start_digest,
    search,
    liveness,
):
    path = episode_file(destination, round_index, offset)
    expected = {
        "contract_sha256": contract["contract_sha256"],
        "model_seed": model_seed,
        "round": round_index + 1,
        "offset": offset + 1,
        "instance_seed": coordinate["instance_seed"],
        "lambda": coordinate["lambda"],
        "workload": coordinate["workload"],
        "validation": coordinate["validation"],
        "round_start_network_digest": start_digest,
    }
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        require(
            all(saved.get(key) == value for key, value in expected.items()),
            f"E204 saved collection coordinate differs: {path}",
        )
        require(
            saved["row"]["strict_safe_complete"]
            and len(saved["samples"]) == saved["row"]["macro_decisions"],
            f"E204 saved collection is incomplete: {path}",
        )
        return saved["row"], tuple(saved["samples"]), False

    agent.base_agent.bind_context_environment(
        env, required_jobs=e191_scenario_jobs(), max_steps=e191.MAX_STEPS
    )
    started = perf_counter()
    with ExitStack() as stack:
        _runtime_stack(stack)
        run = common.run_preference_episode(
            agent,
            env,
            instance_seed=coordinate["instance_seed"],
            preference_lambda=coordinate["lambda"],
            preference_schedule_record={
                "behavior_lambda": coordinate["lambda"],
                "lambda_center": p.LAMBDA_CENTER,
                "round": round_index + 1,
                "episode": offset + 1,
                "workload": coordinate["workload"],
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
    metrics = env.e187_metrics()
    row = {
        "seed": coordinate["instance_seed"],
        "lambda": coordinate["lambda"],
        "workload": coordinate["workload"],
        "validation": coordinate["validation"],
        "strict_safe_complete": bool(run["strict_method_success"]),
        "failure": run["method_failure_reason"],
        "J10": float(metrics["J"]),
        "return": float(run["return"]),
        "steps": int(run["steps"]),
        "rehandles": int(run["physical_rehandles"]),
        "macro_decisions": int(run["macro_decisions"]),
        "instance_id": run["episode_instance_id"],
        "peak_stored": int(metrics["peak_stored"]),
        "mean_stored": float(metrics["mean_stored"]),
        "admission_wait": float(metrics["total_admission_wait"]),
        "wall_seconds": perf_counter() - started,
    }
    if not row["strict_safe_complete"]:
        _write(destination / "failed-collection.json", {**expected, "row": row})
        raise E204Error(
            "Stage B needs a completed Monte Carlo continuation; collection failed"
        )
    samples = tuple(item.numeric_copy() for item in agent.pop_completed_samples())
    require(
        len(samples) == row["macro_decisions"],
        "E204 sample count differs from macro decisions",
    )
    agent.base_agent.decision_log.clear()
    common._atomic_torch_save({**expected, "row": row, "samples": samples}, path)
    return row, samples, True


def e191_scenario_jobs() -> int:
    return int(e191.scenario_for(p.SCENARIO).total_jobs)


def train_seed(
    output: Path,
    model_seed: int,
    *,
    through_round: int,
    device_name: str,
) -> dict:
    contract = authenticate(output)
    require(model_seed in p.MODEL_SEEDS, "E204 model seed must be 0, 1, or 2")
    require(1 <= through_round <= p.ROUNDS, "E204 through-round must be 1..4")
    device = resolve_device(device_name)
    base, search, liveness, base_digest, source = load_base(
        contract, model_seed, device
    )
    destination = stage_b_dir(output, model_seed)
    latest = destination / "latest.pth"
    terminal = destination / "terminal.pth"
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        expected = {
            "training_protocol": p.PROTOCOL,
            "role": "latest",
            "contract_sha256": contract["contract_sha256"],
            "scenario": p.SCENARIO,
            "model_seed": model_seed,
            "base_checkpoint_sha256": source["checkpoint_sha256"],
            "base_policy_digest": base_digest,
        }
        mismatch = {
            key: (saved.get(key), value)
            for key, value in expected.items()
            if saved.get(key) != value
        }
        require(not mismatch, f"E204 latest checkpoint differs: {mismatch}")
        rounds = list(saved["round_records"])
        require(
            len(rounds) == int(saved["completed_rounds"]),
            "E204 latest checkpoint round count differs",
        )
        agent = CenteredHandlingAgent.from_checkpoint(
            saved["agent_checkpoint"],
            base_agent=base,
            expected_base_checkpoint_sha256=source["checkpoint_sha256"],
            expected_base_policy_digest=base_digest,
            seed=263_204_800 + model_seed,
        )
    else:
        require(
            not destination.exists()
            or all(child.name == "collection" for child in destination.iterdir()),
            "E204 nonempty handling output lacks latest checkpoint",
        )
        rounds = []
        agent = new_agent(base, model_seed)
    require(len(rounds) <= through_round, "E204 cannot train backward")
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    new_episode_runs = 0
    for round_index in range(len(rounds), through_round):
        start_digest = state_digest(agent.handling_network.state_dict())
        rows = []
        episode_samples = []
        coordinates = p.collection_coordinates(model_seed, round_index)
        for offset, coordinate in enumerate(coordinates):
            row, samples, executed = collect_episode(
                agent,
                env,
                contract=contract,
                destination=destination,
                model_seed=model_seed,
                round_index=round_index,
                offset=offset,
                coordinate=coordinate,
                start_digest=start_digest,
                search=search,
                liveness=liveness,
            )
            rows.append(row)
            episode_samples.append(samples)
            new_episode_runs += int(executed)
            print(
                f"E204 seed={model_seed} round={round_index + 1}/{p.ROUNDS} "
                f"episode={offset + 1}/{p.EPISODES_PER_ROUND} "
                f"lambda={coordinate['lambda']:g} "
                f"rho={100 * coordinate['workload']:.0f}% "
                f"{'run' if executed else 'reused'} J10={row['J10']:.1f} "
                f"R={row['rehandles']}",
                flush=True,
            )
        validation_indices = {
            index for index, value in enumerate(coordinates) if value["validation"]
        }
        training = tuple(
            sample
            for index, episode in enumerate(episode_samples)
            if index not in validation_indices
            for sample in episode
        )
        validation = tuple(
            sample
            for index, episode in enumerate(episode_samples)
            if index in validation_indices
            for sample in episode
        )
        prior_state = {
            key: value.detach().clone()
            for key, value in agent.handling_network.state_dict().items()
        }
        fit = fit_centered_future_handling(
            agent.handling_network,
            training,
            validation,
            device=device,
            epochs=8,
            batch_size=256,
            learning_rate=5.0e-4,
            seed=263_205_000 + model_seed * 100 + round_index,
        )
        fraction = 1.0 if round_index < p.ROUNDS - 1 else p.LATE_UPDATE_FRACTION
        if fraction != 1.0:
            fitted = agent.handling_network.state_dict()
            agent.handling_network.load_state_dict({
                key: (1.0 - fraction) * prior_state[key] + fraction * fitted[key]
                for key in fitted
            })
        fit["applied_update_fraction"] = fraction
        fit["applied_validation"] = centered_calibration(
            agent.handling_network, validation, device=device
        )
        require(
            state_digest(base.Q_local.state_dict()) == base_digest,
            "E204 mutated frozen E191",
        )
        end_digest = state_digest(agent.handling_network.state_dict())
        require(end_digest != start_digest, "E204 fit did not update the head")
        rounds.append({
            "round": round_index + 1,
            "episodes": rows,
            "training_samples": len(training),
            "validation_samples": len(validation),
            "validation_episode_indices": sorted(validation_indices),
            "network_before_sha256": start_digest,
            "network_after_sha256": end_digest,
            "fit": fit,
        })
        common._atomic_torch_save(
            _checkpoint(agent, contract, model_seed, base_digest, rounds, "latest"),
            latest,
        )
        _write(destination / "training-summary.json", {
            "status": "complete" if len(rounds) == p.ROUNDS else "paused",
            "training_protocol": p.PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "scenario": p.SCENARIO,
            "model_seed": model_seed,
            "completed_rounds": len(rounds),
            "completed_collection_episodes": len(rounds) * p.EPISODES_PER_ROUND,
            "base_selected_episode": source["selected_episode"],
            "base_checkpoint_sha256": source["checkpoint_sha256"],
            "base_policy_digest": base_digest,
            "lambda_center": p.LAMBDA_CENTER,
            "round_records": rounds,
            "development_only": True,
            "test_panel_opened": False,
            "paper_evidence": False,
        })
    if len(rounds) == p.ROUNDS and not terminal.is_file():
        common._atomic_torch_save(
            _checkpoint(agent, contract, model_seed, base_digest, rounds, "terminal"),
            terminal,
        )
    return {
        "status": "complete" if len(rounds) == p.ROUNDS else "paused",
        "model_seed": model_seed,
        "completed_rounds": len(rounds),
        "completed_collection_episodes": len(rounds) * p.EPISODES_PER_ROUND,
        "episode_runs_this_invocation": new_episode_runs,
        "terminal_checkpoint": str(terminal) if terminal.is_file() else None,
        "test_panel_opened": False,
        "paper_evidence": False,
    }


def pilot_collect(output: Path, model_seed: int, device_name: str) -> dict:
    contract = authenticate(output)
    device = resolve_device(device_name)
    base, search, liveness, base_digest, _ = load_base(
        contract, model_seed, device
    )
    destination = stage_b_dir(output, model_seed)
    require(
        not (destination / "latest.pth").exists(),
        "E204 pilot collection is only available before round 1 is fitted",
    )
    agent = new_agent(base, model_seed)
    coordinate = p.collection_coordinates(model_seed, 0)[0]
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    row, samples, executed = collect_episode(
        agent,
        env,
        contract=contract,
        destination=destination,
        model_seed=model_seed,
        round_index=0,
        offset=0,
        coordinate=coordinate,
        start_digest=state_digest(agent.handling_network.state_dict()),
        search=search,
        liveness=liveness,
    )
    require(
        state_digest(base.Q_local.state_dict()) == base_digest,
        "E204 pilot mutated frozen E191",
    )
    return {
        "status": "passed",
        "model_seed": model_seed,
        "episode_runs": int(executed),
        "cached_for_full_run": True,
        "row": row,
        "samples": len(samples),
        "training_runs": 0,
        "test_panel_opened": False,
    }


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    rows = []
    for model_seed in p.MODEL_SEEDS:
        directory = stage_b_dir(output, model_seed)
        summary_path = directory / "training-summary.json"
        if not summary_path.is_file():
            rows.append({"model_seed": model_seed, "status": "not_started"})
            continue
        summary = _read(summary_path)
        rounds = summary["round_records"]
        require(
            summary.get("training_protocol") == p.PROTOCOL
            and summary.get("contract_sha256") == contract["contract_sha256"]
            and int(summary.get("model_seed", -1)) == model_seed
            and int(summary.get("completed_rounds", -1)) == len(rounds),
            f"E204 seed {model_seed} summary identity differs",
        )
        episodes = [episode for round_row in rounds for episode in round_row["episodes"]]
        require(
            len(episodes) == len(rounds) * p.EPISODES_PER_ROUND
            and all(row["strict_safe_complete"] for row in episodes),
            f"E204 seed {model_seed} collection is incomplete",
        )
        terminal = directory / "terminal.pth"
        if len(rounds) == p.ROUNDS:
            require(
                summary["status"] == "complete" and terminal.is_file(),
                f"E204 seed {model_seed} terminal checkpoint is missing",
            )
        rows.append({
            "model_seed": model_seed,
            "status": summary["status"],
            "completed_rounds": len(rounds),
            "collection_episodes": len(episodes),
            "strict_complete_episodes": len(episodes),
            "mean_collection_J10": float(fmean(row["J10"] for row in episodes)),
            "mean_collection_rehandles": float(
                fmean(row["rehandles"] for row in episodes)
            ),
            "mean_rehandles_by_lambda": {
                f"{value:g}": float(fmean(
                    row["rehandles"] for row in episodes if row["lambda"] == value
                ))
                for value in p.LAMBDAS
            },
            "terminal_checkpoint": str(terminal) if terminal.is_file() else None,
            "terminal_checkpoint_sha256": (
                p.sha256(terminal) if terminal.is_file() else None
            ),
        })
    result = {
        "status": (
            "complete" if all(row["status"] == "complete" for row in rows)
            else "partial"
        ),
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "scope": "development Stage-B collection and fit; no quality panel",
        "lambda_center": p.LAMBDA_CENTER,
        "model_seeds": rows,
        "fresh_later_panel_required": True,
        "e203_used_for_selection": False,
        "episode_runs": 0,
        "training_runs": 0,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    _write(output.resolve() / "e204-report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "verify-base",
            "verify-center-parity",
            "pilot-collect",
            "train-seed",
            "analyze",
        ),
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    parser.add_argument("--through-round", type=int, default=p.ROUNDS)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "analyze":
        result = analyze(output)
    else:
        require(args.model_seed is not None, "E204 command requires --model-seed")
        if args.command == "verify-base":
            result = verify_base(output, args.model_seed, args.device)
        elif args.command == "verify-center-parity":
            result = verify_center_parity(output, args.model_seed, args.device)
        elif args.command == "pilot-collect":
            result = pilot_collect(output, args.model_seed, args.device)
        else:
            result = train_seed(
                output,
                args.model_seed,
                through_round=args.through_round,
                device_name=args.device,
            )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
