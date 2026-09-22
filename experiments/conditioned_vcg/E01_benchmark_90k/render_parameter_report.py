#!/usr/bin/env python3
"""Render the paper-facing E1 method and baseline parameter registry.

The report is reconstructed from the sealed E1 contract, the exact selected
checkpoints, and representative authenticated run ledgers.  It deliberately
does not invoke the historical authentication chain: this is a read-only
description of artifacts that already exist, not a new evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_E1_ROOT = PROJECT_ROOT / "results/vcg-conditioned-final-comparison-90k-cpu-v3"

BASE_PATHS = {
    0: PROJECT_ROOT / "results/vcg-dense-v1-1-seed0-500ep/best.pth",
    1: PROJECT_ROOT / "results/vcg-dense-v1-1-seed1-500ep/best.pth",
    2: PROJECT_ROOT / "results/vcg-dense-v1-1-seed2-500ep/best.pth",
}
CONDITIONED_PATHS = {
    0: PROJECT_ROOT / "results/vcg-v1-1-conditioned-handling-seed0-damped-convergence/terminal.pth",
    1: PROJECT_ROOT / "results/vcg-v1-1-conditioned-handling-seed1-convergence-continuation/terminal.pth",
    2: PROJECT_ROOT / "results/vcg-v1-1-conditioned-handling-seeds12-two-phase-development/seed-2/terminal.pth",
}
V23_PATHS = {
    seed: PROJECT_ROOT
    / f"results/vcg-v2-3-seed-stability-85k/seed-{seed}/best-development-candidate.pth"
    for seed in (11, 12, 13)
}
A3C_PATHS = {
    seed: PROJECT_ROOT
    / f"results/vcg-v2-3-kim2020-supplement-85k/training/seed-{seed}/best.pth"
    for seed in (0, 1, 2)
}
A3C_COMPLETION_PATHS = {
    seed: PROJECT_ROOT
    / f"results/vcg-v2-3-kim2020-supplement-evaluation-85k/training-completion/seed-{seed}.json"
    for seed in (0, 1, 2)
}

EXPERIMENT_PATHS = {
    "e03": PROJECT_ROOT / "results/vcg-conditioned-e03-certification-ablation-90k-v2/e03-contract.json",
    "e04": PROJECT_ROOT / "results/vcg-conditioned-e04-ranking-ablation-92k/e04-contract.json",
    "e05": PROJECT_ROOT / "results/vcg-conditioned-e05-handling-ablation-92k/e05-contract.json",
    "e05a": PROJECT_ROOT / "results/vcg-conditioned-e05-handling-ablation-92k/e05a-model-validation.json",
    "e05c": PROJECT_ROOT / "results/vcg-conditioned-e05c-future-consequence-ablation-92k/e05c-contract.json",
    "e08": PROJECT_ROOT / "results/vcg-conditioned-e08-liveness-audit-95k/e8-pilot-contract.json",
    "e11": PROJECT_ROOT / "results/vcg-conditioned-e11-distribution-shift-93k/e11-contract.json",
    "e12_design": PROJECT_ROOT / "results/vcg-conditioned-e12-representation-ablation-94k/e12-contract.json",
    "e12": PROJECT_ROOT / "results/vcg-conditioned-e12-corrected-executor-full-94k/corrected-full-contract.json",
    "e13": PROJECT_ROOT / "results/vcg-conditioned-e13-scalability-95k-v2/e13-contract.json",
    "e14": PROJECT_ROOT / "results/vcg-conditioned-e14-budget-scalability-95k/e14-budget-contract.json",
    "e14_refinement": PROJECT_ROOT / "results/vcg-conditioned-e14-threshold-refinement-95k/threshold-contract.json",
    "e16": PROJECT_ROOT / "results/vcg-conditioned-e16-component-rollout-ablation-92k/e16d-contract.json",
    "e19": PROJECT_ROOT / "results/vcg-conditioned-e19-service-cost/e19-contract.json",
}


class ParameterReportError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ParameterReportError(f"expected JSON object: {path}")
    return value


def _checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ParameterReportError(f"expected checkpoint mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Mapping[str, Any], *, omit: str | None = None) -> str:
    payload = dict(value)
    if omit is not None:
        payload.pop(omit, None)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _same(label: str, values: list[Any]) -> Any:
    if not values:
        raise ParameterReportError(f"no values supplied for {label}")
    canonical = [json.dumps(v, sort_keys=True, default=str) for v in values]
    if len(set(canonical)) != 1:
        raise ParameterReportError(f"{label} differs across selected checkpoints")
    return values[0]


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def _fmt(value: Any) -> str:
    if value is None:
        return "unbounded"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        if value and all(
            isinstance(item, (list, tuple)) and len(item) == 2 for item in value
        ):
            return ", ".join(f"({item[0]}, {item[1]})" for item in value)
        return ", ".join(_fmt(item) for item in value)
    return str(value)


def build_registry(e1_root: Path) -> dict[str, Any]:
    e1_root = e1_root.resolve()
    contract_path = e1_root / "final90-contract.json"
    instance_path = e1_root / "episode-instances/seed-90000.json"
    contract = _json(contract_path)
    instance = _json(instance_path)

    bases = {seed: _checkpoint(path) for seed, path in BASE_PATHS.items()}
    conditioned = {
        seed: _checkpoint(path) for seed, path in CONDITIONED_PATHS.items()
    }
    v23 = {seed: _checkpoint(path) for seed, path in V23_PATHS.items()}
    a3c = {seed: _checkpoint(path) for seed, path in A3C_PATHS.items()}
    a3c_completion = {
        seed: _json(path) for seed, path in A3C_COMPLETION_PATHS.items()
    }

    base_config = _same("VCG operational config", [bases[s]["config"] for s in bases])
    base_environment = _same(
        "VCG operational environment", [bases[s]["environment"] for s in bases]
    )
    base_objective = _same(
        "VCG operational objective", [bases[s]["timing_objective_spec"] for s in bases]
    )
    base_search = _same(
        "VCG operational search", [bases[s]["viability_search"] for s in bases]
    )
    handling_config = _same(
        "conditioned handling config",
        [conditioned[s]["agent_checkpoint"]["config"] for s in conditioned],
    )
    a3c_config = _same("A3C config", [a3c[s]["config"] for s in a3c])
    a3c_geometry = _same("A3C geometry", [a3c[s]["geometry"] for s in a3c])
    v23_config = _same(
        "historical VCG 2.3 config",
        [v23[s]["core_contract"]["agent_config"] for s in v23],
    )
    v23_search = _same(
        "historical VCG 2.3 search",
        [v23[s]["core_contract"]["viability_search_config"] for s in v23],
    )

    dynamic_ledger_path = (
        e1_root
        / "run-ledger/duration_aware_dynamic_pslap/instance-90000-model-none-rng-none.json"
    )
    ga_ledger_path = (
        e1_root
        / "run-ledger/duration_aware_pslap_ga_2009_rolling_capacity_aware_partial"
        / "instance-90000-model-none-rng-0.json"
    )
    dynamic_ledger = _json(dynamic_ledger_path)
    ga_ledger = _json(ga_ledger_path)
    dynamic_audit = dynamic_ledger["run"]["evidence"]["method_audit"]
    ga_audit = ga_ledger["run"]["evidence"]["capacity_aware_source_audit"]
    ga_config = ga_audit["optimizer_base_config"]
    experiments = {name: _json(path) for name, path in EXPERIMENT_PATHS.items()}

    expected_terminal_hashes = contract["conditioned_terminal_sha256"]
    for seed, path in CONDITIONED_PATHS.items():
        if _sha256(path) != expected_terminal_hashes[str(seed)]:
            raise ParameterReportError(f"conditioned terminal hash mismatch for seed {seed}")
    for seed, path in V23_PATHS.items():
        if _sha256(path) != contract["historical_v23_checkpoints"][str(seed)]:
            raise ParameterReportError(f"VCG 2.3 checkpoint hash mismatch for seed {seed}")
    for seed, path in A3C_PATHS.items():
        if _sha256(path) != contract["kim_checkpoints"][str(seed)]:
            raise ParameterReportError(f"A3C checkpoint hash mismatch for seed {seed}")

    for seed in (0, 1, 2):
        base_sha = conditioned[seed]["agent_checkpoint"]["base_checkpoint_sha256"]
        if base_sha != _sha256(BASE_PATHS[seed]):
            raise ParameterReportError(f"conditioned/base binding mismatch for seed {seed}")

    shared = {
        "geometry": {
            "grid": [contract["environment"]["grid_rows"], contract["environment"]["grid_cols"]],
            "start_cell": instance["start_state"],
            "door_cell": instance["door_cell"],
            "pickup_cell": instance["pickup_cell"],
            "waiting_cell": instance["waiting_cell"],
            "exit_cells": instance["exit_cells"],
            "usable_storage_cells": instance["storage_positions"],
            "usable_storage_cell_count": len(instance["storage_positions"]),
        },
        "workload": {
            "blocks_per_episode": contract["environment"]["number_blocks"],
            "arrival_distribution": "exponential inter-arrivals",
            "arrival_rate": contract["environment"]["arrival_rate"],
            "mean_interarrival_steps": 1.0 / contract["environment"]["arrival_rate"],
            "first_arrival_forced_to_step_zero": True,
            "storage_duration_distribution": "Poisson",
            "storage_duration_mean_steps": contract["environment"]["proc_mean"],
            "episode_horizon_primitive_steps": contract["environment"]["max_steps"],
        },
        "environment_reward": {
            "primitive_step_cost": -0.09,
            "first_storage_reward": 5.0,
            "delivery_reward": "40 - 1.5*|e| - 0.5*max(0, |e|-20)",
            "delivery_base": base_objective["dense_b"],
            "absolute_error_coefficient": base_objective["lambda_abs"],
            "outside_window_coefficient": base_objective["lambda_outside"],
            "target_window_steps": base_objective["window"],
        },
        "evaluation": {
            "episode_instance_seeds": contract["instance_seeds"],
            "episode_instance_count": len(contract["instance_seeds"]),
            "same_serialized_instances_for_all_methods": contract[
                "all_rows_use_same_serialized_episode_instances"
            ],
            "aggregation_unit": contract["aggregation_unit"],
            "complete_case_filtering_allowed": contract[
                "complete_case_filtering_allowed"
            ],
            "failed_method_or_coordinate_suppressed": contract[
                "failed_method_or_lambda_has_aggregate_metrics_suppressed"
            ],
            "primary_metrics": contract["primary_tradeoff_plane"],
            "secondary_metrics": contract["secondary_metrics"],
            "within_window_definition": "absolute delivery error <= 20 primitive steps",
            "rehandle_rate_definition": "100 * physical storage-to-storage relocations / 8 required deliveries",
            "execution_device": contract["runtime_contract"]["execution_device"],
            "torch_threads": contract["runtime_contract"]["torch_num_threads"],
            "software_versions": contract["runtime_contract"]["software_versions"],
            "training_during_evaluation": contract["training_or_learning"],
            "checkpoint_selection_on_test": contract["checkpoint_selection"],
            "lambda_selection_on_test": contract["lambda_selection"],
        },
    }

    methods = {
        "conditioned_vcg": {
            "paper_name": "VCG",
            "role": "proposed preference-conditioned controller",
            "model_seeds": contract["model_seeds"],
            "operational_critic": {
                "architecture": bases[0]["network_architecture"],
                "graph_node_features": bases[0]["graph_node_feature_dim"],
                "action_features": bases[0]["action_feature_dim"],
                "graph_hidden_dim": base_config["graph_hidden_dim"],
                "graph_embedding_dim": base_config["graph_embedding_dim"],
                "message_passing_steps": base_config["message_passing_steps"],
                "action_embedding_dim": base_config["action_embedding_dim"],
                "head_hidden_dim": base_config["head_hidden_dim"],
                "gamma": base_config["gamma"],
                "reward_scale": base_config["reward_scale"],
                "optimizer": "Adam",
                "learning_rate": base_config["learning_rate"],
                "batch_size": base_config["batch_size"],
                "replay_capacity": base_config["replay_capacity"],
                "update_every_macro": base_config["update_every"],
                "target_update_every_updates": base_config["target_update_every"],
                "huber_delta": base_config["huber_delta"],
                "gradient_clip": base_config["grad_clip"],
                "training_episodes_per_seed": 500,
                "training_seed_bases": {str(s): bases[s]["train_instance_seeds"][0] for s in bases},
                "epsilon_start": 0.90,
                "epsilon_end": 0.05,
                "epsilon_warmup_decisions": 0,
                "epsilon_decay_decisions": 3000,
                "validation_every_episodes": 25,
                "validation_instance_seeds": bases[0]["validation_instance_seeds"],
                "selected_checkpoint_episode": {
                    str(s): bases[s]["selected_checkpoint_episode"] for s in bases
                },
            },
            "certification_and_selection": {
                "candidate_contract": bases[0]["candidate_interface"],
                "certificate_contract": bases[0]["certification_contract"],
                "unknown_and_unsafe_fail_closed": bases[0]["unsafe_unknown_fail_closed"],
                "search": base_search,
                "modes": bases[0]["modes"],
                "action_types": bases[0]["action_types"],
                "within_mode_temperatures": bases[0]["within_temperatures"],
                "mode_temperature": bases[0]["mode_temperature"],
                "deployment_policy": bases[0]["deployment_policy"],
                "liveness_rule": bases[0]["liveness_rule"],
                "max_nonprogress_recovery_decisions": base_config[
                    "max_nonprogress_recovery_decisions"
                ],
                "force_recovery_witness_when_due": base_config[
                    "force_recovery_witness_when_due"
                ],
            },
            "handling_predictor": {
                "architecture": conditioned[0]["agent_checkpoint"]["network_architecture"],
                "feature_dim": handling_config["feature_dim"],
                "hidden_layers": [handling_config["hidden_dim"], handling_config["hidden_dim"]],
                "activation": "SiLU",
                "output": "softplus nonnegative future rehandles",
                "lambda_input": "lambda/lambda_max",
                "lambda_max": handling_config["lambda_max"],
                "target": conditioned[0]["agent_checkpoint"]["target_contract"],
                "exact_immediate_term": "1 for Reconfigure, 0 otherwise",
                "merit": "Q_op(s,c) - lambda * (1[Reconfigure] + Q_N,future(s,c,lambda))",
                "lambda_zero_behavior": "exact direct delegation to frozen VCG 1.1",
                "operational_critic_frozen": True,
                "collection_lambda": "stratified uniform on [0,0.2], fixed for the whole episode",
                "collection_episodes_by_seed": {"0": 400, "1": 500, "2": 400},
                "episodes_per_round": 50,
                "rounds_by_seed": contract["conditioned_terminal_training_rounds"],
                "fit_batch_size": 256,
                "fit_epochs_per_round": 8,
                "fit_learning_rate": 0.0005,
                "full_update_rounds": 4,
                "polyak_rho_after_round_4": 0.25,
                "whole_episode_monte_carlo": True,
                "checkpoint_selection": False,
            },
            "deployment": {
                "lambda_grid": contract["lambda_grid"],
                "rows": contract["conditioned_rows"],
                "replication": contract["nuisance_averaging"]["vcg_conditioned"],
                "checkpoint_sha256": contract["conditioned_terminal_sha256"],
            },
        },
        "historical_vcg_2_3": {
            "paper_name": "Historical VCG 2.3",
            "role": "historical constrained-VCG comparator; not a lambda coordinate of the final method",
            "model_seeds": [11, 12, 13],
            "architecture": v23[11]["agent_state"]["network_architecture"],
            "graph_hidden_dim": v23_config["graph_hidden_dim"],
            "graph_embedding_dim": v23_config["graph_embedding_dim"],
            "message_passing_steps": v23_config["message_passing_steps"],
            "action_embedding_dim": v23_config["action_embedding_dim"],
            "head_hidden_dim": v23_config["head_hidden_dim"],
            "operational_gamma": v23_config["gamma_operational"],
            "rehandle_gamma": v23_config["gamma_rehandle"],
            "reward_scale": v23_config["operational_reward_scale"],
            "learning_rate": v23_config["learning_rate"],
            "batch_size": v23_config["batch_size"],
            "replay_capacity": v23_config["replay_capacity"],
            "target_update_every": v23_config["target_update_every"],
            "training_episodes_per_seed": 200,
            "dual": {
                "budget_rehandles_per_100_required_deliveries": 20.0,
                "learning_rate": 0.01,
                "lambda_initial": 0.0,
                "lambda_max": 20.0,
                "warmup_episodes": 20,
                "block_episodes": 10,
            },
            "selected_episode": {str(s): v23[s]["completed_episodes"] for s in v23},
            "selected_lambda": {str(s): v23[s]["runtime_lambda"] for s in v23},
            "deployment_policy": "induced nested stochastic Lagrangian policy",
            "selected_within_group_temperatures": [0.01, 0.01, 0.01, 0.01],
            "selected_group_temperature": 0.05,
            "certification_search": v23_search,
            "policy_rng_realizations_per_model_instance": 4,
            "policy_rng_base": contract["baseline_policy_rng"]["v23_base"],
            "rows": contract["baseline_rows"]["vcg_2_3"],
            "replication": contract["nuisance_averaging"]["vcg_2_3"],
            "checkpoint_sha256": contract["historical_v23_checkpoints"],
        },
        "dynamic_pslap": {
            "paper_name": "Duration-aware Dynamic PSLAP",
            "role": "deterministic online heuristic baseline",
            "learned_parameters": False,
            "assignment_rule": "among physically feasible cells preserving access to earlier-due blocks, minimize exit distance, then travel distance, then cell index",
            "relocation_rule": "minimize travel distance, then exit distance, then cell index",
            "retrieval_dispatch": "dispatch when complete predicted retrieval plan has slack <= 0; rank by slack, relocations, plan steps, label",
            "assignment_source_version": dynamic_audit["assignment_source_version"],
            "controller_architecture": dynamic_audit["controller_architecture"],
            "exact_viability_certificate": dynamic_audit["exact_viability_certificate"],
            "deterministic_runs_per_instance": 1,
            "rows": contract["baseline_rows"]["duration_aware_dynamic_pslap"],
            "replication": contract["nuisance_averaging"]["duration_aware_dynamic_pslap"],
        },
        "capacity_aware_ga": {
            "paper_name": "Capacity-aware rolling GA",
            "role": "Park--Seo-inspired arrived-information online adaptation",
            "learned_parameters": False,
            "information_regime": "only stored inventory and blocks whose arrival time has been reached",
            "objective": "lexicographic infeasible known placements, predicted obstruction, route length",
            "population_size": ga_config["population_size"],
            "generations_per_decision": ga_config["generations"],
            "elite_count": ga_config["elite_count"],
            "tournament_size": ga_config["tournament_size"],
            "crossover": "one-point with uniqueness repair",
            "crossover_probability": ga_config["crossover_rate"],
            "mutation_probability": ga_config["mutation_rate"],
            "assignment_source_version": ga_audit["assignment_source_version"],
            "scheduler_defer_action_introduced": ga_audit[
                "scheduler_defer_action_introduced"
            ],
            "optimizer_rng_realizations_per_instance": 4,
            "optimizer_rng_base": contract["baseline_policy_rng"]["ga_base"],
            "rows": contract["baseline_rows"][
                "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial"
            ],
            "replication": contract["nuisance_averaging"][
                "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial"
            ],
        },
        "a3c_adaptation": {
            "paper_name": "A3C adaptation",
            "role": "Kim et al. (2020)-inspired spatial assignment adaptation, not an exact reproduction",
            "source_paper_doi": a3c[0]["source_paper_doi"],
            "exact_paper_reproduction": a3c[0]["exact_paper_reproduction"],
            "model_seeds": [0, 1, 2],
            "input_channels": a3c[0]["input_channels"],
            "channel_semantics": [
                "relative remaining dwell",
                "occupancy",
                "storage mask",
                "traversability",
                "exit proximity",
            ],
            "architecture": a3c[0]["selector_architecture"],
            "hidden_channels": a3c_config["hidden_channels"],
            "learning_rate": a3c_config["learning_rate"],
            "weight_decay": a3c_config["weight_decay"],
            "gamma_per_placement_decision": a3c_config["gamma"],
            "reward_scale": a3c_config["reward_scale"],
            "entropy_coefficient": a3c_config["entropy_coef"],
            "value_coefficient": a3c_config["value_coef"],
            "gradient_clip": a3c_config["grad_clip"],
            "updates_per_eligible_episode": a3c_config["updates_per_episode"],
            "training_reward": "negative relocations attributed to the placed block",
            "training_episodes_per_seed": 1000,
            "validation_every_episodes": 100,
            "validation_instance_seeds": a3c[0]["validation_instance_seeds"],
            "validation_stochastic_rollouts_per_instance": a3c[0]["stochastic_rollouts"],
            "selected_episode": {
                str(s): a3c_completion[s]["selected_episode"] for s in a3c_completion
            },
            "deployment_mode": "stochastic masked categorical",
            "policy_rng_realizations_per_model_instance": 5,
            "policy_rng_base": contract["baseline_policy_rng"]["kim_base"],
            "shared_physical_control": "deterministic navigation and atomic storage/retrieval executors",
            "exact_recovery_stall_fallback_max_nodes": a3c[0]["exact_recovery_max_nodes"],
            "geometry_storage_cell_count": a3c_geometry["storage_cell_count"],
            "rows": contract["baseline_rows"][
                "kim2020_a3c_spatial_adapted__stochastic"
            ],
            "replication": contract["nuisance_averaging"][
                "kim2020_a3c_spatial_adapted__stochastic"
            ],
            "checkpoint_sha256": contract["kim_checkpoints"],
        },
    }

    e11 = experiments["e11"]
    e13 = experiments["e13"]
    experiment_coordinates = {
        "E03_certification_ablation": {
            "candidate_sources": experiments["e03"]["candidate_sources"],
            "preference_lambda": experiments["e03"]["preference_lambda"],
            "model_seeds": experiments["e03"]["model_seeds"],
            "instance_seeds": experiments["e03"]["instance_seeds"],
            "new_physical_only_rows": experiments["e03"]["new_physical_rows"],
            "certified_rows_reused_from_E1": experiments["e03"][
                "certified_control_rows_reused_from_e1"
            ],
        },
        "E04_ranking_ablation": {
            "ranking_signals": experiments["e04"]["ranking_signals"],
            "conditioned_lambdas": experiments["e04"]["conditioned_lambdas"],
            "model_seeds": experiments["e04"]["model_seeds"],
            "instance_seeds": experiments["e04"]["instance_seeds"],
            "random_ranking_seeds": experiments["e04"]["random_ranking_seeds"],
        },
        "E05_handling_model": {
            "validation_fixed_states": experiments["e05a"]["seeds"]["0"][
                "fixed_state_count"
            ],
            "validation_fixed_candidates": experiments["e05a"]["seeds"]["0"][
                "fixed_candidate_count"
            ],
            "validation_lambda_count": experiments["e05a"][
                "fixed_bank_lambda_count"
            ],
            "conditioning_deployment_lambdas": experiments["e05"][
                "deployment_lambdas"
            ],
            "clamped_predictor_input_lambda": experiments["e05"][
                "clamped_handling_input_lambda"
            ],
            "future_consequence_ablation": experiments["e05c"]["ablation"],
            "future_consequence_control": experiments["e05c"]["control"],
        },
        "E08_liveness_audit": {
            "scope": "instrumented descriptive pilot",
            "scenario": experiments["e08"]["scenario_id"],
            "instance_seed": experiments["e08"]["instance_seed"],
            "model_seed": experiments["e08"]["model_seed"],
            "preference_lambda": experiments["e08"]["preference_lambda"],
            "search": experiments["e08"]["search"],
        },
        "E11_distribution_shift": {
            "methods": e11["methods"],
            "deployment_lambdas": e11["deployment_lambdas"],
            "model_seeds": e11["model_seeds"],
            "instance_seeds": e11["instance_seeds"],
            "regimes": [
                {
                    "id": regime["regime_id"],
                    "axis": regime["shift_axis"],
                    "grid": [regime["grid_rows"], regime["grid_cols"]],
                    "mean_interarrival": regime["mean_interarrival"],
                    "duration_profile": regime["duration_profile"],
                    "duration_mean": regime["proc_mean"],
                    "mirrored_entry": regime["mirrored_entry"],
                    "blocks": regime["number_blocks"],
                }
                for regime in e11["regimes"]
            ],
            "expected_rows": e11["expected_full_rows"],
        },
        "E12_representation_ablation": {
            "representations": experiments["e12"]["representations"],
            "regimes": experiments["e12"]["regimes"],
            "deployment_lambdas": experiments["e12"]["deployment_lambdas"],
            "model_seeds": experiments["e12"]["model_seeds"],
            "instance_seeds": experiments["e12"]["instance_seeds"],
            "evaluation_rows": experiments["e12"]["expected_evaluation_rows"],
            "training_runs": 18,
            "operational_training": experiments["e12_design"][
                "operational_training"
            ],
            "handling_training": experiments["e12_design"]["handling_training"],
            "new_training_runs_in_corrected_executor_confirmation": experiments[
                "e12"
            ]["training_runs"],
        },
        "E13_operational_scalability": {
            "model_seed": e13["model_seed"],
            "instance_seeds": e13["instance_seeds"],
            "preference_lambda": e13["preference_lambda"],
            "search": e13["search_config"],
            "relocation_family_certificate": e13[
                "recovery_certification_strategy"
            ],
            "deadline_seconds": e13["deadline_seconds"],
            "wall_limit_seconds_per_episode": e13[
                "wall_limit_seconds_per_episode"
            ],
            "scenarios": [
                {
                    "id": scenario["scenario_id"],
                    "comparison": scenario["comparison"],
                    "grid": [scenario["rows"], scenario["cols"]],
                    "storage_capacity": scenario["storage_capacity"],
                    "initial_occupied_slots": scenario[
                        "initial_occupied_slots"
                    ],
                    "initial_occupancy_ratio": scenario[
                        "initial_occupancy_ratio"
                    ],
                    "total_jobs": scenario["total_jobs"],
                    "max_steps": scenario["max_steps"],
                }
                for scenario in e13["scenarios"]
            ],
        },
        "E14_certification_budget": {
            "predeclared_max_node_budgets": experiments["e14"][
                "budgets_max_nodes"
            ],
            "newly_executed_budgets": experiments["e14"][
                "executed_budgets_max_nodes"
            ],
            "reference_budget_reused_from_E13": experiments["e14"][
                "reference_budget_max_nodes"
            ],
            "post_hoc_refinement_budgets": experiments["e14_refinement"][
                "refinement_budgets_max_nodes"
            ],
            "instance_seed": experiments["e14"]["instance_seed"],
            "model_seed": experiments["e14"]["model_seed"],
            "preference_lambda": experiments["e14"]["preference_lambda"],
            "scenarios": experiments["e14"]["scenarios"],
            "deadline_seconds": experiments["e14"]["deadline_seconds"],
            "wall_limit_seconds_per_episode": experiments["e14"][
                "wall_limit_seconds_per_episode"
            ],
        },
        "E16_prediction_to_ranking": {
            "arms": experiments["e16"]["arms"],
            "decomposition": experiments["e16"]["decomposition"],
            "deployment_lambda": experiments["e16"]["deployment_lambda"],
            "predictor_input_lambda": experiments["e16"][
                "predictor_input_lambda"
            ],
            "model_seeds": experiments["e16"]["model_seeds"],
            "instance_seeds": experiments["e16"]["instance_seeds"],
            "rows": experiments["e16"]["confirmation_rows"],
        },
        "E19_service_cost": {
            "analysis_only": experiments["e19"]["analysis_only"],
            "cost_definition": experiments["e19"]["cost_definition"],
            "tardiness_weights": experiments["e19"]["tardiness_weights"],
            "rehandle_equivalent_steps": experiments["e19"][
                "rehandle_equivalent_steps"
            ],
            "tail_probabilities": experiments["e19"]["tail_probabilities"],
            "new_training_runs": experiments["e19"]["new_training_runs"],
            "new_evaluation_rollouts": experiments["e19"][
                "new_evaluation_rollouts"
            ],
        },
    }

    provenance_paths = {
        "sealed_e1_contract": contract_path,
        "representative_episode_instance": instance_path,
        "representative_dynamic_ledger": dynamic_ledger_path,
        "representative_ga_ledger": ga_ledger_path,
        "conditioned_controller_source": PROJECT_ROOT / "vcg_v11_conditioned_handling.py",
        "operational_training_source": PROJECT_ROOT / "train_vcg_dense_proper.py",
        "dynamic_pslap_source": PROJECT_ROOT / "PSLAP/dynamic_yard.py",
        "ga_source": PROJECT_ROOT / "PSLAP/ga_optimizer.py",
        "a3c_source": PROJECT_ROOT / "PSLAP/kim2020_a3c_spatial.py",
        **{f"base_checkpoint_seed_{s}": p for s, p in BASE_PATHS.items()},
        **{f"conditioned_checkpoint_seed_{s}": p for s, p in CONDITIONED_PATHS.items()},
        **{f"v23_checkpoint_seed_{s}": p for s, p in V23_PATHS.items()},
        **{f"a3c_checkpoint_seed_{s}": p for s, p in A3C_PATHS.items()},
        **{f"{name}_contract_or_report": p for name, p in EXPERIMENT_PATHS.items()},
    }

    registry: dict[str, Any] = {
        "schema_version": 1,
        "title": "Paper parameter registry: final VCG, baselines, and experiment coordinates",
        "scope": {
            "included": "parameters governing the final VCG and E1 baselines, plus the prespecified coordinates of the paper's principal confirmation experiments",
            "excluded": "development-only screens, plot styling, machine-dependent incidental timings, and intermediate superseded contracts",
            "baseline_naming": "Kim2020 implementation is reported as A3C adaptation",
        },
        "shared": shared,
        "methods": methods,
        "experiment_coordinates": experiment_coordinates,
        "raw_run_accounting": {
            "conditioned_vcg": contract["conditioned_rows"],
            "historical_vcg_2_3": contract["baseline_rows"]["vcg_2_3"],
            "dynamic_pslap": contract["baseline_rows"]["duration_aware_dynamic_pslap"],
            "capacity_aware_ga": contract["baseline_rows"][
                "duration_aware_pslap_ga_2009_rolling_capacity_aware_partial"
            ],
            "a3c_adaptation": contract["baseline_rows"][
                "kim2020_a3c_spatial_adapted__stochastic"
            ],
            "interpretation": "all methods see the same 30 EpisodeInstances; extra rows average model, policy, or optimizer nuisance realizations within instance",
        },
        "provenance": {
            name: {"path": _rel(path), "sha256": _sha256(path)}
            for name, path in sorted(provenance_paths.items())
        },
    }
    registry["registry_sha256"] = _canonical_sha256(
        registry, omit="registry_sha256"
    )
    return registry


def _table(rows: list[tuple[str, str]]) -> str:
    return "\n".join(
        ["| Parameter | Value |", "|---|---|"]
        + [f"| {name} | {value} |" for name, value in rows]
    )


def render_markdown(registry: Mapping[str, Any]) -> str:
    shared = registry["shared"]
    methods = registry["methods"]
    vcg = methods["conditioned_vcg"]
    v23 = methods["historical_vcg_2_3"]
    dynamic = methods["dynamic_pslap"]
    ga = methods["capacity_aware_ga"]
    a3c = methods["a3c_adaptation"]
    coordinates = registry["experiment_coordinates"]

    lines = [
        "# Paper parameter report",
        "",
        "This report records the parameters that governed the final VCG controller and every comparator in the common 90k E1 benchmark, followed by the coordinates of the paper's principal confirmation experiments. Values were reconstructed from sealed contracts, exact selected checkpoints, and authenticated run ledgers; no policy was rerun. Development-only screens and superseded contracts are outside this report.",
        "",
        "## Shared problem and evaluation settings",
        "",
        _table([
            ("Grid", f"{shared['geometry']['grid'][0]} x {shared['geometry']['grid'][1]}"),
            ("Usable storage cells", str(shared['geometry']['usable_storage_cell_count'])),
            ("Start / pickup / waiting", f"{shared['geometry']['start_cell']} / {shared['geometry']['pickup_cell']} / {shared['geometry']['waiting_cell']}"),
            ("Exit cells", _fmt(shared['geometry']['exit_cells'])),
            ("Blocks / required deliveries", str(shared['workload']['blocks_per_episode'])),
            ("Arrivals", f"exponential, rate={_fmt(shared['workload']['arrival_rate'])}, mean inter-arrival={_fmt(shared['workload']['mean_interarrival_steps'])} step; first arrival at step 0"),
            ("Storage durations", f"Poisson, mean={_fmt(shared['workload']['storage_duration_mean_steps'])} steps"),
            ("Episode cap", f"{shared['workload']['episode_horizon_primitive_steps']} primitive steps"),
            ("Evaluation panel", "30 frozen EpisodeInstances, seeds 90000--90029"),
            ("Target window", "+/-20 primitive steps"),
            ("Execution", f"CPU, one PyTorch thread; Python {shared['evaluation']['software_versions']['python']}, NumPy {shared['evaluation']['software_versions']['numpy']}, PyTorch {shared['evaluation']['software_versions']['torch']}"),
        ]),
        "",
        "All methods used the same serialized EpisodeInstances. The statistical unit was the EpisodeInstance; model, policy, and optimizer repetitions were averaged within instance. Evaluation performed no learning, checkpoint selection, or preference selection, and complete-case filtering was forbidden.",
        "",
        "### Reward and reported objectives",
        "",
        "For delivery error `e`, the dense delivery reward was `40 - 1.5|e| - 0.5 max(0, |e|-20)`. Every primitive step contributed `-0.09`, and the first successful storage of a block contributed `+5`. Thus episode return was not used as a substitute for either MAE or physical rehandles; those metrics were reported separately.",
        "",
        _table([
            ("Primary coordinates", "MAE; physical rehandles per 100 required deliveries"),
            ("Secondary metrics", "dense return; primitive steps; within-window rate"),
            ("Rehandle rate", "100 x storage-to-storage relocations / 8 required deliveries"),
            ("Failure handling", "retain failures; suppress a method/coordinate aggregate rather than complete-case average"),
        ]),
        "",
        "## Final VCG",
        "",
        "The deployed score was `Q_op(s,c) - lambda [1{Reconfigure} + Q_N,future(s,c,lambda)]`. At `lambda=0`, the implementation bypassed the handling head and delegated exactly to the frozen operational controller.",
        "",
        "### Operational critic and selector",
        "",
        _table([
            ("Representation", f"shared counterfactual graph; {vcg['operational_critic']['graph_node_features']} node and {vcg['operational_critic']['action_features']} action features"),
            ("Graph / embedding / action / head widths", f"{vcg['operational_critic']['graph_hidden_dim']} / {vcg['operational_critic']['graph_embedding_dim']} / {vcg['operational_critic']['action_embedding_dim']} / {vcg['operational_critic']['head_hidden_dim']}"),
            ("Message-passing steps", str(vcg['operational_critic']['message_passing_steps'])),
            ("Discount / reward scale", f"{_fmt(vcg['operational_critic']['gamma'])} / {_fmt(vcg['operational_critic']['reward_scale'])}"),
            ("Optimizer", f"Adam, lr={_fmt(vcg['operational_critic']['learning_rate'])}, batch={vcg['operational_critic']['batch_size']}, replay={vcg['operational_critic']['replay_capacity']}"),
            ("Updates", f"every macro; target every {vcg['operational_critic']['target_update_every_updates']} updates; Huber delta={_fmt(vcg['operational_critic']['huber_delta'])}; gradient clip={_fmt(vcg['operational_critic']['gradient_clip'])}"),
            ("Training", "500 episodes per model seed; epsilon 0.90 to 0.05 over 3,000 macro decisions"),
            ("Validation / selection", "every 25 episodes on seeds 79000--79019; selected episodes seed 0/1/2 = 500/450/375"),
            ("Modes", "Accept, Recover, Defer; Recover contains Deliver and Reconfigure"),
            ("Temperatures", "within-mode 0.1; mode 1.0; deterministic regularized mode argmax then candidate argmax at deployment"),
        ]),
        "",
        "### Exact certificate and liveness",
        "",
        _table([
            ("Search", f"goal-directed; max nodes={vcg['certification_and_selection']['search']['max_nodes']}; depth and primitive-step bounds unbounded"),
            ("Queue cells", "reserved in recovery search"),
            ("UNKNOWN / UNSAFE", "excluded (fail closed)"),
            ("Defer limits", "10 option steps and 16 consecutive defer decisions"),
            ("Nonprogress recovery limit", str(vcg['certification_and_selection']['max_nonprogress_recovery_decisions'])),
        ]),
        "",
        "### Conditioned future-handling head",
        "",
        _table([
            ("Input / hidden layers", "224 frozen candidate features plus normalized lambda; two 128-unit SiLU layers"),
            ("Output", "softplus nonnegative future physical rehandles"),
            ("Target", "undiscounted post-current-macro Monte Carlo rehandles from complete episodes"),
            ("Preference range", "[0, 0.2] during collection; fixed within each episode"),
            ("Collection episodes", "seed 0/1/2 = 400/500/400 in 50-episode rounds"),
            ("Fit", "8 epochs/round, batch 256, lr=5e-4; first 4 full-update rounds, then Polyak rho=0.25"),
            ("Operational critic", "frozen throughout handling-head training"),
            ("E1 lambda grid", _fmt(vcg['deployment']['lambda_grid'])),
        ]),
        "",
        "## Baseline parameters",
        "",
        "### Historical VCG 2.3",
        "",
        _table([
            ("Architecture", "shared graph trunk with operational and nonnegative rehandle heads; widths 64/64/32/128; 3 message-passing steps"),
            ("Discounts", "operational=1.0; rehandle=1.0; operational reward scale=0.01"),
            ("Optimizer", f"lr={_fmt(v23['learning_rate'])}, batch={v23['batch_size']}, replay={v23['replay_capacity']}, target every {v23['target_update_every']} updates"),
            ("Training", "200 episodes per seed; 20-episode critic warm-up, then 10-episode dual blocks"),
            ("Dual", "budget=20 rehandles/100 deliveries; lr=0.01; lambda initial/max=0/20"),
            ("Selected seed operating points", "seed 11: ep120, lambda=.152; seed 12: ep160, lambda=.170; seed 13: ep100, lambda=.213"),
            ("Deployment", "stochastic induced nested Lagrangian policy; within/group temperatures=.01/.05"),
            ("Certificate", "goal-directed, max 100,000 nodes, fail closed"),
        ]),
        "",
        "### Duration-aware Dynamic PSLAP",
        "",
        "No parameters were learned. Inbound assignment retained only reachable empty cells that preserved access to earlier-due blocks, then minimized exit distance, travel distance, and cell index. Relocation minimized travel distance, exit distance, and cell index. Retrieval was dispatched when the complete travel/handling plan had nonpositive slack, ordered by slack, relocation count, plan length, and block label. The baseline was deterministic and did not use the exact VCG certificate.",
        "",
        "### Capacity-aware rolling GA",
        "",
        _table([
            ("Information", "stored inventory plus arrived pending blocks only"),
            ("Population / generations", f"{ga['population_size']} / {ga['generations_per_decision']} at every inbound decision"),
            ("Elites / tournament", f"{ga['elite_count']} / {ga['tournament_size']}"),
            ("Crossover", f"one-point with uniqueness repair; probability={_fmt(ga['crossover_probability'])}"),
            ("Mutation probability", _fmt(ga['mutation_probability'])),
            ("Fitness", "lexicographic infeasible known placements, predicted obstruction, route length"),
            ("Replications", "4 optimizer RNG realizations per EpisodeInstance; never best-of-4"),
        ]),
        "",
        "### A3C adaptation",
        "",
        _table([
            ("Status", "Kim et al. (2020)-inspired adaptation; not an exact reproduction"),
            ("Observation", "5 channels: relative dwell, occupancy, storage mask, traversability, exit proximity"),
            ("Network", "two 3x3 convolution layers with 32 channels; masked per-cell actor and scalar critic"),
            ("Optimizer", f"Adam, lr={_fmt(a3c['learning_rate'])}, weight decay={_fmt(a3c['weight_decay'])}"),
            ("Loss / update", f"gamma={_fmt(a3c['gamma_per_placement_decision'])}; entropy={_fmt(a3c['entropy_coefficient'])}; value={_fmt(a3c['value_coefficient'])}; gradient clip={_fmt(a3c['gradient_clip'])}; 1 update/eligible episode"),
            ("Training reward", "negative relocations attributed to the placed block; incomplete/censored episodes skipped"),
            ("Training / validation", "1,000 episodes per seed; validation every 100 episodes on five fixed instances with five stochastic rollouts"),
            ("Selected episodes", "seed 0/1/2 = 100/200/700"),
            ("Deployment", "stochastic masked categorical placement; 5 policy RNGs per model-instance"),
            ("Physical control", "common deterministic navigation and atomic storage/retrieval executors"),
        ]),
        "",
        "## Raw run accounting",
        "",
        _table([
            ("VCG", "900 = 30 instances x 3 model seeds x 10 lambda values"),
            ("Historical VCG 2.3", "360 = 30 x 3 model seeds x 4 policy RNGs"),
            ("Dynamic PSLAP", "30 = 30 x one deterministic run"),
            ("Capacity-aware GA", "120 = 30 x 4 optimizer RNGs"),
            ("A3C adaptation", "450 = 30 x 3 model seeds x 5 policy RNGs"),
        ]),
        "",
        "These unequal raw counts are intentional nuisance replication. They do not represent different test panels: every method was evaluated on the same 30 instances, and inference was performed at the instance level after within-instance averaging.",
        "",
        "## Principal experiment coordinates",
        "",
        "These settings are experimental coordinates, not additional learned hyperparameters. Reused controls are identified so that reanalysis is not mistaken for independent replication.",
        "",
        _table([
            ("E3: certification", "physical-feasibility-only vs recoverability-certified; lambda=.10; 3 model seeds x 30 instances; 90 new physical-only rows, certified control reused from E1"),
            ("E4: ranking", "random-safe, heuristic-safe, frozen Qop-safe, and conditioned VCG at lambda in {.05,.10,.20}; 3 model seeds x 30 instances; five random-ranking seeds"),
            ("E5(a): model validation", f"{coordinates['E05_handling_model']['validation_fixed_states']} fixed states, {coordinates['E05_handling_model']['validation_fixed_candidates']} candidates, {coordinates['E05_handling_model']['validation_lambda_count']} lambda values"),
            ("E5(b): preference input", "deployment lambda in {.05,.10,.20}; comparator clamps predictor input to .10 while retaining requested deployment multiplier"),
            ("E5(c): future consequences", "immediate Reconfigure cost only vs fixed learned future-handling estimate; lambda in {.05,.10,.20}; 270 new immediate-only rows, controls reused from E4/E5(b)"),
            ("E8: liveness", "descriptive instrumented pilot; 8x8 high occupancy, instance 95100, model seed 0, lambda=.10, 20,000-node exact-search budget and 600-s wall limit"),
            ("E16: score component", "immediate-only, action-type-only, candidate-residual-only, full-future; predictor and deployment lambda=.10; 3 model seeds x 30 instances = 360 rows"),
            ("E19: service cost", "analysis only; tardiness weights {1,2,4}, rehandle equivalents {0,10,25,50} timing steps, job-level quantiles {.50,.90,.95,.99}"),
        ]),
        "",
        "### E11 distribution-shift panel",
        "",
        "All E11 policies were frozen. The panel used model seeds 0--2 and instance seeds 93000--93029. Frozen Qop represented lambda=0; conditioned VCG used lambda in {.05,.10,.20}; the heuristic-safe comparator was deterministic. No deployment-specific preference selection was allowed.",
        "",
        "| Regime | Shift | Mean inter-arrival | Duration | Entry |",
        "|---|---|---:|---|---|",
        *[
            f"| {regime['id']} | {regime['axis']} | {_fmt(regime['mean_interarrival'])} | {regime['duration_profile']} (mean {_fmt(regime['duration_mean'])}) | {'mirrored' if regime['mirrored_entry'] else 'reference'} |"
            for regime in coordinates["E11_distribution_shift"]["regimes"]
        ],
        "",
        "The compared methods were frozen Qop-safe, conditioned VCG, and heuristic-safe. The complete design contained 2,730 rows; the exact same EpisodeInstance was shared within each regime-policy comparison.",
        "",
        "### E12 representation ablation",
        "",
        _table([
            ("Representations", "full relational successor; nonrelational successor; relational current-plus-candidate without explicit successor"),
            ("Regimes", "the seven E11 regimes"),
            ("Preferences", "lambda in {0,.05,.10,.20}"),
            ("Seeds and rows", "3 model seeds x 30 instances; 7,560 corrected-executor rows"),
            ("Training", "18 runs total: 9 operational runs (3 representations x 3 seeds, 500 episodes) and 9 handling runs (400 episodes = 8 rounds x 50); operational critic frozen during handling training"),
            ("Corrected-executor confirmation", "no retraining; 7,560 fresh evaluations of the frozen representation variants"),
        ]),
        "",
        "### E13 scale, occupancy, geometry, and episode length",
        "",
        "E13 used frozen model seed 0 at lambda=.10 on instance seeds 95100--95102. The verifier used goal-directed exact search with a 20,000-node cap, queue-cell reservation, timing-invariant caching, order-preserving path cleanup, and validated relocation-family certificates. Runs were isolated by process with a 1,800-s episode wall limit; latency thresholds were .1, 1, and 5 s.",
        "",
        "| Scenario | Comparison | Grid | Initial occupancy | Total jobs | Step cap |",
        "|---|---|---:|---:|---:|---:|",
        *[
            f"| {scenario['id']} | {scenario['comparison']} | {scenario['grid'][0]}x{scenario['grid'][1]} | {scenario['initial_occupied_slots']}/{scenario['storage_capacity']} ({100.0 * scenario['initial_occupancy_ratio']:.1f}%) | {scenario['total_jobs']} | {scenario['max_steps']} |"
            for scenario in coordinates["E13_operational_scalability"]["scenarios"]
        ],
        "",
        "### E14 certification-budget sensitivity",
        "",
        _table([
            ("Predeclared max-node budgets", "2, 4, 8, 16, and 20,000; the 20,000-node reference reused E13"),
            ("Post-hoc threshold refinement", "32 and 64 nodes, explicitly labeled post hoc"),
            ("Panel", "nine 5x5/8x8/10x10 low/medium/high scenarios; instance 95100, model seed 0, lambda=.10"),
            ("Proof isolation", "fresh per-episode cache; anchors acquired under the arm's own budget; no cross-budget positive-proof reuse; UNKNOWN never reused as UNSAFE"),
            ("Timing", "isolated CPU process; .1/1/5-s decision thresholds; 1,800-s episode wall limit"),
        ]),
        "",
        "## Provenance",
        "",
        f"Machine-readable registry SHA-256: `{registry['registry_sha256']}`.",
        "",
        "The companion `e1-parameter-register.json` contains full checkpoint digests and source-artifact hashes. Historical VCG 2.3 is reported separately because it is not a preference coordinate of the final conditioned VCG. The Kim2020-inspired comparator is named **A3C adaptation** throughout the paper.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-root", type=Path, default=DEFAULT_E1_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--paper-copy",
        type=Path,
        default=PROJECT_ROOT / "docs/PAPER_PARAMETER_REPORT.md",
    )
    args = parser.parse_args()
    output = (args.output_dir or args.e1_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = build_registry(args.e1_root)
    json_path = output / "e1-parameter-register.json"
    md_path = output / "e1-parameter-report.md"
    json_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(registry)
    md_path.write_text(markdown, encoding="utf-8")
    paper_copy = args.paper_copy.resolve()
    paper_copy.parent.mkdir(parents=True, exist_ok=True)
    paper_copy.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "json": str(json_path),
        "markdown": str(md_path),
        "paper_copy": str(paper_copy),
        "registry_sha256": registry["registry_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
