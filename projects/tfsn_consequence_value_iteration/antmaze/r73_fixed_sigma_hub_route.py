#!/usr/bin/env python3
"""Matched fixed-sigma acquisition-memory arm for R72."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r65_r52_area_weighted_hubs as r65,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r67_e549_hub_routed_exploration as r67,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r68_e549_dynamic_hubs as r68,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r72_hub_route_correction as implementation,
)
from projects.tfsn_consequence_value_iteration.shared.latent_detection import (
    SigmaKernelPosterior,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R73_FIXED_SIGMA_HUB_ROUTE_REFERENCE.md"
PROTOCOL = HERE / "r73_fixed_sigma_hub_route_protocol.json"
METHOD_REVISION = "antmaze_e549_5m_fixed_sigma_hub_route_correction_r73"
HORIZON = implementation.HORIZON
EPISODES = implementation.EPISODES
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_fixed_sigma_hub_route_correction_r73/original9_long_budget"
)


def load_protocol() -> dict:
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_preparation_revision": r52.METHOD_REVISION,
        "base_evaluation_revision": implementation.METHOD_REVISION,
        "task_ids": list(range(9)),
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": HORIZON * r48.COMMAND_STEPS,
        "fixed_command_duration": r48.COMMAND_STEPS,
        "evaluation_seed": 607,
        "acquisition_observation_memory": (
            "fixed_R52_landing_scale_soft_Gaussian_complement"
        ),
        "acquisition_search_sigma": (
            "deployed_R52_landing_sigma_fixed_within_evaluation"
        ),
        "latent_radius_posterior_used": False,
        "hub_route_similarity_cutoff": None,
        "hub_route_prior_weight": float(RECORDED_EVIDENCE_PRIOR_WEIGHT),
        "hub_evaluation_cutoff": None,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R73 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R73 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


class AreaWeightedSigmaKernelPosterior(SigmaKernelPosterior):
    """R59 soft memory on R65's area-weighted finite support."""

    def __init__(self, recorded_goal_xy: np.ndarray, sigma: float):
        super().__init__(recorded_goal_xy, sigma)
        weights, _occupied, _origin = r65.area_weights(
            self.goal_xy, float(sigma)
        )
        self.goal_probability = weights.copy()
        self.feasible = self.goal_probability > 0.0
        self.marginal_evidence = 1.0


def fixed_area_posterior(
    nodes: np.ndarray, sigma: float
) -> AreaWeightedSigmaKernelPosterior:
    support = np.unique(nodes[:, :2].astype(np.float64), axis=0)
    return AreaWeightedSigmaKernelPosterior(support, sigma)


def fixed_negative_path_update(
    posterior: AreaWeightedSigmaKernelPosterior, passage_xy: np.ndarray
) -> float:
    """Apply exactly one fixed-sigma no-contact passage likelihood."""
    passage = np.asarray(passage_xy, np.float64).reshape(-1, 2)
    if not len(passage):
        return 0.0
    started = time.perf_counter()
    distance = np.linalg.norm(
        passage[:, None] - posterior.goal_xy[None], axis=2
    ).min(axis=0)
    residual = 1.0 - posterior._kernel(distance)
    unnormalized = posterior.goal_probability * residual
    evidence = float(unnormalized.sum())
    if evidence > 0.0:
        posterior.goal_probability = unnormalized / evidence
        posterior.marginal_evidence *= evidence
    else:
        posterior.goal_probability.fill(0.0)
        posterior.marginal_evidence = 0.0
    posterior.feasible = posterior.goal_probability > 0.0
    posterior.negative_observations += len(passage)
    posterior.assert_not_exhausted("after R73 traversed path")
    return time.perf_counter() - started


def fixed_all_hub_coverage(
    posterior: AreaWeightedSigmaKernelPosterior, hub_xy: np.ndarray
) -> np.ndarray:
    """Vectorized fixed-sigma contact readout for every dynamic hub."""
    posterior.assert_not_exhausted("before R73 dynamic hub scoring")
    hubs = np.asarray(hub_xy, np.float64).reshape(-1, 2)
    chunk = int(r68.PROTOCOL_VALUE["hub_scoring_chunk_computational_only"])
    output = np.empty(len(hubs), np.float64)
    for begin in range(0, len(hubs), chunk):
        stop = min(begin + chunk, len(hubs))
        distance = np.linalg.norm(
            posterior.goal_xy[:, None] - hubs[None, begin:stop], axis=2
        )
        output[begin:stop] = (
            posterior.goal_probability @ posterior._kernel(distance)
        )
    return np.clip(output, 0.0, 1.0)


_R72_LOAD_RUNTIME = implementation._load_runtime
_R72_COMMAND_AUDIT = implementation.command_audit
_R72_COMMAND_EVALUATE = implementation.command_evaluate


def configure_implementation() -> None:
    implementation.REFERENCE = REFERENCE
    implementation.PROTOCOL = PROTOCOL
    implementation.METHOD_REVISION = METHOD_REVISION
    implementation.PROTOCOL_VALUE = PROTOCOL_VALUE
    implementation.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    implementation.PLOT_TITLE = (
        "R73 E549-5M fixed-σ hub-route correction — "
        "20 commands × 10 episodes"
    )
    implementation.PLOT_STEM = "r73_fixed_sigma_hub_route_original9"
    implementation._load_runtime = _load_runtime
    implementation.command_audit = command_audit
    r67._area_posterior = fixed_area_posterior
    r67._negative_path_update = fixed_negative_path_update
    r68._all_hub_coverage = fixed_all_hub_coverage


def _load_runtime(args):
    loaded = _R72_LOAD_RUNTIME(args)
    # R72's binder reinstalls only its reporting contract.  Replace the two
    # acquisition-memory functions after loading so every task uses fixed σ.
    r67._area_posterior = fixed_area_posterior
    r67._negative_path_update = fixed_negative_path_update
    r68._all_hub_coverage = fixed_all_hub_coverage
    planner = loaded[0]
    planner.method_revision = METHOD_REVISION
    planner.planner_name = (
        "E549_R72_route_correction_fixed_sigma_acquisition_memory"
    )
    return loaded


def command_audit(args) -> dict:
    configure_implementation()
    report = _R72_COMMAND_AUDIT(args)
    consequence, nodes = (
        r67._graph_arrays(Path(args.preparation).resolve())[1:3]
    )
    posterior = fixed_area_posterior(
        nodes, float(consequence["landing_sigma_m"])
    )
    coverage_smoke = fixed_all_hub_coverage(posterior, nodes[:3, :2])
    coverage_contract_ok = bool(
        coverage_smoke.shape == (3,)
        and np.isfinite(coverage_smoke).all()
        and np.all((coverage_smoke >= 0.0) & (coverage_smoke <= 1.0))
    )
    report.update({
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "acquisition_observation_memory": PROTOCOL_VALUE[
            "acquisition_observation_memory"
        ],
        "acquisition_search_sigma": PROTOCOL_VALUE[
            "acquisition_search_sigma"
        ],
        "latent_radius_posterior_used": False,
        "fixed_sigma_hub_coverage_contract_ok": coverage_contract_ok,
        "ready_for_evaluation": bool(
            report["ready_for_evaluation"] and coverage_contract_ok
        ),
    })
    print(json.dumps(report, indent=2))
    return report


def command_evaluate(args) -> dict:
    configure_implementation()
    result = _R72_COMMAND_EVALUATE(args)
    result.update({
        "acquisition_observation_memory": PROTOCOL_VALUE[
            "acquisition_observation_memory"
        ],
        "acquisition_search_sigma": PROTOCOL_VALUE[
            "acquisition_search_sigma"
        ],
        "latent_radius_posterior_used": False,
    })
    r48.dump(Path(args.output).resolve() / "summary.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    value.add_argument("--episodes", type=int, default=EPISODES)
    value.add_argument("--preparation", type=Path, default=r52.DEFAULT_OUTPUT)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--graph", type=Path)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("evaluate").set_defaults(func=command_evaluate)
    return value


def main() -> None:
    configure_implementation()
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
