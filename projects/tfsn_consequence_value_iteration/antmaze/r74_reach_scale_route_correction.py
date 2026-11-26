#!/usr/bin/env python3
"""R72 with route evidence generalized at empirical command reach."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze import (
    r48_rule_free_r39 as r48,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r52_e549_preparation as r52,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r67_e549_hub_routed_exploration as r67,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r72_hub_route_correction as implementation,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
)


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R74_REACH_SCALE_ROUTE_CORRECTION_REFERENCE.md"
PROTOCOL = HERE / "r74_reach_scale_route_correction_protocol.json"
METHOD_REVISION = "antmaze_e549_5m_reach_scale_hub_route_correction_r74"
HORIZON = implementation.HORIZON
EPISODES = implementation.EPISODES
DEFAULT_OUTPUT = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "e549_5m_reach_scale_hub_route_correction_r74/original9_long_budget"
)
R72_OUTPUT = implementation.DEFAULT_OUTPUT


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
        "hub_route_similarity": (
            "continuous_gaussian_mahalanobis_over_attempt_source_xy_and_"
            "hub_xy_each_using_the_same_R52_empirical_movement_weighted_"
            "reach_covariance_with_exact_remaining_horizon"
        ),
        "hub_route_similarity_cutoff": None,
        "hub_route_prior_weight": float(RECORDED_EVIDENCE_PRIOR_WEIGHT),
        "route_reach_covariance_estimator": (
            "weighted_central_second_moment_of_R52_realized_displacement_"
            "using_the_declared_continuous_requested_advance_times_"
            "realized_advance_weight"
        ),
        "route_source_scale": "R52_empirical_reach_covariance",
        "route_hub_scale": "R52_empirical_reach_covariance",
        "reachability_computational_backend": "exact_batched_rolling_torch",
        "coverage_computational_backend": "exact_chunked_torch",
        "computational_target_batch_source": (
            "declared_R69_hub_scoring_chunk_64"
        ),
        "bridge_evidence_computational_backend": "unchanged_R71_numpy",
        "computational_equivalence_base_revision": (
            "antmaze_e549_5m_hub_route_online_correction_r72"
        ),
        "computational_equivalence_base_reference_sha256": (
            "1142c7bf331dce08535f3dc3dbbb2817def020663cdfd2088af6cf906c3cc7b3"
        ),
        "computational_equivalence_base_implementation_sha256": (
            "616f7e28cb55ce2ffa207c068ded3629baf8c6dc5e4a5c98ec4890cd18cde441"
        ),
        "corrected_hub_selection": (
            "exact_global_argmax_by_coverage_upper_bound_branch_and_bound"
        ),
        "hub_evaluation_cutoff": None,
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"R74 protocol has another {key}")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE.resolve():
        raise ValueError("R74 method reference changed")
    return value


PROTOCOL_VALUE = load_protocol()


def weighted_central_covariance(
    displacement: np.ndarray, sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean and covariance with no fitted scale or regularizer."""
    displacement = np.asarray(displacement, np.float64).reshape(-1, 2)
    sample_weight = np.asarray(sample_weight, np.float64).reshape(-1)
    if len(displacement) != len(sample_weight):
        raise ValueError("displacements and weights have different lengths")
    if not np.isfinite(displacement).all() or not np.isfinite(
        sample_weight
    ).all():
        raise ValueError("reach observations must be finite")
    if np.any(sample_weight < 0.0):
        raise ValueError("reach weights must be nonnegative")
    total = float(sample_weight.sum())
    if total <= 0.0:
        raise ValueError("reach observations have zero total weight")
    mean = (sample_weight[:, None] * displacement).sum(axis=0) / total
    centered = displacement - mean
    covariance = (
        np.einsum("n,ni,nj->ij", sample_weight, centered, centered)
        / total
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.isfinite(covariance).all() or np.any(eigenvalues <= 0.0):
        raise ValueError("empirical reach covariance is not positive definite")
    return mean, covariance


def empirical_reach_covariance(preparation: Path) -> dict:
    """Derive the shared source/hub route scale from frozen R52 data."""
    arrays = r67._graph_arrays(Path(preparation).resolve())
    paths, consequence, nodes, source, endpoint = arrays[:5]
    with np.load(paths["graph"], allow_pickle=False) as stored:
        commands = stored["commands"].astype(np.float64)
    start = nodes[source, :2].astype(np.float64)
    finish = nodes[endpoint, :2].astype(np.float64)
    sigma = float(consequence["landing_sigma_m"])
    requested_advance = 1.0 - np.exp(
        -np.square(commands - start).sum(axis=1) / (2.0 * sigma**2)
    )
    realized_advance = 1.0 - np.exp(
        -np.square(finish - start).sum(axis=1) / (2.0 * sigma**2)
    )
    weight = requested_advance * realized_advance
    mean, covariance = weighted_central_covariance(finish - start, weight)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return {
        "mean_displacement_m": mean,
        "covariance_m2": covariance,
        "inverse_covariance_per_m2": np.linalg.inv(covariance),
        "principal_standard_deviation_m": np.sqrt(eigenvalues),
        "radial_rms_m": float(np.sqrt(np.trace(covariance))),
        "sample_weight_sum": float(weight.sum()),
        "positive_weight_rows": int(np.count_nonzero(weight > 0.0)),
        "recorded_rows": int(len(weight)),
        "landing_sigma_m": sigma,
        "graph": str(paths["graph"]),
        "graph_sha256": r52.sha256(paths["graph"]),
    }


def reach_route_similarity(
    source_xy: np.ndarray,
    hub_xy: np.ndarray,
    observed_source_xy: np.ndarray,
    observed_hub_xy: np.ndarray,
    inverse_covariance: np.ndarray,
) -> np.ndarray:
    """The declared two-sided Mahalanobis route-evidence kernel."""
    source_delta = (
        np.asarray(source_xy, np.float64)
        - np.asarray(observed_source_xy, np.float64)
    )
    hub_delta = (
        np.asarray(hub_xy, np.float64)
        - np.asarray(observed_hub_xy, np.float64)
    )
    inverse = np.asarray(inverse_covariance, np.float64).reshape(2, 2)
    source_distance = np.einsum(
        "...i,ij,...j->...", source_delta, inverse, source_delta
    )
    hub_distance = np.einsum(
        "...i,ij,...j->...", hub_delta, inverse, hub_delta
    )
    return np.exp(-0.5 * (source_distance + hub_distance))


def route_mass_from_records(
    records: list[dict],
    source_xy: np.ndarray,
    hub_xy: np.ndarray,
    remaining: int,
    inverse_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer uncensored route outcomes at matching decision horizon."""
    hubs = np.asarray(hub_xy, np.float64).reshape(-1, 2)
    support = np.zeros(len(hubs), np.float64)
    arrivals = np.zeros(len(hubs), np.float64)
    for record in records:
        if record.get("censored"):
            continue
        if int(record["remaining_commands"]) != int(remaining):
            continue
        similarity = reach_route_similarity(
            np.asarray(source_xy, np.float64)[None],
            hubs,
            np.asarray(record["source_xy"], np.float64)[None],
            np.asarray(record["hub_xy"], np.float64)[None],
            inverse_covariance,
        )
        support += similarity
        arrivals += similarity * float(record["arrived"])
    return support, arrivals


class ReachScaleHubRouteCorrectedScorer(
    implementation.HubRouteCorrectedScorer
):
    """R72 scorer with the same empirical reach scale on both route sides."""

    def __init__(
        self, planner, arrays, hub_nodes, reach_covariance, device="cuda",
    ):
        super().__init__(planner, arrays, hub_nodes, device=device)
        self.route_reach_covariance = np.asarray(
            reach_covariance, np.float64
        ).reshape(2, 2)
        self.route_reach_inverse = np.linalg.inv(
            self.route_reach_covariance
        )

    def _route_mass(self, source_xy, hub_xy, remaining):
        return route_mass_from_records(
            self.route_records,
            np.asarray(source_xy, np.float64),
            np.asarray(hub_xy, np.float64),
            int(remaining),
            self.route_reach_inverse,
        )

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        self.planner.hub_selection_extra.update({
            "route_source_scale": "R52_empirical_reach_covariance",
            "route_hub_scale": "R52_empirical_reach_covariance",
            "route_reach_covariance_m2": (
                self.route_reach_covariance.tolist()
            ),
            "route_similarity_cutoff": None,
        })
        return result


_R72_COMMAND_EVALUATE = implementation.command_evaluate


def configure_implementation() -> None:
    implementation.REFERENCE = REFERENCE
    implementation.PROTOCOL = PROTOCOL
    implementation.METHOD_REVISION = METHOD_REVISION
    implementation.PROTOCOL_VALUE = PROTOCOL_VALUE
    implementation.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    implementation.PLOT_TITLE = (
        "R74 E549-5M empirical-reach hub-route correction — "
        "20 commands × 10 episodes"
    )
    implementation.PLOT_STEM = "r74_reach_scale_route_correction_original9"
    implementation._load_runtime = _load_runtime
    implementation.command_audit = command_audit


def _load_runtime(args):
    # Start from the same R71 runtime as R72, then install the R74 scorer
    # directly.  This avoids allocating and discarding a second GPU Bellman
    # solver while preserving the loaded planner, executor, graph and table.
    loaded = list(implementation.r71._load_runtime(args))
    implementation._bind_r72_contract()
    reach = empirical_reach_covariance(args.preparation)
    planner = loaded[0]
    planner.method_revision = METHOD_REVISION
    planner.planner_name = (
        "E549_R51_sigma_link_and_empirical_reach_route_correction"
    )
    scorer = ReachScaleHubRouteCorrectedScorer(
        planner,
        r67._graph_arrays(args.preparation),
        loaded[3],
        reach["covariance_m2"],
        args.device,
    )
    planner.hub_score_override = scorer
    planner.hub_attempt_observer = scorer
    loaded[0] = planner
    return tuple(loaded)


def _offline_transfer_audit(
    inverse_covariance: np.ndarray, landing_sigma: float,
) -> dict:
    """Replay R72 route outcomes without rerunning any hidden-goal task."""
    tasks = [1, 4, 6, 7, 8]
    rows = []
    compared = []
    for task_id in tasks:
        path = R72_OUTPUT / f"task_{task_id:03d}.jsonl"
        if not path.is_file():
            continue
        records = []
        for row in implementation._jsonl(path):
            record = row.get("hub_route_attempt_outcome")
            if not record or record.get("censored"):
                continue
            source = np.asarray(record["source_xy"], np.float64)
            hub = np.asarray(record["hub_xy"], np.float64)
            remaining = int(record["remaining_commands"])
            reach_support, reach_arrival = route_mass_from_records(
                records, source, hub[None], remaining, inverse_covariance
            )
            sigma_support = np.zeros(1, np.float64)
            sigma_arrival = np.zeros(1, np.float64)
            for previous in records:
                if int(previous["remaining_commands"]) != remaining:
                    continue
                similarity = implementation.hub_route_similarity(
                    source[None],
                    hub[None],
                    np.asarray(previous["source_xy"], np.float64),
                    np.asarray(previous["hub_xy"], np.float64),
                    float(landing_sigma),
                )
                sigma_support += similarity
                sigma_arrival += similarity * float(previous["arrived"])
            prior = float(record["predicted_reachability"])
            reach_corrected = float(
                implementation.correct_hub_route_probability(
                    [prior], reach_support, reach_arrival
                )[0]
            )
            sigma_corrected = float(
                implementation.correct_hub_route_probability(
                    [prior], sigma_support, sigma_arrival
                )[0]
            )
            item = {
                "task_id": task_id,
                "remaining_commands": remaining,
                "arrived": bool(record["arrived"]),
                "prior_reachability": prior,
                "reach_scale_support": float(reach_support[0]),
                "landing_sigma_support": float(sigma_support[0]),
                "reach_scale_corrected_probability": reach_corrected,
                "landing_sigma_corrected_probability": sigma_corrected,
            }
            rows.append(item)
            if not record["arrived"] and records:
                compared.append(item)
            records.append(record)
    strictly_more = [
        item["reach_scale_support"] > item["landing_sigma_support"]
        for item in compared
        if item["reach_scale_support"] > 0.0
        or item["landing_sigma_support"] > 0.0
    ]
    return {
        "source": str(R72_OUTPUT),
        "tasks": tasks,
        "route_outcomes": len(rows),
        "repeated_failed_route_outcomes_compared": len(strictly_more),
        "reach_scale_support_strictly_exceeds_sigma_count": int(
            sum(strictly_more)
        ),
        "reach_scale_support_strictly_exceeds_sigma_fraction": (
            float(np.mean(strictly_more)) if strictly_more else None
        ),
        "rows": rows,
    }


def _inherited_equivalence_report() -> tuple[Path, dict | None, bool]:
    path = Path(PROTOCOL_VALUE["computational_equivalence_report"])
    if not path.is_absolute():
        path = MANAGEMENT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        return path, None, False
    report = json.loads(path.read_text())
    valid = bool(
        report.get("passed") is True
        and report.get("method_revision")
        == PROTOCOL_VALUE["computational_equivalence_base_revision"]
        and report.get("method_reference_sha256")
        == PROTOCOL_VALUE[
            "computational_equivalence_base_reference_sha256"
        ]
        and report.get("implementation_sha256")
        == PROTOCOL_VALUE[
            "computational_equivalence_base_implementation_sha256"
        ]
        and report.get("same_hub_selected_every_time") is True
        and report.get("bridge_evidence_spread_identical") is True
        and report.get("exact_global_argmax_retained") is True
        and report.get("shortlist_used") is False
    )
    return path, report, valid


def command_audit(args) -> dict:
    configure_implementation()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reach = empirical_reach_covariance(args.preparation)
    transfer = _offline_transfer_audit(
        reach["inverse_covariance_per_m2"], reach["landing_sigma_m"]
    )
    transfer_rows_path = output / "reach_scale_transfer_rows.json"
    r48.dump(transfer_rows_path, transfer)
    transfer_summary = {
        key: value for key, value in transfer.items() if key != "rows"
    }
    table_report = implementation.r69._audit(
        implementation._r69_args(args)
    )
    paths = r52.output_paths(args.preparation)
    required = [
        paths["consequence"],
        paths["graph"],
        paths["runtime"] / "planner/bridges.npz",
        implementation.r68._artifact_paths(
            implementation.r69.DEFAULT_OUTPUT
        )["table"],
        implementation.r68._artifact_paths(
            implementation.r69.DEFAULT_OUTPUT
        )["metadata"],
    ]
    equivalence_path, equivalence, equivalence_ok = (
        _inherited_equivalence_report()
    )
    covariance_ok = bool(
        np.all(np.linalg.eigvalsh(reach["covariance_m2"]) > 0.0)
    )
    transfer_ok = bool(
        transfer["repeated_failed_route_outcomes_compared"] > 0
        and transfer[
            "reach_scale_support_strictly_exceeds_sigma_count"
        ] > 0
    )
    report = {
        "schema_version": 1,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": r52.sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "claim_level": PROTOCOL_VALUE["claim_level"],
        "declared_deviations": PROTOCOL_VALUE["declared_deviations"],
        "oracle_inputs": [],
        "task_ids": PROTOCOL_VALUE["task_ids"],
        "episodes": EPISODES,
        "commands_per_episode": HORIZON,
        "primitive_steps_per_episode": HORIZON * r48.COMMAND_STEPS,
        "landing_sigma_m": reach["landing_sigma_m"],
        "hub_route_outcome": PROTOCOL_VALUE["hub_route_outcome"],
        "hub_route_prior_weight": float(RECORDED_EVIDENCE_PRIOR_WEIGHT),
        "hub_route_similarity": PROTOCOL_VALUE["hub_route_similarity"],
        "hub_route_similarity_cutoff": None,
        "hub_evaluation_cutoff": None,
        "reachability_computational_backend": PROTOCOL_VALUE[
            "reachability_computational_backend"
        ],
        "coverage_computational_backend": PROTOCOL_VALUE[
            "coverage_computational_backend"
        ],
        "computational_equivalence_required": True,
        "computational_equivalence_report": str(equivalence_path),
        "computational_equivalence_passed": equivalence_ok,
        "computational_equivalence_summary": (
            {
                "base_revision": equivalence.get("method_revision"),
                "same_hub_selected_every_time": equivalence.get(
                    "same_hub_selected_every_time"
                ),
                "bridge_evidence_spread_identical": equivalence.get(
                    "bridge_evidence_spread_identical"
                ),
            }
            if equivalence_ok and equivalence is not None else None
        ),
        "route_source_scale": PROTOCOL_VALUE["route_source_scale"],
        "route_hub_scale": PROTOCOL_VALUE["route_hub_scale"],
        "route_reach_covariance_estimator": PROTOCOL_VALUE[
            "route_reach_covariance_estimator"
        ],
        "route_reach_mean_displacement_m": (
            reach["mean_displacement_m"].tolist()
        ),
        "route_reach_covariance_m2": reach["covariance_m2"].tolist(),
        "route_reach_principal_standard_deviation_m": (
            reach["principal_standard_deviation_m"].tolist()
        ),
        "route_reach_radial_rms_m": reach["radial_rms_m"],
        "route_reach_sample_weight_sum": reach["sample_weight_sum"],
        "route_reach_positive_weight_rows": reach[
            "positive_weight_rows"
        ],
        "route_reach_recorded_rows": reach["recorded_rows"],
        "route_reach_graph_sha256": reach["graph_sha256"],
        "route_reach_covariance_positive_definite": covariance_ok,
        "offline_R72_route_evidence_transfer_audit": transfer_summary,
        "offline_R72_route_evidence_transfer_rows": str(
            transfer_rows_path
        ),
        "offline_route_evidence_transfer_observed": transfer_ok,
        "required_artifacts": [
            {"path": str(path), "exists": path.is_file()}
            for path in required
        ],
        "ready_for_evaluation": bool(
            table_report["ready_for_evaluation"]
            and all(path.is_file() for path in required)
            and equivalence_ok
            and covariance_ok
            and transfer_ok
        ),
        "hidden_goal_visible_to_planner_executor_or_models": False,
        "success_radius_read_by_planner": False,
        "task_gradient_updates": 0,
        "confirmation_panel_accessed": False,
    }
    r48.dump(output / "reach_scale_audit.json", report)
    print(json.dumps(report, indent=2))
    return report


def command_evaluate(args) -> dict:
    configure_implementation()
    result = _R72_COMMAND_EVALUATE(args)
    reach = empirical_reach_covariance(args.preparation)
    result.update({
        "route_source_scale": PROTOCOL_VALUE["route_source_scale"],
        "route_hub_scale": PROTOCOL_VALUE["route_hub_scale"],
        "route_reach_covariance_m2": reach["covariance_m2"].tolist(),
        "route_reach_principal_standard_deviation_m": (
            reach["principal_standard_deviation_m"].tolist()
        ),
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
