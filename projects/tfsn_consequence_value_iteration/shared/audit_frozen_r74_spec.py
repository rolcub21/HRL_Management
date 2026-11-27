"""Verify the canonical frozen R74/R15 cross-domain specification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .frozen_hub_method import (
    alias_hub_candidates,
    area_weights,
    empirical_reach_covariance,
)
from .latent_detection import exact_support_diameter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SETTINGS = HERE / "frozen_r74_settings.json"
SPECIFICATION = ROOT / (
    "projects/tfsn_consequence_value_iteration/"
    "FROZEN_R74_METHOD_SPECIFICATION.md"
)
FREEZE = ROOT / (
    "projects/tfsn_consequence_value_iteration/"
    "FROZEN_R74_CROSS_DOMAIN_FREEZE.json"
)
BASE = ROOT / "projects/tfsn_consequence_value_iteration/results"
ANT_GRAPH = BASE / (
    "antmaze/e549_5m_preparation_r52/runtime/recorded_transition_graph.npz"
)
ANT_TASK = BASE / (
    "antmaze/e549_5m_r74_twenty_episode_r75/original9/task_000.json"
)
ANT_FREEZE = ROOT / (
    "projects/tfsn_consequence_value_iteration/antmaze/R74_FREEZE.json"
)
POINT_GRAPH = BASE / (
    "pointmaze/continuous_reference_roadmap_r11/recorded_graph.npz"
)
POINT_BRIDGES = BASE / (
    "pointmaze/continuous_reference_planner_r11/bridges.json"
)
POINT_GATE = BASE / "pointmaze/frozen_r74_port_r15/gate.json"
POINT_PROTOCOL = ROOT / (
    "projects/tfsn_consequence_value_iteration/pointmaze/"
    "frozen_r74_port_protocol.json"
)
ANT_PROTOCOL = ROOT / (
    "projects/tfsn_consequence_value_iteration/antmaze/"
    "r74_reach_scale_route_correction_protocol.json"
)
SHARED_MANIFEST = HERE / "method_manifest.json"


def _load(path: Path):
    return json.loads(path.read_text())


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(left, right, *, atol=1e-10):
    return bool(np.allclose(left, right, rtol=0.0, atol=atol))


def _require(condition, message, failures):
    if not condition:
        failures.append(message)


def _audit_global(settings, failures):
    manifest = _load(SHARED_MANIFEST)["categories"]["global_computational"]
    mapping = {
        "command_duration_tau_primitive_steps": "command_steps",
        "consequence_ensemble_members": "ensemble_size",
        "mixture_components_per_head": "mixture_components",
        "consequence_hidden_dimension": "hidden_dimension",
        "consequence_hidden_layers": "hidden_layers",
        "training_batch_size": "batch_size",
        "training_epochs": "epochs",
        "training_learning_rate": "learning_rate",
        "validation_fraction": "validation_fraction",
        "standard_deviation_floor": "std_floor",
        "bridge_candidates_per_node": "bridge_candidates",
        "maximum_alignment_rounds": "maximum_alignment_rounds",
        "alignment_relative_stop_tolerance": "alignment_relative_tolerance",
        "alignment_tasks_per_phase": "alignment_tasks_per_phase",
        "alignment_commands_per_task": "alignment_commands_per_task",
        "gaussian_circle_quadrature_points": "gaussian_circle_quadrature_points",
        "passage_bins": "passage_bins",
        "consumptive_lookahead_depth": "consumptive_lookahead_depth",
        "consumptive_lookahead_beam_width": "consumptive_lookahead_beam_width",
        "recorded_reliability_neighbors": "recorded_reliability_neighbors",
        "online_evidence_prior_weight": "recorded_evidence_prior_weight",
        "situation_density_truncation_sigmas": "situation_density_truncation_sigmas",
        "pair_inference_batch": "pair_inference_batch",
        "artifact_node_shard": "artifact_node_shard",
        "field_evaluation_batch": "field_evaluation_batch",
    }
    declared = settings["global_computational"]
    for name, source_name in mapping.items():
        _require(name in declared, f"missing global setting {name}", failures)
        if name in declared:
            _require(
                declared[name]["value"] == manifest[source_name],
                f"global setting changed: {name}",
                failures,
            )
    point_protocol = _load(POINT_PROTOCOL)
    _require(
        declared["exact_hub_target_batch"]["value"]
        == point_protocol["hub_target_batch_computational"] == 64,
        "exact hub target batch changed",
        failures,
    )
    alignment_tasks = int(manifest["alignment_tasks_per_phase"])
    commands_per_task = int(manifest["alignment_commands_per_task"])
    total = int(declared["preparation_commands_per_seed"]["value"])
    bootstrap_tasks = int(declared["bootstrap_tasks_per_seed"]["value"])
    _require(total == 36_000, "preparation command budget changed", failures)
    _require(
        declared["bootstrap_commands_per_task"]["value"]
        == commands_per_task,
        "bootstrap command count differs from the shared task length",
        failures,
    )
    _require(
        bootstrap_tasks * commands_per_task
        + 2 * alignment_tasks * commands_per_task
        == total,
        "preparation phase allocation does not sum to its budget",
        failures,
    )


def _audit_domain(name, graph_path, declared, failures):
    _require(graph_path.is_file(), f"missing {name} recorded graph", failures)
    if not graph_path.is_file():
        return None
    _require(
        _sha256(graph_path) == declared["recorded_graph_sha256"],
        f"{name} recorded graph hash changed",
        failures,
    )
    with np.load(graph_path, allow_pickle=False) as graph:
        nodes = graph["nodes"]
        source = graph["source_index"]
        endpoint = graph["endpoint_index"]
        commands = graph["commands"]
    sigma = float(declared["landing_sigma_m"])
    support = np.unique(nodes[:, :2].astype(np.float64), axis=0)
    _prior, occupied, _origin = area_weights(support, sigma)
    hubs, _inverse, _hub_origin = alias_hub_candidates(nodes[:, :2], sigma)
    reach = empirical_reach_covariance(
        nodes[:, :2], source, endpoint, commands, sigma
    )
    diameter = exact_support_diameter(support)
    checks = {
        "unique_recorded_xy": len(support),
        "occupied_sigma_cells_and_hubs": occupied,
        "hub_count": len(hubs),
        "recorded_nodes": len(nodes),
        "recorded_edges": len(source),
    }
    for key, actual in checks.items():
        expected_key = (
            "occupied_sigma_cells_and_hubs" if key == "hub_count" else key
        )
        _require(
            int(actual) == int(declared[expected_key]),
            f"{name} derived {key} changed",
            failures,
        )
    _require(
        _close(diameter, declared["rho_max_m"]),
        f"{name} support diameter/rho_max changed",
        failures,
    )
    _require(
        _close(sigma, declared["rho_min_m"]),
        f"{name} rho_min is not landing sigma",
        failures,
    )
    for key in (
        "mean_displacement_m",
        "covariance_m2",
        "principal_standard_deviation_m",
        "radial_rms_m",
    ):
        declared_key = f"reach_{key}"
        _require(
            _close(reach[key], declared[declared_key]),
            f"{name} derived {declared_key} changed",
            failures,
        )
    tau = 100
    T = int(declared["evaluation_T_primitive_steps"])
    _require(T % tau == 0, f"{name} T is not divisible by tau", failures)
    _require(
        T // tau == int(declared["derived_H_commands"]),
        f"{name} H is not T/tau",
        failures,
    )
    return {
        "landing_sigma_m": sigma,
        "rho_bounds_m": [sigma, diameter],
        "reach_covariance_m2": reach["covariance_m2"].tolist(),
        "hubs": int(len(hubs)),
        "H": int(T // tau),
    }


def audit():
    failures = []
    settings = _load(SETTINGS)
    _require(SPECIFICATION.is_file(), "canonical specification is missing", failures)
    _require(
        settings.get("specification_revision")
        == "tfsn_frozen_cross_domain_r74_point_r15",
        "specification revision changed",
        failures,
    )
    _require(
        settings.get("environment_specific_policy_rules") == [],
        "environment-specific policy rule was declared",
        failures,
    )
    freeze = _load(FREEZE)
    _require(
        freeze.get("specification_revision")
        == settings.get("specification_revision"),
        "cross-domain freeze revision differs from settings inventory",
        failures,
    )
    _require(
        freeze.get("canonical_specification_sha256") == _sha256(SPECIFICATION),
        "cross-domain freeze does not bind the specification",
        failures,
    )
    _require(
        freeze.get("settings_inventory_sha256") == _sha256(SETTINGS),
        "cross-domain freeze does not bind the settings inventory",
        failures,
    )
    for domain, row in freeze.get("domain_implementations", {}).items():
        implementation = ROOT / row["implementation"]
        _require(
            implementation.is_file()
            and _sha256(implementation) == row["implementation_sha256"],
            f"cross-domain freeze implementation changed: {domain}",
            failures,
        )
    _audit_global(settings, failures)

    domains = settings["domain_realizations"]
    derived = {
        "antmaze": _audit_domain(
            "antmaze", ANT_GRAPH, domains["antmaze"], failures
        ),
        "pointmaze": _audit_domain(
            "pointmaze", POINT_GRAPH, domains["pointmaze"], failures
        ),
    }

    ant_task = _load(ANT_TASK)
    ant_freeze = _load(ANT_FREEZE)
    _require(
        ant_freeze["method_revision"]
        == settings["source_method_revisions"]["antmaze"],
        "AntMaze frozen revision changed",
        failures,
    )
    for key in (
        "executor_checkpoint_sha256",
        "factorized_consequence_sha256",
        "mobility_sha256",
        "bridges_sha256",
    ):
        _require(
            ant_task[key] == domains["antmaze"][key],
            f"AntMaze frozen artifact changed: {key}",
            failures,
        )

    point_bridge = _load(POINT_BRIDGES)
    point_artifact_keys = {
        "recorded_graph_sha256": "recorded_graph_sha256",
        "factorized_consequence_sha256": "factorized_consequence_sha256",
        "mobility_sha256": "mobility_sha256",
        "bridges_sha256": "artifact_sha256",
    }
    for declared_key, artifact_key in point_artifact_keys.items():
        _require(
            domains["pointmaze"][declared_key] == point_bridge[artifact_key],
            f"PointMaze frozen artifact changed: {declared_key}",
            failures,
        )
    _require(
        _close(
            domains["pointmaze"]["landing_sigma_m"],
            point_bridge["landing_sigma_m"],
        ),
        "PointMaze landing sigma changed",
        failures,
    )

    for protocol_path in (ANT_PROTOCOL, POINT_PROTOCOL):
        protocol = _load(protocol_path)
        for key in (
            "link_similarity_cutoff",
            "hub_route_similarity_cutoff",
            "route_similarity_cutoff",
            "hub_evaluation_cutoff",
        ):
            if key in protocol:
                _require(
                    protocol[key] is None,
                    f"{protocol_path.name} introduced {key}",
                    failures,
                )
        _require(
            protocol.get("hidden_goal_visible_to_planner_executor_or_models")
            is False,
            f"{protocol_path.name} exposes the hidden goal",
            failures,
        )
        _require(
            protocol.get("success_radius_read_by_planner") is False,
            f"{protocol_path.name} exposes the success radius",
            failures,
        )

    gate = _load(POINT_GATE)
    _require(gate.get("passed") is True, "PointMaze 24-goal gate failed", failures)
    _require(
        gate.get("frozen_r74_port_acquired") == 24
        and gate.get("tasks") == 24,
        "PointMaze cross-domain preservation is not 24/24",
        failures,
    )
    _require(
        gate.get("confirmation_panel_accessed") is False,
        "PointMaze confirmation panel was accessed",
        failures,
    )

    report = {
        "schema_version": 1,
        "specification_revision": settings["specification_revision"],
        "canonical_specification": str(SPECIFICATION),
        "canonical_specification_sha256": _sha256(SPECIFICATION),
        "settings_inventory": str(SETTINGS),
        "settings_inventory_sha256": _sha256(SETTINGS),
        "cross_domain_freeze": str(FREEZE),
        "cross_domain_freeze_sha256": _sha256(FREEZE),
        "global_settings_verified": len(settings["global_computational"]),
        "derived_domain_realizations": derived,
        "environment_specific_policy_rules": [],
        "pointmaze_development_gate_acquired": gate.get(
            "frozen_r74_port_acquired"
        ),
        "pointmaze_confirmation_panel_accessed": False,
        "failures": failures,
        "passed": not failures,
    }
    if failures:
        raise ValueError(json.dumps(report, indent=2))
    return report


def main():
    print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
