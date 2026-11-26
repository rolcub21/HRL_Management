"""Rules-only AntMaze port of R39 plus the deployed R46 correction.

This revision deliberately preserves the immediate adaptive-greedy search and
the R46 online target semantics.  It changes only the cross-domain rule
violations declared in ``r48_rule_free_protocol.json``.  Historical R31--R47
source and artifacts remain immutable evidence about earlier methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from projects.tfsn_consequence_value_iteration.antmaze import (
    hidden_goal_empirical_passage_r27 as passage,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    lazy_alias_advance as lazy,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    mobility_stitched_planning as mobility,
)
from projects.tfsn_consequence_value_iteration.antmaze import (
    r31_aligned_hidden_goal as r31,
)
from projects.tfsn_consequence_value_iteration.antmaze.factorized_consequence import (
    FactorizedConsequenceEnsemble,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    RecordedTransitionGraph,
    consequence_inputs,
)
from projects.tfsn_consequence_value_iteration.shared.factorized import (
    diagonal_gaussian_kernel_expectation,
)
from projects.tfsn_consequence_value_iteration.shared.reach import (
    EmpiricalReachQuery,
    representative_reach_offsets,
)
from projects.tfsn_consequence_value_iteration.shared.reliability import (
    recorded_transition_reliability,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    ARTIFACT_NODE_SHARD,
    BATCH_SIZE,
    BRIDGE_CANDIDATES,
    COMMAND_STEPS,
    EPOCHS,
    HIDDEN_DIMENSION,
    LEARNING_RATE,
    MANIFEST_PATH,
    METHOD_MANIFEST,
    PAIR_INFERENCE_BATCH,
    PASSAGE_BINS,
    RECORDED_EVIDENCE_PRIOR_WEIGHT,
    RECORDED_RELIABILITY_NEIGHBORS,
    VALIDATION_FRACTION,
    finite_horizon,
)
from scripts.antmaze.current_tfsn.executor import (
    FaramaSACExecutor,
    full_state,
    make_env,
)
from scripts.antmaze.current_tfsn.fields import TaskFields
from scripts.antmaze.current_tfsn.main import dump, paths


HERE = Path(__file__).resolve().parent
MANAGEMENT_ROOT = HERE.parents[2]
REFERENCE = HERE / "R48_RULE_FREE_R39_REFERENCE.md"
PROTOCOL = HERE / "r48_rule_free_protocol.json"
METHOD_REVISION = "antmaze_r39_r46_rule_removal_r48"
PREPARATION_CLAIM_LEVEL = "reward_free_rule_removal_preparation"
DEFAULT_RESULT_DIRECTORY = (
    MANAGEMENT_ROOT
    / "projects/tfsn_consequence_value_iteration/results/antmaze"
    / "r39_r46_rule_removal_r48"
)
RESULT_DIRECTORY = DEFAULT_RESULT_DIRECTORY
R30_CHECKPOINT = r31.R30_CHECKPOINT
R30_SUMMARY = r31.R30_SUMMARY
SIGMA = float(r31.SIGMA)
STATE_DIM = int(r31.ant_constants.STATE_DIM)
CANDIDATE_BUDGET = int(BRIDGE_CANDIDATES)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_contract():
    """Immutable shared declarations every R48 artifact must record."""
    return {
        "shared_method_manifest": str(MANIFEST_PATH),
        "shared_method_manifest_sha256": sha256(MANIFEST_PATH),
        "shared_method_revision": METHOD_MANIFEST["method_revision"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
    }


def configure_result_directory(path):
    global RESULT_DIRECTORY
    RESULT_DIRECTORY = Path(path).resolve()


def _hash_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_contract():
    value = json.loads(PROTOCOL.read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        "base_search_revision": "antmaze_live_posterior_passage_r39",
        "base_online_revision": "antmaze_r39_online_consequence_correction_r46",
        "shared_method_revision": METHOD_MANIFEST["method_revision"],
        "gate_task_ids": [3, 5],
        "development_task_ids": list(range(15)),
        "episodes_per_task": 5,
    }
    for key, item in expected.items():
        if value.get(key) != item:
            raise ValueError(f"R48 protocol has another {key}")
    expected_sets = {
        "gate2": "consumed_development_rules_only_removal_gate",
        "all15": "consumed_development_full_panel",
    }
    for name, claim_level in expected_sets.items():
        if value.get("evaluation_sets", {}).get(name, {}).get(
            "claim_level"
        ) != claim_level:
            raise ValueError(f"R48 protocol has another {name} claim level")
    reference = Path(value["method_reference"])
    if not reference.is_absolute():
        reference = MANAGEMENT_ROOT / reference
    if reference.resolve() != REFERENCE:
        raise ValueError("R48 method reference changed")
    manifest = Path(value["shared_method_manifest"])
    if not manifest.is_absolute():
        manifest = MANAGEMENT_ROOT / manifest
    if manifest.resolve() != MANIFEST_PATH.resolve():
        raise ValueError("R48 is not bound to the shared method manifest")
    evaluator = value["evaluator"]
    horizon = finite_horizon(evaluator["episode_primitive_steps"])
    if int(value["derived_horizon_commands"]) != horizon:
        raise ValueError("R48 horizon is not T/tau")
    if value["planner_inputs"].get("success_radius") != "forbidden":
        raise ValueError("success radius must be forbidden to the planner")
    return value


PROTOCOL_VALUE = load_contract()
HORIZON = finite_horizon(
    PROTOCOL_VALUE["evaluator"]["episode_primitive_steps"]
)


def graph_path(_root=None):
    return RESULT_DIRECTORY / "recorded_transition_graph.npz"


def artifact_paths(_root=None):
    directory = RESULT_DIRECTORY / "planner"
    return {
        "directory": directory,
        "landmarks": directory / "landmarks.npz",
        "landmark_meta": directory / "landmarks.json",
        "mobility_shards": directory / "mobility_shards",
        "mobility_labels": directory / "mobility_labels.npz",
        "mobility": directory / "mobility.pt",
        "mobility_meta": directory / "mobility.json",
        "bridge_shards": directory / "bridge_shards",
        "bridges": directory / "bridges.npz",
        "bridge_meta": directory / "bridges.json",
    }


def passage_paths(_root=None):
    directory = RESULT_DIRECTORY / "empirical_passage"
    return {
        "directory": directory,
        "metadata": directory / "metadata.json",
        "recorded": directory / "recorded_passages.npz",
        "bridge_shards": directory / "bridge_shards",
    }


class ManifestRecordedTransitionGraph(RecordedTransitionGraph):
    """Recorded graph whose numerical estimator is wholly manifest-bound."""

    def command_support(self, current_state, commands):
        state = np.asarray(current_state, np.float32).reshape(STATE_DIM)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        query = np.concatenate(
            (
                np.repeat((state / self.state_scale)[None], len(commands), 0),
                commands / SIGMA,
            ),
            axis=1,
        )
        count = min(RECORDED_RELIABILITY_NEIGHBORS, self.edge_count)
        distance, index = self.joint_tree.query(query, k=count, workers=-1)
        if count == 1:
            distance, index = distance[:, None], index[:, None]
        similarity = np.exp(-0.5 * np.square(distance))
        support = similarity.sum(axis=1)
        outcome = (similarity * self.edge_reliability[index]).sum(axis=1)
        empirical = np.divide(
            outcome, support, out=np.zeros_like(outcome), where=support > 0.0
        )
        posterior = outcome / (support + RECORDED_EVIDENCE_PRIOR_WEIGHT)
        return {
            "aimed_support_mass": support.astype(np.float32),
            "local_recorded_reliability": empirical.astype(np.float32),
            "pessimistic_success": posterior.astype(np.float32),
        }

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path) as stored:
            arrays = {name: stored[name] for name in stored.files}
        return cls(
            metadata=json.loads(path.with_suffix(".json").read_text()),
            **arrays,
        )


class AntConsequenceAdapter(FactorizedConsequenceEnsemble):
    """Ant state/posture adapter with shared exact mixture operations."""

    def component_mean_endpoint_outcomes(self, states, commands, member):
        states = np.asarray(states, np.float32).reshape(-1, STATE_DIM)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        features = consequence_inputs(states, commands)
        spatial_weight, residual, _spatial_std = self.spatial_parameters(
            features, member
        )
        endpoint_xy = commands[:, None] + residual
        posture_weight, posture_mean, _posture_std = self._posture_parameters(
            features, endpoint_xy, member
        )
        posture_components = posture_weight.shape[2]
        xy = np.repeat(endpoint_xy[:, :, None], posture_components, axis=2)
        joint_weight = spatial_weight[:, :, None] * posture_weight
        endpoint = np.concatenate((xy, posture_mean), axis=3)
        return (
            joint_weight.reshape(len(states), -1).astype(np.float32),
            endpoint.reshape(len(states), -1, STATE_DIM).astype(np.float32),
        )

    def stall_probability(self, states, commands, sigma, member):
        states = np.asarray(states, np.float32).reshape(-1, STATE_DIM)
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        features = consequence_inputs(states, commands)
        weights, residual, stds = self.spatial_parameters(features, member)
        means = commands[:, None] + residual
        component = diagonal_gaussian_kernel_expectation(
            means, stds, states[:, :2], sigma
        )
        return (weights * component).sum(axis=1).astype(np.float32)


def _bind_adapter():
    """Route inherited generic Ant machinery through shared declarations."""
    mobility.LANDING_SIGMA = SIGMA
    mobility.CANDIDATES = CANDIDATE_BUDGET
    mobility.LANDMARK_NEIGHBORS = CANDIDATE_BUDGET
    mobility.NODE_SHARD_SIZE = ARTIFACT_NODE_SHARD
    mobility.PAIR_INFERENCE_BATCH = PAIR_INFERENCE_BATCH
    mobility.BATCH_SIZE = BATCH_SIZE
    mobility.EPOCHS = EPOCHS
    mobility.HIDDEN_DIM = HIDDEN_DIMENSION
    mobility.LEARNING_RATE = LEARNING_RATE
    mobility.MOBILITY_VALIDATION_FRACTION = VALIDATION_FRACTION
    mobility.METHOD_REVISION = METHOD_REVISION
    mobility.CLAIM_LEVEL = PREPARATION_CLAIM_LEVEL
    lazy.CANDIDATES = CANDIDATE_BUDGET
    lazy.LANDMARK_NEIGHBORS = CANDIDATE_BUDGET
    lazy.LANDING_SIGMA = SIGMA
    lazy.alias_kernel = lambda distance, sigma=SIGMA: np.exp(
        -np.square(np.asarray(distance, np.float32)) / (2.0 * float(sigma) ** 2)
    ).astype(np.float32)
    r31.r22.SIGMA = SIGMA
    r31.r22.BRIDGE_CANDIDATES = CANDIDATE_BUDGET
    r31.r22.graph_path = graph_path
    r31.r22.artifact_paths = artifact_paths
    passage.r22.SIGMA = SIGMA
    passage.r22.graph_path = graph_path
    passage.r22.artifact_paths = artifact_paths
    passage.PASSAGE_BINS = PASSAGE_BINS
    passage.NODE_SHARD_SIZE = ARTIFACT_NODE_SHARD
    passage.METHOD_REVISION = METHOD_REVISION
    passage.BASE_METHOD_REVISION = METHOD_REVISION
    passage.artifact_paths = passage_paths


_bind_adapter()


def verify_graph(path):
    graph = ManifestRecordedTransitionGraph.load(path)
    expected = {
        "method_revision": METHOD_REVISION,
        **manifest_contract(),
        "recorded_reliability_neighbors": RECORDED_RELIABILITY_NEIGHBORS,
        "recorded_evidence_prior_weight": RECORDED_EVIDENCE_PRIOR_WEIGHT,
        "field_sigma_m": SIGMA,
        "deployed_consequence_sha256": sha256(R30_CHECKPOINT),
    }
    for key, item in expected.items():
        if graph.metadata.get(key) != item:
            raise ValueError(f"R48 graph has another {key}")
    return graph


def command_prepare_graph(args):
    output = Path(args.graph)
    if output.exists() and output.with_suffix(".json").exists():
        graph = verify_graph(output)
        print(json.dumps({
            "artifact": str(output), "nodes": graph.node_count,
            "edges": graph.edge_count, "resumed": True,
        }, indent=2))
        return
    source = r31.verify_graph(r31.graph_path(args.root))
    starts = source.nodes[source.source_index]
    endpoints = source.nodes[source.endpoint_index]
    reliability, empirical, support = recorded_transition_reliability(
        starts,
        source.commands,
        endpoints,
        source.state_scale,
        SIGMA,
    )
    graph = ManifestRecordedTransitionGraph(
        source.nodes,
        source.source_index,
        source.endpoint_index,
        source.commands,
        reliability,
        source.state_scale,
        metadata=dict(source.metadata),
    )
    graph.metadata.update({
        "method_revision": METHOD_REVISION,
        **manifest_contract(),
        "claim_level": PREPARATION_CLAIM_LEVEL,
        "recorded_reliability_neighbors": RECORDED_RELIABILITY_NEIGHBORS,
        "recorded_evidence_prior_weight": RECORDED_EVIDENCE_PRIOR_WEIGHT,
        "recorded_reliability_estimator": (
            "shared_kernel_posterior_of_repeatable_realized_deltas"
        ),
        "reliability_empirical_mean": float(empirical.mean()),
        "reliability_support_median": float(np.median(support)),
        "deployed_consequence_sha256": sha256(R30_CHECKPOINT),
        "field_sigma_m": SIGMA,
        "task_reward_used": False,
        "hidden_goals_used": False,
    })
    graph.save(output)
    print(json.dumps({
        "artifact": str(output), "artifact_sha256": sha256(output),
        "nodes": graph.node_count, "edges": graph.edge_count,
        "reliability_mean": float(reliability.mean()),
    }, indent=2))


def movement_reach_weight(graph):
    start = graph.nodes[graph.source_index, :2]
    endpoint = graph.nodes[graph.endpoint_index, :2]
    requested = np.exp(
        -np.square(graph.commands - start).sum(axis=1) / (2.0 * SIGMA**2)
    )
    achieved = np.exp(
        -np.square(endpoint - start).sum(axis=1) / (2.0 * SIGMA**2)
    )
    return ((1.0 - requested) * (1.0 - achieved)).astype(np.float32)


def _prepare_reach(graph, args):
    artifact = artifact_paths()
    artifact["directory"].mkdir(parents=True, exist_ok=True)
    displacement = (
        graph.nodes[graph.endpoint_index, :2]
        - graph.nodes[graph.source_index, :2]
    ).astype(np.float32)
    sample_weight = movement_reach_weight(graph)
    offsets = representative_reach_offsets(
        displacement, CANDIDATE_BUDGET, sample_weight=sample_weight
    )
    contract = {
        "method_revision": METHOD_REVISION,
        **manifest_contract(),
        "base_graph_sha256": sha256(args.graph),
        "factorized_consequence_sha256": sha256(R30_CHECKPOINT),
        "candidate_budget_per_family": CANDIDATE_BUDGET,
        "reach_measure": METHOD_MANIFEST["categories"]["data_derived"][
            "reach_measure"
        ],
        "reach_weight": METHOD_MANIFEST["categories"]["data_derived"][
            "reach_weight"
        ],
        "reach_offsets_sha256": _hash_arrays(offsets),
    }
    if artifact["landmarks"].exists() and artifact["landmark_meta"].exists():
        metadata = json.loads(artifact["landmark_meta"].read_text())
        for key, item in contract.items():
            if metadata.get(key) != item:
                raise ValueError(f"R48 reach artifact has another {key}")
        if metadata.get("artifact_sha256") != sha256(artifact["landmarks"]):
            raise ValueError("R48 reach artifact hash changed")
        with np.load(artifact["landmarks"]) as stored:
            return (
                artifact,
                stored["landmark_index"].astype(np.int32),
                stored["target_index"].astype(np.int32),
                stored["target_distance"].astype(np.float32),
                metadata,
            )
    landmark = np.unique(graph.endpoint_index).astype(np.int32)
    query = EmpiricalReachQuery(graph.nodes[landmark, :2], offsets)
    distance, local = query.query(graph.nodes[:, :2], k=CANDIDATE_BUDGET)
    targets = landmark[local].astype(np.int32)
    mobility._save_npz_atomic(
        artifact["landmarks"], landmark_index=landmark,
        target_index=targets, target_distance=distance,
        reach_offsets=offsets, reach_sample_weight=sample_weight,
    )
    metadata = {
        **contract,
        "artifact": str(artifact["landmarks"]),
        "artifact_sha256": sha256(artifact["landmarks"]),
        "landmarks": int(len(landmark)),
        "nodes": graph.node_count,
        "global_landmark_neighbors": int(targets.shape[1]),
        "positive_reach_weight_rows": int(np.sum(sample_weight > 0.0)),
        "reach_weight_mean": float(sample_weight.mean()),
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
    }
    dump(artifact["landmark_meta"], metadata)
    return artifact, landmark, targets, distance, metadata


def movement_probability(model, states, commands):
    members = [
        1.0 - model.stall_probability(states, commands, SIGMA, member)
        for member in range(len(model.members))
    ]
    return np.stack(members, axis=1).min(axis=1).astype(np.float32)


def _prepare_mobility_labels(graph, targets, model, args, artifact):
    artifact["mobility_shards"].mkdir(parents=True, exist_ok=True)
    contract = {
        "method_revision": METHOD_REVISION,
        **manifest_contract(),
        "stage": "analytic_sigma_stall_mobility_labels",
        "base_graph_sha256": sha256(args.graph),
        "factorized_consequence_sha256": sha256(R30_CHECKPOINT),
        "landmarks_sha256": sha256(artifact["landmarks"]),
        "stall_probability": METHOD_MANIFEST["categories"]["data_derived"][
            "stall_probability"
        ],
        "candidate_budget_per_family": CANDIDATE_BUDGET,
    }
    mobility._bind_shards(artifact["mobility_shards"], contract)
    for begin in range(0, graph.node_count, ARTIFACT_NODE_SHARD):
        stop = min(begin + ARTIFACT_NODE_SHARD, graph.node_count)
        path = artifact["mobility_shards"] / f"nodes_{begin:06d}_{stop:06d}.npz"
        if path.exists() and mobility._valid_npz(path, "mobility", (stop - begin,)):
            continue
        target = targets[begin:stop]
        states = np.repeat(
            graph.nodes[begin:stop, None], target.shape[1], axis=1
        ).reshape(-1, STATE_DIM)
        commands = graph.nodes[target, :2].reshape(-1, 2)
        rows = []
        for pair_begin in range(0, len(states), PAIR_INFERENCE_BATCH):
            pair_stop = min(pair_begin + PAIR_INFERENCE_BATCH, len(states))
            rows.append(movement_probability(
                model, states[pair_begin:pair_stop], commands[pair_begin:pair_stop]
            ))
        values = np.concatenate(rows).reshape(stop - begin, -1)
        asked = np.any(
            commands.reshape(stop - begin, -1, 2)
            != states.reshape(stop - begin, -1, STATE_DIM)[..., :2],
            axis=2,
        )
        values = np.where(asked, values, -np.inf)
        labels = values.max(axis=1)
        labels[~np.isfinite(labels)] = 0.0
        mobility._save_npz_atomic(path, mobility=labels.astype(np.float32))
        print(
            f"R48 mobility labels nodes={stop}/{graph.node_count} "
            f"median={np.median(labels):.4f}", flush=True,
        )
    labels = []
    for begin in range(0, graph.node_count, ARTIFACT_NODE_SHARD):
        stop = min(begin + ARTIFACT_NODE_SHARD, graph.node_count)
        with np.load(
            artifact["mobility_shards"] / f"nodes_{begin:06d}_{stop:06d}.npz"
        ) as stored:
            labels.append(stored["mobility"])
    labels = np.concatenate(labels).astype(np.float32)
    mobility._save_npz_atomic(artifact["mobility_labels"], mobility=labels)
    return labels


def command_prepare_mobility(args):
    graph = verify_graph(args.graph)
    model = AntConsequenceAdapter(R30_CHECKPOINT, args.device)
    artifact, _landmark, targets, _distance, metadata = _prepare_reach(
        graph, args
    )
    label_contract = artifact["mobility_shards"] / "contract.json"
    if artifact["mobility_labels"].exists() and not label_contract.exists():
        raise ValueError(
            "R48 mobility labels predate their manifest-bound shard contract"
        )
    if artifact["mobility_labels"].exists():
        saved_contract = json.loads(label_contract.read_text())
        expected_contract = {
            "method_revision": METHOD_REVISION,
            **manifest_contract(),
            "stage": "analytic_sigma_stall_mobility_labels",
            "base_graph_sha256": sha256(args.graph),
            "factorized_consequence_sha256": sha256(R30_CHECKPOINT),
            "landmarks_sha256": sha256(artifact["landmarks"]),
            "stall_probability": METHOD_MANIFEST["categories"]["data_derived"][
                "stall_probability"
            ],
            "candidate_budget_per_family": CANDIDATE_BUDGET,
        }
        if saved_contract != expected_contract:
            raise ValueError("R48 mobility labels use a stale manifest contract")
        with np.load(artifact["mobility_labels"]) as stored:
            labels = stored["mobility"]
    else:
        labels = _prepare_mobility_labels(graph, targets, model, args, artifact)
    if artifact["mobility"].exists() and artifact["mobility_meta"].exists():
        saved = json.loads(artifact["mobility_meta"].read_text())
        for key, expected in manifest_contract().items():
            if saved.get(key) != expected:
                raise ValueError(f"R48 mobility artifact has another {key}")
        if saved.get("artifact_sha256") != sha256(artifact["mobility"]):
            raise ValueError("R48 mobility artifact hash changed")
    _model, report = mobility._train_mobility(
        args, graph, labels, artifact, metadata
    )
    report.update({
        **manifest_contract(),
        "stall_probability": METHOD_MANIFEST["categories"]["data_derived"][
            "stall_probability"
        ],
        "commands_asking_for_movement": "exact_nonidentity_requests_only",
        "physical_mobility_threshold": None,
        "shared_method_revision": METHOD_MANIFEST["method_revision"],
    })
    dump(artifact["mobility_meta"], report)
    print(json.dumps({
        "labels": int(len(labels)),
        "label_median": float(np.median(labels)),
        "validation_mse": report["validation_mse_best"],
    }, indent=2))


def verify_mobility_artifact(artifact):
    if not artifact["mobility"].exists() or not artifact["mobility_meta"].exists():
        raise FileNotFoundError("run prepare-mobility first")
    metadata = json.loads(artifact["mobility_meta"].read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        **manifest_contract(),
        "artifact_sha256": sha256(artifact["mobility"]),
        "landmark_sha256": sha256(artifact["landmarks"]),
    }
    for key, item in expected.items():
        if metadata.get(key) != item:
            raise ValueError(f"R48 mobility artifact has another {key}")
    return metadata


def exact_member_connection(model, endpoint_mobility, source, command, target_xy):
    values = []
    for member in range(len(model.members)):
        weight, endpoint = model.component_mean_endpoint_outcomes(
            source, command, member
        )
        spatial = np.exp(
            -np.square(endpoint[..., :2] - target_xy[:, None]).sum(axis=2)
            / (2.0 * SIGMA**2)
        )
        movable = endpoint_mobility.predict(
            endpoint.reshape(-1, STATE_DIM)
        ).reshape(endpoint.shape[:2])
        values.append((weight * spatial * movable).sum(axis=1))
    return np.stack(values, axis=1).astype(np.float32)


def command_prepare_bridges(args):
    graph = verify_graph(args.graph)
    model = AntConsequenceAdapter(R30_CHECKPOINT, args.device)
    artifact, landmark, targets, _distance, reach_meta = _prepare_reach(
        graph, args
    )
    verify_mobility_artifact(artifact)
    endpoint_mobility = mobility.MobilityModel(artifact["mobility"], args.device)
    artifact["bridge_shards"].mkdir(parents=True, exist_ok=True)
    contract = {
        "method_revision": METHOD_REVISION,
        **manifest_contract(),
        "base_graph_sha256": sha256(args.graph),
        "factorized_consequence_sha256": sha256(R30_CHECKPOINT),
        "mobility_sha256": sha256(artifact["mobility"]),
        "candidate_budget_per_family": CANDIDATE_BUDGET,
        "integration": "exact_joint_spatial_posture_component_means",
        "landing_sigma_m": SIGMA,
    }
    mobility._bind_shards(artifact["bridge_shards"], contract)
    for begin in range(0, graph.node_count, ARTIFACT_NODE_SHARD):
        stop = min(begin + ARTIFACT_NODE_SHARD, graph.node_count)
        path = artifact["bridge_shards"] / f"nodes_{begin:06d}_{stop:06d}.npz"
        if path.exists() and mobility._valid_npz(
            path, "weight", (stop - begin, targets.shape[1])
        ):
            continue
        target = targets[begin:stop]
        source = np.repeat(
            graph.nodes[begin:stop, None], target.shape[1], axis=1
        ).reshape(-1, STATE_DIM)
        target_xy = graph.nodes[target, :2].reshape(-1, 2)
        rows = []
        for pair_begin in range(0, len(source), PAIR_INFERENCE_BATCH):
            pair_stop = min(pair_begin + PAIR_INFERENCE_BATCH, len(source))
            member = exact_member_connection(
                model,
                endpoint_mobility,
                source[pair_begin:pair_stop],
                target_xy[pair_begin:pair_stop],
                target_xy[pair_begin:pair_stop],
            )
            rows.append(member.min(axis=1))
        weight = np.concatenate(rows).reshape(stop - begin, -1)
        mobility._save_npz_atomic(path, weight=weight.astype(np.float32))
        print(
            f"R48 exact bridges nodes={stop}/{graph.node_count} "
            f"median={np.median(weight):.6g}", flush=True,
        )
    weights = []
    for begin in range(0, graph.node_count, ARTIFACT_NODE_SHARD):
        stop = min(begin + ARTIFACT_NODE_SHARD, graph.node_count)
        with np.load(
            artifact["bridge_shards"] / f"nodes_{begin:06d}_{stop:06d}.npz"
        ) as stored:
            weights.append(stored["weight"])
    weight = np.concatenate(weights).astype(np.float32)
    mobility._save_npz_atomic(
        artifact["bridges"], landmark_index=landmark,
        target_index=targets, weight=weight,
    )
    report = {
        **contract,
        "claim_level": PREPARATION_CLAIM_LEVEL,
        "artifact": str(artifact["bridges"]),
        "artifact_sha256": sha256(artifact["bridges"]),
        "nodes": graph.node_count,
        "model_bridges": int(weight.size),
        "weight_mean": float(weight.mean()),
        "weight_median": float(np.median(weight)),
        "weight_p90": float(np.quantile(weight, 0.9)),
        "endpoint_samples": None,
        "reach_metadata": reach_meta,
        "task_reward_used": False,
        "hidden_goals_used": False,
        "oracle_inputs": [],
    }
    dump(artifact["bridge_meta"], report)
    print(json.dumps(report, indent=2))


def _load_bridges(args):
    graph = verify_graph(args.graph)
    artifact, landmark, targets, _distance, _meta = _prepare_reach(graph, args)
    verify_mobility_artifact(artifact)
    with np.load(artifact["bridges"]) as stored:
        saved_landmark = stored["landmark_index"].astype(np.int32)
        saved_targets = stored["target_index"].astype(np.int32)
        weight = stored["weight"].astype(np.float32)
    if not np.array_equal(landmark, saved_landmark):
        raise ValueError("R48 bridge landmark set changed")
    if not np.array_equal(targets, saved_targets):
        raise ValueError("R48 bridge target set changed")
    report = json.loads(artifact["bridge_meta"].read_text())
    expected = {
        "method_revision": METHOD_REVISION,
        **manifest_contract(),
        "base_graph_sha256": sha256(args.graph),
        "factorized_consequence_sha256": sha256(R30_CHECKPOINT),
        "mobility_sha256": sha256(artifact["mobility"]),
        "candidate_budget_per_family": CANDIDATE_BUDGET,
        "integration": "exact_joint_spatial_posture_component_means",
        "landing_sigma_m": SIGMA,
    }
    for key, item in expected.items():
        if report.get(key) != item:
            raise ValueError(f"R48 bridge artifact has another {key}")
    if report.get("artifact_sha256") != sha256(artifact["bridges"]):
        raise ValueError("R48 bridge hash changed")
    return graph, artifact, landmark, targets, weight


def verify_passage_metadata(args, metadata):
    expected = {
        "method_revision": METHOD_REVISION,
        **manifest_contract(),
        "graph_sha256": sha256(args.graph),
        "r22_bridges_sha256": sha256(artifact_paths()["bridges"]),
        "reach_landmarks_sha256": sha256(artifact_paths()["landmarks"]),
        "integration_sigma_m": SIGMA,
        "passage_bins_computational_only_global": PASSAGE_BINS,
    }
    for key, item in expected.items():
        if metadata.get(key) != item:
            raise ValueError(f"R48 passage artifact has another {key}")
    if metadata.get("recorded_passages_sha256") != sha256(
        passage_paths()["recorded"]
    ):
        raise ValueError("R48 recorded passage artifact hash changed")
    for row in metadata.get("bridge_shards", []):
        path = passage_paths()["bridge_shards"] / row["file"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"R48 passage shard hash changed: {path}")
    return metadata


def command_prepare_passage(args):
    graph, _artifact, _landmark, targets, weight = _load_bridges(args)
    metadata_path = passage_paths()["metadata"]
    existed = metadata_path.exists()
    metadata = passage.prepare_geometry(args.root, graph, targets, weight)
    if existed:
        verify_passage_metadata(args, metadata)
    else:
        metadata.update(manifest_contract())
        dump(metadata_path, metadata)
        verify_passage_metadata(args, metadata)
    print(json.dumps({
        "artifact": str(passage_paths()["metadata"]),
        "method_revision": metadata["method_revision"],
        "passage_bins": metadata["passage_bins_computational_only_global"],
        "active_bridges": metadata["active_bridges"],
    }, indent=2))


def command_prepare_all(args):
    command_prepare_graph(args)
    command_prepare_mobility(args)
    command_prepare_bridges(args)
    command_prepare_passage(args)


def _quantiles(values, probabilities=(0.5, 0.9, 0.95, 0.99, 1.0)):
    values = np.asarray(values, np.float64)
    return {
        f"p{int(round(probability * 100)):02d}": float(
            np.quantile(values, probability)
        )
        for probability in probabilities
    }


def _weighted_quantiles(values, weights, probabilities=(0.5, 0.9, 0.95, 0.99, 1.0)):
    values = np.asarray(values, np.float64)
    weights = np.asarray(weights, np.float64)
    positive = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(positive):
        raise ValueError("passage audit has no positive finite bridge weights")
    values, weights = values[positive], weights[positive]
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    targets = np.asarray(probabilities, np.float64) * cumulative[-1]
    index = np.searchsorted(cumulative, targets, side="left")
    index = np.minimum(index, len(values) - 1)
    return {
        f"p{int(round(probability * 100)):02d}": float(values[item])
        for probability, item in zip(probabilities, index)
    }


def command_audit_passage_matching(args):
    """Audit all prototype matches without task reward or environment access."""
    graph, _artifact, _landmark, targets, weight = _load_bridges(args)
    metadata = verify_passage_metadata(
        args, json.loads(passage_paths()["metadata"].read_text())
    )
    recorded_start = graph.nodes[graph.source_index, :2]
    recorded_displacement = (
        graph.nodes[graph.endpoint_index, :2] - recorded_start
    )
    source_rows, displacement_rows = [], []
    standardized_rows, weight_rows = [], []
    width = int(targets.shape[1])
    for row in metadata["bridge_shards"]:
        path = passage_paths()["bridge_shards"] / row["file"]
        with np.load(path) as stored:
            active = stored["active_index"].astype(np.int64)
            edge = stored["passage_edge_index"].astype(np.int64)
        local_node = active // width
        local_target = active % width
        node = int(row["begin"]) + local_node
        source = graph.nodes[node, :2]
        target = targets[node, local_target]
        desired_displacement = graph.nodes[target, :2] - source
        source_delta = recorded_start[edge] - source
        displacement_delta = recorded_displacement[edge] - desired_displacement
        source_error = np.linalg.norm(source_delta, axis=1)
        displacement_error = np.linalg.norm(displacement_delta, axis=1)
        standardized = np.sqrt(
            np.square(source_error / SIGMA)
            + np.square(displacement_error / SIGMA)
        )
        source_rows.append(source_error.astype(np.float32))
        displacement_rows.append(displacement_error.astype(np.float32))
        standardized_rows.append(standardized.astype(np.float32))
        weight_rows.append(
            weight[int(row["begin"]):int(row["stop"])].reshape(-1)[active]
        )
    source_error = np.concatenate(source_rows)
    displacement_error = np.concatenate(displacement_rows)
    standardized = np.concatenate(standardized_rows)
    bridge_weight = np.concatenate(weight_rows)
    if len(standardized) != int(metadata["active_bridges"]):
        raise ValueError("passage audit did not cover every active bridge")
    expected = {
        "match_standardized_distance_median": np.median(standardized),
        "match_standardized_distance_p90": np.quantile(standardized, 0.9),
        "matched_source_error_median_m": np.median(source_error),
        "matched_source_error_p90_m": np.quantile(source_error, 0.9),
        "matched_displacement_error_median_m": np.median(displacement_error),
        "matched_displacement_error_p90_m": np.quantile(
            displacement_error, 0.9
        ),
    }
    for key, value in expected.items():
        if not np.isclose(metadata[key], value, rtol=1e-5, atol=1e-6):
            raise ValueError(f"passage audit disagrees with preparation: {key}")
    combined_error = np.hypot(source_error, displacement_error)
    report = {
        "schema_version": 1,
        "experiment_id": "antmaze_r48_passage_matching_audit",
        "method_revision": METHOD_REVISION,
        **manifest_contract(),
        "claim_level": "reward_free_passage_matching_diagnostic_only",
        "declared_deviations": [
            "diagnostic_reports_continuous_error_distributions_without_an_acceptance_cutoff",
            "does_not_change_passage_matching_or_planner_behavior",
        ],
        "oracle_inputs": [],
        "task_reward_used": False,
        "hidden_goals_used": False,
        "environment_steps": 0,
        "affected_quantity": (
            "empirical_trajectory_geometry_used_for_acquisition_passage_gain"
        ),
        "unaffected_quantities": [
            "R30_consequence_model_training",
            "recorded_transition_reliability",
            "analytic_sigma_stall_mobility",
            "movement_weighted_reach_bank",
            "exact_mixture_bridge_connection_weights",
        ],
        "landing_sigma_m": SIGMA,
        "active_bridges": int(len(standardized)),
        "recorded_edges": int(graph.edge_count),
        "graph_sha256": sha256(args.graph),
        "bridges_sha256": sha256(artifact_paths()["bridges"]),
        "passage_metadata_sha256": sha256(passage_paths()["metadata"]),
        "unweighted": {
            "joint_match_distance_sigma_units": _quantiles(standardized),
            "source_error_m": _quantiles(source_error),
            "source_error_sigma_units": _quantiles(source_error / SIGMA),
            "displacement_error_m": _quantiles(displacement_error),
            "displacement_error_sigma_units": _quantiles(
                displacement_error / SIGMA
            ),
            "combined_error_m": _quantiles(combined_error),
        },
        "bridge_weighted": {
            "weight_sum": float(bridge_weight.sum(dtype=np.float64)),
            "joint_match_distance_sigma_units": _weighted_quantiles(
                standardized, bridge_weight
            ),
            "source_error_m": _weighted_quantiles(
                source_error, bridge_weight
            ),
            "source_error_sigma_units": _weighted_quantiles(
                source_error / SIGMA, bridge_weight
            ),
            "displacement_error_m": _weighted_quantiles(
                displacement_error, bridge_weight
            ),
            "displacement_error_sigma_units": _weighted_quantiles(
                displacement_error / SIGMA, bridge_weight
            ),
            "combined_error_m": _weighted_quantiles(
                combined_error, bridge_weight
            ),
        },
        "interpretation": (
            "diagnostic_only_no_acceptance_threshold_and_no_planner_change"
        ),
    }
    output = RESULT_DIRECTORY / "diagnostics" / "passage_matching_audit.json"
    dump(output, report)
    print(json.dumps(report, indent=2))


class RuleFreeR39Planner(r31.AlignedBoundedConsumptionPlanner):
    """R39 greedy search with learned-sigma fields and exact mixtures."""

    method_revision = METHOD_REVISION
    planner_name = "r39_rule_removal_with_manifest_inputs"
    horizon = HORIZON

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        displacement = (
            self.graph.nodes[self.graph.endpoint_index, :2]
            - self.graph.nodes[self.graph.source_index, :2]
        ).astype(np.float32)
        offsets = representative_reach_offsets(
            displacement,
            CANDIDATE_BUDGET,
            sample_weight=movement_reach_weight(self.graph),
        )
        self.landmark_tree = EmpiricalReachQuery(
            self.graph.nodes[self.landmark_index, :2], offsets
        )

    def configure_hidden_fields(self, landing_sigma):
        if float(landing_sigma) != SIGMA:
            raise ValueError("R48 must retain learned landing precision")
        self.acquisition_search_mode = "fine"
        self.coarse_search_sigma = SIGMA
        self.coverage_reference_xy = None

    def _exact_connections(self, state, commands, target_index):
        source = np.repeat(
            np.asarray(state, np.float32).reshape(1, STATE_DIM),
            len(commands), axis=0,
        )
        target_xy = self.graph.nodes[target_index, :2]
        return exact_member_connection(
            self.consequence, self.mobility, source,
            np.asarray(commands, np.float32), target_xy,
        )

    def _candidate_value(
        self, state, commands, target_index, remaining, reliability, seed,
        return_members=False,
    ):
        del seed
        member_connection = self._exact_connections(
            state, commands, target_index
        )
        connection = member_connection.min(axis=1)
        reliability = np.asarray(reliability, np.float32).reshape(-1)
        if self.acquisition_active:
            gain = self.passage_geometry.live_segment_gain(
                state[:2], self.graph.nodes[target_index, :2],
                self.current_fields,
            )
            member_q = (
                member_connection * reliability[:, None] * gain[:, None]
            )
            self._last_live_gain = gain
        else:
            continuation = (
                self.node_field[target_index]
                + self.values[remaining - 1, target_index]
            )
            member_q = (
                member_connection
                * reliability[:, None]
                * continuation[:, None]
            )
        q = member_q.min(axis=1)
        if return_members:
            return q, connection, member_q, member_connection
        return q, connection

    def _positive_evidence_value(self, state, commands, fields, remaining, seed):
        del seed
        nearest = self.node_xy_tree.query(commands, k=1, workers=-1)[1]
        continuation = (
            fields.reward_rate(commands)
            + self.values[remaining - 1, nearest]
        )
        source = np.repeat(state[None], len(commands), axis=0)
        member = exact_member_connection(
            self.consequence, self.mobility, source, commands, commands
        )
        return (
            (member * continuation[:, None]).min(axis=1),
            member.min(axis=1),
        )

    def choose(self, state, command_index, fields, *args, **kwargs):
        command, diagnostic = super().choose(
            state, command_index, fields, *args, **kwargs
        )
        diagnostic.update({
            "planner": self.planner_name,
            "method_revision": self.method_revision,
            "planning_horizon_commands": HORIZON,
            "candidate_budget_per_family": CANDIDATE_BUDGET,
            "consequence_integration": (
                "exact_joint_spatial_posture_component_means"
            ),
        })
        if not fields.contacted:
            residual = float(
                self.passage_geometry.residual_reachable_mass_m2 or 0.0
            )
            diagnostic.update({
                "phase": "acquisition",
                "acquisition_search_mode": "learned_sigma_residual_field",
                "acquisition_backup": "adaptive_greedy_live_passage_mass",
                "acquisition_immediate_readout": (
                    "normalized_learned_sigma_residual_passage_mass"
                ),
                "selected_expected_posterior_mass_fraction": float(
                    diagnostic["selected_score"]
                ),
                "posterior_exhausted": bool(residual <= 0.0),
                "posterior_exhaustion_behavior": "report_zero_gain",
            })
        return command, diagnostic


def bayesian_local_connection(prior, support, outcome_mass):
    prior = np.asarray(prior, np.float32)
    support = np.asarray(support, np.float32).reshape(-1)
    outcome_mass = np.asarray(outcome_mass, np.float32).reshape(-1)
    if prior.ndim != 2 or prior.shape[0] != len(support):
        raise ValueError("prior must have one row per candidate")
    return np.clip(
        (
            RECORDED_EVIDENCE_PRIOR_WEIGHT * prior + outcome_mass[:, None]
        )
        / (RECORDED_EVIDENCE_PRIOR_WEIGHT + support[:, None]),
        0.0, 1.0,
    ).astype(np.float32)


class RuleFreeOnlinePlanner(RuleFreeR39Planner):
    planner_name = "r39_r46_rule_removal_with_manifest_inputs"

    def reset_online_correction_task(self):
        self.online_starts = []
        self.online_commands = []
        self.online_outcomes = []
        self.pending_start = None
        self.pending_command = None
        self.pending_online_length = 0
        self._online_candidate_batches = []

    def _ingest_pending(self, online_states):
        new_outcome = None
        if (
            self.pending_start is not None
            and len(online_states) > int(self.pending_online_length)
        ):
            endpoint = np.asarray(online_states[-1], np.float32)
            outcome = float(np.exp(
                -np.square(endpoint[:2] - self.pending_command).sum()
                / (2.0 * SIGMA**2)
            ))
            self.online_starts.append(self.pending_start.copy())
            self.online_commands.append(self.pending_command.copy())
            self.online_outcomes.append(outcome)
            new_outcome = outcome
        self.pending_start = None
        self.pending_command = None
        return new_outcome

    def _online_mass(self, state, commands):
        commands = np.asarray(commands, np.float32).reshape(-1, 2)
        if not self.online_starts:
            zeros = np.zeros(len(commands), np.float32)
            return zeros, zeros
        starts = np.asarray(self.online_starts, np.float32)
        observed = np.asarray(self.online_commands, np.float32)
        outcomes = np.asarray(self.online_outcomes, np.float32)
        state_distance = np.square(
            (starts - np.asarray(state, np.float32)[None])
            / self.graph.state_scale[None]
        ).sum(axis=1, dtype=np.float64)
        command_distance = np.square(
            (observed[:, None] - commands[None]) / SIGMA
        ).sum(axis=2, dtype=np.float64)
        similarity = np.exp(-0.5 * (state_distance[:, None] + command_distance))
        return (
            similarity.sum(axis=0).astype(np.float32),
            (similarity * outcomes[:, None]).sum(axis=0).astype(np.float32),
        )

    def _candidate_value(
        self, state, commands, target_index, remaining, reliability, seed,
        return_members=False,
    ):
        q, connection, member_q, member_connection = super()._candidate_value(
            state, commands, target_index, remaining, reliability, seed,
            return_members=True,
        )
        del q
        support, outcome = self._online_mass(state, commands)
        corrected_connection = bayesian_local_connection(
            member_connection, support, outcome
        )
        continuation = np.divide(
            member_q, member_connection, out=np.zeros_like(member_q),
            where=member_connection > 0.0,
        )
        corrected_q = continuation * corrected_connection
        empirical = np.divide(
            outcome, support, out=np.zeros_like(outcome), where=support > 0.0
        )
        self._online_candidate_batches.append({
            "commands": np.asarray(commands, np.float32).copy(),
            "offline_connection": np.asarray(connection, np.float32).copy(),
            "corrected_connection": corrected_connection.min(axis=1),
            "support": support,
            "empirical": empirical,
        })
        if return_members:
            return (
                corrected_q.min(axis=1),
                corrected_connection.min(axis=1),
                corrected_q,
                corrected_connection,
            )
        return corrected_q.min(axis=1), corrected_connection.min(axis=1)

    def _selected_online_diagnostic(self, command, selected_connection):
        if not self._online_candidate_batches:
            return {}
        commands = np.concatenate([x["commands"] for x in self._online_candidate_batches])
        exact = np.flatnonzero(np.all(commands == command[None], axis=1))
        if not len(exact):
            return {}
        offline = np.concatenate([x["offline_connection"] for x in self._online_candidate_batches])
        corrected = np.concatenate([x["corrected_connection"] for x in self._online_candidate_batches])
        support = np.concatenate([x["support"] for x in self._online_candidate_batches])
        empirical = np.concatenate([x["empirical"] for x in self._online_candidate_batches])
        index = exact[np.argmin(np.abs(corrected[exact] - selected_connection))]
        return {
            "selected_offline_connection_weight": float(offline[index]),
            "selected_online_corrected_connection_weight": float(corrected[index]),
            "selected_online_support_mass": float(support[index]),
            "selected_online_empirical_connection": (
                float(empirical[index]) if support[index] > 0.0 else None
            ),
        }

    def choose(self, state, command_index, fields, online_states=(), online_trajectories=()):
        if not hasattr(self, "online_starts"):
            self.reset_online_correction_task()
        new_outcome = self._ingest_pending(online_states)
        self._online_candidate_batches = []
        command, diagnostic = super().choose(
            state, command_index, fields, online_states, online_trajectories
        )
        diagnostic.update(self._selected_online_diagnostic(
            np.asarray(command, np.float32),
            float(diagnostic["selected_connection_weight"]),
        ))
        diagnostic.update({
            "online_consequence_observations": len(self.online_starts),
            "new_online_outcome_from_previous_command": new_outcome,
            "online_offline_prior_effective_count_global": (
                RECORDED_EVIDENCE_PRIOR_WEIGHT
            ),
        })
        self.pending_start = np.asarray(state, np.float32).copy()
        self.pending_command = np.asarray(command, np.float32).copy()
        self.pending_online_length = len(online_states)
        return command, diagnostic


def _load_runtime(args):
    graph, artifact, landmark, targets, weight = _load_bridges(args)
    consequence = AntConsequenceAdapter(R30_CHECKPOINT, args.device)
    endpoint_mobility = mobility.MobilityModel(artifact["mobility"], args.device)
    planner = RuleFreeOnlinePlanner(
        consequence, endpoint_mobility, graph, landmark, targets, weight,
        np.random.default_rng(args.seed), args.seed,
    )
    geometry = r31.r28.BoundedPassageGeometry(
        args.root, graph, targets, weight
    )
    verify_passage_metadata(args, geometry.metadata)
    planner.configure_hidden_fields(SIGMA)
    planner.configure_empirical_passage(geometry)
    planner.artifact_hashes = {
        "graph_sha256": sha256(args.graph),
        "factorized_consequence_sha256": sha256(R30_CHECKPOINT),
        "mobility_sha256": sha256(artifact["mobility"]),
        "bridges_sha256": sha256(artifact["bridges"]),
        "reach_landmarks_sha256": sha256(artifact["landmarks"]),
        "empirical_passage_geometry_sha256": sha256(passage_paths()["metadata"]),
        "shared_method_manifest_sha256": sha256(MANIFEST_PATH),
    }
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    env = make_env(max_episode_steps=10_000_000)
    return planner, executor, env


def _coverage_cell(xy):
    return tuple(int(value) for value in np.floor(np.asarray(xy, np.float32)))


def _execute_evaluator_command(
    env, executor, state, command, hidden_goal, success_radius
):
    states, rewards = [state.copy()], []
    for _ in range(COMMAND_STEPS):
        observation, _reward, terminated, truncated, _info = env.step(
            executor.action(state, command)
        )
        state = full_state(observation)
        rewards.append(
            np.linalg.norm(state[:2] - hidden_goal) <= float(success_radius)
        )
        states.append(state.copy())
        if terminated or truncated:
            raise RuntimeError("environment ended inside a fixed command")
    return np.asarray(states), np.asarray(rewards, np.uint8)


def evaluate_hidden_task(
    env, executor, planner, goal, reset_seed, episodes, log_path,
    success_radius,
):
    fields = TaskFields(SIGMA, search_mode=planner.acquisition_search_mode)
    online, completed = [], []
    contacts, successes, logs = None, [], []
    first_step, elapsed = None, 0
    visited, endpoints = set(), []
    acquisition_commands = acquisition_new = acquisition_zero = 0
    acquisition_revisited = acquisition_cycles = 0
    for episode in range(int(episodes)):
        observation, _ = env.reset(seed=int(reset_seed))
        state = full_state(observation)
        trajectory_memory = [state.copy()]
        fields.add(state[:2], 0)
        visited.add(_coverage_cell(state[:2]))
        episode_success = False
        for command_index in range(HORIZON):
            fields.rebuild()
            phase = "return" if fields.contacted else "acquisition"
            command, diagnostic = planner.choose(
                state, command_index, fields, online,
                completed + [trajectory_memory],
            )
            trajectory, rewards = _execute_evaluator_command(
                env, executor, state, command, goal, success_radius
            )
            cells = {_coverage_cell(xy) for xy in trajectory[1:, :2]}
            new = cells - visited
            endpoint = _coverage_cell(trajectory[-1, :2])
            revisited = endpoint in visited
            cycle = bool(
                len(endpoints) >= 2
                and endpoint == endpoints[-2]
                and endpoint != endpoints[-1]
            )
            if phase == "acquisition":
                acquisition_commands += 1
                acquisition_new += len(new)
                acquisition_zero += int(not new)
                acquisition_revisited += int(revisited)
                acquisition_cycles += int(cycle)
                endpoints.append(endpoint)
            visited.update(cells)
            for next_state, reward in zip(trajectory[1:], rewards):
                fields.add(next_state[:2], reward)
                online.append(next_state.copy())
                trajectory_memory.append(next_state.copy())
            if first_step is None and rewards.any():
                first_step = elapsed + int(np.flatnonzero(rewards)[0]) + 1
            elapsed += COMMAND_STEPS
            episode_success = bool(rewards.any())
            diagnostic.update({
                "episode": episode,
                "command_index": command_index,
                "command": np.asarray(command).tolist(),
                "actual_endpoint": trajectory[-1, :2].tolist(),
                "moved_distance": float(np.linalg.norm(trajectory[-1, :2] - state[:2])),
                "reward": episode_success,
                "new_1m_cells": len(new),
                "revisited_endpoint_1m_cell": revisited,
                "acquisition_two_cycle_1m_cell": cycle if phase == "acquisition" else None,
            })
            logs.append(diagnostic)
            state = trajectory[-1]
            if episode_success:
                if contacts is None:
                    contacts = episode
                break
        completed.append(np.asarray(trajectory_memory))
        successes.append(episode_success)
    Path(log_path).write_text(
        "\n".join(json.dumps(row) for row in logs) + "\n"
    )
    return {
        "successes": successes,
        "first_contact_episode": contacts,
        "contact_latency": first_step if first_step is not None else int(episodes) * HORIZON * COMMAND_STEPS,
        "mean_e2e": float(np.mean(successes)),
        "conditional_return": (
            float(np.mean(successes[contacts + 1:]))
            if contacts is not None and contacts + 1 < int(episodes) else None
        ),
        "acquisition_commands": acquisition_commands,
        "acquisition_new_1m_cells": acquisition_new,
        "acquisition_new_1m_cells_per_command": (
            acquisition_new / acquisition_commands if acquisition_commands else None
        ),
        "acquisition_zero_new_1m_cell_fraction": (
            acquisition_zero / acquisition_commands if acquisition_commands else None
        ),
        "acquisition_revisited_endpoint_fraction": (
            acquisition_revisited / acquisition_commands if acquisition_commands else None
        ),
        "acquisition_two_cycle_count": acquisition_cycles,
    }


def task_ids(set_name):
    if set_name == "gate2":
        return tuple(int(x) for x in PROTOCOL_VALUE["gate_task_ids"])
    if set_name == "all15":
        return tuple(int(x) for x in PROTOCOL_VALUE["development_task_ids"])
    raise ValueError(set_name)


def evaluation_contract(set_name):
    return PROTOCOL_VALUE["evaluation_sets"][set_name]


def evaluation_directory(set_name):
    return RESULT_DIRECTORY / f"evaluation_{set_name}"


def task_result_path(set_name, task_id):
    return evaluation_directory(set_name) / f"task_{int(task_id):03d}.json"


def _evaluate_task(args, planner, executor, env, task_id):
    task = r31.r22._task(args.root, task_id)
    contract = evaluation_contract(args.set_name)
    output = evaluation_directory(args.set_name)
    output.mkdir(parents=True, exist_ok=True)
    result_path = task_result_path(args.set_name, task_id)
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        expected = {
            "task_id": int(task_id),
            "goal": task["goal"],
            "reset_seed": task["reset_seed"],
            "method_revision": METHOD_REVISION,
            "method_reference_sha256": sha256(REFERENCE),
            "shared_method_manifest_sha256": sha256(MANIFEST_PATH),
            "claim_level": contract["claim_level"],
            "episodes": int(args.episodes),
            **planner.artifact_hashes,
        }
        for key, item in expected.items():
            if saved.get(key) != item:
                raise ValueError(f"existing R48 result has another {key}")
        print(json.dumps(saved))
        return saved
    seed = int(args.seed) + 1_000_003 * int(task_id)
    planner.rng = np.random.default_rng(seed)
    planner.seed = seed
    planner.goal_key = None
    planner.reset_online_correction_task()
    row = evaluate_hidden_task(
        env, executor, planner, np.asarray(task["goal"], np.float32),
        task["reset_seed"], int(args.episodes),
        output / f"task_{int(task_id):03d}.jsonl",
        PROTOCOL_VALUE["evaluator"]["success_radius_m_evaluator_only"],
    )
    row.update(task)
    row.update({
        "evaluation_arm": "R48_R39_plus_R46_rules_only_removal",
        "set_name": args.set_name,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "shared_method_revision": METHOD_MANIFEST["method_revision"],
        "claim_level": contract["claim_level"],
        "declared_deviations": contract["declared_deviations"],
        "oracle_inputs": [],
        "planner_inputs": PROTOCOL_VALUE["planner_inputs"],
        "episodes": int(args.episodes),
        "horizon_formula": "episode_primitive_steps/command_steps",
        "episode_primitive_steps": PROTOCOL_VALUE["evaluator"]["episode_primitive_steps"],
        "command_steps": COMMAND_STEPS,
        "horizon_commands": HORIZON,
        "success_radius_evaluator_only_m": PROTOCOL_VALUE["evaluator"]["success_radius_m_evaluator_only"],
        "success_radius_read_by_planner": False,
        "landing_sigma_m": SIGMA,
        "candidate_budget_per_family": CANDIDATE_BUDGET,
        "online_prior_weight": RECORDED_EVIDENCE_PRIOR_WEIGHT,
        **planner.artifact_hashes,
    })
    dump(result_path, row)
    print(json.dumps(row), flush=True)
    return row


def command_evaluate_set(args):
    if int(args.episodes) != int(PROTOCOL_VALUE["episodes_per_task"]):
        raise ValueError("R48 evaluation requires the declared episode count")
    planner, executor, env = _load_runtime(args)
    try:
        for task_id in task_ids(args.set_name):
            _evaluate_task(args, planner, executor, env, task_id)
    finally:
        env.close()
    command_summarize(args)


def command_summarize(args):
    contract = evaluation_contract(args.set_name)
    selected = task_ids(args.set_name)
    rows = [
        json.loads(task_result_path(args.set_name, task_id).read_text())
        for task_id in selected
    ]
    for row, task_id in zip(rows, selected):
        expected = {
            "task_id": int(task_id),
            "method_revision": METHOD_REVISION,
            "method_reference_sha256": sha256(REFERENCE),
            "shared_method_manifest_sha256": sha256(MANIFEST_PATH),
            "claim_level": contract["claim_level"],
            "episodes": int(args.episodes),
        }
        for key, item in expected.items():
            if row.get(key) != item:
                raise ValueError(f"R48 summary row has another {key}")
    contacts = [row["first_contact_episode"] for row in rows]
    successes = [value for row in rows for value in row["successes"]]
    returns = []
    for row in rows:
        contact = row["first_contact_episode"]
        if contact is not None:
            returns.extend(row["successes"][int(contact) + 1:])
    result = {
        "schema_version": 1,
        "experiment_id": PROTOCOL_VALUE["experiment_id"],
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "method_revision": METHOD_REVISION,
        "shared_method_revision": METHOD_MANIFEST["method_revision"],
        "shared_method_manifest_sha256": sha256(MANIFEST_PATH),
        "claim_level": contract["claim_level"],
        "declared_deviations": contract["declared_deviations"],
        "oracle_inputs": [],
        "set_name": args.set_name,
        "tasks": len(rows),
        "task_ids": list(selected),
        "episodes_per_task": int(args.episodes),
        "acquired_tasks": int(sum(x is not None for x in contacts)),
        "acquisition_at_1": float(np.mean([x == 0 for x in contacts])),
        "acquisition_at_5": float(np.mean([x is not None and x < 5 for x in contacts])),
        "pooled_conditional_return": float(np.mean(returns)) if returns else None,
        "mean_e2e": float(np.mean(successes)),
        "horizon_commands": HORIZON,
        "candidate_budget_per_family": CANDIDATE_BUDGET,
        "success_radius_read_by_planner": False,
        "acquisition_new_1m_cells_per_command": float(np.mean([
            row["acquisition_new_1m_cells_per_command"] for row in rows
        ])),
        "acquisition_zero_new_1m_cell_fraction": float(np.mean([
            row["acquisition_zero_new_1m_cell_fraction"] for row in rows
        ])),
        "acquisition_revisited_endpoint_fraction": float(np.mean([
            row["acquisition_revisited_endpoint_fraction"] for row in rows
        ])),
        "acquisition_two_cycle_count": int(sum(
            row["acquisition_two_cycle_count"] for row in rows
        )),
        "rows": rows,
    }
    dump(evaluation_directory(args.set_name) / "summary.json", result)
    print(json.dumps({
        key: result[key] for key in (
            "tasks", "acquired_tasks", "acquisition_at_1", "acquisition_at_5",
            "pooled_conditional_return", "mean_e2e",
        )
    }, indent=2))


def command_audit(_args):
    source = Path(__file__).read_text()
    forbidden = (
        "CONTACT" + "_RADIUS",
        "DETECTION" + "_FOOTPRINT",
        "STALL" + "_RADIUS",
        "ENDPOINT" + "_SAMPLES",
    )
    violations = [token for token in forbidden if token in source]
    report = {
        "method_revision": METHOD_REVISION,
        "method_reference": str(REFERENCE),
        "method_reference_sha256": sha256(REFERENCE),
        "shared_method_manifest": str(MANIFEST_PATH),
        "shared_method_manifest_sha256": sha256(MANIFEST_PATH),
        "shared_method_revision": METHOD_MANIFEST["method_revision"],
        "conformance_audit": PROTOCOL_VALUE["conformance_audit"],
        "active_source_forbidden_tokens": violations,
        "conforms": not violations and all(
            row["status"] == "pass"
            for row in PROTOCOL_VALUE["conformance_audit"]
        ),
    }
    if not report["conforms"]:
        raise ValueError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    value.add_argument("--graph", type=Path)
    value.add_argument("--output", type=Path, default=DEFAULT_RESULT_DIRECTORY)
    value.add_argument("--device", default="cuda")
    value.add_argument("--seed", type=int, default=607)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("audit").set_defaults(func=command_audit)
    commands.add_parser("prepare-graph").set_defaults(func=command_prepare_graph)
    commands.add_parser("prepare-mobility").set_defaults(func=command_prepare_mobility)
    commands.add_parser("prepare-bridges").set_defaults(func=command_prepare_bridges)
    commands.add_parser("prepare-passage").set_defaults(func=command_prepare_passage)
    commands.add_parser("prepare-all").set_defaults(func=command_prepare_all)
    commands.add_parser("audit-passage-matching").set_defaults(
        func=command_audit_passage_matching
    )
    evaluate = commands.add_parser("evaluate-gate2")
    evaluate.add_argument(
        "--episodes", type=int, default=int(PROTOCOL_VALUE["episodes_per_task"])
    )
    evaluate.set_defaults(func=command_evaluate_set, set_name="gate2")
    evaluate_set = commands.add_parser("evaluate-set")
    evaluate_set.add_argument("--set-name", choices=("gate2", "all15"), required=True)
    evaluate_set.add_argument(
        "--episodes", type=int, default=int(PROTOCOL_VALUE["episodes_per_task"])
    )
    evaluate_set.set_defaults(func=command_evaluate_set)
    summarize = commands.add_parser("summarize")
    summarize.add_argument(
        "--set-name", choices=("gate2", "all15"), default="gate2"
    )
    summarize.add_argument(
        "--episodes", type=int, default=int(PROTOCOL_VALUE["episodes_per_task"])
    )
    summarize.set_defaults(func=command_summarize)
    return value


def main():
    args = parser().parse_args()
    configure_result_directory(args.output)
    if args.graph is None:
        args.graph = graph_path(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
