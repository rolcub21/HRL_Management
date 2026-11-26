"""SoRB-style stitching with factorized consequences and endpoint mobility.

Every R6 node and recorded edge is preserved.  A sigma-net subset of the
recorded nodes supplies only candidate command destinations.  Model bridges
connect every recorded node to its global-k nearest landmarks.  Their weight
is the expected spatial landing kernel times endpoint mobility.

Endpoint mobility is the best predicted probability of movement among the
same landmark commands.  It is computed from the verified spatial consequence
head on reward-free graph nodes, then amortized as a scalar function of full
state so predicted endpoint postures can be scored without a strict posture
match to one recorded representative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr
from scipy.spatial import cKDTree

from scripts.antmaze.current_tfsn.collect import evaluate_supplied_goal_task
from scripts.antmaze.current_tfsn.constants import (
    BATCH_SIZE,
    CANDIDATES,
    ENDPOINT_SAMPLES,
    EPISODE_STEPS,
    EPOCHS,
    HIDDEN_DIM,
    LANDING_SIGMA,
    LEARNING_RATE,
    STATE_DIM,
)
from scripts.antmaze.current_tfsn.executor import FaramaSACExecutor, make_env
from scripts.antmaze.current_tfsn.main import dump, panel, paths

from projects.tfsn_consequence_value_iteration.antmaze.factorized_consequence import (
    FactorizedConsequenceEnsemble,
    factorized_checkpoint,
)
from projects.tfsn_consequence_value_iteration.antmaze.oracle_contract_diagnostic import (
    STALL_RADIUS_M,
)
from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    RecordedTransitionGraph,
    consequence_inputs,
    sha256_file,
)


METHOD_REVISION = "antmaze_mobility_stitched_value_iteration_r10"
CLAIM_LEVEL = "consumed_development_supplied_goal_diagnostic"
LANDMARK_NEIGHBORS = CANDIDATES
NODE_SHARD_SIZE = 128
PAIR_INFERENCE_BATCH = 2048
MOBILITY_VALIDATION_FRACTION = 0.10
STALL_QUADRATURE_POINTS = 32
_STALL_QUADRATURE_X, _STALL_QUADRATURE_WEIGHT = (
    np.polynomial.legendre.leggauss(STALL_QUADRATURE_POINTS)
)


def _artifact_paths(root):
    directory = paths(root)["root"] / "model_planning" / "mobility_stitch_r10"
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


def _default_graph(root):
    return (
        Path(root)
        / "model_planning"
        / "round2_recorded_transition_graph_r6.npz"
    )


def _hash_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _bind_shards(directory, contract):
    """Bind resumable shards to the exact immutable inputs that created them."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "contract.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != contract:
            raise ValueError(
                f"Existing resumable shards use a different contract: {path}"
            )
    else:
        dump(path, contract)


def _save_npz_atomic(path, **arrays):
    """Write a generated NPZ so an interrupted run cannot leave a valid name."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _valid_npz(path, key, shape):
    try:
        with np.load(path) as data:
            return key in data and data[key].shape == tuple(shape)
    except (OSError, ValueError, EOFError):
        return False


def sigma_landmarks(xy, sigma):
    """Deterministic greedy sigma-net over exact support coordinates."""
    xy = np.asarray(xy, np.float32).reshape(-1, 2)
    sigma = float(sigma)
    cells = {}
    selected = []
    for index, point in enumerate(xy):
        cell = tuple(np.floor(point / sigma).astype(np.int64))
        keep = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in cells.get((cell[0] + dx, cell[1] + dy), ()):
                    if np.linalg.norm(point - xy[other]) < sigma:
                        keep = False
                        break
                if not keep:
                    break
            if not keep:
                break
        if keep:
            selected.append(index)
            cells.setdefault(cell, []).append(index)
    return np.asarray(selected, np.int32)


def landmark_neighbors(nodes, landmark_index, count=LANDMARK_NEIGHBORS):
    nodes = np.asarray(nodes, np.float32)
    landmark_index = np.asarray(landmark_index, np.int32)
    count = min(int(count), len(landmark_index))
    tree = cKDTree(nodes[landmark_index, :2])
    distance, local = tree.query(nodes[:, :2], k=count, workers=-1)
    if count == 1:
        distance = distance[:, None]
        local = local[:, None]
    return landmark_index[local].astype(np.int32), np.asarray(
        distance, np.float32
    )


class MobilityNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(STATE_DIM, HIDDEN_DIM),
            torch.nn.SiLU(),
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            torch.nn.SiLU(),
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            torch.nn.SiLU(),
            torch.nn.Linear(HIDDEN_DIM, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, states):
        return self.network(states).squeeze(-1)


class MobilityModel:
    def __init__(self, checkpoint, device):
        payload = torch.load(
            Path(checkpoint), map_location="cpu", weights_only=False
        )
        if payload.get("artifact_type") != "endpoint_mobility":
            raise ValueError("not an endpoint mobility checkpoint")
        self.device = torch.device(device)
        self.mean = np.asarray(payload["state_mean"], np.float32)
        self.std = np.asarray(payload["state_std"], np.float32)
        self.network = MobilityNetwork().to(self.device)
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()
        self.payload = payload

    def predict(self, states, batch_size=65_536):
        states = np.asarray(states, np.float32).reshape(-1, STATE_DIM)
        result = []
        with torch.no_grad():
            for begin in range(0, len(states), batch_size):
                value = (states[begin:begin + batch_size] - self.mean) / self.std
                tensor = torch.as_tensor(
                    value, dtype=torch.float32, device=self.device
                )
                result.append(self.network(tensor).cpu().numpy())
        return np.concatenate(result).astype(np.float32)


def _verify_base(graph, consequence):
    if graph.metadata.get("transition_model_inside_graph") is not False:
        raise ValueError("base graph contains learned transitions")
    if graph.metadata.get("predicted_endpoint_projection_inside_graph") is not False:
        raise ValueError("base graph contains endpoint projection")
    scale = consequence.input_std[:STATE_DIM].copy()
    scale[:2] = LANDING_SIGMA
    if not np.array_equal(scale, graph.state_scale):
        raise ValueError("factorized model and base graph use different scales")


def _load_base(args):
    graph = RecordedTransitionGraph.load(args.graph)
    consequence_path = factorized_checkpoint(args.root)
    consequence = FactorizedConsequenceEnsemble(consequence_path, args.device)
    _verify_base(graph, consequence)
    return graph, consequence, consequence_path


def _prepare_landmarks(args, graph, consequence_path):
    artifact = _artifact_paths(args.root)
    artifact["directory"].mkdir(parents=True, exist_ok=True)
    graph_hash = sha256_file(args.graph)
    consequence_hash = sha256_file(consequence_path)
    if artifact["landmarks"].exists() and artifact["landmark_meta"].exists():
        metadata = json.loads(artifact["landmark_meta"].read_text())
        if metadata["base_graph_sha256"] != graph_hash:
            raise ValueError("saved landmarks refer to a different graph")
        if metadata["factorized_consequence_sha256"] != consequence_hash:
            raise ValueError("saved landmarks refer to a different consequence model")
        with np.load(artifact["landmarks"]) as data:
            landmark_index = data["landmark_index"]
            targets = data["target_index"]
            distance = data["target_distance"]
        return artifact, landmark_index, targets, distance, metadata

    landmark_index = sigma_landmarks(graph.nodes[:, :2], LANDING_SIGMA)
    targets, distance = landmark_neighbors(
        graph.nodes, landmark_index, LANDMARK_NEIGHBORS
    )
    tree = cKDTree(graph.nodes[landmark_index, :2])
    cover_distance = tree.query(graph.nodes[:, :2], k=1, workers=-1)[0]
    _save_npz_atomic(
        artifact["landmarks"],
        landmark_index=landmark_index,
        target_index=targets,
        target_distance=distance,
    )
    metadata = {
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "oracle_inputs": [],
        "landmark_rule": "deterministic_greedy_sigma_net",
        "landmark_sigma_m": float(LANDING_SIGMA),
        "landmarks_are_exact_recorded_nodes": True,
        "recorded_nodes_deleted": False,
        "recorded_edges_deleted": False,
        "global_landmark_neighbors": int(targets.shape[1]),
        "nodes": int(graph.node_count),
        "landmarks": int(len(landmark_index)),
        "maximum_support_to_landmark_distance_m": float(cover_distance.max()),
        "median_neighbor_radius_m": float(np.median(distance[:, -1])),
        "p90_neighbor_radius_m": float(np.quantile(distance[:, -1], 0.9)),
        "base_graph": str(Path(args.graph).resolve()),
        "base_graph_sha256": graph_hash,
        "factorized_consequence": str(consequence_path.resolve()),
        "factorized_consequence_sha256": consequence_hash,
    }
    dump(artifact["landmark_meta"], metadata)
    return artifact, landmark_index, targets, distance, metadata


def _diagonal_gaussian_circle_probability(means, stds, centers, radius):
    """Deterministic quadrature for mass inside a 2-D circle."""
    means = np.asarray(means, np.float64)
    stds = np.asarray(stds, np.float64)
    centers = np.asarray(centers, np.float64)
    radius = float(radius)
    quadrature_x = _STALL_QUADRATURE_X[None, None]
    absolute_x = centers[:, None, 0, None] + radius * quadrature_x
    half_height = radius * np.sqrt(1.0 - np.square(quadrature_x))
    standardized_x = (
        absolute_x - means[..., 0, None]
    ) / stds[..., 0, None]
    density_x = (
        np.exp(-0.5 * np.square(standardized_x))
        / (np.sqrt(2.0 * np.pi) * stds[..., 0, None])
    )
    upper = (
        centers[:, None, 1, None]
        + half_height
        - means[..., 1, None]
    ) / stds[..., 1, None]
    lower = (
        centers[:, None, 1, None]
        - half_height
        - means[..., 1, None]
    ) / stds[..., 1, None]
    probability = radius * np.sum(
        _STALL_QUADRATURE_WEIGHT[None, None]
        * density_x
        * (ndtr(upper) - ndtr(lower)),
        axis=2,
    )
    return np.clip(probability, 0.0, 1.0).astype(np.float32)


def _movement_probability(consequence, states, commands):
    """Pessimistic ensemble probability that an endpoint moves >= 0.2 m."""
    states = np.asarray(states, np.float32).reshape(-1, STATE_DIM)
    commands = np.asarray(commands, np.float32).reshape(-1, 2)
    inputs = consequence_inputs(states, commands)
    member = []
    for member_index in range(len(consequence.members)):
        weights, residual, stds = consequence.spatial_parameters(
            inputs, member_index
        )
        means = commands[:, None] + residual
        stalled = (
            weights
            * _diagonal_gaussian_circle_probability(
                means, stds, states[:, :2], STALL_RADIUS_M
            )
        ).sum(axis=1)
        member.append(1.0 - stalled)
    return np.stack(member, axis=1).min(axis=1).astype(np.float32)


def _mobility_labels(
    args, graph, consequence, consequence_path, targets, artifact
):
    _bind_shards(artifact["mobility_shards"], {
        "method_revision": METHOD_REVISION,
        "stage": "endpoint_mobility_labels",
        "base_graph_sha256": sha256_file(args.graph),
        "factorized_consequence_sha256": sha256_file(consequence_path),
        "landmarks_sha256": sha256_file(artifact["landmarks"]),
        "global_landmark_neighbors": int(targets.shape[1]),
        "stall_probability_estimator": "deterministic_gauss_legendre_quadrature",
        "stall_quadrature_points": int(STALL_QUADRATURE_POINTS),
        "movement_radius_m": float(STALL_RADIUS_M),
        "seed": int(args.seed),
    })
    for begin in range(0, graph.node_count, NODE_SHARD_SIZE):
        stop = min(begin + NODE_SHARD_SIZE, graph.node_count)
        shard = artifact["mobility_shards"] / f"nodes_{begin:06d}_{stop:06d}.npz"
        if shard.exists() and _valid_npz(
            shard, "mobility", (stop - begin,)
        ):
            continue
        target = targets[begin:stop]
        state = np.repeat(
            graph.nodes[begin:stop, None], target.shape[1], axis=1
        ).reshape(-1, STATE_DIM)
        command = graph.nodes[target, :2].reshape(-1, 2)
        values = []
        for pair_begin in range(0, len(state), PAIR_INFERENCE_BATCH):
            pair_stop = min(pair_begin + PAIR_INFERENCE_BATCH, len(state))
            values.append(_movement_probability(
                consequence,
                state[pair_begin:pair_stop],
                command[pair_begin:pair_stop],
            ))
        movement = np.concatenate(values).reshape(stop - begin, -1)
        mobility = movement.max(axis=1).astype(np.float32)
        _save_npz_atomic(shard, mobility=mobility)
        print(
            f"mobility labels nodes={stop}/{graph.node_count} "
            f"median={np.median(mobility):.4f}",
            flush=True,
        )
    labels = []
    for begin in range(0, graph.node_count, NODE_SHARD_SIZE):
        stop = min(begin + NODE_SHARD_SIZE, graph.node_count)
        shard = artifact["mobility_shards"] / f"nodes_{begin:06d}_{stop:06d}.npz"
        with np.load(shard) as data:
            labels.append(data["mobility"])
    mobility = np.concatenate(labels).astype(np.float32)
    _save_npz_atomic(artifact["mobility_labels"], mobility=mobility)
    return mobility


def _train_mobility(args, graph, labels, artifact, landmark_meta):
    if artifact["mobility"].exists() and artifact["mobility_meta"].exists():
        metadata = json.loads(artifact["mobility_meta"].read_text())
        if metadata["labels_sha256"] != _hash_arrays(labels):
            raise ValueError("saved mobility checkpoint uses different labels")
        return MobilityModel(artifact["mobility"], args.device), metadata

    rng = np.random.default_rng(args.seed + 10)
    permutation = rng.permutation(graph.node_count)
    validation_count = max(
        1, int(round(MOBILITY_VALIDATION_FRACTION * graph.node_count))
    )
    validation = permutation[:validation_count]
    training = permutation[validation_count:]
    state_mean = graph.nodes[training].mean(axis=0).astype(np.float32)
    state_std = np.maximum(
        graph.nodes[training].std(axis=0), 1e-6
    ).astype(np.float32)
    x = torch.as_tensor(
        (graph.nodes - state_mean) / state_std, dtype=torch.float32
    )
    y = torch.as_tensor(labels, dtype=torch.float32)
    torch.manual_seed(args.seed + 11)
    network = MobilityNetwork().to(args.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(EPOCHS):
        network.train()
        order = rng.permutation(training)
        losses = []
        for begin in range(0, len(order), BATCH_SIZE):
            index = order[begin:begin + BATCH_SIZE]
            prediction = network(x[index].to(args.device))
            loss = torch.mean(torch.square(prediction - y[index].to(args.device)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        network.eval()
        with torch.no_grad():
            prediction = network(x[validation].to(args.device)).cpu()
        validation_mse = float(torch.mean(
            torch.square(prediction - y[validation])
        ))
        validation_mae = float(torch.mean(
            torch.abs(prediction - y[validation])
        ))
        history.append({
            "epoch": epoch + 1,
            "train_mse": float(np.mean(losses)),
            "validation_mse": validation_mse,
            "validation_mae": validation_mae,
        })
        if validation_mse < best_loss:
            best_loss = validation_mse
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in network.state_dict().items()
            }
        if (epoch + 1) % 25 == 0 or epoch + 1 == EPOCHS:
            print(
                f"mobility epoch={epoch + 1}/{EPOCHS} "
                f"validation_mae={validation_mae:.6f}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("mobility training produced no checkpoint")
    payload = {
        "artifact_type": "endpoint_mobility",
        "method_revision": METHOD_REVISION,
        "definition": "one_minus_minimum_predicted_stall_over_bridge_candidates",
        "state_mean": state_mean,
        "state_std": state_std,
        "state_dict": best_state,
        "labels_sha256": _hash_arrays(labels),
        "training_nodes": int(len(training)),
        "validation_nodes": int(len(validation)),
        "seed": int(args.seed),
    }
    torch.save(payload, artifact["mobility"])
    model = MobilityModel(artifact["mobility"], args.device)
    prediction = model.predict(graph.nodes)
    metadata = {
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "oracle_inputs": [],
        "artifact": str(artifact["mobility"].resolve()),
        "artifact_sha256": sha256_file(artifact["mobility"]),
        "definition": payload["definition"],
        "labels_sha256": payload["labels_sha256"],
        "label_minimum": float(labels.min()),
        "label_median": float(np.median(labels)),
        "label_maximum": float(labels.max()),
        "validation_mse_best": best_loss,
        "all_node_mae": float(np.mean(np.abs(prediction - labels))),
        "training_nodes": payload["training_nodes"],
        "validation_nodes": payload["validation_nodes"],
        "landmark_sha256": sha256_file(artifact["landmarks"]),
        "global_landmark_neighbors": landmark_meta[
            "global_landmark_neighbors"
        ],
        "history": history,
    }
    dump(artifact["mobility_meta"], metadata)
    return model, metadata


def command_prepare_mobility(args):
    graph, consequence, consequence_path = _load_base(args)
    artifact, _landmarks, targets, _distance, landmark_meta = (
        _prepare_landmarks(args, graph, consequence_path)
    )
    if artifact["mobility_labels"].exists():
        with np.load(artifact["mobility_labels"]) as data:
            labels = data["mobility"]
    else:
        labels = _mobility_labels(
            args,
            graph,
            consequence,
            consequence_path,
            targets,
            artifact,
        )
    _model, metadata = _train_mobility(
        args, graph, labels, artifact, landmark_meta
    )
    print(json.dumps({
        "landmarks": landmark_meta["landmarks"],
        "mobility_labels": int(len(labels)),
        "mobility_label_median": float(np.median(labels)),
        "mobility_validation_mse": metadata["validation_mse_best"],
    }, indent=2))


def _bridge_weight_chunk(
    consequence,
    mobility,
    source,
    target_state,
    command,
    seed,
):
    inputs = consequence_inputs(source, command)
    samples = ENDPOINT_SAMPLES // len(consequence.members)
    member_values = []
    for member in range(len(consequence.members)):
        rng = np.random.default_rng(seed + 1_000_003 * member)
        delta = consequence.sample_member(inputs, member, samples, rng)
        endpoint = source[:, None] + delta
        spatial = np.exp(
            -np.square(
                endpoint[..., :2] - target_state[:, None, :2]
            ).sum(axis=2)
            / (2.0 * LANDING_SIGMA**2)
        )
        endpoint_mobility = mobility.predict(
            endpoint.reshape(-1, STATE_DIM)
        ).reshape(len(source), samples)
        member_values.append((spatial * endpoint_mobility).mean(axis=1))
    return np.stack(member_values, axis=1).min(axis=1).astype(np.float32)


def command_prepare_bridges(args):
    graph, consequence, consequence_path = _load_base(args)
    artifact, landmark_index, targets, _distance, landmark_meta = (
        _prepare_landmarks(args, graph, consequence_path)
    )
    if not artifact["mobility"].exists():
        raise FileNotFoundError("run prepare-stitched-mobility first")
    mobility = MobilityModel(artifact["mobility"], args.device)
    _bind_shards(artifact["bridge_shards"], {
        "method_revision": METHOD_REVISION,
        "stage": "mobility_weighted_model_bridges",
        "base_graph_sha256": sha256_file(args.graph),
        "factorized_consequence_sha256": sha256_file(consequence_path),
        "mobility_sha256": sha256_file(artifact["mobility"]),
        "landmarks_sha256": sha256_file(artifact["landmarks"]),
        "global_landmark_neighbors": int(targets.shape[1]),
        "endpoint_samples": int(ENDPOINT_SAMPLES),
        "landing_sigma_m": float(LANDING_SIGMA),
        "seed": int(args.seed),
    })
    for begin in range(0, graph.node_count, NODE_SHARD_SIZE):
        stop = min(begin + NODE_SHARD_SIZE, graph.node_count)
        shard = artifact["bridge_shards"] / f"nodes_{begin:06d}_{stop:06d}.npz"
        if shard.exists() and _valid_npz(
            shard, "weight", (stop - begin, targets.shape[1])
        ):
            continue
        target = targets[begin:stop]
        source = np.repeat(
            graph.nodes[begin:stop, None], target.shape[1], axis=1
        ).reshape(-1, STATE_DIM)
        target_state = graph.nodes[target].reshape(-1, STATE_DIM)
        command = target_state[:, :2]
        values = []
        for pair_begin in range(0, len(source), PAIR_INFERENCE_BATCH):
            pair_stop = min(pair_begin + PAIR_INFERENCE_BATCH, len(source))
            values.append(_bridge_weight_chunk(
                consequence,
                mobility,
                source[pair_begin:pair_stop],
                target_state[pair_begin:pair_stop],
                command[pair_begin:pair_stop],
                args.seed + begin * 1009 + pair_begin,
            ))
        weight = np.concatenate(values).reshape(stop - begin, -1)
        _save_npz_atomic(shard, weight=weight.astype(np.float32))
        print(
            f"bridge weights nodes={stop}/{graph.node_count} "
            f"median={np.median(weight):.6g}",
            flush=True,
        )
    weights = []
    for begin in range(0, graph.node_count, NODE_SHARD_SIZE):
        stop = min(begin + NODE_SHARD_SIZE, graph.node_count)
        shard = artifact["bridge_shards"] / f"nodes_{begin:06d}_{stop:06d}.npz"
        with np.load(shard) as data:
            weights.append(data["weight"])
    weight = np.concatenate(weights).astype(np.float32)
    _save_npz_atomic(
        artifact["bridges"],
        landmark_index=landmark_index,
        target_index=targets,
        weight=weight,
    )
    metadata = {
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "hidden_goals_accessed": False,
        "task_reward_used": False,
        "oracle_inputs": [],
        "artifact": str(artifact["bridges"].resolve()),
        "artifact_sha256": sha256_file(artifact["bridges"]),
        "base_graph_sha256": sha256_file(args.graph),
        "factorized_consequence_sha256": sha256_file(consequence_path),
        "mobility_sha256": sha256_file(artifact["mobility"]),
        "nodes": int(graph.node_count),
        "recorded_edges_preserved": int(graph.edge_count),
        "model_bridges": int(weight.size),
        "landmarks": int(len(landmark_index)),
        "global_landmark_neighbors": int(targets.shape[1]),
        "bridge_weight": (
            "pessimistic_ensemble_expectation_of_spatial_landing_kernel_"
            "times_predicted_endpoint_mobility"
        ),
        "weight_mean": float(weight.mean()),
        "weight_median": float(np.median(weight)),
        "weight_p90": float(np.quantile(weight, 0.9)),
        "positive_weight_fraction": float(np.mean(weight > 0)),
        "landmark_metadata": landmark_meta,
    }
    dump(artifact["bridge_meta"], metadata)
    print(json.dumps(metadata, indent=2))


def _verify_bridges(
    args, graph, consequence_path, artifact, landmark_index, targets, weight
):
    if not artifact["bridge_meta"].exists():
        raise FileNotFoundError("bridge metadata is missing")
    metadata = json.loads(artifact["bridge_meta"].read_text())
    expected = {
        "base_graph_sha256": sha256_file(args.graph),
        "factorized_consequence_sha256": sha256_file(consequence_path),
        "mobility_sha256": sha256_file(artifact["mobility"]),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"bridge artifact has a different {key}")
    if metadata.get("artifact_sha256") != sha256_file(artifact["bridges"]):
        raise ValueError("bridge artifact hash does not match its metadata")
    if weight.shape != (graph.node_count, targets.shape[1]):
        raise ValueError("bridge weight matrix has the wrong shape")
    if not np.isfinite(weight).all() or np.any(weight < 0) or np.any(weight > 1):
        raise ValueError("bridge weights must be finite probabilities")
    if len(landmark_index) != metadata.get("landmarks"):
        raise ValueError("bridge landmark count changed")


def stitched_values(graph, targets, weight, goal, horizon=10):
    goal = np.asarray(goal, np.float32)
    node_field = np.exp(
        -np.square(graph.nodes[:, :2] - goal[None]).sum(axis=1)
        / (2.0 * LANDING_SIGMA**2)
    ).astype(np.float32)
    values = np.zeros((horizon + 1, graph.node_count), np.float32)
    coverage = []
    for remaining in range(1, horizon + 1):
        recorded_q = graph.edge_reliability * (
            node_field[graph.endpoint_index]
            + values[remaining - 1, graph.endpoint_index]
        )
        row = np.zeros(graph.node_count, np.float32)
        np.maximum.at(row, graph.source_index, recorded_q)
        bridge_q = weight * (
            node_field[targets] + values[remaining - 1, targets]
        )
        row = np.maximum(row, bridge_q.max(axis=1))
        values[remaining] = row
        coverage.append({
            "remaining": remaining,
            "positive_nodes": int(np.sum(row > 0)),
            "positive_fraction": float(np.mean(row > 0)),
            "meaningful_nodes": int(np.sum(row > 1e-6)),
            "meaningful_fraction": float(np.mean(row > 1e-6)),
            "maximum": float(row.max()),
        })
    return values, coverage, node_field


class MobilityStitchedPlanner:
    def __init__(
        self,
        consequence,
        mobility,
        graph,
        landmark_index,
        targets,
        weight,
        rng,
        seed,
    ):
        self.consequence = consequence
        self.mobility = mobility
        self.graph = graph
        self.landmark_index = np.asarray(landmark_index, np.int32)
        self.targets = np.asarray(targets, np.int32)
        self.weight = np.asarray(weight, np.float32)
        self.landmark_tree = cKDTree(
            graph.nodes[self.landmark_index, :2]
        )
        self.rng = rng
        self.seed = int(seed)
        self.goal_key = None
        self.values = None
        self.coverage = None
        self.node_field = None

    def _set_goal(self, fields):
        key = tuple(np.asarray(fields.goal, np.float32))
        if key != self.goal_key:
            self.values, self.coverage, self.node_field = stitched_values(
                self.graph, self.targets, self.weight, fields.goal
            )
            self.goal_key = key

    def _candidate_value(
        self,
        state,
        commands,
        target_index,
        remaining,
        reliability,
        seed,
        return_members=False,
    ):
        source = np.repeat(state[None], len(commands), axis=0)
        target_state = self.graph.nodes[target_index]
        inputs = consequence_inputs(source, commands)
        samples = ENDPOINT_SAMPLES // len(self.consequence.members)
        member_q = []
        member_weight = []
        continuation = (
            self.node_field[target_index]
            + self.values[remaining - 1, target_index]
        )
        for member in range(len(self.consequence.members)):
            rng = np.random.default_rng(seed + 1_000_003 * member)
            delta = self.consequence.sample_member(
                inputs, member, samples, rng
            )
            endpoint = source[:, None] + delta
            spatial = np.exp(
                -np.square(
                    endpoint[..., :2] - target_state[:, None, :2]
                ).sum(axis=2)
                / (2.0 * LANDING_SIGMA**2)
            )
            endpoint_mobility = self.mobility.predict(
                endpoint.reshape(-1, STATE_DIM)
            ).reshape(len(commands), samples)
            bridge = (spatial * endpoint_mobility).mean(axis=1)
            member_weight.append(bridge)
            member_q.append(bridge * continuation * reliability)
        member_q = np.stack(member_q, axis=1)
        member_weight = np.stack(member_weight, axis=1)
        pessimistic_q = member_q.min(axis=1)
        pessimistic_weight = member_weight.min(axis=1)
        if return_members:
            return (
                pessimistic_q,
                pessimistic_weight,
                member_q,
                member_weight,
            )
        return pessimistic_q, pessimistic_weight

    def choose(
        self,
        state,
        command_index,
        fields,
        online_states=(),
        online_trajectories=(),
    ):
        del online_states, online_trajectories
        self._set_goal(fields)
        remaining = 10 - int(command_index)
        state = np.asarray(state, np.float32).reshape(STATE_DIM)

        _distance, local = self.landmark_tree.query(
            state[:2],
            k=min(LANDMARK_NEIGHBORS, len(self.landmark_index)),
        )
        bridge_target = self.landmark_index[np.atleast_1d(local)]
        bridge_command = self.graph.nodes[bridge_target, :2]
        bridge_q, bridge_weight = self._candidate_value(
            state,
            bridge_command,
            bridge_target,
            remaining,
            np.ones(len(bridge_target), np.float32),
            self.seed + command_index * 10_007,
        )

        recorded_command, recorded_edge, proposal = self.graph.sample_commands(
            self.rng, state, CANDIDATES
        )
        recorded_target = self.graph.endpoint_index[recorded_edge]
        recorded_q, recorded_weight = self._candidate_value(
            state,
            recorded_command,
            recorded_target,
            remaining,
            self.graph.edge_reliability[recorded_edge],
            self.seed + command_index * 10_007 + 97,
        )

        command = np.concatenate((bridge_command, recorded_command), axis=0)
        score = np.concatenate((bridge_q, recorded_q), axis=0)
        connection = np.concatenate((bridge_weight, recorded_weight), axis=0)
        kind = np.asarray(
            ["model_bridge"] * len(bridge_command)
            + ["recorded_edge"] * len(recorded_command)
        )
        winner = int(np.argmax(score))
        diagnostic = {
            "planner": "mobility_stitched_value_iteration",
            "method_revision": METHOD_REVISION,
            "phase": "supplied_goal",
            "remaining": int(remaining),
            "selected_score": float(score[winner]),
            "selected_kind": str(kind[winner]),
            "selected_connection_weight": float(connection[winner]),
            "selected_command_distance": float(
                np.linalg.norm(command[winner] - state[:2])
            ),
            "positive_value_fraction": self.coverage[-1][
                "positive_fraction"
            ],
            "meaningful_value_fraction": self.coverage[-1][
                "meaningful_fraction"
            ],
            "top_scores": np.sort(score)[-5:][::-1].tolist(),
            "search_enabled": False,
            "acquisition_search_mode": "supplied_goal",
            "fine_search_sigma": float(fields.sigma),
            "coarse_search_sigma": float(fields.sigma),
            "coarse_search_weight": 0.0,
            "fine_unsearched_mass": None,
            "coarse_unsearched_mass": None,
        }
        diagnostic.update(proposal)
        return command[winner].copy(), diagnostic


def command_evaluate(args):
    graph, consequence, consequence_path = _load_base(args)
    artifact, landmark_index, targets, _distance, _landmark_meta = (
        _prepare_landmarks(args, graph, consequence_path)
    )
    if not artifact["bridges"].exists():
        raise FileNotFoundError("run prepare-stitched-bridges first")
    with np.load(artifact["bridges"]) as data:
        saved_landmarks = data["landmark_index"]
        saved_targets = data["target_index"]
        weight = data["weight"]
    if not np.array_equal(landmark_index, saved_landmarks):
        raise ValueError("bridge landmark set changed")
    if not np.array_equal(targets, saved_targets):
        raise ValueError("bridge targets changed")
    _verify_bridges(
        args,
        graph,
        consequence_path,
        artifact,
        landmark_index,
        targets,
        weight,
    )
    mobility = MobilityModel(artifact["mobility"], args.device)
    planner = MobilityStitchedPlanner(
        consequence,
        mobility,
        graph,
        landmark_index,
        targets,
        weight,
        np.random.default_rng(args.seed),
        args.seed,
    )
    executor = FaramaSACExecutor(paths(args.root)["executor"], args.device)
    selected = panel(args.root, "development", 9, 37701)
    output = (
        paths(args.root)["eval"]
        / "development"
        / "mobility_stitched_value_iteration_r10"
    )
    output.mkdir(parents=True, exist_ok=True)
    env = make_env(max_episode_steps=10_000)
    rows = []
    coverage = []
    for task in selected["tasks"]:
        task_seed = int(args.seed) + 1_000_003 * int(task["task_id"])
        planner.rng = np.random.default_rng(task_seed)
        planner.seed = task_seed
        planner.goal_key = None
        row = evaluate_supplied_goal_task(
            env,
            executor,
            planner,
            np.asarray(task["goal"], np.float32),
            task["reset_seed"],
            LANDING_SIGMA,
            max_steps=EPISODE_STEPS,
            log_path=output / f"task_{task['task_id']:03d}.jsonl",
        )
        row.update(task)
        row["value_coverage"] = planner.coverage
        rows.append(row)
        coverage.append(planner.coverage[-1])
        print(json.dumps(row), flush=True)
    env.close()
    result = {
        "method_revision": METHOD_REVISION,
        "claim_level": CLAIM_LEVEL,
        "declared_deviations": [
            "supplied_goal_consumed_development_diagnostic",
            "model_bridges_between_exact_recorded_transition_strands",
        ],
        "oracle_inputs": [],
        "panel": "development",
        "tasks": len(rows),
        "successes": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_steps_to_goal_success": (
            float(np.mean([
                row["steps_to_goal"] for row in rows if row["success"]
            ]))
            if any(row["success"] for row in rows) else None
        ),
        "mean_positive_value_coverage": float(np.mean([
            row["positive_fraction"] for row in coverage
        ])),
        "mean_meaningful_value_coverage": float(np.mean([
            row["meaningful_fraction"] for row in coverage
        ])),
        "recorded_nodes_preserved": int(graph.node_count),
        "recorded_edges_preserved": int(graph.edge_count),
        "model_bridges": int(weight.size),
        "landmarks": int(len(landmark_index)),
        "global_landmark_neighbors": int(targets.shape[1]),
        "graph_sha256": sha256_file(args.graph),
        "factorized_consequence_sha256": sha256_file(consequence_path),
        "mobility_sha256": sha256_file(artifact["mobility"]),
        "bridges_sha256": sha256_file(artifact["bridges"]),
        "fixed_contract_oracle_reference": {"successes": 8, "tasks": 9},
        "fresh_panel_accessed": False,
        "task_gradient_updates": 0,
        "rows": rows,
    }
    dump(output / "summary.json", result)
    print(json.dumps({
        "successes": result["successes"],
        "tasks": result["tasks"],
        "success_rate": result["success_rate"],
        "mean_positive_value_coverage": result[
            "mean_positive_value_coverage"
        ],
        "mean_meaningful_value_coverage": result[
            "mean_meaningful_value_coverage"
        ],
    }, indent=2))


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("/home/ai_diagnosis/SG/e377")
    )
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=605)
    subparsers = parser.add_subparsers(dest="command", required=True)
    mobility = subparsers.add_parser("prepare-mobility")
    mobility.set_defaults(func=command_prepare_mobility)
    bridges = subparsers.add_parser("prepare-bridges")
    bridges.set_defaults(func=command_prepare_bridges)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main():
    args = parser().parse_args()
    if args.graph is None:
        args.graph = _default_graph(args.root)
    args.func(args)


if __name__ == "__main__":
    main()
