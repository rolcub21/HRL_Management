"""Learn endpoint mobility and cache model-bridge reliability."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from projects.tfsn_consequence_value_iteration.shared.factorized import (
    FactorizedConsequenceEnsemble,
)
from projects.tfsn_consequence_value_iteration.shared.settings import (
    ARTIFACT_NODE_SHARD,
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_DIMENSION,
    LEARNING_RATE,
    PAIR_INFERENCE_BATCH,
)

from .core import atomic_json, atomic_npz, atomic_torch, load_json, protocol, sha256
from .roadmap import RecordedRoadmap


METHOD_REVISION = "pointmaze_initial_mobility_weighted_bridges_r1"
ALIGNED_METHOD_REVISION = "pointmaze_current_aligned_mobility_weighted_bridges_r4"
MOVEMENT_REACH_METHOD_REVISION = (
    "pointmaze_density_movement_reach_mobility_weighted_bridges_r7"
)
REFERENCE_METHOD_REVISION = "pointmaze_continuous_reference_bridges_r11"


class MobilityNetwork(torch.nn.Module):
    def __init__(self, state_dim=4):
        super().__init__()
        layers = []
        width = int(state_dim)
        for _ in range(3):
            layers.extend((torch.nn.Linear(width, HIDDEN_DIMENSION), torch.nn.SiLU()))
            width = HIDDEN_DIMENSION
        layers.extend((torch.nn.Linear(width, 1), torch.nn.Sigmoid()))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, states):
        return self.network(states).squeeze(-1)


class MobilityModel:
    def __init__(self, checkpoint, device):
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if payload.get("artifact_type") not in (
            "pointmaze_endpoint_mobility", "executor_goal_endpoint_mobility"
        ):
            raise ValueError("not an endpoint-mobility checkpoint")
        self.device = torch.device(device)
        self.mean = np.asarray(payload["state_mean"], np.float32)
        self.std = np.asarray(payload["state_std"], np.float32)
        self.network = MobilityNetwork(len(self.mean)).to(self.device)
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()

    def predict(self, states, batch_size=65_536):
        values = np.asarray(states, np.float32).reshape(-1, len(self.mean))
        rows = []
        with torch.no_grad():
            for begin in range(0, len(values), int(batch_size)):
                normalized = (
                    values[begin:begin + batch_size] - self.mean
                ) / self.std
                tensor = torch.as_tensor(
                    normalized, dtype=torch.float32, device=self.device
                )
                rows.append(self.network(tensor).cpu().numpy())
        return np.concatenate(rows).astype(np.float32)


def artifact_paths(output):
    output = Path(output)
    return {
        "mobility_shards": output / "mobility_shards",
        "mobility_labels": output / "mobility_labels.npz",
        "mobility": output / "mobility.pt",
        "mobility_meta": output / "mobility.json",
        "bridge_shards": output / "bridge_shards",
        "bridges": output / "bridges.npz",
        "bridge_meta": output / "bridges.json",
    }


def load_inputs(models, roadmap, device):
    metrics = load_json(Path(models) / "factorized_consequence_metrics.json")
    checkpoint = Path(metrics["artifact"])
    roadmeta = load_json(Path(roadmap) / "roadmap.json")
    if sha256(checkpoint) != metrics["artifact_sha256"]:
        raise ValueError("factorized consequence checkpoint changed")
    if sha256(Path(roadmeta["graph"])) != roadmeta["graph_sha256"]:
        raise ValueError("recorded graph changed")
    if roadmeta["factorized_consequence_sha256"] != metrics["artifact_sha256"]:
        raise ValueError("roadmap and model do not match")
    graph = RecordedRoadmap.load(roadmeta["graph"])
    with np.load(roadmeta["reach_candidates"], allow_pickle=False) as stored:
        targets = stored["target_index"].astype(np.int32)
        landmarks = stored["landmark_index"].astype(np.int32)
    model = FactorizedConsequenceEnsemble(checkpoint, device)
    return metrics, roadmeta, graph, targets, landmarks, model


def movement_probability(model, states, commands, sigma):
    member = []
    for index in range(len(model.spatial_members)):
        member.append(1.0 - model.stall_probability(states, commands, sigma, index))
    return np.stack(member, axis=1).min(axis=1).astype(np.float32)


def prepare_mobility_labels(
    graph, targets, model, sigma, output, graph_hash, model_hash
):
    paths = artifact_paths(output)
    paths["mobility_shards"].mkdir(parents=True, exist_ok=True)
    for begin in range(0, graph.node_count, ARTIFACT_NODE_SHARD):
        stop = min(begin + ARTIFACT_NODE_SHARD, graph.node_count)
        path = paths["mobility_shards"] / f"nodes_{begin:06d}_{stop:06d}.npz"
        if path.exists():
            with np.load(path) as stored:
                if stored["mobility"].shape == (stop - begin,):
                    continue
        target = targets[begin:stop]
        states = np.repeat(
            graph.nodes[begin:stop, None], target.shape[1], axis=1
        ).reshape(-1, graph.nodes.shape[1])
        commands = graph.nodes[
            target, : graph.goal_dim
        ].reshape(-1, graph.goal_dim)
        rows = []
        for pair_begin in range(0, len(states), PAIR_INFERENCE_BATCH):
            pair_stop = min(pair_begin + PAIR_INFERENCE_BATCH, len(states))
            rows.append(movement_probability(
                model,
                states[pair_begin:pair_stop],
                commands[pair_begin:pair_stop],
                sigma,
            ))
        mobility = np.concatenate(rows).reshape(stop - begin, -1).max(axis=1)
        atomic_npz(path, mobility=mobility.astype(np.float32))
        print(
            f"point mobility labels nodes={stop}/{graph.node_count} "
            f"median={np.median(mobility):.4f}", flush=True
        )
    labels = []
    for begin in range(0, graph.node_count, ARTIFACT_NODE_SHARD):
        stop = min(begin + ARTIFACT_NODE_SHARD, graph.node_count)
        with np.load(
            paths["mobility_shards"] / f"nodes_{begin:06d}_{stop:06d}.npz"
        ) as stored:
            labels.append(stored["mobility"])
    labels = np.concatenate(labels).astype(np.float32)
    atomic_npz(paths["mobility_labels"], mobility=labels)
    return labels


def train_mobility(
    graph, labels, output, device, schedule, method_revision=METHOD_REVISION,
    seed=947, artifact_type="pointmaze_endpoint_mobility",
):
    paths = artifact_paths(output)
    epochs = 2 if schedule == "smoke" else EPOCHS
    rng = np.random.default_rng(seed)
    order = rng.permutation(graph.node_count)
    validation_count = max(1, int(round(0.10 * graph.node_count)))
    validation, training = order[:validation_count], order[validation_count:]
    mean = graph.nodes[training].mean(axis=0).astype(np.float32)
    std = np.maximum(graph.nodes[training].std(axis=0), 1e-6).astype(np.float32)
    x = torch.as_tensor((graph.nodes - mean) / std, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    torch.manual_seed(seed + 1)
    network = MobilityNetwork(graph.nodes.shape[1]).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        network.train()
        losses = []
        epoch_order = rng.permutation(training)
        for begin in range(0, len(epoch_order), BATCH_SIZE):
            index = epoch_order[begin:begin + BATCH_SIZE]
            prediction = network(x[index].to(device))
            loss = torch.square(prediction - y[index].to(device)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        network.eval()
        with torch.no_grad():
            prediction = network(x[validation].to(device)).cpu()
        validation_mse = float(torch.square(prediction - y[validation]).mean())
        validation_mae = float(torch.abs(prediction - y[validation]).mean())
        history.append({
            "epoch": epoch + 1,
            "train_mse": float(np.mean(losses)),
            "validation_mse": validation_mse,
            "validation_mae": validation_mae,
        })
        if validation_mse < best_loss:
            best_loss = validation_mse
            best_state = copy.deepcopy(network.state_dict())
        if (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            print(
                f"point mobility epoch={epoch + 1}/{epochs} "
                f"validation_mae={validation_mae:.6f}", flush=True
            )
    payload = {
        "artifact_type": str(artifact_type),
        "method_revision": method_revision,
        "state_mean": mean,
        "state_std": std,
        "state_dict": best_state,
        "definition": "one_minus_minimum_predicted_stall_over_reach_candidates",
        "schedule": schedule,
        "seed": seed,
    }
    atomic_torch(paths["mobility"], payload)
    model = MobilityModel(paths["mobility"], device)
    prediction = model.predict(graph.nodes)
    report = {
        "schema_version": 1,
        "method_revision": method_revision,
        "schedule": schedule,
        "artifact": str(paths["mobility"].resolve()),
        "artifact_sha256": sha256(paths["mobility"]),
        "definition": payload["definition"],
        "nodes": graph.node_count,
        "label_mean": float(labels.mean()),
        "label_median": float(np.median(labels)),
        "validation_mse_best": best_loss,
        "all_node_mae": float(np.abs(prediction - labels).mean()),
        "history": history,
    }
    atomic_json(paths["mobility_meta"], report)
    return model, report


def bridge_member_weight(
    model, mobility, source, command, sigma, goal_dim=None
):
    goal_dim = int(command.shape[1] if goal_dim is None else goal_dim)
    state_dim = int(source.shape[1])
    member_rows = []
    for member in range(len(model.spatial_members)):
        weights, endpoint_state = model.component_mean_endpoint_outcomes(
            source, command, member
        )
        distance = np.linalg.norm(
            endpoint_state[..., :goal_dim] - command[:, None], axis=2
        )
        connection = np.exp(-np.square(distance) / (2.0 * float(sigma) ** 2))
        endpoint_mobility = mobility.predict(
            endpoint_state.reshape(-1, state_dim)
        ).reshape(endpoint_state.shape[:2])
        member_rows.append(
            (weights * connection * endpoint_mobility).sum(axis=1)
        )
    return np.stack(member_rows, axis=1).astype(np.float32)


def prepare_bridge_weights(
    graph, targets, model, mobility, sigma, output
):
    paths = artifact_paths(output)
    paths["bridge_shards"].mkdir(parents=True, exist_ok=True)
    members = len(model.spatial_members)
    for begin in range(0, graph.node_count, ARTIFACT_NODE_SHARD):
        stop = min(begin + ARTIFACT_NODE_SHARD, graph.node_count)
        path = paths["bridge_shards"] / f"nodes_{begin:06d}_{stop:06d}.npz"
        expected_shape = (stop - begin, targets.shape[1], members)
        if path.exists():
            with np.load(path) as stored:
                if stored["member_weight"].shape == expected_shape:
                    continue
        target = targets[begin:stop]
        source = np.repeat(
            graph.nodes[begin:stop, None], target.shape[1], axis=1
        ).reshape(-1, graph.nodes.shape[1])
        command = graph.nodes[
            target, : graph.goal_dim
        ].reshape(-1, graph.goal_dim)
        rows = []
        for pair_begin in range(0, len(source), PAIR_INFERENCE_BATCH):
            pair_stop = min(pair_begin + PAIR_INFERENCE_BATCH, len(source))
            rows.append(bridge_member_weight(
                model,
                mobility,
                source[pair_begin:pair_stop],
                command[pair_begin:pair_stop],
                sigma, graph.goal_dim,
            ))
        member_weight = np.concatenate(rows).reshape(expected_shape)
        atomic_npz(path, member_weight=member_weight)
        print(
            f"point bridge weights nodes={stop}/{graph.node_count} "
            f"median={np.median(member_weight):.4f}", flush=True
        )
    rows = []
    for begin in range(0, graph.node_count, ARTIFACT_NODE_SHARD):
        stop = min(begin + ARTIFACT_NODE_SHARD, graph.node_count)
        with np.load(
            paths["bridge_shards"] / f"nodes_{begin:06d}_{stop:06d}.npz"
        ) as stored:
            rows.append(stored["member_weight"])
    member_weight = np.concatenate(rows).astype(np.float32)
    weight = member_weight.min(axis=2).astype(np.float32)
    atomic_npz(paths["bridges"], member_weight=member_weight, weight=weight)
    return member_weight, weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("mobility", "bridges", "all"))
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--roadmap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--schedule", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    started = time.time()
    metrics, roadmeta, graph, targets, landmarks, model = load_inputs(
        args.models, args.roadmap, args.device
    )
    if roadmeta["schedule"] != args.schedule:
        raise ValueError("bridge and roadmap preparation schedules do not match")
    if args.schedule == "full" and metrics["schedule"] != "full":
        raise ValueError("full bridge preparation requires a full consequence model")
    output = args.output.resolve()
    paths = artifact_paths(output)
    sigma = float(metrics["kernel_sigma_m"])
    movement_reach = roadmeta["method_revision"] == (
        "pointmaze_movement_conditioned_reach_roadmap_r7"
    )
    # The reference property is semantic, not tied to one historical revision
    # name.  Multiseed rebuilds retain continuous reach weighting under their
    # own preparation revision.
    reference = bool(roadmeta.get("continuous_reach_weighting", False))
    aligned = reference or movement_reach or roadmeta["method_revision"] == (
        "pointmaze_current_aligned_recorded_roadmap_r4"
    )
    method_revision = (
        REFERENCE_METHOD_REVISION
        if reference
        else MOVEMENT_REACH_METHOD_REVISION
        if movement_reach
        else (ALIGNED_METHOD_REVISION if aligned else METHOD_REVISION)
    )
    if paths["mobility"].exists() and paths["mobility_meta"].exists():
        mobility = MobilityModel(paths["mobility"], args.device)
        mobility_meta = load_json(paths["mobility_meta"])
    else:
        labels = prepare_mobility_labels(
            graph, targets, model, sigma, output,
            roadmeta["graph_sha256"], metrics["artifact_sha256"],
        )
        mobility, mobility_meta = train_mobility(
            graph, labels, output, args.device, args.schedule,
            method_revision=method_revision,
        )
    if args.command in ("bridges", "all"):
        member_weight, weight = prepare_bridge_weights(
            graph, targets, model, mobility, sigma, output
        )
        report = {
            "schema_version": 1,
            "method_reference": protocol()["method_reference"],
            "method_revision": method_revision,
            "claim_level": (
                "smoke_only" if args.schedule == "smoke"
                else (
                    "reward_free_current_aligned_planner_preparation_only"
                    if aligned else "reward_free_initial_planner_preparation_only"
                )
            ),
            "declared_deviations": [],
            "oracle_inputs": [],
            "schedule": args.schedule,
            "factorized_consequence_schedule": metrics["schedule"],
            "artifact": str(paths["bridges"].resolve()),
            "artifact_sha256": sha256(paths["bridges"]),
            "recorded_graph_sha256": roadmeta["graph_sha256"],
            "factorized_consequence_sha256": metrics["artifact_sha256"],
            "mobility_sha256": mobility_meta["artifact_sha256"],
            "landing_sigma_m": sigma,
            "nodes": graph.node_count,
            "recorded_edges": graph.edge_count,
            "landmarks": int(len(landmarks)),
            "bridges": int(weight.size),
            "bridge_weight_mean": float(weight.mean()),
            "bridge_weight_median": float(np.median(weight)),
            "bridge_weight_p90": float(np.quantile(weight, 0.9)),
            "ensemble_reduction": "pessimistic_minimum",
            "movement_conditioned_reach": bool(movement_reach),
            "continuous_reach_weighting": bool(reference),
            "bridge_outcome_integration": (
                "all_joint_spatial_posture_component_means_weighted_by_"
                "joint_mixture_probability"
            ),
            "task_reward_used": False,
            "hidden_goals_used": False,
            "wall_clock_seconds": time.time() - started,
        }
        atomic_json(paths["bridge_meta"], report)
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps(mobility_meta, indent=2))


if __name__ == "__main__":
    main()
