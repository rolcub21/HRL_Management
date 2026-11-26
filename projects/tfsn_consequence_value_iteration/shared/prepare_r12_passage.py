"""AntMaze adapter for the shared R12 learned-passage training rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.antmaze.recorded_transition_planning import (
    RecordedTransitionGraph,
)
from scripts.antmaze.current_tfsn.data import part_paths

from .learned_passage import PassageTrainingConfig, train_passage_ensemble


HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "method_manifest_r12.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config(manifest: dict) -> PassageTrainingConfig:
    value = manifest["categories"]["global_computational"]
    return PassageTrainingConfig(
        passage_bins=int(value["passage_bins"]),
        ensemble_size=int(value["ensemble_size"]),
        hidden_dimension=int(value["hidden_dimension"]),
        hidden_layers=int(value["hidden_layers"]),
        batch_size=int(value["batch_size"]),
        epochs=int(value["epochs"]),
        learning_rate=float(value["learning_rate"]),
        validation_fraction=float(value["validation_fraction"]),
        std_floor=float(value["std_floor"]),
        seed=int(value["random_seed"]),
        early_stopping_patience=int(value["early_stopping_patience"]),
        early_stopping_relative_improvement=float(
            value["early_stopping_relative_improvement"]
        ),
        progress_every_epochs=int(value["progress_every_epochs"]),
    )


def _antmaze(args):
    graph = RecordedTransitionGraph.load(args.graph)
    if graph.metadata.get("method_revision") != "antmaze_r39_r46_rule_removal_r48":
        raise ValueError("AntMaze passage training requires the R48 graph")
    if graph.metadata.get("task_reward_used") is not False or graph.metadata.get(
        "hidden_goals_used"
    ) is not False:
        raise ValueError("AntMaze passage source is not reward-free")
    rounds = [Path(value) for value in graph.metadata["reward_free_rounds"]]
    state_rows, command_rows, primitive_rows, group_rows = [], [], [], []
    for round_index, directory in enumerate(rounds):
        for path in part_paths(directory):
            with np.load(path, allow_pickle=False) as stored:
                states = np.asarray(stored["states"], np.float32)
                commands = np.asarray(stored["commands"], np.float32)
                task = np.asarray(stored["task_id"], np.int64)
                episode = np.asarray(stored["episode_id"], np.int64)
                reset = np.asarray(stored["reset_seed"], np.int64)
            state_rows.append(states[:, 0])
            command_rows.append(commands)
            primitive_rows.append(states[:, 1:, :2])
            triples = np.stack((
                np.full(len(states), round_index, np.int64), reset,
                task * 1_000_000 + episode,
            ), axis=1)
            group_rows.append(triples)
    states = np.concatenate(state_rows)
    commands = np.concatenate(command_rows)
    primitive = np.concatenate(primitive_rows)
    group_tuple = np.concatenate(group_rows)
    _unique, group = np.unique(group_tuple, axis=0, return_inverse=True)
    if len(states) != graph.edge_count:
        raise ValueError("AntMaze passage rows do not match recorded graph")
    if not np.array_equal(states, graph.nodes[graph.source_index]):
        raise ValueError("AntMaze passage starts do not match recorded graph")
    if not np.array_equal(commands, graph.commands):
        raise ValueError("AntMaze passage commands do not match recorded graph")
    expected_end = graph.nodes[graph.endpoint_index]
    # The graph records exact command-boundary states.
    if not np.array_equal(
        np.concatenate((primitive[:, -1, :2], expected_end[:, 2:]), axis=1),
        expected_end,
    ):
        raise ValueError("AntMaze passage endpoints do not match recorded graph")
    return {
        "states": states,
        "commands": commands,
        "primitive_xy": primitive,
        "group_id": group,
        "sigma": float(graph.metadata["field_sigma_m"]),
        "provenance": {
            "domain_adapter": "antmaze_state29_phi_first2",
            "recorded_graph": str(args.graph.resolve()),
            "recorded_graph_sha256": _sha256(args.graph),
            "reward_free_rounds": [str(path.resolve()) for path in rounds],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("antmaze",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--graph", type=Path)
    args = parser.parse_args()
    if args.graph is None:
        parser.error("AntMaze requires --graph")
    print(
        f"[r49][prepare] loading AntMaze reward-free passages from {args.graph}",
        flush=True,
    )
    data = _antmaze(args)
    print(
        "[r49][prepare] "
        f"loaded rows={len(data['states'])} state_dim={data['states'].shape[1]} "
        f"primitive_steps={data['primitive_xy'].shape[1]} sigma={data['sigma']:.8g}",
        flush=True,
    )
    manifest = json.loads(MANIFEST_PATH.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        **data.pop("provenance"),
        "shared_method_manifest": str(MANIFEST_PATH),
        "shared_method_manifest_sha256": _sha256(MANIFEST_PATH),
        "shared_method_revision": manifest["method_revision"],
    }
    metadata = train_passage_ensemble(
        **data,
        config=_config(manifest),
        artifact_path=args.output / "passage.pt",
        metadata_path=args.output / "passage.json",
        provenance=provenance,
        device=args.device,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
