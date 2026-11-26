"""Train the shared passage estimator from PointMaze reward-free data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from projects.tfsn_consequence_value_iteration.shared.learned_passage import (
    PassageTrainingConfig,
    train_passage_ensemble,
)

from .dataset import load_current_aligned_dataset
from .roadmap import RecordedRoadmap


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "method_manifest_factorial_r12.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config(manifest):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roadmap", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(f"[point-r12][prepare] loading roadmap={args.roadmap}", flush=True)
    roadmap_meta = json.loads((args.roadmap / "roadmap.json").read_text())
    if roadmap_meta.get("method_revision") != "pointmaze_continuous_reference_roadmap_r11":
        raise ValueError("PointMaze factorial passage training requires the R11 roadmap")
    if roadmap_meta.get("task_reward_used") is not False or roadmap_meta.get(
        "hidden_goals_used"
    ) is not False:
        raise ValueError("PointMaze passage source is not reward-free")
    graph_path = Path(roadmap_meta["graph"])
    if sha256(graph_path) != roadmap_meta["graph_sha256"]:
        raise ValueError("PointMaze recorded graph changed")
    graph = RecordedRoadmap.load(graph_path)
    dataset = load_current_aligned_dataset(args.alignment)
    if dataset.rows != graph.edge_count:
        raise ValueError("PointMaze passage rows do not match the R11 graph")
    if not np.array_equal(dataset.starts, graph.nodes[graph.source_index]):
        raise ValueError("PointMaze passage starts do not match the R11 graph")
    if not np.array_equal(dataset.commands, graph.commands):
        raise ValueError("PointMaze passage commands do not match the R11 graph")
    if not np.array_equal(dataset.primitives[:, :, :2], graph.primitive_xy):
        raise ValueError("PointMaze primitive passages do not match the R11 graph")
    manifest = json.loads(MANIFEST.read_text())
    print(
        "[point-r12][prepare] "
        f"rows={dataset.rows} state_dim={dataset.starts.shape[1]} "
        f"primitive_steps={dataset.primitives.shape[1]} "
        f"sigma={roadmap_meta['landing_sigma_m']}",
        flush=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = train_passage_ensemble(
        states=dataset.starts,
        commands=dataset.commands,
        primitive_xy=dataset.primitives[:, :, :2],
        group_id=dataset.trajectory_id,
        sigma=float(roadmap_meta["landing_sigma_m"]),
        config=config(manifest),
        artifact_path=args.output / "passage.pt",
        metadata_path=args.output / "passage.json",
        provenance={
            "domain_adapter": "pointmaze_observation4_phi_first2",
            "recorded_graph": str(graph_path),
            "recorded_graph_sha256": sha256(graph_path),
            "alignment": str(args.alignment.resolve()),
            "dataset_sources": list(dataset.source_names),
        },
        device=args.device,
        manifest_path=MANIFEST,
        progress_label="point-r12",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
