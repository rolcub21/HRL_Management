"""Machine-check the isolated PointMaze R12 factorial artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from projects.tfsn_consequence_value_iteration.shared.latent_detection import (
    LatentRadiusPosterior,
)
from projects.tfsn_consequence_value_iteration.shared.learned_passage import (
    LOSS_NAME,
    MODEL_CLASS,
    TRAINING_RULE,
    LearnedPassageEnsemble,
)

from .factorial_planner_r12 import MANIFEST
from .roadmap import RecordedRoadmap


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "factorial_protocol_r12.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--roadmap", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--passage-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    road = json.loads((args.roadmap / "roadmap.json").read_text())
    bridges = json.loads((args.planner / "bridges.json").read_text())
    models = json.loads(
        (args.models / "factorized_consequence_metrics.json").read_text()
    )
    if road.get("method_revision") != "pointmaze_continuous_reference_roadmap_r11":
        raise ValueError("factorial screen does not use the R11 roadmap")
    graph_path = Path(road["graph"])
    if sha256(graph_path) != road["graph_sha256"]:
        raise ValueError("R11 graph hash changed")
    if bridges["recorded_graph_sha256"] != road["graph_sha256"]:
        raise ValueError("R11 bridge cache uses another graph")
    if bridges["factorized_consequence_sha256"] != models["artifact_sha256"]:
        raise ValueError("R11 bridge cache uses another consequence model")
    graph = RecordedRoadmap.load(graph_path)
    posterior = LatentRadiusPosterior(
        graph.nodes[:, :2], float(road["landing_sigma_m"])
    )
    metadata_path = args.passage_metadata.resolve()
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "model_class": MODEL_CLASS,
        "loss": LOSS_NAME,
        "training_rule": TRAINING_RULE,
        "reward_used": False,
        "hidden_goal_used": False,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise ValueError(f"PointMaze passage artifact has another {name}")
    if metadata["provenance"].get("shared_method_manifest_sha256") != sha256(MANIFEST):
        raise ValueError("PointMaze passage artifact has a stale manifest hash")
    artifact = Path(metadata["artifact"])
    if metadata.get("artifact_sha256") != sha256(artifact):
        raise ValueError("PointMaze passage checkpoint hash changed")
    runtime = LearnedPassageEnsemble(
        artifact, "cpu", manifest_path=MANIFEST
    )
    if runtime.state_dim != 4:
        raise ValueError("PointMaze passage checkpoint has another state adapter")
    if runtime.sigma != posterior.rho_min:
        raise ValueError("PointMaze passage checkpoint has another landing sigma")
    source = HERE / "factorial_planner_r12.py"
    local_model = "class PassageMLP" in source.read_text() or "nn.Module" in source.read_text()
    if local_model:
        raise ValueError("PointMaze defines a domain-local passage model")
    protocol = json.loads(PROTOCOL.read_text())
    if not all(row["status"] == "pass" for row in protocol["conformance_audit"]):
        raise ValueError("PointMaze protocol contains a failed conformance item")
    report = {
        "schema_version": 1,
        "conforms": True,
        "method_revision": protocol["method_revision"],
        "method_manifest_sha256": sha256(MANIFEST),
        "protocol_sha256": sha256(PROTOCOL),
        "candidate_goal_contract": {
            "source": "unique_recorded_roadmap_state_xy",
            "prior": "uniform_over_support_points",
            "candidate_count": len(posterior.goal_xy),
            "rho_min": posterior.rho_min,
            "rho_max": posterior.rho_max,
        },
        "posterior_exhaustion": "fatal_no_fallback",
        "passage_artifact": {
            "metadata": str(metadata_path),
            "artifact_sha256": metadata["artifact_sha256"],
            "model_class": metadata["model_class"],
            "loss": metadata["loss"],
            "training_rule": metadata["training_rule"],
        },
        "r11_inputs": {
            "factorized_consequence_sha256": models["artifact_sha256"],
            "recorded_graph_sha256": road["graph_sha256"],
            "bridges_sha256": bridges["artifact_sha256"],
        },
        "domain_local_passage_model": False,
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
