"""Machine-check the R12 support, exhaustion, and shared-model contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .latent_detection import LatentRadiusPosterior
from .learned_passage import (
    LOSS_NAME,
    MODEL_CLASS,
    TRAINING_RULE,
    LearnedPassageEnsemble,
)
from .r12_settings import R12_MANIFEST, R12_MANIFEST_PATH


ROOT = Path(__file__).resolve().parents[3]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def passage_contract(path, domain, expected_state_dim, expected_sigma):
    metadata = json.loads(Path(path).read_text())
    expected = {
        "model_class": MODEL_CLASS,
        "loss": LOSS_NAME,
        "training_rule": TRAINING_RULE,
        "reward_used": False,
        "hidden_goal_used": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{domain} passage artifact has another {key}")
    provenance = metadata["provenance"]
    if provenance.get("shared_method_manifest_sha256") != sha256(R12_MANIFEST_PATH):
        raise ValueError(f"{domain} passage artifact has a stale manifest hash")
    artifact = Path(metadata["artifact"])
    if metadata.get("artifact_sha256") != sha256(artifact):
        raise ValueError(f"{domain} passage checkpoint hash changed")
    runtime = LearnedPassageEnsemble(artifact, "cpu")
    if runtime.state_dim != int(expected_state_dim):
        raise ValueError(f"{domain} passage checkpoint has another state adapter")
    if runtime.sigma != float(expected_sigma):
        raise ValueError(f"{domain} passage checkpoint has another landing sigma")
    return {
        "metadata": str(Path(path).resolve()),
        "artifact_sha256": metadata["artifact_sha256"],
        "model_class": metadata["model_class"],
        "loss": metadata["loss"],
        "training_rule": metadata["training_rule"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ant-graph", type=Path, required=True)
    parser.add_argument("--ant-passage-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with np.load(args.ant_graph, allow_pickle=False) as stored:
        ant_nodes = stored["nodes"]
    ant = LatentRadiusPosterior(ant_nodes[:, :2], sigma=float(
        json.loads(args.ant_graph.with_suffix(".json").read_text())["field_sigma_m"]
    ))
    domain_sources = [
        ROOT / "projects/tfsn_consequence_value_iteration/antmaze/r49_latent_passage_factorial.py",
    ]
    local_model_declarations = []
    for source in domain_sources:
        text = source.read_text()
        if "class PassageMLP" in text or "nn.Module" in text:
            local_model_declarations.append(str(source))
    if local_model_declarations:
        raise ValueError(
            "domain-local passage model implementation found: "
            + ", ".join(local_model_declarations)
        )
    protocols = [
        ROOT / "projects/tfsn_consequence_value_iteration/antmaze/r49_latent_passage_factorial_protocol.json",
    ]
    for path in protocols:
        value = json.loads(path.read_text())
        if not all(row["status"] == "pass" for row in value["conformance_audit"]):
            raise ValueError(f"failed conformance item in {path}")
    report = {
        "schema_version": 1,
        "shared_method_revision": R12_MANIFEST["method_revision"],
        "shared_method_manifest_sha256": sha256(R12_MANIFEST_PATH),
        "candidate_goal_contract": {
            "source": "unique_recorded_roadmap_state_xy",
            "prior": "uniform_over_support_points",
            "antmaze_candidate_count": len(ant.goal_xy),
            "antmaze_rho_min": ant.rho_min,
            "antmaze_rho_max": ant.rho_max,
        },
        "posterior_exhaustion": "fatal_no_fallback",
        "domain_local_passage_model_declarations": local_model_declarations,
        "passage_artifacts": {
            "antmaze": passage_contract(
                args.ant_passage_metadata, "antmaze", ant_nodes.shape[1], ant.rho_min
            ),
        },
        "conforms": True,
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
