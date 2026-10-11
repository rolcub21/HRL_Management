#!/usr/bin/env python3
"""Prospective VCG-Dense v1.2 candidate-pool training.

This entry point deliberately does *not* change the frozen VCG-Dense v1.1
learning recipe.  It trains fresh model seeds 3--5 by driving the authenticated
v1.1 trainer in 25-episode segments, then preserves the deployment weights at
every validation epoch as an immutable candidate pool.  Candidate selection is
a separate, later protocol; no A/B selection or test panel is opened here.

The output layout is::

    OUTPUT/
      prospective-training-contract.json
      prospective-snapshot-manifest.json
      snapshots/episode-0025.pth ... episode-0500.pth
      v1_1_training/                 # authenticated resumable trainer state

Only runtime placement, model seed, device, logging, and pause/resume are
configurable.  Objective, gamma, architecture, environment, training budget,
validation schedule, and instance namespaces remain frozen by
``train_vcg_dense_proper``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

import run_vcg_objective_gamma_audit as audit
import train_vcg_dense_proper as base
from train_viability_graph_smdp import resolve_device
from viability_graph_episodic_audit import (
    EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
)


PROSPECTIVE_TRAINING_PROTOCOL = (
    "vcg_dense_v1_2_guarded_prospective_training_v1"
)
PROSPECTIVE_METHOD_VERSION = "vcg_dense_v1_2_guarded"
PROSPECTIVE_TRAINER_SCHEMA_VERSION = 1
PROSPECTIVE_SNAPSHOT_MANIFEST_SCHEMA_VERSION = 1
SNAPSHOT_ROLE = "prospective_validation_snapshot_v1"

FRESH_MODEL_SEEDS = (3, 4, 5)
SNAPSHOT_EPISODES = tuple(
    range(base.FROZEN_EVAL_EVERY, base.FROZEN_TOTAL_EPISODES + 1,
          base.FROZEN_EVAL_EVERY)
)
BASE_STATE_DIRECTORY = "v1_1_training"
SNAPSHOT_DIRECTORY = "snapshots"
CONTRACT_FILENAME = "prospective-training-contract.json"
SNAPSHOT_MANIFEST_FILENAME = "prospective-snapshot-manifest.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one fresh VCG-Dense v1.2 candidate pool with the exact "
            "frozen v1.1 recipe"
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-seed", type=int, choices=FRESH_MODEL_SEEDS, required=True
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument(
        "--stop-after-episode",
        type=int,
        choices=SNAPSHOT_EPISODES,
        help="pause only at an authenticated validation/snapshot boundary",
    )
    parser.add_argument("--log-every", type=int, default=5)
    return parser


def _base_runtime(runtime, *, stop_after_episode: int) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=Path(runtime.output_dir) / BASE_STATE_DIRECTORY,
        model_seed=int(runtime.model_seed),
        device=str(runtime.device),
        resume_existing=False,
        stop_after_episode=int(stop_after_episode),
        log_every=int(runtime.log_every),
    )


def _prospective_contract(runtime) -> tuple[dict, dict, int]:
    """Return prospective contract, exact base contract, and train seed base."""

    base_runtime = _base_runtime(
        runtime, stop_after_episode=SNAPSHOT_EPISODES[0]
    )
    args = base._frozen_audit_args(base_runtime)
    train_seed_base = base._assert_frozen_recipe(args)
    base_contract = base._resume_contract(args, train_seed_base)
    contract = {
        "prospective_training_protocol": PROSPECTIVE_TRAINING_PROTOCOL,
        "prospective_method_version": PROSPECTIVE_METHOD_VERSION,
        "prospective_trainer_schema_version": (
            PROSPECTIVE_TRAINER_SCHEMA_VERSION
        ),
        "candidate_selection_performed": False,
        "performance_claim_authorized": False,
        "model_seed": int(runtime.model_seed),
        "allowed_fresh_model_seeds": FRESH_MODEL_SEEDS,
        "total_training_episodes": base.FROZEN_TOTAL_EPISODES,
        "snapshot_episodes": SNAPSHOT_EPISODES,
        "snapshot_filename_template": "snapshots/episode-{episode:04d}.pth",
        "base_state_directory": BASE_STATE_DIRECTORY,
        "base_training_protocol": base.TRAINING_PROTOCOL,
        "base_method_version": base.METHOD_VERSION,
        "base_frozen_recipe_source": base.FROZEN_RECIPE_SOURCE,
        "base_resume_contract": base_contract,
        "base_resume_contract_sha256": audit._contract_hash(base_contract),
        "training_instance_seed_base": int(train_seed_base),
        "legacy_training_validation_instance_seeds": (
            base.FROZEN_VALIDATION_SEEDS
        ),
        "future_a_b_selection_panels_opened": False,
        "sealed_test_panels_opened": False,
        "immutable_model_only_snapshot_pool": True,
    }
    return contract, base_contract, int(train_seed_base)


def _snapshot_relative_path(episode: int) -> Path:
    episode = int(episode)
    if episode not in SNAPSHOT_EPISODES:
        raise ValueError(f"episode {episode} is not a snapshot boundary")
    return Path(SNAPSHOT_DIRECTORY) / f"episode-{episode:04d}.pth"


def _initial_manifest(contract: dict) -> dict:
    return {
        "manifest_schema_version": (
            PROSPECTIVE_SNAPSHOT_MANIFEST_SCHEMA_VERSION
        ),
        "prospective_training_protocol": PROSPECTIVE_TRAINING_PROTOCOL,
        "prospective_method_version": PROSPECTIVE_METHOD_VERSION,
        "prospective_training_contract_sha256": audit._contract_hash(contract),
        "base_resume_contract_sha256": contract[
            "base_resume_contract_sha256"
        ],
        "model_seed": int(contract["model_seed"]),
        "snapshot_episodes": SNAPSHOT_EPISODES,
        "snapshot_filename_template": contract[
            "snapshot_filename_template"
        ],
        "entries": [],
        "completed_training_episodes": 0,
        "candidate_pool_complete": False,
        "candidate_selection_performed": False,
        "deployment_checkpoint_eligible": False,
        "performance_claim_authorized": False,
        "future_a_b_selection_panels_opened": False,
        "sealed_test_panels_opened": False,
        # The V1.1 reference is defined by the frozen legacy-validation
        # lexicographic rule.  It becomes knowable only after episode 500;
        # the guarded alternative remains a later A/B-selection decision.
        "v1_1_reference_checkpoint_episode": None,
        "v1_1_reference_checkpoint_sha256": None,
        "v1_1_best_checkpoint_sha256": None,
        # Panel A ranks guarded alternatives relative to the frozen V1.1
        # reference; it must never silently redefine a second reference.
        "fresh_a_reference_checkpoint_episode": None,
        "fresh_a_reference_checkpoint_sha256": None,
    }


def _load_json(path: Path, *, label: str) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read authenticated {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"authenticated {label} must be a JSON object")
    return value


def _validation_record(payload: Mapping, episode: int) -> dict:
    matches = [
        record
        for record in payload.get("validation_history", ())
        if int(record.get("checkpoint_episode", -1)) == int(episode)
    ]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise ValueError(
            f"base checkpoint must contain exactly one validation record "
            f"for episode {episode}"
        )
    record = dict(matches[0])
    if not record.get("selection_score_fields") or not record.get(
        "selection_score"
    ):
        raise ValueError("snapshot validation record has no selection identity")
    return record


def _same_policy_weights(left: Mapping, right: Mapping) -> bool:
    """Byte-semantic equality for the online and target network states."""

    try:
        for network_name in ("Q_local", "Q_target"):
            left_state = left["agent_state"][network_name]
            right_state = right["agent_state"][network_name]
            if set(left_state) != set(right_state):
                return False
            if any(
                not torch.equal(left_state[name], right_state[name])
                for name in left_state
            ):
                return False
    except (KeyError, TypeError):
        return False
    return True


def _snapshot_payload(
    source: Mapping,
    *,
    episode: int,
    source_latest_sha256: str,
    prospective_contract: dict,
    validation_record: dict,
) -> dict:
    """Create a directly deployable, non-resumable candidate checkpoint."""

    state = source.get("agent_state")
    if not isinstance(state, Mapping):
        raise ValueError("base latest checkpoint has no agent_state mapping")
    if "Q_local" not in state or "Q_target" not in state:
        raise ValueError("base latest checkpoint has no Q-network weights")
    # A candidate is deliberately weights-only.  Deployment reconstructs all
    # counters, optimizer state, replay, RNG, and episode-local liveness state
    # from safe defaults; none can influence the frozen greedy policy.
    model_state = {
        "Q_local": state["Q_local"],
        "Q_target": state["Q_target"],
    }

    payload = dict(source)
    payload["agent_state"] = model_state
    payload.update(
        {
            "trainer_resumable": False,
            "checkpoint_role": SNAPSHOT_ROLE,
            "completed_training_episodes": int(episode),
            "global_rng_state": None,
            "training_history": (),
            "validation_history": (),
            "best_validation_record": None,
            "best_candidate_checkpoint_filename": None,
            "best_candidate_checkpoint_sha256": None,
            "best_checkpoint_sha256": None,
            "selection_finalized_after_total_episodes": False,
            "selection_finalized_at_training_episode": None,
            "selected_checkpoint_episode": None,
            "deployment_checkpoint_eligible": False,
            "performance_claim_authorized": False,
            "formal_test_authorized": False,
            "prospective_training_protocol": PROSPECTIVE_TRAINING_PROTOCOL,
            "prospective_method_version": PROSPECTIVE_METHOD_VERSION,
            "prospective_trainer_schema_version": (
                PROSPECTIVE_TRAINER_SCHEMA_VERSION
            ),
            "prospective_training_contract_sha256": audit._contract_hash(
                prospective_contract
            ),
            "base_resume_contract_sha256": prospective_contract[
                "base_resume_contract_sha256"
            ],
            "source_latest_checkpoint_sha256": source_latest_sha256,
            "snapshot_episode": int(episode),
            "snapshot_validation_record": validation_record,
            "snapshot_selection_status": "unselected_candidate",
            "candidate_selection_performed": False,
            "future_a_b_selection_panels_opened": False,
            "sealed_test_panels_opened": False,
        }
    )
    return payload


def _validate_snapshot_payload(
    payload: Mapping,
    *,
    episode: int,
    prospective_contract: dict,
    source_latest_sha256: Optional[str] = None,
) -> None:
    expected = {
        "checkpoint_family": EPISODIC_VIABILITY_GRAPH_CHECKPOINT_FAMILY,
        "training_protocol": base.TRAINING_PROTOCOL,
        "method_version": base.METHOD_VERSION,
        "checkpoint_role": SNAPSHOT_ROLE,
        "trainer_resumable": False,
        "model_seed": int(prospective_contract["model_seed"]),
        "completed_training_episodes": int(episode),
        "snapshot_episode": int(episode),
        "timing_objective_spec": base.FROZEN_OBJECTIVE_SPEC.to_dict(),
        "gamma": base.FROZEN_GAMMA,
        "prospective_training_protocol": PROSPECTIVE_TRAINING_PROTOCOL,
        "prospective_method_version": PROSPECTIVE_METHOD_VERSION,
        "prospective_training_contract_sha256": audit._contract_hash(
            prospective_contract
        ),
        "base_resume_contract_sha256": prospective_contract[
            "base_resume_contract_sha256"
        ],
        "candidate_selection_performed": False,
        "selection_finalized_after_total_episodes": False,
        "deployment_checkpoint_eligible": False,
        "performance_claim_authorized": False,
        "future_a_b_selection_panels_opened": False,
        "sealed_test_panels_opened": False,
        "protocol_training_complete": (
            int(episode) == base.FROZEN_TOTAL_EPISODES
        ),
        "exact_full": True,
        "viability_critic_enabled": False,
        "baseline_teacher": False,
        "baseline_policy_query": False,
        "future_schedule_visible_to_policy": False,
        "resume_contract_sha256": prospective_contract[
            "base_resume_contract_sha256"
        ],
    }
    mismatches = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if payload.get("resume_contract") != prospective_contract[
        "base_resume_contract"
    ]:
        mismatches["resume_contract"] = (
            payload.get("resume_contract"),
            prospective_contract["base_resume_contract"],
        )
    if source_latest_sha256 is not None and payload.get(
        "source_latest_checkpoint_sha256"
    ) != source_latest_sha256:
        mismatches["source_latest_checkpoint_sha256"] = (
            payload.get("source_latest_checkpoint_sha256"),
            source_latest_sha256,
        )
    state = payload.get("agent_state")
    if not isinstance(state, Mapping):
        mismatches["agent_state"] = (type(state).__name__, "mapping")
    else:
        state_keys = tuple(sorted(state))
        expected_state_keys = ("Q_local", "Q_target")
        if state_keys != expected_state_keys:
            mismatches["model_only_agent_state"] = (
                state_keys, expected_state_keys
            )
    record = payload.get("snapshot_validation_record")
    if not isinstance(record, Mapping) or int(
        record.get("checkpoint_episode", -1)
    ) != int(episode):
        mismatches["snapshot_validation_record"] = (
            None if not isinstance(record, Mapping) else record.get(
                "checkpoint_episode"
            ),
            int(episode),
        )
    if mismatches:
        raise ValueError(
            f"incompatible prospective snapshot episode {episode}: "
            f"{mismatches!r}"
        )


def _entry_from_snapshot(
    root: Path,
    *,
    episode: int,
    source_latest_sha256: str,
    prospective_contract: dict,
    validation_record: dict,
) -> dict:
    relative = _snapshot_relative_path(episode)
    path = root / relative
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"existing snapshot is not a mapping: {path}")
        _validate_snapshot_payload(
            payload,
            episode=episode,
            prospective_contract=prospective_contract,
            source_latest_sha256=source_latest_sha256,
        )
        if audit._json_safe(payload["snapshot_validation_record"]) != (
            audit._json_safe(validation_record)
        ):
            raise ValueError(
                f"existing snapshot validation record mismatch: {path}"
            )
    else:
        source_path = root / BASE_STATE_DIRECTORY / "latest.pth"
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        if not isinstance(source, Mapping):
            raise ValueError("base latest checkpoint must be a mapping")
        payload = _snapshot_payload(
            source,
            episode=episode,
            source_latest_sha256=source_latest_sha256,
            prospective_contract=prospective_contract,
            validation_record=validation_record,
        )
        _validate_snapshot_payload(
            payload,
            episode=episode,
            prospective_contract=prospective_contract,
            source_latest_sha256=source_latest_sha256,
        )
        audit._atomic_torch_save(payload, path)
    return {
        "episode": int(episode),
        "relative_path": relative.as_posix(),
        "sha256": audit._sha256(path),
        "source_latest_relative_path": (
            f"{BASE_STATE_DIRECTORY}/latest.pth"
        ),
        "source_latest_sha256": source_latest_sha256,
        "base_resume_contract_sha256": prospective_contract[
            "base_resume_contract_sha256"
        ],
        "validation_record_sha256": audit._contract_hash(validation_record),
        "validation_selection_score": tuple(
            float(value) for value in validation_record["selection_score"]
        ),
        "validation_selection_score_fields": tuple(
            str(value)
            for value in validation_record["selection_score_fields"]
        ),
    }


def _validate_manifest(
    root: Path, manifest: Mapping, prospective_contract: dict
) -> None:
    expected = {
        "manifest_schema_version": (
            PROSPECTIVE_SNAPSHOT_MANIFEST_SCHEMA_VERSION
        ),
        "prospective_training_protocol": PROSPECTIVE_TRAINING_PROTOCOL,
        "prospective_method_version": PROSPECTIVE_METHOD_VERSION,
        "prospective_training_contract_sha256": audit._contract_hash(
            prospective_contract
        ),
        "base_resume_contract_sha256": prospective_contract[
            "base_resume_contract_sha256"
        ],
        "model_seed": int(prospective_contract["model_seed"]),
        "snapshot_episodes": list(SNAPSHOT_EPISODES),
        "snapshot_filename_template": prospective_contract[
            "snapshot_filename_template"
        ],
        "candidate_selection_performed": False,
        "deployment_checkpoint_eligible": False,
        "performance_claim_authorized": False,
        "future_a_b_selection_panels_opened": False,
        "sealed_test_panels_opened": False,
    }
    mismatches = {
        name: (manifest.get(name), value)
        for name, value in expected.items()
        if manifest.get(name) != value
    }
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        mismatches["entries"] = (type(entries).__name__, "list")
        entries = []
    episodes = [int(entry.get("episode", -1)) for entry in entries]
    expected_prefix = list(SNAPSHOT_EPISODES[: len(entries)])
    if episodes != expected_prefix:
        mismatches["entry_episode_prefix"] = (episodes, expected_prefix)
    if int(manifest.get("completed_training_episodes", -1)) != (
        episodes[-1] if episodes else 0
    ):
        mismatches["completed_training_episodes"] = (
            manifest.get("completed_training_episodes"),
            episodes[-1] if episodes else 0,
        )
    for entry in entries:
        episode = int(entry["episode"])
        expected_relative = _snapshot_relative_path(episode).as_posix()
        if entry.get("relative_path") != expected_relative:
            mismatches[f"entry_{episode}_relative_path"] = (
                entry.get("relative_path"),
                expected_relative,
            )
            continue
        path = root / expected_relative
        expected_entry_metadata = {
            "source_latest_relative_path": (
                f"{BASE_STATE_DIRECTORY}/latest.pth"
            ),
            "base_resume_contract_sha256": prospective_contract[
                "base_resume_contract_sha256"
            ],
        }
        for name, expected_value in expected_entry_metadata.items():
            if entry.get(name) != expected_value:
                mismatches[f"entry_{episode}_{name}"] = (
                    entry.get(name), expected_value
                )
        if not path.is_file():
            mismatches[f"entry_{episode}_file"] = (False, True)
            continue
        actual_hash = audit._sha256(path)
        if entry.get("sha256") != actual_hash:
            mismatches[f"entry_{episode}_sha256"] = (
                entry.get("sha256"), actual_hash
            )
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            mismatches[f"entry_{episode}_payload"] = (
                type(payload).__name__, "mapping"
            )
            continue
        try:
            _validate_snapshot_payload(
                payload,
                episode=episode,
                prospective_contract=prospective_contract,
                source_latest_sha256=entry.get(
                    "source_latest_sha256"
                ),
            )
        except ValueError as error:
            mismatches[f"entry_{episode}_payload"] = (str(error), "valid")
            continue
        record = payload["snapshot_validation_record"]
        expected_record_hash = audit._contract_hash(dict(record))
        if entry.get("validation_record_sha256") != expected_record_hash:
            mismatches[f"entry_{episode}_validation_record_sha256"] = (
                entry.get("validation_record_sha256"), expected_record_hash
            )
        expected_score = [float(value) for value in record["selection_score"]]
        if entry.get("validation_selection_score") != expected_score:
            mismatches[f"entry_{episode}_validation_selection_score"] = (
                entry.get("validation_selection_score"), expected_score
            )
        expected_fields = [
            str(value) for value in record["selection_score_fields"]
        ]
        if entry.get("validation_selection_score_fields") != expected_fields:
            mismatches[
                f"entry_{episode}_validation_selection_score_fields"
            ] = (
                entry.get("validation_selection_score_fields"),
                expected_fields,
            )
    complete = len(entries) == len(SNAPSHOT_EPISODES)
    if bool(manifest.get("candidate_pool_complete")) != complete:
        mismatches["candidate_pool_complete"] = (
            manifest.get("candidate_pool_complete"), complete
        )
    reference_episode = manifest.get("v1_1_reference_checkpoint_episode")
    reference_hash = manifest.get("v1_1_reference_checkpoint_sha256")
    best_hash = manifest.get("v1_1_best_checkpoint_sha256")
    if complete:
        entry_by_episode = {int(entry["episode"]): entry for entry in entries}
        if reference_episode not in entry_by_episode:
            mismatches["v1_1_reference_checkpoint_episode"] = (
                reference_episode, "one snapshot episode"
            )
        elif reference_hash != entry_by_episode[reference_episode]["sha256"]:
            mismatches["v1_1_reference_checkpoint_sha256"] = (
                reference_hash, entry_by_episode[reference_episode]["sha256"]
            )
        if not base._is_sha256(best_hash):
            mismatches["v1_1_best_checkpoint_sha256"] = (
                best_hash, "SHA-256"
            )
        else:
            base_best_path = root / BASE_STATE_DIRECTORY / "best.pth"
            if not base_best_path.is_file():
                mismatches["v1_1_best_checkpoint_file"] = (False, True)
            elif audit._sha256(base_best_path) != best_hash:
                mismatches["v1_1_best_checkpoint_file_sha256"] = (
                    audit._sha256(base_best_path), best_hash
                )
            elif reference_episode in entry_by_episode:
                reference_path = root / entry_by_episode[reference_episode][
                    "relative_path"
                ]
                base_best_payload = torch.load(
                    base_best_path, map_location="cpu", weights_only=False
                )
                reference_payload = torch.load(
                    reference_path, map_location="cpu", weights_only=False
                )
                if not isinstance(base_best_payload, Mapping) or not isinstance(
                    reference_payload, Mapping
                ) or not _same_policy_weights(
                    base_best_payload, reference_payload
                ):
                    mismatches["v1_1_reference_policy_weights"] = (
                        "different", "identical"
                    )
        final_latest_path = root / BASE_STATE_DIRECTORY / "latest.pth"
        final_entry = entry_by_episode.get(base.FROZEN_TOTAL_EPISODES)
        if not final_latest_path.is_file():
            mismatches["base_final_latest_file"] = (False, True)
        elif final_entry is not None and audit._sha256(
            final_latest_path
        ) != final_entry.get("source_latest_sha256"):
            mismatches["base_final_latest_sha256"] = (
                audit._sha256(final_latest_path),
                final_entry.get("source_latest_sha256"),
            )
    elif any(
        value is not None
        for value in (reference_episode, reference_hash, best_hash)
    ):
        mismatches["premature_v1_1_reference"] = (
            (reference_episode, reference_hash, best_hash),
            (None, None, None),
        )
    if manifest.get("fresh_a_reference_checkpoint_episode") is not None or (
        manifest.get("fresh_a_reference_checkpoint_sha256") is not None
    ):
        mismatches["fresh_a_must_not_redefine_reference"] = (
            (
                manifest.get("fresh_a_reference_checkpoint_episode"),
                manifest.get("fresh_a_reference_checkpoint_sha256"),
            ),
            (None, None),
        )
    if mismatches:
        raise ValueError(
            f"prospective snapshot manifest mismatch: {mismatches!r}"
        )


def _prepare_root(runtime, contract: dict) -> dict:
    root = Path(runtime.output_dir)
    contract_path = root / CONTRACT_FILENAME
    manifest_path = root / SNAPSHOT_MANIFEST_FILENAME
    expected_contract_document = {
        "prospective_training_protocol": PROSPECTIVE_TRAINING_PROTOCOL,
        "prospective_method_version": PROSPECTIVE_METHOD_VERSION,
        "prospective_training_contract_sha256": audit._contract_hash(contract),
        "contract": contract,
    }
    if root.exists() and any(root.iterdir()):
        if not runtime.resume_existing:
            raise FileExistsError(
                "fresh prospective VCG-Dense training refuses a nonempty "
                "output directory; pass --resume-existing or choose a new one"
            )
        if not contract_path.is_file():
            raise ValueError(
                "prospective resume requires its authenticated root contract"
            )
        contract_document = _load_json(
            contract_path, label="prospective training contract"
        )
        if contract_document != audit._json_safe(expected_contract_document):
            raise ValueError("prospective training contract mismatch")
        if not manifest_path.is_file():
            # The only recoverable initialization crash is the atomic gap
            # after committing the exact contract and before committing the
            # empty manifest.  Any other artifact proves work may have begun.
            artifact_names = sorted(path.name for path in root.iterdir())
            if artifact_names != [CONTRACT_FILENAME]:
                raise ValueError(
                    "prospective resume refuses non-pristine contract-only "
                    f"recovery state: {artifact_names!r}"
                )
            recovered = _initial_manifest(contract)
            audit._atomic_json_save(recovered, manifest_path)
            return audit._json_safe(recovered)
        manifest = _load_json(
            manifest_path, label="prospective snapshot manifest"
        )
        _validate_manifest(root, manifest, contract)
        return manifest

    if runtime.resume_existing and root.exists() and not any(root.iterdir()):
        raise ValueError(
            "--resume-existing cannot authenticate an empty output directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    audit._atomic_json_save(expected_contract_document, contract_path)
    manifest = _initial_manifest(contract)
    audit._atomic_json_save(manifest, manifest_path)
    return audit._json_safe(manifest)


def _base_completed(root: Path, base_contract: dict) -> int:
    latest_path = root / BASE_STATE_DIRECTORY / "latest.pth"
    if not latest_path.exists():
        return 0
    payload = torch.load(latest_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("base latest checkpoint must be a mapping")
    base._validate_resume(payload, base_contract)
    return int(payload["completed_training_episodes"])


def _commit_snapshot(
    runtime,
    manifest: dict,
    *,
    episode: int,
    prospective_contract: dict,
    base_contract: dict,
) -> dict:
    root = Path(runtime.output_dir)
    latest_path = root / BASE_STATE_DIRECTORY / "latest.pth"
    source = torch.load(latest_path, map_location="cpu", weights_only=False)
    if not isinstance(source, Mapping):
        raise ValueError("base latest checkpoint must be a mapping")
    base._validate_resume(source, base_contract)
    if int(source["completed_training_episodes"]) != int(episode):
        raise ValueError(
            "snapshot source/latest episode mismatch: "
            f"{source.get('completed_training_episodes')} != {episode}"
        )
    source_sha256 = audit._sha256(latest_path)
    validation_record = _validation_record(source, episode)
    entry = _entry_from_snapshot(
        root,
        episode=episode,
        source_latest_sha256=source_sha256,
        prospective_contract=prospective_contract,
        validation_record=validation_record,
    )
    entries = list(manifest["entries"])
    if len(entries) >= 1 and int(entries[-1]["episode"]) >= episode:
        existing = {
            int(item["episode"]): item for item in entries
        }.get(int(episode))
        if existing != audit._json_safe(entry):
            raise ValueError(
                f"immutable manifest entry mismatch at episode {episode}"
            )
        return manifest
    expected_next = SNAPSHOT_EPISODES[len(entries)]
    if int(episode) != int(expected_next):
        raise ValueError(
            f"snapshot sequence gap: expected {expected_next}, got {episode}"
        )
    entries.append(entry)
    manifest = dict(manifest)
    manifest["entries"] = entries
    manifest["completed_training_episodes"] = int(episode)
    complete = int(episode) == base.FROZEN_TOTAL_EPISODES
    manifest["candidate_pool_complete"] = complete
    if complete:
        # Authenticate that the exact base recipe did finish and finalize its
        # historical selector, without importing that selector's decision
        # into the prospective candidate-pool manifest.
        base_best_path = root / BASE_STATE_DIRECTORY / "best.pth"
        if not base_best_path.is_file():
            raise ValueError("complete base run has no finalized best.pth")
        base_best_hash = audit._sha256(base_best_path)
        if source.get("best_checkpoint_sha256") != base_best_hash:
            raise ValueError("base best.pth SHA-256 does not match latest")
        best_record = source.get("best_validation_record")
        if not isinstance(best_record, Mapping):
            raise ValueError("complete base run has no V1.1 reference record")
        reference_episode = int(best_record["checkpoint_episode"])
        entry_by_episode = {
            int(item["episode"]): item for item in entries
        }
        if reference_episode not in entry_by_episode:
            raise ValueError("V1.1 reference episode is absent from snapshot pool")
        reference_path = root / entry_by_episode[reference_episode][
            "relative_path"
        ]
        base_best = torch.load(
            base_best_path, map_location="cpu", weights_only=False
        )
        reference_snapshot = torch.load(
            reference_path, map_location="cpu", weights_only=False
        )
        if not isinstance(base_best, Mapping) or not isinstance(
            reference_snapshot, Mapping
        ):
            raise ValueError("V1.1 reference artifacts must be mappings")
        if not _same_policy_weights(base_best, reference_snapshot):
            raise ValueError(
                "V1.1 finalized best weights do not match reference snapshot"
            )
        manifest["v1_1_reference_checkpoint_episode"] = reference_episode
        manifest["v1_1_reference_checkpoint_sha256"] = entry_by_episode[
            reference_episode
        ]["sha256"]
        manifest["v1_1_best_checkpoint_sha256"] = base_best_hash
    audit._atomic_json_save(
        manifest, root / SNAPSHOT_MANIFEST_FILENAME
    )
    manifest = audit._json_safe(manifest)
    _validate_manifest(root, manifest, prospective_contract)
    return manifest


def _run(runtime, *, device: torch.device) -> dict:
    prospective_contract, base_contract, train_seed_base = (
        _prospective_contract(runtime)
    )
    manifest = _prepare_root(runtime, prospective_contract)
    root = Path(runtime.output_dir)
    completed = _base_completed(root, base_contract)
    manifested = int(manifest["completed_training_episodes"])
    if completed not in (manifested, manifested + base.FROZEN_EVAL_EVERY):
        raise ValueError(
            "base/manifest progress mismatch is not an atomic one-snapshot "
            f"recovery gap: base={completed}, manifest={manifested}"
        )
    if completed == manifested + base.FROZEN_EVAL_EVERY:
        manifest = _commit_snapshot(
            runtime,
            manifest,
            episode=completed,
            prospective_contract=prospective_contract,
            base_contract=base_contract,
        )

    stop_at = (
        base.FROZEN_TOTAL_EPISODES
        if runtime.stop_after_episode is None
        else int(runtime.stop_after_episode)
    )
    if int(manifest["completed_training_episodes"]) > stop_at:
        raise ValueError("resume state is beyond --stop-after-episode")

    while int(manifest["completed_training_episodes"]) < stop_at:
        episode = SNAPSHOT_EPISODES[len(manifest["entries"])]
        base_runtime = _base_runtime(runtime, stop_after_episode=episode)
        args = base._frozen_audit_args(base_runtime)
        args.resume_existing = bool(
            (root / BASE_STATE_DIRECTORY / "training-contract.json").exists()
        )
        authenticated_seed_base = base._assert_frozen_recipe(args)
        if authenticated_seed_base != train_seed_base:
            raise ValueError("prospective/base training seed authentication drift")
        base._run(args, device=device, train_seed_base=train_seed_base)
        manifest = _commit_snapshot(
            runtime,
            manifest,
            episode=episode,
            prospective_contract=prospective_contract,
            base_contract=base_contract,
        )
        print(
            f"Prospective snapshot {episode:04d}/0500 | "
            f"sha256={manifest['entries'][-1]['sha256']}",
            flush=True,
        )

    return {
        "status": (
            "complete" if manifest["candidate_pool_complete"] else "paused"
        ),
        "prospective_training_protocol": PROSPECTIVE_TRAINING_PROTOCOL,
        "prospective_method_version": PROSPECTIVE_METHOD_VERSION,
        "model_seed": int(runtime.model_seed),
        "completed_training_episodes": int(
            manifest["completed_training_episodes"]
        ),
        "snapshot_count": len(manifest["entries"]),
        "candidate_pool_complete": bool(manifest["candidate_pool_complete"]),
        "candidate_selection_performed": False,
        "deployment_checkpoint_eligible": False,
        "v1_1_reference_checkpoint_episode": manifest[
            "v1_1_reference_checkpoint_episode"
        ],
        "snapshot_manifest": str(
            (root / SNAPSHOT_MANIFEST_FILENAME).resolve()
        ),
        "base_training_summary": str(
            (root / BASE_STATE_DIRECTORY / "training-summary.json").resolve()
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> dict:
    runtime = _build_parser().parse_args(argv)
    if runtime.log_every <= 0:
        raise ValueError("--log-every must be positive")
    device = resolve_device(runtime.device)
    result = _run(runtime, device=device)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
