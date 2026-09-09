#!/usr/bin/env python3
"""E4: certified-frontier ranking ablation on a prospective 92k panel.

Every arm receives the same exact-SAFE candidate frontier and uses the same
recovery-witness guard, normalized mode aggregation, deterministic tie breaks,
and macro executor.  Only the scalar merit assigned to each SAFE candidate is
changed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev
import sys
from typing import Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from PSLAP.viability_candidates import ViabilityActionType, ViabilityMode
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_conditioned_handling_seed0_85k as conditioned_seed0
import run_vcg_v11_nested_handling_pilot as pilot
from viability_graph_hierarchy import ID_TO_MODE, ViabilityGraphDecision
from viability_graph_preference_conditioned import select_hierarchical_index


PROTOCOL = "vcg_conditioned_e04_certified_frontier_ranking_ablation_92k_v1"
SCHEMA_VERSION = 1
MODEL_SEEDS = final90.MODEL_SEEDS
INSTANCE_SEEDS = tuple(range(92_000, 92_030))
PILOT_INSTANCE_COUNT = 5
CONDITIONED_LAMBDAS = (0.05, 0.10, 0.20)
RANDOM_RANKING_SEEDS = (4_204_001, 4_204_002, 4_204_003, 4_204_004, 4_204_005)
EXPECTED_BLOCKS = final90.EXPECTED_BLOCKS

RANDOM_SAFE = "random_safe"
HEURISTIC_SAFE = "rank_urgency_distance_heuristic_safe"
QOP_SAFE = "frozen_vcg1_1_qop"
CONDITIONED_SAFE = "conditioned_vcg"

CONTRACT_NAME = "e04-contract.json"
MANIFEST_NAME = "episode-instance-manifest.json"
REPORT_NAME = "e04-report.json"
TABLE_NAME = "e04-results-table.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vcg-conditioned-e04-ranking-ablation-92k"


class E4Error(RuntimeError):
    pass


def _canonical_bytes(value: Mapping) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: Mapping, *, hash_field: Optional[str] = None) -> str:
    payload = dict(value)
    if hash_field is not None:
        payload.pop(hash_field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _with_hash(value: Mapping, field: str) -> dict:
    result = dict(value)
    result[field] = _digest(result)
    return result


def _sha256(path: Path) -> str:
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise E4Error(f"missing canonical file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict:
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise E4Error(f"missing canonical {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise E4Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise E4Error(f"{label} must contain an object")
    return value


def _verify_hash(value: Mapping, field: str, *, label: str) -> None:
    if value.get(field) != _digest(value, hash_field=field):
        raise E4Error(f"{label} self hash mismatch")


def _atomic_json(path: Path, value: Mapping) -> None:
    final90._atomic_json(path, value)


def _source_contract(project_root: Path) -> dict:
    authenticated = final90._authenticate_inputs(project_root)["conditioned"]
    source_paths = (
        Path(__file__).resolve(),
        project_root / "viability_graph_hierarchy.py",
        project_root / "vcg_v11_conditioned_handling.py",
        project_root / "run_vcg_conditioned_final_comparison_90k.py",
        project_root / "run_vcg_v11_nested_handling_pilot.py",
    )
    return {
        "conditioned_terminal_sha256": {
            str(seed): authenticated["terminal_sha256"][seed]
            for seed in MODEL_SEEDS
        },
        "source_sha256": {
            str(path.relative_to(project_root)): _sha256(path)
            for path in source_paths
        },
    }


def _contract(project_root: Path) -> dict:
    source = _source_contract(project_root)
    semantic = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "scientific_question": "how_much_does_ranking_matter_inside_the_same_certified_frontier",
        "instance_seeds": list(INSTANCE_SEEDS),
        "model_seeds": list(MODEL_SEEDS),
        "conditioned_lambdas": list(CONDITIONED_LAMBDAS),
        "random_ranking_seeds": list(RANDOM_RANKING_SEEDS),
        "ranking_signals": [
            RANDOM_SAFE,
            HEURISTIC_SAFE,
            QOP_SAFE,
            CONDITIONED_SAFE,
        ],
        "random_score_contract": (
            "sha256(ranking_seed,episode_instance_id,decision_epoch,candidate_key)_uniform_v1"
        ),
        "heuristic_score_contract": (
            "10*exact_rank_delta-clipped_remaining_time/20-macro_steps/25+action_offset_v1"
        ),
        "carrier_for_nonlearned_rankers": "model_seed_0_configuration_only",
        "held_fixed": [
            "exact_safe_candidate_source",
            "candidate_generation",
            "recovery_witness_liveness_guard",
            "normalized_mode_logmeanexp_aggregation",
            "candidate_and_mode_tie_breaking",
            "macro_execution",
            "episode_instances",
        ],
        "learning_or_checkpoint_selection": False,
        "pilot_scope": {
            "model_seed": 0,
            "instance_count": PILOT_INSTANCE_COUNT,
            "random_ranking_seed_count": 1,
            "interpretation": "debugging_and_mechanism_check_only",
            "scientific_results_do_not_authorize_retuning": True,
        },
        **source,
    }
    return _with_hash(semantic, "contract_sha256")


def _contract_path(output_dir: Path) -> Path:
    return output_dir / CONTRACT_NAME


def _instance_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "episode-instances" / f"seed-{seed}.json"


def prepare(project_root: Path, output_dir: Path) -> dict:
    output_dir = output_dir.absolute()
    if output_dir.is_symlink():
        raise E4Error("output directory must not be a symlink")
    expected = _contract(project_root)
    path = _contract_path(output_dir)
    if path.is_file():
        observed = _load_json(path, label="E4 contract")
        _verify_hash(observed, "contract_sha256", label="E4 contract")
        if observed != expected:
            raise E4Error("E4 contract, frozen inputs, or sources changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise E4Error("nonempty E4 output has no contract")
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, expected)

    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = authenticate_manifest(project_root, output_dir)
    else:
        env = final90.final86._new_environment()
        records = []
        for index, seed in enumerate(INSTANCE_SEEDS):
            instance = env.sample_episode_instance(seed)
            instance_path = _instance_path(output_dir, seed)
            final90._atomic_text(instance_path, instance.to_json())
            canonical = instance.to_json().encode("utf-8")
            records.append(
                {
                    "seed": seed,
                    "instance_index": index,
                    "relative_path": str(instance_path.relative_to(output_dir)),
                    "raw_sha256": _sha256(instance_path),
                    "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
                    "episode_instance_id": instance.instance_id,
                    "schedule_id": instance.schedule_id,
                }
            )
        manifest = _with_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "contract_sha256": expected["contract_sha256"],
                "instance_count": len(records),
                "instances": records,
                "all_arms_load_exact_serialized_instances": True,
                "sampled_before_any_E4_rollout": True,
            },
            "manifest_sha256",
        )
        _atomic_json(manifest_path, manifest)
    return {
        "status": "prepared",
        "contract": str(path),
        "manifest": str(manifest_path),
        "instances": manifest["instance_count"],
    }


def authenticate_contract(project_root: Path, output_dir: Path) -> dict:
    observed = _load_json(_contract_path(output_dir), label="E4 contract")
    _verify_hash(observed, "contract_sha256", label="E4 contract")
    if observed != _contract(project_root):
        raise E4Error("E4 contract, frozen inputs, or sources changed")
    return observed


def authenticate_manifest(project_root: Path, output_dir: Path) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    manifest = _load_json(output_dir / MANIFEST_NAME, label="E4 instance manifest")
    _verify_hash(manifest, "manifest_sha256", label="E4 instance manifest")
    records = manifest.get("instances")
    if (
        manifest.get("contract_sha256") != contract["contract_sha256"]
        or not isinstance(records, list)
        or [record.get("seed") for record in records] != list(INSTANCE_SEEDS)
    ):
        raise E4Error("E4 instance manifest grid changed")
    env = final90.final86._new_environment()
    for record in records:
        path = output_dir / str(record["relative_path"])
        if _sha256(path) != record["raw_sha256"]:
            raise E4Error("serialized E4 EpisodeInstance bytes changed")
        instance = final90.final86.EpisodeInstance.from_json(
            path.read_text(encoding="utf-8")
        )
        instance.validate_for(env)
        if (
            instance.seed != record["seed"]
            or instance.instance_id != record["episode_instance_id"]
            or instance.schedule_id != record["schedule_id"]
            or hashlib.sha256(instance.to_json().encode("utf-8")).hexdigest()
            != record["canonical_sha256"]
        ):
            raise E4Error("serialized E4 EpisodeInstance identity changed")
    return manifest


def _load_instance(output_dir: Path, record: Mapping):
    path = output_dir / str(record["relative_path"])
    if _sha256(path) != record["raw_sha256"]:
        raise E4Error("E4 EpisodeInstance changed before rollout")
    return final90.final86.EpisodeInstance.from_json(path.read_text(encoding="utf-8"))


def _hash_score(
    ranking_seed: int,
    episode_instance_id: str,
    decision_epoch: int,
    candidate_key: str,
) -> float:
    payload = (
        f"{int(ranking_seed)}\x1f{episode_instance_id}\x1f"
        f"{int(decision_epoch)}\x1f{candidate_key}"
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(1 << 64)


def _macro_steps(candidate) -> int:
    if candidate.recovery_action is not None:
        return int(candidate.recovery_action.steps)
    if candidate.horizon_steps is not None:
        return int(candidate.horizon_steps)
    if candidate.source is not None and candidate.destination is not None:
        return (
            abs(candidate.source[0] - candidate.destination[0])
            + abs(candidate.source[1] - candidate.destination[1])
            + 2
        )
    return 0


def _heuristic_score(snapshot, candidate) -> float:
    """Frozen rank/urgency/distance score using only current public state."""

    rank_delta = float(candidate.rank_delta or 0)
    block = (
        None
        if candidate.target_label is None
        else snapshot.recovery_state.block(candidate.target_label)
    )
    remaining = 0.0 if block is None else float(block.remaining_time)
    urgency = -max(-100.0, min(100.0, remaining)) / 20.0
    action_offset = {
        ViabilityActionType.ACCEPT: 0.0,
        ViabilityActionType.DELIVER: 0.50,
        ViabilityActionType.RECONFIGURE: 0.25,
        ViabilityActionType.DEFER: -0.25,
    }[candidate.action_type]
    return (
        10.0 * rank_delta
        + urgency
        - float(_macro_steps(candidate)) / 25.0
        + action_offset
    )


class SafeMeritAgent:
    """Replace only SAFE-candidate merits; preserve the base deployment stack."""

    def __init__(self, base_agent, *, ranking: str, ranking_seed: Optional[int] = None):
        if ranking not in (RANDOM_SAFE, HEURISTIC_SAFE):
            raise ValueError("unsupported nonlearned ranking")
        if ranking == RANDOM_SAFE and ranking_seed is None:
            raise ValueError("random-safe requires a declared ranking seed")
        self.base_agent = base_agent
        self.config = base_agent.config
        self.ranking = ranking
        self.ranking_seed = None if ranking_seed is None else int(ranking_seed)

    def reset_episode_state(self):
        return self.base_agent.reset_episode_state()

    on_episode_reset = reset_episode_state

    def _scores(self, snapshot, prepared) -> torch.Tensor:
        candidates = tuple(
            snapshot.candidates[index] for index in prepared.source_indices
        )
        if self.ranking == RANDOM_SAFE:
            values = [
                _hash_score(
                    self.ranking_seed,
                    str(snapshot.episode_instance_id),
                    int(snapshot.decision_epoch),
                    record.key,
                )
                for record in prepared.records
            ]
        else:
            values = [_heuristic_score(snapshot, candidate) for candidate in candidates]
        return torch.as_tensor(values, dtype=torch.float32, device=self.base_agent.device)

    def select(self, snapshot, *, training=False, epsilon=0.0):
        if training or float(epsilon) != 0.0:
            raise E4Error("E4 ranking arms are deterministic frozen deployments")
        prepared, liveness_forced = self.base_agent._admissible_prepared(snapshot)
        if not prepared.records:
            raise E4Error("E4 ranker received an empty exact-SAFE frontier")
        merits = self._scores(snapshot, prepared)
        hierarchy = select_hierarchical_index(
            merits,
            prepared.mode_ids,
            self.base_agent.within_temperatures,
            candidate_keys=tuple(record.key for record in prepared.records),
        )
        selected_index = 0 if liveness_forced else hierarchy.selected_index
        source = (
            "exact_recovery_witness_guard"
            if liveness_forced
            else f"e04_{self.ranking}_same_hierarchical_selector"
        )
        source_index = prepared.source_indices[selected_index]
        candidate = snapshot.candidates[source_index]
        exact_rank_progress = bool(
            snapshot.audit.recovery_rank_exact
            and candidate.mode is ViabilityMode.RECOVER
            and candidate.rank_delta is not None
            and candidate.rank_delta > 0
        )
        decision = ViabilityGraphDecision(
            candidate=candidate,
            record=prepared.records[selected_index],
            prepared_snapshot=prepared,
            q_values=tuple(float(value) for value in merits.detach().cpu()),
            mode_values=tuple(
                (ID_TO_MODE[mode], value) for mode, value in hierarchy.mode_values
            ),
            explored=False,
            selection_source=source,
            liveness_forced=liveness_forced,
            exact_rank_progress=exact_rank_progress,
        )
        base = self.base_agent
        base.decision_count += 1
        base.mode_decisions[candidate.mode.value] += 1
        base.action_decisions[candidate.action_type.value] += 1
        base.selection_sources[source] += 1
        base.safe_candidates_scored += len(prepared.records)
        base.interface_rejections += prepared.interface_rejection_count
        base.exact_rejections_seen += int(snapshot.audit.fail_closed_rejection_count)
        return decision

    def observe_outcome(self, decision, *, next_snapshot, done):
        return self.base_agent.observe_outcome(
            decision, next_snapshot=next_snapshot, done=done
        )


def _identity(record: Mapping) -> dict:
    return {
        "instance_seed": int(record["seed"]),
        "instance_index": int(record["instance_index"]),
        "episode_instance_id": record["episode_instance_id"],
        "schedule_id": record["schedule_id"],
        "episode_instance_sha256": record["canonical_sha256"],
    }


def _row(raw: Mapping, compact: Mapping, *, identity: Mapping, spec: Mapping) -> dict:
    strict = bool(compact["strict_safe_complete"])
    timing = final90._timing(raw["delivery_deviations"]) if strict else {
        "mean_signed_deviation": None,
        "mean_absolute_error": None,
        "mean_tardiness": None,
        "mean_earliness": None,
        "within_target_window_rate": None,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        **dict(identity),
        **dict(spec),
        "strict_safe_complete": strict,
        "dense_return": float(compact["dense_return"]) if strict else None,
        **timing,
        "steps": int(compact["steps"]) if strict else None,
        "physical_storage_relocations": (
            int(compact["physical_rehandles"]) if strict else None
        ),
        "physical_rehandles_per_100_required_deliveries": (
            float(compact["physical_rehandles_per_100"]) if strict else None
        ),
        "required_deliveries": EXPECTED_BLOCKS,
        "all_selected_candidates_exact_safe": bool(
            raw.get("complete_frontier_exactly_verified", False)
        ),
        "method_failure_reason": raw.get("method_failure_reason"),
        "behavior_digest": raw.get("behavior_digest"),
        "evaluation_learning": False,
    }


def _failed_row(error: Exception, *, identity: Mapping, spec: Mapping) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        **dict(identity),
        **dict(spec),
        "strict_safe_complete": False,
        "dense_return": None,
        "mean_signed_deviation": None,
        "mean_absolute_error": None,
        "mean_tardiness": None,
        "mean_earliness": None,
        "within_target_window_rate": None,
        "steps": None,
        "physical_storage_relocations": None,
        "physical_rehandles_per_100_required_deliveries": None,
        "required_deliveries": EXPECTED_BLOCKS,
        "all_selected_candidates_exact_safe": None,
        "method_failure_reason": f"{type(error).__name__}: {error}",
        "behavior_digest": None,
        "evaluation_learning": False,
    }


def _spec_token(spec: Mapping) -> str:
    parts = [str(spec["ranking_signal"])]
    if spec.get("model_seed") is not None:
        parts.append(f"model-{int(spec['model_seed'])}")
    if spec.get("preference_lambda") is not None:
        parts.append(f"lambda-{float(spec['preference_lambda']):.3f}")
    if spec.get("ranking_seed") is not None:
        parts.append(f"rankseed-{int(spec['ranking_seed'])}")
    return "__".join(parts)


def _ledger_path(output_dir: Path, spec: Mapping, instance_seed: int) -> Path:
    return output_dir / "run-ledger" / _spec_token(spec) / f"instance-{instance_seed}.json"


def _run_specs(
    *, selected_seed: Optional[int], pilot: bool
) -> tuple[dict, ...]:
    learned_seeds = MODEL_SEEDS if selected_seed is None else (selected_seed,)
    random_seeds = RANDOM_RANKING_SEEDS[:1] if pilot else RANDOM_RANKING_SEEDS
    specs = [
        {
            "ranking_signal": RANDOM_SAFE,
            "model_seed": None,
            "preference_lambda": None,
            "ranking_seed": ranking_seed,
        }
        for ranking_seed in random_seeds
    ]
    specs.append(
        {
            "ranking_signal": HEURISTIC_SAFE,
            "model_seed": None,
            "preference_lambda": None,
            "ranking_seed": None,
        }
    )
    for model_seed in learned_seeds:
        specs.append(
            {
                "ranking_signal": QOP_SAFE,
                "model_seed": model_seed,
                "preference_lambda": 0.0,
                "ranking_seed": None,
            }
        )
        specs.extend(
            {
                "ranking_signal": CONDITIONED_SAFE,
                "model_seed": model_seed,
                "preference_lambda": value,
                "ranking_seed": None,
            }
            for value in CONDITIONED_LAMBDAS
        )
    return tuple(specs)


def _factory(project_root: Path, auth: Mapping, spec: Mapping, arm, device):
    ranking = spec["ranking_signal"]

    def factory(base):
        if ranking == RANDOM_SAFE:
            return SafeMeritAgent(
                base, ranking=RANDOM_SAFE, ranking_seed=spec["ranking_seed"]
            )
        if ranking == HEURISTIC_SAFE:
            return SafeMeritAgent(base, ranking=HEURISTIC_SAFE)
        if ranking == QOP_SAFE:
            return base
        agent = final90._load_conditioned_agent(
            project_root,
            auth,
            model_seed=int(spec["model_seed"]),
            base=base,
            device=device,
        )
        agent.set_epsilon(0.0)
        return conditioned_seed0._FixedLambdaAgent(
            agent, float(spec["preference_lambda"])
        )

    return factory


def run(
    project_root: Path,
    output_dir: Path,
    *,
    selected_seed: Optional[int],
    instance_limit: Optional[int],
    pilot_mode: bool,
    device_name: str,
) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    manifest = authenticate_manifest(project_root, output_dir)
    if selected_seed is not None and selected_seed not in MODEL_SEEDS:
        raise E4Error("model seed must be 0, 1, or 2")
    if pilot_mode and selected_seed != 0:
        raise E4Error("the declared E4 pilot is seed 0 only")
    device = pilot._device(device_name)
    if device.type != "cpu":
        raise E4Error("E4 is frozen to CPU execution")
    auth = final90._authenticate_inputs(project_root)
    conditioned = auth["conditioned"]
    records = manifest["instances"]
    if instance_limit is not None:
        if instance_limit < 1 or instance_limit > len(records):
            raise E4Error("invalid E4 instance limit")
        records = records[:instance_limit]

    specs = _run_specs(selected_seed=selected_seed, pilot=pilot_mode)
    complete = safe = 0
    for spec in specs:
        carrier_seed = (
            0 if spec["model_seed"] is None else int(spec["model_seed"])
        )
        arm = conditioned["inputs"]["arms"][carrier_seed]
        for record in records:
            identity = _identity(record)
            path = _ledger_path(output_dir, spec, identity["instance_seed"])
            if path.is_file():
                ledger = _load_json(path, label="E4 rollout ledger")
                _verify_hash(ledger, "ledger_sha256", label="E4 rollout ledger")
                if (
                    ledger.get("contract_sha256") != contract["contract_sha256"]
                    or ledger.get("manifest_sha256") != manifest["manifest_sha256"]
                    or ledger.get("spec") != spec
                ):
                    raise E4Error("E4 rollout ledger binding changed")
                row = ledger["run"]
            else:
                instance = _load_instance(output_dir, record)
                try:
                    raw = pilot._run_raw(
                        arm,
                        instance,
                        device=device,
                        wrapper_factory=_factory(
                            project_root, auth, spec, arm, device
                        ),
                    )
                    compact = pilot._compact_row(raw, instance)
                    row = _row(raw, compact, identity=identity, spec=spec)
                except Exception as error:  # retain failed cells prospectively
                    row = _failed_row(error, identity=identity, spec=spec)
                ledger = _with_hash(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "protocol": PROTOCOL,
                        "contract_sha256": contract["contract_sha256"],
                        "manifest_sha256": manifest["manifest_sha256"],
                        "spec": dict(spec),
                        "run": row,
                    },
                    "ledger_sha256",
                )
                _atomic_json(path, ledger)
            complete += 1
            safe += int(row["strict_safe_complete"])
            if complete % 10 == 0:
                print(f"E4 {complete}/{len(specs) * len(records)} | safe={safe}", flush=True)
    return {
        "status": "complete",
        "rows": complete,
        "strict_safe_complete_rows": safe,
        "pilot_mode": pilot_mode,
        "model_seeds": list(MODEL_SEEDS if selected_seed is None else (selected_seed,)),
        "instances": len(records),
    }


METRICS = (
    "dense_return",
    "mean_absolute_error",
    "within_target_window_rate",
    "steps",
    "physical_rehandles_per_100_required_deliveries",
)


def _all_ledgers(output_dir: Path) -> list[dict]:
    rows = []
    root = output_dir / "run-ledger"
    if not root.is_dir():
        return rows
    for path in sorted(root.glob("**/instance-*.json")):
        ledger = _load_json(path, label="E4 rollout ledger")
        _verify_hash(ledger, "ledger_sha256", label="E4 rollout ledger")
        rows.append(ledger["run"])
    return rows


def _group_label(row: Mapping) -> str:
    ranking = row["ranking_signal"]
    if ranking == RANDOM_SAFE:
        return RANDOM_SAFE
    if ranking == HEURISTIC_SAFE:
        return HEURISTIC_SAFE
    if ranking == QOP_SAFE:
        return QOP_SAFE
    return f"{CONDITIONED_SAFE}_lambda_{float(row['preference_lambda']):.2f}"


def _summarize(rows: Sequence[Mapping]) -> dict:
    result = {}
    groups = defaultdict(list)
    for row in rows:
        groups[_group_label(row)].append(row)
    for label, group in sorted(groups.items()):
        safe = [row for row in group if row["strict_safe_complete"]]
        summary = {
            "rows": len(group),
            "strict_safe_complete": len(safe),
            "strict_completion_rate": len(safe) / len(group),
            "complete_case_metrics_suppressed": len(safe) != len(group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in safe]
            summary[metric] = None if len(safe) != len(group) else fmean(values)
            summary[f"{metric}_sd"] = (
                None if len(safe) != len(group) or len(values) < 2 else stdev(values)
            )
        result[label] = summary
    return result


def _table(report: Mapping) -> str:
    lines = [
        "# E4 certified-frontier ranking ablation",
        "",
        "| Ranking signal | Strict complete | Return | MAE | Within ±20 | Steps | Rehandles/100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in report["aggregate"].items():
        def value(name: str, digits: int = 2) -> str:
            observed = item[name]
            return "—" if observed is None else f"{observed:.{digits}f}"
        lines.append(
            f"| {label} | {item['strict_safe_complete']}/{item['rows']} | "
            f"{value('dense_return')} | {value('mean_absolute_error')} | "
            f"{value('within_target_window_rate', 3)} | {value('steps')} | "
            f"{value('physical_rehandles_per_100_required_deliveries')} |"
        )
    lines.extend(
        [
            "",
            "Complete-case performance is suppressed whenever an arm has any failed row.",
            "The pilot is a mechanism check and is not paper evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(project_root: Path, output_dir: Path, *, allow_partial: bool) -> dict:
    contract = authenticate_contract(project_root, output_dir)
    manifest = authenticate_manifest(project_root, output_dir)
    rows = _all_ledgers(output_dir)
    full_expected = (
        len(RANDOM_RANKING_SEEDS) * len(INSTANCE_SEEDS)
        + len(INSTANCE_SEEDS)
        + len(MODEL_SEEDS) * len(INSTANCE_SEEDS)
        + len(MODEL_SEEDS) * len(CONDITIONED_LAMBDAS) * len(INSTANCE_SEEDS)
    )
    if not allow_partial and len(rows) != full_expected:
        raise E4Error(f"E4 grid is incomplete: {len(rows)}/{full_expected} rows")
    report = _with_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "status": "partial" if len(rows) != full_expected else "complete",
            "contract_sha256": contract["contract_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "rows": len(rows),
            "expected_rows": full_expected,
            "paper_evidence": len(rows) == full_expected,
            "aggregate": _summarize(rows),
        },
        "report_sha256",
    )
    _atomic_json(output_dir / REPORT_NAME, report)
    final90._atomic_text(output_dir / TABLE_NAME, _table(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run", "run-all", "analyze")
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--instance-limit", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.command == "prepare":
        result = prepare(project_root, output_dir)
    elif args.command == "run":
        result = run(
            project_root,
            output_dir,
            selected_seed=args.seed,
            instance_limit=args.instance_limit,
            pilot_mode=args.pilot,
            device_name=args.device,
        )
    elif args.command == "run-all":
        result = run(
            project_root,
            output_dir,
            selected_seed=None,
            instance_limit=None,
            pilot_mode=False,
            device_name=args.device,
        )
    else:
        result = analyze(
            project_root, output_dir, allow_partial=args.allow_partial
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
