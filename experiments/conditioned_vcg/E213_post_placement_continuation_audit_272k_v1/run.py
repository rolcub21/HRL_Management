#!/usr/bin/env python3
"""Run E213's post-placement continuation intervention."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import torch

import train_vcg_preference_conditioned as common
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    protocol as e191,
)
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1 import (
    run as e204,
)
from experiments.conditioned_vcg.E204_e191_centered_rehandle_stage_b_263k_v1.centered import (
    detached_e191_features,
)
from experiments.conditioned_vcg.E205_e204_paired_tradeoff_development_264k_v1 import (
    run as e205_run,
)
from experiments.conditioned_vcg.E207_e204_immediate_only_ablation_266k_v1.immediate import (
    from_stage_b_agent,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)

from . import protocol as p


class E213Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E213Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E213 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E213 JSON: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _declaration(contract: dict) -> dict:
    return {
        key: contract[key]
        for key in (
            "method_reference", "method_revision", "claim_level",
            "declared_deviations", "oracle_inputs", "nonconforming_shortcuts",
            "conformance_audit",
        )
    }


def prepare(output: Path) -> dict:
    output = output.resolve()
    contract = p.contract()
    validate_protocol(contract, source=p.PROTOCOL)
    path = output / "contract.json"
    if path.is_file():
        require(_read(path) == contract, "E213 prepared contract changed")
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E213 refuses nonempty uncontracted output",
        )
        _write(path, contract)
    return {
        "status": "prepared",
        "contract_sha256": contract["contract_sha256"],
        "forced_placement_key": p.FORCED_PLACEMENT_KEY,
        "expected_complete_episode_runs": 2,
        "test_panel_opened": False,
    }


def authenticate(output: Path) -> dict:
    output = output.resolve()
    if not (output / "contract.json").is_file():
        prepare(output)
    contract = _read(output / "contract.json")
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    require(p.digest(payload) == contract.get("contract_sha256"), "E213 contract digest differs")
    render_validated_conformance_audit(_declaration(contract))
    for relative, expected in contract["source_sha256"].items():
        require(p.sha256(p.ROOT / relative) == expected, f"E213 source changed: {relative}")
    for relative, expected in contract["source_artifacts"].items():
        require(p.sha256(p.ROOT / relative) == expected, f"E213 source artifact changed: {relative}")
    require(
        _read(p.SOURCE_E212 / "contract.json")["contract_sha256"]
        == contract["source_e212_contract_sha256"],
        "E213 E212 contract identity differs",
    )
    return contract


def _state_signature(decision) -> dict:
    records = tuple(decision.prepared_snapshot.records)
    require(bool(records), "E213 cannot sign an empty prepared frontier")
    state_reprs = {repr(record.current_state) for record in records}
    require(len(state_reprs) == 1, "E213 prepared records disagree on current state")
    state_repr = next(iter(state_reprs))
    return {
        "decision_epoch": int(decision.prepared_snapshot.decision_epoch),
        "state_sha256": hashlib.sha256(state_repr.encode("utf-8")).hexdigest(),
        "prepared_keys": [str(record.key) for record in records],
    }


def _forced_decision(controller, natural, snapshot, forced_key: str):
    prepared = natural.prepared_snapshot
    indices = [
        index for index, record in enumerate(prepared.records)
        if str(record.key) == str(forced_key)
    ]
    require(len(indices) == 1, f"forced E213 key is not uniquely certified: {forced_key}")
    selected_index = indices[0]
    source_index = prepared.source_indices[selected_index]
    candidate = snapshot.candidates[source_index]
    record = prepared.records[selected_index]
    require(str(candidate.key) == str(record.key) == str(forced_key), "E213 forced binding differs")
    require(not natural.liveness_forced or natural.record.key == record.key, "E213 cannot override a forced liveness action")
    require(
        candidate.action_type is natural.candidate.action_type
        and candidate.mode is natural.candidate.mode,
        "E213 prefix intervention would change action type or temporal mode",
    )
    features = detached_e191_features(controller.base_agent.Q_local, prepared.records)
    exact_rank_progress = bool(
        snapshot.audit.recovery_rank_exact
        and candidate.mode.value == "recover"
        and candidate.rank_delta is not None
        and candidate.rank_delta > 0
    )
    forced_base = replace(
        natural.base_decision,
        candidate=candidate,
        record=record,
        explored=False,
        selection_source="e213_forced_observed_prefix",
        liveness_forced=False,
        exact_rank_progress=exact_rank_progress,
    )
    return replace(
        natural,
        base_decision=forced_base,
        chosen_feature=features[selected_index].detach().cpu().clone(),
        chosen_immediate=int(candidate.action_type.value == "reconfigure"),
    )


class PrefixInterventionAgent:
    def __init__(self, controller, *, forced_prefix: tuple[str, ...] | None = None):
        self.controller = controller
        self.forced_prefix = forced_prefix
        self.chosen_keys: list[str] = []
        self.natural_keys: list[str] = []
        self.forced_records: list[dict] = []
        self.handoff_signature: dict | None = None

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def select(self, snapshot, **kwargs):
        natural = self.controller.select(snapshot, **kwargs)
        index = len(self.chosen_keys)
        if index == p.DIVERGENCE_INDEX + 1:
            self.handoff_signature = _state_signature(natural)
        forced_key = None
        if self.forced_prefix is not None and index < len(self.forced_prefix):
            forced_key = self.forced_prefix[index]
        elif self.forced_prefix is None and index == p.DIVERGENCE_INDEX:
            forced_key = p.FORCED_PLACEMENT_KEY
        chosen = natural
        if forced_key is not None and str(natural.record.key) != str(forced_key):
            chosen = _forced_decision(self.controller, natural, snapshot, forced_key)
            self.forced_records.append({
                "decision_index": index,
                "natural_key": str(natural.record.key),
                "forced_key": str(forced_key),
                "action_type": str(chosen.candidate.action_type.value),
            })
        self.natural_keys.append(str(natural.record.key))
        self.chosen_keys.append(str(chosen.record.key))
        return chosen

    def audit(self) -> dict:
        return {
            "chosen_prefix": self.chosen_keys[: p.DIVERGENCE_INDEX + 1],
            "natural_prefix": self.natural_keys[: p.DIVERGENCE_INDEX + 1],
            "forced_records": self.forced_records,
            "handoff_signature": self.handoff_signature,
        }


def _episode(agent: PrefixInterventionAgent, search, liveness, value: float, label: str):
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    agent.base_agent.bind_context_environment(
        env,
        required_jobs=e191.scenario_for(p.SCENARIO).total_jobs,
        max_steps=e191.MAX_STEPS,
    )
    started = perf_counter()
    with e205_run.heartbeat(f"E213 {label}"), ExitStack() as stack:
        e204._runtime_stack(stack)
        result = common.run_preference_episode(
            agent,
            env,
            instance_seed=p.HEAVY_INSTANCE_SEED,
            preference_lambda=float(value),
            preference_schedule_record={
                "behavior_lambda": float(value),
                "lambda_center": 10.0,
                "diagnostic": p.PROTOCOL,
                "forced_prefix_through_decision_index": p.DIVERGENCE_INDEX,
                "fixed_for_complete_episode": True,
            },
            max_steps=e191.MAX_STEPS,
            search_config=search,
            liveness_rule=liveness,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0,
            updates_per_macro=1,
        )
    samples = agent.pop_completed_samples()
    require(bool(result["strict_method_success"]), f"E213 {label} episode failed: {result}")
    require(len(samples) == int(result["macro_decisions"]), "E213 sample count differs")
    require(agent.handoff_signature is not None, "E213 did not reach the post-placement handoff")
    metrics = env.e187_metrics()
    components = {key: float(item) for key, item in metrics["components"].items()}
    rehandles = int(result["physical_rehandles"])
    j10 = float(metrics["J"])
    return {
        "status": "complete",
        "arm": label,
        "strict_complete": True,
        "lambda": float(value),
        "instance_seed": p.HEAVY_INSTANCE_SEED,
        "instance_id": env.current_episode_instance.instance_id,
        "J10": j10,
        "J_lambda": j10 + (float(value) - 10.0) * rehandles,
        "physical_rehandles": rehandles,
        "timing_cost_2T_plus_E": (
            2.0 * components["storage_tardiness_accrual"]
            + components["storage_earliness_at_delivery"]
        ),
        "admission_wait_cost_0p5W": 0.5 * components["admission_wait_accrual"],
        "equipment_cost_0p1L": 0.1 * components["equipment_work"],
        "steps": int(result["steps"]),
        "macro_decisions": int(result["macro_decisions"]),
        "peak_stored": int(metrics["peak_stored"]),
        "intervention": agent.audit(),
        "wall_seconds": perf_counter() - started,
        "test_panel_opened": False,
    }


def _source_rows() -> dict:
    return {
        "plain_e191_natural": _read(p.SOURCE_E205 / "rows/rho-80/lambda-10.json"),
        "learned_lambda20_natural": _read(p.SOURCE_E205 / "rows/rho-80/lambda-20.json"),
        "immediate_only_lambda20_natural": _read(p.SOURCE_E207 / "rows/rho-80/lambda-20.json"),
    }


def _row_path(output: Path, arm: str) -> Path:
    return output.resolve() / "rows" / f"{arm}.json"


def _run_immediate(output: Path, contract: dict, device_name: str) -> dict:
    path = _row_path(output, "immediate-only-lambda20-after-learned-placement")
    if path.is_file():
        return {**_read(path), "new_episode_run": 0}
    stage_b, search, liveness, base_digest, stage_b_identity, _device = e205_run.load_agent(device_name)
    controller = from_stage_b_agent(stage_b, seed=272_213_700)
    before = e204.state_digest(controller.base_agent.Q_local.state_dict())
    wrapped = PrefixInterventionAgent(controller)
    print("E213 immediate-only continuation start", flush=True)
    row = _episode(wrapped, search, liveness, p.IMMEDIATE_LAMBDA, "immediate_only_lambda20")
    require(e204.state_digest(controller.base_agent.Q_local.state_dict()) == before == base_digest, "E213 immediate continuation mutated E191")
    divergence = _read(p.SOURCE_E212 / "first-divergence.json")
    require(
        wrapped.natural_keys[p.DIVERGENCE_INDEX]
        == divergence["immediate_only_decision"]["chosen_key"],
        "E213 immediate natural divergence choice differs from E212",
    )
    require(wrapped.chosen_keys[p.DIVERGENCE_INDEX] == p.FORCED_PLACEMENT_KEY, "E213 did not force learned placement")
    row.update({
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "model_seed": p.MODEL_SEED,
        "stage_b_terminal_sha256": stage_b_identity["terminal_checkpoint_sha256"],
        "new_episode_run": 1,
        "paper_evidence": False,
    })
    _write(path, row)
    print(f"E213 immediate-only done R={row['physical_rehandles']} J10={row['J10']:.1f}", flush=True)
    return row


def _run_plain(output: Path, contract: dict, device_name: str, forced_prefix: tuple[str, ...], expected_handoff: dict) -> dict:
    path = _row_path(output, "plain-e191-lambda10-after-learned-placement")
    if path.is_file():
        return {**_read(path), "new_episode_run": 0}
    controller, search, liveness, base_digest, stage_b_identity, _device = e205_run.load_agent(device_name)
    before = e204.state_digest(controller.base_agent.Q_local.state_dict())
    wrapped = PrefixInterventionAgent(controller, forced_prefix=forced_prefix)
    print("E213 plain E191 continuation start", flush=True)
    row = _episode(wrapped, search, liveness, p.PLAIN_E191_LAMBDA, "plain_e191_lambda10")
    require(e204.state_digest(controller.base_agent.Q_local.state_dict()) == before == base_digest, "E213 plain continuation mutated E191")
    require(tuple(wrapped.chosen_keys[: len(forced_prefix)]) == forced_prefix, "E213 plain forced prefix differs")
    require(wrapped.handoff_signature == expected_handoff, "E213 post-placement handoff states differ")
    row.update({
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "model_seed": p.MODEL_SEED,
        "stage_b_terminal_sha256": stage_b_identity["terminal_checkpoint_sha256"],
        "new_episode_run": 1,
        "paper_evidence": False,
    })
    _write(path, row)
    print(f"E213 plain E191 done R={row['physical_rehandles']} J10={row['J10']:.1f}", flush=True)
    return row


def _closer(value: int, natural: int, collapse: int) -> str:
    natural_distance = abs(int(value) - int(natural))
    collapse_distance = abs(int(value) - int(collapse))
    if natural_distance < collapse_distance:
        return "natural_controller_baseline"
    if collapse_distance < natural_distance:
        return "learned_lambda20_collapse"
    return "equidistant"


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    immediate = _read(_row_path(output, "immediate-only-lambda20-after-learned-placement"))
    plain = _read(_row_path(output, "plain-e191-lambda10-after-learned-placement"))
    sources = _source_rows()
    natural_plain = int(sources["plain_e191_natural"]["physical_rehandles"])
    natural_immediate = int(sources["immediate_only_lambda20_natural"]["physical_rehandles"])
    collapse = int(sources["learned_lambda20_natural"]["physical_rehandles"])
    immediate_anchor = _closer(immediate["physical_rehandles"], natural_immediate, collapse)
    plain_anchor = _closer(plain["physical_rehandles"], natural_plain, collapse)
    if immediate_anchor == plain_anchor == "learned_lambda20_collapse":
        verdict = "placement_dominated"
    elif immediate_anchor == plain_anchor == "natural_controller_baseline":
        verdict = "subsequent_learned_dial_choices_dominated"
    else:
        verdict = "placement_controller_interaction"
    report = {
        "status": "complete",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "forced_prefix": immediate["intervention"]["chosen_prefix"],
        "post_placement_handoff_signature": immediate["intervention"]["handoff_signature"],
        "natural_baselines": {
            "plain_e191_lambda10_rehandles": natural_plain,
            "immediate_only_lambda20_rehandles": natural_immediate,
            "learned_lambda20_collapse_rehandles": collapse,
        },
        "continuations": {"immediate_only_lambda20": immediate, "plain_e191_lambda10": plain},
        "anchor_classification": {"immediate_only_lambda20": immediate_anchor, "plain_e191_lambda10": plain_anchor},
        "verdict": verdict,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    _write(output.resolve() / "e213-report.json", report)
    audit = render_validated_conformance_audit(_declaration(contract))
    lines = [
        audit,
        "RESULTS",
        "=" * 72,
        "",
        f"Forced prefix through decision 7: {report['forced_prefix']}",
        f"Post-placement state digest: {report['post_placement_handoff_signature']['state_sha256']}",
        "",
        f"Natural anchors: plain E191 R={natural_plain}; immediate-only lambda20 R={natural_immediate}; learned lambda20 collapse R={collapse}.",
        f"After B3->(6,6), immediate-only lambda20: R={immediate['physical_rehandles']}, J10={immediate['J10']:.1f}, timing={immediate['timing_cost_2T_plus_E']:.1f}, steps={immediate['steps']}.",
        f"After B3->(6,6), plain E191 lambda10: R={plain['physical_rehandles']}, J10={plain['J10']:.1f}, timing={plain['timing_cost_2T_plus_E']:.1f}, steps={plain['steps']}.",
        "",
        f"Verdict: {verdict}.",
        "Nearest-anchor classification is predeclared: each continuation is compared with its own natural baseline and the 309-rehandle learned collapse.",
        "",
        "Development diagnostic only; sealed test panel remains closed.",
        "",
    ]
    (output.resolve() / "REPORT.txt").write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "complete",
        "verdict": verdict,
        "immediate_only_after_placement_rehandles": int(immediate["physical_rehandles"]),
        "plain_e191_after_placement_rehandles": int(plain["physical_rehandles"]),
        "report": str(output.resolve() / "REPORT.txt"),
        "test_panel_opened": False,
    }


def run(output: Path, device_name: str) -> dict:
    contract = authenticate(output)
    immediate = _run_immediate(output, contract, device_name)
    forced_prefix = tuple(immediate["intervention"]["chosen_prefix"])
    require(len(forced_prefix) == p.DIVERGENCE_INDEX + 1, "E213 immediate prefix length differs")
    require(forced_prefix[-1] == p.FORCED_PLACEMENT_KEY, "E213 prefix does not end at learned placement")
    plain = _run_plain(
        output,
        contract,
        device_name,
        forced_prefix,
        immediate["intervention"]["handoff_signature"],
    )
    result = analyze(output)
    return {
        **result,
        "new_complete_episode_runs": int(immediate.get("new_episode_run", 0)) + int(plain.get("new_episode_run", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "run":
        result = run(args.output, args.device)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
