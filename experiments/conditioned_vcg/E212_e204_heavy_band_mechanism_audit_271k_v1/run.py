#!/usr/bin/env python3
"""Run E212's frozen first-divergence replay and local lambda sweep."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
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


class E212Error(RuntimeError):
    pass


class DivergenceFound(BaseException):
    def __init__(self, record: dict):
        super().__init__("first controller divergence found")
        self.record = record


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E212Error(message)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E212 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E212 JSON: {path}")
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
        require(_read(path) == contract, "E212 prepared contract changed")
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E212 refuses nonempty uncontracted output",
        )
        _write(path, contract)
    return {
        "status": "prepared",
        "contract_sha256": contract["contract_sha256"],
        "heavy_instance_seed": p.HEAVY_INSTANCE_SEED,
        "trace_lambda": p.TRACE_LAMBDA,
        "sweep_lambdas": list(p.SWEEP_LAMBDAS),
        "new_complete_episode_runs": 4,
        "test_panel_opened": False,
    }


def authenticate(output: Path) -> dict:
    output = output.resolve()
    if not (output / "contract.json").is_file():
        prepare(output)
    contract = _read(output / "contract.json")
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    require(p.digest(payload) == contract.get("contract_sha256"), "E212 contract digest differs")
    render_validated_conformance_audit(_declaration(contract))
    for relative, expected in contract["source_sha256"].items():
        require(p.sha256(p.ROOT / relative) == expected, f"E212 source changed: {relative}")
    for relative, expected in contract["source_artifacts"].items():
        require(p.sha256(p.ROOT / relative) == expected, f"E212 source artifact changed: {relative}")
    require(
        _read(p.SOURCE_E205 / "contract.json")["contract_sha256"]
        == contract["source_e205_contract_sha256"],
        "E212 E205 contract identity differs",
    )
    require(
        _read(p.SOURCE_E207 / "contract.json")["contract_sha256"]
        == contract["source_e207_contract_sha256"],
        "E212 E207 contract identity differs",
    )
    return contract


def _candidate_rows(decision, value: float) -> list[dict]:
    correction = decision.base_decision.q_values
    rows = []
    for index, record in enumerate(decision.prepared_snapshot.records):
        immediate = int(record.action_type == "reconfigure")
        handling = float(decision.handling_values[index])
        rows.append({
            "index": index,
            "key": str(record.key),
            "action_type": str(record.action_type),
            "mode_id": int(record.mode_id),
            "operational_Q10": float(decision.operational_values[index]),
            "immediate_rehandle": immediate,
            "predicted_future_rehandles": handling - immediate,
            "predicted_total_rehandles": handling,
            "corrected_merit": float(correction[index]),
            "selected": bool(record.key == decision.record.key),
        })
    return rows


def _decision_row(decision, value: float, index: int) -> dict:
    return {
        "decision_index": int(index),
        "decision_epoch": int(decision.prepared_snapshot.decision_epoch),
        "lambda": float(value),
        "chosen_key": str(decision.record.key),
        "chosen_action_type": str(decision.record.action_type),
        "chosen_mode_id": int(decision.record.mode_id),
        "liveness_forced": bool(decision.liveness_forced),
        "selection_source": str(decision.selection_source),
        "mode_values": [list(item) for item in decision.base_decision.mode_values],
        "frontier": _candidate_rows(decision, value),
    }


def _attach_trace(agent, value: float) -> list[dict]:
    trace: list[dict] = []
    original = agent.select

    def select(snapshot, **kwargs):
        decision = original(snapshot, **kwargs)
        trace.append(_decision_row(decision, value, len(trace)))
        return decision

    agent.select = select
    return trace


def _episode(agent, search, liveness, value: float, *, capture_trace: bool):
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    agent.base_agent.bind_context_environment(
        env,
        required_jobs=e191.scenario_for(p.SCENARIO).total_jobs,
        max_steps=e191.MAX_STEPS,
    )
    trace = _attach_trace(agent, value) if capture_trace else []
    started = perf_counter()
    with e205_run.heartbeat(f"E212 lambda={value:g}"), ExitStack() as stack:
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
    require(bool(result["strict_method_success"]), f"E212 episode failed: {result}")
    require(len(samples) == int(result["macro_decisions"]), "E212 sample count differs")
    if capture_trace:
        require(len(trace) == len(samples), "E212 trace count differs")
        for row, sample in zip(trace, samples):
            row["realized_immediate_rehandles"] = int(sample.immediate_rehandles)
            row["realized_future_rehandles"] = int(sample.future_rehandles)
            row["realized_total_rehandles"] = int(
                sample.immediate_rehandles + sample.future_rehandles
            )
    metrics = env.e187_metrics()
    components = {key: float(item) for key, item in metrics["components"].items()}
    rehandles = int(result["physical_rehandles"])
    j10 = float(metrics["J"])
    return {
        "status": "complete",
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
        "wall_seconds": perf_counter() - started,
        "test_panel_opened": False,
    }, trace


def _load_learned(device_name: str):
    agent, search, liveness, base_digest, stage_b, _device = e205_run.load_agent(device_name)
    return agent, search, liveness, base_digest, stage_b


def _source_row(value: float) -> dict:
    label = f"{float(value):g}".replace(".", "p")
    return _read(p.SOURCE_E205 / f"rows/rho-80/lambda-{label}.json")


def _sweep_path(output: Path, value: float) -> Path:
    label = f"{float(value):g}".replace(".", "p")
    return output.resolve() / "sweep" / f"lambda-{label}.json"


def _run_sweep_value(output: Path, contract: dict, value: float, device_name: str) -> dict:
    path = _sweep_path(output, value)
    if path.is_file():
        return {**_read(path), "new_episode_run": 0}
    agent, search, liveness, base_digest, stage_b = _load_learned(device_name)
    before = e204.state_digest(agent.base_agent.Q_local.state_dict())
    print(f"E212 sweep start lambda={value:g}", flush=True)
    row, _trace = _episode(agent, search, liveness, value, capture_trace=False)
    require(e204.state_digest(agent.base_agent.Q_local.state_dict()) == before == base_digest, "E212 mutated E191")
    row.update({
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "model_seed": p.MODEL_SEED,
        "stage_b_terminal_sha256": stage_b["terminal_checkpoint_sha256"],
        "new_episode_run": 1,
        "paper_evidence": False,
    })
    _write(path, row)
    print(
        f"E212 sweep done lambda={value:g} R={row['physical_rehandles']} "
        f"Jlambda={row['J_lambda']:.1f} timing={row['timing_cost_2T_plus_E']:.1f}",
        flush=True,
    )
    return row


def _trace_replay(output: Path, contract: dict, device_name: str) -> dict:
    path = output.resolve() / "first-divergence.json"
    if path.is_file():
        return {**_read(path), "new_episode_runs": 0}
    learned, search, liveness, base_digest, stage_b = _load_learned(device_name)
    before = e204.state_digest(learned.base_agent.Q_local.state_dict())
    print("E212 learned lambda=20 trace start", flush=True)
    learned_row, learned_trace = _episode(
        learned, search, liveness, p.TRACE_LAMBDA, capture_trace=True
    )
    require(e204.state_digest(learned.base_agent.Q_local.state_dict()) == before == base_digest, "E212 trace mutated E191")
    source_learned = _source_row(p.TRACE_LAMBDA)
    for key in ("physical_rehandles", "steps", "macro_decisions"):
        require(learned_row[key] == source_learned[key], f"E212 learned replay differs on {key}")
    require(math.isclose(learned_row["J10"], source_learned["J10"], abs_tol=1e-6), "E212 learned replay J differs")

    stage_b_agent, immediate_search, immediate_liveness, immediate_digest, _stage_b = _load_learned(device_name)
    immediate = from_stage_b_agent(stage_b_agent, seed=271_212_700)
    immediate_trace: list[dict] = []
    original = immediate.select

    def select(snapshot, **kwargs):
        decision = original(snapshot, **kwargs)
        row = _decision_row(decision, p.TRACE_LAMBDA, len(immediate_trace))
        immediate_trace.append(row)
        index = row["decision_index"]
        require(index < len(learned_trace), "E212 immediate prefix exceeded learned trace")
        learned_row_at_index = learned_trace[index]
        if row["chosen_key"] != learned_row_at_index["chosen_key"]:
            raise DivergenceFound(row)
        require(
            [item["key"] for item in row["frontier"]]
            == [item["key"] for item in learned_row_at_index["frontier"]],
            "E212 frontiers differed before action divergence",
        )
        return decision

    immediate.select = select
    env = e191.TargetDenseEnv(p.SCENARIO, e191.SPEC)
    immediate.base_agent.bind_context_environment(
        env, required_jobs=e191.scenario_for(p.SCENARIO).total_jobs,
        max_steps=e191.MAX_STEPS,
    )
    divergence = None
    try:
        with ExitStack() as stack:
            e204._runtime_stack(stack)
            common.run_preference_episode(
                immediate, env, instance_seed=p.HEAVY_INSTANCE_SEED,
                preference_lambda=p.TRACE_LAMBDA,
                preference_schedule_record={
                    "behavior_lambda": p.TRACE_LAMBDA,
                    "lambda_center": 10.0,
                    "diagnostic": p.PROTOCOL,
                    "fixed_for_complete_episode": True,
                },
                max_steps=e191.MAX_STEPS, search_config=immediate_search,
                liveness_rule=immediate_liveness, epsilon_start=0.0,
                epsilon_end=0.0, epsilon_decay_decisions=1,
                epsilon_warmup_decisions=0, updates_per_macro=1,
            )
    except DivergenceFound as found:
        divergence = found.record
    require(divergence is not None, "E212 controllers did not diverge")
    require(e204.state_digest(immediate.base_agent.Q_local.state_dict()) == immediate_digest, "E212 immediate prefix mutated E191")

    index = int(divergence["decision_index"])
    learned_decision = learned_trace[index]
    require(
        [item["key"] for item in divergence["frontier"]]
        == [item["key"] for item in learned_decision["frontier"]],
        "E212 first-divergence frontiers differ",
    )
    learned_by_key = {item["key"]: item for item in learned_decision["frontier"]}
    immediate_by_key = {item["key"]: item for item in divergence["frontier"]}
    learned_choice = learned_by_key[learned_decision["chosen_key"]]
    immediate_choice_under_learned_head = learned_by_key[divergence["chosen_key"]]
    immediate_choice = immediate_by_key[divergence["chosen_key"]]
    prior_rehandles = sum(
        int(row["realized_immediate_rehandles"])
        for row in learned_trace[:index]
    )
    source_immediate = _read(p.SOURCE_E207 / "rows/rho-80/lambda-20.json")
    immediate_remaining_after_choice = (
        int(source_immediate["physical_rehandles"])
        - prior_rehandles
        - int(immediate_choice["immediate_rehandle"])
    )
    predicted_saving = (
        float(immediate_choice_under_learned_head["predicted_total_rehandles"])
        - float(learned_choice["predicted_total_rehandles"])
    )
    learned_realized_after_choice = int(learned_decision["realized_future_rehandles"])
    result = {
        "status": "complete",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "model_seed": p.MODEL_SEED,
        "instance_seed": p.HEAVY_INSTANCE_SEED,
        "lambda": p.TRACE_LAMBDA,
        "first_divergence_index": index,
        "identical_prefix_decisions": index,
        "prior_realized_rehandles": prior_rehandles,
        "learned_decision": learned_decision,
        "immediate_only_decision": divergence,
        "learned_choice": learned_choice,
        "immediate_choice_evaluated_by_learned_head": immediate_choice_under_learned_head,
        "predicted_rehandles_saved_by_learned_choice": predicted_saving,
        "learned_realized_future_rehandles_after_choice": learned_realized_after_choice,
        "immediate_realized_future_rehandles_after_choice": immediate_remaining_after_choice,
        "learned_total_episode_rehandles": int(learned_row["physical_rehandles"]),
        "immediate_total_episode_rehandles": int(source_immediate["physical_rehandles"]),
        "hypothesis_conditions": {
            "learned_chose_reconfigure": learned_choice["action_type"] == "reconfigure",
            "immediate_chose_progress": immediate_choice["action_type"] != "reconfigure",
            "learned_head_predicted_positive_saving": predicted_saving > 0.0,
            "predicted_saving_failed_to_materialize": (
                learned_realized_after_choice > immediate_remaining_after_choice
            ),
        },
        "stage_b_terminal_sha256": stage_b["terminal_checkpoint_sha256"],
        "new_episode_runs": 1,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    _write(path, result)
    return result


def analyze(output: Path) -> dict:
    contract = authenticate(output)
    divergence = _read(output.resolve() / "first-divergence.json")
    sweep = []
    for value in p.SWEEP_LAMBDAS:
        if value in p.NEW_SWEEP_LAMBDAS:
            row = _read(_sweep_path(output, value))
        else:
            row = _source_row(value)
        sweep.append({
            "lambda": float(value),
            "physical_rehandles": int(row["physical_rehandles"]),
            "J10": float(row["J10"]),
            "J_lambda": float(row["J_lambda"]),
            "timing_cost_2T_plus_E": float(row["timing_cost_2T_plus_E"]),
            "steps": int(row["steps"]),
            "peak_stored": int(row["peak_stored"]),
        })
    report = {
        "status": "complete",
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "first_divergence": divergence,
        "lambda_sweep": sweep,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    _write(output.resolve() / "e212-report.json", report)
    conditions = divergence["hypothesis_conditions"]
    audit = render_validated_conformance_audit(_declaration(contract))
    lines = [
        audit,
        "RESULTS",
        "=" * 72,
        "",
        f"First divergence after {divergence['identical_prefix_decisions']} identical decisions.",
        f"Learned choice: {divergence['learned_choice']['action_type']} ({divergence['learned_choice']['key']}).",
        f"Immediate-only choice: {divergence['immediate_choice_evaluated_by_learned_head']['action_type']} ({divergence['immediate_choice_evaluated_by_learned_head']['key']}).",
        f"Predicted rehandles saved by learned choice: {divergence['predicted_rehandles_saved_by_learned_choice']:.3f}.",
        f"Realized future rehandles, learned versus immediate: {divergence['learned_realized_future_rehandles_after_choice']} versus {divergence['immediate_realized_future_rehandles_after_choice']}.",
        f"Mechanism hypothesis confirmed: {all(conditions.values())}.",
        "",
        "Lambda sweep on the exact heavy instance:",
    ]
    lines.extend(
        f"  lambda={row['lambda']:g}: R={row['physical_rehandles']}, Jlambda={row['J_lambda']:.1f}, timing={row['timing_cost_2T_plus_E']:.1f}, steps={row['steps']}"
        for row in sweep
    )
    lines.extend(("", "Development diagnostic only; sealed test panel remains closed.", ""))
    (output.resolve() / "REPORT.txt").write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "complete",
        "first_divergence_index": divergence["first_divergence_index"],
        "mechanism_hypothesis_confirmed": all(conditions.values()),
        "sweep": sweep,
        "report": str(output.resolve() / "REPORT.txt"),
        "test_panel_opened": False,
    }


def run(output: Path, device_name: str) -> dict:
    contract = authenticate(output)
    divergence = _trace_replay(output, contract, device_name)
    new = int(divergence.get("new_episode_runs", 0))
    for value in p.NEW_SWEEP_LAMBDAS:
        row = _run_sweep_value(output, contract, value, device_name)
        new += int(row.get("new_episode_run", 0))
    result = analyze(output)
    return {**result, "new_complete_episode_runs": new}


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
