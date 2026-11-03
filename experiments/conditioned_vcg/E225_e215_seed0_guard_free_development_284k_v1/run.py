#!/usr/bin/env python3
"""Run the E225 frozen seed-0 guard-free development replay."""

from __future__ import annotations

import argparse
import builtins
from collections import defaultdict
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

import torch

import run_vcg_objective_gamma_audit as audit
from experiments.conditioned_vcg.E14_certification_scalability_95k import reuse
from experiments.conditioned_vcg.E100_accelerated_release_clock_163k_v1 import (
    fast_inputs,
)
from experiments.conditioned_vcg.E107_queue_context_stage_a_8x8_171k_v1 import (
    representation as e107_representation,
)
from experiments.conditioned_vcg.E189_e188_roundoff_repair_248k_v1.roundoff import (
    repaired_audit,
)
from experiments.conditioned_vcg.E191_fresh_long_stay_crowding_250k_v1 import (
    instrumentation as e191_instrumentation,
    protocol as e191,
    scenario as e191_scenario,
    stage as e191_stage,
)
from projects.temporal_consequence_realization.method_conformance import (
    render_validated_conformance_audit,
    validate_protocol,
)

from .guard_free import FrozenGuardFreeQueueAgent
from . import protocol as p


class E225Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E225Error(message)


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable(value), indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _read(path: Path) -> dict:
    require(Path(path).is_file(), f"missing E225 artifact: {path}")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"invalid E225 artifact: {path}")
    return value


def _checkpoint_payload() -> tuple[dict, dict]:
    identity = p.checkpoint_identity()
    payload = torch.load(
        identity["source_checkpoint"], map_location="cpu", weights_only=False
    )
    require(
        int(payload.get("model_seed", -1)) == p.MODEL_SEED
        and int(payload.get("selected_checkpoint_episode", -1)) == 300
        and payload.get("checkpoint_role") == "best_deployment_finalized",
        "E225 source checkpoint identity differs",
    )
    return identity, payload


def _reference_runs(payload: dict) -> dict[int, dict]:
    record = payload.get("best_validation_record", {})
    require(
        int(record.get("checkpoint_episode", -1)) == 300,
        "E225 source lacks the selected episode-300 validation",
    )
    runs = tuple(record.get("runs", ()))
    by_seed = {int(run["instance_seed"]): dict(run) for run in runs}
    require(
        len(runs) == len(by_seed) == len(p.INSTANCE_SEEDS)
        and set(by_seed) == set(p.INSTANCE_SEEDS),
        "E225 embedded guard-on reference coordinates differ",
    )
    return by_seed


def _make_instance(seed: int):
    env = e191_scenario.LongStayCrowdingEnv(e191.SCENARIO, e191.SPEC)
    instance = env.sample_episode_instance(seed=int(seed))
    instance.validate_for(env)
    return instance


def _reference_path(output: Path, seed: int) -> Path:
    return output / "rows/guard_on_reference" / f"instance-{int(seed)}.json"


def _guard_free_path(output: Path, seed: int) -> Path:
    return output / "rows/guard_free" / f"instance-{int(seed)}.json"


def prepare(output: Path) -> dict:
    output = output.resolve()
    contract = p.contract()
    validate_protocol(contract, source=p.PROTOCOL)
    identity, payload = _checkpoint_payload()
    references = _reference_runs(payload)
    contract_path = output / "contract.json"
    if contract_path.is_file():
        require(_read(contract_path) == contract, "E225 contract changed")
    else:
        require(
            not output.exists() or not any(output.iterdir()),
            "E225 refuses a nonempty uncontracted output",
        )
        _write(contract_path, contract)

    for seed, run in references.items():
        instance = _make_instance(seed)
        require(
            run.get("episode_instance_id") == instance.instance_id,
            f"E225 reference instance {seed} differs",
        )
        row = {
            "protocol": p.PROTOCOL,
            "contract_sha256": contract["contract_sha256"],
            "method": "guard_on_reference",
            "model_seed": p.MODEL_SEED,
            "selected_episode": 300,
            "instance_seed": seed,
            "workload": p.workload_for_seed(seed),
            "instance_id": instance.instance_id,
            "checkpoint_sha256": identity["source_checkpoint_sha256"],
            "strict_complete": bool(run.get("strict_method_success")),
            "run": run,
            "metrics": dict(run.get("e187_metrics", {})),
            "source": "selected checkpoint embedded best-validation record",
            "episode_runs": 0,
            "training_runs": 0,
            "test_panel_opened": False,
            "paper_evidence": False,
        }
        path = _reference_path(output, seed)
        if path.is_file():
            require(_read(path) == _jsonable(row), f"E225 reference row {seed} changed")
        else:
            _write(path, row)
    return {
        "status": "prepared",
        "contract_sha256": contract["contract_sha256"],
        "reference_rows": len(references),
        "guard_free_episode_runs": 0,
        "test_panel_opened": False,
    }


def authenticate(output: Path) -> dict:
    prepare(output)
    return _read(output.resolve() / "contract.json")


def _load_agent(device: str):
    identity, payload = _checkpoint_payload()
    agent = FrozenGuardFreeQueueAgent.from_checkpoint(
        payload, device=device, resumable=False, seed=p.MODEL_SEED
    )
    agent.set_epsilon(0.0)
    agent.Q_local.eval()
    agent.Q_target.eval()
    return agent, identity, payload


def _search(payload: dict):
    from PSLAP.viability_filter import ViabilitySearchConfig

    value = payload["viability_search"]
    return ViabilitySearchConfig(
        max_depth=value["max_depth"],
        max_nodes=int(value["max_nodes"]),
        max_primitive_steps=value["max_primitive_steps"],
        reserve_queue_cells=bool(value["reserve_queue_cells"]),
        search_order=str(value["search_order"]),
    )


def _liveness(payload: dict):
    from PSLAP.viability_candidates import BoundedEventDeferRule

    value = payload["liveness_rule"]
    return BoundedEventDeferRule(
        max_option_steps=int(value["max_option_steps"]),
        max_consecutive_defer_decisions=int(
            value["max_consecutive_defer_decisions"]
        ),
    )


def _decision_summary(agent: FrozenGuardFreeQueueAgent) -> dict:
    decisions = tuple(agent.guard_free_decision_records)
    require(
        all(
            not row["liveness_forced"] and not row["liveness_restricted"]
            for row in decisions
        ),
        "E225 observed guard intervention",
    )
    return {
        "audit": agent.guard_free_audit(),
        "records": decisions,
        "selection_sources": dict(CounterLike(
            row["selection_source"] for row in decisions
        )),
    }


class CounterLike(defaultdict):
    def __init__(self, values=()):
        super().__init__(int)
        for value in values:
            self[str(value)] += 1


def run_one(output: Path, seed: int, device: str) -> dict:
    output = output.resolve()
    contract = authenticate(output)
    require(seed in p.INSTANCE_SEEDS, "E225 seed is outside the development panel")
    require(device in ("cpu", "cuda"), "E225 device must be cpu or cuda")
    if device == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    path = _guard_free_path(output, seed)
    identity_fields = {
        "protocol": p.PROTOCOL,
        "contract_sha256": contract["contract_sha256"],
        "method": "guard_free",
        "model_seed": p.MODEL_SEED,
        "selected_episode": 300,
        "instance_seed": int(seed),
    }
    if path.is_file():
        row = _read(path)
        require(
            all(row.get(key) == value for key, value in identity_fields.items()),
            f"E225 saved row differs: {path}",
        )
        return {**row, "reused": True, "episode_runs": 0}

    agent, checkpoint, payload = _load_agent(device)
    expected = _make_instance(seed)
    env = e191_scenario.LongStayCrowdingEnv(e191.SCENARIO, e191.SPEC)
    original_print = builtins.print

    def e225_print(*values, **kwargs):
        if values and isinstance(values[0], str) and values[0].startswith("E191 "):
            values = ("E225 " + values[0][len("E191 "):], *values[1:])
        return original_print(*values, **kwargs)

    print(
        f"E225 start guard_free seed={seed} workload={p.workload_for_seed(seed)}",
        flush=True,
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(
            audit,
            "enumerate_viability_candidates",
            e191_stage._accelerated_enumerator(
                audit.enumerate_viability_candidates
            ),
        ))
        stack.enter_context(patch.object(
            audit, "ViabilityCertificateCache", reuse.TimingInvariantCertificateCache
        ))
        stack.enter_context(patch.object(
            audit, "_rescore_run", e191_instrumentation.single_objective_rescore
        ))
        stack.enter_context(patch.object(builtins, "print", e225_print))
        stack.enter_context(reuse.path_cleanup_active())
        stack.enter_context(fast_inputs.enabled())
        stack.enter_context(e107_representation.acceleration_enabled())
        stack.enter_context(repaired_audit())
        run = e191_instrumentation.run_episode(
            agent,
            env,
            objective_spec=e191.SPEC,
            instance_seed=int(seed),
            training=False,
            max_steps=e191_scenario.MAX_STEPS,
            search_config=_search(payload),
            liveness_rule=_liveness(payload),
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_decisions=1,
            epsilon_warmup_decisions=0,
            updates_per_macro=1,
            run_protocol=p.PROTOCOL,
        )
    require(
        env.current_episode_instance.instance_id == expected.instance_id,
        "E225 executed another instance",
    )
    decisions = _decision_summary(agent)
    require(
        len(decisions["records"]) == int(run["macro_decisions"])
        and int(decisions["audit"]["frontiers"]) == int(run["macro_decisions"]),
        "E225 decision-frontier and macro counts differ",
    )
    require(
        int(run.get("audit", {}).get("liveness_forced_decisions", 0)) == 0
        and run.get("selection_source_counts", {}).get(
            "exact_recovery_witness_guard", 0
        ) == 0,
        "E225 run contains a forced witness decision",
    )
    row = {
        **identity_fields,
        "workload": p.workload_for_seed(seed),
        "instance_id": expected.instance_id,
        "checkpoint_sha256": checkpoint["source_checkpoint_sha256"],
        "status": "complete",
        "strict_complete": bool(run["strict_method_success"]),
        "method_failure_reason": run["method_failure_reason"],
        "run": run,
        "metrics": dict(run["e187_metrics"]),
        "guard_free": decisions,
        "episode_runs": 1,
        "training_runs": 0,
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    _write(path, row)
    return row


def _mean(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return fmean(values) if values else None


def _metric(row: dict, name: str):
    metrics = row.get("metrics", {})
    if name in metrics:
        return metrics[name]
    return metrics.get("components", {}).get(name)


def _paired_row(reference: dict, guard_free: dict) -> dict:
    reference_j = _metric(reference, "J")
    free_j = _metric(guard_free, "J")
    reference_r = _metric(reference, "physical_rehandles")
    free_r = _metric(guard_free, "physical_rehandles")
    reference_steps = reference["run"].get("steps")
    free_steps = guard_free["run"].get("steps")
    return {
        "instance_seed": int(guard_free["instance_seed"]),
        "workload": guard_free["workload"],
        "reference_strict_complete": bool(reference["strict_complete"]),
        "guard_free_strict_complete": bool(guard_free["strict_complete"]),
        "reference_J": reference_j,
        "guard_free_J": free_j,
        "delta_J": None if reference_j is None or free_j is None else free_j - reference_j,
        "relative_delta_J": (
            None if reference_j in (None, 0) or free_j is None
            else (free_j - reference_j) / reference_j
        ),
        "reference_rehandles": reference_r,
        "guard_free_rehandles": free_r,
        "delta_rehandles": (
            None if reference_r is None or free_r is None
            else free_r - reference_r
        ),
        "reference_steps": reference_steps,
        "guard_free_steps": free_steps,
        "delta_steps": (
            None if reference_steps is None or free_steps is None
            else free_steps - reference_steps
        ),
        "reference_forced_decisions": int(
            reference["run"].get("e188_guard", {}).get("forced_decisions", 0)
        ),
        "reference_masked_decisions": int(
            reference["run"].get("e188_guard", {}).get("masked_decisions", 0)
        ),
        "guard_free_threshold_frontiers": int(
            guard_free["guard_free"]["audit"]["threshold_frontiers"]
        ),
        "guard_free_counterfactual_force_frontiers": int(
            guard_free["guard_free"]["audit"][
                "old_guard_would_force_frontiers"
            ]
        ),
        "guard_free_singleton_reconfigures": int(
            guard_free["guard_free"]["audit"][
                "singleton_reconfigure_decisions"
            ]
        ),
        "guard_free_critic_reconfigures": int(
            guard_free["guard_free"]["audit"]["critic_reconfigure_decisions"]
        ),
    }


def _render_report(contract: dict, report: dict) -> str:
    lines = [
        render_validated_conformance_audit(contract).rstrip(),
        "",
        "RESULTS",
        "=======",
        "",
        f"Status: {report['status']}",
        f"Verdict: {report['verdict']}",
        "",
        "| workload | seed | J guard | J no guard | R guard | R no guard | "
        "forced | strict no guard |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["paired_rows"]:
        lines.append(
            f"| {row['workload']} | {row['instance_seed']} | "
            f"{row['reference_J']:.1f} | {row['guard_free_J']:.1f} | "
            f"{row['reference_rehandles']:.0f} | "
            f"{row['guard_free_rehandles']:.0f} | "
            f"{row['reference_forced_decisions']} | "
            f"{int(row['guard_free_strict_complete'])} |"
        )
    lines.extend((
        "",
        "The no-guard arm retains the frozen four-value non-progress context "
        "because it is part of the trained critic input. It cannot mask a "
        "candidate, activate a witness, or bypass critic selection.",
        "",
    ))
    return "\n".join(lines)


def analyze(output: Path) -> dict:
    output = output.resolve()
    contract = authenticate(output)
    paired = []
    missing = []
    for seed in p.INSTANCE_SEEDS:
        reference = _read(_reference_path(output, seed))
        path = _guard_free_path(output, seed)
        if not path.is_file():
            missing.append(seed)
            continue
        paired.append(_paired_row(reference, _read(path)))

    complete = len(paired) == len(p.INSTANCE_SEEDS)
    all_strict = complete and all(
        row["guard_free_strict_complete"] for row in paired
    )
    relative = [
        row["relative_delta_J"] for row in paired
        if row["relative_delta_J"] is not None
    ]
    mean_reference_j = _mean(
        [{"value": row["reference_J"]} for row in paired], "value"
    )
    mean_free_j = _mean(
        [{"value": row["guard_free_J"]} for row in paired], "value"
    )
    pooled_relative = (
        None if mean_reference_j in (None, 0) or mean_free_j is None
        else (mean_free_j - mean_reference_j) / mean_reference_j
    )
    essentially_unchanged = bool(
        complete
        and all_strict
        and pooled_relative is not None
        and abs(pooled_relative) <= 0.05
        and all(abs(value) <= 0.10 for value in relative)
    )
    if not complete:
        verdict = "partial"
    elif not all_strict:
        verdict = "guard_removal_changed_completion"
    elif essentially_unchanged:
        verdict = "completion_and_cost_essentially_unchanged"
    else:
        verdict = "completion_held_but_behavior_changed_materially"
    report = {
        "status": "complete" if complete else "partial",
        "verdict": verdict,
        "model_seed": p.MODEL_SEED,
        "selected_episode": 300,
        "paired_rows": paired,
        "missing_guard_free_seeds": missing,
        "summary": {
            "cases": len(paired),
            "guard_free_strict_completion_rate": (
                None if not paired else sum(
                    row["guard_free_strict_complete"] for row in paired
                ) / len(paired)
            ),
            "mean_reference_J": mean_reference_j,
            "mean_guard_free_J": mean_free_j,
            "relative_mean_J_change": pooled_relative,
            "worst_absolute_case_relative_J_change": (
                None if not relative else max(abs(value) for value in relative)
            ),
            "reference_forced_decisions": sum(
                row["reference_forced_decisions"] for row in paired
            ),
            "reference_masked_decisions": sum(
                row["reference_masked_decisions"] for row in paired
            ),
            "guard_free_counterfactual_force_frontiers": sum(
                row["guard_free_counterfactual_force_frontiers"] for row in paired
            ),
            "guard_free_singleton_reconfigures": sum(
                row["guard_free_singleton_reconfigures"] for row in paired
            ),
            "guard_free_critic_reconfigures": sum(
                row["guard_free_critic_reconfigures"] for row in paired
            ),
        },
        "essentially_unchanged_rule": (
            "all nine strict-complete; absolute pooled mean J change <=5%; "
            "absolute per-case J change <=10%"
        ),
        "contract_sha256": contract["contract_sha256"],
        "test_panel_opened": False,
        "paper_evidence": False,
    }
    _write(output / p.REPORT_JSON, report)
    (output / p.REPORT_NAME).write_text(
        _render_report(contract, report), encoding="utf-8"
    )
    return report


def self_check(output: Path) -> dict:
    prepared = prepare(output)
    agent, identity, _payload = _load_agent("cpu")
    require(
        isinstance(agent, FrozenGuardFreeQueueAgent),
        "E225 did not construct its guard-free agent",
    )
    require(
        agent.guard_free_audit()["forced_decisions"] == 0
        and agent.guard_free_audit()["masked_frontiers"] == 0,
        "E225 guard-free counters are not clean",
    )
    return {
        **prepared,
        "status": "pass",
        "checkpoint_sha256": identity["source_checkpoint_sha256"],
        "selected_episode": identity["selected_episode"],
        "guard_free_contract": agent.guard_free_audit()["contract"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "self-check", "run-one", "run-all", "analyze")
    )
    parser.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, choices=p.INSTANCE_SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output.resolve()
    if args.command == "prepare":
        result = prepare(output)
    elif args.command == "self-check":
        result = self_check(output)
    elif args.command == "analyze":
        result = analyze(output)
    else:
        require(
            args.command != "run-one" or args.seed is not None,
            "run-one requires --seed",
        )
        seeds = (args.seed,) if args.command == "run-one" else p.INSTANCE_SEEDS
        rows = [run_one(output, seed, args.device) for seed in seeds]
        result = {
            "status": "complete",
            "rows": len(rows),
            "new_episode_runs": sum(int(row.get("episode_runs", 0)) for row in rows),
            "analysis": analyze(output),
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=str))


if __name__ == "__main__":
    main()

