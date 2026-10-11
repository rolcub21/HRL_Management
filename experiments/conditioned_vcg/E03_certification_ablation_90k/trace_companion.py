#!/usr/bin/env python3
"""Replay the frozen E3 companion fork and record primitive execution paths.

This is a post-hoc explanatory replay, not an additional performance row.  It
uses the E3-selected model/instance/decision and holds the checkpoint,
preference, selector, and EpisodeInstance fixed.  The only differing factor is
the E3 candidate source: physical feasibility or recoverability certification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import benchmark_viability_critic_priority as benchmark
import render_vcg_unified_behavior_gifs as behavior
import run_vcg_conditioned_final_comparison_90k as final90
import run_vcg_v11_nested_handling_pilot as pilot
from experiments.conditioned_vcg.E03_certification_ablation_90k import run as e3


PROTOCOL = "vcg_conditioned_e03_certification_companion_trace_v2"
DEFAULT_OUTPUT_DIR = e3.DEFAULT_OUTPUT
DEFAULT_TRACE = DEFAULT_OUTPUT_DIR / "e03-companion-trace.json"


class CompanionTraceError(RuntimeError):
    pass


class TraceComplete(Exception):
    pass


def _canonical_sha(value: Mapping) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _same_json(left, right) -> bool:
    return json.dumps(
        left, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) == json.dumps(
        right, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _primitive_info(info: Mapping) -> dict:
    keys = (
        "picked_block",
        "stored_block",
        "relocated_block",
        "delivered_block",
        "delivery_error_time",
        "illegal_drop",
    )
    result = {}
    for key in keys:
        if key not in info:
            continue
        value = info[key]
        if isinstance(value, float) and not math.isfinite(value):
            raise CompanionTraceError(f"non-finite primitive info: {key}")
        result[key] = value
    return result


def _trace_arm(
    *,
    project_root: Path,
    auth: Mapping,
    instance,
    arm,
    candidate_source: str,
    fork_index: int,
    expected_key: str,
    macro_limit: int | None,
    expected_common: Mapping,
    expected_first_after: Mapping | None,
    device: torch.device,
) -> dict:
    state = {
        "decision_index": 0,
        "active": False,
        "pending": None,
        "common_snapshot": None,
        "macros": [],
        "environment": None,
    }
    original_make_env = benchmark._make_env
    original_enumerator = benchmark.enumerate_viability_candidates
    original_execute = benchmark.execute_certified_macro

    def capture_environment(payload):
        environment = original_make_env(payload)
        state["environment"] = environment
        return environment

    def candidate_enumerator(environment, **kwargs):
        certified = original_enumerator(environment, **kwargs)
        if candidate_source == e3.CERTIFIED:
            return certified
        cache = kwargs.get("cache")
        if cache is None:
            raise CompanionTraceError("physical replay has no certificate cache")
        return e3._physical_snapshot(
            environment,
            certified,
            search_config=kwargs["search_config"],
            cache=cache,
            max_replans=int(kwargs.get("max_replans", 8)),
        )

    def factory(base):
        agent = final90._load_conditioned_agent(
            project_root,
            auth,
            model_seed=int(auth["selected_model_seed"]),
            base=base,
            device=device,
        )
        agent.set_epsilon(0.0)

        class ControlledAgent:
            config = agent.config

            def reset_episode_state(self):
                agent.reset_episode_state()

            def select(self, snapshot, *, training=False, epsilon=0.0):
                if (
                    macro_limit is not None
                    and state["active"]
                    and len(state["macros"]) >= macro_limit
                ):
                    raise TraceComplete
                index = int(state["decision_index"])
                if index == fork_index:
                    environment = state["environment"]
                    if environment is None:
                        raise CompanionTraceError("environment was not captured")
                    common = behavior._snapshot(environment)
                    if not _same_json(common, expected_common):
                        raise CompanionTraceError(
                            f"{candidate_source} companion state drifted"
                        )
                    state["common_snapshot"] = common
                    state["active"] = True
                decision = agent.select(
                    snapshot,
                    preference_lambda=e3.PREFERENCE_LAMBDA,
                    training=False,
                    epsilon=0.0,
                )
                if index == fork_index and decision.candidate.key != expected_key:
                    raise CompanionTraceError(
                        f"{candidate_source} fork changed: {decision.candidate.key}"
                    )
                if state["active"]:
                    shadow = e3._shadow_certificate(decision.candidate)
                    state["pending"] = {
                        "decision_index": index,
                        "selected_key": decision.candidate.key,
                        "selected_action_type": decision.candidate.action_type.value,
                        "target_label": decision.candidate.target_label,
                        "source": decision.candidate.source,
                        "destination": decision.candidate.destination,
                        "shadow_status": shadow.status.value,
                        "exact_safe": shadow.status.value == "SAFE",
                    }
                state["decision_index"] = index + 1
                return decision

            def observe_outcome(self, decision, *, next_snapshot, done):
                return agent.observe_outcome(
                    decision,
                    next_snapshot=next_snapshot,
                    done=done,
                )

        return ControlledAgent()

    def traced_execute(environment, candidate, **kwargs):
        if not state["active"]:
            return original_execute(environment, candidate, **kwargs)
        pending = state["pending"]
        if not isinstance(pending, dict) or pending["selected_key"] != candidate.key:
            raise CompanionTraceError("decision and macro execution drifted")
        before = behavior._snapshot(environment)
        primitives = []
        original_step = environment.step
        had_instance_step = "step" in environment.__dict__
        previous_instance_step = environment.__dict__.get("step")

        def traced_step(action):
            primitive_before = behavior._snapshot(environment)
            result = original_step(action)
            primitive_after = behavior._snapshot(environment)
            _, reward, terminal, info = result
            primitives.append(
                {
                    "index": len(primitives),
                    "action_id": int(action),
                    "action": str(environment.ACTION_NAMES[int(action)]),
                    "reward": float(reward),
                    "terminal": bool(terminal),
                    "info": _primitive_info(info),
                    "before": primitive_before,
                    "after": primitive_after,
                }
            )
            return result

        environment.step = traced_step
        try:
            execution = original_execute(environment, candidate, **kwargs)
        finally:
            if had_instance_step:
                environment.step = previous_instance_step
            else:
                del environment.step
        row = dict(pending)
        row.update(
            {
                "duration": int(execution.duration),
                "physical_rehandles": int(execution.relocations),
                "delivery_deviations": tuple(
                    float(value) for value in execution.delivery_deviations
                ),
                "option_success": bool(execution.option_success),
                "failure_reason": execution.failure_reason,
                "truncated": bool(execution.truncated),
                "before": before,
                "after": behavior._snapshot(environment),
                "primitive_steps": tuple(primitives),
            }
        )
        if (
            row["option_success"] is not True
            or row["failure_reason"] is not None
            or row["truncated"] is not False
            or row["duration"] != len(primitives)
        ):
            raise CompanionTraceError(
                f"{candidate_source} replay contains a failed macro"
            )
        state["macros"].append(row)
        state["pending"] = None
        return execution

    raw = None
    with (
        patch.object(benchmark, "_make_env", capture_environment),
        patch.object(
            benchmark,
            "enumerate_viability_candidates",
            candidate_enumerator,
        ),
        patch.object(benchmark, "execute_certified_macro", traced_execute),
    ):
        try:
            raw = pilot._run_raw(
                arm,
                instance,
                device=device,
                wrapper_factory=factory,
            )
        except TraceComplete:
            pass

    if state["common_snapshot"] is None:
        raise CompanionTraceError(f"{candidate_source} never reached the fork")
    if macro_limit is not None and len(state["macros"]) != macro_limit:
        raise CompanionTraceError(
            f"{candidate_source} trace is incomplete: "
            f"{len(state['macros'])}/{macro_limit}"
        )
    if macro_limit is None and (
        raw is None
        or raw.get("strict_method_success") is not True
        or raw.get("terminal") is not True
        or raw.get("method_failure_reason") is not None
    ):
        raise CompanionTraceError(
            f"{candidate_source} full continuation did not complete strictly"
        )
    first = state["macros"][0]
    if expected_first_after is not None and not _same_json(
        first["after"], expected_first_after
    ):
        raise CompanionTraceError(f"{candidate_source} successor drifted")
    if candidate_source == e3.PHYSICAL and first["shadow_status"] != "UNSAFE":
        raise CompanionTraceError("physical companion no longer selects UNSAFE")
    if candidate_source == e3.CERTIFIED and any(
        not row["exact_safe"] for row in state["macros"]
    ):
        raise CompanionTraceError("certified continuation contains an unsafe macro")
    return {
        "candidate_source": candidate_source,
        "common_snapshot": state["common_snapshot"],
        "macros": tuple(state["macros"]),
        "terminal_snapshot": behavior._snapshot(state["environment"]),
        "full_episode_summary": (
            None
            if raw is None
            else {
                "strict_method_success": bool(raw["strict_method_success"]),
                "terminal": bool(raw["terminal"]),
                "method_failure_reason": raw["method_failure_reason"],
                "macro_decisions": int(raw["macro_decisions"]),
                "completed_deliveries": len(raw["delivery_deviations"]),
                "physical_rehandles": int(raw["relocations"]),
                "steps": int(raw["steps"]),
            }
        ),
    }


def collect(project_root: Path, output_dir: Path) -> dict:
    contract = e3.authenticate_contract(project_root, output_dir)
    report = e3._load_json(output_dir / e3.REPORT_NAME, label="E3 report")
    e3._verify_hash(report, "report_sha256", label="E3 report")
    if report.get("status") != "complete":
        raise CompanionTraceError("E3 report is not complete")
    case = e3._load_json(output_dir / e3.COMPANION_NAME, label="E3 companion")
    e3._verify_hash(case, "case_sha256", label="E3 companion")
    if report.get("companion_case", {}).get("case_sha256") != case["case_sha256"]:
        raise CompanionTraceError("report and companion case disagree")

    model_seed = int(case["model_seed"])
    instance_seed = int(case["instance_seed"])
    event = case["event"]
    selected = event["selected"]
    counterfactual = event["certified_counterfactual"]
    if (
        event.get("selected_shadow_status") != "UNSAFE"
        or counterfactual.get("shadow_status") != "SAFE"
        or float(case["critic_merit_gap_selected_minus_certified"]) <= 0.0
    ):
        raise CompanionTraceError("selected case no longer satisfies the E3 fork")

    manifest = final90.authenticate_manifest(project_root, final90.DEFAULT_OUTPUT)
    record = next(
        (row for row in manifest["instances"] if int(row["seed"]) == instance_seed),
        None,
    )
    if record is None:
        raise CompanionTraceError("selected EpisodeInstance is unavailable")
    instance = final90._load_instance(final90.DEFAULT_OUTPUT, record)
    if instance.instance_id != case["episode_instance_id"]:
        raise CompanionTraceError("selected EpisodeInstance identity drifted")
    inputs = final90._authenticate_inputs(project_root)
    arm = inputs["conditioned"]["inputs"]["arms"][model_seed]
    auth = dict(inputs)
    auth["selected_model_seed"] = model_seed
    device = torch.device("cpu")

    physical = _trace_arm(
        project_root=project_root,
        auth=auth,
        instance=instance,
        arm=arm,
        candidate_source=e3.PHYSICAL,
        fork_index=int(case["decision_index"]),
        expected_key=selected["key"],
        macro_limit=1,
        expected_common=event["before"],
        expected_first_after=event["after"],
        device=device,
    )
    certified = _trace_arm(
        project_root=project_root,
        auth=auth,
        instance=instance,
        arm=arm,
        candidate_source=e3.CERTIFIED,
        fork_index=int(case["decision_index"]),
        expected_key=counterfactual["key"],
        macro_limit=None,
        expected_common=event["before"],
        expected_first_after=None,
        device=device,
    )
    if not _same_json(
        physical["common_snapshot"], certified["common_snapshot"]
    ):
        raise CompanionTraceError("candidate-source arms do not share the fork state")

    result = {
        "protocol": PROTOCOL,
        "role": "post_hoc_explanation_not_performance_evidence",
        "source_contract_sha256": contract["contract_sha256"],
        "source_report_sha256": report["report_sha256"],
        "source_case_sha256": case["case_sha256"],
        "model_seed": model_seed,
        "instance_seed": instance_seed,
        "episode_instance_id": instance.instance_id,
        "fork_decision_index": int(case["decision_index"]),
        "preference_lambda": e3.PREFERENCE_LAMBDA,
        "critic_merit_gap_selected_minus_certified": float(
            case["critic_merit_gap_selected_minus_certified"]
        ),
        "training_or_learning": False,
        "checkpoint_selection": False,
        "arms": {
            e3.PHYSICAL: physical,
            e3.CERTIFIED: certified,
        },
    }
    result["trace_sha256"] = _canonical_sha(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_TRACE)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output = args.output.resolve()
    torch.set_num_threads(1)
    trace = collect(project_root, output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    final90._atomic_json(output, trace)
    print(
        json.dumps(
            {
                "output": str(output),
                "trace_sha256": trace["trace_sha256"],
                "actions": {
                    source: [row["selected_key"] for row in arm["macros"]]
                    for source, arm in trace["arms"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
