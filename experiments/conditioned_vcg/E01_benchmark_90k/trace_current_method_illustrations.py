#!/usr/bin/env python3
"""Trace controlled E1 mechanism forks with the final conditioned controller."""

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


PROTOCOL = "vcg_conditioned_controlled_policy_filmstrip_trace_v1"
MODEL_SEED = 0
INSTANCE_SEED = 90_001
MACRO_LIMIT = 5
SCENARIOS = (
    {
        "id": "storage_placement",
        "fork_decision_index": 0,
        "positive_lambda": 0.1,
        "selected_macro_indices_1_based": {
            "lambda0": (1, 11, 12, 14, 22),
            "lambda_positive": (1, 8, 11, 16, 19),
        },
        "aligned_events": (
            ("accept", "B1"),
            ("accept", "B8"),
            ("deliver", "B5"),
            ("deliver", "B6"),
            ("deliver", "B8"),
        ),
        "expected": {
            "lambda0": "accept:B1:3:1",
            "lambda_positive": "accept:B1:1:1",
        },
    },
    {
        "id": "rehandle_vs_delivery",
        "fork_decision_index": 12,
        "positive_lambda": 0.1,
        "selected_macro_indices_1_based": {
            "lambda0": (13, 14, 15, 16, 17),
            "lambda_positive": (1, 2, 3, 4, 5),
        },
        "expected": {
            "lambda0": "reconfigure:B6:3:2",
            "lambda_positive": "deliver:B6:4:2",
        },
    },
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results/vcg-conditioned-final-comparison-90k-cpu-v3/illustrations"
)


class IllustrationTraceError(RuntimeError):
    pass


class TraceComplete(Exception):
    pass


def _canonical_sha(value: Mapping) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _primitive_info(info: Mapping) -> dict:
    keep = (
        "picked_block",
        "stored_block",
        "relocated_block",
        "delivered_block",
        "delivery_error_time",
        "illegal_drop",
    )
    result = {}
    for key in keep:
        if key not in info:
            continue
        value = info[key]
        if isinstance(value, float) and not math.isfinite(value):
            raise IllustrationTraceError(f"non-finite primitive info: {key}")
        result[key] = value
    return result


def _trace_arm(
    *,
    project_root: Path,
    auth: Mapping,
    instance,
    arm,
    label: str,
    fixed_lambda: float,
    fork_decision_index: int,
    expected_key: str,
    device: torch.device,
    macro_limit: int | None = MACRO_LIMIT,
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
    original_execute = benchmark.execute_certified_macro

    def capture_environment(payload):
        environment = original_make_env(payload)
        state["environment"] = environment
        return environment

    def factory(base):
        agent = final90._load_conditioned_agent(
            project_root,
            auth,
            model_seed=MODEL_SEED,
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
                value = 0.0 if index < fork_decision_index else fixed_lambda
                if index == fork_decision_index:
                    environment = state["environment"]
                    if environment is None:
                        raise IllustrationTraceError("environment was not captured")
                    state["common_snapshot"] = behavior._snapshot(environment)
                    state["active"] = True
                decision = agent.select(
                    snapshot,
                    preference_lambda=value,
                    training=False,
                    epsilon=0.0,
                )
                if index == fork_decision_index and decision.candidate.key != expected_key:
                    raise IllustrationTraceError(
                        f"{label} fork selection changed: {decision.candidate.key}"
                    )
                if state["active"]:
                    state["pending"] = {
                        "decision_index": index,
                        "selected_key": decision.candidate.key,
                        "selected_action_type": decision.candidate.action_type.value,
                        "target_label": decision.candidate.target_label,
                        "source": decision.candidate.source,
                        "destination": decision.candidate.destination,
                        "exact_safe": (
                            decision.candidate.certificate.status.value == "SAFE"
                        ),
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
            raise IllustrationTraceError("decision and macro execution drifted")
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
                "lambda": float(fixed_lambda),
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
            row["exact_safe"] is not True
            or row["option_success"] is not True
            or row["failure_reason"] is not None
            or row["truncated"] is not False
            or row["duration"] != len(primitives)
        ):
            raise IllustrationTraceError(f"{label} traced a non-strict macro")
        state["macros"].append(row)
        state["pending"] = None
        return execution

    raw = None
    with (
        patch.object(benchmark, "_make_env", capture_environment),
        patch.object(benchmark, "execute_certified_macro", traced_execute),
    ):
        try:
            raw = pilot._run_raw(
                arm, instance, device=device, wrapper_factory=factory
            )
        except TraceComplete:
            pass
    if state["common_snapshot"] is None:
        raise IllustrationTraceError(f"{label} never reached the controlled fork")
    if macro_limit is not None and len(state["macros"]) != macro_limit:
        raise IllustrationTraceError(
            f"{label} trace is incomplete: {len(state['macros'])}/{macro_limit}"
        )
    if macro_limit is None and (
        raw is None or raw.get("strict_method_success") is not True
    ):
        raise IllustrationTraceError(f"{label} full continuation was not strict")
    return {
        "label": label,
        "lambda": float(fixed_lambda),
        "common_snapshot": state["common_snapshot"],
        "macros": tuple(state["macros"]),
        "full_episode_summary": (
            None
            if raw is None
            else {
                "strict_safe_complete": bool(raw["strict_method_success"]),
                "total_macro_decisions": int(raw["macro_decisions"]),
                "total_physical_rehandles": int(raw["relocations"]),
                "steps": int(raw["steps"]),
                "delivery_deviations": tuple(raw["delivery_deviations"]),
            }
        ),
    }


def _selected_arm(source: Mapping, indices: tuple[int, ...], *, common_snapshot=None) -> dict:
    if len(indices) != MACRO_LIMIT or tuple(sorted(set(indices))) != indices:
        raise IllustrationTraceError("filmstrip indices must be five increasing values")
    macros = source["macros"]
    if indices[0] < 1 or indices[-1] > len(macros):
        raise IllustrationTraceError("filmstrip index exceeds the full trace")
    selected = []
    previous = 0
    for index in indices:
        row = dict(macros[index - 1])
        row["source_macro_index_1_based"] = index
        row["omitted_before"] = index - previous - 1
        selected.append(row)
        previous = index
    return {
        "label": source["label"],
        "lambda": float(source["lambda"]),
        "common_snapshot": (
            source["common_snapshot"] if common_snapshot is None else common_snapshot
        ),
        "macros": tuple(selected),
        "full_episode_summary": source["full_episode_summary"],
    }


def _validate_aligned_events(arms: Mapping, expected: tuple[tuple[str, str], ...]) -> None:
    if len(expected) != MACRO_LIMIT:
        raise IllustrationTraceError("aligned event count does not match the filmstrip")
    for column, event in enumerate(expected):
        for label, arm in arms.items():
            row = arm["macros"][column]
            actual = (row["selected_action_type"], row["target_label"])
            if actual != event:
                raise IllustrationTraceError(
                    f"{label} column {column + 1} is {actual}, expected {event}"
                )


def _write_trace(output_dir: Path, trace: dict) -> dict:
    trace["trace_sha256"] = _canonical_sha(trace)
    path = output_dir / f"e1-current-{trace['scenario'].replace('_', '-')}-trace.json"
    path.write_text(
        json.dumps(trace, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "trace": str(path.resolve()),
        "trace_sha256": trace["trace_sha256"],
        "actions": {
            key: [row["selected_action_type"] for row in value["macros"]]
            for key, value in trace["arms"].items()
        },
        "displayed_rehandles": {
            key: sum(int(row["physical_rehandles"]) for row in value["macros"])
            for key, value in trace["arms"].items()
        },
        "full_episode_rehandles": {
            key: value["full_episode_summary"]["total_physical_rehandles"]
            for key, value in trace["arms"].items()
        },
    }


def collect(project_root: Path, output_dir: Path) -> dict:
    auth = final90._authenticate_inputs(project_root)
    manifest = final90.authenticate_manifest(project_root, final90.DEFAULT_OUTPUT)
    record = next(
        (item for item in manifest["instances"] if int(item["seed"]) == INSTANCE_SEED),
        None,
    )
    if record is None:
        raise IllustrationTraceError("E1 source EpisodeInstance is unavailable")
    instance = final90._load_instance(final90.DEFAULT_OUTPUT, record)
    arm = auth["conditioned"]["inputs"]["arms"][MODEL_SEED]
    checkpoint_sha = auth["conditioned"]["terminal_sha256"][MODEL_SEED]
    device = torch.device("cpu")

    storage = SCENARIOS[0]
    rehandle = SCENARIOS[1]
    full_lambda0 = _trace_arm(
        project_root=project_root,
        auth=auth,
        instance=instance,
        arm=arm,
        label="lambda0",
        fixed_lambda=0.0,
        fork_decision_index=0,
        expected_key=storage["expected"]["lambda0"],
        device=device,
        macro_limit=None,
    )
    full_lambda_positive = _trace_arm(
        project_root=project_root,
        auth=auth,
        instance=instance,
        arm=arm,
        label="lambda_positive",
        fixed_lambda=float(storage["positive_lambda"]),
        fork_decision_index=0,
        expected_key=storage["expected"]["lambda_positive"],
        device=device,
        macro_limit=None,
    )
    switched_lambda_positive = _trace_arm(
        project_root=project_root,
        auth=auth,
        instance=instance,
        arm=arm,
        label="lambda_positive",
        fixed_lambda=float(rehandle["positive_lambda"]),
        fork_decision_index=int(rehandle["fork_decision_index"]),
        expected_key=rehandle["expected"]["lambda_positive"],
        device=device,
        macro_limit=None,
    )

    storage_arms = {
        "lambda0": _selected_arm(
            full_lambda0, storage["selected_macro_indices_1_based"]["lambda0"]
        ),
        "lambda_positive": _selected_arm(
            full_lambda_positive,
            storage["selected_macro_indices_1_based"]["lambda_positive"],
        ),
    }
    _validate_aligned_events(storage_arms, storage["aligned_events"])
    rehandle_common = full_lambda0["macros"][
        int(rehandle["fork_decision_index"])
    ]["before"]
    lambda0_rehandle = _selected_arm(
        full_lambda0,
        rehandle["selected_macro_indices_1_based"]["lambda0"],
        common_snapshot=rehandle_common,
    )
    for index, row in enumerate(lambda0_rehandle["macros"]):
        row["omitted_before"] = 0
        row["source_macro_index_1_based"] = index + 1
    lambda_positive_rehandle = _selected_arm(
        switched_lambda_positive,
        rehandle["selected_macro_indices_1_based"]["lambda_positive"],
    )
    if rehandle_common != lambda_positive_rehandle["common_snapshot"]:
        raise IllustrationTraceError("controlled rehandle arms do not share a state")
    rehandle_arms = {
        "lambda0": lambda0_rehandle,
        "lambda_positive": lambda_positive_rehandle,
    }

    common = {
        "protocol": PROTOCOL,
        "role": "post_hoc_mechanism_illustration_not_performance_evidence",
        "method": final90.CONDITIONED_METHOD,
        "controller_architecture": "vcg_v1_1_conditioned_future_handling_v1",
        "model_seed": MODEL_SEED,
        "conditioned_terminal_sha256": checkpoint_sha,
        "instance_seed": INSTANCE_SEED,
        "episode_instance_id": instance.instance_id,
        "episode_instance_sha256": record["canonical_sha256"],
        "macro_count_per_arm": MACRO_LIMIT,
        "training_or_learning": False,
    }
    traces = (
        {
            **common,
            "scenario": storage["id"],
            "fork_decision_index": 0,
            "arms": storage_arms,
            "fixed_lambda_from_episode_start": True,
            "event_selection_is_post_hoc_for_visualization": True,
            "column_alignment": "same_operational_event_not_wall_clock_time",
            "aligned_events": tuple(
                {"action_type": action, "target_label": target}
                for action, target in storage["aligned_events"]
            ),
            "selected_macro_indices_1_based": storage[
                "selected_macro_indices_1_based"
            ],
        },
        {
            **common,
            "scenario": rehandle["id"],
            "fork_decision_index": int(rehandle["fork_decision_index"]),
            "arms": rehandle_arms,
            "common_prefix_uses_lambda_zero": True,
            "event_selection_is_post_hoc_for_visualization": False,
        },
    )
    return {trace["scenario"]: _write_trace(output_dir, trace) for trace in traces}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    print(json.dumps(collect(project_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
