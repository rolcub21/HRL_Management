#!/usr/bin/env python3
"""Trace two policy continuations from one authenticated half-full VCG state.

This is a post-hoc illustration utility.  It replays the frozen lambda-zero
prefix to a previously authenticated decision boundary, switches only the
evaluation clone's lambda at that boundary, and records the next few real
macro decisions plus every primitive environment step.  It does not train,
select a checkpoint, or create performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import MethodType
from typing import Mapping

import evaluate_vcg_unified_frozen_lambda_confirmation as confirmation
import probe_vcg_unified_decision_mechanism as mechanism
import render_vcg_unified_behavior_gifs as behavior
import train_vcg_constrained_v2_3 as trainer
import evaluate_vcg_unified_frozen_lambda_seed_stability as development


PROTOCOL = "vcg_unified_controlled_policy_filmstrip_trace_v1"
DEFAULT_PROBE = (
    confirmation.DEFAULT_OUTPUT.parent
    / "vcg-unified-frozen-lambda-confirmation-87k-behavior-view"
    / "half-full-decision-fork-probe.json"
)
DEFAULT_OUTPUT = DEFAULT_PROBE.with_name("half-full-policy-filmstrip-trace.json")


class FilmstripTraceError(RuntimeError):
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
        if key in info:
            value = info[key]
            if isinstance(value, float) and not math.isfinite(value):
                raise FilmstripTraceError(f"non-finite primitive info: {key}")
            result[key] = value
    return result


def _trace_arm(
    *,
    label: str,
    fixed_lambda: float,
    parent: Mapping,
    entry: Mapping,
    selected: Mapping,
    model_seed: int,
    instance_seed: int,
    rng_index: int,
    macro_limit: int,
    device: str,
) -> dict:
    class TraceComplete(Exception):
        pass

    target_index = int(selected["decision_index"])
    expected_selection = selected["same_rng_counterfactual_samples"][label]
    macros: list[dict] = []
    common_snapshot = None

    with confirmation._runtime_context(model_seed):
        runtime = mechanism._probe_runtime(parent, device=device)
        trainer._install_schedule(
            runtime,
            trainer.schedule_for_episode(confirmation.EPISODE, validation=True),
        )
        runtime.set_dual_lambda(0.0)
        instance = confirmation._load_instance(
            confirmation.DEFAULT_OUTPUT / entry["relative_path"], entry, runtime.env
        )
        original_factory = runtime._evaluation_agent
        original_execute = runtime._execute_macro
        state = {"active": False, "pending": None, "decision_count": 0}

        def factory(*, seed):
            clone = original_factory(seed=seed)
            original_select = clone.select

            def select(self, snapshot, *, training=True, epsilon=None):
                nonlocal common_snapshot
                index = state["decision_count"]
                state["decision_count"] += 1
                if state["active"] and len(macros) >= macro_limit:
                    raise TraceComplete()
                if index == target_index:
                    observed = behavior._snapshot(runtime.env)
                    if mechanism._json_safe(observed) != mechanism._json_safe(
                        selected["physical_snapshot"]
                    ):
                        raise FilmstripTraceError("controlled fork state drifted")
                    common_snapshot = observed
                    self.set_dual_lambda(fixed_lambda)
                    state["active"] = True
                decision = original_select(
                    snapshot, training=training, epsilon=epsilon
                )
                if index == target_index and decision.record.key != expected_selection["selected_key"]:
                    raise FilmstripTraceError(
                        f"{label} shared-RNG selection drifted: {decision.record.key}"
                    )
                if state["active"]:
                    state["pending"] = {
                        "decision_index": index,
                        "selected_key": decision.record.key,
                        "selected_action_type": decision.candidate.action_type.value,
                        "target_label": decision.candidate.target_label,
                        "source": decision.candidate.source,
                        "destination": decision.candidate.destination,
                        "exact_safe": decision.candidate.certificate.status.value == "SAFE",
                    }
                return decision

            clone.select = MethodType(select, clone)
            return clone

        def traced_execute(candidate, *, remaining_steps: int, evaluation: bool):
            if not state["active"]:
                return original_execute(
                    candidate,
                    remaining_steps=remaining_steps,
                    evaluation=evaluation,
                )
            pending = state["pending"]
            if not isinstance(pending, dict) or pending["selected_key"] != candidate.key:
                raise FilmstripTraceError("selected decision/macro binding drifted")
            before = behavior._snapshot(runtime.env)
            primitives = []
            underlying = runtime.env._environment
            original_step = underlying.step
            had_instance_step = "step" in underlying.__dict__
            previous_instance_step = underlying.__dict__.get("step")

            def traced_step(action):
                primitive_before = behavior._snapshot(runtime.env)
                result = original_step(action)
                primitive_after = behavior._snapshot(runtime.env)
                _, reward, terminal, info = result
                primitives.append(
                    {
                        "index": len(primitives),
                        "action_id": int(action),
                        "action": str(underlying.ACTION_NAMES[int(action)]),
                        "reward": float(reward),
                        "terminal": bool(terminal),
                        "info": _primitive_info(info),
                        "before": primitive_before,
                        "after": primitive_after,
                    }
                )
                return result

            underlying.step = traced_step
            try:
                execution = original_execute(
                    candidate,
                    remaining_steps=remaining_steps,
                    evaluation=evaluation,
                )
            finally:
                if had_instance_step:
                    underlying.step = previous_instance_step
                else:
                    del underlying.step
            after = behavior._snapshot(runtime.env)
            if int(execution.duration) != len(primitives):
                raise FilmstripTraceError("primitive trace length differs from macro duration")
            row = dict(pending)
            row.update(
                {
                    "lambda": fixed_lambda,
                    "duration": int(execution.duration),
                    "physical_rehandles": int(execution.relocations),
                    "delivery_deviations": tuple(
                        float(value) for value in execution.delivery_deviations
                    ),
                    "option_success": bool(execution.option_success),
                    "failure_reason": execution.failure_reason,
                    "truncated": bool(execution.truncated),
                    "before": before,
                    "after": after,
                    "primitive_steps": tuple(primitives),
                }
            )
            if (
                row["exact_safe"] is not True
                or row["option_success"] is not True
                or row["failure_reason"] is not None
                or row["truncated"] is not False
            ):
                raise FilmstripTraceError(f"{label} traced a non-strict macro")
            macros.append(row)
            state["pending"] = None
            return execution

        runtime._evaluation_agent = factory
        runtime._execute_macro = traced_execute
        runtime.begin_validation_batch()
        original_environment = runtime.env
        runtime.env = confirmation._BoundEnvironment(original_environment, instance)
        try:
            try:
                runtime.run_episode(
                    instance_seed=instance_seed,
                    training=False,
                    max_steps=confirmation.MAX_STEPS,
                    policy_rng_index=rng_index,
                    policy_rng_seed=confirmation._policy_rng(
                        instance_seed - confirmation.INSTANCE_SEEDS[0], rng_index
                    ),
                )
            except TraceComplete:
                pass
        finally:
            runtime.env = original_environment
            runtime._evaluation_agent = original_factory
            runtime._execute_macro = original_execute
            batch = dict(runtime.end_validation_batch())
        if batch.get("training_agent_unchanged") is not True:
            raise FilmstripTraceError("filmstrip replay mutated the frozen agent")
    if common_snapshot is None or len(macros) != macro_limit:
        raise FilmstripTraceError(
            f"{label} trace incomplete: common={common_snapshot is not None}, macros={len(macros)}"
        )
    return {
        "label": label,
        "lambda": fixed_lambda,
        "common_snapshot": common_snapshot,
        "macros": tuple(macros),
    }


def collect(*, probe_path: Path, macro_limit: int, device: str) -> dict:
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    selected = probe.get("selected_probe")
    if (
        probe.get("exactly_reproduces_persisted_confirmation_row") is not True
        or probe.get("probe_execution_device") not in {"cuda", "cpu"}
        or not isinstance(selected, dict)
        or not isinstance(selected.get("occupied_storage_cells"), int)
        or not 0 <= int(selected["occupied_storage_cells"]) <= 8
    ):
        raise FilmstripTraceError("input is not the authenticated CUDA half-full probe")
    model_seed = int(probe["model_seed"])
    instance_seed = int(probe["instance_seed"])
    rng_index = int(probe["policy_rng_index"])
    contract = confirmation.prepare(
        confirmation.DEFAULT_OUTPUT, confirmation.DEFAULT_PARENT, device="cuda"
    )
    manifest = confirmation._validate_manifest(confirmation.DEFAULT_OUTPUT, contract)
    entries = {int(item["instance_seed"]): item for item in manifest["instances"]}
    parent = development._parent(confirmation.DEFAULT_PARENT, model_seed)
    arms = {
        "lambda0": _trace_arm(
            label="lambda0",
            fixed_lambda=0.0,
            parent=parent,
            entry=entries[instance_seed],
            selected=selected,
            model_seed=model_seed,
            instance_seed=instance_seed,
            rng_index=rng_index,
            macro_limit=macro_limit,
            device=device,
        ),
        "lambda005": _trace_arm(
            label="lambda005",
            fixed_lambda=0.05,
            parent=parent,
            entry=entries[instance_seed],
            selected=selected,
            model_seed=model_seed,
            instance_seed=instance_seed,
            rng_index=rng_index,
            macro_limit=macro_limit,
            device=device,
        ),
    }
    if mechanism._json_safe(arms["lambda0"]["common_snapshot"]) != mechanism._json_safe(
        arms["lambda005"]["common_snapshot"]
    ):
        raise FilmstripTraceError("the two arms do not share the same fork state")
    result = {
        "protocol": PROTOCOL,
        "role": "post_hoc_visual_process_not_performance_evidence",
        "source_probe": str(probe_path.resolve()),
        "source_probe_sha256": probe["probe_sha256"],
        "model_seed": model_seed,
        "instance_seed": instance_seed,
        "policy_rng_index": rng_index,
        "policy_rng_seed": int(probe["policy_rng_seed"]),
        "fork_decision_index": int(selected["decision_index"]),
        "fork_occupied_storage_cells": int(selected["occupied_storage_cells"]),
        "macro_count_per_arm": macro_limit,
        "arms": arms,
        "training_or_learning": False,
    }
    result["trace_sha256"] = _canonical_sha(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--macros", type=int, default=5, choices=range(1, 33))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = collect(
        probe_path=args.probe,
        macro_limit=args.macros,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "trace_sha256": result["trace_sha256"],
        "actions": {
            key: [macro["selected_action_type"] for macro in arm["macros"]]
            for key, arm in result["arms"].items()
        },
        "rehandles": {
            key: sum(int(macro["physical_rehandles"]) for macro in arm["macros"])
            for key, arm in result["arms"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
