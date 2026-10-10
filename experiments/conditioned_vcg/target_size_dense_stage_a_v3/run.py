#!/usr/bin/env python3
"""Stage-A v3 orchestration using the verified no-gradient TD acceleration."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch

import train_vcg_dense_proper as dense
from viability_graph_episodic_audit import EpisodicViabilityGraphHierarchyAgent

from experiments.conditioned_vcg.target_size_dense_stage_a_v2 import run as v2run
from experiments.conditioned_vcg.target_size_dense_stage_a_v3 import acceleration
from experiments.conditioned_vcg.target_size_dense_stage_a_v3 import protocol as p


def _with_protocol(function, *args, **kwargs):
    with patch.object(v2run, "p", p):
        return function(*args, **kwargs)


def train(args):
    original_result = dense._result_payload
    original_checkpoint = EpisodicViabilityGraphHierarchyAgent.checkpoint
    original_validate = dense._validate_resume

    def result_payload(*positional, **keywords):
        value = original_result(*positional, **keywords)
        value["td_backup_acceleration"] = acceleration.audit_payload(
            keywords["agent"]
        )
        return value

    def checkpoint(agent, *positional, **keywords):
        value = original_checkpoint(agent, *positional, **keywords)
        value["td_backup_implementation"] = p.TD_BACKUP_IMPLEMENTATION
        value["td_backup_acceleration_audit"] = acceleration.audit_payload(agent)
        return value

    def validate_resume(payload, contract):
        original_validate(payload, contract)
        if payload.get("td_backup_implementation") != p.TD_BACKUP_IMPLEMENTATION:
            raise ValueError("checkpoint was not produced by Stage-A v3 deduplication")

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                EpisodicViabilityGraphHierarchyAgent,
                "_td_batch",
                acceleration.accelerated_episodic_td_batch,
            )
        )
        stack.enter_context(
            patch.object(
                EpisodicViabilityGraphHierarchyAgent,
                "checkpoint",
                checkpoint,
            )
        )
        stack.enter_context(patch.object(dense, "_result_payload", result_payload))
        stack.enter_context(patch.object(dense, "_validate_resume", validate_resume))
        return _with_protocol(
            v2run.train_stage_a,
            args.output,
            args.scenario,
            args.model_seed,
            device_name=args.device,
            stop_after_episode=args.stop_after_episode,
            allow_cpu_pilot=args.allow_cpu_pilot,
        )


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("command", choices=("hardware", "self-check", "prepare", "train-stage-a"))
    value.add_argument("--output", type=Path, default=p.DEFAULT_OUTPUT)
    value.add_argument("--scenario", choices=tuple(p.SCENARIOS))
    value.add_argument("--model-seed", type=int, choices=p.MODEL_SEEDS)
    value.add_argument("--device", choices=("cuda", "auto", "cpu"), default="cuda")
    value.add_argument("--stop-after-episode", type=int)
    value.add_argument("--allow-cpu-pilot", action="store_true")
    return value


def main():
    args = parser().parse_args()
    if args.command == "hardware":
        result = v2run.hardware()
    elif args.command == "self-check":
        result = _with_protocol(v2run.self_check)
    elif args.command == "prepare":
        result = _with_protocol(v2run.prepare, args.output)
    else:
        if args.scenario is None or args.model_seed is None:
            raise ValueError("train-stage-a requires --scenario and --model-seed")
        result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
