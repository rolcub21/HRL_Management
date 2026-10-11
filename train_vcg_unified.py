#!/usr/bin/env python3
"""Unified VCG development trainer.

This module deliberately reuses the VCG V2.3 controller and changes one
thing only: how the multiplier on the already-trained physical-rehandle head
is controlled.

``vcg``
    Keep lambda identically zero.  Q_op and Q_N are both trained, but action
    merit is Q_op.

``vcg-handling-constraint``
    Use the V2.3 projected complete-block dual update with a declared B_N
    physical rehandles per 100 required deliveries.  Action merit is
    Q_op - lambda * Q_N.

``vcg-fixed-handling-weight``
    Keep a declared positive lambda fixed after the common critic warm-up.
    This directly maps operating points without making a budget-convergence
    claim.

The graph encoder, two-headed critic, safe frontier, Hold rule, SMDP backup,
discounts, temperatures, stochastic policy, training instances, action RNGs,
validation instances, and validation action RNGs are inherited unchanged
from V2.3.  Both variants use the terminal episode-200 checkpoint; there is
no constraint-dependent best-checkpoint selection in this development run.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib
import json
from pathlib import Path
from statistics import fmean
from typing import Mapping, Optional, Sequence

import torch

import train_vcg_constrained_v2_1 as atomic_io
import train_vcg_constrained_v2_3 as v23


METHOD_VERSION = "vcg_unified_v1"
TRAINING_PROTOCOL = "vcg_unified_v1_paired_lambda_control_development"
CHECKPOINT_FAMILY = "vcg_unified_v1_terminal_development"
CONTRACT_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1

VCG = "vcg"
VCG_HANDLING_CONSTRAINT = "vcg-handling-constraint"
VCG_FIXED_HANDLING_WEIGHT = "vcg-fixed-handling-weight"
VARIANTS = (VCG, VCG_HANDLING_CONSTRAINT, VCG_FIXED_HANDLING_WEIGHT)
HANDLING_BUDGET_PER_100 = 20.0
MODEL_SEED = 14
TRAIN_SEED_BASE = 61_004_000
TRAINING_POLICY_RNG_BASE = 610_004_000
REPLAY_RNG_SEED = 610_104_010
VALIDATION_POLICY_RNG_BASE = 621_000_000


class UnifiedVCGError(ValueError):
    """Raised when the paired unified-VCG protocol is violated."""


@contextmanager
def activated_unified_seed_profile():
    """Install the one fresh paired seed profile, then restore V2.3.

    V2.3 intentionally freezes its original development seed constants at
    module scope.  The unified experiment needs a fresh *common* namespace,
    so both arms enter this same narrowly-scoped profile.  No architecture or
    policy constant is changed.
    """

    names = (
        "DEFAULT_MODEL_SEED",
        "FRESH_TRAIN_SEED_BASE",
        "TRAINING_POLICY_RNG_BASE",
        "REPLAY_RNG_SEED",
        "VALIDATION_POLICY_RNG_BASE",
    )
    snapshot = {name: getattr(v23, name) for name in names}
    core = importlib.import_module("viability_graph_constrained_v2_3")
    core_replay_seed = core.REPLAY_RNG_SEED
    try:
        v23.DEFAULT_MODEL_SEED = MODEL_SEED
        v23.FRESH_TRAIN_SEED_BASE = TRAIN_SEED_BASE
        v23.TRAINING_POLICY_RNG_BASE = TRAINING_POLICY_RNG_BASE
        v23.REPLAY_RNG_SEED = REPLAY_RNG_SEED
        v23.VALIDATION_POLICY_RNG_BASE = VALIDATION_POLICY_RNG_BASE
        core.REPLAY_RNG_SEED = REPLAY_RNG_SEED
        yield
    finally:
        core.REPLAY_RNG_SEED = core_replay_seed
        for name, value in snapshot.items():
            setattr(v23, name, value)


def _canonical_hash(value: Mapping) -> str:
    return v23.contract_hash(value)


def _validate_variant(value: str) -> str:
    if value not in VARIANTS:
        raise UnifiedVCGError(
            f"variant must be one of {VARIANTS!r}; received {value!r}"
        )
    return value


@dataclass(frozen=True)
class UnifiedVCGVariant:
    """The sole treatment variable in the unified comparison."""

    name: str
    handling_budget_per_100: float = HANDLING_BUDGET_PER_100
    fixed_lambda: Optional[float] = None

    def __post_init__(self) -> None:
        _validate_variant(self.name)
        budget = float(self.handling_budget_per_100)
        if not torch.isfinite(torch.tensor(budget)) or budget < 0.0:
            raise UnifiedVCGError("handling budget must be finite and nonnegative")
        if self.name == VCG_FIXED_HANDLING_WEIGHT:
            if self.fixed_lambda is None:
                raise UnifiedVCGError("fixed-weight VCG requires --fixed-lambda")
            value = float(self.fixed_lambda)
            if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
                raise UnifiedVCGError("fixed lambda must be finite and positive")
        elif self.fixed_lambda not in (None, 0, 0.0):
            raise UnifiedVCGError("fixed lambda is valid only for fixed-weight VCG")

    @property
    def constrained(self) -> bool:
        return self.name == VCG_HANDLING_CONSTRAINT

    @property
    def fixed_weight(self) -> bool:
        return self.name == VCG_FIXED_HANDLING_WEIGHT

    @property
    def uses_handling_weight(self) -> bool:
        return self.constrained or self.fixed_weight

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "q_operational_trained": True,
            "q_physical_rehandle_trained": True,
            "q_physical_rehandle_monitored": True,
            "cost_loss_weight": 1.0,
            "decision_merit": (
                "Q_op(s,c) - lambda * Q_N(s,c)"
                if self.uses_handling_weight
                else "Q_op(s,c) (lambda identically zero)"
            ),
            "lambda_control": (
                "projected_complete_block_dual_update"
                if self.constrained
                else ("fixed_positive_after_warmup" if self.fixed_weight else "fixed_zero")
            ),
            "handling_budget_per_100_required_deliveries": (
                float(self.handling_budget_per_100) if self.constrained else None
            ),
            "fixed_lambda_after_warmup": (
                float(self.fixed_lambda) if self.fixed_weight else None
            ),
        }


class UnifiedLambdaController:
    """One block-clock implementation with fixed or adaptive lambda control.

    Residuals are computed and recorded for every variant.  Only the
    constrained variant applies the projected update.  Keeping this logic in
    one class prevents the unconstrained run from acquiring a separate
    training path.
    """

    def __init__(self, variant: UnifiedVCGVariant) -> None:
        if not isinstance(variant, UnifiedVCGVariant):
            raise TypeError("variant must be UnifiedVCGVariant")
        self.variant = variant
        self.config = v23.V23DualConfig(
            budget_per_100_required_deliveries=variant.handling_budget_per_100,
            learning_rate=0.01,
            lambda_initial=0.0,
            lambda_max=20.0,
        )
        self.lambda_value = (
            float(variant.fixed_lambda) if variant.fixed_weight else 0.0
        )
        self.observed_block_count = 0
        self.applied_update_count = 0
        self.last_completed_block = v23.WARMUP_BLOCKS
        self.last_mean_residual: Optional[float] = None
        self.saturation_count = 0

    def process_complete_block(
        self,
        episodes: Sequence[Mapping],
        *,
        block_number: int,
        terminal: bool,
    ) -> dict:
        block = int(block_number)
        if block != self.last_completed_block + 1:
            raise UnifiedVCGError("lambda blocks must be processed in order")
        if block <= v23.WARMUP_BLOCKS:
            raise UnifiedVCGError("warm-up episodes must not enter the dual clock")
        if len(episodes) != v23.BLOCK_EPISODES:
            raise UnifiedVCGError("lambda control requires one complete block")

        residuals = tuple(
            v23.episode_budget_residual(
                int(run["physical_rehandles"]),
                int(run["required_deliveries"]),
                self.config,
            )
            for run in episodes
        )
        before = float(self.lambda_value)
        mean_residual = float(fmean(residuals))
        dual_proposed = min(
            max(before + self.config.learning_rate * mean_residual, 0.0),
            self.config.lambda_max,
        )
        should_apply = bool(self.variant.constrained and not terminal)
        after = float(dual_proposed if should_apply else before)
        if not self.variant.uses_handling_weight:
            after = 0.0

        self.lambda_value = after
        self.observed_block_count += 1
        self.applied_update_count += int(should_apply)
        self.last_completed_block = block
        self.last_mean_residual = mean_residual
        saturated = bool(
            should_apply
            and after == self.config.lambda_max
            and mean_residual > 0.0
        )
        self.saturation_count += int(saturated)
        return {
            "protocol": "unified_optional_projected_complete_block_dual",
            "variant": self.variant.name,
            "block_number": block,
            "lambda_before": before,
            "lambda_proposed": float(
                before if self.variant.fixed_weight else dual_proposed
            ),
            "lambda_after": after,
            "mean_budget_residual": mean_residual,
            "episode_residuals": residuals,
            "episode_count": len(residuals),
            "applied": should_apply,
            "terminal": bool(terminal),
            "reason_not_applied": (
                None
                if should_apply
                else (
                    "lambda_fixed_zero"
                    if not self.variant.uses_handling_weight
                    else (
                        "fixed_handling_weight"
                        if self.variant.fixed_weight
                        else "no_subsequent_primal_block"
                    )
                )
            ),
            "saturated_with_positive_residual": saturated,
        }

    def state_dict(self) -> dict:
        return {
            "protocol": "unified_optional_projected_complete_block_dual",
            "variant": self.variant.name,
            "config": asdict(self.config),
            "lambda_value": float(self.lambda_value),
            "observed_block_count": int(self.observed_block_count),
            "applied_update_count": int(self.applied_update_count),
            "last_completed_block": int(self.last_completed_block),
            "last_mean_residual": self.last_mean_residual,
            "saturation_count": int(self.saturation_count),
        }


def _shared_contract(base_contract: Mapping) -> dict:
    agent = dict(base_contract["agent_config"])
    if float(agent.get("cost_loss_weight", -1.0)) != 1.0:
        raise UnifiedVCGError("Q_N must have the frozen nonzero loss weight")
    return {
        "implementation_basis": "vcg_constrained_v2_3_gamma1",
        "base_contract_sha256": base_contract["contract_sha256"],
        "controller": base_contract["controller"],
        "backup_version": base_contract["backup_version"],
        "policy_schedule_protocol": base_contract["policy_schedule_protocol"],
        "validation_protocol": base_contract["validation_protocol"],
        "model_seed": base_contract["model_seed"],
        "train_seed_base": base_contract["train_seed_base"],
        "episodes": base_contract["episodes"],
        "training_policy_rng_formula": base_contract["training_policy_rng_formula"],
        "replay_rng_seed": base_contract["replay_rng_seed"],
        "validation_seeds": base_contract["validation_seeds"],
        "validation_policy_rng_grid": base_contract["validation_policy_rng_grid"],
        "environment": base_contract["environment"],
        "max_steps": base_contract["max_steps"],
        "dense_objective_spec": base_contract["dense_objective_spec"],
        "certified_hold": base_contract["certified_hold"],
        "viability_search_config": base_contract["viability_search_config"],
        "agent_config": agent,
        "gamma_operational": base_contract["gamma_operational"],
        "gamma_rehandle": base_contract["gamma_rehandle"],
        "reward_scale": base_contract["reward_scale"],
        "temperature_schedule": base_contract["temperature_schedule"],
        "policy_realization": v23.POLICY_REALIZATION,
        "same_policy_for_behavior_backup_validation_deployment": True,
        "terminal_checkpoint_comparison": True,
        "constraint_dependent_checkpoint_selection": False,
        "q_operational_trained": True,
        "q_physical_rehandle_trained": True,
        "q_physical_rehandle_monitored": True,
    }


def build_unified_contract(args: argparse.Namespace) -> tuple[dict, dict]:
    """Return ``(unified_contract, runtime_v23_contract)``.

    The runtime contract is deliberately preserved byte-for-byte between the
    two variants.  The top-level treatment record is the only difference.
    """

    variant = UnifiedVCGVariant(
        args.variant,
        handling_budget_per_100=float(args.rehandle_budget_per_100),
        fixed_lambda=args.fixed_lambda,
    )
    with activated_unified_seed_profile():
        base_contract = v23.build_training_contract(args)
    # The inherited V2.3 builder historically rendered this one field as a
    # literal seed-10 string even though its RNG function uses the active
    # profile.  Bind the declaration to the fresh unified profile too.
    base_contract["training_policy_rng_formula"] = (
        f"{TRAINING_POLICY_RNG_BASE} + episode_number - 1"
    )
    base_contract.pop("contract_sha256", None)
    base_contract["contract_sha256"] = _canonical_hash(base_contract)
    shared = _shared_contract(base_contract)
    shared_hash = _canonical_hash(shared)
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "training_protocol": TRAINING_PROTOCOL,
        "method_version": METHOD_VERSION,
        "experiment_role": "paired_unified_constraint_ablation_development",
        "variant": variant.to_dict(),
        "shared_configuration": shared,
        "shared_configuration_sha256": shared_hash,
        "only_treatment_difference": "lambda control",
        "base_v2_3_runtime_contract": base_contract,
        "terminal_checkpoint_only": True,
        "development_only": True,
        "deployment_checkpoint_eligible": False,
        "final_86xxx_panel_opened": False,
    }
    contract["contract_sha256"] = _canonical_hash(contract)
    return contract, base_contract


def _decorate_run(row: Mapping, variant: UnifiedVCGVariant) -> dict:
    result = dict(row)
    result.update(
        unified_method_version=METHOD_VERSION,
        unified_variant=variant.name,
        q_physical_rehandle_trained=True,
        q_physical_rehandle_monitored=True,
    )
    return result


def _validation_rows(runtime, args, schedule: Mapping, dual_lambda: float) -> tuple[list[dict], dict]:
    nuisance_grid = tuple(
        tuple(v23.validation_policy_rng_seed(i, j) for j in range(v23.VALIDATION_POLICY_RNG_COUNT))
        for i in range(len(v23.DEFAULT_VALIDATION_SEEDS))
    )
    raw_rows: list[dict] = []
    runtime.begin_validation_batch()
    try:
        for instance_index, instance_seed in enumerate(args.validation_seeds):
            for rng_index, rng_seed in enumerate(nuisance_grid[instance_index]):
                raw = dict(runtime.run_episode(
                    instance_seed=int(instance_seed),
                    training=False,
                    max_steps=int(args.max_steps),
                    policy_rng_index=rng_index,
                    policy_rng_seed=int(rng_seed),
                ))
                if raw.get("instance_seed") != int(instance_seed):
                    raise RuntimeError("runtime did not echo the validation seed")
                raw["instance_index"] = instance_index
                raw_rows.append(raw)
    finally:
        batch_audit = dict(runtime.end_validation_batch())
    if batch_audit.get("training_agent_unchanged") is not True:
        raise RuntimeError("validation mutated the training agent")

    rows = []
    for index, raw in enumerate(raw_rows):
        raw["validation_batch_state_unchanged"] = True
        rows.append(v23._normalize_authenticated_run(
            raw,
            index=index,
            schedule=schedule,
            dual_lambda=dual_lambda,
            training=False,
        ))
    return rows, batch_audit


def _instance_identities(rows: Sequence[Mapping], validation_seeds: Sequence[int]) -> tuple[dict, ...]:
    identities: dict[int, dict] = {}
    for row in rows:
        seed = int(row["instance_seed"])
        identity = {
            "instance_seed": seed,
            "episode_instance_id": row["episode_instance_id"],
            "schedule_id": row["schedule_id"],
            "episode_instance_sha256": row["episode_instance_sha256"],
        }
        if seed in identities and identities[seed] != identity:
            raise RuntimeError("validation instance drifted across action RNG rows")
        identities[seed] = identity
    ordered = tuple(identities[int(seed)] for seed in validation_seeds)
    if ordered != v23.PARENT_V2_2_VALIDATION_INSTANCE_IDENTITIES:
        raise RuntimeError("unified VCG did not use the frozen V2.3 validation instances")
    return ordered


def _checkpoint_payload(
    *,
    contract: Mapping,
    runtime,
    controller: UnifiedLambdaController,
    completed_episodes: int,
    schedule: Mapping,
    include_replay: bool,
) -> dict:
    return {
        "checkpoint_family": CHECKPOINT_FAMILY,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method_version": METHOD_VERSION,
        "unified_variant": controller.variant.name,
        "completed_episodes": int(completed_episodes),
        "training_contract_sha256": contract["contract_sha256"],
        "shared_configuration_sha256": contract["shared_configuration_sha256"],
        "terminal_checkpoint": completed_episodes == v23.TOTAL_EPISODES,
        "development_only": True,
        "deployment_checkpoint_eligible": False,
        "final_86xxx_panel_opened": False,
        "lambda_state": controller.state_dict(),
        "runtime_lambda": float(runtime.dual_lambda),
        "schedule_state": dict(schedule),
        "q_operational_trained": True,
        "q_physical_rehandle_trained": True,
        "q_physical_rehandle_monitored": True,
        "base_v2_3_core_contract": dict(runtime.core_contract),
        "agent_state": runtime.checkpoint_state(include_replay=include_replay),
    }


def run_unified_training(args: argparse.Namespace, *, runtime=None) -> dict:
    with activated_unified_seed_profile():
        return _run_unified_training_active(args, runtime=runtime)


def _run_unified_training_active(args: argparse.Namespace, *, runtime=None) -> dict:
    contract, base_contract = build_unified_contract(args)
    variant = UnifiedVCGVariant(
        args.variant,
        handling_budget_per_100=float(args.rehandle_budget_per_100),
        fixed_lambda=args.fixed_lambda,
    )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise UnifiedVCGError("unified VCG requires a fresh output directory")
    atomic_io._atomic_json(contract, output_dir / "training-contract.json")
    if args.contract_only:
        summary = {
            "status": "contract_only",
            "method_version": METHOD_VERSION,
            "variant": variant.name,
            "episodes_executed": 0,
            "shared_configuration_sha256": contract["shared_configuration_sha256"],
            "final_86xxx_panel_opened": False,
        }
        atomic_io._atomic_json(summary, output_dir / "training-summary.json")
        return summary

    runtime = runtime or v23.load_default_runtime(args, base_contract)
    v23._validate_runtime(runtime, base_contract)
    controller = UnifiedLambdaController(variant)
    runtime.set_dual_lambda(0.0)
    training_history: list[dict] = []
    validation_history: list[dict] = []
    lambda_history: list[dict] = []
    pending: list[dict] = []
    validation_manifest: Optional[dict] = None
    latest_path = output_dir / "latest.pth"

    for episode_number in range(1, v23.TOTAL_EPISODES + 1):
        state = v23.schedule_for_episode(episode_number)
        schedule = v23._install_schedule(runtime, state)
        block_lambda = float(
            0.0
            if state.block_number <= v23.WARMUP_BLOCKS
            else controller.lambda_value
        )
        runtime.set_dual_lambda(block_lambda)
        raw = runtime.run_episode(
            instance_seed=int(args.train_seed_base) + episode_number - 1,
            training=True,
            max_steps=int(args.max_steps),
            policy_rng_index=episode_number - 1,
            policy_rng_seed=v23.training_policy_rng_seed(episode_number),
        )
        run = v23._normalize_authenticated_run(
            raw,
            index=episode_number - 1,
            schedule=schedule,
            dual_lambda=block_lambda,
            training=True,
        )
        run.update(training_episode_number=episode_number, block_number=state.block_number)
        run = _decorate_run(run, variant)
        training_history.append(run)
        if state.dual_updates_enabled:
            pending.append(run)
        print(
            f"Unified VCG [{variant.name}] Ep {episode_number:4d} | "
            f"TrainR {run['dense_return']:8.2f} | "
            f"Reh {run['physical_rehandles']:3d}/{run['required_deliveries']:3d} | "
            f"Lambda {block_lambda:7.4f}",
            flush=True,
        )

        validation_summary = None
        checkpoint_schedule = schedule
        # One common terminal look avoids both the runtime cost and the
        # constraint-specific selection confound of V2.3's best-of-seven gate.
        if episode_number == v23.TOTAL_EPISODES:
            checkpoint_schedule = v23._install_schedule(
                runtime,
                v23.schedule_for_episode(episode_number, validation=True),
            )
            rows, batch_audit = _validation_rows(
                runtime, args, checkpoint_schedule, block_lambda
            )
            identities = _instance_identities(rows, args.validation_seeds)
            if validation_manifest is None:
                validation_manifest = {
                    "schema_version": 1,
                    "validation_protocol": v23.VALIDATION_PROTOCOL,
                    "common_across_unified_variants": True,
                    "instances": identities,
                    "final_86xxx_panel_opened": False,
                }
                validation_manifest["manifest_sha256"] = _canonical_hash(validation_manifest)
                atomic_io._atomic_json(
                    validation_manifest,
                    output_dir / "validation-instance-manifest.json",
                )
            elif identities != tuple(validation_manifest["instances"]):
                raise RuntimeError("validation EpisodeInstances drifted across looks")

            validation_summary = v23.summarize_validation(
                rows,
                controller.config,
                checkpoint_episode=episode_number,
                validation_lambda=block_lambda,
                schedule_state=checkpoint_schedule,
                validation_seeds=args.validation_seeds,
                positive_residual_saturation_count=controller.saturation_count,
            )
            validation_summary.update(
                unified_method_version=METHOD_VERSION,
                unified_variant=variant.name,
                handling_budget_role=("constraint" if variant.constrained else "monitor_only"),
                terminal_checkpoint_comparison=episode_number == v23.TOTAL_EPISODES,
                validation_batch_audit=batch_audit,
            )
            validation_history.append(validation_summary)
            ledger = {
                "schema_version": 1,
                "method_version": METHOD_VERSION,
                "unified_variant": variant.name,
                "checkpoint_episode": episode_number,
                "validated_lambda": block_lambda,
                "rows": tuple(_decorate_run(row, variant) for row in rows),
                "row_count": len(rows),
                "complete_case_filtering_used": False,
            }
            ledger["ledger_sha256"] = _canonical_hash(ledger)
            atomic_io._atomic_json(
                ledger,
                output_dir / "validation-ledger" / f"episode-{episode_number:04d}.json",
            )
            mae = validation_summary["mean_absolute_error"]
            mae_text = "NA" if mae is None else f"{mae:7.2f}"
            print(
                f"Unified VCG [{variant.name}] Validation Ep {episode_number:4d} | "
                f"R {validation_summary['mean_dense_return']:8.2f} | "
                f"MAE {mae_text} | "
                f"Reh/100 {validation_summary['expected_physical_rehandles_per_100_required_deliveries']:7.2f}",
                flush=True,
            )

        if episode_number % v23.BLOCK_EPISODES == 0 and state.dual_updates_enabled:
            update = controller.process_complete_block(
                pending,
                block_number=state.block_number,
                terminal=state.block_number == v23.TOTAL_BLOCKS,
            )
            lambda_history.append(update)
            pending.clear()
            runtime.set_dual_lambda(controller.lambda_value)

        if episode_number == v23.TOTAL_EPISODES:
            atomic_io._atomic_torch(
                _checkpoint_payload(
                    contract=contract,
                    runtime=runtime,
                    controller=controller,
                    completed_episodes=episode_number,
                    schedule=checkpoint_schedule,
                    include_replay=True,
                ),
                latest_path,
            )
            atomic_io._atomic_json(training_history, output_dir / "training-history.json")
            atomic_io._atomic_json(validation_history, output_dir / "validation-history.json")
            atomic_io._atomic_json(lambda_history, output_dir / "lambda-history.json")

    if pending:
        raise RuntimeError("unified VCG ended with an incomplete lambda block")
    if len(validation_history) != 1:
        raise RuntimeError("unified VCG requires exactly one terminal validation")
    if variant.name == VCG and controller.lambda_value != 0.0:
        raise RuntimeError("unconstrained unified VCG changed lambda")

    final_path = output_dir / "final-model.pth"
    atomic_io._atomic_torch(
        _checkpoint_payload(
            contract=contract,
            runtime=runtime,
            controller=controller,
            completed_episodes=v23.TOTAL_EPISODES,
            schedule=checkpoint_schedule,
            include_replay=False,
        ),
        final_path,
    )
    summary = {
        "status": "complete",
        "method_version": METHOD_VERSION,
        "variant": variant.name,
        "completed_training_episodes": len(training_history),
        "terminal_checkpoint_only": True,
        "final_validation": validation_history[-1],
        "lambda_state": controller.state_dict(),
        "training_contract": str(output_dir / "training-contract.json"),
        "final_checkpoint": str(final_path),
        "latest_checkpoint": str(latest_path),
        "shared_configuration_sha256": contract["shared_configuration_sha256"],
        "development_only": True,
        "deployment_checkpoint_eligible": False,
        "final_86xxx_panel_opened": False,
    }
    atomic_io._atomic_json(summary, output_dir / "training-summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = v23.build_parser()
    parser.description = "Train one unified-VCG lambda-control variant"
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--fixed-lambda", type=float)
    parser.set_defaults(model_seed=MODEL_SEED, train_seed_base=TRAIN_SEED_BASE)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_unified_training(args)
    print(json.dumps(v23._json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "HANDLING_BUDGET_PER_100",
    "METHOD_VERSION",
    "MODEL_SEED",
    "REPLAY_RNG_SEED",
    "TRAINING_POLICY_RNG_BASE",
    "TRAIN_SEED_BASE",
    "TRAINING_PROTOCOL",
    "UnifiedLambdaController",
    "UnifiedVCGError",
    "UnifiedVCGVariant",
    "VCG",
    "VCG_HANDLING_CONSTRAINT",
    "VCG_FIXED_HANDLING_WEIGHT",
    "VARIANTS",
    "VALIDATION_POLICY_RNG_BASE",
    "activated_unified_seed_profile",
    "build_parser",
    "build_unified_contract",
    "run_unified_training",
]
