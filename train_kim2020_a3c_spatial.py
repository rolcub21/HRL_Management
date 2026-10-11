#!/usr/bin/env python3
"""Train the Kim et al. (2020)-inspired spatial A3C adaptation.

This is a conceptual adaptation, not an exact reproduction of the paper.  It
learns only the storage-cell decision and leaves transport, retrieval, timing,
and the shared feasibility mask to the neutral Track-A protocol.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os

# CUDA reads this setting when its runtime initializes.  Establish a
# deterministic default before importing torch while preserving either
# supported value supplied by the caller.
_CUBLAS_WORKSPACE_CONFIG_ENV = "CUBLAS_WORKSPACE_CONFIG"
_DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_SUPPORTED_CUBLAS_WORKSPACE_CONFIGS = (":4096:8", ":16:8")
os.environ.setdefault(
    _CUBLAS_WORKSPACE_CONFIG_ENV, _DEFAULT_CUBLAS_WORKSPACE_CONFIG
)

from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
import torch

from example.episode_instance import EpisodeInstance
from example.small_rooms_env import SmallRoomsEnv
from example.yard_geometry import geometry_metadata, make_shipyard_env
from PSLAP.kim2020_a3c_spatial import (
    DEPLOYMENT_MAP,
    DEPLOYMENT_STOCHASTIC,
    Kim2020A3CSpatialSource,
    Kim2020Config,
)
from PSLAP.online_policy import (
    EXACT_RECOVERY_FALLBACK_CONTRACT,
    EXACT_RECOVERY_MAX_NODES,
)
from PSLAP.track_a import (
    TRACK_A_KIM2020_A3C_SPATIAL,
    run_track_a_episode,
)


TRAINER_VERSION = "kim2020_a3c_spatial_adapted_trainer_v7"
ADAPTATION_CONTRACT = (
    "kim_jeong_shin_2020_spatial_a3c_conceptual_adaptation_v6"
)
VALIDATION_CONTRACT = (
    "fixed_instances_map_once_stochastic_fixed_policy_seed_rollouts_v1"
)
MODEL_SELECTION_CONTRACT = (
    "primary_stochastic_strict_success_then_mean_obstructive_moves_v2"
)
PAPER_DOI = "10.1080/00207543.2020.1748247"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Kim et al. (2020)-inspired spatial A3C baseline. "
            "This is an explicit adaptation, not an exact reproduction."
        )
    )
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=50.0)
    parser.add_argument("--grid-rows", type=int, default=10)
    parser.add_argument("--grid-cols", type=int, default=10)
    parser.add_argument("--exit-width", type=int)
    parser.add_argument("--number-blocks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=Path,
        help=(
            "resume from latest.pth in output-dir; --episodes is the new "
            "total episode target"
        ),
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help=(
            "discount between successive placement decisions in the "
            "terminal episodic Monte Carlo return"
        ),
    )
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--updates-per-episode", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument(
        "--validation-seeds",
        nargs="+",
        type=int,
        default=[10_000, 10_001, 10_002, 10_003, 10_004],
        help="fixed schedule seeds used only for model selection",
    )
    parser.add_argument(
        "--stochastic-rollouts",
        type=int,
        default=5,
        help="independent policy rollouts per fixed validation instance",
    )
    parser.add_argument(
        "--validation-policy-seed-base",
        type=int,
        default=10_000_000,
        help=(
            "separate fixed seed namespace for stochastic validation actions; "
            "these seeds never generate schedules"
        ),
    )
    parser.add_argument(
        "--training-instance-seed-base",
        type=int,
        default=200_000,
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request deterministic PyTorch kernels where available",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/kim2020-a3c-spatial-adapted"),
    )
    args = parser.parse_args(argv)

    positive = {
        "episodes": args.episodes,
        "max-steps": args.max_steps,
        "eval-every": args.eval_every,
        "grid-rows": args.grid_rows,
        "grid-cols": args.grid_cols,
        "number-blocks": args.number_blocks,
        "learning-rate": args.learning_rate,
        "reward-scale": args.reward_scale,
        "grad-clip": args.grad_clip,
        "hidden-channels": args.hidden_channels,
        "updates-per-episode": args.updates_per_episode,
        "stochastic-rollouts": args.stochastic_rollouts,
    }
    invalid_positive = [name for name, value in positive.items() if value <= 0]
    if invalid_positive:
        parser.error(
            "these arguments must be positive: " + ", ".join(invalid_positive)
        )
    if args.grid_rows < 4 or args.grid_cols < 4:
        parser.error("grid dimensions must both be at least 4")
    if args.exit_width is not None and not 1 <= args.exit_width <= args.grid_cols - 2:
        parser.error("exit-width must be between 1 and grid-cols - 2")
    if args.lam < 0:
        parser.error("lambda must be nonnegative")
    if args.mu <= 0:
        parser.error("mu must be positive")
    if not 0.0 < args.gamma <= 1.0:
        parser.error("gamma must be in (0, 1]")
    if args.weight_decay < 0 or args.entropy_coef < 0 or args.value_coef < 0:
        parser.error(
            "weight-decay, entropy-coef, and value-coef must be nonnegative"
        )
    if not args.validation_seeds:
        parser.error("at least one validation seed is required")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        parser.error("validation-seeds must be unique")
    if any(seed < 0 for seed in args.validation_seeds):
        parser.error("validation-seeds must be nonnegative")
    if args.validation_policy_seed_base < 0:
        parser.error("validation-policy-seed-base must be nonnegative")
    if args.training_instance_seed_base < 0:
        parser.error("training-instance-seed-base must be nonnegative")
    if not 0 <= args.seed <= np.iinfo(np.uint32).max:
        parser.error("seed must be in [0, 2**32 - 1]")
    training_seed_start = (
        args.training_instance_seed_base + args.seed * 1_000_000 + 1
    )
    training_seed_end = training_seed_start + args.episodes - 1
    if any(
        training_seed_start <= seed <= training_seed_end
        for seed in args.validation_seeds
    ):
        parser.error(
            "validation schedule seeds must not overlap training instances"
        )
    return args


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def validate_cuda_determinism(
    *, device: str, deterministic_algorithms: bool
) -> str | None:
    """Validate and return the process's effective cuBLAS configuration."""

    workspace_config = os.environ.get(_CUBLAS_WORKSPACE_CONFIG_ENV)
    if (
        device == "cuda"
        and deterministic_algorithms
        and workspace_config not in _SUPPORTED_CUBLAS_WORKSPACE_CONFIGS
    ):
        supported = " or ".join(_SUPPORTED_CUBLAS_WORKSPACE_CONFIGS)
        raise RuntimeError(
            "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG to be set "
            f"before CUDA initializes to {supported}; got "
            f"{workspace_config!r}. Restart the process with a supported "
            "value."
        )
    return workspace_config


def seed_everything(seed: int, *, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_algorithms


def make_env(args: argparse.Namespace) -> SmallRoomsEnv:
    return make_shipyard_env(
        arrival_rate=args.lam,
        proc_mean=args.mu,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        exit_width=args.exit_width,
        number_blocks=args.number_blocks,
    )


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@contextmanager
def preserve_global_rng_state():
    """Prevent validation model construction from perturbing training RNGs."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def build_validation_instances(
    env: SmallRoomsEnv, seeds: Iterable[int]
) -> tuple[tuple[int, EpisodeInstance], ...]:
    instances = []
    for seed in seeds:
        instance = env.sample_episode_instance(int(seed))
        instance.validate_for(env)
        instances.append((int(seed), instance))
    return tuple(instances)


def write_validation_instances(
    output_dir: Path,
    instances: Sequence[tuple[int, EpisodeInstance]],
) -> list[dict]:
    instance_dir = output_dir / "validation-instances"
    instance_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed, instance in instances:
        path = instance_dir / f"seed-{seed}.json"
        content = instance.to_json() + "\n"
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f"refusing to replace changed instance: {path}")
        path.write_text(content)
        manifest.append(
            {
                "seed": seed,
                "instance_id": instance.instance_id,
                "schedule_id": instance.schedule_id,
                "path": str(path.resolve()),
            }
        )
    _atomic_json(output_dir / "validation-instances.json", manifest)
    return manifest


def _policy_seed(
    base: int,
    *,
    rollout_index: int,
) -> int:
    # This namespace is deliberately independent of the instance seed.  It is
    # crossed with every fixed schedule and reused across validation epochs,
    # making both schedule and policy randomness paired between checkpoints.
    return int(base + rollout_index)


def _run_row(
    payload: dict,
    args: argparse.Namespace,
    *,
    instance_seed: int,
    instance: EpisodeInstance,
    deployment_mode: str,
    policy_seed: int,
    rollout_index: int,
) -> dict:
    env = make_env(args)
    source = Kim2020A3CSpatialSource.from_checkpoint(
        env,
        payload,
        learning_enabled=False,
        device="cpu",
        seed=policy_seed,
        policy_seed=policy_seed,
        deployment_mode=deployment_mode,
    )
    result = run_track_a_episode(
        env,
        TRACK_A_KIM2020_A3C_SPATIAL,
        max_steps=args.max_steps,
        episode_instance=instance,
        assignment_source=source,
    )
    source_audit = source.audit()
    errors = np.asarray(result["delivery_errors"], dtype=float)
    return {
        "deployment_mode": deployment_mode,
        "instance_seed": int(instance_seed),
        "instance_id": result["instance_id"],
        "policy_seed": int(policy_seed),
        "rollout_index": int(rollout_index),
        "return": float(result["return"]),
        "success": float(result["success"]),
        "strict_method_success": float(result["strict_method_success"]),
        "completion_fraction": float(result["completion_fraction"]),
        "steps": int(result["steps"]),
        "obstructive_moves": int(result["obstructive_moves"]),
        "obstructive_moves_per_delivered_block": float(
            result["obstructive_moves_per_delivered_block"]
        ),
        "mean_absolute_error": (
            float(np.abs(errors).mean()) if errors.size else None
        ),
        "mean_signed_delivery_error": (
            float(errors.mean()) if errors.size else None
        ),
        "mean_tardiness": (
            float(np.maximum(errors, 0.0).mean()) if errors.size else None
        ),
        "delivery_count": int(result["delivery_count"]),
        "invalid_assignment_count": int(result["invalid_assignment_count"]),
        "fallback_count": int(result["fallback_count"]),
        "infeasible_epoch_count": int(result["infeasible_epoch_count"]),
        "retrieval_live_plan_failure_count": int(
            result["retrieval_live_plan_failure_count"]
        ),
        "inbound_approach_defer_count": int(
            result["inbound_approach_defer_count"]
        ),
        "exact_recovery_search_count": int(
            result["exact_recovery_search_count"]
        ),
        "exact_recovery_fallback_count": int(
            result["exact_recovery_fallback_count"]
        ),
        "exact_recovery_failure_count": int(
            result["exact_recovery_failure_count"]
        ),
        "exact_recovery_explored_nodes": int(
            result["exact_recovery_explored_nodes"]
        ),
        "source_setup_seconds": float(result["source_setup_seconds"]),
        "assignment_planning_seconds": float(
            result["assignment_planning_seconds"]
        ),
        "episode_loop_seconds": float(result["episode_loop_seconds"]),
        "source_audit": json_safe(source_audit),
    }


def _mean_present(rows: Sequence[dict], key: str):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def summarize_rows(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty validation set")
    obstructive = np.asarray(
        [row["obstructive_moves"] for row in rows], dtype=float
    )
    return {
        "run_count": len(rows),
        "instance_count": len({row["instance_id"] for row in rows}),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "strict_method_success_rate": float(
            np.mean([row["strict_method_success"] for row in rows])
        ),
        "all_strict_method_success": bool(
            all(row["strict_method_success"] == 1.0 for row in rows)
        ),
        "mean_obstructive_moves": float(obstructive.mean()),
        "std_obstructive_moves": float(obstructive.std(ddof=0)),
        "mean_obstructive_moves_per_delivered_block": _mean_present(
            rows, "obstructive_moves_per_delivered_block"
        ),
        "mean_return": _mean_present(rows, "return"),
        "mean_steps": _mean_present(rows, "steps"),
        "mean_absolute_error": _mean_present(rows, "mean_absolute_error"),
        "mean_signed_delivery_error": _mean_present(
            rows, "mean_signed_delivery_error"
        ),
        "mean_tardiness": _mean_present(rows, "mean_tardiness"),
        "mean_delivery_count": _mean_present(rows, "delivery_count"),
        "mean_invalid_assignment_count": _mean_present(
            rows, "invalid_assignment_count"
        ),
        "mean_fallback_count": _mean_present(rows, "fallback_count"),
        "mean_infeasible_epoch_count": _mean_present(
            rows, "infeasible_epoch_count"
        ),
        "mean_retrieval_live_plan_failure_count": _mean_present(
            rows, "retrieval_live_plan_failure_count"
        ),
        "mean_inbound_approach_defer_count": _mean_present(
            rows, "inbound_approach_defer_count"
        ),
        "mean_exact_recovery_search_count": _mean_present(
            rows, "exact_recovery_search_count"
        ),
        "mean_exact_recovery_fallback_count": _mean_present(
            rows, "exact_recovery_fallback_count"
        ),
        "mean_exact_recovery_failure_count": _mean_present(
            rows, "exact_recovery_failure_count"
        ),
        "mean_exact_recovery_explored_nodes": _mean_present(
            rows, "exact_recovery_explored_nodes"
        ),
        "mean_source_setup_seconds": _mean_present(
            rows, "source_setup_seconds"
        ),
        "mean_assignment_planning_seconds": _mean_present(
            rows, "assignment_planning_seconds"
        ),
        "mean_episode_loop_seconds": _mean_present(
            rows, "episode_loop_seconds"
        ),
        "rows": list(rows),
    }


def evaluation_selection(validation: dict) -> dict:
    map_summary = validation["map"]
    stochastic = validation["stochastic"]
    primary_strict = bool(stochastic["all_strict_method_success"])
    return {
        "eligible": primary_strict,
        "primary_deployment_mode": DEPLOYMENT_STOCHASTIC,
        "primary_strict_method_success_rate": float(
            stochastic["strict_method_success_rate"]
        ),
        "primary_mean_obstructive_moves": float(
            stochastic["mean_obstructive_moves"]
        ),
        "primary_mean_return": float(stochastic["mean_return"]),
        "secondary_map_all_strict_method_success": bool(
            map_summary["all_strict_method_success"]
        ),
        "secondary_map_strict_method_success_rate": float(
            map_summary["strict_method_success_rate"]
        ),
        "secondary_map_mean_obstructive_moves": float(
            map_summary["mean_obstructive_moves"]
        ),
        "contract": MODEL_SELECTION_CONTRACT,
    }


def model_selection_key(validation: dict) -> tuple[float, ...]:
    selection = validation["selection"]
    return (
        float(selection["eligible"]),
        float(selection["primary_strict_method_success_rate"]),
        -float(selection["primary_mean_obstructive_moves"]),
        float(selection["primary_mean_return"]),
    )


def evaluate(
    training_source: Kim2020A3CSpatialSource,
    *,
    args: argparse.Namespace,
    instances: Sequence[tuple[int, EpisodeInstance]],
) -> dict:
    payload = training_source.checkpoint()
    map_rows = []
    stochastic_rows = []
    with preserve_global_rng_state():
        for instance_seed, instance in instances:
            map_seed = _policy_seed(
                args.validation_policy_seed_base,
                rollout_index=args.stochastic_rollouts,
            )
            map_rows.append(
                _run_row(
                    payload,
                    args,
                    instance_seed=instance_seed,
                    instance=instance,
                    deployment_mode=DEPLOYMENT_MAP,
                    policy_seed=map_seed,
                    rollout_index=0,
                )
            )
            for rollout_index in range(args.stochastic_rollouts):
                policy_seed = _policy_seed(
                    args.validation_policy_seed_base,
                    rollout_index=rollout_index,
                )
                stochastic_rows.append(
                    _run_row(
                        payload,
                        args,
                        instance_seed=instance_seed,
                        instance=instance,
                        deployment_mode=DEPLOYMENT_STOCHASTIC,
                        policy_seed=policy_seed,
                        rollout_index=rollout_index,
                    )
                )
    validation = {
        "contract": VALIDATION_CONTRACT,
        "validation_instance_seeds": [seed for seed, _ in instances],
        "validation_policy_seed_base": args.validation_policy_seed_base,
        "stochastic_policy_seeds": [
            _policy_seed(
                args.validation_policy_seed_base,
                rollout_index=rollout_index,
            )
            for rollout_index in range(args.stochastic_rollouts)
        ],
        "map_policy_seed": _policy_seed(
            args.validation_policy_seed_base,
            rollout_index=args.stochastic_rollouts,
        ),
        "stochastic_rollouts_per_instance": args.stochastic_rollouts,
        "map": summarize_rows(map_rows),
        "stochastic": summarize_rows(stochastic_rows),
    }
    validation["selection"] = evaluation_selection(validation)
    validation["selection_key"] = list(model_selection_key(validation))
    return validation


def _tail_mean(value, count: int = 100):
    if not isinstance(value, (list, tuple)) or not value:
        return None
    return float(np.mean([float(item) for item in value[-count:]]))


def source_diagnostics(source: Kim2020A3CSpatialSource) -> dict:
    scalar_names = (
        "assignment_count",
        "gradient_steps",
        "completed_outcome_count",
        "censored_outcome_count",
        "training_episode_count",
        "skipped_episode_count",
    )
    diagnostics = {
        name: json_safe(getattr(source, name))
        for name in scalar_names
        if hasattr(source, name)
    }
    for name in (
        "loss_history",
        "actor_loss_history",
        "value_loss_history",
        "entropy_history",
        "return_history",
    ):
        if hasattr(source, name):
            diagnostics[f"{name}_ma100"] = _tail_mean(getattr(source, name))
    audit = getattr(source, "audit", None)
    if callable(audit):
        audit_value = audit()
        if isinstance(audit_value, dict):
            diagnostics["audit"] = json_safe(audit_value)
    return diagnostics


def save_checkpoint(
    source: Kim2020A3CSpatialSource,
    path: Path,
    *,
    checkpoint_kind: str,
    args: argparse.Namespace,
    config: Kim2020Config,
    geometry: dict,
    completed_episodes: int,
    validation: dict,
    python_hash_seed: str,
    training_device: str,
    cublas_workspace_config: str | None,
) -> None:
    payload = source.checkpoint(
        trainer_version=TRAINER_VERSION,
        checkpoint_kind=checkpoint_kind,
        method=TRACK_A_KIM2020_A3C_SPATIAL,
        adaptation_contract=ADAPTATION_CONTRACT,
        exact_paper_reproduction=False,
        source_paper_doi=PAPER_DOI,
        model_selection_contract=MODEL_SELECTION_CONTRACT,
        validation_contract=VALIDATION_CONTRACT,
        exact_recovery_fallback_contract=EXACT_RECOVERY_FALLBACK_CONTRACT,
        exact_recovery_max_nodes=EXACT_RECOVERY_MAX_NODES,
        deployment_modes_evaluated=[DEPLOYMENT_MAP, DEPLOYMENT_STOCHASTIC],
        training_seed=args.seed,
        python_hash_seed=python_hash_seed,
        deterministic_algorithms=args.deterministic_algorithms,
        training_device=training_device,
        cublas_workspace_config=cublas_workspace_config,
        actor_coefficient=1.0,
        training_lambda=args.lam,
        training_mu=args.mu,
        geometry=geometry,
        completed_training_episodes=completed_episodes,
        training_instance_seed_base=args.training_instance_seed_base,
        validation_instance_seeds=list(args.validation_seeds),
        validation_policy_seed_base=args.validation_policy_seed_base,
        stochastic_rollouts=args.stochastic_rollouts,
        max_steps=args.max_steps,
        trainer_config=asdict(config),
        validation_evaluation=validation,
        resumed_from_checkpoint=(
            str(args.resume.resolve()) if args.resume is not None else None
        ),
        resume_start_episode=int(
            getattr(args, "resume_start_episode", 0)
        ),
    )
    _atomic_torch_save(payload, path)


def _check_output_directory(
    output_dir: Path, *, resume: Path | None = None
) -> None:
    protected = (
        "best.pth",
        "latest.pth",
        "training-history.json",
        "training-summary.json",
    )
    collisions = [name for name in protected if (output_dir / name).exists()]
    if resume is None and collisions:
        raise FileExistsError(
            "output directory already contains training artifacts: "
            + ", ".join(collisions)
        )
    if resume is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    if not output_dir.is_dir():
        raise FileNotFoundError(
            "resume requires an existing output directory"
        )
    if not resume.is_file():
        raise FileNotFoundError(resume)
    if resume.resolve().parent != output_dir.resolve():
        raise ValueError("resume checkpoint must be inside output-dir")
    if resume.name != "latest.pth":
        raise ValueError("resume requires the run's latest.pth checkpoint")
    if not (output_dir / "training-history.json").is_file():
        raise FileNotFoundError(
            "resume requires output-dir/training-history.json"
        )


def _load_training_history(path: Path) -> list[dict]:
    history = json.loads(path.read_text())
    if not isinstance(history, list):
        raise ValueError("training history must be a JSON list")
    episodes = [int(record["episode"]) for record in history]
    if episodes != sorted(set(episodes)):
        raise ValueError("training history episodes must be unique and ordered")
    return history


def _validate_resume_payload(
    payload: dict,
    *,
    args: argparse.Namespace,
    config: Kim2020Config,
    python_hash_seed: str,
    training_device: str,
    cublas_workspace_config: str | None,
) -> int:
    expected = {
        "trainer_version": TRAINER_VERSION,
        "checkpoint_kind": "latest",
        "training_seed": args.seed,
        "python_hash_seed": python_hash_seed,
        "training_lambda": args.lam,
        "training_mu": args.mu,
        "training_instance_seed_base": args.training_instance_seed_base,
        "validation_instance_seeds": list(args.validation_seeds),
        "validation_policy_seed_base": args.validation_policy_seed_base,
        "stochastic_rollouts": args.stochastic_rollouts,
        "max_steps": args.max_steps,
        "deterministic_algorithms": args.deterministic_algorithms,
        "training_device": training_device,
        "cublas_workspace_config": cublas_workspace_config,
        "exact_recovery_fallback_contract": (
            EXACT_RECOVERY_FALLBACK_CONTRACT
        ),
        "exact_recovery_max_nodes": EXACT_RECOVERY_MAX_NODES,
        "config": asdict(config),
    }
    mismatches = [
        key for key, value in expected.items() if payload.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "resume checkpoint does not match requested run: "
            + ", ".join(mismatches)
        )
    completed = int(payload.get("completed_training_episodes", 0))
    if completed <= 0:
        raise ValueError("resume checkpoint has no completed training episodes")
    if args.episodes <= completed:
        raise ValueError(
            "--episodes must exceed the resume checkpoint's completed count"
        )
    return completed


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed is None or hash_seed.lower() == "random":
        raise RuntimeError(
            "PYTHONHASHSEED must be fixed before interpreter startup"
        )
    device = resolve_device(args.device)
    cublas_workspace_config = validate_cuda_determinism(
        device=device,
        deterministic_algorithms=args.deterministic_algorithms,
    )
    seed_everything(
        args.seed, deterministic_algorithms=args.deterministic_algorithms
    )
    _check_output_directory(args.output_dir, resume=args.resume)

    config = Kim2020Config(
        hidden_channels=args.hidden_channels,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        reward_scale=args.reward_scale,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        grad_clip=args.grad_clip,
        updates_per_episode=args.updates_per_episode,
    )
    env = make_env(args)
    geometry = geometry_metadata(env, requested_exit_width=args.exit_width)
    resume_start_episode = 0
    if args.resume is None:
        source = Kim2020A3CSpatialSource(
            env,
            config,
            seed=args.seed,
            policy_seed=args.seed,
            learning_enabled=True,
            device=device,
            deployment_mode=DEPLOYMENT_STOCHASTIC,
        )
        history = []
    else:
        resume_payload = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        resume_start_episode = _validate_resume_payload(
            resume_payload,
            args=args,
            config=config,
            python_hash_seed=hash_seed,
            training_device=device,
            cublas_workspace_config=cublas_workspace_config,
        )
        source = Kim2020A3CSpatialSource.from_checkpoint(
            env,
            resume_payload,
            learning_enabled=True,
            device=device,
            seed=args.seed,
            policy_seed=args.seed,
            deployment_mode=DEPLOYMENT_STOCHASTIC,
        )
        history = _load_training_history(
            args.output_dir / "training-history.json"
        )
        if not history or int(history[-1]["episode"]) != resume_start_episode:
            raise ValueError(
                "training history does not end at the resume checkpoint episode"
            )
    args.resume_start_episode = resume_start_episode

    best_key = None
    best_episode = None
    best_validation = None
    for prior_record in history:
        prior_validation = prior_record["validation"]
        if not prior_validation["selection"]["eligible"]:
            continue
        prior_key = model_selection_key(prior_validation)
        if best_key is None or prior_key > best_key:
            best_key = prior_key
            best_episode = int(prior_record["episode"])
            best_validation = prior_validation
    if best_episode is not None and not (args.output_dir / "best.pth").is_file():
        raise FileNotFoundError(
            "training history records an eligible best but best.pth is missing"
        )

    # A rejected resume must be read-only.  Authenticate the checkpoint,
    # source geometry, history, and prior best artifact above before touching
    # the persisted validation manifest or its instance files.
    validation_instances = build_validation_instances(
        env, args.validation_seeds
    )
    validation_manifest = write_validation_instances(
        args.output_dir, validation_instances
    )
    recent_returns: list[float] = []
    recent_successes: list[float] = []
    recent_obstructive_moves: list[float] = []
    print(
        f"Kim2020 A3C spatial adaptation | device={device} | "
        f"seed={args.seed} | episodes={args.episodes} | "
        f"resume_start={resume_start_episode} | "
        f"cublas_workspace={cublas_workspace_config} | "
        f"stochastic_validation_rollouts={args.stochastic_rollouts}",
        flush=True,
    )

    for episode in range(resume_start_episode + 1, args.episodes + 1):
        instance_seed = (
            args.training_instance_seed_base
            + args.seed * 1_000_000
            + episode
        )
        instance = env.sample_episode_instance(instance_seed)
        result = run_track_a_episode(
            env,
            TRACK_A_KIM2020_A3C_SPATIAL,
            max_steps=args.max_steps,
            episode_instance=instance,
            assignment_source=source,
        )
        recent_returns.append(float(result["return"]))
        recent_successes.append(float(result["success"]))
        recent_obstructive_moves.append(float(result["obstructive_moves"]))
        recent_returns = recent_returns[-100:]
        recent_successes = recent_successes[-100:]
        recent_obstructive_moves = recent_obstructive_moves[-100:]

        if episode % args.eval_every != 0 and episode != args.episodes:
            continue
        validation = evaluate(
            source, args=args, instances=validation_instances
        )
        diagnostics = source_diagnostics(source)
        record = {
            "episode": episode,
            "training_instance_seed": instance_seed,
            "training_return_ma100": float(np.mean(recent_returns)),
            "training_success_ma100": float(np.mean(recent_successes)),
            "training_obstructive_moves_ma100": float(
                np.mean(recent_obstructive_moves)
            ),
            "source_diagnostics": diagnostics,
            "validation": validation,
        }
        history.append(record)
        save_checkpoint(
            source,
            args.output_dir / "latest.pth",
            checkpoint_kind="latest",
            args=args,
            config=config,
            geometry=geometry,
            completed_episodes=episode,
            validation=validation,
            python_hash_seed=hash_seed,
            training_device=device,
            cublas_workspace_config=cublas_workspace_config,
        )
        key = model_selection_key(validation)
        if validation["selection"]["eligible"] and (
            best_key is None or key > best_key
        ):
            best_key = key
            best_episode = episode
            best_validation = validation
            save_checkpoint(
                source,
                args.output_dir / "best.pth",
                checkpoint_kind="best",
                args=args,
                config=config,
                geometry=geometry,
                completed_episodes=episode,
                validation=validation,
                python_hash_seed=hash_seed,
                training_device=device,
                cublas_workspace_config=cublas_workspace_config,
            )
        _atomic_json(args.output_dir / "training-history.json", history)
        print(
            f"Ep {episode:5d} | "
            f"Train moves {record['training_obstructive_moves_ma100']:.2f} | "
            f"MAP moves {validation['map']['mean_obstructive_moves']:.2f} | "
            "Stochastic moves "
            f"{validation['stochastic']['mean_obstructive_moves']:.2f} | "
            "Stochastic eligible "
            f"{int(validation['selection']['eligible'])}",
            flush=True,
        )

    summary = {
        "status": (
            "completed"
            if best_episode is not None
            else "completed_no_eligible_checkpoint"
        ),
        "trainer_version": TRAINER_VERSION,
        "method": TRACK_A_KIM2020_A3C_SPATIAL,
        "adaptation_contract": ADAPTATION_CONTRACT,
        "exact_paper_reproduction": False,
        "source_paper_doi": PAPER_DOI,
        "model_selection_contract": MODEL_SELECTION_CONTRACT,
        "validation_contract": VALIDATION_CONTRACT,
        "exact_recovery_fallback_contract": (
            EXACT_RECOVERY_FALLBACK_CONTRACT
        ),
        "exact_recovery_max_nodes": EXACT_RECOVERY_MAX_NODES,
        "deployment_modes_evaluated": [
            DEPLOYMENT_MAP,
            DEPLOYMENT_STOCHASTIC,
        ],
        "source_feature_version": getattr(source, "feature_version", None),
        "source_architecture": getattr(source, "architecture_name", None),
        "config": asdict(config),
        "lambda": args.lam,
        "mu": args.mu,
        "geometry": geometry,
        "training_seed": args.seed,
        "python_hash_seed": hash_seed,
        "deterministic_algorithms": args.deterministic_algorithms,
        "training_device": device,
        "cublas_workspace_config": cublas_workspace_config,
        "actor_coefficient": 1.0,
        "episodes": args.episodes,
        "resumed_from_checkpoint": (
            str(args.resume.resolve()) if args.resume is not None else None
        ),
        "resume_start_episode": resume_start_episode,
        "max_steps": args.max_steps,
        "training_instance_seed_start": (
            args.training_instance_seed_base + args.seed * 1_000_000 + 1
        ),
        "training_instance_seed_end": (
            args.training_instance_seed_base
            + args.seed * 1_000_000
            + args.episodes
        ),
        "validation_instances": validation_manifest,
        "validation_policy_seed_base": args.validation_policy_seed_base,
        "stochastic_rollouts": args.stochastic_rollouts,
        "best_episode": best_episode,
        "best_selection_key": list(best_key) if best_key is not None else None,
        "best_validation": best_validation,
        "final_validation": history[-1]["validation"],
        "source_diagnostics": source_diagnostics(source),
        "best_checkpoint": (
            str((args.output_dir / "best.pth").resolve())
            if best_episode is not None
            else None
        ),
        "latest_checkpoint": str((args.output_dir / "latest.pth").resolve()),
        "history": str(
            (args.output_dir / "training-history.json").resolve()
        ),
    }
    _atomic_json(args.output_dir / "training-summary.json", summary)
    print(json.dumps(json_safe(summary), indent=2), flush=True)
    if best_episode is None:
        raise RuntimeError(
            "training completed without a checkpoint that passed strict "
            "stochastic validation; latest.pth is diagnostic only"
        )


if __name__ == "__main__":
    main()
