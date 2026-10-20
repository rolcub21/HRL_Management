"""Frozen two-arm, from-scratch E118 training recipe."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from experiments.conditioned_vcg.E115_release_et_cost_vcg_8x8_179k_v1 import (
    protocol as base,
)


ROOT = base.ROOT
PROTOCOL = "vcg_conditioned_e118_short_exploration_earliness_8x8_182k_v1"
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e118-short-exploration-earliness-8x8-182k-v1"
ARMS = {"schedule_only": 1.0, "early125": 1.25}
MODEL_SEED = 0
MAX_EPISODE = 120
EPSILON_START = 0.90
EPSILON_END = 0.05
EPSILON_DECAY_DECISIONS = 8_000
LR_PHASES = ((8_000, 5.0e-5), (float("inf"), 1.0e-5))
LATE_CHECKPOINTS = (96, 102, 108, 114, 120)
HORIZON_RULE = (
    "fresh seed-0 training through 120 episodes; compare the fixed final five "
    "greedy validation checkpoints (96,102,108,114,120), with no best-checkpoint "
    "selection; epsilon reaches 0.05 at decision 8000"
)
REFERENCE_WEIGHTS = (2.0, 1.0, 0.1, 10.0)
ALTERNATIVE_WEIGHTS = (2.0, 1.25, 0.1, 10.0)
SCENARIO = base.SCENARIO


def cost_weights(arm: str) -> tuple[float, ...]:
    if arm not in ARMS:
        raise ValueError(f"unknown E118 arm: {arm}")
    return (2.0, ARMS[arm], 0.1, 10.0)


def learning_rate_at(decisions: int) -> float:
    if isinstance(decisions, bool) or int(decisions) != decisions or decisions < 0:
        raise ValueError("decision count must be a nonnegative integer")
    return 5.0e-5 if decisions < EPSILON_DECAY_DECISIONS else 1.0e-5


@lru_cache(maxsize=1)
def contract() -> dict:
    inherited = base.contract()
    directory = Path(__file__).resolve().parent
    sources = dict(inherited["source_sha256"])
    for name in ("protocol.py", "run.py", "run.sh"):
        path = directory / name
        sources[str(path.relative_to(ROOT))] = base.sha256(path)
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scope": "two matched one-seed development training pilots",
        "architecture_name": base.ARCHITECTURE_NAME,
        "base_recipe_contract_sha256": inherited["contract_sha256"],
        "scenario": SCENARIO,
        "scenario_public": inherited["scenario_public"],
        "workload": inherited["workload"],
        "fresh_initialization": True,
        "source_checkpoint": None,
        "model_seeds": [MODEL_SEED],
        "maximum_episode": MAX_EPISODE,
        "train_instance_seed_base": base.TRAIN_BASE[SCENARIO],
        "instance_rule": "identical ordered seeds and arrival/duration generator in both arms",
        "validation_every": inherited["validation_every"],
        "validation_seeds": inherited["validation_seeds"],
        "validation_role": "reused E115/E116 development panel; no independent test",
        "arms": {arm: {"earliness_weight": value, "cost_weights": list(cost_weights(arm))}
                 for arm, value in ARMS.items()},
        "reference_weights": list(REFERENCE_WEIGHTS),
        "alternative_rescore_weights": list(ALTERNATIVE_WEIGHTS),
        "objective": {
            **inherited["objective"],
            "formula": "2*T + w_E*E + 0.1*L + 10*R",
            "earliness_weight": "arm-specific: 1.0 or 1.25",
            "scalar_weights_in_q_units": "-0.01 times each arm's cost weights",
        },
        "epsilon": {"start": EPSILON_START, "end": EPSILON_END,
                    "warmup_decisions": 0, "decay_decisions": EPSILON_DECAY_DECISIONS},
        "learning_rate_schedule": [
            {"before_decision": 8_000, "learning_rate": 5.0e-5},
            {"from_decision": 8_000, "learning_rate": 1.0e-5},
        ],
        "gamma": 1.0,
        "reward_scale": 0.01,
        "critic": {**inherited["critic"],
                   "component_loss_weights": "same objective-derived rule in each arm"},
        "conservative_update": inherited["conservative_update"],
        "td_backup": inherited["td_backup"],
        "td_backup_implementation": base.TD_BACKUP_IMPLEMENTATION,
        "certification_and_guard": inherited["certification_and_guard"],
        "horizon_rule": HORIZON_RULE,
        "late_checkpoints": list(LATE_CHECKPOINTS),
        "analysis": (
            "rescore every trajectory under both fixed objectives; average the final "
            "five checkpoints within instance before comparing arms across instances; "
            "report checkpoint variability and actual low-epsilon exposure"
        ),
        "checkpoint_selection": "final-five window; best-candidate file is diagnostic only",
        "stage_b_authorized": False,
        "independent_test_opened": False,
        "paper_evidence": False,
        "source_sha256": sources,
    }
    return {**payload, "contract_sha256": base.digest(payload)}


def arm_contract(arm: str) -> dict:
    weights = cost_weights(arm)
    common = contract()
    payload = {
        **{key: value for key, value in common.items() if key != "contract_sha256"},
        "protocol": f"{PROTOCOL}_{arm}",
        "method_version": f"release_et_cost_vcg_e118_{arm}_v1",
        "parent_contract_sha256": common["contract_sha256"],
        "arm": arm,
        "objective": {**common["objective"], "earliness_weight": ARMS[arm],
                      "scalar_weights_in_q_units": [-0.01 * value for value in weights]},
        "component_loss_weights": list(weights),
    }
    return {**payload, "contract_sha256": base.digest(payload)}


def training_view(output: Path, arm: str):
    """Build the E115 wrapper's configuration without changing its source."""
    frozen = arm_contract(arm)
    trainer = SimpleNamespace(**vars(base.trainer_protocol()))
    trainer.PROTOCOL = frozen["protocol"]
    trainer.METHOD_VERSION = frozen["method_version"]
    trainer.MODEL_SEEDS = (MODEL_SEED,)
    trainer.EPISODES_BY_SCENARIO = {SCENARIO: MAX_EPISODE}
    trainer.TOTAL_EPISODES = MAX_EPISODE
    trainer.TRAINING_JOB_BUDGET = MAX_EPISODE * base.e107.e105.e78.SCENARIOS[SCENARIO].total_jobs
    trainer.LATE_CHECKPOINTS = LATE_CHECKPOINTS
    trainer.DEFAULT_OUTPUT = output
    trainer.contract = lambda: frozen
    return SimpleNamespace(
        PROTOCOL=frozen["protocol"], ARCHITECTURE_NAME=base.ARCHITECTURE_NAME,
        SCENARIO=SCENARIO, MAX_EPISODE=MAX_EPISODE,
        EPSILON_START=EPSILON_START, EPSILON_END=EPSILON_END,
        EPSILON_WARMUP_DECISIONS=0, EPSILON_DECAY_DECISIONS=EPSILON_DECAY_DECISIONS,
        DECISION_CHECKPOINTS=LATE_CHECKPOINTS, HORIZON_RULE=HORIZON_RULE,
        TD_BACKUP_IMPLEMENTATION=base.TD_BACKUP_IMPLEMENTATION,
        e107=base.e107, contract=lambda: frozen, trainer_protocol=lambda: trainer,
    )
