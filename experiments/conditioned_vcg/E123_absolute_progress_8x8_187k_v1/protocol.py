"""Three matched fresh-training recipes; the absolute arm needs explicit opt-in."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from experiments.conditioned_vcg.E122_joint_flow_contract_8x8_186k_v1 import protocol as parent
from experiments.conditioned_vcg.E118_short_exploration_earliness_8x8_182k_v1 import protocol as e118

ROOT = parent.ROOT
PROTOCOL = "vcg_conditioned_e123_absolute_progress_8x8_187k_v1"
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e123-absolute-progress-8x8-187k-v1"
ARMS = {"A_outflow_bounded": (False, False), "B_joint_bounded": (True, False), "C_joint_absolute": (True, True)}
MAX_EPISODE = 120
VALIDATION_SEEDS = tuple(range(187800, 187805))
PILOT_SEED = 187990
TRAIN_ORIGIN = 187_000_000
TRAIN_STRIDE = 10_000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=1)
def contract():
    inherited = parent.contract()
    hashes = dict(inherited["source_sha256"])
    for name, expected in hashes.items():
        if parent.parent.attribution.sha256(ROOT / name) != expected:
            raise RuntimeError(f"E123 inherited implementation changed: {name}")
    for path in Path(__file__).resolve().parent.iterdir():
        if path.suffix in (".py", ".sh"):
            hashes[str(path.relative_to(ROOT))] = parent.parent.attribution.sha256(path)
    hashes["tests/test_vcg_e123_absolute_progress.py"] = parent.parent.attribution.sha256(ROOT / "tests/test_vcg_e123_absolute_progress.py")
    value = {
        "protocol": PROTOCOL, "architecture_name": "Progress-Observed Joint-Flow VCG",
        "source_contract_sha256": inherited["contract_sha256"], "source_sha256": hashes,
        "arms": {arm: {"joint_flow": spec[0], "absolute_clock": spec[1]} for arm, spec in ARMS.items()},
        "initialization": "from scratch, identical seed and extended input schema across compared arms; no source checkpoint",
        "model_seeds": [0, 1], "default_training_seeds": [0],
        "train_seed_origin": TRAIN_ORIGIN, "model_seed_stride": TRAIN_STRIDE,
        "validation_seeds": list(VALIDATION_SEEDS), "adversarial_pilot_seed": PILOT_SEED,
        "maximum_episode": MAX_EPISODE, "validation_every": 6, "fixed_late_checkpoints": [96, 102, 108, 114, 120],
        "objective": "2T+E+0.1L+10R, release-fixed due dates, +/-20 tolerance, gamma=1, raw reward scaled by .01",
        "regime": "empty 8x8 n36; release gap 30; Poisson mean 80",
        "exploration": {"start": .9, "floor": .05, "decay_training_decisions": 8000},
        "learning_rate": "5e-5 before decision 8000, then 1e-5",
        "same_successor_for_all_cost_heads": True, "target_period": 200,
        "absolute_bound": {"W": 30, "certified_progress_prefix_budget": 20, "events": 72,
                           "conditional_horizon_bound": 3600,
                           "nonprogress_macros": "must fit remaining W; otherwise excluded",
                           "deadline": "only bounded Accept/Deliver, or commit a bounded prefix through first Deliver",
                           "external_events": "never reset clock; no-work idle uses declared <=30 release-gap assumption"},
        "critic": "old queue/guard inputs plus absolute progress age, remaining W, Defer streak, committed progress budget; replay stores all",
        "absolute_arm_training": "requires passed gate and explicit --authorize-absolute-arm; not included in train-matched default",
        "checkpoint_selection": "fixed final-five window within instance, then aggregate; best checkpoint not selected",
        "validation": "greedy frozen clone; no training-state/RNG advancement",
        "test_panel_opened": False, "paper_evidence": False,
    }
    return {**value, "contract_sha256": digest(value)}


def train_seed(model_seed, episode):
    return TRAIN_ORIGIN + int(model_seed) * TRAIN_STRIDE + int(episode) - 1
