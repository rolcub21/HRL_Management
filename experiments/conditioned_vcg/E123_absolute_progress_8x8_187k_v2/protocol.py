"""Versioned execution correction, not a changed learning experiment."""

from functools import lru_cache
import json
from pathlib import Path

from experiments.conditioned_vcg.E123_absolute_progress_8x8_187k_v1 import protocol as source
from .acceleration import IMPLEMENTATION, INPUT_CAPACITY

ROOT, parent, e118 = source.ROOT, source.parent, source.e118
ARMS, MAX_EPISODE = source.ARMS, source.MAX_EPISODE
VALIDATION_SEEDS, PILOT_SEED = source.VALIDATION_SEEDS, source.PILOT_SEED
train_seed = source.train_seed
PROTOCOL = "vcg_conditioned_e123_absolute_progress_8x8_187k_v2"
SOURCE_OUTPUT = source.DEFAULT_OUTPUT
DEFAULT_OUTPUT = ROOT / "results/vcg-conditioned-e123-absolute-progress-8x8-187k-v2"


@lru_cache(maxsize=1)
def contract():
    original = source.contract()
    actual = json.loads((SOURCE_OUTPUT / "contract.json").read_text())
    if actual != original:
        raise RuntimeError("E123-v2 source contract changed")
    hashes = dict(original["source_sha256"])
    for path in Path(__file__).resolve().parent.iterdir():
        if path.suffix in (".py", ".sh"):
            hashes[str(path.relative_to(ROOT))] = parent.parent.attribution.sha256(path)
    test = ROOT / "tests/test_vcg_e123_v2_cache_resume.py"
    hashes[str(test.relative_to(ROOT))] = parent.parent.attribution.sha256(test)
    value = {**original, "protocol": PROTOCOL, "source_contract_sha256": original["contract_sha256"],
             "source_sha256": hashes, "execution_implementation": IMPLEMENTATION,
             "graph_input_capacity": INPUT_CAPACITY,
             "initialization": "authenticated v1 resume; preserve weights, optimizer, replay values/order, counters and RNG",
             "changes": ["share value-identical complete current observations", "retain bounded input cache across episodes and validation"],
             "unchanged": ["all candidates", "four-head combined-action Double-DQN", "objective", "certification",
                           "guard", "exploration", "training instances", "target refresh", "optimizer", "episode budget"],
             "timing": "v1 history retained but excluded from post-migration v2 runtime averages"}
    value.pop("contract_sha256")
    return {**value, "contract_sha256": source.digest(value)}
