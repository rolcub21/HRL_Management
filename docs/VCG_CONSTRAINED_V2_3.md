# Constrained VCG V2.3: operational-discount ablation

V2.3 is a controlled development ablation of V2.2. Its only algorithmic
change is:

`gamma_operational: 0.99 -> 1.0`.

`gamma_rehandle` remains 1.0. The raw dense reward, physical-rehandle cost and
budget, exact-safe frontier, certified Hold, stochastic policy, vector critic,
network, optimizer, replay settings, dual, temperature schedule, candidate
gates, model seed, training instances, and action/replay RNG streams remain
frozen to V2.2.

The protocol is bound to the completed parent artifacts, not only their seed
labels:

- V2.2 training-contract canonical SHA-256:
  `d764582412a73d6bc5e4f8688c28a354e87bc24e7f162c974857f1149738e0c1`;
- V2.2 validation-manifest canonical SHA-256:
  `cd63f0672e6557c0fa53c598e67c12609c8d6393fb8a45f8f46e2d116b108833`.

At the first validation look, V2.3 authenticates every instance seed, instance
ID, schedule ID, and canonical EpisodeInstance SHA against that parent
manifest, then rejects drift at later looks.

This tests whether discounting the operational continuation—but not the
reported finite-horizon objective or the rehandle continuation—caused the
remaining late-delivery timing collapse. Because it deliberately reuses the
already-open V2.2 training instances and 85000–85011 development panel, V2.3
is a paired development diagnosis, not a fresh prospective result.

## Frozen comparison

- Model seed: 10.
- Training EpisodeInstances: 61000000–61000199.
- Training action RNG: `610000000 + episode_number - 1`.
- Replay RNG: 610100010.
- Validation: the same 85000–85011 instances, each crossed with four
  action-only RNG streams 620000000–620000047.
- Training length: 200 episodes; validation every 20 episodes.
- Candidate looks: 80, 100, 120, 140, 160, 180, and 200.
- Candidate gates: unchanged V2.2 strict/exact/full-completion integrity,
  rehandle point estimate and Bonferroni cluster-UCB at most 20, MAE at most
  20, floor temperature, and no current positive-residual lambda saturation.

The V2.2 output must not be used to initialize V2.3. Both runs start from the
same declared model seed and consume the same declared training and policy RNG
streams under their own incompatible checkpoint families.

## Checkpoints and protected panel

Every candidate look writes a unique model-only diagnostic checkpoint under
`candidate-look-checkpoints/`, even when the look fails eligibility. These
files exclude replay, optimizer, and training RNG state and are recorded in a
self-hashed manifest. Their top-level candidate and deployment eligibility
flags are always false. Only `best-development-candidate.pth`, created from an
actually eligible validation and selected by the frozen ranking rule, may be a
development candidate. `latest.pth` remains audit-only and non-resumable.

V2.3 has new method, protocol, and checkpoint-family identities. V2, V2.1,
V2.2, and V2.3 loaders reject cross-family checkpoints.

The unopened final panel remains 86000–86029 crossed with action RNGs
622000000–622000119. Both namespaces are refused by V2.3. No V2.3 development
artifact authorizes a performance claim or deployment.

Diagnostic artifacts must be opened through
`load_candidate_look_diagnostic(...)`. The loader authenticates the contract
and manifest self-hashes, canonical path containment, file SHA-256, diagnostic
role and ineligibility flags, episode/schedule/lambda binding, model-only
contents, and V2.3 nested checkpoint family before returning a frozen agent.
Directly unwrapping `agent_state` is not an authenticated diagnostic load.

The selected `best-development-candidate.pth` must be opened through
`load_best_development_candidate(...)`, never by directly unwrapping its nested
`agent_state`. The caller supplies independently recorded SHA-256 pins for the
raw best file and the canonical complete candidate-look manifest. The loader
then reconstructs the complete frozen V2.3 contract, authenticates the exact
V2.2-bound validation-instance manifest, opens all seven manifest-pinned
diagnostics and their self-hashed validation ledgers, recomputes every
validation summary and the frozen selection rule, and requires the selected
diagnostic and best file to have identical model weights, policy metadata,
schedule, lambda, and dual lifecycle. Unknown outer, nested, or agent-state
fields fail closed. The best artifact remains development-only and does not
become deployment-eligible through loading.

For example, after an independent audit records the two digests:

```python
from pathlib import Path
from train_vcg_constrained_v2_3 import load_best_development_candidate

root = Path("results/vcg-constrained-v2-3-gamma1-ablation-seed10-200ep")
authenticated = load_best_development_candidate(
    root / "best-development-candidate.pth",
    expected_best_sha256=BEST_RAW_SHA_FROM_AUDIT,
    manifest_path=root / "candidate-look-checkpoint-manifest.json",
    expected_manifest_sha256=CANDIDATE_MANIFEST_CANONICAL_SHA_FROM_AUDIT,
    validation_instance_manifest_path=root / "validation-instance-manifest.json",
    contract=root / "training-contract.json",
    device="cpu",
)
agent = authenticated["agent"]
```
