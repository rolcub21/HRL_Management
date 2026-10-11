# VCG constrained V2.3 gamma ablation

This runs the frozen 200-episode V2.3 controlled ablation. Relative to V2.2,
only `gamma_operational` changes from 0.99 to 1.0; `gamma_rehandle` remains
1.0. The V2.2 training and 85000–85011 development instances are intentionally
reused, so this is a paired development diagnosis rather than a fresh
prospective experiment.

The contract binds the run to the canonical completed V2.2 contract and
validation-manifest hashes and authenticates the exact 12 parent instance
identities, not merely their seeds.

From the repository root:

```bash
bash experiments/vcg_constrained_v2_3/run_development_gamma1_ablation.sh
```

The output directory must be new. The run writes the authenticated contract,
fixed-instance manifest, complete validation ledgers, a self-hashed manifest
of seven immutable model-only candidate-look diagnostics, audit-only latest
checkpoint, eligible best candidate if one exists, and final summary. It
refuses the unopened 86xxx/622000xxx final panel.

Use `train_vcg_constrained_v2_3.load_candidate_look_diagnostic` to inspect a
saved candidate-look model. It verifies the contract, manifest, file hash,
outer envelope, schedule/lambda binding, and nested V2.3 checkpoint before
loading weights.
