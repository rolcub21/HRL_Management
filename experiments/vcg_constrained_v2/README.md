# Constrained VCG-SMDP V2

This is an isolated, development-only implementation of the constrained V2
method. It does not load, modify, or resume VCG V1/V1.2 artifacts.

The controller uses:

- exact-SAFE candidate certification and the retained recovery-witness guard;
- four cardinality-normalized groups: Accept, Deliver, Reconfigure, Hold;
- one operational-return critic and one nonnegative physical-rehandle critic;
- one shared elapsed-weighted Lagrangian policy for both Bellman targets;
- an episode-level projected dual for a predeclared rehandle allowance; and
- a certified, event-interruptible Hold with a primitive idle-step budget.

The frozen mathematical contract is in
[`docs/VCG_CONSTRAINED_V2.md`](../../docs/VCG_CONSTRAINED_V2.md).

## Verified CPU smoke

The implementation has already completed a real three-training-episode CPU
smoke plus frozen validation. It exercised all four controls, raw vector replay,
eight gradient steps, exact certification, physical event accounting, and the
dual update. The smoke is an integration result, not a performance result.

## GPU development calibration

Run from the repository root:

```bash
bash experiments/vcg_constrained_v2/run_development_calibration.sh
```

The script uses training seeds beginning at 60,000,000 and development
validation seeds 84,000--84,002. It never opens the prospective 83xxx panel.
The output directory must be new because the calibration trainer is
intentionally fresh-run-only and labels its latest artifact non-resumable.

Watch these fields:

- `Strict`: exact-safe, complete execution integrity;
- `Reh`: observed physical storage-to-storage moves / required workload;
- `Lambda`: the projected constraint multiplier;
- `Reh/100` and `Budget`: frozen MAP validation compliance; and
- `Eligible`: development-candidate eligibility only, never deployment status.

Twenty-five episodes are a mechanism calibration. They are not enough for a
capacity claim. Proper multi-seed training and an untouched evaluation panel
should only be frozen after this run demonstrates stable learning and a
non-saturated dual.
