# Constrained VCG-SMDP V2.1 calibration

V2.1 is the isolated two-timescale repair of the constrained V2 development
method. It starts a fresh model and does not load or resume V1, V1.2, or V2
artifacts.

The calibration freezes 100 training episodes in ten blocks:

- blocks 1–2 (episodes 1–20): critic warm-up with `lambda=0`;
- blocks 3–8: primal–dual training with block-constant temperatures annealed
  from `(within=0.10, group=1.0)` to `(within=0.01, group=0.05)`;
- blocks 9–10: low-temperature stabilization at those floors.

The same temperatures induce behavior and Bellman continuation. Lambda is held
fixed for each complete ten-episode block. Each block is evaluated under
deterministic nested MAP before its dual proposal is computed. Proposals after
blocks 3–9 apply only to the next block; the block-10 proposal is recorded but
not applied because no later primal block could authenticate it.

Run from the repository root:

```bash
bash experiments/vcg_constrained_v2_1/run_development_calibration.sh
```

The command matches the V2 calibration workload: a 5x5 environment, eight
blocks per episode, 2,000 maximum steps, training paths beginning at
60,000,000, and MAP validation paths 84,000–84,002. The output directory must
be absent or empty:

```text
results/vcg-constrained-v2-1-development-seed0-100ep
```

Watch the block-end validation line:

- `R`, `MAE`: dense operational return and timing accuracy;
- `Reh/100`, `Budget`: physical rehandles per required deliveries and budget
  compliance;
- `Hold`, `Deliver`: deterministic MAP action shares, useful for detecting the
  earlier Hold-to-Deliver ranking flip;
- `OpP`, `InP`: maximum probability under the regularized operator at the
  outer-group and selected within-group levels. These diagnose concentration;
  validation execution itself remains deterministic MAP;
- `Eligible`: a development candidate only when block 8 or later is strict,
  budget-feasible, and has mean absolute error no greater than the existing
  20-step service window.

Every artifact remains development-only, non-resumable, and ineligible for a
deployment or manuscript performance claim. The prospective 83xxx panel is
not opened.
