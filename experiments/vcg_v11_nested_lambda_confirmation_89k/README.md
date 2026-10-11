# Nested VCG lambda-frontier confirmation on unseen 89k instances

This inference-only experiment confirms the complete lambda frontier selected
on the 85k development panel. For each of the three frozen model seeds, the
same VCG 1.1 operational controller and detached handling head are evaluated
at `lambda in {0, .025, .05, .10, .20}`. Lambda changes only the action merit;
there is no training, tuning, checkpoint selection, or lambda selection.

The confirmation creates 30 new `EpisodeInstance`s (`89000..89029`) once,
serializes them, and loads those exact same files for all 15 policy cells. The
complete grid therefore contains 450 deterministic rows.

Run from the repository root:

```bash
bash experiments/vcg_v11_nested_lambda_confirmation_89k/run.sh prepare
bash experiments/vcg_v11_nested_lambda_confirmation_89k/run.sh run
```

`prepare` authenticates and freezes the experiment contract. It does not
create, sample, or inspect any 89k episode. `run` is the explicit panel-opening
step: it writes the activation marker, samples and serializes the 30 instances,
evaluates all 450 rows, and writes `confirmation-report.json`.
It also renders the sampled MAE–rehandles frontier and cumulative-
EpisodeInstance curves for return, MAE, and rehandles. Every cumulative curve
endpoint is checked against the authenticated report before either figure is
accepted.

Authentication also binds the recorded CUDA, PyTorch, and thread runtime.
Run later analysis under the prepared runtime; a runtime mismatch is reported
as contract drift even when the source and result bytes are unchanged.

If execution is interrupted, resume from completed row ledgers with:

```bash
bash experiments/vcg_v11_nested_lambda_confirmation_89k/run.sh resume
```

The report suppresses aggregate metrics if any row is absent or not
strict-safe-complete. Otherwise it reports return, MAE, steps, and physical
rehandles/100 for every lambda, both by model seed and over all 90 rows. The
predeclared descriptive frontier check is the same check that authorized this
confirmation: at least three aggregate nondominated MAE–rehandling points,
support in at least two of three model seeds, aggregate nonincreasing handling,
and at most one model seed with a material handling reversal. It does not make
a baseline-comparison claim; baselines would need evaluation on these exact
serialized 89k instances.
