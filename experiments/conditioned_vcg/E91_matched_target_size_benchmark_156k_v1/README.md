# E91: fresh matched target-size benchmark

This is the **8×8 test**, not E81's zero-shot 8×8→10×10 transfer test and not
E82's three-instance development profile. It evaluates the completed original
VCG cohort: three episode-112 base-80 Stage-A models, each with its own E80
round-8 Stage-B head. At λ=0 the head delegates exactly to Stage A. No model
is trained or selected here.

Thirty previously unused `empty_8x8_n36` instances are frozen by an instance
manifest. The yard starts empty; arrivals are spaced 10 primitive steps apart
and durations follow the same Poisson(80) generator used in training. Each
instance is reused for all methods. The pilot uses a disjoint seed and cannot
be promoted into the test panel.

Methods: VCG at λ=0/.05/.10/.20 for each of three model seeds; Dynamic PSLAP
once per instance; capacity-aware GA with four stochastic replicates per
instance, matching E25. The baselines use E28's empty-start common scheduler,
including strict retrieval, legal blocker relocation, standalone
reconfiguration, and bounded wait. No target-size-trained A3C cohort exists,
so A3C is omitted rather than importing a 5×5 policy.

Prepare and verify (no episodes):

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/conditioned_vcg/E91_matched_target_size_benchmark_156k_v1/run.sh prepare
bash experiments/conditioned_vcg/E91_matched_target_size_benchmark_156k_v1/run.sh verify-models
```

Run the six-episode pilot yourself, then inspect its report:

```bash
bash experiments/conditioned_vcg/E91_matched_target_size_benchmark_156k_v1/run.sh run-panel --panel pilot
bash experiments/conditioned_vcg/E91_matched_target_size_benchmark_156k_v1/run.sh analyze --panel pilot
```

The **test** is 510 episodes: 360 VCG, 120 GA, and 30 Dynamic PSLAP. Runs
resume by exact coordinate. To work in small chunks, `--from-instance` is
inclusive and `--through-instance` exclusive; both index the frozen 30-instance
list. For example, run the first five matched instances across every method:

```bash
bash experiments/conditioned_vcg/E91_matched_target_size_benchmark_156k_v1/run.sh run-panel --panel test --from-instance 0 --through-instance 5
```

Continue with `5..10`, `10..15`, and so on. A rerun skips authenticated saved
rows. To restrict a chunk to one method, add `--method vcg`,
`--method dynamic_pslap`, or `--method ga`. Once all 30 are complete:

```bash
bash experiments/conditioned_vcg/E91_matched_target_size_benchmark_156k_v1/run.sh analyze --panel test
```

The report is
`results/vcg-conditioned-e91-matched-target-size-156k-v1/e91-test-report.json`.
Per-run rows include strict completion, release- and storage-referenced MAE,
storage-referenced within ±20, rehandles per 100, steps per delivery, mean
arrival-to-storage wait, elapsed episode time, and amortized wall time per
decision. VCG additionally records selector-only and full-frontier time per
decision. Amortized wall time includes search and execution; it is not pure
neural inference latency.

Analysis first averages stochastic replicates or model seeds **within each
instance** and then gives each of the 30 instances equal weight. It reports
paired 95% t intervals and two-sided p-values for VCG minus each baseline;
Holm adjusts the full family of 48 p-values (four λ values × two baselines ×
six quality metrics). The descriptively stronger baseline is the one with
lower mean release MAE. Because that label is chosen after observing test
data, both baselines remain in the Holm family. Intervals are nominal; Holm
applies to p-values, not interval endpoints. If any arm is incomplete, strict
completion remains unconditional and **primary quality inference is refused**;
partial complete-case quality cannot masquerade as a full-panel result.

10×10 requires a separate matched-size cohort and contract. E79 Stage A is
on another server/unfinished locally, and there is no three-seed 10×10 Stage-B
head here. This E91 runner intentionally cannot substitute E81 zero-shot
models or the uncertified direct-legal V6 controller for that missing cohort.
