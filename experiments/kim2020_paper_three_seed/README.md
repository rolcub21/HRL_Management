# Frozen three-training-seed Kim adaptation protocol

This is the paper protocol for the Kim et al. (2020)-inspired spatial source.
Training seeds `0`, `1`, and `2` are independent model initializations and use
disjoint training schedule namespaces. They share the frozen architecture,
hyperparameters, model-selection rule, and validation schedules.

Seed 0 is the existing v7 reference. Seeds 1 and 2 each run 1,000 episodes and
are evaluated every 100 episodes on schedules `34000..34004`, using five fixed
stochastic deployment seeds. The best checkpoint is selected by the declared
stochastic strict-success/mean-rearrangement rule. A verifier compares every
fixed checkpoint field with seed 0 and rejects drift.

The earlier `45000..45049` result was inspected before the move to a
three-training-seed design and is therefore a preliminary single-model result.
The three-model paper comparison is preregistered on the previously unused
schedule seeds `97000..97049`. All three models use the same schedules and the
same five policy-seed labels; the checkpoint digest is part of stochastic
action realization.

Train one model at a time from the project root:

```bash
./experiments/kim2020_paper_three_seed/run.sh train1
./experiments/kim2020_paper_three_seed/run.sh train2
```

If a process is interrupted after writing `latest.pth`, continue it with
`resume1` or `resume2`. Resume restores the saved optimizer and RNG state and
is checked against the same frozen configuration.

After both verification reports pass, run the frozen paper evaluation:

```bash
./experiments/kim2020_paper_three_seed/run.sh eval-all
./experiments/kim2020_paper_three_seed/run.sh aggregate
```

Every training directory and evaluation artifact is write-once: the runner
refuses to overwrite an existing target. Do not tune or replace a seed based on
its validation or final result. A failed or weak independent seed remains part
of the paper result.

Aggregation must first average five policy rolls within each
training-seed/schedule pair. Report model-specific results, mean and sample
standard deviation across the three trained models, and paired schedule
contrasts after equally averaging the three models. Baselines need not be
treated as three independent observations merely because their deterministic
runs are repeated in each artifact.
