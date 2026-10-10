# E28-v2: empty-start, own-placement scale comparison

E28 asks whether frozen VCG and the three adapted Table 3 baselines remain
operationally competitive as they build their **own** 5×5, 8×8, and 10×10
yards from empty storage. This is a new post-hoc, system-level comparison; it
does not replace E27's shared-certified-frontier ranking test. E27 can continue
without any changes to its code, manifests, or result directory. The v1
four-arm development run was superseded before any confirmation episodes;
its output is not paper evidence.

Each paired EpisodeInstance has zero initially stored blocks. All 8, 36, or
64 jobs arrive online at steps 0, 10, 20, and so on. The first job is
waiting at pickup at reset, while **storage itself is empty**. Durations are seeded
Poisson(80). Each method sees the same arrival times and durations as they
become observable, but each builds a different yard through its own choices.
The usable-storage capacities are 8, 35, and 63 slots respectively. We report
**realized mean and peak occupancy** instead of assigning an initial low,
medium, or high occupancy label to an empty yard.

The four frozen methods are the **full ten-point VCG preference family**
at λ∈{0,.025,.0375,.05,.075,.10,.125,.15,.175,.20} (frozen model seed 0),
Dynamic PSLAP
(one deterministic run), capacity-aware GA (four optimizer seeds), and the
A3C adaptation (three frozen models × five policy streams). There are three
paired instance seeds per size: **9 instances and 270 new episodes**. No
training or historical control-row reuse is involved. The adaptation executes
every method through its existing placement proposal and reserved-cell
admission contract. The non-certified baselines share one declared scheduler:
duration-aware dispatch; strict retrieval, which can legally clear blocking
inventory; a standalone legal reconfiguration when needed; then bounded
defer if no action can admit or retrieve. They are **not** bitwise
implementations of online controllers specified in the cited placement
papers. The adapter's blocked waits, standalone and target-bound rehandles,
invalid proposals, and fallbacks are recorded so an implementation failure
is not mislabeled as a placement-policy failure.

VCG retains its frozen exact-SAFE candidate generator, 20,000-node verifier
budget, E14 cleanup/timing-invariant cache, D12 family proofs, guard, and
macro executor. The baselines use physical action legality but no
recoverability certificates. This compares *integrated deployed systems*;
it does not isolate learned ranking (E27 does that). Only the authenticated
scale seed-0 VCG model is used here, matching E27's controller; the baseline
replications represent their frozen nuisance/model-seed grids, not matched VCG
model seeds. The ten VCG preferences are evaluated without choosing a
per-scenario favorite. Completion, MAE,
within-window rate, rehandles per 100 deliveries, steps per delivery, and
occupancy are reported. Quality is conditional on strict completion, with
all failures and wall-clock censoring kept in the completion denominator.
Each episode is limited to 1,800 wall seconds and max(2000, 100×jobs)
simulated steps. The wall limit is a resource bound, not evidence of
infeasibility. Analysis produces a four-panel PDF/PNG with strict completion,
placement MAE, rehandles per 100, and steps per delivery. The blue band is
the range across VCG preferences, **not** an uncertainty interval; the blue
line marks the highest declared preference, λ=.20, at every size (not a
per-scenario best preference). Missing quality points remain blank when no run completed;
they are not plotted as zero.

This 5×5 coordinate is **not an E1 replication**: E1's 30 saved instances
release all eight jobs by step 0 or 1, whereas E28 releases them at steps
0, 10, …, 70. E1 averages three VCG model seeds over 30 instances; E28 uses
seed 0 over three new instances per size. The 5×5 geometry and seed-0
checkpoint match, but workload timing and model/instance coverage do not.
For E28 at λ=.20, VCG has the lowest observed mean placement MAE and ties
Dynamic PSLAP for the fewest rehandles, but uses more steps per delivery
than every baseline. These are descriptive three-instance results, not a
replication of E1's comparative estimates or proof of the cause of the gap.

Prepare and inspect the four-arm 5×5 development pilot. The pilot is an
integration gate, **not** paper evidence and not a quality-tuning screen:

```bash
bash experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.sh prepare
bash experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.sh run-pilot
bash experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.sh freeze-confirmation
```

Once the gate passes, run one method at a time or all 270 episodes. Coordinates
resume from authenticated ledgers, so an interrupted run does not erase prior
work. Run sequentially on otherwise idle hardware if you intend to compare
latency:

```bash
bash experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.sh run-method --method vcg
bash experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.sh run-method --method duration_aware_dynamic_pslap
bash experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.sh run-method --method duration_aware_pslap_ga_2009_rolling_capacity_aware_partial
bash experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.sh run-method --method kim2020_a3c_spatial_adapted__stochastic
bash experiments/conditioned_vcg/E28_empty_start_scale_baselines_100k/run.sh analyze
```

`run-all` runs those methods sequentially. `run-one --method ... --scenario
... --seed ... --replicate ...` is available for a single coordinate. The
default result directory is
`results/vcg-conditioned-e28-empty-start-scale-baselines-100k-v2/`.

Reporting erratum: the frozen VCG runner records λ and the replicate index in
each ledger filename but omits the redundant `replicate` field inside its 90
VCG row objects. The `run.sh analyze` command applies a checked, in-memory
coordinate reconstruction before invoking the frozen analysis. It validates
each ledger's self-hash and requires filename index `K` to match the saved
λ=`LAMBDAS[K]`. It does not rewrite ledgers or rerun episodes. The resulting
`e28-analysis-erratum.json` records the 90 affected paths and correction
source hash.

The reporting wrapper also uses `render_presentation.py` to highlight λ=.20
without modifying the contract-hashed original renderer. The figure's
`e28-empty-start-scale-comparison-data.json` records the exact plotted values,
report hash, renderer hash, and fixed highlight rule. No rollout is rerun.
