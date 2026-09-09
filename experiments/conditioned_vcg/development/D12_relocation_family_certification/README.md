# D12 — Relocation-family certification

This separate development experiment asks whether one validated current-state
completion witness can certify many distinct relocation successors. It does
not import or modify D11, patch the native verifier, change the controller,
train models, or write E14 artifacts. Its output directory is separate.

For an already legal relocation from `s` to `x`, the rule constructs:

1. A legal relocation of the same block back to its original cell.
2. A freshly computed canonical approach and first macro from the anchor's
   completion witness.
3. The anchor's remaining, previously validated witness suffix.

Step 2 must reach exactly the anchor witness's first successor, including the
transporter position, geometry, block attributes apart from erased clocks,
obstacles, and reservations. Undo alone usually does not restore the original
transporter position. Every new macro uses the native deterministic pathfinder
and the native destination legality rules. The anchor is replayed with native
legal-action validation once per family; shared suffixes need no new search.

Only `SAFE` anchors under the native closed-admission contract are accepted.
The new witness is a completion upper bound, never a shortest recovery rank.
Depth and primitive-step limits are checked against the complete constructed
witness. Breadth-first requests decline the shortcut to preserve rank
semantics. A failed construction supplies no negative proof. The standalone
`certify_with_fallback` API runs the unchanged native exact search on any miss.

## Offline screen

Run from the repository root, on an otherwise idle CPU, separately from D11
and other latency experiments:

```bash
.venv/bin/python -m experiments.conditioned_vcg.development.D12_relocation_family_certification.run
```

The default deterministic sample chooses up to six evenly spaced relocation
queries from every completed E14 medium frontier. The loop has a 180-second
budget, checked between queries; one query can exceed the remaining time.
Unprocessed queries are recorded as censoring, not failure. Authentication and
input loading occur before this loop. The runner authenticates E14's contract,
capture, and path-cleanup records and writes the selected query indices and
source hashes before measurement. It refuses to overwrite an existing output
directory. Use `--output` for an additional run, `--per-frontier` for a larger
sample, and `--max-seconds` for its budget.

`queries.jsonl` retains each constructed certificate, miss reason, baseline
cost, connection cost, independent validation cost, and witness-length change.
Every served witness is independently replayed to completion using the native
enumerator. `report.json` separates setup, construction, independent replay,
and estimated avoided search cost. Setup includes once-per-family anchor
validation and relocation enumeration. The baseline's current-state search
is common to both designs and is not counted as an avoided search.

Fallback outcomes and costs in this offline screen come from authenticated
saved E14 path-cleanup results; they are not freshly executed. The standalone
API's real exact fallback is tested separately. Savings are sampled estimates
against historical per-query search measurements, not matched online latency
or a claim about accumulated cache performance. Sampling is balanced across
frontiers rather than representative of the full candidate population.

## Interpretation and next gate

This screen establishes coverage, constructive soundness on the sample, and
whether potential search savings justify a larger experiment. It does not
establish frontier identity, operational quality, or online liveness.
Constructed witnesses may differ from native-search witnesses, and a valid
proof could resolve a baseline `UNKNOWN`. An exact `UNSAFE` contradiction
aborts the screen.

Any later online integration must retain and advance the active liveness
witness instead of continually replacing it with a new undo plan. Compare
frontier coverage, selected actions, fallback activations, completion, and
extra handling before adopting this mechanism in the controller.

Tests:

```bash
.venv/bin/python -m pytest -q tests/test_vcg_d12_relocation_family.py
```

## Initial medium-occupancy screen

The default sample completed all 106 selected queries without censoring.
All 106 received a family proof and passed independent native completion
replay. Historical E14 path-cleanup search time for these queries totaled
26.75 seconds. Family setup took 3.97 seconds, connection construction 0.033
seconds, and independent full replay 19.09 seconds. Estimated net savings
were 22.75 seconds excluding independent audit replay, or 3.66 seconds when
charging that replay too. This demonstrates why reusable suffix validation
must remain distinct from full replay in a future runtime implementation.

Constructed witnesses averaged 1.07 additional macros relative to the saved
baseline witnesses; the maximum increase was 36 primitive steps. These are
proof lengths, not observed handling costs. The result supports a larger
controlled evaluation, not production adoption or a claim of universal
relocation coverage. A corridor regression test explicitly demonstrates a
legal undo whose first-witness-macro connection fails.

Artifacts are in `results/vcg-d12-relocation-family-screen-95k/`. The selected
indices and hashes are in `contract.json`; detailed proofs are in
`queries.jsonl`; aggregate results are in `report.json`.

## Full-workload and online confirmation

The extended screen covers all completed relocation queries in E14's saved
medium trace. Its `prefix` validation independently executes both connection
macros with native legal-action validation and checks equality with the anchor
join state. The suffix has already passed native replay during anchor setup.
This is compositional validation of every proof, not sampling of proof validity.

```bash
.venv/bin/python -m experiments.conditioned_vcg.development.D12_relocation_family_certification.run \
  --all-relocations --validation-mode prefix --max-seconds 900 \
  --output results/vcg-d12-relocation-family-full-95k
```

After the offline screen, run the separate online arm sequentially:

```bash
.venv/bin/python -m experiments.conditioned_vcg.development.D12_relocation_family_certification.online
```

This uses the same frozen model, preference, medium instance, search budgets,
and E14 path cleanup, with a 1,200-second censoring limit. Scoped overrides
exist only inside the D12 process; D11 and production files are not edited.
The native current-state checker runs before building each family. Family
proofs are deliberately not inserted into the native outcome cache: later
current-state checks and the retained liveness witness keep their native
semantics. The existing guard continues to advance its witness cursor.

The arm compares ordered candidate keys and decision epochs, and selected
actions and decision epochs, with the authenticated completed E14 path-cleanup
episode. Certificate digests are expected to change and are not a behavioral
equivalence criterion. It records completion, physical relocations, primitive
steps, forced liveness decisions, macro failures, and illegal drops. On
censoring, completed frontier records survive; the benchmark's completed
decision list is available only if its episode call returns.

Online results go to `results/vcg-d12-relocation-family-online-95k/`, including
the run contract, streaming frontier records, the full returned episode, and
an aggregate report. Latency is compared with a historical reference, not a
fresh paired repeat. Neither identical choices on this instance nor safe
completion establishes general coverage or size-independent certification.
Use the report's `verifier_counts` to distinguish family proofs from native
recovery checks. Legacy `exact_cache_misses` fields in the returned benchmark
episode also count family-served verifier requests; they are not counts of
fresh native searches in this experimental arm.

The full offline run completed all 5,450 saved relocation queries across the
21 completed frontiers. Every connection passed native prefix validation;
none required fallback. Historical query search time totaled 1,719.01 seconds.
Family setup and construction took 5.96 seconds, with 106.76 seconds of
independent native connection validation. Net savings including that validation
were estimated at 1,606.30 seconds. Witnesses averaged 1.001 extra macros;
the maximum increase remained 36 primitive steps. This covers the complete
saved relocation prefix, not the entire medium episode or other occupancies.

## Online medium result

The online arm completed safely, without censoring:

| Measurement | Historical E14 path cleanup | D12 family arm |
|---|---:|---:|
| Episode wall time | 2,640.70 s | 602.97 s |
| Completed decisions | 51 | 51 |
| Delivered jobs | 34 | 34 |
| Primitive steps | 765 | 765 |
| Physical reconfiguration macros | 4 | 4 |
| Median frontier time, all 51 | 12.03 s | 2.25 s |
| Maximum frontier time | 191.78 s | 39.14 s |

All 51 ordered candidate-key lists and all 51 selected actions matched the
reference, including decision epochs. D12 issued 9,451 family proofs, executed
34 liveness-forced decisions, and recorded zero macro failures or illegal
drops. The 4.38× historical wall-time ratio corresponds to a 77.2% reduction.
This is one medium instance compared with a historical run, not a replicated
paired latency estimate or a high-occupancy result.

Anchor validation and connection construction together took 3.65 seconds
online. The benchmark still reports 550.51 seconds of certification-related
time within the 602.97-second episode. The remaining exact current-state,
admission, and delivery checks therefore remain a substantial cost; the
family rule addresses relocation successors specifically. Its 657 recorded
native recovery fallbacks exclude admission checks, which continue through
their separate unchanged helper.

The longer candidate certificates did not increase executed relocation count
or primitive steps in this instance. The native current-state witness and
retained guard are preserved by the adapter, rather than treating arbitrary
replacement of an active fallback plan as safe for progress. No production
integration or D11 changes were made. The next evidence needed is a matched
repeat and a high-occupancy confirmation before generalizing this result.

## Integrated confirmation

The accepted proof rule now lives in
`PSLAP/relocation_family_certification.py` and is an explicit optional strategy
of `enumerate_viability_candidates`. The generator passes the native
current-state certificate, recovery action, and modeled successor directly to
the proof rule. This removes the development adapter's assumption that the
first verifier invocation identifies the current state. The default remains
native exact-only, so historical experiments and unrelated callers do not
silently change behavior.

Integrated family certificates are not stored in the outcome cache. Native
current-state certificates therefore continue to supply liveness witnesses.
Every family miss runs the unchanged native exact checker. Frontier telemetry
separates family attempts, proofs, misses, construction time, and remaining
native recovery searches from ordinary cache hits and misses.

The bounded confirmation uses the frozen seed-0 controller at lambda 0.10,
the same 95,100 medium/high 10x10 instances, timing-invariant cache, and E14
path/enumeration cleanup. It performs no training. Medium has the complete
historical 51-decision E14 reference; high has the completed frontier prefix
from the censored D10 latency diagnostic. Historical inputs are authenticated
by their stored self-hashes and raw instance/checkpoint hashes because the
integrated source files intentionally differ from the earlier contracts.

The v1 medium run completed strictly and matched all 51 reference frontiers and
selected decisions. It is authenticated and imported into the v2 report rather
than rerun. The first high run also completed, reached 64 frontiers, and matched
all 21 saved reference frontiers, but a harness error incorrectly required its
length to equal that deliberately censored reference and fired before writing
the terminal ledger. V2 requires complete equality only for the complete medium
reference; high must cover and match the full 21-frontier prefix. It also saves
the returned episode before applying post-run comparison gates.

Run sequentially on an otherwise idle CPU:

```bash
bash experiments/conditioned_vcg/development/D12_relocation_family_certification/run.sh authenticate-inputs
bash experiments/conditioned_vcg/development/D12_relocation_family_certification/run.sh prepare
bash experiments/conditioned_vcg/development/D12_relocation_family_certification/run.sh run-high
```

Each episode has a predeclared 1,800-second wall limit. A timeout is recorded
as censoring, never infeasibility. Only high needs to be rerun; `run-high`
writes its authenticated v2 ledger and refreshes the report containing both the
imported medium result and current high result. The final analysis is:

```bash
bash experiments/conditioned_vcg/development/D12_relocation_family_certification/run.sh analyze
```

The confirmation requires identical candidate keys and epochs on the common
reference prefix. Medium additionally requires identical selected actions and
equal complete frontier/decision lengths. Certificate digests are not required
to match because the constructive witnesses deliberately differ from native
search witnesses. Latency comparisons remain historical rather than fresh
paired repeats.

### Integrated confirmation result

Both confirmation scenarios completed strictly, with no macro failures or
illegal drops:

| Measurement | 10x10 medium | 10x10 high |
|---|---:|---:|
| Delivered jobs | 34/34 | 46/46 |
| Completed decisions/frontiers | 51 | 64 |
| Primitive steps | 765 | 925 |
| Physical reconfiguration macros | 4 | 6 |
| Relocation-family proofs | 9,451 | 11,600 |
| Relocation-family misses | 0 | 0 |
| Remaining native recovery searches | 657 | 957 |
| Episode wall time | 422.49 s | 754.04 s |
| Frontier time | 395.22 s | 723.00 s |
| Remaining exact-search time | 372.53 s | 691.08 s |

For medium occupancy, all 51 candidate frontiers and all 51 selected decisions
match the completed E14 path-cleanup reference. Relative to that authenticated
historical run, frontier time fell from 2,617.13 to 395.22 seconds (84.9%, or
6.62x), and episode wall time fell from 2,640.70 to 422.49 seconds (84.0%, or
6.25x). This is a historical timing comparison on the same instance and frozen
controller, not a fresh replicated latency estimate.

For high occupancy, all 21 candidate frontiers available from the censored D10
reference match exactly in keys and decision epochs. The integrated run then
continued to 64 decisions and completed all 46 deliveries in 754.04 seconds;
the prior implementation reached the 3,600-second limit after 21 completed
frontiers. Over that common 21-frontier prefix, recorded frontier time fell from
3,521.69 to 348.99 seconds (90.1%, or 10.09x). This high comparison measures the
combined integrated stack, including the E14 cleanup as well as D12, so the
whole reduction cannot be attributed to relocation-family certification alone.
The censored reference also lacks returned decision records, preventing a
selected-action identity claim for high occupancy.

The constructive rule served every attempted relocation successor in both
episodes, and its setup plus connection construction took only 2.84 seconds at
medium occupancy and 3.95 seconds at high occupancy. Nevertheless, frontier
certification remained 93.5% and 95.9% of wall time, respectively. The remaining
cost is therefore in current-state, admission, delivery, and other unchanged
exact checks rather than relocation-family misses. D12 establishes that exact
constructive proof sharing is both behavior-preserving on the complete medium
reference and sufficient for strict completion of this high-occupancy case; it
does not establish universal family coverage or remove the need for further
certifier acceleration.

The authenticated combined report is
`results/vcg-d12-relocation-family-integrated-confirmation-95k-v2/report.json`.
