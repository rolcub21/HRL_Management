# E193 — observable long-stay residual VCG

E193 is the second fresh model for the exact E191 long-stay workload. It keeps
the E191 yard, job generation, objective, certification, progress guard,
training budget, exploration schedule, validation panel, and exact speed-ups.
It changes only the learning interface that E191 exposed poorly:

- the public context uses long-stay time units and observed arrival history;
- Defer advances to an observed release or a known stored-block window event,
  without reading a future release timestamp or imposing a 160-step cutoff;
- the network learns a scalar residual while the exact observable nominal
  five-component macro cost is added to every selection and TD score.

All network tensors, optimizer state, replay, counters, and caches start fresh.
The model still uses the established scalar-Q graph architecture; this is not
a multi-head replacement. The target is still the same undiscounted
`2T + E + 0.5W + 0.1L + 10R` objective.

Run the non-episodic checks first:

```bash
MPLCONFIGDIR=/tmp/e193-mpl \
bash experiments/conditioned_vcg/E193_observable_longstay_residual_252k_v1/run.sh \
  self-check --device cpu
```

Launch the first learning/validation segment through episode 20:

```bash
cd /home/ai_diagnosis/HRL_Management
set -o pipefail

MPLCONFIGDIR=/tmp/e193-mpl \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E193_observable_longstay_residual_252k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 20 --device cuda 2>&1 | \
tee -a results/vcg-conditioned-e193-observable-longstay-residual-252k-v1/seed-0-run.log
```

The command is resumable. To continue the same run after inspecting the
episode-20 development panel, replace `20` by the next desired checkpoint or
by `300`. Validation runs every 20 episodes on the same nine sealed-development
cases as E191. The test panel remains unopened, and the 9,000-step horizon is
an error cap rather than an action rule.

The retained acceleration stack is relocation-family certification,
timing-invariant certificate caching, path cleanup, device graph-input caching,
identity-deduplicated batched scoring, cached batched TD backups, and
cost-aware frozen-target caching. The E193 adapter explicitly makes these
caches store total `Q = known nominal cost + learned residual`.
