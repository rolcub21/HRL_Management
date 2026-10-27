# E187 — wait-corrected E78

E187 keeps E78's 8x8 workload, physical frontier, exact certification, and
manuscript progress guard, but removes the admission-wait loophole.

The learned primitive objective is exactly

```
J = 2*T_storage + E_storage + 0.5*W_release_to_admission + 0.1*L + 10*R
```

`T_storage` and `E_storage` are outside a fixed +/-20 step window around
`stored_time + storage_duration`. `W` has no grace interval and accrues for
every released, unadmitted job on every primitive step. Gamma is 1, so the
finite-episode return is exactly `-J`.

The critic sees only observable current queue state. Future releases and the
durations of unreleased jobs remain hidden. The graph encoder is warm-started
from the finalized E78 checkpoint for the corresponding seed; the action
encoder, queue-context encoder, Q head, optimizer, replay, counters, and guard
memory are freshly initialized.

All exact accelerations remain active: relocation-family certification,
timing-invariant certificate caching and path cleanup, device graph-input
caching, identity-deduplicated frontier scoring, and cached/batched TD backup.

The 3,600-step horizon is an error detector, not an action rule. Any horizon
hit is reported as a failed episode.

## Commands

```bash
bash experiments/conditioned_vcg/E187_wait_corrected_e78_246k_v1/run.sh prepare
bash experiments/conditioned_vcg/E187_wait_corrected_e78_246k_v1/run.sh self-check

MPLCONFIGDIR=/tmp/e187-mpl CUBLAS_WORKSPACE_CONFIG=:4096:8 \
bash experiments/conditioned_vcg/E187_wait_corrected_e78_246k_v1/run.sh \
  train-seed --model-seed 0 --through-episode 300 --device cuda
```

Training is resumable at every completed episode. Validation uses five fresh
E187-only development instances every 20 episodes. They are disjoint from the
panel that selected the warm-start checkpoints; the sealed test panel remains
closed.

