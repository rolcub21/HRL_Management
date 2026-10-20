# E120: missing-Accept attribution

Replays the five **nominal-rule** E119 development episodes with the same frozen
E118 schedule-only episode-120 model. No training, controller changes, new test
instances, relaxed certificates, or inbound-wait intervention.

The audit records occupancy and every stage of admission: physical mask,
bound-option executability, post-admission SAFE/UNSAFE/UNKNOWN certification,
and the final admissible ranker frontier. Failed option initiation is decomposed
into binding and routing reasons. Certification results come from the original
calls, not a second or relaxed verifier. Every choice, boundary time, delivery
error, and episode aggregate must match E119 before the results are accepted.

```bash
bash experiments/conditioned_vcg/E120_accept_frontier_attribution_8x8_184k_v1/run.sh run-panel
```

Requires the completed E119 repair1 directory, including its small frozen model
and five nominal rows, plus the same Python environment. CPU, one thread,
300-second bound per episode. Completed coordinates are reused. Results go to
`results/vcg-conditioned-e120-accept-frontier-attribution-8x8-184k-v1`.

The nominal due cutoff is retained. E119's shifted cutoff was isolated and never
changed E118 training. Attribution must precede any separately labelled bounded
inbound-wait controller experiment. A missing physical/executable option is not
evidence of a certification rejection; low occupancy alone is not evidence of
an accessible pickup or storage route.
