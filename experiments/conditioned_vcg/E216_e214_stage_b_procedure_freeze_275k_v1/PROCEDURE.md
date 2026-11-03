# Frozen Stage B: E214 on-policy trace future-rehandle head

E216 freezes E214 as the Stage-B procedure used by subsequent VCG experiments.
E214 itself is not edited: its completed contract, report, training summary and
round-4 endpoint remain content-addressed evidence.

The frozen procedure is:

1. Freeze the selected Stage-A operational critic, whose reference objective
   already prices a physical rehandle at 10.
2. Attach a preference-conditioned head estimating future physical rehandles
   under the controller that uses that head.
3. Collect four on-policy rounds.  In each round, freeze the round-start head
   while collecting nine episodes: three workloads crossed with continuous
   draws from alpha bands 10–20, 20–30 and 30–40.
4. Construct eta-returns with eta 0.99 using the same frozen round-start head
   for bootstrap terms.  Fit for 40 epochs with smooth-L1 loss, batch size 128,
   learning rate 2e-4 and gradient clipping at 5.  Use only the current round's
   data, then discard it.
5. Deploy with

   `Q_stage_a - 0.01*(alpha-10)*(I_reconfigure + N_alpha)`.

The primary calibration measure is held-out Monte-Carlo future-rehandle count
error, always compared with the exact zero-head predictor.  Evaluation must use
paired instances and retain the immediate-only preference controller as the
ablation.

## Claim boundary

Alpha is a preference coordinate, not an exact rehandle price.  E214 supports
the procedure on one 8x8 development seed over alpha 10–40.  It does not show
monotonic rehandles on every instance, multiseed superiority, or valid
extrapolation beyond 40.  In particular, the heavy alpha-20 row was 7.4% worse
than immediate-only and must remain visible.

The strongest current finding is that alpha 40 beat immediate-only on the
light, medium and heavy development days while removing the earlier collapse.
Forced into the old decision-7 placement, the repaired continuation used 29
rehandles rather than 309; avoiding that placement is not itself the success.

Future protocols import `STAGE_B_PROCEDURE` from this package and declare every
changed field.  Testing alpha 80 or 160 requires new on-policy collection and a
separately versioned range extension.
