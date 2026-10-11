# E1 computational-data audit for Block 6

The authenticated E1 result contains outcome ledgers for conditioned VCG,
historical VCG 2.3, Dynamic PSLAP, capacity-aware GA, and Kim2020. None of
those row schemas records episode wall time, decision latency, or an isolated
hardware profile. The aggregate E1 report likewise contains no computational
timing field.

Consequently, E1 supports the common-instance operational comparison but not
a retrospective runtime comparison. Historical shell duration cannot be
substituted: the methods used different row multiplicities, the launches were
not isolated, and elapsed process time was not attached to individual runs.

If a paper-facing established-method runtime comparison is required, it must
be a separate companion evaluation on common frozen instances, sequentially
executed on stated idle hardware. It should report per-episode wall time and,
where the method has an online decision boundary, per-decision latency. The
VCG arm must use the same frozen E14-cleanup+D12 implementation evaluated in
E13. Such a companion does not alter E1's operational conclusions and should
not be merged with the E13 scaling curve.
