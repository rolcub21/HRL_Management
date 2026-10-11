# Preference-conditioned handling VCG

This is the public package for the final method. Import the controller with:

```python
from methods.conditioned_vcg import ConditionedHandlingAgent
```

The method combines three layers:

1. the exact viability frontier and recovery-witness guard;
2. the frozen VCG 1.1 operational critic;
3. a lambda-conditioned nonnegative future-handling estimate with a structural
   immediate Reconfigure cost.

The exact frontier also exposes an optional constructive
`RELOCATION_FAMILY_CERTIFICATION` strategy. It validates one native
current-state completion witness, then certifies a relocation successor only
when a legal undo and rebuilt first macro reach the exact suffix-join state.
Unresolved cases use native exact search, and constructed candidate proofs are
not stored as current-state/liveness witnesses. This changes certification
cost, not the safe-set definition or preference ranking.

D12 accepts this strategy as the constructive amortization component for
subsequent scalability evaluation. On the frozen seed-0 controller at
`lambda=0.10`, the integrated confirmation strictly completed both tested
10x10 medium/high occupancy episodes and supplied 21,051 relocation-family
proofs with zero misses. Native exact-only remains the API default so callers
must enable amortization explicitly; this prevents historical experiments from
changing silently. Current-state, admission, and delivery checks remain exact.

Selection is

```text
M_lambda(s,c) = Qop_V1.1(s,c) - lambda * QN(s,c,lambda)
```

and `lambda=0` delegates directly to the original VCG 1.1 selector.

The installed root implementation, `vcg_v11_conditioned_handling.py`, is an
immutable evidence source for the completed checkpoints and 90k contract.
`controller.py` intentionally re-exports it rather than duplicating or moving
it. This preserves authentication while giving future code one stable package
path.

See [the full method specification](../../docs/PREFERENCE_CONDITIONED_VCG.md)
and [paper experiment E1](../../experiments/conditioned_vcg/E01_benchmark_90k/).
The architecture-selection trail is retained separately as the
[D1--D9 development history](../../experiments/conditioned_vcg/development/).

The controlled representation variants used only for E12 live in
`representation_ablation.py`. They do not modify the authenticated final
controller or its E1--E11 evidence.

New integrations can import the certification strategy from:

```python
from methods.conditioned_vcg import RELOCATION_FAMILY_CERTIFICATION
```
