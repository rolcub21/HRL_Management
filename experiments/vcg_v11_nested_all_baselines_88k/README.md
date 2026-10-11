# Nested VCG with all matched comparators on 88k

This additive, inference-only experiment evaluates the missing comparators on
the exact serialized `88000..88029` EpisodeInstances already used by the
nested-VCG confirmation. It does not regenerate the panel, retrain a policy, or
rerun either nested VCG arm.

The combined figure contains:

- VCG (`lambda = 0`), which is exactly the frozen VCG 1.1 controller;
- VCG with handling augmentation (`lambda = 0.025`);
- historical VCG 2.3;
- Duration-aware Dynamic PSLAP;
- the capacity-aware rolling GA; and
- the Kim2020-inspired spatial policy.

New grid: VCG 2.3 360 rows, PSLAP 30, GA 120, and Kim2020 450, for
960 new resumable rows. This comparison was added after the nested confirmation
outcomes and is therefore a matched secondary analysis; it does not rewrite the
prospective confirmation gate.

Run:

```bash
bash experiments/vcg_v11_nested_all_baselines_88k/run.sh run-all
```

The phases can be resumed independently with `run-v23`, `run-baselines`, and
`run-kim`; then use `plot`. If any required row is unsafe or incomplete, that
method's aggregate endpoint is suppressed rather than computed on complete
cases. The visual may show its contiguous safe prefix ending at the first
failure.
