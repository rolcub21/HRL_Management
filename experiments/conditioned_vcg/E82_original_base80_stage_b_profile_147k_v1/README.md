# E82: 8×8 operating profile of the original VCG after Stage B

E82 evaluates the **frozen terminal** E78/E70 Stage-A critics and E80 Stage-B
handling heads for all three model seeds. It does no training and does not
choose a checkpoint or a preferred λ. The yard is the same 8×8/36-job
empty-start workload used in training, but the episode instances are fresh.
This is a target-size operating-profile test, separate from E81's ongoing
Stage-A zero-shot 10×10 transfer test. E81 files and results are untouched.

The four prespecified preferences are λ = 0, .05, .10, and .20. At λ = 0,
the conditioned controller delegates exactly to the Stage-A operational
policy. Positive λ uses the frozen E80 future-rehandling predictor in the
original mode hierarchy. All runs share the exact SAFE frontier, D12
relocation-family certification, E14 timing-invariant cache and path cleanup,
the same progress guard, and the same physical executor. Policies do not
learn from evaluation reward; reports compare physical outcomes, not raw
returns. They include unconditional strict completion, arrival-to-storage
wait, release- and storage-referenced timing, steps per delivery, and physical
rehandles per 100 deliveries. Quality metrics are shown only for strictly
complete episodes; incomplete runs retain delivered count, steps, rehandles,
and censored storage-flow counts.

The pilot is one fresh instance × seed 0 × four λ values: **four episodes**.
The confirmation is three different fresh instances × three model seeds ×
four λ values: **36 episodes**. All λ values for a model/instance share the
same exact workload. Completed coordinates are cached and a panel can be
resumed. Both panels are development evidence, not a sealed paper test.

From the repository root, run the zero-episode checks first:

```bash
bash experiments/conditioned_vcg/E82_original_base80_stage_b_profile_147k_v1/run.sh prepare
bash experiments/conditioned_vcg/E82_original_base80_stage_b_profile_147k_v1/run.sh verify-models
```

Then run and inspect the pilot yourself:

```bash
bash experiments/conditioned_vcg/E82_original_base80_stage_b_profile_147k_v1/run.sh run-panel --panel pilot
bash experiments/conditioned_vcg/E82_original_base80_stage_b_profile_147k_v1/run.sh analyze --panel pilot
```

If the pilot executes cleanly, run the whole prespecified panel:

```bash
bash experiments/conditioned_vcg/E82_original_base80_stage_b_profile_147k_v1/run.sh run-panel --panel confirmation
bash experiments/conditioned_vcg/E82_original_base80_stage_b_profile_147k_v1/run.sh analyze --panel confirmation
```

For one coordinate, use `run-one --panel confirmation --model-seed 1
--instance-seed 148000 --lambda 0.10`. Pilot instance seed is `147990`;
confirmation instance seeds are `148000`, `148001`, and `148002`.
Results go to
`results/vcg-conditioned-e82-original-base80-stage-b-profile-147k-v1/`.

No preference should be chosen using the pilot or confirmation results and
then evaluated on the same instances. If one operating point is to be
reported as a selected policy, freeze that choice and open another panel.
