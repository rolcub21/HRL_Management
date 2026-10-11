# Matched Kim2020 / V2.3 supplement (85k development panel)

This isolated protocol adds the missing geometry-matched
`kim2020_a3c_spatial_adapted` comparator without changing the active V2.3
seed-stability experiment. It also exposes the already evaluated
`duration_aware_dynamic_pslap` row under the presentation label
**Duration-aware dynamic PSLAP**.

The Kim implementation is a repository adaptation inspired by Kim et al.
(2020), not an exact reproduction of their unavailable implementation. Its v7
obstruction-reduction objective and model-selection rule remain unchanged.

## Frozen training design

- three fresh models, seeds 0, 1, and 2;
- 5x5 yard, default three-cell exit, 8 blocks, arrival rate 10, processing-time
  mean 80;
- 1,000 episodes/model and the established Kim-v7 hyperparameters;
- training schedules 70,000,001..70,001,000 for seed 0,
  71,000,001..71,001,000 for seed 1, and
  72,000,001..72,001,000 for seed 2;
- model selection on schedules 73,000,000..73,000,004 with five stochastic
  policy seeds beginning at 74,000,000;
- fresh, write-once, non-resumable jobs in protocol v1.

The later primary comparison is exactly 3 models x 12 authenticated 85k
instances x 5 stochastic policy rolls = 180 rows. Policy RNG seeds use
`632000000 + 1000*model_seed + 10*panel_index + rollout`. Rollouts are averaged
within instance, instances within model, and models with equal weight. The
statistical unit remains the 12 EpisodeInstances. MAP, if added, is diagnostic
only.

The existing Dynamic PSLAP trajectory rows are reused from the authenticated
repair-V2 report. They are never rerun or duplicated. `legacy_adapted_pslap`
remains an unreliable internal ablation, and the offline full-schedule GA is
not part of the online dominance screen.

The supplemental comparative table keeps the new and relabeled methods beside
the complete safety-eligible online comparison set:

| Method | Dense return | MAE | Steps | Physical rehandles / 100 | Status |
|---|---:|---:|---:|---:|---|
| VCG constrained V2.3, gamma=1 (episode 160) | 130.2902 | 16.1406 | 237.9792 | 5.7292 | authenticated |
| VCG dense V1.1 (three selected training seeds) | 182.1758 | 12.5868 | 193.4167 | 28.1250 | authenticated |
| Duration-aware nearest-free | -349.1925 | 47.1563 | 167.4167 | 0.0000 | authenticated |
| **Duration-aware dynamic PSLAP** | **24.2375** | **22.6667** | **169.5833** | **9.3750** | **authenticated** |
| Park-Seo (2009)-inspired rolling GA, capacity-aware | 148.1117 | 14.4271 | 181.1667 | 19.7917 | authenticated |
| Duration-aware rolling GA, capacity-aware | 119.9417 | 16.5104 | 173.3333 | 17.7083 | authenticated |
| Operational rolling GA, capacity-aware | 132.4300 | 15.5938 | 173.0000 | 16.6667 | authenticated |
| Enhanced-complete rolling GA, capacity-aware | 119.9417 | 16.5104 | 173.3333 | 17.7083 | authenticated |
| **Kim et al. (2020)-inspired spatial A3C (stochastic)** | **pending matched training** | **pending** | **pending** | **pending** | **prepared** |

The final table replaces only the Kim placeholders from authenticated 180-row
evaluation output. Every existing value is reloaded from the pinned repair-V2
report rather than copied from this documentation. The four superseded GA
adapters that failed strict completion remain recorded in the source report but
are not mixed into this numeric table.

## Preparation and training

After these files are installed in the project root:

```bash
bash experiments/vcg_v2_3_kim2020_supplement_85k/run.sh prepare
bash experiments/vcg_v2_3_kim2020_supplement_85k/run.sh train_seed 0
bash experiments/vcg_v2_3_kim2020_supplement_85k/run.sh train_seed 1
bash experiments/vcg_v2_3_kim2020_supplement_85k/run.sh train_seed 2
```

Do not run Kim training concurrently on the same GPU as the active V2.3 jobs.
If a Kim job is interrupted, archive the partial directory and diagnose it;
this frozen protocol deliberately does not accept resume.

No 86xxx EpisodeInstance or 622m policy seed may be opened. This is a reused
development-panel sensitivity analysis, not confirmation. The original V2.3
stability verdict is immutable; the future Kim table creates a separate,
stricter extended dominance screen.
