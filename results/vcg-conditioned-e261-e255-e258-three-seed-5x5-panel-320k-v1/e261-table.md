# E261 — Three corrected-temperature seeds on the E201 5×5 panel

Completion precedes cost. Corrected VCG costs are averaged across model seeds within each instance.

| Method | Complete | J | Storage MAE | Wait | Rehandles/100 | Peak |
|---|---:|---:|---:|---:|---:|---:|
| Corrected-temperature VCG, three-seed mean | 90/90 | 63.927 | 9.606 | 9.004 | 7.778 | 4.200 |
| Original E195, three-seed mean | 90/90 | 79.631 | 11.622 | 9.344 | 11.528 | 4.322 |
| Kim-A3C, three seeds × five trials | 449/450 | 64.622 | 3.210 | 9.301 | 6.682 | 4.269 |
| Dynamic PSLAP | 30/30 | 113.463 | 11.850 | 10.025 | 2.500 | 3.700 |
| GA | 30/30 | 185.195 | 24.360 | 8.930 | 0.312 | 3.275 |

Kim-A3C costs use completed stochastic trials; its completion denominator retains all 450 runs.
All corrected-temperature checkpoints were selected on development validation before panel evaluation.
