# E259 causal action-ranking audit

- States: 9
- Certified branches: 176
- Critic-selected realized-best rate: 0.556
- Mean selected J regret: 0.911
- Within-mode pairwise accuracy: 0.7709876543209877
- Early state identical: True
- Early realized best differs by future: True

| workload | state | candidates | selected | realized best | J regret | dominant regret |
|---|---|---:|---|---|---:|---|
| 20% | accept_1 | 8 | accept:B1:3:3 | accept:B1:1:2 | 2.2 | admission_wait_accrual |
| 20% | accept_4 | 17 | accept:B4:2:3 | accept:B4:1:2 | 0.8 | admission_wait_accrual |
| 20% | first_timing_choice | 19 | defer:10 | defer:10 | 0.0 | none |
| 50% | accept_1 | 8 | accept:B1:3:3 | accept:B1:3:3 | 0.0 | none |
| 50% | accept_4 | 29 | accept:B4:3:1 | accept:B4:3:1 | 0.0 | none |
| 50% | first_timing_choice | 29 | defer:10 | defer:10 | 0.0 | none |
| 80% | accept_1 | 8 | accept:B1:3:3 | accept:B1:2:1 | 4.3 | admission_wait_accrual |
| 80% | accept_4 | 29 | accept:B4:1:1 | accept:B4:2:1 | 0.9 | storage_earliness_at_delivery |
| 80% | first_timing_choice | 29 | defer:10 | defer:10 | 0.0 | none |
