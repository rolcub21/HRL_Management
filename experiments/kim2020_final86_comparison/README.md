# Kim2020 adaptation on the final 86k panel

This additive comparison evaluates the three already-trained Kim-inspired
spatial A3C checkpoints on the exact serialized EpisodeInstances 86000..86029.
It runs five stochastic policy realizations per model-instance: 3 x 30 x 5 =
450 rows. Existing VCG, Dynamic PSLAP, and GA rows are reused without reruns.

```bash
cd /home/ai_diagnosis/HRL_Management
bash experiments/kim2020_final86_comparison/run.sh run
```

Rows are saved individually, so reissuing `run` resumes by loading completed
rows. Any unsafe or incomplete Kim row suppresses Kim's aggregate; successful
rows are never filtered into a complete-case average.

The combined output is
`results/kim2020-final86-comparison/comparison-report.json`.

