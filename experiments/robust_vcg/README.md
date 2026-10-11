# Robust VCG extension

Robust certification is kept separate from the paper-facing E-series.
The current sequence is:

| ID | Experiment directory | Purpose |
|---|---|---|
| R1 | [`vcg_robust_execution_probe/`](../vcg_robust_execution_probe/) | finite one-macro disturbance mechanism |
| R2 | [`vcg_recursive_robust_kernel_reduced_yard/`](../vcg_recursive_robust_kernel_reduced_yard/) | recursive fixed-point kernel on a finite reduced yard |
| R3 | [`vcg_robust_recovery_snapshot_panel_5x5/`](../vcg_robust_recovery_snapshot_panel_5x5/) | real 5x5 snapshot recovery checks |
| R4 | [`vcg_dynamic_robust_filter_panel_5x5/`](../vcg_dynamic_robust_filter_panel_5x5/) | dynamic 5x5 disturbance filtering |
| R5 | [`vcg_dynamic_budgeted_robust_filter_panel_5x5/`](../vcg_dynamic_budgeted_robust_filter_panel_5x5/) | budgeted dynamic robust filtering |
| R6 | [`vcg_exhaustive_live_recovery_tree_5x5/`](../vcg_exhaustive_live_recovery_tree_5x5/) | exhaustive allowed-branch live recovery tree |

These remain theoretical/mechanism experiments. They do not alter the E1
deterministic final-performance claim.
