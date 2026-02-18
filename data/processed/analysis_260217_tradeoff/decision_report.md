# GOx Trade-off Diagnostic Report

## Scope
- Runs analyzed: 11
- Polymer-run rows: 141
- Screen-candidate rows: 114

## Key Findings
- Heat=0 fit window start median (GOx vs non-GOx): 6.00 s vs 6.00 s.
- Heat=0 start-index median (GOx vs non-GOx): 0.00 vs 0.00.
- Runs with negative Spearman(abs0, FoG): 8/8 (rank inversion pressure present in multiple runs).
- Strong rank-up examples (abs rank -> FoG rank, delta>=4): 32 cases.
- Best PMBTA beats PMPC in 8/8 runs; beats PMTAC in 5/8 runs.

## Policy Simulation Summary
| policy_id | n_runs_selected | top_family_pmbta_count | top_family_pmpc_count | top_family_pmtac_count | top_native_rel_median | top_fog_median | unique_top_polymer_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cons_0.40 | 8 | 5 | 0 | 3 | 0.7209 | 1.61 | 4 |
| cons_0.50 | 8 | 6 | 0 | 2 | 0.7418 | 1.538 | 4 |
| cons_0.60 | 8 | 6 | 0 | 2 | 0.7797 | 1.45 | 4 |
| cons_0.70 | 8 | 6 | 0 | 2 | 0.7797 | 1.45 | 4 |
| cons_0.75 | 8 | 6 | 0 | 2 | 0.8307 | 1.45 | 4 |
| cons_0.75_r2_0.85 | 8 | 5 | 1 | 2 | 0.8307 | 1.347 | 5 |
| cons_0.75_r2_0.90 | 8 | 5 | 1 | 2 | 0.8307 | 1.347 | 5 |
| cons_0.75_r2_0.90_rea20_20 | 8 | 5 | 1 | 2 | 0.8307 | 1.347 | 5 |
| cons_0.80 | 8 | 7 | 0 | 1 | 0.8506 | 1.432 | 5 |
| cons_0.90 | 8 | 7 | 1 | 0 | 1.032 | 1.091 | 7 |
| fog_adj_clip | 8 | 6 | 1 | 1 | 0.8718 | 1.45 | 5 |
| fog_raw | 8 | 5 | 0 | 3 | 0.7209 | 1.61 | 4 |
| logw_0.50 | 8 | 7 | 0 | 1 | 0.7825 | 1.45 | 5 |
| logw_1.00 | 8 | 8 | 0 | 0 | 1.195 | 1.291 | 3 |

## Recommended Next Implementation Steps
1. Switch default selection objective to constrained FoG with theta=0.70 (main) and keep raw FoG as diagnostic only.
2. Make main ranking figure feasible-only bar plot; move trade-off scatter to supplementary.
3. Add run-wise control panel (PMPC, PMTAC, best PMBTA) to communicate reproducibility and avoid overclaim.
4. Track PMTAC as a flagged trace type: high FoG + low native_rel should be auto-annotated in ranking outputs.

## Immediate Decision Support
| run_id | top_raw | top_cons_070 |
| --- | --- | --- |
| 260209-1 | PMBTA-3 | PMBTA-3 |
| 260209-2 | PMBTA-10 | PMBTA-10 |
| 260209-3 | PMBTA-11 | PMBTA-11 |
| 260211-1 | PMTAC | PMBTA-11 |
| 260211-2 | PMBTA-10 | PMBTA-11 |
| 260216-1 | PMTAC | PMTAC |
| 260216-2 | PMTAC | PMTAC |
| 260216-3 | PMBTA-10 | PMBTA-10 |