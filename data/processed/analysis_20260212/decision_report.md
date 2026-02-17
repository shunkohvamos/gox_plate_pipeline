# GOx Trade-off Diagnostic Report

## Scope
- Runs analyzed: 7
- Polymer-run rows: 90
- Screen-candidate rows: 75

## Key Findings
- Heat=0 fit window start median (GOx vs non-GOx): 6.00 s vs 6.00 s.
- Heat=0 start-index median (GOx vs non-GOx): 0.00 vs 0.00.
- Runs with negative Spearman(abs0, FoG): 5/5 (rank inversion pressure present in multiple runs).
- Strong rank-up examples (abs rank -> FoG rank, delta>=4): 19 cases.
- Best PMBTA beats PMPC in 5/5 runs; beats PMTAC in 4/5 runs.

## Policy Simulation Summary
| policy_id | n_runs_selected | top_family_pmbta_count | top_family_pmpc_count | top_family_pmtac_count | top_native_rel_median | top_fog_median | unique_top_polymer_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cons_0.40 | 5 | 4 | 0 | 1 | 0.7163 | 1.481 | 4 |
| cons_0.50 | 5 | 5 | 0 | 0 | 0.8013 | 1.42 | 3 |
| cons_0.60 | 5 | 4 | 0 | 1 | 0.8013 | 1.18 | 4 |
| cons_0.70 | 5 | 5 | 0 | 0 | 0.8395 | 1.18 | 3 |
| cons_0.75 | 5 | 5 | 0 | 0 | 0.845 | 1.072 | 3 |
| cons_0.75_r2_0.85 | 5 | 4 | 1 | 0 | 0.845 | 1.072 | 4 |
| cons_0.75_r2_0.90 | 5 | 4 | 1 | 0 | 0.845 | 1.072 | 4 |
| cons_0.75_r2_0.90_rea20_20 | 5 | 4 | 1 | 0 | 0.8855 | 1.072 | 4 |
| cons_0.80 | 5 | 5 | 0 | 0 | 0.845 | 1.072 | 3 |
| cons_0.90 | 5 | 4 | 1 | 0 | 1.028 | 0.9867 | 5 |
| fog_adj_clip | 5 | 4 | 1 | 0 | 1.01 | 1.072 | 3 |
| fog_raw | 5 | 4 | 0 | 1 | 0.7163 | 1.481 | 4 |
| logw_0.50 | 5 | 5 | 0 | 0 | 0.8395 | 1.42 | 3 |
| logw_1.00 | 5 | 5 | 0 | 0 | 1.028 | 1.072 | 1 |

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