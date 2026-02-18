# Objective Profile Sensitivity Summary (260216)

- Source table: `data/processed/analysis_260217_tradeoff/objective_profile_sensitivity_summary_260216.csv`
- Sign consistency (U0* above/below 1.0) across runs: 12/13 polymers.

## Run-wise top-1 by profile

- 260216-1: default=PMBTA-4, penalty-only=PMBTA-5, gentle=PMBTA-4, strong=PMBTA-4
- 260216-2: default=PMBTA-6, penalty-only=PMBTA-6, gentle=PMBTA-6, strong=PMBTA-6
- 260216-3: default=PMBTA-6, penalty-only=PMBTA-6, gentle=PMBTA-6, strong=PMBTA-6
- 260216-group_mean: default=PMBTA-4, penalty-only=PMBTA-4, gentle=PMBTA-4, strong=PMBTA-4

## Interpretation

- Top-1 is stable in most settings/runs; rank movement exists mainly in lower ranks.
- PMTAC behavior is profile-sensitive (as expected for low U0* and high FoG* tradeoff), so reporting profile sensitivity is essential.
- Keep default objective fixed for BO, but publish the profile-sensitivity grid/heatmap as robustness evidence.
