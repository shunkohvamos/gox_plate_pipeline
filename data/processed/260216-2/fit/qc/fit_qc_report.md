# Fit QC Report

- Generated: 2026-02-16 19:51:49.100134

## (a) OK / EXCLUDED
- Total wells: 112
- OK: 112
- EXCLUDED: 0
- OK rate: 100.0%

- CSV: csv/fit_qc_summary_overall.csv
- CSV (by plate): csv/fit_qc_summary_by_plate.csv
- CSV (by heat): csv/fit_qc_summary_by_heat.csv

## (b) Selected t_end distribution
- t_end min/max: 96 / 606 s
- q10: 96 s
- q25: 156 s
- q50: 216 s
- q75: 606 s
- q90: 606 s

- t_end ≤ 30 s : 0.0%
- t_end ≤ 60 s : 0.0%
- t_end ≤ 120 s : 13.4%
- t_end ≤ 240 s : 55.4%

![t_end hist](fit_qc_t_end_hist.png)

## (c) Slope vs t_end
- N (finite): 112
- Pearson r: -0.6649
- Spearman ρ: -0.7086

![slope vs t_end](fit_qc_slope_vs_t_end.png)

## (d) select_method_used breakdown (OK only)
- method column used: select_method_used
- force_whole* fraction (among OK): 0.0%
- force_whole* fraction (among ALL wells): 0.0%

- CSV: csv/fit_qc_select_method_counts.csv
- initial_positive_promote_long_ext: 43 (38.4%)
- initial_positive: 39 (34.8%)
- initial_positive_tangent: 14 (12.5%)
- initial_positive_ext: 6 (5.4%)
- initial_positive_promote_long_ext_tangent: 3 (2.7%)
- initial_positive_tangent_post_broad_overfit_ext_neighbor_recheck_right: 1 (0.9%)
- initial_positive_ext_skip2: 1 (0.9%)
- initial_positive_promote_long_ext_tangent_col1_consensus_t0: 1 (0.9%)
- initial_positive_post_broad_overfit_ext_neighbor_recheck_right: 1 (0.9%)
- full_range_outlier_skip: 1 (0.9%)
- initial_positive_early_steep_delayed_steep_intskip1: 1 (0.9%)
- initial_positive_intskip1_tangent: 1 (0.9%)

![select_method_used](fit_qc_select_method_bar.png)

## (e) Distributions (OK only)
### R²
- R² min/max: 0.9654 / 0.9999
- R² q10: 0.9872
- R² q25: 0.9975
- R² q50: 0.9991
- R² q75: 0.9996
- R² q90: 0.9998

![r2 hist](fit_qc_r2_hist.png)

### mono_frac
- mono_frac min/max: 0.8889 / 1
- mono_frac q10: 1
- mono_frac q25: 1
- mono_frac q50: 1
- mono_frac q75: 1
- mono_frac q90: 1

![mono_frac hist](fit_qc_mono_frac_hist.png)

### snr
- snr min/max: 8.148 / 401
- snr q10: 22.96
- snr q25: 49.41
- snr q50: 91.67
- snr q75: 132.7
- snr q90: 187.9

![snr hist](fit_qc_snr_hist_log10.png)

## (f) Exclude reasons (EXCLUDED only)
- excluded wells: 0
