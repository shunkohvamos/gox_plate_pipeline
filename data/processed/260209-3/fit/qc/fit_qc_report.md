# Fit QC Report

- Generated: 2026-02-11 10:15:49.516744

## (a) OK / EXCLUDED
- Total wells: 112
- OK: 109
- EXCLUDED: 3
- OK rate: 97.3%

- CSV: fit_qc_summary_overall.csv
- CSV (by plate): fit_qc_summary_by_plate.csv
- CSV (by heat): fit_qc_summary_by_heat.csv

## (b) Selected t_end distribution
- t_end min/max: 96 / 606 s
- q10: 96 s
- q25: 156 s
- q50: 156 s
- q75: 606 s
- q90: 606 s

- t_end ≤ 30 s : 0.0%
- t_end ≤ 60 s : 0.0%
- t_end ≤ 120 s : 16.5%
- t_end ≤ 240 s : 58.7%
- t_end ≤ 600 s : 70.6%

![t_end hist](fit_qc_t_end_hist.png)

## (c) Slope vs t_end
- N (finite): 109
- Pearson r: -0.6266
- Spearman ρ: -0.6113

![slope vs t_end](fit_qc_slope_vs_t_end.png)

## (d) select_method_used breakdown (OK only)
- method column used: select_method_used
- force_whole* fraction (among OK): 0.0%
- force_whole* fraction (among ALL wells): 0.0%

- CSV: fit_qc_select_method_counts.csv
- initial_positive: 42 (38.5%)
- initial_positive_promote_long_ext: 39 (35.8%)
- initial_positive_tangent: 16 (14.7%)
- initial_positive_ext: 6 (5.5%)
- initial_positive_promote_long_ext_tangent_post_broad_overfit_ext: 1 (0.9%)
- full_range_outlier_skip: 1 (0.9%)
- initial_positive_ext_intskip1: 1 (0.9%)
- initial_positive_post_broad_overfit_ext: 1 (0.9%)
- initial_positive_promote_long_ext_tangent: 1 (0.9%)
- initial_positive_ext_post_broad_overfit_ext: 1 (0.9%)

![select_method_used](fit_qc_select_method_bar.png)

## (e) Distributions (OK only)
### R²
- R² min/max: 0.6545 / 1
- R² q10: 0.9844
- R² q25: 0.9967
- R² q50: 0.9991
- R² q75: 0.9995
- R² q90: 0.9998

![r2 hist](fit_qc_r2_hist.png)

### mono_frac
- mono_frac min/max: 0.8947 / 1
- mono_frac q10: 1
- mono_frac q25: 1
- mono_frac q50: 1
- mono_frac q75: 1
- mono_frac q90: 1

![mono_frac hist](fit_qc_mono_frac_hist.png)

### snr
- snr min/max: 2.373 / 467.6
- snr q10: 16.88
- snr q25: 40.85
- snr q50: 83.48
- snr q75: 141.6
- snr q90: 180.5

![snr hist](fit_qc_snr_hist_log10.png)

## (f) Exclude reasons (EXCLUDED only)
- CSV: fit_qc_exclude_reason_norm_counts.csv
- Monotonicity / steps: 1 (33.3%)
- Slope < slope_min: 1 (33.3%)
- R² < r2_min: 1 (33.3%)

![exclude_reason_norm](fit_qc_exclude_reason_norm_bar.png)
