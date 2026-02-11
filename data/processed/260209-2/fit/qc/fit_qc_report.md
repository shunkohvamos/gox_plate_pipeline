# Fit QC Report

- Generated: 2026-02-11 22:52:58.391526

## (a) OK / EXCLUDED
- Total wells: 112
- OK: 107
- EXCLUDED: 5
- OK rate: 95.5%

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
- t_end ≤ 120 s : 21.5%
- t_end ≤ 240 s : 63.6%

![t_end hist](fit_qc_t_end_hist.png)

## (c) Slope vs t_end
- N (finite): 107
- Pearson r: -0.6562
- Spearman ρ: -0.6475

![slope vs t_end](fit_qc_slope_vs_t_end.png)

## (d) select_method_used breakdown (OK only)
- method column used: select_method_used
- force_whole* fraction (among OK): 0.0%
- force_whole* fraction (among ALL wells): 0.0%

- CSV: fit_qc_select_method_counts.csv
- initial_positive: 35 (32.7%)
- initial_positive_promote_long_ext: 32 (29.9%)
- initial_positive_tangent: 22 (20.6%)
- initial_positive_ext: 13 (12.1%)
- full_range_outlier_skip: 2 (1.9%)
- initial_positive_ext_tangent: 1 (0.9%)
- initial_positive_promote_long_ext_tangent: 1 (0.9%)
- initial_positive_promote_long_ext_delayed_steep: 1 (0.9%)

![select_method_used](fit_qc_select_method_bar.png)

## (e) Distributions (OK only)
### R²
- R² min/max: 0.7034 / 0.9999
- R² q10: 0.9888
- R² q25: 0.9969
- R² q50: 0.9993
- R² q75: 0.9996
- R² q90: 0.9998

![r2 hist](fit_qc_r2_hist.png)

### mono_frac
- mono_frac min/max: 1 / 1
- mono_frac q10: 1
- mono_frac q25: 1
- mono_frac q50: 1
- mono_frac q75: 1
- mono_frac q90: 1

![mono_frac hist](fit_qc_mono_frac_hist.png)

### snr
- snr min/max: 2.121 / 348.1
- snr q10: 13.11
- snr q25: 41.05
- snr q50: 99.03
- snr q75: 138.9
- snr q90: 171.6

![snr hist](fit_qc_snr_hist_log10.png)

## (f) Exclude reasons (EXCLUDED only)
- CSV: fit_qc_exclude_reason_norm_counts.csv
- R² < r2_min: 4 (80.0%)
- t_end > max_t_end: 1 (20.0%)

![exclude_reason_norm](fit_qc_exclude_reason_norm_bar.png)
