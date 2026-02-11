# Fit QC Report

- Generated: 2026-02-11 10:15:49.977798

## (a) OK / EXCLUDED
- Total wells: 112
- OK: 111
- EXCLUDED: 1
- OK rate: 99.1%

- CSV: fit_qc_summary_overall.csv
- CSV (by plate): fit_qc_summary_by_plate.csv
- CSV (by heat): fit_qc_summary_by_heat.csv

## (b) Selected t_end distribution
- t_end min/max: 66 / 606 s
- q10: 96 s
- q25: 156 s
- q50: 156 s
- q75: 606 s
- q90: 606 s

- t_end ≤ 30 s : 0.0%
- t_end ≤ 60 s : 0.0%
- t_end ≤ 120 s : 21.6%
- t_end ≤ 240 s : 58.6%
- t_end ≤ 600 s : 66.7%

![t_end hist](fit_qc_t_end_hist.png)

## (c) Slope vs t_end
- N (finite): 111
- Pearson r: -0.6777
- Spearman ρ: -0.6734

![slope vs t_end](fit_qc_slope_vs_t_end.png)

## (d) select_method_used breakdown (OK only)
- method column used: select_method_used
- force_whole* fraction (among OK): 0.0%
- force_whole* fraction (among ALL wells): 0.0%

- CSV: fit_qc_select_method_counts.csv
- initial_positive: 38 (34.2%)
- initial_positive_promote_long_ext: 32 (28.8%)
- initial_positive_tangent: 23 (20.7%)
- initial_positive_ext: 10 (9.0%)
- initial_positive_post_broad_overfit_ext: 3 (2.7%)
- initial_positive_early_steep_delayed_steep: 1 (0.9%)
- initial_positive_early_steep_ext: 1 (0.9%)
- initial_positive_tangent_post_broad_overfit_ext: 1 (0.9%)
- initial_positive_promote_long_ext_tangent_post_broad_overfit_ext: 1 (0.9%)
- last_resort: 1 (0.9%)

![select_method_used](fit_qc_select_method_bar.png)

## (e) Distributions (OK only)
### R²
- R² min/max: 0.6 / 1
- R² q10: 0.9795
- R² q25: 0.9966
- R² q50: 0.999
- R² q75: 0.9995
- R² q90: 0.9997

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
- snr min/max: 2.981 / 640.5
- snr q10: 16.26
- snr q25: 38.67
- snr q50: 79.08
- snr q75: 118.5
- snr q90: 158.4

![snr hist](fit_qc_snr_hist_log10.png)

## (f) Exclude reasons (EXCLUDED only)
- CSV: fit_qc_exclude_reason_norm_counts.csv
- R² < r2_min: 1 (100.0%)

![exclude_reason_norm](fit_qc_exclude_reason_norm_bar.png)
