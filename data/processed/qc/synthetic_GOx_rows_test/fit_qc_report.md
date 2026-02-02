# Fit QC Report

- Generated: 2026-02-02 15:13:27.707174

## (a) OK / EXCLUDED
- Total wells: 56
- OK: 52
- EXCLUDED: 4
- OK rate: 92.9%

- CSV: fit_qc_summary_overall.csv
- CSV (by plate): fit_qc_summary_by_plate.csv
- CSV (by heat): fit_qc_summary_by_heat.csv

## (b) Selected t_end distribution
- t_end min/max: 126 / 606 s
- q10: 186 s
- q25: 238 s
- q50: 351 s
- q75: 606 s
- q90: 606 s

- t_end ≤ 30 s : 0.0%
- t_end ≤ 60 s : 0.0%
- t_end ≤ 120 s : 0.0%
- t_end ≤ 240 s : 25.0%
- t_end ≤ 600 s : 57.7%

![t_end hist](fit_qc_t_end_hist.png)

## (c) Slope vs t_end
- N (finite): 52
- Pearson r: -0.09736
- Spearman ρ: 0.1051

![slope vs t_end](fit_qc_slope_vs_t_end.png)

## (d) select_method_used breakdown (OK only)
- method column used: select_method_used
- force_whole* fraction (among OK): 0.0%
- force_whole* fraction (among ALL wells): 0.0%

- CSV: fit_qc_select_method_counts.csv
- initial_positive_ext: 18 (34.6%)
- initial_positive_ext_tangent: 11 (21.2%)
- initial_positive: 7 (13.5%)
- last_resort: 6 (11.5%)
- full_range_outlier_skip_intskip1: 5 (9.6%)
- initial_positive_ext_intskip1: 2 (3.8%)
- full_range_outlier_skip_intskip1_tangent: 2 (3.8%)
- outlier_removed: 1 (1.9%)

![select_method_used](fit_qc_select_method_bar.png)

## (e) Distributions (OK only)
### R²
- R² min/max: 0.5571 / 1
- R² q10: 0.8092
- R² q25: 0.9598
- R² q50: 0.9929
- R² q75: 0.9986
- R² q90: 0.9996

![r2 hist](fit_qc_r2_hist.png)

### mono_frac
- mono_frac min/max: 0.7143 / 1
- mono_frac q10: 1
- mono_frac q25: 1
- mono_frac q50: 1
- mono_frac q75: 1
- mono_frac q90: 1

![mono_frac hist](fit_qc_mono_frac_hist.png)

### snr
- snr min/max: 2.773 / 747.1
- snr q10: 6.084
- snr q25: 13.35
- snr q50: 26.06
- snr q75: 43.78
- snr q90: 85.9

![snr hist](fit_qc_snr_hist_log10.png)

## (f) Exclude reasons (EXCLUDED only)
- CSV: fit_qc_exclude_reason_norm_counts.csv
- R² < r2_min: 4 (100.0%)

![exclude_reason_norm](fit_qc_exclude_reason_norm_bar.png)
