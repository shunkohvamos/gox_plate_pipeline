# Fit QC Report

- Generated: 2026-02-06 12:54:16.356266

## (a) OK / EXCLUDED
- Total wells: 56
- OK: 56
- EXCLUDED: 0
- OK rate: 100.0%

- CSV: fit_qc_summary_overall.csv
- CSV (by plate): fit_qc_summary_by_plate.csv
- CSV (by heat): fit_qc_summary_by_heat.csv

## (b) Selected t_end distribution
- t_end min/max: 96 / 606 s
- q10: 96 s
- q25: 96 s
- q50: 126 s
- q75: 276 s
- q90: 576 s

- t_end ≤ 30 s : 0.0%
- t_end ≤ 60 s : 0.0%
- t_end ≤ 120 s : 37.5%
- t_end ≤ 240 s : 64.3%
- t_end ≤ 600 s : 91.1%

![t_end hist](fit_qc_t_end_hist.png)

## (c) Slope vs t_end
- N (finite): 56
- Pearson r: -0.818
- Spearman ρ: -0.9195

![slope vs t_end](fit_qc_slope_vs_t_end.png)

## (d) select_method_used breakdown (OK only)
- method column used: select_method_used
- force_whole* fraction (among OK): 0.0%
- force_whole* fraction (among ALL wells): 0.0%

- CSV: fit_qc_select_method_counts.csv
- initial_positive_tangent: 19 (33.9%)
- initial_positive_ext_tangent: 17 (30.4%)
- initial_positive: 8 (14.3%)
- initial_positive_ext: 5 (8.9%)
- full_range_outlier_skip: 3 (5.4%)
- initial_positive_ext_intskip1_tangent: 2 (3.6%)
- initial_positive_ext_skip1: 1 (1.8%)
- initial_positive_post_long_ext: 1 (1.8%)

![select_method_used](fit_qc_select_method_bar.png)

## (e) Distributions (OK only)
### R²
- R² min/max: 0.8815 / 0.9994
- R² q10: 0.9604
- R² q25: 0.9762
- R² q50: 0.9899
- R² q75: 0.9949
- R² q90: 0.9968

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
- snr min/max: 7.03 / 111.3
- snr q10: 11.1
- snr q25: 17.73
- snr q50: 26.94
- snr q75: 39.79
- snr q90: 49.21

![snr hist](fit_qc_snr_hist_log10.png)

## (f) Exclude reasons (EXCLUDED only)
- excluded wells: 0
