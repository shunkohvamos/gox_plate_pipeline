# Fit QC Report

- Generated: 2026-02-12 00:15:23.157686

## (a) OK / EXCLUDED
- Total wells: 112
- OK: 112
- EXCLUDED: 0
- OK rate: 100.0%

- CSV: fit_qc_summary_overall.csv
- CSV (by plate): fit_qc_summary_by_plate.csv
- CSV (by heat): fit_qc_summary_by_heat.csv

## (b) Selected t_end distribution
- t_end min/max: 96 / 606 s
- q10: 96 s
- q25: 96 s
- q50: 156 s
- q75: 471 s
- q90: 606 s

- t_end ≤ 30 s : 0.0%
- t_end ≤ 60 s : 0.0%
- t_end ≤ 120 s : 39.3%
- t_end ≤ 240 s : 67.9%
- t_end ≤ 600 s : 75.0%

![t_end hist](fit_qc_t_end_hist.png)

## (c) Slope vs t_end
- N (finite): 112
- Pearson r: -0.6937
- Spearman ρ: -0.7746

![slope vs t_end](fit_qc_slope_vs_t_end.png)

## (d) select_method_used breakdown (OK only)
- method column used: select_method_used
- force_whole* fraction (among OK): 0.0%
- force_whole* fraction (among ALL wells): 0.0%

- CSV: fit_qc_select_method_counts.csv
- initial_positive_tangent: 42 (37.5%)
- initial_positive_promote_long_ext: 30 (26.8%)
- initial_positive: 26 (23.2%)
- initial_positive_ext: 3 (2.7%)
- initial_positive_promote_long_ext_tangent: 3 (2.7%)
- initial_positive_early_steep_ext: 2 (1.8%)
- initial_positive_post_tangent_under: 2 (1.8%)
- initial_positive_ext_intskip1: 1 (0.9%)
- initial_positive_early_steep_tangent: 1 (0.9%)
- initial_positive_post_broad_overfit_ext: 1 (0.9%)
- initial_positive_intskip1: 1 (0.9%)

![select_method_used](fit_qc_select_method_bar.png)

## (e) Distributions (OK only)
### R²
- R² min/max: 0.9459 / 0.9999
- R² q10: 0.9876
- R² q25: 0.9977
- R² q50: 0.999
- R² q75: 0.9994
- R² q90: 0.9996

![r2 hist](fit_qc_r2_hist.png)

### mono_frac
- mono_frac min/max: 0.9474 / 1
- mono_frac q10: 1
- mono_frac q25: 1
- mono_frac q50: 1
- mono_frac q75: 1
- mono_frac q90: 1

![mono_frac hist](fit_qc_mono_frac_hist.png)

### snr
- snr min/max: 10.93 / 287
- snr q10: 22.81
- snr q25: 47.45
- snr q50: 78.87
- snr q75: 112.9
- snr q90: 154.3

![snr hist](fit_qc_snr_hist_log10.png)

## (f) Exclude reasons (EXCLUDED only)
- excluded wells: 0
