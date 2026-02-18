# Methods (one-page summary)

Single-page summary of definitions and settings for **thermal stability assay**, **initial-rate fitting**, **t50 / FoG**, and **Bayesian optimization**. Suitable for paper Methods or SI. Values below are filled from pipeline code defaults when you run **Update Methods one-page** (Run and Debug) or `scripts/update_methods_one_page.py`.

---

## 1. Thermal stability assay and REA

- **Signal**: Product accumulation over time (e.g. Amplex Red / resorufin). Per-well time series are recorded at several heating times (e.g. 0, 5, 10, 15, 20, 30 min at target temperature).
- **Absolute activity**: Initial rate (slope of the linear phase) at each heating time, from the selected window (see §2).
- **REA (Retained Enzyme Activity), %**: For each (plate, polymer, heat_min), REA_percent = 100 × (absolute activity at that heat_min) / (absolute activity at heat_min = 0). So 100% = no loss; lower values indicate thermal inactivation.

---

## 2. Initial-rate window selection (design principles)

- **Goal**: Estimate the **initial rate** as the slope of a **local linear approximation** in the **initial regime** (before substrate depletion, product inhibition, or strong curvature). Do not fit a straight line into an already curved or flat region.
- **Window selection priority** (in order):
  1. **Earlier start** is preferred (late start is penalized).
  2. **Higher linearity** (e.g. R²) within that early region.
  3. **Sufficient point count** (minimum number of points respected).
- **Leading-point skip**: Allowed only to skip a **true lag** (sustained flat or noisy start, e.g. mixing or equilibration). A shallow dip or single-point noise is not treated as lag; the first points remain eligible. A **skip cap** (number of points or time) is enforced.
- **Default implementation parameters** (from `compute_rates_and_rea` / `select_fit`):
  - Method: `initial_positive`.
  - Minimum points in window: 6 (rescue down to 4–5 with relaxed R²).
  - Maximum points: 30.
  - R² minimum: 0.96 (relaxed in rescue steps, e.g. 0.80 → 0.70 → 0.60 → 0.50 for shorter windows).
  - Start detection: `find_start=true`, `start_max_shift=5`, `start_window=3`.
  - Monotonicity: `mono_min_frac=0.85`, `min_pos_steps=2`, `min_snr=3.0`.
  - Maximum t_end: 240.0 s (optional cap so that only the initial phase is used).
- **Audit**: When the start index is advanced (leading points skipped), the skip count or first fitted time is recorded in the output (CSV/log).

---

## 3. t50 (half-life) definition

- **t50**: Heating time (minutes) at which REA reaches a defined **threshold**. Unit: **min** (throughout the pipeline).
- **Two canonical modes** (selected at run time; default: **y0_half**):

  | Mode      | Threshold              | Typical use |
  |-----------|------------------------|-------------|
  | **y0_half** | threshold = 0.5 * fitted y0 | Same-run relative; adapts if baseline differs. |
  | **rea50**   | Fixed REA = 50%        | Cross-run / cross-study comparison. |

- **Fitting**: REA vs. heating time (min) is fitted per (run, plate, polymer). Preferred: exponential decay (or exp with plateau); fallback: linear interpolation to threshold. Minimum 3 points; at least 2 distinct heating times.
- **Default in scripts**: `--t50_definition y0_half` (e.g. `fit_initial_rates.py`, `build_fog_plate_aware.py`, `run_bayesian_optimization.py` uses FoG built with the same definition).

---

## 4. FoG (Fold over GOx)

- **Definition**:  
  **FoG = t50_polymer / t50_bare_GOx**  
  (same run, or same plate / same round when using plate-aware FoG).
- **Interpretation**: FoG > 1 means the polymer–enzyme sample retains activity longer than bare GOx under the same conditions; FoG is a dimensionless thermal-stability ratio.
- **Constrained objective (BO)**: Only polymers with **sufficient native (unheated) activity** are included in the optimization objective. Default: `native_activity_rel_at_0 >= 0.7` (70% of GOx baseline at heat = 0). The objective column is **log_fog_native_constrained** (log(FoG) where the constraint is satisfied; NaN otherwise).
- **Plate-aware denominator**: For each (run, plate, polymer), the GOx t50 used in the denominator is from the same (run, plate) when available; otherwise a round-level representative (e.g. median) is used. Outlier GOx t50 values can be detected and optionally excluded; same-plate values outside a guard range fall back to the round representative.

---

## 5. Bayesian optimization (BO) settings

- **Objective**: Maximize **log(FoG)** (log_fog_native_constrained) on the design space of **ternary composition** (MPC, BMA, MTAC; fractions sum to 1).
- **Model**: Gaussian process regression (GPR), **Matern-5/2** kernel. Default: **ARD** (one length scale per component); for sparse design points (e.g. ≤10 unique points), **isotropic** kernel is used by default to avoid unstable anisotropy (optional: `--disable_sparse_isotropic`).
- **Acquisition**: **EI (Expected Improvement)** by default; optional **UCB**. Batch proposals combine **exploit** (EI or UCB) and **explore** (high predictive uncertainty).
  - Default: `--acquisition ei`, `--exploration_ratio 0.35`, `--n_suggestions 8`.
  - EI: `--ei_xi 0.01` (jitter). UCB: `--ucb_kappa 2.0` (or `--ucb_beta` if set).
- **Batch composition**: Part of the batch is reserved for **anchor** (fixed-composition re-measurements, e.g. PMPC,PMTAC) and **replicates** (re-measurements of high-value compositions). Default: `--anchor_fraction 0.12`, `--replicate_fraction 0.12`, `--anchor_polymer_ids PMPC,PMTAC`.
- **Constraints**: Composition fractions in [0.02, 0.95] per component (default `--min_component 0.02`, `--max_component 0.95`); sum = 1; **minimum distance** between suggested points and between suggestions and existing training points (`--min_distance_between 0.06`, `--min_distance_to_train 0.03`).
- **Round correction**: **Anchor-based correction** of log(FoG) across rounds is **off by default** (`--anchor_correction` to enable), with a minimum number of shared anchor polymers (default: 2).
- **Proposal grid**: Candidates are evaluated on a simplex grid (e.g. `--candidate_step 0.02`); `--n_random_candidates 5000` for random (x,y) sampling where applicable.
- **Outputs**: Each BO run produces a unique **bo_run_id**; outputs include proposal log (candidates, acquisition values, constraints), **next_experiment_top5** table, surrogate maps (ternary: mean, std, EI, UCB; optional 2×2 panels), Observed vs. Predicted plot, and ranking tables (t50, FoG). All referenced run_ids and parameters are stored in the run manifest.

---

## 6. Traceability

- Every derived output (CSV, figures) is tied to a **run_id** (and **bo_run_id** for BO). A **run manifest** (JSON) records run_id, timestamp, git commit, input paths and hashes, and main parameters.
- Lineage (e.g. well-level ancestry) is kept in separate CSVs where needed; long lineage strings are not embedded in summary tables. Figure filenames include run_id or bo_run_id; raw paths are not written in figures.

---

## Regenerating this page from code

This file (**METHODS_ONE_PAGE.md**) is generated from **METHODS_ONE_PAGE_TEMPLATE.md** and the default values in code.

- **Run/debug で更新**: VS Code の Run and Debug で **「Update Methods one-page」** を実行すると、このファイルがコードのデフォルトで上書きされます。
- **手動**: リポジトリルートで `PYTHONPATH=src .venv/bin/python scripts/update_methods_one_page.py` を実行しても同じです。
- **コード変更時に更新** (optional): `pre-commit install` でフックを有効にすると、t50/FoG/初速/BO のデフォルトを定義しているファイルを変更してコミットするときに、自動でこのページが再生成されます。変更後に `research strategy/METHODS_ONE_PAGE.md` を add してから再度コミットしてください。

Defaults are defined in:

- **t50**: `src/gox_plate_pipeline/polymer_timeseries.py`; `src/gox_plate_pipeline/fog.py`.
- **Initial rate**: `src/gox_plate_pipeline/fitting/pipeline.py`, `selection.py`; `.cursor/rules/` (design rules).
- **BO**: `scripts/run_bayesian_optimization.py`; `src/gox_plate_pipeline/bo_engine.py`.
