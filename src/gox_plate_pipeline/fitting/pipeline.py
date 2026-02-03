# src/gox_plate_pipeline/fitting/pipeline.py
"""
Main pipeline for computing rates and REA.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .core import FitSelectionError
from .preprocessing import add_heat_time
from .candidates import fit_initial_rate_one_well
from .selection import (
    select_fit,
    try_skip_extend,
    try_extend_fit,
    _enforce_final_safety,
    find_best_short_window,
    find_fit_with_outlier_removal,
    detect_internal_outliers,
    detect_curvature_and_shorten,
    fit_with_outlier_skip_full_range,
)
from .plotting import plot_fit_diagnostic
from .qc import write_fit_qc_report


def compute_rates_and_rea(
    tidy: pd.DataFrame,
    heat_times: List[float],
    min_points: int = 6,
    max_points: int = 20,
    min_span_s: float = 0.0,
    select_method: str = "initial_positive",
    r2_min: float = 0.98,
    slope_min: float = 0.0,
    max_t_end: Optional[float] = 240.0,
    mono_eps: Optional[float] = None,
    min_delta_y: Optional[float] = None,
    find_start: bool = True,
    start_max_shift: int = 5,
    start_window: int = 3,
    start_allow_down_steps: int = 1,
    mono_min_frac: float = 0.85,
    mono_max_down_steps: int = 1,
    min_pos_steps: int = 2,
    min_snr: float = 3.0,
    slope_drop_frac: float = 0.18,
    plot_dir: Optional[Path] = None,
    plot_mode: str = "all",
    qc_report_dir: Optional[Path] = None,
    qc_prefix: str = "fit_qc",
    force_whole: bool = False,
    force_whole_n_min: int = 10,
    force_whole_r2_min: float = 0.985,
    force_whole_mono_min_frac: float = 0.70,
    min_t_start_s: float = 0.0,
    down_step_min_frac: Optional[float] = None,
    fit_method: str = "ols",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-well initial rates (absolute activity) and REA (%).

    Absolute activity:
      - slope from selected linear window per well (no row-averaging)

    REA (%):
      - for each (plate_id, polymer_id), normalize each heat_min by baseline at heat_min==0
      - REA_percent = 100 * abs_activity / baseline_abs_activity
    """
    df = add_heat_time(tidy, heat_times=heat_times)

    selected_rows: list[dict] = []

    group_cols = ["plate_id", "well"]
    if not all(c in df.columns for c in group_cols):
        raise ValueError(f"tidy must contain columns {group_cols}, got: {df.columns.tolist()}")

    plot_mode = str(plot_mode).lower().strip()
    if plot_mode not in {"all", "ok", "excluded"}:
        raise ValueError("plot_mode must be one of: all, ok, excluded")

    for (plate_id, well), g in df.groupby(group_cols, sort=False):
        def _nunique_nonnull(s: pd.Series) -> int:
            if s is None or s.empty:
                return 0
            x = s.dropna()
            x = x[~(x.astype(str).str.strip() == "")]
            return int(x.nunique())

        meta_fields = ["polymer_id", "heat_min", "col", "row", "sample_name"]
        bad = []
        for k in meta_fields:
            if k in g.columns:
                if _nunique_nonnull(g[k]) > 1:
                    bad.append(k)

        if bad:
            raise ValueError(
                f"Inconsistent metadata within group (plate_id={plate_id}, well={well}): {bad}. "
                "Rowmap/merge may be broken."
            )

        base = {
            "plate_id": plate_id,
            "well": well,
            "row": g["row"].iloc[0] if "row" in g.columns else "",
            "col": int(g["col"].iloc[0]) if "col" in g.columns and pd.notna(g["col"].iloc[0]) else np.nan,
            "heat_min": float(g["heat_min"].iloc[0]) if "heat_min" in g.columns else np.nan,
            "polymer_id": g["polymer_id"].iloc[0] if "polymer_id" in g.columns else "",
            "sample_name": g["sample_name"].iloc[0] if "sample_name" in g.columns else "",
            "source_file": g["source_file"].iloc[0] if "source_file" in g.columns else "",
        }

        status = "excluded"
        exclude_reason = ""
        sel: Optional[pd.Series] = None

        try:
            cands = fit_initial_rate_one_well(
                g,
                min_points=int(min_points),
                max_points=int(max_points),
                min_span_s=float(min_span_s),
                mono_eps=mono_eps,
                min_delta_y=min_delta_y,
                find_start=bool(find_start),
                start_max_shift=int(start_max_shift),
                start_window=int(start_window),
                start_allow_down_steps=int(start_allow_down_steps),
                min_t_start_s=float(min_t_start_s),
                down_step_min_frac=down_step_min_frac,
                fit_method=str(fit_method),
            )

            # Fallback attempts: progressively relax parameters for difficult data.
            # Final fallback uses r2_min=0.80 to avoid fitting pure noise.
            fallback_attempts = [
                {"r2_min": r2_min, "min_delta_y": min_delta_y, "mono_min_frac": mono_min_frac, "min_pos_steps": min_pos_steps},
                {"r2_min": 0.92, "min_delta_y": min_delta_y, "mono_min_frac": mono_min_frac, "min_pos_steps": min_pos_steps},
                {"r2_min": r2_min, "min_delta_y": 0.0, "mono_min_frac": mono_min_frac, "min_pos_steps": min_pos_steps},
                {"r2_min": 0.88, "min_delta_y": 0.0, "mono_min_frac": 0.75, "min_pos_steps": min_pos_steps},
                {"r2_min": 0.80, "min_delta_y": 0.0, "mono_min_frac": 0.65, "min_pos_steps": 0},
            ]
            sel = None
            last_err = None
            used_params = fallback_attempts[0]
            for attempt_params in fallback_attempts:
                try:
                    sel = select_fit(
                        cands,
                        method=select_method,
                        r2_min=attempt_params["r2_min"],
                        slope_min=slope_min,
                        max_t_end=max_t_end,
                        min_delta_y=attempt_params["min_delta_y"],
                        mono_min_frac=attempt_params["mono_min_frac"],
                        mono_max_down_steps=mono_max_down_steps,
                        min_pos_steps=attempt_params["min_pos_steps"],
                        min_snr=min_snr,
                        slope_drop_frac=slope_drop_frac,
                        force_whole=force_whole,
                        force_whole_n_min=force_whole_n_min,
                        force_whole_r2_min=force_whole_r2_min,
                        force_whole_mono_min_frac=force_whole_mono_min_frac,
                    )
                    used_params = attempt_params
                    break
                except FitSelectionError as e:
                    last_err = e
                    continue
            if sel is None:
                # Last resort 1: try to find best short window for very noisy data
                t_arr = g["time_s"].to_numpy(dtype=float)
                y_arr = g["signal"].to_numpy(dtype=float)
                sel = find_best_short_window(
                    t=t_arr,
                    y=y_arr,
                    min_points=4,
                    max_points=8,
                    r2_min=0.55,  # Relaxed to allow noisy initial sections
                    slope_min=slope_min,
                    min_snr=1.5,  # Slightly relaxed SNR for initial rate priority
                )
                if sel is not None:
                    # Found a decent short window - use relaxed thresholds
                    used_params = {
                        "r2_min": 0.55,  # Match find_best_short_window threshold
                        "min_delta_y": 0.0,
                        "mono_min_frac": 0.0,
                        "min_pos_steps": 0,
                        "mono_max_down_steps": 99,  # Relaxed for last_resort
                        "min_snr": 1.5,  # Match find_best_short_window threshold
                    }
                else:
                    # Last resort 2: try fitting with single outlier removal
                    sel = find_fit_with_outlier_removal(
                        t=t_arr,
                        y=y_arr,
                        min_points=6,
                        r2_min=0.80,
                        slope_min=slope_min,
                        min_snr=2.0,
                        outlier_sigma=3.0,
                    )
                    if sel is not None:
                        used_params = {
                            "r2_min": 0.80,
                            "min_delta_y": 0.0,
                            "mono_min_frac": 0.0,
                            "min_pos_steps": 0,
                            "mono_max_down_steps": 99,
                            "min_snr": 2.0,
                        }
                    else:
                        raise last_err  # type: ignore[misc]

            t_arr = g["time_s"].to_numpy(dtype=float)
            y_arr = g["signal"].to_numpy(dtype=float)
            
            # Step 1: Try full-range outlier skip first (C1, C3 case)
            # This handles single outliers where we want to include points
            # BEFORE and AFTER the outlier
            full_range_fit = fit_with_outlier_skip_full_range(
                t=t_arr,
                y=y_arr,
                outlier_sigma=3.5,
                r2_min=0.98,
                min_points=10,
                r2_improvement_min=0.005,
            )
            # Prefer full-range one-outlier rescue when quality is comparable.
            # This keeps the intended contiguous domain instead of truncating
            # before an isolated outlier.
            full_range_r2_tolerance = 0.005
            if (
                full_range_fit is not None
                and float(full_range_fit["r2"]) + float(full_range_r2_tolerance) >= float(sel["r2"])
            ):
                sel = full_range_fit
                used_params = {
                    "r2_min": 0.98,
                    "min_delta_y": 0.0,
                    "mono_min_frac": 0.0,
                    "min_pos_steps": 0,
                    "mono_max_down_steps": 99,
                    "min_snr": 2.0,
                }
            
            # Step 2: Try to extend
            sel = try_extend_fit(
                sel,
                t=t_arr,
                y=y_arr,
                r2_min=r2_min,
                r2_drop_tolerance=0.02,
            )
            
            # Step 3: Try to skip outliers at the end and extend
            sel = try_skip_extend(
                sel,
                t=t_arr,
                y=y_arr,
                r2_min=r2_min,
                max_skip=2,
                min_extend_points=3,
                residual_threshold_sigma=4.5,
                max_t_end=None,
            )
            
            # Step 4: Try extending again after skipping outliers
            sel = try_extend_fit(
                sel,
                t=t_arr,
                y=y_arr,
                r2_min=r2_min,
                r2_drop_tolerance=0.02,
            )
            
            # Step 5: Detect and remove internal outliers (C2 case)
            # Only remove if R² improves by at least 0.005 (keep marginal points for initial velocity)
            sel = detect_internal_outliers(
                sel,
                t=t_arr,
                y=y_arr,
                outlier_sigma=3.0,
                r2_min=0.98,
                r2_improvement_min=0.005,
                max_internal_outliers=2,
            )
            
            # Step 6: Detect curvature and shorten for tangent fit (D1-D5 case)
            # If we already selected a full-range one-outlier rescue,
            # keep that full range and do not shorten to a tangent window.
            method_before_curvature = str(sel.get("select_method_used", ""))
            if "full_range_outlier_skip" not in method_before_curvature:
                sel = detect_curvature_and_shorten(
                    sel,
                    t=t_arr,
                    y=y_arr,
                    r2_min=0.97,
                    curvature_threshold=0.15,
                )

            select_method_used = str(sel.get("select_method_used", select_method))

            # Relax max_t_end for extended/skip fits and outlier-removed fits
            extended = ("skip" in select_method_used) or ("_ext" in select_method_used) or ("outlier" in select_method_used)
            safety_max_t_end = None if extended else max_t_end

            _enforce_final_safety(
                sel,
                slope_min=slope_min,
                r2_min=used_params["r2_min"],
                max_t_end=safety_max_t_end,
                min_delta_y=used_params["min_delta_y"],
                mono_min_frac=used_params["mono_min_frac"],
                mono_max_down_steps=used_params.get("mono_max_down_steps", mono_max_down_steps),
                min_pos_steps=used_params["min_pos_steps"],
                min_snr=used_params.get("min_snr", min_snr),
            )

            status = "ok"

            row = {
                **base,
                "status": status,
                "abs_activity": float(sel["slope"]),
                "slope": float(sel["slope"]),
                "intercept": float(sel["intercept"]),
                "r2": float(sel["r2"]),
                "n": int(sel["n"]),
                "t_start": float(sel["t_start"]),
                "t_end": float(sel["t_end"]),
                "select_method": select_method_used,
                "select_method_used": select_method_used,
                "exclude_reason": "",
                "dy": float(sel["dy"]),
                "mono_frac": float(sel["mono_frac"]),
                "down_steps": int(sel["down_steps"]),
                "pos_steps": int(sel["pos_steps"]),
                "pos_steps_eps": int(sel["pos_steps_eps"]) if "pos_steps_eps" in sel.index else np.nan,
                "pos_eps": float(sel["pos_eps"]) if "pos_eps" in sel.index else np.nan,
                "rmse": float(sel["rmse"]) if pd.notna(sel["rmse"]) else np.nan,
                "snr": float(sel["snr"]) if pd.notna(sel["snr"]) else np.nan,
                "start_idx_used": int(sel["start_idx_used"]) if "start_idx_used" in sel.index else np.nan,
                "skip_indices": str(sel.get("skip_indices", "")) if "skip_indices" in sel.index else "",
                "skip_count": int(sel.get("skip_count", 0)) if "skip_count" in sel.index else 0,
            }

        except FitSelectionError as e:
            status = "excluded"
            exclude_reason = str(e)

            row = {
                **base,
                "status": status,
                "abs_activity": np.nan,
                "slope": np.nan,
                "intercept": np.nan,
                "r2": np.nan,
                "n": np.nan,
                "t_start": np.nan,
                "t_end": np.nan,
                "select_method": select_method,
                "select_method_used": select_method,
                "exclude_reason": exclude_reason,
                "dy": np.nan,
                "mono_frac": np.nan,
                "down_steps": np.nan,
                "pos_steps": np.nan,
                "rmse": np.nan,
                "snr": np.nan,
                "start_idx_used": np.nan,
            }

        except Exception:
            raise

        selected_rows.append(row)

        if plot_dir is not None:
            do_plot = (
                (plot_mode == "all")
                or (plot_mode == "ok" and status == "ok")
                or (plot_mode == "excluded" and status != "ok")
            )
            if do_plot:
                out_png = plot_dir / f"{plate_id}" / f"{well}.png"
                plot_fit_diagnostic(
                    df_well=g,
                    meta=base,
                    selected=sel if status == "ok" else None,
                    status=status,
                    exclude_reason=exclude_reason,
                    out_png=out_png,
                )

    if selected_rows:
        selected = pd.DataFrame(selected_rows)
    else:
        selected = pd.DataFrame(
            columns=[
                "plate_id", "well", "row", "col", "heat_min", "polymer_id",
                "sample_name", "source_file", "status", "abs_activity", "slope",
                "intercept", "r2", "n", "t_start", "t_end", "select_method",
                "select_method_used", "exclude_reason", "dy", "mono_frac",
                "down_steps", "pos_steps", "pos_steps_eps", "pos_eps", "rmse",
                "snr", "start_idx_used",
            ]
        )

    if qc_report_dir is not None:
        qc_dir = Path(qc_report_dir)
        qc_dir.mkdir(parents=True, exist_ok=True)
        write_fit_qc_report(
            selected=selected,
            out_dir=qc_dir,
            max_t_end=max_t_end,
            prefix=str(qc_prefix),
        )

    baseline_mask = (selected["status"] == "ok") & np.isfinite(selected["heat_min"].to_numpy(dtype=float))
    baseline_mask = baseline_mask & np.isclose(selected["heat_min"].to_numpy(dtype=float), 0.0, atol=1e-12)

    baseline = (
        selected[baseline_mask]
        .groupby(["plate_id", "polymer_id"], dropna=False)["abs_activity"]
        .median()
        .rename("baseline_abs_activity")
        .reset_index()
    )

    rea = selected.merge(baseline, on=["plate_id", "polymer_id"], how="left")
    rea["REA_percent"] = 100.0 * rea["abs_activity"] / rea["baseline_abs_activity"]

    return selected, rea
