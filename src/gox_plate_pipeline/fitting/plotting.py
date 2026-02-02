# src/gox_plate_pipeline/fitting/plotting.py
"""
Diagnostic plotting functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from .core import apply_paper_style, PAPER_FIGSIZE_SINGLE


def _format_heat(heat_min: object) -> str:
    if heat_min is None or (isinstance(heat_min, float) and np.isnan(heat_min)):
        return "NA"
    try:
        return f"{float(heat_min):g} min"
    except Exception:
        return str(heat_min)


def plot_fit_diagnostic(
    df_well: pd.DataFrame,
    meta: dict,
    selected: Optional[pd.Series],
    status: str,
    exclude_reason: str,
    out_png: Path,
) -> None:
    """
    Diagnostic plot for one well.

    Requirements:
      - If excluded: do NOT draw fit line (show "cannot fit" style)
      - If ok: draw dotted line across full plot x-range (edge-to-edge)
      - Title is minimal: plate_id + well + polymer_id (and "EXCLUDED" when excluded)
      - Info box contains: heat_min (+ sample_name if present) and (if ok) slope + R² + n
      - Use unicode R² (superscript 2)
      - Line should be fine dotted
    """
    d = df_well.copy()
    d["time_s"] = pd.to_numeric(d["time_s"], errors="coerce")
    d["signal"] = pd.to_numeric(d["signal"], errors="coerce")
    d = d.dropna(subset=["time_s", "signal"]).sort_values("time_s").reset_index(drop=True)

    mask_finite = np.isfinite(d["time_s"].to_numpy(dtype=float)) & np.isfinite(d["signal"].to_numpy(dtype=float))
    d = d.loc[mask_finite].reset_index(drop=True)

    t = d["time_s"].to_numpy(dtype=float)
    y = d["signal"].to_numpy(dtype=float)

    well = str(meta.get("well", "NA"))
    polymer_id = str(meta.get("polymer_id", "") or "")
    heat_txt = _format_heat(meta.get("heat_min", np.nan))

    _raw_sample = meta.get("sample_name", "")
    sample_name = "" if pd.isna(_raw_sample) else str(_raw_sample).strip()
    if sample_name.lower() in {"nan", "none", "na", "n/a"}:
        sample_name = ""

    # Colors: paper-grade palette
    c_point = "#0072B2"         # blue (fitted points)
    c_fit = "#E07020"           # burnt orange (fit line) - warm, sophisticated
    c_drop = "#FF3B3B"          # fluorescent red (skipped outlier points)
    c_edge = "#2D2D2D"          # dark gray, almost black (marker edge for all points)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(apply_paper_style()):
        # Paper-grade figure size: ~1 column width (90mm ≈ 3.5 in)
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE_SINGLE)

        # highlight only the points that were explicitly excluded by trim or skip
        drop_mask = np.zeros(len(t), dtype=bool)

        if status == "ok" and selected is not None:
            method_used = str(selected.get("select_method_used", ""))
            # trim-dropped points
            if ("trim1" in method_used) or ("trim2" in method_used):
                for key in ["trim1_drop_idx", "trim2_drop_idx1", "trim2_drop_idx2"]:
                    if key in selected.index and pd.notna(selected[key]):
                        j = int(selected[key])
                        if 0 <= j < len(drop_mask):
                            drop_mask[j] = True
            # skip-extended outlier points
            if "skip_indices" in selected.index and pd.notna(selected.get("skip_indices", "")):
                skip_str = str(selected["skip_indices"])
                if skip_str:
                    for idx_str in skip_str.split(","):
                        idx_str = idx_str.strip()
                        if idx_str.isdigit():
                            j = int(idx_str)
                            if 0 <= j < len(drop_mask):
                                drop_mask[j] = True

        # Paper-grade axes (full frame, uniform width)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.3")
            spine.set_linewidth(0.6)
        ax.tick_params(axis="both", which="both", width=0.5, colors="0.35")

        # thin polyline to connect points (very light, paper-grade)
        if status == "ok" and selected is not None and len(t) > 1:
            t_line = t[~drop_mask]
            y_line = y[~drop_mask]
            if t_line.size >= 2:
                ax.plot(
                    t_line,
                    y_line,
                    linewidth=0.4,  # paper-grade: minimal
                    alpha=0.25,
                    color="0.5",
                    zorder=2.6,
                )

        # scatter with edge lines - paper-grade marker size
        if len(t) > 0:
            # Fitted points: full opacity with gray edge
            ax.scatter(
                t[~drop_mask], y[~drop_mask],
                s=12,
                color=c_point,
                edgecolors=c_edge,
                linewidths=0.4,
                alpha=0.95,
                zorder=3,
            )
            # Skipped/excluded points: moderate opacity with gray edge
            if np.any(drop_mask):
                ax.scatter(
                    t[drop_mask], y[drop_mask],
                    s=12,
                    color=c_drop,
                    edgecolors=c_edge,
                    linewidths=0.4,
                    alpha=0.70,
                    zorder=4,
                )

        x0 = float(np.min(t)) if len(t) else 0.0
        x1 = float(np.max(t)) if len(t) else 1.0

        slope_txt = "NA"
        r2_txt = "NA"
        n_txt = "NA"

        if status == "ok" and selected is not None and len(t) > 0:
            slope = float(selected["slope"])
            intercept = float(selected["intercept"])
            r2 = float(selected["r2"])
            n_txt = str(int(selected.get("n", np.nan))) if pd.notna(selected.get("n", np.nan)) else "NA"

            slope_txt = f"{slope:.3g}"
            r2_txt = f"{r2:.3f}"

            # selected window shading
            t0 = float(selected["t_start"])
            t1 = float(selected["t_end"])
            ax.axvspan(t0, t1, facecolor="0.85", alpha=0.18, zorder=2)

            # fit line across full plot x-range (edge-to-edge)
            xa, xb = ax.get_xlim()
            xx = np.array([xa, xb], dtype=float)
            yy = slope * xx + intercept
            ax.plot(
                xx,
                yy,
                linestyle=(0, (1, 2)),
                linewidth=0.8,  # paper-grade: 0.6-0.9 pt
                alpha=0.9,
                color=c_fit,
                zorder=2.7,
            )
            ax.set_xlim(xa, xb)

        plate_txt = str(meta.get("plate_id", "NA"))
        poly_txt = polymer_id or "NA"
        base_title = f"{plate_txt} | Well {well} | {poly_txt}"
        ax.set_title(base_title, pad=4)  # uses rcParams font size

        ax.set_xlabel("Time (s)")  # uses rcParams font size
        ax.set_ylabel("Signal (a.u.)")  # uses rcParams font size

        # info box (paper-grade: 6 pt font)
        info_lines = [f"heat: {heat_txt}"]

        if status == "ok":
            if sample_name:
                info_lines.append(f"sample: {sample_name}")
            info_lines.append(f"slope: {slope_txt}")
            info_lines.append(f"R\u00b2: {r2_txt}")
            info_lines.append(f"n: {n_txt}")

        txt = ax.text(
            0.02,
            0.98,
            "\n".join(info_lines),
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=6,  # paper-grade: 5-8 pt
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.7", linewidth=0.4),
            zorder=10,
        )
        # Add soft shadow to info box (right-bottom 45°)
        if txt.get_bbox_patch() is not None:
            txt.get_bbox_patch().set_path_effects([
                path_effects.SimplePatchShadow(
                    offset=(2.5, -2.5),  # 45° right-bottom, slightly further
                    shadow_rgbFace="0.5",  # lighter gray
                    alpha=0.12,  # more transparent
                    rho=0.5,  # blur radius (higher = more blur)
                ),
                path_effects.Normal(),
            ])

        fig.tight_layout(pad=0.3)
        fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
