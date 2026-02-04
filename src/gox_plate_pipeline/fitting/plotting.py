# src/gox_plate_pipeline/fitting/plotting.py
"""
Diagnostic plotting functions and plate grid assembly.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patheffects as path_effects

from .core import (
    apply_paper_style,
    INFO_BOX_MARGIN_PT,
    INFO_BOX_FACE_COLOR,
    INFO_BOX_PAD_DEFAULT,
    get_info_box_gradient_shadow,
    PAPER_FIGSIZE_SINGLE,
)

# Well label pattern: one letter A-H, then 1-2 digits (col 1-12); we use cols 1-7 only
_WELL_PATTERN = re.compile(r"^([A-H])(\d+)$")


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
    c_drop = "#FF1744"          # fluorescent red (skipped points), distinct from fit orange
    c_edge = "#2D2D2D"          # dark gray, almost black (marker edge for all points)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(apply_paper_style()):
        # Paper-grade figure size: ~1 column width (90mm ≈ 3.5 in)
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE_SINGLE)

        # highlight only the points that were explicitly excluded by trim or skip
        drop_mask = np.zeros(len(t), dtype=bool)

        if status == "ok" and selected is not None:
            fit_start_idx = 0
            fit_end_idx = len(t) - 1
            if ("start_idx" in selected.index) and pd.notna(selected.get("start_idx", np.nan)):
                fit_start_idx = max(0, int(selected["start_idx"]))
            if ("end_idx" in selected.index) and pd.notna(selected.get("end_idx", np.nan)):
                fit_end_idx = min(len(t) - 1, int(selected["end_idx"]))
            if fit_end_idx < fit_start_idx:
                fit_start_idx, fit_end_idx = 0, len(t) - 1

            method_used = str(selected.get("select_method_used", ""))
            # trim-dropped points
            if ("trim1" in method_used) or ("trim2" in method_used):
                for key in ["trim1_drop_idx", "trim2_drop_idx1", "trim2_drop_idx2"]:
                    if key in selected.index and pd.notna(selected[key]):
                        j = int(selected[key])
                        if fit_start_idx <= j <= fit_end_idx:
                            drop_mask[j] = True
            # skip-extended outlier points
            if "skip_indices" in selected.index and pd.notna(selected.get("skip_indices", "")):
                skip_str = str(selected["skip_indices"])
                if skip_str:
                    for idx_str in skip_str.split(","):
                        idx_str = idx_str.strip()
                        if idx_str.isdigit():
                            j = int(idx_str)
                            if fit_start_idx <= j <= fit_end_idx:
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

        txt = ax.annotate(
            "\n".join(info_lines),
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(INFO_BOX_MARGIN_PT, -INFO_BOX_MARGIN_PT),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=6,  # paper-grade: 5-8 pt
            bbox=dict(
                boxstyle=f"round,pad={INFO_BOX_PAD_DEFAULT}",
                facecolor=INFO_BOX_FACE_COLOR,
                alpha=0.95,
                edgecolor="none",
            ),
            zorder=10,
        )
        # Add gradient shadow to info box
        if txt.get_bbox_patch() is not None:
            txt.get_bbox_patch().set_path_effects(get_info_box_gradient_shadow())

        fig.tight_layout(pad=0.3)
        fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def _well_to_rc(well: str) -> Optional[tuple[int, int]]:
    """Convert well label (e.g. 'A1') to (row_index, col_index). Row A=0..H=7, Col 1=0..7=6. Returns None if invalid or col not in 1-7."""
    m = _WELL_PATTERN.match(str(well).strip())
    if not m:
        return None
    row_char, col_str = m.group(1), m.group(2)
    row_idx = ord(row_char.upper()) - ord("A")
    col_val = int(col_str)
    if not (1 <= col_val <= 7):
        return None
    col_idx = col_val - 1
    if not (0 <= row_idx <= 7):
        return None
    return (row_idx, col_idx)


def write_plate_grid(run_plot_dir: Path, run_id: str) -> list[Path]:
    """
    Assemble per-well PNGs into plate grid images (paper-grade, English only).

    One output is generated per plate directory under run_plot_dir:
      - run_plot_dir / plate_grid__{run_id}__{plate_id}.png

    Grid layout: rows A--H (8 max), cols 1--7. If only A--D have data, output
    a 4x7 grid (no empty E,F,G,H rows). Uses existing PNGs only (no redraw).

    Parameters
    ----------
    run_plot_dir : Path
        Directory containing plate subdirs (e.g. plate1/) with well PNGs (A1.png, ...).
    run_id : str
        Run identifier for the output filename.

    Returns
    -------
    list[Path]
        Paths to the saved plate-grid PNGs.
    """
    run_plot_dir = Path(run_plot_dir)
    # Collect per plate: (row_idx, col_idx) -> path for wells in cols 1-7
    per_plate_grid: dict[str, dict[tuple[int, int], Path]] = {}
    for plate_path in sorted(run_plot_dir.iterdir()):
        if not plate_path.is_dir():
            continue
        plate_id = plate_path.name
        plate_grid: dict[tuple[int, int], Path] = {}
        for png_path in sorted(plate_path.glob("*.png")):
            if png_path.name.startswith("plate_grid"):
                continue
            well = png_path.stem
            rc = _well_to_rc(well)
            if rc is None:
                continue
            plate_grid[rc] = png_path
        if plate_grid:
            per_plate_grid[plate_id] = plate_grid

    if not per_plate_grid:
        return []

    # Clean stale grid files first
    expected = {f"plate_grid__{run_id}__{plate_id}.png" for plate_id in per_plate_grid}
    for stale in run_plot_dir.glob(f"plate_grid__{run_id}*.png"):
        if stale.name not in expected and stale.name != f"plate_grid__{run_id}.png":
            try:
                stale.unlink()
            except Exception:
                pass

    saved_paths: list[Path] = []
    grid_dpi = 600

    for plate_id in sorted(per_plate_grid.keys()):
        grid = per_plate_grid[plate_id]

        max_row = max(r for (r, c) in grid)
        n_rows = max_row + 1
        n_cols = 7

        # Get aspect ratio from first available image (preserve original aspect, no stretch)
        first_path = next(grid[k] for k in sorted(grid))
        try:
            with Image.open(first_path) as sample_img:
                w, h = sample_img.size
            cell_aspect = float(w) / float(h) if h > 0 else 1.0
        except Exception:
            cell_aspect = 3.5 / 2.6

        # Figure size so each cell has the same aspect as source images (no distortion)
        cell_height_in = 2.6
        cell_width_in = cell_height_in * cell_aspect
        fig_w = n_cols * cell_width_in
        fig_h = n_rows * cell_height_in

        with plt.rc_context(apply_paper_style()):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            for r in range(n_rows):
                for c in range(n_cols):
                    ax = axes[r, c]
                    ax.set_axis_off()
                    key = (r, c)
                    if key not in grid:
                        continue
                    try:
                        # Use Pillow->uint8 arrays to avoid float32 expansion (large memory spike).
                        with Image.open(grid[key]) as im:
                            img = np.asarray(im.convert("RGBA"), dtype=np.uint8)
                        # Preserve original aspect ratio (no stretch, no squeeze)
                        ax.imshow(img, aspect="equal", interpolation="none")
                        del img
                    except Exception:
                        pass
                    well_label = f"{chr(ord('A') + r)}{c + 1}"
                    ax.text(0.02, 0.98, well_label, transform=ax.transAxes, ha="left", va="top", fontsize=7, color="0.2")

            fig.tight_layout(pad=0.15)
            out_path = run_plot_dir / f"plate_grid__{run_id}__{plate_id}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, format="png", dpi=grid_dpi, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            # Ensure per-plate image buffers are released before next plate.
            import gc
            gc.collect()
            saved_paths.append(out_path)

    # Backward-compat legacy name only when exactly one plate exists.
    legacy_path = run_plot_dir / f"plate_grid__{run_id}.png"
    if len(saved_paths) == 1:
        try:
            shutil.copyfile(saved_paths[0], legacy_path)
        except Exception:
            pass
    elif legacy_path.exists():
        try:
            legacy_path.unlink()
        except Exception:
            pass

    return saved_paths
