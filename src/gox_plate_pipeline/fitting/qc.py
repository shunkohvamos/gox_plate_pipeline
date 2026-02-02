# src/gox_plate_pipeline/fitting/qc.py
"""
QC report generation for fit quality assessment.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .core import apply_paper_style, PAPER_FIGSIZE_SINGLE


def _normalize_exclude_reason(reason: object) -> str:
    """
    Normalize verbose FitSelectionError messages into coarse buckets.

    Buckets:
      - R² < r2_min
      - Δy < min_delta_y
      - Monotonicity / steps
      - SNR < min_snr
      - t_end > max_t_end
      - Slope < slope_min
      - No candidate windows
      - Other
    """
    s = "" if reason is None else str(reason)
    s = s.strip()
    if (not s) or (s.lower() in {"nan", "none"}):
        return "Other"

    low = s.lower()

    # Multi-failure message with per-filter rejection counts
    if "rejected counts out of" in low:

        def _m(pat: str) -> int:
            m = re.search(pat, s)
            return int(m.group(1)) if m else 0

        dy = _m(r"dy<[^:]*:\s*(\d+)")
        mono = _m(r"mono_frac<[^:]*:\s*(\d+)")
        down = _m(r"down_steps>[^:]*:\s*(\d+)")
        pos = _m(r"(?:pos_steps_eps|pos_steps)<[^:]*:\s*(\d+)")
        snr = _m(r"snr<[^:]*:\s*(\d+)")

        mono_total = int(mono + down + pos)

        candidates = [
            ("Δy < min_delta_y", dy),
            ("Monotonicity / steps", mono_total),
            ("SNR < min_snr", snr),
        ]
        best = max(candidates, key=lambda x: x[1])
        if best[1] > 0:
            return best[0]

        if ("dy<" in low) or ("min_delta_y" in low) or ("selected dy" in low):
            return "Δy < min_delta_y"
        if ("mono_frac" in low) or ("down_steps" in low) or ("pos_steps" in low):
            return "Monotonicity / steps"
        if ("snr" in low) or ("min_snr" in low):
            return "SNR < min_snr"
        return "Other"

    if ("max_t_end" in low) or ("ended after max_t_end" in low):
        return "t_end > max_t_end"

    if ("r2_min" in low) or ("no candidates met r2_min" in low) or ("selected r2" in low):
        return "R² < r2_min"

    if ("min_delta_y" in low) or ("dy<" in low) or ("selected dy" in low):
        return "Δy < min_delta_y"

    if ("mono_frac" in low) or ("down_steps" in low) or ("pos_steps" in low):
        return "Monotonicity / steps"

    if ("min_snr" in low) or ("snr<" in low) or ("selected snr" in low):
        return "SNR < min_snr"

    if ("slope_min" in low) or ("slope range" in low) or ("selected slope" in low):
        return "Slope < slope_min"

    if ("no candidate windows were generated" in low) or ("no candidates were generated" in low):
        return "No candidate windows"

    return "Other"


def write_fit_qc_report(
    selected: pd.DataFrame,
    out_dir: Path,
    max_t_end: Optional[float] = None,
    prefix: str = "fit_qc",
) -> Path:
    """
    Write a lightweight QC report for fit selection quality.

    Outputs (in out_dir):
      - {prefix}_summary_overall.csv
      - {prefix}_summary_by_plate.csv
      - {prefix}_summary_by_heat.csv
      - {prefix}_t_end_hist.png
      - {prefix}_slope_vs_t_end.png
      - {prefix}_select_method_counts.csv
      - {prefix}_select_method_bar.png
      - {prefix}_r2_hist.png
      - {prefix}_mono_frac_hist.png
      - {prefix}_snr_hist_log10.png
      - {prefix}_exclude_reason_norm_counts.csv
      - {prefix}_exclude_reason_norm_bar.png
      - {prefix}_report.md

    Returns path to the markdown report.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = selected.copy()

    if "status" not in df.columns or len(df) == 0:
        overall = pd.DataFrame(
            [{"n_total": 0, "n_ok": 0, "n_excluded": 0, "ok_rate": np.nan}]
        )
        overall.to_csv(out_dir / f"{prefix}_summary_overall.csv", index=False)
        report_md = out_dir / f"{prefix}_report.md"
        report_md.write_text(
            "# Fit QC Report\n\nNo wells were processed (empty or invalid selected).\n",
            encoding="utf-8",
        )
        return report_md

    # All plots use paper-grade style via rc_context below

    status = df["status"].astype(str)
    n_total = int(len(df))
    n_ok = int((status == "ok").sum())
    n_ex = int(n_total - n_ok)
    ok_rate = float(n_ok / n_total) if n_total > 0 else np.nan

    overall = pd.DataFrame(
        [
            {
                "n_total": n_total,
                "n_ok": n_ok,
                "n_excluded": n_ex,
                "ok_rate": ok_rate,
            }
        ]
    )
    overall_csv = out_dir / f"{prefix}_summary_overall.csv"
    overall.to_csv(overall_csv, index=False)

    by_plate_csv = None
    if "plate_id" in df.columns:
        tmp = df.copy()
        tmp["is_ok"] = (tmp["status"].astype(str) == "ok")
        by_plate = (
            tmp.groupby("plate_id", dropna=False)
            .agg(total=("status", "size"), ok=("is_ok", "sum"))
            .reset_index()
        )
        by_plate["excluded"] = by_plate["total"] - by_plate["ok"]
        by_plate["ok_rate"] = by_plate["ok"] / by_plate["total"]
        by_plate_csv = out_dir / f"{prefix}_summary_by_plate.csv"
        by_plate.to_csv(by_plate_csv, index=False)

    by_heat_csv = None
    if "heat_min" in df.columns:
        tmp = df.copy()
        tmp["is_ok"] = (tmp["status"].astype(str) == "ok")
        tmp["heat_min_num"] = pd.to_numeric(tmp["heat_min"], errors="coerce")
        by_heat = (
            tmp.groupby("heat_min_num", dropna=False)
            .agg(total=("status", "size"), ok=("is_ok", "sum"))
            .reset_index()
            .rename(columns={"heat_min_num": "heat_min"})
            .sort_values("heat_min")
        )
        by_heat["excluded"] = by_heat["total"] - by_heat["ok"]
        by_heat["ok_rate"] = by_heat["ok"] / by_heat["total"]
        by_heat_csv = out_dir / f"{prefix}_summary_by_heat.csv"
        by_heat.to_csv(by_heat_csv, index=False)

    ok = df[df["status"].astype(str) == "ok"].copy()

    if "t_end" not in ok.columns:
        ok["t_end"] = np.nan
    if "slope" not in ok.columns:
        ok["slope"] = np.nan

    ok["t_end"] = pd.to_numeric(ok["t_end"], errors="coerce")
    ok["slope"] = pd.to_numeric(ok["slope"], errors="coerce")

    t_end = ok["t_end"].to_numpy(dtype=float)
    slope = ok["slope"].to_numpy(dtype=float)
    mask = np.isfinite(t_end) & np.isfinite(slope)
    t_end_f = t_end[mask]
    slope_f = slope[mask]
    n_corr = int(t_end_f.size)

    pearson_r = np.nan
    spearman_rho = np.nan
    if n_corr >= 2:
        pearson_r = float(np.corrcoef(t_end_f, slope_f)[0, 1])
        rx = pd.Series(t_end_f).rank(method="average")
        ry = pd.Series(slope_f).rank(method="average")
        spearman_rho = float(rx.corr(ry))

    png_hist = out_dir / f"{prefix}_t_end_hist.png"
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE_SINGLE)
        if np.isfinite(t_end).sum() > 0:
            te = t_end[np.isfinite(t_end)]
            ax.hist(te, bins=30, color="#0072B2", edgecolor="white", linewidth=0.3)
            if max_t_end is not None and np.isfinite(float(max_t_end)):
                ax.axvline(float(max_t_end), linestyle=(0, (4, 2)), color="0.3", linewidth=0.7)
            ax.set_title("Selected t_end Distribution")
            ax.set_xlabel("t_end (s)")
            ax.set_ylabel("Count")
        else:
            ax.text(0.5, 0.5, "No OK fits", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Selected t_end Distribution")
            ax.set_xlabel("t_end (s)")
            ax.set_ylabel("Count")
        fig.tight_layout(pad=0.3)
        fig.savefig(png_hist, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    png_scatter = out_dir / f"{prefix}_slope_vs_t_end.png"
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE_SINGLE)
        if n_corr > 0:
            ax.scatter(t_end_f, slope_f, s=9, color="#0072B2", linewidths=0, alpha=0.8)
            ax.set_title("Slope vs t_end")
            ax.set_xlabel("t_end (s)")
            ax.set_ylabel("Slope (a.u./s)")
            txt = (
                f"N={n_corr}\n"
                f"Pearson r={pearson_r:.3f}\n"
                f"Spearman \u03c1={spearman_rho:.3f}"
            )
            ax.text(
                0.02,
                0.98,
                txt,
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.7", linewidth=0.4),
            )
        else:
            ax.text(0.5, 0.5, "No OK fits with finite t_end & slope", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Slope vs t_end")
            ax.set_xlabel("t_end (s)")
            ax.set_ylabel("Slope (a.u./s)")
        fig.tight_layout(pad=0.3)
        fig.savefig(png_scatter, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    method_counts_csv = None
    png_method_bar = None
    method_col = None
    if "select_method_used" in ok.columns:
        method_col = "select_method_used"
    elif "select_method" in ok.columns:
        method_col = "select_method"

    method_lines: list[str] = []
    force_ok_frac = np.nan
    force_all_frac = np.nan

    if method_col is not None:
        m = ok[method_col].fillna("NA").astype(str)
        if len(m) > 0:
            vc = m.value_counts(dropna=False)
            method_counts = vc.rename_axis("select_method_used").reset_index(name="count")
            method_counts["fraction"] = method_counts["count"] / float(method_counts["count"].sum())

            method_counts_csv = out_dir / f"{prefix}_select_method_counts.csv"
            method_counts.to_csv(method_counts_csv, index=False)

            png_method_bar = out_dir / f"{prefix}_select_method_bar.png"
            with plt.rc_context(apply_paper_style()):
                fig, ax = plt.subplots(figsize=PAPER_FIGSIZE_SINGLE)
                ax.bar(
                    method_counts["select_method_used"].astype(str),
                    method_counts["count"].to_numpy(dtype=int),
                    color="#0072B2",
                    edgecolor="white",
                    linewidth=0.3,
                )
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_title("select_method_used (OK only)")
                ax.set_xlabel("method")
                ax.set_ylabel("count")
                fig.tight_layout(pad=0.3)
                fig.savefig(png_method_bar, dpi=600, bbox_inches="tight", pad_inches=0.02)
                plt.close(fig)

            force_mask_ok = m.str.startswith("force_whole")
            force_ok_frac = float(force_mask_ok.mean())

            if (method_col in df.columns) and (n_total > 0):
                m_all = df[method_col].fillna("NA").astype(str)
                force_all_frac = float(m_all.str.startswith("force_whole").mean())

            for _, r in method_counts.iterrows():
                name = str(r["select_method_used"])
                cnt = int(r["count"])
                frac = float(r["fraction"]) * 100.0
                method_lines.append(f"- {name}: {cnt} ({frac:.1f}%)")

    png_r2 = out_dir / f"{prefix}_r2_hist.png"
    r2v = ok["r2"].to_numpy(dtype=float)
    r2v = r2v[np.isfinite(r2v)]

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE_SINGLE)
        if r2v.size > 0:
            ax.hist(r2v, bins=30, color="#0072B2", edgecolor="white", linewidth=0.3)
            ax.set_title("Selected R\u00b2 Distribution")
            ax.set_xlabel("R\u00b2")
            ax.set_ylabel("Count")
        else:
            ax.text(0.5, 0.5, "No OK fits", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Selected R\u00b2 Distribution")
            ax.set_xlabel("R\u00b2")
            ax.set_ylabel("Count")
        fig.tight_layout(pad=0.3)
        fig.savefig(png_r2, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    png_mono = out_dir / f"{prefix}_mono_frac_hist.png"
    mfv = ok["mono_frac"].to_numpy(dtype=float)
    mfv = mfv[np.isfinite(mfv)]
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE_SINGLE)
        if mfv.size > 0:
            ax.hist(mfv, bins=30, color="#0072B2", edgecolor="white", linewidth=0.3)
            ax.set_title("Selected mono_frac Distribution")
            ax.set_xlabel("mono_frac")
            ax.set_ylabel("Count")
        else:
            ax.text(0.5, 0.5, "No OK fits", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Selected mono_frac Distribution")
            ax.set_xlabel("mono_frac")
            ax.set_ylabel("Count")
        fig.tight_layout(pad=0.3)
        fig.savefig(png_mono, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    png_snr = out_dir / f"{prefix}_snr_hist_log10.png"
    snv = ok["snr"].to_numpy(dtype=float)
    snv = snv[np.isfinite(snv)]
    snv = snv[snv > 0.0]
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE_SINGLE)
        if snv.size > 0:
            sn_log = np.log10(snv + 1e-12)
            ax.hist(sn_log, bins=30, color="#0072B2", edgecolor="white", linewidth=0.3)
            ax.set_title("Selected SNR Distribution (log10)")
            ax.set_xlabel("log10(snr)")
            ax.set_ylabel("Count")
        else:
            ax.text(0.5, 0.5, "No OK fits", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Selected SNR Distribution (log10)")
            ax.set_xlabel("log10(snr)")
            ax.set_ylabel("Count")
        fig.tight_layout(pad=0.3)
        fig.savefig(png_snr, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    ex_counts_csv = None
    png_ex_bar = None
    ex_lines: list[str] = []

    ex = df[df["status"].astype(str) != "ok"].copy()
    if "exclude_reason" not in ex.columns:
        ex["exclude_reason"] = ""

    if len(ex) > 0:
        ex["exclude_reason_norm"] = ex["exclude_reason"].apply(_normalize_exclude_reason)

        vc = ex["exclude_reason_norm"].fillna("NA").astype(str).value_counts(dropna=False)
        ex_counts = vc.rename_axis("exclude_reason_norm").reset_index(name="count")
        ex_counts["fraction"] = ex_counts["count"] / float(ex_counts["count"].sum())

        ex_counts_csv = out_dir / f"{prefix}_exclude_reason_norm_counts.csv"
        ex_counts.to_csv(ex_counts_csv, index=False)

        png_ex_bar = out_dir / f"{prefix}_exclude_reason_norm_bar.png"
        with plt.rc_context(apply_paper_style()):
            fig, ax = plt.subplots(figsize=PAPER_FIGSIZE_SINGLE)
            ax.bar(
                ex_counts["exclude_reason_norm"].astype(str),
                ex_counts["count"].to_numpy(dtype=int),
                color="#D55E00",  # orange-red for excluded
                edgecolor="white",
                linewidth=0.3,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_title("exclude_reason_norm (EXCLUDED only)")
            ax.set_xlabel("reason")
            ax.set_ylabel("count")
            fig.tight_layout(pad=0.3)
            fig.savefig(png_ex_bar, dpi=600, bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)

        for _, r in ex_counts.iterrows():
            name = str(r["exclude_reason_norm"])
            cnt = int(r["count"])
            frac = float(r["fraction"]) * 100.0
            ex_lines.append(f"- {name}: {cnt} ({frac:.1f}%)")

    md_path = out_dir / f"{prefix}_report.md"

    te_ok = ok["t_end"].dropna()
    te_stats = {}
    if len(te_ok) > 0:
        qs = te_ok.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        te_stats = {f"q{int(k*100):02d}": float(v) for k, v in qs.items()}
        te_min = float(te_ok.min())
        te_max = float(te_ok.max())
    else:
        te_min = np.nan
        te_max = np.nan

    thresholds = [30.0, 60.0, 120.0, 240.0]
    if max_t_end is not None and np.isfinite(float(max_t_end)):
        mt = float(max_t_end)
        if mt not in thresholds:
            thresholds.append(mt)
    thresholds = sorted(set(thresholds))

    frac_lines = []
    if len(te_ok) > 0:
        for th in thresholds:
            frac = float((te_ok <= th).mean())
            frac_lines.append(f"- t_end ≤ {th:g} s : {frac*100:.1f}%")
    else:
        frac_lines.append("- t_end: no OK fits")

    def _q_lines(series: pd.Series, label: str) -> list[str]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if len(s) == 0:
            return [f"- {label}: no OK fits"]
        qs = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        out = [f"- {label} min/max: {float(s.min()):.4g} / {float(s.max()):.4g}"]
        for k in [0.1, 0.25, 0.5, 0.75, 0.9]:
            out.append(f"- {label} q{int(k*100):02d}: {float(qs[k]):.4g}")
        return out

    now_txt = str(pd.Timestamp.now())

    lines: list[str] = []
    lines.append("# Fit QC Report")
    lines.append("")
    lines.append(f"- Generated: {now_txt}")
    lines.append("")
    lines.append("## (a) OK / EXCLUDED")
    lines.append(f"- Total wells: {n_total}")
    lines.append(f"- OK: {n_ok}")
    lines.append(f"- EXCLUDED: {n_ex}")
    lines.append(f"- OK rate: {ok_rate*100:.1f}%")
    lines.append("")
    lines.append(f"- CSV: {overall_csv.name}")
    if by_plate_csv is not None:
        lines.append(f"- CSV (by plate): {by_plate_csv.name}")
    if by_heat_csv is not None:
        lines.append(f"- CSV (by heat): {by_heat_csv.name}")
    lines.append("")

    lines.append("## (b) Selected t_end distribution")
    if len(te_ok) > 0:
        lines.append(f"- t_end min/max: {te_min:.3g} / {te_max:.3g} s")
        for k in ["q10", "q25", "q50", "q75", "q90"]:
            if k in te_stats:
                lines.append(f"- {k}: {te_stats[k]:.3g} s")
    lines.append("")
    lines.extend(frac_lines)
    lines.append("")
    lines.append(f"![t_end hist]({png_hist.name})")
    lines.append("")

    lines.append("## (c) Slope vs t_end")
    lines.append(f"- N (finite): {n_corr}")
    lines.append(f"- Pearson r: {pearson_r:.4g}")
    lines.append(f"- Spearman ρ: {spearman_rho:.4g}")
    lines.append("")
    lines.append(f"![slope vs t_end]({png_scatter.name})")
    lines.append("")

    lines.append("## (d) select_method_used breakdown (OK only)")
    if method_col is None or (method_counts_csv is None) or (png_method_bar is None) or (len(method_lines) == 0):
        lines.append("- select_method_used: not available (column missing or no OK fits)")
    else:
        lines.append(f"- method column used: {method_col}")
        if np.isfinite(force_ok_frac):
            lines.append(f"- force_whole* fraction (among OK): {force_ok_frac*100:.1f}%")
        if np.isfinite(force_all_frac):
            lines.append(f"- force_whole* fraction (among ALL wells): {force_all_frac*100:.1f}%")
        lines.append("")
        lines.append(f"- CSV: {method_counts_csv.name}")
        lines.extend(method_lines)
        lines.append("")
        lines.append(f"![select_method_used]({png_method_bar.name})")
    lines.append("")

    lines.append("## (e) Distributions (OK only)")
    lines.append("### R²")
    lines.extend(_q_lines(ok["r2"], "R²"))
    lines.append("")
    lines.append(f"![r2 hist]({png_r2.name})")
    lines.append("")
    lines.append("### mono_frac")
    lines.extend(_q_lines(ok["mono_frac"], "mono_frac"))
    lines.append("")
    lines.append(f"![mono_frac hist]({png_mono.name})")
    lines.append("")
    lines.append("### snr")
    lines.extend(_q_lines(ok["snr"], "snr"))
    lines.append("")
    lines.append(f"![snr hist]({png_snr.name})")
    lines.append("")

    lines.append("## (f) Exclude reasons (EXCLUDED only)")
    if ex_counts_csv is None or png_ex_bar is None or len(ex_lines) == 0:
        lines.append("- excluded wells: 0")
    else:
        lines.append(f"- CSV: {ex_counts_csv.name}")
        lines.extend(ex_lines)
        lines.append("")
        lines.append(f"![exclude_reason_norm]({png_ex_bar.name})")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path
