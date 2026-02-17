#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def df_to_simple_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    cols = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals: List[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("NA")
                else:
                    vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def normalize_pid(pid: Any) -> str:
    return str(pid).strip().upper()


def classify_family(pid: str) -> str:
    p = normalize_pid(pid)
    if p == "GOX":
        return "reference"
    if p.startswith("GOX WITH"):
        return "solvent_control"
    if p == "PMPC":
        return "homopolymer_pmpc"
    if p == "PMTAC":
        return "homopolymer_pmtac"
    if p.startswith("PMBTA"):
        return "pmbta"
    if p == "BACKGROUND":
        return "background"
    return "other"


def is_screen_candidate(pid: str) -> bool:
    fam = classify_family(pid)
    return fam in {"pmbta", "homopolymer_pmpc", "homopolymer_pmtac"}


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([np.nan] * len(df))


def value_at_time_linear(t: np.ndarray, y: np.ndarray, at_time: float) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if t.size == 0:
        return np.nan
    if t.size == 1:
        return float(y[0])
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    return float(np.interp(float(at_time), t, y))


def load_run_table(run_fit_dir: Path) -> pd.DataFrame:
    run_id = run_fit_dir.parent.name
    summary_path = run_fit_dir / "summary_simple.csv"
    fog_path = run_fit_dir / f"fog_summary__{run_id}.csv"
    rates_path = run_fit_dir / "rates_selected.csv"
    t50_path = run_fit_dir / "t50" / "csv" / f"t50__{run_id}.csv"
    if not t50_path.is_file():
        t50_path = run_fit_dir / "t50" / f"t50__{run_id}.csv"

    if not summary_path.is_file() or not fog_path.is_file() or not rates_path.is_file():
        return pd.DataFrame()

    summary = pd.read_csv(summary_path)
    fog = pd.read_csv(fog_path)
    rates = pd.read_csv(rates_path)
    t50_df = pd.read_csv(t50_path) if t50_path.is_file() else pd.DataFrame()

    summary["pid_norm"] = summary["polymer_id"].map(normalize_pid)
    fog["pid_norm"] = fog["polymer_id"].map(normalize_pid)
    rates["pid_norm"] = rates["polymer_id"].map(normalize_pid)
    if not t50_df.empty and "polymer_id" in t50_df.columns:
        t50_df["pid_norm"] = t50_df["polymer_id"].map(normalize_pid)

    for col in ["heat_min", "abs_activity", "REA_percent"]:
        summary[col] = pd.to_numeric(_safe_series(summary, col), errors="coerce")
    for col in ["heat_min", "abs_activity", "r2", "n", "t_start", "t_end", "snr", "start_idx_used"]:
        rates[col] = pd.to_numeric(_safe_series(rates, col), errors="coerce")
    rates["status"] = _safe_series(rates, "status").astype(str).str.lower()

    for col in ["t50_min", "fog", "rea_at_20_percent", "abs_activity_at_0", "native_activity_rel_at_0"]:
        fog[col] = pd.to_numeric(_safe_series(fog, col), errors="coerce")
    for col in ["fit_r2", "t50_exp_min", "t50_linear_min"]:
        if not t50_df.empty:
            t50_df[col] = pd.to_numeric(_safe_series(t50_df, col), errors="coerce")

    all_pids = sorted(set(summary["pid_norm"].dropna().unique()).union(set(fog["pid_norm"].dropna().unique())))
    rows: List[Dict[str, Any]] = []
    for pid in all_pids:
        gsum = summary[summary["pid_norm"] == pid].copy()
        gfog = fog[fog["pid_norm"] == pid].copy()
        grates = rates[(rates["pid_norm"] == pid) & (rates["status"] == "ok")].copy()
        gt50 = t50_df[t50_df["pid_norm"] == pid].copy() if not t50_df.empty else pd.DataFrame()

        t = gsum["heat_min"].to_numpy(dtype=float) if not gsum.empty else np.array([], dtype=float)
        abs_y = gsum["abs_activity"].to_numpy(dtype=float) if not gsum.empty else np.array([], dtype=float)
        rea_y = gsum["REA_percent"].to_numpy(dtype=float) if not gsum.empty else np.array([], dtype=float)

        abs0_summary = value_at_time_linear(t, abs_y, 0.0) if t.size else np.nan
        rea20_summary = value_at_time_linear(t, rea_y, 20.0) if t.size else np.nan
        rea_auc = np.nan
        if t.size >= 2 and np.any(np.isfinite(rea_y)):
            mask = np.isfinite(t) & np.isfinite(rea_y)
            tt = t[mask]
            yy = rea_y[mask]
            if tt.size >= 2:
                order = np.argsort(tt)
                tt = tt[order]
                yy = yy[order]
                rea_auc = float(np.trapezoid(yy, tt))

        h0 = grates[np.isclose(grates["heat_min"], 0.0, atol=1e-9)].copy()
        abs0_rate_median = float(h0["abs_activity"].median()) if not h0.empty else np.nan
        abs0_rate_mean = float(h0["abs_activity"].mean()) if not h0.empty else np.nan
        fit_r2_h0_median = float(h0["r2"].median()) if not h0.empty else np.nan
        fit_tstart_h0_median = float(h0["t_start"].median()) if not h0.empty else np.nan
        fit_start_idx_h0_median = float(h0["start_idx_used"].median()) if not h0.empty else np.nan
        fit_n_h0_median = float(h0["n"].median()) if not h0.empty else np.nan

        abs0 = abs0_rate_median if np.isfinite(abs0_rate_median) else abs0_summary
        rea20 = rea20_summary
        t50 = pd.to_numeric(_safe_series(gfog, "t50_min"), errors="coerce")
        t50_val = float(t50.iloc[0]) if len(t50) else np.nan
        fog_val = pd.to_numeric(_safe_series(gfog, "fog"), errors="coerce")
        fog_val = float(fog_val.iloc[0]) if len(fog_val) else np.nan
        if not np.isfinite(rea20):
            rea20_fog = pd.to_numeric(_safe_series(gfog, "rea_at_20_percent"), errors="coerce")
            if len(rea20_fog):
                rea20 = float(rea20_fog.iloc[0])

        use_for_bo_series = _safe_series(gfog, "use_for_bo")
        use_for_bo = bool(use_for_bo_series.iloc[0]) if len(use_for_bo_series) else True
        fit_r2 = pd.to_numeric(_safe_series(gt50, "fit_r2"), errors="coerce")
        fit_r2 = float(fit_r2.iloc[0]) if len(fit_r2) else np.nan
        fit_model = str(_safe_series(gt50, "fit_model").iloc[0]) if len(gt50) and "fit_model" in gt50.columns else ""
        t50_exp = pd.to_numeric(_safe_series(gt50, "t50_exp_min"), errors="coerce")
        t50_exp = float(t50_exp.iloc[0]) if len(t50_exp) else np.nan
        t50_lin = pd.to_numeric(_safe_series(gt50, "t50_linear_min"), errors="coerce")
        t50_lin = float(t50_lin.iloc[0]) if len(t50_lin) else np.nan

        rows.append(
            {
                "run_id": run_id,
                "polymer_id": str(gsum["polymer_id"].iloc[0]) if not gsum.empty else str(gfog["polymer_id"].iloc[0]),
                "pid_norm": pid,
                "family": classify_family(pid),
                "is_screen_candidate": int(is_screen_candidate(pid)),
                "use_for_bo": int(bool(use_for_bo)),
                "abs0": abs0,
                "abs0_summary": abs0_summary,
                "abs0_rate_median": abs0_rate_median,
                "abs0_rate_mean": abs0_rate_mean,
                "rea20": rea20,
                "rea_auc_0_60": rea_auc,
                "t50_min": t50_val,
                "fog": fog_val,
                "fit_r2": fit_r2,
                "fit_model": fit_model,
                "t50_exp_min": t50_exp,
                "t50_linear_min": t50_lin,
                "fit_r2_h0_median": fit_r2_h0_median,
                "fit_tstart_h0_median": fit_tstart_h0_median,
                "fit_start_idx_h0_median": fit_start_idx_h0_median,
                "fit_n_h0_median": fit_n_h0_median,
                "n_summary_rows": int(len(gsum)),
                "n_rate_rows_h0": int(len(h0)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    ref_rows = out[out["pid_norm"] == "GOX"]
    ref_abs0 = float(ref_rows["abs0"].iloc[0]) if not ref_rows.empty and np.isfinite(ref_rows["abs0"].iloc[0]) else np.nan
    ref_t50 = float(ref_rows["t50_min"].iloc[0]) if not ref_rows.empty and np.isfinite(ref_rows["t50_min"].iloc[0]) else np.nan
    out["ref_abs0"] = ref_abs0
    out["ref_t50"] = ref_t50
    out["has_ref_abs0"] = int(np.isfinite(ref_abs0) and ref_abs0 > 0.0)
    out["has_ref_t50"] = int(np.isfinite(ref_t50) and ref_t50 > 0.0)
    out["native_rel"] = np.where(
        np.isfinite(out["abs0"]) & np.isfinite(ref_abs0) & (ref_abs0 > 0.0),
        out["abs0"] / ref_abs0,
        np.nan,
    )
    if "native_activity_rel_at_0" in fog.columns:
        fog_native = fog[["pid_norm", "native_activity_rel_at_0"]].copy()
        fog_native["native_activity_rel_at_0"] = pd.to_numeric(fog_native["native_activity_rel_at_0"], errors="coerce")
        out = out.merge(fog_native, on="pid_norm", how="left")
        out["native_rel"] = np.where(np.isfinite(out["native_activity_rel_at_0"]), out["native_activity_rel_at_0"], out["native_rel"])
        out = out.drop(columns=["native_activity_rel_at_0"])
    return out


def build_unified_table(processed_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for run_fit in sorted(processed_dir.glob("*/fit")):
        tab = load_run_table(run_fit)
        if not tab.empty:
            rows.append(tab)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def infer_run_date_token(run_id: Any) -> str:
    s = str(run_id).strip()
    if not s:
        return ""
    head = s.split("-")[0]
    if head.isdigit() and len(head) >= 6:
        return head[:6]
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 6:
        return digits[:6]
    return ""


def parse_theta_list(raw: str) -> List[float]:
    vals: List[float] = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        v = float(t)
        if v <= 0:
            continue
        vals.append(float(v))
    if not vals:
        vals = [0.60, 0.75]
    return sorted(set(vals))


def build_fit_bias_check(unified: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run, g in unified.groupby("run_id", sort=True):
        g_h0 = g[np.isfinite(g["abs0_rate_median"])].copy()
        if g_h0.empty:
            continue
        gox = g_h0[g_h0["pid_norm"] == "GOX"]
        non = g_h0[g_h0["pid_norm"] != "GOX"]
        rows.append(
            {
                "run_id": run,
                "gox_abs0": float(gox["abs0_rate_median"].median()) if not gox.empty else np.nan,
                "non_abs0_median": float(non["abs0_rate_median"].median()) if not non.empty else np.nan,
                "gox_tstart_h0_median": float(gox["fit_tstart_h0_median"].median()) if not gox.empty else np.nan,
                "non_tstart_h0_median": float(non["fit_tstart_h0_median"].median()) if not non.empty else np.nan,
                "gox_start_idx_h0_median": float(gox["fit_start_idx_h0_median"].median()) if not gox.empty else np.nan,
                "non_start_idx_h0_median": float(non["fit_start_idx_h0_median"].median()) if not non.empty else np.nan,
                "gox_r2_h0_median": float(gox["fit_r2_h0_median"].median()) if not gox.empty else np.nan,
                "non_r2_h0_median": float(non["fit_r2_h0_median"].median()) if not non.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rank_inversion_table(unified: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []
    for run, g in unified.groupby("run_id", sort=True):
        cand = g[(g["is_screen_candidate"] == 1) & np.isfinite(g["fog"]) & np.isfinite(g["abs0"])].copy()
        if cand.empty:
            continue
        cand["rank_abs_desc"] = cand["abs0"].rank(ascending=False, method="min")
        cand["rank_rea20_desc"] = cand["rea20"].rank(ascending=False, method="min")
        cand["rank_fog_desc"] = cand["fog"].rank(ascending=False, method="min")
        cand["delta_abs_to_fog"] = cand["rank_abs_desc"] - cand["rank_fog_desc"]
        cand["delta_abs_to_rea20"] = cand["rank_abs_desc"] - cand["rank_rea20_desc"]
        corr_abs_fog = cand["abs0"].corr(cand["fog"], method="spearman")
        corr_abs_rea20 = cand["abs0"].corr(cand["rea20"], method="spearman") if np.isfinite(cand["rea20"]).sum() >= 3 else np.nan
        for _, r in cand.iterrows():
            out_rows.append(
                {
                    "run_id": run,
                    "polymer_id": r["polymer_id"],
                    "family": r["family"],
                    "abs0": r["abs0"],
                    "rea20": r["rea20"],
                    "fog": r["fog"],
                    "native_rel": r["native_rel"],
                    "rank_abs_desc": r["rank_abs_desc"],
                    "rank_rea20_desc": r["rank_rea20_desc"],
                    "rank_fog_desc": r["rank_fog_desc"],
                    "delta_abs_to_fog": r["delta_abs_to_fog"],
                    "delta_abs_to_rea20": r["delta_abs_to_rea20"],
                    "spearman_abs_vs_fog_run": corr_abs_fog,
                    "spearman_abs_vs_rea20_run": corr_abs_rea20,
                }
            )
    return pd.DataFrame(out_rows)


def policy_specs() -> List[Dict[str, Any]]:
    return [
        {"policy_id": "fog_raw", "display": "FoG raw", "kind": "fog"},
        {"policy_id": "fog_adj_clip", "display": "FoG * clip(native,0,1)", "kind": "adj_clip"},
        {"policy_id": "cons_0.40", "display": "Constrained theta=0.40", "kind": "constrained", "theta": 0.40},
        {"policy_id": "cons_0.50", "display": "Constrained theta=0.50", "kind": "constrained", "theta": 0.50},
        {"policy_id": "cons_0.60", "display": "Constrained theta=0.60", "kind": "constrained", "theta": 0.60},
        {"policy_id": "cons_0.70", "display": "Constrained theta=0.70", "kind": "constrained", "theta": 0.70},
        {"policy_id": "cons_0.75", "display": "Constrained theta=0.75", "kind": "constrained", "theta": 0.75},
        {"policy_id": "cons_0.80", "display": "Constrained theta=0.80", "kind": "constrained", "theta": 0.80},
        {"policy_id": "cons_0.90", "display": "Constrained theta=0.90", "kind": "constrained", "theta": 0.90},
        {
            "policy_id": "cons_0.75_r2_0.85",
            "display": "Constrained theta=0.75 + fit_r2>=0.85",
            "kind": "constrained",
            "theta": 0.75,
            "min_fit_r2": 0.85,
        },
        {
            "policy_id": "cons_0.75_r2_0.90",
            "display": "Constrained theta=0.75 + fit_r2>=0.90",
            "kind": "constrained",
            "theta": 0.75,
            "min_fit_r2": 0.90,
        },
        {
            "policy_id": "cons_0.75_r2_0.90_rea20_20",
            "display": "Constrained theta=0.75 + fit_r2>=0.90 + REA20>=20",
            "kind": "constrained",
            "theta": 0.75,
            "min_fit_r2": 0.90,
            "min_rea20": 20.0,
        },
        {"policy_id": "logw_0.50", "display": "log(FoG)+0.5*log(native)", "kind": "log_weighted", "alpha": 0.50},
        {"policy_id": "logw_1.00", "display": "log(FoG)+1.0*log(native)", "kind": "log_weighted", "alpha": 1.00},
    ]


def compute_policy_score(kind: str, fog: float, native: float, theta: Optional[float], alpha: Optional[float]) -> float:
    if not np.isfinite(fog) or fog <= 0.0:
        return np.nan
    if kind == "fog":
        return float(fog)
    if kind == "adj_clip":
        if not np.isfinite(native):
            return np.nan
        return float(fog) * float(np.clip(native, 0.0, 1.0))
    if kind == "constrained":
        if not np.isfinite(native) or theta is None:
            return np.nan
        return float(fog) if native >= float(theta) else np.nan
    if kind == "log_weighted":
        if not np.isfinite(native) or native <= 0.0 or alpha is None:
            return np.nan
        return float(np.log(fog) + float(alpha) * np.log(native))
    return np.nan


def simulate_policies(unified: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    top_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    specs = policy_specs()

    for spec in specs:
        pid = str(spec["policy_id"])
        top_per_run: List[Dict[str, Any]] = []
        for run, g in unified.groupby("run_id", sort=True):
            cand = g[
                (g["is_screen_candidate"] == 1)
                & (g["use_for_bo"] == 1)
                & np.isfinite(g["fog"])
                & np.isfinite(g["native_rel"])
            ].copy()
            if cand.empty:
                continue
            min_fit_r2 = spec.get("min_fit_r2")
            if min_fit_r2 is not None:
                cand = cand[np.isfinite(cand["fit_r2"]) & (cand["fit_r2"] >= float(min_fit_r2))].copy()
            min_rea20 = spec.get("min_rea20")
            if min_rea20 is not None:
                cand = cand[np.isfinite(cand["rea20"]) & (cand["rea20"] >= float(min_rea20))].copy()
            if cand.empty:
                continue
            cand["score"] = cand.apply(
                lambda r: compute_policy_score(
                    kind=str(spec["kind"]),
                    fog=float(r["fog"]),
                    native=float(r["native_rel"]),
                    theta=spec.get("theta"),
                    alpha=spec.get("alpha"),
                ),
                axis=1,
            )
            cand = cand[np.isfinite(cand["score"])].copy()
            if cand.empty:
                continue
            cand = cand.sort_values(["score", "fog", "native_rel"], ascending=[False, False, False], kind="mergesort")
            top = cand.iloc[0]
            row = {
                "policy_id": pid,
                "policy_display": spec["display"],
                "run_id": run,
                "top_polymer_id": top["polymer_id"],
                "top_family": top["family"],
                "top_score": float(top["score"]),
                "top_fog": float(top["fog"]),
                "top_native_rel": float(top["native_rel"]),
            }
            if "theta" in spec:
                row["theta"] = float(spec["theta"])
            else:
                row["theta"] = np.nan
            if "alpha" in spec:
                row["alpha"] = float(spec["alpha"])
            else:
                row["alpha"] = np.nan
            row["min_fit_r2"] = float(spec["min_fit_r2"]) if "min_fit_r2" in spec else np.nan
            row["min_rea20"] = float(spec["min_rea20"]) if "min_rea20" in spec else np.nan
            top_per_run.append(row)
            top_rows.append(row)

        top_df = pd.DataFrame(top_per_run)
        if top_df.empty:
            summary_rows.append(
                {
                    "policy_id": pid,
                    "policy_display": spec["display"],
                    "n_runs_selected": 0,
                    "top_family_pmbta_count": 0,
                    "top_family_pmpc_count": 0,
                    "top_family_pmtac_count": 0,
                    "top_native_rel_median": np.nan,
                    "top_fog_median": np.nan,
                    "unique_top_polymer_count": 0,
                }
            )
            continue
        summary_rows.append(
            {
                "policy_id": pid,
                "policy_display": spec["display"],
                "n_runs_selected": int(len(top_df)),
                "top_family_pmbta_count": int((top_df["top_family"] == "pmbta").sum()),
                "top_family_pmpc_count": int((top_df["top_family"] == "homopolymer_pmpc").sum()),
                "top_family_pmtac_count": int((top_df["top_family"] == "homopolymer_pmtac").sum()),
                "top_native_rel_median": float(top_df["top_native_rel"].median()),
                "top_fog_median": float(top_df["top_fog"].median()),
                "unique_top_polymer_count": int(top_df["top_polymer_id"].nunique()),
            }
        )

    top_all = pd.DataFrame(top_rows)
    summary = pd.DataFrame(summary_rows).sort_values("policy_id")
    return top_all, summary


def runwise_homopolymer_vs_best_pmbta(unified: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run, g in unified.groupby("run_id", sort=True):
        cand = g[g["is_screen_candidate"] == 1].copy()
        if cand.empty:
            continue
        pmpc = cand[cand["family"] == "homopolymer_pmpc"]
        pmtac = cand[cand["family"] == "homopolymer_pmtac"]
        pmb = cand[cand["family"] == "pmbta"]
        pmpc_fog = float(pmpc["fog"].iloc[0]) if (not pmpc.empty and np.isfinite(pmpc["fog"].iloc[0])) else np.nan
        pmtac_fog = float(pmtac["fog"].iloc[0]) if (not pmtac.empty and np.isfinite(pmtac["fog"].iloc[0])) else np.nan
        best_pmb_id = ""
        best_pmb_fog = np.nan
        if not pmb.empty and np.isfinite(pmb["fog"]).any():
            pmb2 = pmb[np.isfinite(pmb["fog"])].sort_values("fog", ascending=False, kind="mergesort")
            best_pmb_id = str(pmb2.iloc[0]["polymer_id"])
            best_pmb_fog = float(pmb2.iloc[0]["fog"])
        rows.append(
            {
                "run_id": run,
                "pmpc_fog": pmpc_fog,
                "pmtac_fog": pmtac_fog,
                "best_pmbta_id": best_pmb_id,
                "best_pmbta_fog": best_pmb_fog,
                "best_pmbta_beats_pmpc": int(np.isfinite(best_pmb_fog) and np.isfinite(pmpc_fog) and best_pmb_fog > pmpc_fog),
                "best_pmbta_beats_pmtac": int(np.isfinite(best_pmb_fog) and np.isfinite(pmtac_fog) and best_pmb_fog > pmtac_fog),
            }
        )
    return pd.DataFrame(rows)


def _policy_id_from_theta(theta: float) -> str:
    return f"cons_{float(theta):.2f}"


def build_claim_gate_tables(
    *,
    unified: pd.DataFrame,
    top_policy: pd.DataFrame,
    main_theta: float = 0.70,
    sensitivity_thetas: Iterable[float] = (0.60, 0.75),
    beats_both_min_fraction: float = 0.80,
    top_family_stability_min_fraction: float = 0.70,
    claim_native_min: float = 0.85,
    claim_fog_min: float = 1.20,
    claim_effect_min_runs: int = 2,
    claim_effect_min_dates: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: List[Dict[str, Any]] = []
    sens_list = sorted(set(float(v) for v in sensitivity_thetas))
    main_policy_id = _policy_id_from_theta(main_theta)
    sens_policy_ids = [_policy_id_from_theta(v) for v in sens_list]

    top_lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not top_policy.empty:
        for _, r in top_policy.iterrows():
            k = (str(r.get("policy_id", "")), str(r.get("run_id", "")))
            top_lookup[k] = {
                "top_family": str(r.get("top_family", "")),
                "top_polymer_id": str(r.get("top_polymer_id", "")),
                "top_fog": pd.to_numeric(r.get("top_fog"), errors="coerce"),
                "top_native_rel": pd.to_numeric(r.get("top_native_rel"), errors="coerce"),
            }

    for run, g in unified.groupby("run_id", sort=True):
        g = g.copy()
        run_date = infer_run_date_token(run)
        has_ref_col = pd.to_numeric(_safe_series(g, "has_ref_t50"), errors="coerce")
        has_reference = int(np.nanmax(has_ref_col) > 0) if np.isfinite(has_ref_col).any() else 0

        pmpc = g[g["family"] == "homopolymer_pmpc"].copy()
        pmtac = g[g["family"] == "homopolymer_pmtac"].copy()
        pmb = g[g["family"] == "pmbta"].copy()

        pmpc_fog = np.nan
        if not pmpc.empty:
            vals = pd.to_numeric(_safe_series(pmpc, "fog"), errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals):
                pmpc_fog = float(vals.iloc[0])
        pmtac_fog = np.nan
        if not pmtac.empty:
            vals = pd.to_numeric(_safe_series(pmtac, "fog"), errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals):
                pmtac_fog = float(vals.iloc[0])

        best_pmbta_id = ""
        best_pmbta_fog = np.nan
        best_pmbta_native = np.nan
        best_pmbta_t50 = np.nan
        if not pmb.empty:
            pmb = pmb[np.isfinite(pd.to_numeric(_safe_series(pmb, "fog"), errors="coerce"))].copy()
            if not pmb.empty:
                pmb["fog"] = pd.to_numeric(_safe_series(pmb, "fog"), errors="coerce")
                pmb["native_rel"] = pd.to_numeric(_safe_series(pmb, "native_rel"), errors="coerce")
                pmb["t50_min"] = pd.to_numeric(_safe_series(pmb, "t50_min"), errors="coerce")
                pmb = pmb.sort_values(["fog", "native_rel"], ascending=[False, False], kind="mergesort")
                top = pmb.iloc[0]
                best_pmbta_id = str(top.get("polymer_id", ""))
                best_pmbta_fog = float(top["fog"]) if np.isfinite(top["fog"]) else np.nan
                best_pmbta_native = float(top["native_rel"]) if np.isfinite(top["native_rel"]) else np.nan
                best_pmbta_t50 = float(top["t50_min"]) if np.isfinite(top["t50_min"]) else np.nan

        beats_pmpc = int(np.isfinite(best_pmbta_fog) and np.isfinite(pmpc_fog) and best_pmbta_fog > pmpc_fog)
        beats_pmtac = int(np.isfinite(best_pmbta_fog) and np.isfinite(pmtac_fog) and best_pmbta_fog > pmtac_fog)
        beats_both = int(beats_pmpc == 1 and beats_pmtac == 1)
        has_controls = int(np.isfinite(pmpc_fog) and np.isfinite(pmtac_fog))
        include_for_claim = int(has_reference == 1 and has_controls == 1 and np.isfinite(best_pmbta_fog))
        effect_size_claim_pass = int(
            include_for_claim == 1
            and np.isfinite(best_pmbta_native)
            and np.isfinite(best_pmbta_fog)
            and best_pmbta_native >= float(claim_native_min)
            and best_pmbta_fog >= float(claim_fog_min)
        )

        row: Dict[str, Any] = {
            "run_id": run,
            "run_date_token": run_date,
            "has_reference": has_reference,
            "has_controls": has_controls,
            "include_for_claim": include_for_claim,
            "pmpc_fog": pmpc_fog,
            "pmtac_fog": pmtac_fog,
            "best_pmbta_id": best_pmbta_id,
            "best_pmbta_fog": best_pmbta_fog,
            "best_pmbta_native_rel": best_pmbta_native,
            "best_pmbta_t50_min": best_pmbta_t50,
            "best_pmbta_beats_pmpc": beats_pmpc,
            "best_pmbta_beats_pmtac": beats_pmtac,
            "best_pmbta_beats_both": beats_both,
            "effect_size_claim_pass": effect_size_claim_pass,
        }

        main_top = top_lookup.get((main_policy_id, str(run)), {})
        row["top_family_main"] = str(main_top.get("top_family", ""))
        row["top_polymer_main"] = str(main_top.get("top_polymer_id", ""))
        row["top_fog_main"] = float(main_top["top_fog"]) if np.isfinite(main_top.get("top_fog", np.nan)) else np.nan
        row["top_native_rel_main"] = (
            float(main_top["top_native_rel"]) if np.isfinite(main_top.get("top_native_rel", np.nan)) else np.nan
        )

        sens_families: List[str] = []
        sens_available = True
        for theta, policy_id in zip(sens_list, sens_policy_ids):
            sens_top = top_lookup.get((policy_id, str(run)), {})
            fam = str(sens_top.get("top_family", ""))
            row[f"top_family_theta_{theta:.2f}"] = fam
            row[f"top_polymer_theta_{theta:.2f}"] = str(sens_top.get("top_polymer_id", ""))
            sens_families.append(fam)
            if not fam:
                sens_available = False

        main_fam = str(row.get("top_family_main", ""))
        row["theta_sensitivity_evaluable"] = int(bool(main_fam) and sens_available)
        if row["theta_sensitivity_evaluable"] == 1:
            row["theta_family_consistent"] = int(all(f == main_fam for f in sens_families))
            row["theta_all_pmbta"] = int(main_fam == "pmbta" and all(f == "pmbta" for f in sens_families))
        else:
            row["theta_family_consistent"] = 0
            row["theta_all_pmbta"] = 0

        run_rows.append(row)

    runwise = pd.DataFrame(run_rows).sort_values("run_id").reset_index(drop=True)
    if runwise.empty:
        summary = pd.DataFrame(
            columns=[
                "criterion_id",
                "description",
                "observed_value",
                "threshold",
                "pass",
                "numerator",
                "denominator",
                "details",
            ]
        )
        return runwise, summary

    eligible = runwise[runwise["include_for_claim"] == 1].copy()
    n_eligible = int(len(eligible))

    c1_num = int(eligible["best_pmbta_beats_both"].sum()) if n_eligible > 0 else 0
    c1_den = n_eligible
    c1_frac = float(c1_num / c1_den) if c1_den > 0 else np.nan
    c1_pass = bool(c1_den > 0 and c1_frac >= float(beats_both_min_fraction))

    c2_dates = (
        int(eligible[eligible["best_pmbta_beats_both"] == 1]["run_date_token"].replace("", np.nan).dropna().nunique())
        if n_eligible > 0
        else 0
    )
    c2_pass = bool(c2_dates >= 2)

    effect_rows = eligible[eligible["effect_size_claim_pass"] == 1].copy()
    c3_run_count = int(len(effect_rows))
    c3_date_count = int(effect_rows["run_date_token"].replace("", np.nan).dropna().nunique()) if c3_run_count > 0 else 0
    c3_pass = bool(c3_run_count >= int(claim_effect_min_runs) and c3_date_count >= int(claim_effect_min_dates))

    sens_eval = eligible[eligible["theta_sensitivity_evaluable"] == 1].copy()
    c4_num = int(sens_eval["theta_family_consistent"].sum()) if len(sens_eval) > 0 else 0
    c4_den = int(len(sens_eval))
    c4_frac = float(c4_num / c4_den) if c4_den > 0 else np.nan
    c4_pass = bool(c4_den > 0 and c4_frac >= float(top_family_stability_min_fraction))

    n_missing_ref_runs = int((runwise["has_reference"] == 0).sum())
    c5_pass = bool(n_eligible > 0)

    summary_rows = [
        {
            "criterion_id": "C1",
            "description": "Main theta: best PMBTA beats both PMPC and PMTAC in >= target fraction.",
            "observed_value": c1_frac,
            "threshold": float(beats_both_min_fraction),
            "pass": int(c1_pass),
            "numerator": c1_num,
            "denominator": c1_den,
            "details": "",
        },
        {
            "criterion_id": "C2",
            "description": "Reproducibility across independent dates for C1-positive runs.",
            "observed_value": float(c2_dates),
            "threshold": 2.0,
            "pass": int(c2_pass),
            "numerator": c2_dates,
            "denominator": np.nan,
            "details": "",
        },
        {
            "criterion_id": "C3",
            "description": "Claim-effect runs satisfy native and FoG minima with run/date minimums.",
            "observed_value": float(c3_run_count),
            "threshold": float(claim_effect_min_runs),
            "pass": int(c3_pass),
            "numerator": c3_run_count,
            "denominator": n_eligible,
            "details": (
                f"native_min={float(claim_native_min):.3f}, fog_min={float(claim_fog_min):.3f}, "
                f"date_count={c3_date_count}, required_dates={int(claim_effect_min_dates)}"
            ),
        },
        {
            "criterion_id": "C4",
            "description": "Theta sensitivity stability (main vs sensitivity thetas) in >= target fraction.",
            "observed_value": c4_frac,
            "threshold": float(top_family_stability_min_fraction),
            "pass": int(c4_pass),
            "numerator": c4_num,
            "denominator": c4_den,
            "details": f"main_theta={float(main_theta):.2f}, sensitivity={','.join(f'{v:.2f}' for v in sens_list)}",
        },
        {
            "criterion_id": "C5",
            "description": "Reference eligibility enforced (runs without GOx reference excluded from claim set).",
            "observed_value": float(n_eligible),
            "threshold": 1.0,
            "pass": int(c5_pass),
            "numerator": n_eligible,
            "denominator": int(len(runwise)),
            "details": f"excluded_no_reference_runs={n_missing_ref_runs}",
        },
    ]
    overall_pass = bool(c1_pass and c2_pass and c3_pass and c4_pass and c5_pass)
    summary_rows.append(
        {
            "criterion_id": "OVERALL",
            "description": "All claim-gate criteria pass.",
            "observed_value": float(int(overall_pass)),
            "threshold": 1.0,
            "pass": int(overall_pass),
            "numerator": int(sum(int(r["pass"]) for r in summary_rows)),
            "denominator": int(len(summary_rows)),
            "details": "",
        }
    )
    summary = pd.DataFrame(summary_rows)
    return runwise, summary


def build_claim_gate_report(
    *,
    runwise: pd.DataFrame,
    summary: pd.DataFrame,
    out_md: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Claim Gate Report")
    lines.append("")
    if summary.empty:
        lines.append("- No claim-gate rows.")
        out_md.write_text("\n".join(lines), encoding="utf-8")
        return

    overall = summary[summary["criterion_id"] == "OVERALL"]
    overall_pass = int(overall["pass"].iloc[0]) if not overall.empty else 0
    lines.append(f"- Overall pass: **{overall_pass}**")
    lines.append(f"- Runs in runwise table: {len(runwise)}")
    lines.append(f"- Claim-eligible runs: {int((runwise['include_for_claim'] == 1).sum()) if not runwise.empty else 0}")
    lines.append("")

    show_cols = ["criterion_id", "pass", "observed_value", "threshold", "numerator", "denominator", "details"]
    lines.append("## Criteria Summary")
    lines.append(df_to_simple_markdown(summary[show_cols]))
    lines.append("")

    if not runwise.empty:
        run_cols = [
            "run_id",
            "run_date_token",
            "has_reference",
            "include_for_claim",
            "best_pmbta_id",
            "best_pmbta_fog",
            "best_pmbta_native_rel",
            "best_pmbta_beats_both",
            "effect_size_claim_pass",
            "top_family_main",
            "theta_family_consistent",
        ]
        available = [c for c in run_cols if c in runwise.columns]
        lines.append("## Runwise Snapshot")
        lines.append(df_to_simple_markdown(runwise[available].sort_values("run_id")))

    out_md.write_text("\n".join(lines), encoding="utf-8")


def plot_runwise_controls_vs_pmbta(comp: pd.DataFrame, out_png: Path) -> None:
    if comp.empty:
        return
    d = comp.sort_values("run_id").copy()
    x = np.arange(len(d), dtype=float)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7}):
        fig, ax = plt.subplots(figsize=(7.8, 2.8))
        ax.plot(x, d["pmpc_fog"], marker="o", linewidth=1.0, color="#1f77b4", label="PMPC")
        ax.plot(x, d["pmtac_fog"], marker="o", linewidth=1.0, color="#d62728", label="PMTAC")
        ax.plot(x, d["best_pmbta_fog"], marker="o", linewidth=1.2, color="#2ca02c", label="Best PMBTA")
        ax.axhline(1.0, color="0.5", linestyle=(0, (3, 2)), linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(d["run_id"].tolist(), rotation=20, ha="right")
        ax.set_ylabel("FoG")
        ax.set_xlabel("Run ID")
        ax.set_title("Run-wise FoG: PMPC / PMTAC / best PMBTA")
        ax.legend(frameon=True, fontsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def plot_policy_top_matrix(top_df: pd.DataFrame, out_png: Path) -> None:
    if top_df.empty:
        return
    runs = sorted(top_df["run_id"].unique().tolist())
    policies = sorted(top_df["policy_id"].unique().tolist())
    fam_to_int = {"pmbta": 2, "homopolymer_pmpc": 1, "homopolymer_pmtac": 0}
    mat = np.full((len(runs), len(policies)), np.nan, dtype=float)
    labels: Dict[Tuple[int, int], str] = {}
    for _, r in top_df.iterrows():
        i = runs.index(str(r["run_id"]))
        j = policies.index(str(r["policy_id"]))
        fam = str(r["top_family"])
        mat[i, j] = fam_to_int.get(fam, np.nan)
        labels[(i, j)] = str(r["top_polymer_id"])

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7}):
        fig, ax = plt.subplots(figsize=(8.2, 3.0))
        cmap = plt.get_cmap("RdYlGn")
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=2)
        ax.set_xticks(np.arange(len(policies)))
        ax.set_xticklabels(policies, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(runs)))
        ax.set_yticklabels(runs)
        ax.set_title("Top-1 family by policy and run (0=PMTAC, 1=PMPC, 2=PMBTA)")
        for i in range(len(runs)):
            for j in range(len(policies)):
                if (i, j) in labels:
                    txt = labels[(i, j)]
                    ax.text(j, i, txt.replace("PMBTA-", "B"), ha="center", va="center", fontsize=5.5, color="black")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def plot_abs_vs_fog_by_run(unified: pd.DataFrame, out_png: Path) -> None:
    data = unified[
        (unified["is_screen_candidate"] == 1)
        & np.isfinite(unified["abs0"])
        & np.isfinite(unified["fog"])
        & np.isfinite(unified["native_rel"])
    ].copy()
    if data.empty:
        return
    runs = sorted(data["run_id"].unique().tolist())
    n = len(runs)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fam_color = {
        "pmbta": "#2ca02c",
        "homopolymer_pmpc": "#1f77b4",
        "homopolymer_pmtac": "#d62728",
    }
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7}):
        fig, axes = plt.subplots(nrows, ncols, figsize=(8.8, 2.8 * nrows), squeeze=False)
        for idx, run in enumerate(runs):
            ax = axes[idx // ncols][idx % ncols]
            g = data[data["run_id"] == run].copy()
            for _, r in g.iterrows():
                c = fam_color.get(str(r["family"]), "#777777")
                marker = "o"
                ax.scatter(float(r["native_rel"]), float(r["fog"]), s=22, color=c, edgecolors="0.2", linewidths=0.3, marker=marker)
                if str(r["family"]) in {"homopolymer_pmtac", "homopolymer_pmpc"}:
                    ax.text(float(r["native_rel"]) + 0.01, float(r["fog"]), str(r["polymer_id"]), fontsize=5.5, color=c)
            ax.axhline(1.0, color="0.5", linestyle=(0, (3, 2)), linewidth=0.7)
            ax.set_title(run)
            ax.set_xlabel("Native activity at 0 min (rel to GOx)")
            ax.set_ylabel("FoG")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(left=0.0)
            ax.set_ylim(bottom=0.0)
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")
        fig.suptitle("Abs-to-FoG structure by run (screen candidates only)", y=0.995, fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def build_report(
    unified: pd.DataFrame,
    fit_bias: pd.DataFrame,
    inversion: pd.DataFrame,
    top_policy: pd.DataFrame,
    policy_summary: pd.DataFrame,
    comp: pd.DataFrame,
    out_md: Path,
) -> None:
    lines: List[str] = []
    lines.append("# GOx Trade-off Diagnostic Report")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Runs analyzed: {unified['run_id'].nunique()}")
    lines.append(f"- Polymer-run rows: {len(unified)}")
    lines.append(f"- Screen-candidate rows: {int((unified['is_screen_candidate'] == 1).sum())}")
    lines.append("")

    lines.append("## Key Findings")
    if not fit_bias.empty:
        gox_vs_non_tstart = fit_bias[["gox_tstart_h0_median", "non_tstart_h0_median"]].dropna()
        gox_vs_non_idx = fit_bias[["gox_start_idx_h0_median", "non_start_idx_h0_median"]].dropna()
        if not gox_vs_non_tstart.empty:
            lines.append(
                f"- Heat=0 fit window start median (GOx vs non-GOx): "
                f"{gox_vs_non_tstart['gox_tstart_h0_median'].median():.2f} s vs "
                f"{gox_vs_non_tstart['non_tstart_h0_median'].median():.2f} s."
            )
        if not gox_vs_non_idx.empty:
            lines.append(
                f"- Heat=0 start-index median (GOx vs non-GOx): "
                f"{gox_vs_non_idx['gox_start_idx_h0_median'].median():.2f} vs "
                f"{gox_vs_non_idx['non_start_idx_h0_median'].median():.2f}."
            )
    if not inversion.empty:
        corr_tbl = (
            inversion.groupby("run_id", as_index=False)["spearman_abs_vs_fog_run"]
            .first()
            .dropna()
            .sort_values("run_id")
        )
        if not corr_tbl.empty:
            neg = int((corr_tbl["spearman_abs_vs_fog_run"] < 0).sum())
            lines.append(
                f"- Runs with negative Spearman(abs0, FoG): {neg}/{len(corr_tbl)} "
                f"(rank inversion pressure present in multiple runs)."
            )
        big_up = inversion[inversion["delta_abs_to_fog"] >= 4]
        if not big_up.empty:
            lines.append(
                f"- Strong rank-up examples (abs rank -> FoG rank, delta>=4): {len(big_up)} cases."
            )
    if not comp.empty:
        valid = comp[np.isfinite(comp["best_pmbta_fog"]) & np.isfinite(comp["pmpc_fog"]) & np.isfinite(comp["pmtac_fog"])]
        if not valid.empty:
            lines.append(
                f"- Best PMBTA beats PMPC in {int(valid['best_pmbta_beats_pmpc'].sum())}/{len(valid)} runs; "
                f"beats PMTAC in {int(valid['best_pmbta_beats_pmtac'].sum())}/{len(valid)} runs."
            )
    lines.append("")

    lines.append("## Policy Simulation Summary")
    if policy_summary.empty:
        lines.append("- No policy simulation rows.")
    else:
        show_cols = [
            "policy_id",
            "n_runs_selected",
            "top_family_pmbta_count",
            "top_family_pmpc_count",
            "top_family_pmtac_count",
            "top_native_rel_median",
            "top_fog_median",
            "unique_top_polymer_count",
        ]
        lines.append(df_to_simple_markdown(policy_summary[show_cols]))
    lines.append("")

    lines.append("## Recommended Next Implementation Steps")
    lines.append("1. Switch default selection objective to constrained FoG with theta=0.70 (main) and keep raw FoG as diagnostic only.")
    lines.append("2. Make main ranking figure feasible-only bar plot; move trade-off scatter to supplementary.")
    lines.append("3. Add run-wise control panel (PMPC, PMTAC, best PMBTA) to communicate reproducibility and avoid overclaim.")
    lines.append("4. Track PMTAC as a flagged trace type: high FoG + low native_rel should be auto-annotated in ranking outputs.")
    lines.append("")

    lines.append("## Immediate Decision Support")
    if not top_policy.empty:
        base = top_policy[top_policy["policy_id"] == "fog_raw"][["run_id", "top_polymer_id"]].rename(columns={"top_polymer_id": "top_raw"})
        cons = top_policy[top_policy["policy_id"] == "cons_0.70"][["run_id", "top_polymer_id"]].rename(columns={"top_polymer_id": "top_cons_070"})
        joined = base.merge(cons, on="run_id", how="outer").sort_values("run_id")
        lines.append(df_to_simple_markdown(joined))

    out_md.write_text("\n".join(lines), encoding="utf-8")


def _safe_metric(policy_summary: pd.DataFrame, policy_id: str, col: str) -> Optional[float]:
    g = policy_summary[policy_summary["policy_id"] == policy_id]
    if g.empty or col not in g.columns:
        return None
    v = pd.to_numeric(g.iloc[0][col], errors="coerce")
    return float(v) if np.isfinite(v) else None


def build_roadmap(
    *,
    policy_summary: pd.DataFrame,
    top_policy: pd.DataFrame,
    comp: pd.DataFrame,
    out_md: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Next Tasks Roadmap (Decision-Critical)")
    lines.append("")
    lines.append("## What was learned from simulation")
    raw_pmtac = _safe_metric(policy_summary, "fog_raw", "top_family_pmtac_count")
    cons70_pmtac = _safe_metric(policy_summary, "cons_0.70", "top_family_pmtac_count")
    cons70_pmbta = _safe_metric(policy_summary, "cons_0.70", "top_family_pmbta_count")
    cons90_pmpc = _safe_metric(policy_summary, "cons_0.90", "top_family_pmpc_count")
    if raw_pmtac is not None:
        lines.append(f"- Raw FoG (`fog_raw`) selects PMTAC top in {int(raw_pmtac)} run(s) among evaluable runs.")
    if cons70_pmtac is not None and cons70_pmbta is not None:
        lines.append(
            f"- Constrained objective with `theta=0.70` reduces PMTAC top picks to {int(cons70_pmtac)} "
            f"and selects PMBTA top in {int(cons70_pmbta)} run(s)."
        )
    if cons90_pmpc is not None:
        lines.append(
            f"- Overly strict setting (`theta=0.90`) increases PMPC top picks ({int(cons90_pmpc)} run(s)) "
            "and can suppress stability gains."
        )
    if not comp.empty:
        valid = comp[np.isfinite(comp["best_pmbta_fog"]) & np.isfinite(comp["pmpc_fog"]) & np.isfinite(comp["pmtac_fog"])]
        if not valid.empty:
            lines.append(
                f"- Best PMBTA beats PMPC in {int(valid['best_pmbta_beats_pmpc'].sum())}/{len(valid)} runs; "
                f"beats PMTAC in {int(valid['best_pmbta_beats_pmtac'].sum())}/{len(valid)} runs."
            )
    lines.append("")

    lines.append("## Recommended default policy")
    lines.append("1. Objective for selection/BO: `fog_native_constrained` with `theta=0.70` (main).")
    lines.append("2. Keep diagnostics: `fog_raw` and `fog_native_constrained_tradeoff` (supplementary only).")
    lines.append("3. Keep quality flag (not hard exclusion by default): highlight rows with low `fit_r2`.")
    lines.append("")

    lines.append("## Expected behavior after implementation")
    lines.append("- Low-native PMTAC-like spikes should stop dominating top rank.")
    lines.append("- Top picks should concentrate on PMBTA candidates with both usable native activity and improved FoG.")
    lines.append("- Figure narrative shifts from rank inversion confusion to feasible high-stability selection.")
    lines.append("")

    lines.append("## Remaining risks to manage")
    lines.append("- Threshold sensitivity: top candidate may switch near the boundary (`theta` sensitivity must be reported).")
    lines.append("- Reference fallback bias: when same-run reference is missing, same-round substitution can shift native_0.")
    lines.append("- Fit uncertainty: low-SNR traces can inflate t50/FoG; keep fit-quality columns in review tables.")
    lines.append("- Mechanistic ambiguity: low native_0 may be inhibition, viscosity, or assay interference (not always true instability).")
    lines.append("- External validity: current conclusions are run-limited; avoid universal claims across all enzymes/polymer families.")
    lines.append("")

    lines.append("## Figure plan for manuscript")
    lines.append("1. Main: `fog_native_constrained_decision__{run}.png` (native gate + FoG ranking).")
    lines.append("2. Main support: `all_polymers__{run}.png` with right decision chart panel.")
    lines.append("3. Supplement: `fog_native_constrained_tradeoff__{run}.png` (trade-off scatter).")
    lines.append("4. Supplement: representative REA+t50 curves (GOx / PMPC / PMTAC / best PMBTA).")
    lines.append("")

    lines.append("## Immediate decision table (raw vs constrained)")
    if not top_policy.empty:
        base = top_policy[top_policy["policy_id"] == "fog_raw"][["run_id", "top_polymer_id"]].rename(columns={"top_polymer_id": "top_raw"})
        cons = top_policy[top_policy["policy_id"] == "cons_0.70"][["run_id", "top_polymer_id"]].rename(columns={"top_polymer_id": "top_cons_070"})
        joined = base.merge(cons, on="run_id", how="outer").sort_values("run_id")
        lines.append(df_to_simple_markdown(joined))

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep GOx ranking inversion diagnostics and objective-policy simulation.")
    ap.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/processed/analysis_20260212"))
    ap.add_argument("--claim_main_theta", type=float, default=0.70, help="Main theta for claim gate (default: 0.70).")
    ap.add_argument(
        "--claim_sensitivity_thetas",
        type=str,
        default="0.60,0.75",
        help="Comma-separated sensitivity thetas for claim gate (default: 0.60,0.75).",
    )
    ap.add_argument(
        "--claim_beats_both_min_fraction",
        type=float,
        default=0.80,
        help="Required run fraction where best PMBTA beats both PMPC and PMTAC (default: 0.80).",
    )
    ap.add_argument(
        "--claim_top_family_stability_min_fraction",
        type=float,
        default=0.70,
        help="Required sensitivity stability fraction (default: 0.70).",
    )
    ap.add_argument("--claim_native_min", type=float, default=0.85, help="Claim native_rel minimum (default: 0.85).")
    ap.add_argument("--claim_fog_min", type=float, default=1.20, help="Claim FoG minimum (default: 1.20).")
    ap.add_argument(
        "--claim_effect_min_runs",
        type=int,
        default=2,
        help="Minimum number of runs passing claim-effect thresholds (default: 2).",
    )
    ap.add_argument(
        "--claim_effect_min_dates",
        type=int,
        default=2,
        help="Minimum number of independent dates passing claim-effect thresholds (default: 2).",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    unified = build_unified_table(args.processed_dir)
    if unified.empty:
        raise SystemExit("No analyzable run data found.")

    fit_bias = build_fit_bias_check(unified)
    inversion = rank_inversion_table(unified)
    top_policy, policy_summary = simulate_policies(unified)
    comp = runwise_homopolymer_vs_best_pmbta(unified)
    claim_runwise, claim_summary = build_claim_gate_tables(
        unified=unified,
        top_policy=top_policy,
        main_theta=float(args.claim_main_theta),
        sensitivity_thetas=parse_theta_list(args.claim_sensitivity_thetas),
        beats_both_min_fraction=float(args.claim_beats_both_min_fraction),
        top_family_stability_min_fraction=float(args.claim_top_family_stability_min_fraction),
        claim_native_min=float(args.claim_native_min),
        claim_fog_min=float(args.claim_fog_min),
        claim_effect_min_runs=int(args.claim_effect_min_runs),
        claim_effect_min_dates=int(args.claim_effect_min_dates),
    )

    unified.to_csv(out_dir / "unified_polymer_run_metrics.csv", index=False)
    fit_bias.to_csv(out_dir / "fit_bias_check_h0.csv", index=False)
    inversion.to_csv(out_dir / "rank_inversion_abs_rea_fog.csv", index=False)
    top_policy.to_csv(out_dir / "policy_top1_by_run.csv", index=False)
    policy_summary.to_csv(out_dir / "policy_summary.csv", index=False)
    comp.to_csv(out_dir / "runwise_controls_vs_best_pmbta.csv", index=False)
    claim_runwise.to_csv(out_dir / "claim_gate_runwise.csv", index=False)
    claim_summary.to_csv(out_dir / "claim_gate_summary.csv", index=False)

    plot_runwise_controls_vs_pmbta(comp, out_dir / "runwise_controls_vs_best_pmbta_fog.png")
    plot_policy_top_matrix(top_policy, out_dir / "policy_top1_matrix.png")
    plot_abs_vs_fog_by_run(unified, out_dir / "abs_native_vs_fog_by_run.png")

    build_report(
        unified=unified,
        fit_bias=fit_bias,
        inversion=inversion,
        top_policy=top_policy,
        policy_summary=policy_summary,
        comp=comp,
        out_md=out_dir / "decision_report.md",
    )
    build_roadmap(
        policy_summary=policy_summary,
        top_policy=top_policy,
        comp=comp,
        out_md=out_dir / "next_tasks_roadmap.md",
    )
    build_claim_gate_report(
        runwise=claim_runwise,
        summary=claim_summary,
        out_md=out_dir / "claim_gate_report.md",
    )

    print(f"[done] wrote analysis outputs to: {out_dir}")


if __name__ == "__main__":
    main()
