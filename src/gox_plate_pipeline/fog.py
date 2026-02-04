# src/gox_plate_pipeline/fog.py
"""
FoG (Fold over GOx) summary for BO: t50_polymer / t50_bare_GOx.

- Per-run: fog_summary__{run_id}.csv with FoG = t50_polymer / t50_GOx_same_run (denominator = same run GOx).
- Round-averaged: build_round_averaged_fog() averages FoG by (round_id, polymer_id). If a round has no GOx
  in any run, raises ValueError.
- Plate-aware FoG: build_fog_plate_aware() uses denominator rule "same plate GOx → same round GOx".
  Per-plate t50 is computed from rates_with_rea (REA vs heat_min per plate); then FoG = t50_polymer / gox_t50
  with gox_t50 from same (run, plate) if present, else mean GOx t50 in that round.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# t50 unit used in outputs (documented for BO and figures)
T50_UNIT = "min"


@dataclass
class FogWarningInfo:
    """Warning information collected during FoG calculation."""
    outlier_gox: List[Dict[str, any]] = field(default_factory=list)  # List of outlier GOx t50 info
    missing_rates_files: List[Dict[str, str]] = field(default_factory=list)  # List of missing rates_with_rea.csv


def write_fog_warning_file(warning_info: FogWarningInfo, out_path: Path, exclude_outlier_gox: bool = False) -> None:
    """
    Write warning information to a markdown file.
    
    Args:
        warning_info: FogWarningInfo object containing warning details.
        out_path: Path to output warning file.
        exclude_outlier_gox: Whether outlier exclusion was enabled.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("# FoG計算時の警告情報")
    lines.append("")
    lines.append(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    if not warning_info.outlier_gox and not warning_info.missing_rates_files:
        lines.append("## 警告なし")
        lines.append("")
        lines.append("警告は検出されませんでした。")
    else:
        if warning_info.outlier_gox:
            lines.append("## ⚠️ 異常GOx t50値の検出")
            lines.append("")
            for i, outlier_info in enumerate(warning_info.outlier_gox, 1):
                lines.append(f"### {i}. Round {outlier_info['round_id']}")
                lines.append("")
                lines.append(f"- **GOx t50 中央値**: {outlier_info['median_gox_t50']:.3f} min")
                lines.append(f"- **異常値の閾値**: [{outlier_info['low_threshold']:.3f}, {outlier_info['high_threshold']:.3f}] min")
                lines.append("- **検出された異常値**:")
                for outlier in outlier_info['outliers']:
                    lines.append(f"  - {outlier['run_id']}/{outlier['plate_id']}: {outlier['gox_t50']:.3f} min")
                lines.append("")
                if outlier_info['excluded']:
                    lines.append("**処理**: 異常値はround平均GOx t50の計算から**除外されました**。")
                else:
                    lines.append("**処理**: 異常値はround平均GOx t50の計算に**含まれています**。")
                lines.append("")
        
        if warning_info.missing_rates_files:
            lines.append("## ⚠️ 見つからないrates_with_rea.csv")
            lines.append("")
            for i, missing_info in enumerate(warning_info.missing_rates_files, 1):
                lines.append(f"### {i}. Round {missing_info['round_id']}, Run {missing_info['run_id']}")
                lines.append("")
                lines.append(f"- **期待されるパス**: `{missing_info['expected_path']}`")
                lines.append("- **影響**: このrunはFoG計算からスキップされます。")
                lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("## 「FoG（同一プレート→同一ラウンド）計算（異常GOx除外）」を実行した場合")
        lines.append("")
        lines.append("`--exclude_outlier_gox`フラグを有効にして実行すると、以下の変更が行われます：")
        lines.append("")
        lines.append("### 変更点")
        lines.append("")
        lines.append("1. **異常GOx t50値の除外**:")
        lines.append("   - 検出された異常GOx t50値がround平均GOx t50の計算から除外されます")
        lines.append("   - これにより、round平均GOx t50がより安定した値になります")
        lines.append("")
        lines.append("2. **FoG値への影響**:")
        lines.append("   - 異常GOx t50値を持つplateのポリマー:")
        lines.append("     - そのplateにGOxがある場合: `same_plate`のGOxを使うため、**影響なし**")
        lines.append("     - そのplateにGOxがない場合: `same_round`のGOxを使うため、**round平均GOx t50の変更の影響を受ける**")
        lines.append("   - 他のplateのポリマー:")
        lines.append("     - `same_round`のGOxを使う場合: **round平均GOx t50の変更の影響を受ける**")
        lines.append("     - `same_plate`のGOxを使う場合: **影響なし**")
        lines.append("")
        lines.append("3. **出力ファイルの変更**:")
        lines.append("   - `fog_plate_aware.csv`: 各ポリマーのFoG値が変更される可能性があります")
        lines.append("   - `fog_plate_aware_round_averaged.csv`: Round平均FoG値が変更される可能性があります")
        lines.append("")
        lines.append("### 実行方法")
        lines.append("")
        lines.append("VS Codeの「実行とデバッグ」パネルから、以下の設定を選択してください：")
        lines.append("")
        lines.append("- **「FoG（同一プレート→同一ラウンド）計算（異常GOx除外）」**")
        lines.append("")
        lines.append("または、コマンドラインから：")
        lines.append("")
        lines.append("```bash")
        lines.append("python scripts/build_fog_plate_aware.py \\")
        lines.append("  --run_round_map meta/bo_run_round_map.tsv \\")
        lines.append("  --processed_dir data/processed \\")
        lines.append("  --out_dir data/processed \\")
        lines.append("  --exclude_outlier_gox")
        lines.append("```")
        lines.append("")
        lines.append("### 注意事項")
        lines.append("")
        lines.append("- 異常値を除外すると、round平均GOx t50が変わるため、FoG値も変わります")
        lines.append("- 除外するかどうかは、データの品質と研究の目的に応じて判断してください")
        lines.append("- 異常値が多数ある場合、除外によりround平均GOx t50が大きく変わる可能性があります")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_fog_summary(
    t50_path: Path,
    run_id: str,
    *,
    manifest_path: Optional[Path] = None,
    input_tidy_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build FoG summary from t50 CSV: FoG = t50_polymer / t50_bare_GOx (same run only).

    - t50 unit: minutes (T50_UNIT).
    - t50_censored: 1 if t50 is missing or REA plateau > 50% (did not reach 50%); 0 otherwise.
    - gox_t50_same_run: bare GOx t50 in same run (min). NaN if no GOx in run.
    - fog: t50 / gox_t50_same_run. NaN if no GOx in run or t50 missing.
    - log_fog: log(fog) for BO; NaN when fog <= 0 or missing (handle consistently).
    - fog_missing_reason: e.g. "no_bare_gox_in_run" when GOx absent in run; empty otherwise.
    - Lineage: run_id, input_t50_file, input_tidy (from manifest or passed).
    """
    t50_path = Path(t50_path)
    run_id = str(run_id).strip()
    df = pd.read_csv(t50_path)

    if "run_id" not in df.columns:
        df["run_id"] = run_id
    if "polymer_id" not in df.columns:
        raise ValueError(f"t50 CSV must have polymer_id, got: {list(df.columns)}")

    # Prefer t50_exp_min, fallback to t50_linear_min (unit: min)
    t50_exp = pd.to_numeric(df.get("t50_exp_min", np.nan), errors="coerce")
    t50_lin = pd.to_numeric(df.get("t50_linear_min", np.nan), errors="coerce")
    df["t50_min"] = np.where(np.isfinite(t50_exp), t50_exp, t50_lin)

    # Censored: no t50 or plateau > 50% (REA did not reach 50%)
    fit_plateau = pd.to_numeric(df.get("fit_plateau", np.nan), errors="coerce")
    fit_model = df.get("fit_model", "").astype(str)
    no_t50 = ~np.isfinite(df["t50_min"])
    plateau_above_50 = (fit_model.str.contains("exp_plateau", na=False)) & (fit_plateau > 50.0)
    df["t50_censored"] = (no_t50 | plateau_above_50).astype(int)

    # Bare GOx t50 in same run (this CSV is single-run, so one value)
    gox = df[df["polymer_id"].astype(str).str.strip().str.upper() == "GOX"]
    if gox.empty:
        gox_t50_same_run = np.nan
        fog_missing_reason_default = "no_bare_gox_in_run"
    else:
        gox_t50_same_run = float(gox["t50_min"].iloc[0]) if np.isfinite(gox["t50_min"].iloc[0]) else np.nan
        fog_missing_reason_default = "" if np.isfinite(gox_t50_same_run) else "no_bare_gox_in_run"

    df["gox_t50_same_run_min"] = gox_t50_same_run
    df["fog_missing_reason"] = fog_missing_reason_default

    # FoG and log FoG
    with np.errstate(divide="ignore", invalid="ignore"):
        fog = np.where(
            np.isfinite(df["t50_min"]) & np.isfinite(df["gox_t50_same_run_min"]) & (df["gox_t50_same_run_min"] > 0),
            df["t50_min"] / df["gox_t50_same_run_min"],
            np.nan,
        )
    df["fog"] = fog
    # log_fog: only when fog > 0 and finite; else NaN (consistent for BO)
    log_fog = np.full_like(fog, np.nan, dtype=float)
    ok = np.isfinite(fog) & (fog > 0)
    log_fog[ok] = np.log(fog[ok])
    df["log_fog"] = log_fog

    # Lineage columns
    df["input_t50_file"] = t50_path.name
    if input_tidy_path is not None:
        df["input_tidy"] = str(Path(input_tidy_path).resolve())
    elif manifest_path is not None and manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            inputs = payload.get("manifest", {}).get("input_files", [])
            first_path = inputs[0].get("path", "") if inputs else ""
            df["input_tidy"] = first_path
        except Exception:
            df["input_tidy"] = ""
    else:
        df["input_tidy"] = ""

    # Output columns (order and names for BO and lineage)
    out_cols = [
        "run_id",
        "polymer_id",
        "t50_min",
        "t50_censored",
        "gox_t50_same_run_min",
        "fog",
        "log_fog",
        "fog_missing_reason",
        "n_points",
        "input_t50_file",
        "input_tidy",
    ]
    # Keep only columns that exist
    available = [c for c in out_cols if c in df.columns]
    return df[available].copy()


def write_fog_summary_csv(
    fog_df: pd.DataFrame,
    out_path: Path,
    *,
    t50_unit: str = T50_UNIT,
) -> None:
    """Write FoG summary CSV. t50 and gox_t50 are in minutes (t50_unit)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fog_df.to_csv(out_path, index=False)
    # Document unit in a comment line would require custom write; unit is in T50_UNIT and README/doc.


def build_round_averaged_fog(
    run_round_map: Dict[str, str],
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Build round-averaged FoG from per-run fog_summary CSVs.

    - FoG is currently computed per run (same-run GOx as denominator). This function
      averages FoG by (round_id, polymer_id) across all runs in that round.
    - GOx row is excluded from the output. Only rows with finite fog and fog > 0 are averaged.
    - If a round has no run with GOx (all runs have gox_t50_same_run_min NaN), raises ValueError.
    - Output columns: round_id, polymer_id, mean_fog, mean_log_fog, n_observations, run_ids.
    """
    processed_dir = Path(processed_dir)
    # round_id -> list of run_ids
    round_to_runs: Dict[str, List[str]] = {}
    for rid, oid in run_round_map.items():
        rid, oid = str(rid).strip(), str(oid).strip()
        if not oid:
            continue
        round_to_runs.setdefault(oid, []).append(rid)

    if not round_to_runs:
        return pd.DataFrame(columns=["round_id", "polymer_id", "mean_fog", "mean_log_fog", "n_observations", "run_ids"])

    rows: List[dict] = []
    for round_id, run_ids in sorted(round_to_runs.items()):
        run_ids = sorted(run_ids)
        # Load fog_summary for each run in this round
        dfs: List[pd.DataFrame] = []
        round_has_gox = False
        for rid in run_ids:
            path = processed_dir / rid / "fit" / f"fog_summary__{rid}.csv"
            if not path.is_file():
                continue
            df = pd.read_csv(path)
            df["run_id"] = df.get("run_id", rid)
            df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
            gox_t50 = df.get("gox_t50_same_run_min", pd.Series(dtype=float))
            if gox_t50.notna().any() and np.isfinite(gox_t50).any():
                round_has_gox = True
            dfs.append(df)

        if not dfs:
            raise ValueError(f"Round {round_id!r} has no fog_summary CSV under processed_dir for runs {run_ids}.")
        if not round_has_gox:
            raise ValueError(
                f"Round {round_id!r} has no GOx in any run (all runs have no bare GOx). "
                "FoG cannot be computed for this round."
            )

        combined = pd.concat(dfs, ignore_index=True)
        # Exclude GOx row
        combined = combined[combined["polymer_id"].str.upper() != "GOX"].copy()
        # Only valid fog
        fog = pd.to_numeric(combined.get("fog", np.nan), errors="coerce")
        log_fog = pd.to_numeric(combined.get("log_fog", np.nan), errors="coerce")
        valid = np.isfinite(fog) & (fog > 0)
        combined = combined.loc[valid].copy()
        combined["_fog"] = fog[valid].values
        combined["_log_fog"] = log_fog[valid].values

        for polymer_id, g in combined.groupby("polymer_id", sort=False):
            pid = str(polymer_id).strip()
            mean_fog = float(g["_fog"].mean())
            mean_log_fog = float(g["_log_fog"].mean())
            n_obs = int(len(g))
            run_list = sorted(g["run_id"].astype(str).unique().tolist())
            rows.append({
                "round_id": round_id,
                "polymer_id": pid,
                "mean_fog": mean_fog,
                "mean_log_fog": mean_log_fog,
                "n_observations": n_obs,
                "run_ids": ",".join(run_list),
            })

    return pd.DataFrame(rows)


def build_round_gox_traceability(
    run_round_map: Dict[str, str],
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Build per-round GOx traceability: abs_activity (and REA_percent) by heat_min for each run,
    listing all pre-averaged well-level values so which GOx was used for that round is auditable.

    - Reads rates_with_rea.csv for each run in each round; keeps only rows with polymer_id == GOx.
    - Output columns: round_id, run_id, heat_min, plate_id, well, abs_activity, REA_percent.
    - One row per (round_id, run_id, heat_min, well); no averaging (all raw values for that run).
    """
    processed_dir = Path(processed_dir)
    round_to_runs: Dict[str, List[str]] = {}
    for rid, oid in run_round_map.items():
        rid, oid = str(rid).strip(), str(oid).strip()
        if not oid:
            continue
        round_to_runs.setdefault(oid, []).append(rid)

    rows: List[dict] = []
    for round_id, run_ids in sorted(round_to_runs.items()):
        run_ids = sorted(run_ids)
        for run_id in run_ids:
            path = processed_dir / run_id / "fit" / "rates_with_rea.csv"
            if not path.is_file():
                continue
            df = pd.read_csv(path)
            df["polymer_id"] = df.get("polymer_id", pd.Series(dtype=str)).astype(str).str.strip()
            gox = df[df["polymer_id"].str.upper() == "GOX"].copy()
            if gox.empty:
                continue
            for _, row in gox.iterrows():
                rows.append({
                    "round_id": round_id,
                    "run_id": run_id,
                    "heat_min": pd.to_numeric(row.get("heat_min", np.nan), errors="coerce"),
                    "plate_id": str(row.get("plate_id", "")),
                    "well": str(row.get("well", "")),
                    "abs_activity": pd.to_numeric(row.get("abs_activity", np.nan), errors="coerce"),
                    "REA_percent": pd.to_numeric(row.get("REA_percent", np.nan), errors="coerce"),
                })
    out_cols = ["round_id", "run_id", "heat_min", "plate_id", "well", "abs_activity", "REA_percent"]
    return pd.DataFrame(rows, columns=out_cols)


def compute_per_plate_t50_from_rates(rates_df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """
    Compute t50 per (run_id, plate_id, polymer_id) from rates_with_rea-style DataFrame.

    - Uses only rows with status == 'ok' and finite REA_percent.
    - Aggregates by (plate_id, polymer_id, heat_min) -> mean REA_percent, then fits t50 per (plate_id, polymer_id).
    - Prefers exp/exp_plateau t50; fallback t50_linear. Output: run_id, plate_id, polymer_id, t50_min, n_points.
    """
    from gox_plate_pipeline.polymer_timeseries import fit_exponential_decay, t50_linear  # avoid circular import

    df = rates_df.copy()
    df["polymer_id"] = df.get("polymer_id", pd.Series(dtype=str)).astype(str).str.strip()
    df["plate_id"] = df.get("plate_id", pd.Series(dtype=str)).astype(str)
    df["heat_min"] = pd.to_numeric(df.get("heat_min", np.nan), errors="coerce")
    df["REA_percent"] = pd.to_numeric(df.get("REA_percent", np.nan), errors="coerce")
    df["status"] = df.get("status", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    ok = df[(df["status"] == "ok") & np.isfinite(df["REA_percent"]) & (df["REA_percent"] > 0)].copy()
    if ok.empty:
        return pd.DataFrame(columns=["run_id", "plate_id", "polymer_id", "t50_min", "n_points", "fit_model"])

    # Aggregate by (plate_id, polymer_id, heat_min) -> mean REA
    agg = ok.groupby(["plate_id", "polymer_id", "heat_min"], dropna=False).agg(
        REA_percent=("REA_percent", "mean"),
    ).reset_index()

    rows: List[dict] = []
    for (plate_id, polymer_id), g in agg.groupby(["plate_id", "polymer_id"], dropna=False):
        g = g.sort_values("heat_min")
        t = g["heat_min"].to_numpy(dtype=float)
        y = g["REA_percent"].to_numpy(dtype=float)
        n_pts = int(len(g))
        if t.size < 3 or np.unique(t).size < 2:
            continue
        y0_init = float(np.nanmax(y)) if np.any(np.isfinite(y)) else None
        fit = fit_exponential_decay(t, y, y0=y0_init, min_points=3)
        t50_exp = float(fit.t50) if (fit is not None and fit.t50 is not None and np.isfinite(fit.t50)) else np.nan
        y0_lin = float(y0_init) if y0_init is not None and np.isfinite(y0_init) else float(np.nanmax(y))
        t50_lin_val = t50_linear(t, y, y0=y0_lin, target_frac=0.5)
        t50_lin_f = float(t50_lin_val) if t50_lin_val is not None and np.isfinite(t50_lin_val) else np.nan
        t50_min = t50_exp if np.isfinite(t50_exp) and t50_exp > 0 else t50_lin_f
        if not np.isfinite(t50_min) or t50_min <= 0:
            continue
        fit_model = "exp" if (fit is not None and fit.model == "exp") else ("exp_plateau" if (fit is not None and fit.model == "exp_plateau") else "linear_only")
        rows.append({
            "run_id": run_id,
            "plate_id": str(plate_id),
            "polymer_id": str(polymer_id).strip(),
            "t50_min": t50_min,
            "n_points": n_pts,
            "fit_model": fit_model,
        })
    return pd.DataFrame(rows)


def build_fog_plate_aware(
    run_round_map: Dict[str, str],
    processed_dir: Path,
    *,
    exclude_outlier_gox: bool = False,
    gox_outlier_low_threshold: float = 0.33,
    gox_outlier_high_threshold: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FogWarningInfo]:
    """
    Build FoG with denominator rule: same plate GOx → same round GOx.

    - Per-plate t50 is computed from rates_with_rea per run (REA vs heat_min per plate).
    - For each (run_id, plate_id, polymer_id): gox_t50 = GOx t50 from same (run_id, plate_id) if present,
      else mean of GOx t50 across all (run_id, plate_id) in that round. FoG = t50_polymer / gox_t50.
    - Round average GOx t50 calculation: All (run, plate) GOx t50 values are averaged with equal weight
      (run-level weighting is not applied; each plate contributes equally).
    - Outlier detection: GOx t50 values that are < median * gox_outlier_low_threshold or
      > median * gox_outlier_high_threshold are detected and warned. If exclude_outlier_gox=True,
      outliers are excluded from round average GOx t50 calculation.
    - If a round has no GOx in any (run, plate), raises ValueError.
    - Returns (per_row_df, round_averaged_df, gox_traceability_df, warning_info).
      per_row_df columns: run_id, plate_id, polymer_id, t50_min, gox_t50_used_min, denominator_source, fog, log_fog.
      warning_info: FogWarningInfo object containing warning details.

    Args:
        exclude_outlier_gox: If True, exclude outlier GOx t50 values from round average calculation.
        gox_outlier_low_threshold: Lower threshold multiplier for outlier detection (default: 0.33).
        gox_outlier_high_threshold: Upper threshold multiplier for outlier detection (default: 3.0).
    """
    processed_dir = Path(processed_dir)
    warning_info = FogWarningInfo()
    round_to_runs: Dict[str, List[str]] = {}
    for rid, oid in run_round_map.items():
        rid, oid = str(rid).strip(), str(oid).strip()
        if not oid:
            continue
        round_to_runs.setdefault(oid, []).append(rid)

    if not round_to_runs:
        empty = pd.DataFrame(columns=[
            "round_id", "run_id", "plate_id", "polymer_id", "t50_min", "gox_t50_used_min", "denominator_source", "fog", "log_fog"
        ])
        return empty, pd.DataFrame(columns=["round_id", "polymer_id", "mean_fog", "mean_log_fog", "n_observations", "run_ids"]), empty, FogWarningInfo()

    # Collect per-plate t50 for all runs in all rounds
    run_plate_t50: Dict[str, pd.DataFrame] = {}  # run_id -> DataFrame run_id, plate_id, polymer_id, t50_min
    missing_rates_files: List[tuple[str, str]] = []  # (round_id, run_id) pairs
    for round_id, run_ids in round_to_runs.items():
        for run_id in run_ids:
            if run_id in run_plate_t50:
                continue
            path = processed_dir / run_id / "fit" / "rates_with_rea.csv"
            if not path.is_file():
                missing_rates_files.append((round_id, run_id))
                warning_info.missing_rates_files.append({
                    "round_id": round_id,
                    "run_id": run_id,
                    "expected_path": str(path),
                })
                import warnings
                import sys
                # Print a more visible warning message
                print("\n" + "=" * 80, file=sys.stderr)
                print("⚠️  警告: rates_with_rea.csv が見つかりません", file=sys.stderr)
                print("=" * 80, file=sys.stderr)
                print(f"Round: {round_id}", file=sys.stderr)
                print(f"Run ID: {run_id}", file=sys.stderr)
                print(f"期待されるパス: {path}", file=sys.stderr)
                print("→ このrunはFoG計算からスキップされます。", file=sys.stderr)
                print("=" * 80 + "\n", file=sys.stderr)
                # Also issue standard warning for programmatic access
                warnings.warn(
                    f"Round {round_id!r}, run {run_id!r}: rates_with_rea.csv not found at {path}. "
                    "This run will be skipped in FoG calculation.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            df = pd.read_csv(path)
            run_plate_t50[run_id] = compute_per_plate_t50_from_rates(df, run_id)

    per_row_rows: List[dict] = []
    round_gox_t50: Dict[str, List[float]] = {}  # round_id -> list of GOx t50 values (all run,plate in that round)

    for round_id, run_ids in sorted(round_to_runs.items()):
        run_ids = sorted(run_ids)
        gox_in_round: List[float] = []
        gox_plate_info: List[tuple[str, str, float]] = []  # (run_id, plate_id, t50) for outlier reporting
        for run_id in run_ids:
            pt50 = run_plate_t50.get(run_id)
            if pt50 is None or pt50.empty:
                continue
            gox_plate = pt50[pt50["polymer_id"].str.upper() == "GOX"]
            for _, r in gox_plate.iterrows():
                plate_id = str(r.get("plate_id", ""))
                t = pd.to_numeric(r.get("t50_min", np.nan), errors="coerce")
                if np.isfinite(t) and t > 0:
                    gox_in_round.append(float(t))
                    gox_plate_info.append((run_id, plate_id, float(t)))

        if not gox_in_round:
            raise ValueError(
                f"Round {round_id!r} has no GOx t50 in any (run, plate). "
                "FoG with same_plate/same_round denominator cannot be computed."
            )

        # Outlier detection: median-based
        import warnings
        median_gox = float(np.median(gox_in_round))
        low_threshold = median_gox * gox_outlier_low_threshold
        high_threshold = median_gox * gox_outlier_high_threshold
        outliers: List[tuple[str, str, float]] = []
        for run_id, plate_id, t50_val in gox_plate_info:
            if t50_val < low_threshold or t50_val > high_threshold:
                outliers.append((run_id, plate_id, t50_val))

        if outliers:
            outlier_details = ", ".join([f"{run_id}/{plate_id}={t50_val:.3f}min" for run_id, plate_id, t50_val in outliers])
            # Collect warning info
            warning_info.outlier_gox.append({
                "round_id": round_id,
                "median_gox_t50": float(median_gox),
                "low_threshold": float(low_threshold),
                "high_threshold": float(high_threshold),
                "outliers": [{"run_id": run_id, "plate_id": plate_id, "gox_t50": float(t50_val)} for run_id, plate_id, t50_val in outliers],
                "excluded": exclude_outlier_gox,
            })
            import sys
            # Print a more visible warning message
            print("\n" + "=" * 80, file=sys.stderr)
            print("⚠️  警告: 異常GOx t50値が検出されました", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(f"Round: {round_id}", file=sys.stderr)
            print(f"GOx t50 中央値: {median_gox:.3f} min", file=sys.stderr)
            print(f"異常値の閾値: [{low_threshold:.3f}, {high_threshold:.3f}] min", file=sys.stderr)
            print(f"検出された異常値: {outlier_details}", file=sys.stderr)
            if exclude_outlier_gox:
                print("→ 異常値はround平均GOx t50の計算から除外されます。", file=sys.stderr)
            else:
                print("→ 異常値はround平均GOx t50の計算に含まれます。", file=sys.stderr)
                print("→ 除外する場合は、--exclude_outlier_gox フラグを使用してください。", file=sys.stderr)
            print("=" * 80 + "\n", file=sys.stderr)
            # Also issue standard warning for programmatic access
            warnings.warn(
                f"Round {round_id!r} has outlier GOx t50 values (median={median_gox:.3f}min, "
                f"thresholds=[{low_threshold:.3f}, {high_threshold:.3f}]min): {outlier_details}. "
                f"{'Excluding outliers from round average.' if exclude_outlier_gox else 'Outliers are included in round average. Use --exclude_outlier_gox to exclude.'}",
                UserWarning,
                stacklevel=2,
            )
            if exclude_outlier_gox:
                outlier_t50s = {t50_val for _, _, t50_val in outliers}
                gox_in_round = [t for t in gox_in_round if t not in outlier_t50s]
                if not gox_in_round:
                    raise ValueError(
                        f"Round {round_id!r} has no GOx t50 remaining after excluding outliers. "
                        "Cannot compute FoG."
                    )

        round_gox_t50[round_id] = gox_in_round
        # Round average GOx t50: simple mean of all (run, plate) values with equal weight
        # (run-level weighting is not applied; each plate contributes equally)
        mean_gox_round = float(np.mean(gox_in_round))

        for run_id in run_ids:
            pt50 = run_plate_t50.get(run_id)
            if pt50 is None or pt50.empty:
                continue
            gox_by_plate: Dict[str, float] = {}
            gox_plate = pt50[pt50["polymer_id"].str.upper() == "GOX"]
            for _, r in gox_plate.iterrows():
                plate_id = str(r.get("plate_id", ""))
                t = pd.to_numeric(r.get("t50_min", np.nan), errors="coerce")
                if np.isfinite(t) and t > 0:
                    # Check for duplicate GOx t50 for the same plate (should not happen)
                    if plate_id in gox_by_plate:
                        existing_t50 = gox_by_plate[plate_id]
                        raise ValueError(
                            f"Run {run_id!r}, plate {plate_id!r} has multiple GOx t50 values: "
                            f"{existing_t50:.3f}min and {t:.3f}min. "
                            "Each plate should have exactly one GOx t50 value."
                        )
                    gox_by_plate[plate_id] = float(t)

            polymers = pt50[pt50["polymer_id"].str.upper() != "GOX"]
            for _, r in polymers.iterrows():
                plate_id = str(r.get("plate_id", ""))
                polymer_id = str(r.get("polymer_id", "")).strip()
                t50_min = pd.to_numeric(r.get("t50_min", np.nan), errors="coerce")
                if not np.isfinite(t50_min) or t50_min <= 0:
                    continue
                gox_t50 = gox_by_plate.get(plate_id)
                if gox_t50 is not None and gox_t50 > 0:
                    denominator_source = "same_plate"
                else:
                    gox_t50 = mean_gox_round
                    denominator_source = "same_round"
                fog = t50_min / gox_t50
                log_fog = np.log(fog) if fog > 0 else np.nan
                per_row_rows.append({
                    "round_id": round_id,
                    "run_id": run_id,
                    "plate_id": plate_id,
                    "polymer_id": polymer_id,
                    "t50_min": t50_min,
                    "gox_t50_used_min": gox_t50,
                    "denominator_source": denominator_source,
                    "fog": fog,
                    "log_fog": log_fog,
                })

    per_row_df = pd.DataFrame(per_row_rows)

    # Round-averaged: by (round_id, polymer_id)
    round_av_rows: List[dict] = []
    for round_id in sorted(round_to_runs.keys()):
        sub = per_row_df[per_row_df["round_id"] == round_id].copy() if not per_row_df.empty else pd.DataFrame()
        if sub.empty:
            continue
        fog = pd.to_numeric(sub["fog"], errors="coerce")
        log_fog = pd.to_numeric(sub["log_fog"], errors="coerce")
        valid = np.isfinite(fog) & (fog > 0)
        sub = sub.loc[valid].copy()
        for polymer_id, g in sub.groupby("polymer_id", sort=False):
            pid = str(polymer_id).strip()
            round_av_rows.append({
                "round_id": round_id,
                "polymer_id": pid,
                "mean_fog": float(g["fog"].mean()),
                "mean_log_fog": float(g["log_fog"].mean()),
                "n_observations": int(len(g)),
                "run_ids": ",".join(sorted(g["run_id"].astype(str).unique().tolist())),
            })
    round_averaged_df = pd.DataFrame(round_av_rows)

    gox_trace_df = build_round_gox_traceability(run_round_map, processed_dir)
    return per_row_df, round_averaged_df, gox_trace_df, warning_info

