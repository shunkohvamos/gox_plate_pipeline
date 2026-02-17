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
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import trim_mean
import yaml


# t50 unit used in outputs (documented for BO and figures)
T50_UNIT = "min"

# Native-activity feasibility threshold for constrained FoG objective.
# Objective policy: maximize FoG only among polymers that keep sufficient baseline activity.
# Main-analysis default is 0.70 (sensitivity analyses should test 0.50-0.90).
NATIVE_ACTIVITY_MIN_REL_DEFAULT = 0.70
DEFAULT_REFERENCE_POLYMER_ID = "GOX"


def _ensure_csv_subdir(base_dir: Path) -> Path:
    csv_dir = Path(base_dir) / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir


def _remove_legacy_csv(legacy_path: Path, current_path: Path) -> None:
    try:
        if legacy_path.resolve() == current_path.resolve():
            return
    except FileNotFoundError:
        pass
    if legacy_path.is_file():
        legacy_path.unlink(missing_ok=True)


def _normalize_polymer_id_token(value: object) -> str:
    return str(value).strip().upper()


def _is_reference_polymer_id(
    polymer_id: object,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    return _normalize_polymer_id_token(polymer_id) == _normalize_polymer_id_token(reference_polymer_id)


def _missing_reference_reason(reference_polymer_id: str) -> str:
    if _is_reference_polymer_id(reference_polymer_id, DEFAULT_REFERENCE_POLYMER_ID):
        return "no_bare_gox_in_run"
    return "no_reference_polymer_in_run"


def _aggregate_reference_values(
    values: List[float],
    *,
    method: str = "median",
    trimmed_mean_proportion: float = 0.1,
) -> float:
    """Aggregate positive finite reference values with robust method."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return np.nan
    m = str(method).strip().lower()
    if m == "median":
        return float(np.nanmedian(arr))
    if m == "trimmed_mean":
        n = int(arr.size)
        if n >= 3:
            p = min(float(trimmed_mean_proportion), (n - 1) / (2 * n))
            return float(trim_mean(arr, proportiontocut=p))
        return float(np.nanmedian(arr))
    if m == "mean":
        return float(np.nanmean(arr))
    raise ValueError(f"Unknown reference aggregation method: {method!r}")


def _u_col_name(heat_min: float) -> str:
    """Column name for U(t) at a heat time."""
    h = float(heat_min)
    if math.isclose(h, round(h), rel_tol=0.0, abs_tol=1e-9):
        return f"U_{int(round(h))}"
    txt = f"{h:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")
    return f"U_{txt}"


def _compute_t_theta_from_series(
    t: np.ndarray,
    u: np.ndarray,
    *,
    theta: float,
) -> Tuple[float, str]:
    """
    Compute t_theta where U(t) first falls below theta.

    Returns:
      (t_theta, censor_flag)
        - censor_flag = "" for interpolated crossing
        - "already_below" if first finite U is below theta
        - "never_cross" if never below theta (right-censored at max heat time)
        - "missing_profile" if no finite (t,u) points
    """
    tt = np.asarray(t, dtype=float)
    uu = np.asarray(u, dtype=float)
    mask = np.isfinite(tt) & np.isfinite(uu)
    tt = tt[mask]
    uu = uu[mask]
    if tt.size == 0:
        return np.nan, "missing_profile"
    order = np.argsort(tt)
    tt = tt[order]
    uu = uu[order]
    th = float(theta)
    if uu[0] < th:
        return float(tt[0]), "already_below"
    for i in range(int(tt.size) - 1):
        t0 = float(tt[i])
        t1 = float(tt[i + 1])
        u0 = float(uu[i])
        u1 = float(uu[i + 1])
        if (u0 >= th and u1 < th) or (u0 > th and u1 <= th):
            if math.isclose(u1, u0, rel_tol=0.0, abs_tol=1e-12):
                return float(t0), ""
            frac = (th - u0) / (u1 - u0)
            return float(t0 + frac * (t1 - t0)), ""
    return float(tt[-1]), "never_cross"


@dataclass
class FogWarningInfo:
    """Warning information collected during FoG calculation."""
    outlier_gox: List[Dict[str, Any]] = field(default_factory=list)  # List of outlier GOx t50 info
    guarded_same_plate: List[Dict[str, Any]] = field(default_factory=list)  # same_plate guard fallback details
    missing_rates_files: List[Dict[str, str]] = field(default_factory=list)  # List of missing rates_with_rea.csv
    reference_qc_fail_runs: List[Dict[str, Any]] = field(default_factory=list)  # run-level reference QC fails


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
    
    if (
        not warning_info.outlier_gox
        and not warning_info.guarded_same_plate
        and not warning_info.missing_rates_files
        and not warning_info.reference_qc_fail_runs
    ):
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

        if warning_info.guarded_same_plate:
            lines.append("## ⚠️ same_plate分母ガードによるフォールバック")
            lines.append("")
            lines.append("異常なsame_plate GOx t50が検出されたため、分母をround代表値へフォールバックした行です。")
            lines.append("")
            for i, info in enumerate(warning_info.guarded_same_plate, 1):
                lines.append(
                    f"### {i}. Round {info['round_id']} / Run {info['run_id']} / Plate {info['plate_id']}"
                )
                lines.append("")
                lines.append(f"- **same_plate GOx t50**: {info['same_plate_gox_t50_min']:.3f} min")
                lines.append(f"- **round代表GOx t50**: {info['fallback_gox_t50_min']:.3f} min")
                lines.append(
                    f"- **ガード閾値**: [{info['guard_low_threshold_min']:.3f}, {info['guard_high_threshold_min']:.3f}] min"
                )
                lines.append("- **処理**: `denominator_source = same_round` へフォールバック")
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

        if warning_info.reference_qc_fail_runs:
            lines.append("## ⚠️ 参照abs(0) QC fail run")
            lines.append("")
            lines.append("同一run内の参照abs_activity_at_0のばらつきまたは絶対値に問題があるため、QC failとなったrunです。")
            lines.append("")
            for i, info in enumerate(warning_info.reference_qc_fail_runs, 1):
                lines.append(f"### {i}. Round {info.get('round_id', '')}, Run {info.get('run_id', '')}")
                lines.append("")
                lines.append(f"- **n_reference_wells**: {info.get('n_reference_wells', '')}")
                lines.append(f"- **ref_abs0_median**: {info.get('ref_abs0_median', '')}")
                lines.append(f"- **ref_abs0_mad**: {info.get('ref_abs0_mad', '')}")
                lines.append(f"- **ref_abs0_rel_mad**: {info.get('ref_abs0_rel_mad', '')}")
                lines.append(f"- **reason**: {info.get('reason', '')}")
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
    row_map_path: Optional[Path] = None,
    native_activity_min_rel: float = NATIVE_ACTIVITY_MIN_REL_DEFAULT,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> pd.DataFrame:
    """
    Build FoG summary from t50 CSV: FoG = t50_polymer / t50_bare_GOx (same run only).

    - t50 unit: minutes (T50_UNIT).
    - t50_censored: 1 if t50 is missing or REA plateau >= t50 target (did not reach target); 0 otherwise.
      Target uses t50_target_rea_percent when present (new schema), else legacy 50%.
    - gox_t50_same_run: bare GOx t50 in same run (min). NaN if no GOx in run.
    - fog: t50 / gox_t50_same_run. NaN if no GOx in run or t50 missing.
    - log_fog: log(fog) for BO; NaN when fog <= 0 or missing (handle consistently).
    - fog_missing_reason: e.g. "no_bare_gox_in_run" when GOx absent in run; empty otherwise.
    - use_for_bo: if row_map_path is provided, reads use_for_bo from metadata TSV (defaults to True).
      Controls whether this polymer should be included in Bayesian optimization learning data.
    - abs_activity_at_0: absolute activity anchor (heat=0; from t50 CSV when available).
    - gox_abs_activity_at_0_ref: GOx absolute-activity reference at heat=0 from per-run/round fallback resolver.
    - native_activity_rel_at_0: abs_activity_at_0 / gox_abs_activity_at_0.
    - native_activity_feasible: 1 when native_activity_rel_at_0 >= native_activity_min_rel, else 0.
    - fog_native_constrained: fog masked by native_activity_feasible (NaN when infeasible).
    - log_fog_native_constrained: log(fog_native_constrained), NaN when infeasible.
    - Lineage: run_id, input_t50_file, input_tidy (from manifest or passed).
    """
    t50_path = Path(t50_path)
    run_id = str(run_id).strip()
    reference_polymer_id = str(reference_polymer_id).strip() or DEFAULT_REFERENCE_POLYMER_ID
    reference_polymer_norm = _normalize_polymer_id_token(reference_polymer_id)
    df = pd.read_csv(t50_path)

    if "run_id" not in df.columns:
        df["run_id"] = run_id
    if "polymer_id" not in df.columns:
        raise ValueError(f"t50 CSV must have polymer_id, got: {list(df.columns)}")
    if "reference_polymer_id" in df.columns:
        df["reference_polymer_id"] = df["reference_polymer_id"].astype(str).str.strip()
        df["reference_polymer_id"] = np.where(
            df["reference_polymer_id"] != "",
            df["reference_polymer_id"],
            reference_polymer_id,
        )
    else:
        df["reference_polymer_id"] = reference_polymer_id

    native_activity_min_rel = float(native_activity_min_rel)
    if not np.isfinite(native_activity_min_rel) or native_activity_min_rel <= 0.0:
        raise ValueError(
            f"native_activity_min_rel must be a positive finite number, got {native_activity_min_rel!r}"
        )

    # Prefer t50_exp_min, fallback to t50_linear_min (unit: min)
    t50_exp = pd.to_numeric(df.get("t50_exp_min", np.nan), errors="coerce")
    t50_lin = pd.to_numeric(df.get("t50_linear_min", np.nan), errors="coerce")
    df["t50_min"] = np.where(np.isfinite(t50_exp), t50_exp, t50_lin)
    df["abs_activity_at_0"] = pd.to_numeric(df.get("abs_activity_at_0", np.nan), errors="coerce")
    df["abs_activity_at_20"] = pd.to_numeric(df.get("abs_activity_at_20", np.nan), errors="coerce")
    df["gox_abs_activity_at_0_ref"] = pd.to_numeric(df.get("gox_abs_activity_at_0_ref", np.nan), errors="coerce")
    df["gox_abs_activity_at_20_ref"] = pd.to_numeric(df.get("gox_abs_activity_at_20_ref", np.nan), errors="coerce")
    df["functional_activity_at_20_rel"] = pd.to_numeric(df.get("functional_activity_at_20_rel", np.nan), errors="coerce")
    df["functional_reference_source"] = df.get("functional_reference_source", pd.Series([""] * len(df))).astype(str)
    df["functional_reference_round_id"] = df.get("functional_reference_round_id", pd.Series([""] * len(df))).astype(str)
    df["functional_reference_run_id"] = df.get("functional_reference_run_id", pd.Series([""] * len(df))).astype(str)

    # Censored: no t50 or plateau >= target (REA did not reach t50 target)
    fit_plateau = pd.to_numeric(df.get("fit_plateau", np.nan), errors="coerce")
    fit_model = df.get("fit_model", "").astype(str)
    no_t50 = ~np.isfinite(df["t50_min"])
    target = pd.to_numeric(df.get("t50_target_rea_percent", np.nan), errors="coerce")
    target = np.where(np.isfinite(target), target, 50.0)
    plateau_not_reached = (
        fit_model.str.contains("exp_plateau", na=False)
        & np.isfinite(fit_plateau)
        & np.isfinite(target)
        & (fit_plateau >= target)
    )
    df["t50_censored"] = (no_t50 | plateau_not_reached).astype(int)

    # Bare reference-polymer t50 in same run (this CSV is single-run, so one value)
    ref_rows = df[
        df["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) == reference_polymer_norm)
    ]
    if ref_rows.empty:
        gox_t50_same_run = np.nan
        fog_missing_reason_default = _missing_reference_reason(reference_polymer_id)
    else:
        gox_t50_same_run = float(ref_rows["t50_min"].iloc[0]) if np.isfinite(ref_rows["t50_min"].iloc[0]) else np.nan
        fog_missing_reason_default = "" if np.isfinite(gox_t50_same_run) else _missing_reference_reason(reference_polymer_id)

    df["gox_t50_same_run_min"] = gox_t50_same_run
    df["fog_missing_reason"] = fog_missing_reason_default

    # Native-activity guard baseline: same-run reference-polymer absolute activity at heat=0.
    gox_abs0_same_run = np.nan
    if not ref_rows.empty:
        gox_abs0_vals = pd.to_numeric(ref_rows.get("abs_activity_at_0", np.nan), errors="coerce")
        gox_abs0_vals = gox_abs0_vals[np.isfinite(gox_abs0_vals) & (gox_abs0_vals > 0)]
        if not gox_abs0_vals.empty:
            gox_abs0_same_run = float(gox_abs0_vals.iloc[0])
    ref_abs0 = pd.to_numeric(df.get("gox_abs_activity_at_0_ref", np.nan), errors="coerce")
    if np.isfinite(gox_abs0_same_run):
        ref_abs0 = np.where(np.isfinite(ref_abs0), ref_abs0, float(gox_abs0_same_run))
    df["gox_abs_activity_at_0"] = pd.to_numeric(ref_abs0, errors="coerce")

    native_rel = np.where(
        np.isfinite(df["abs_activity_at_0"])
        & np.isfinite(df["gox_abs_activity_at_0"])
        & (df["gox_abs_activity_at_0"] > 0.0),
        df["abs_activity_at_0"] / df["gox_abs_activity_at_0"],
        np.nan,
    )
    df["native_activity_rel_at_0"] = native_rel
    df["native_activity_min_rel_threshold"] = float(native_activity_min_rel)
    native_feasible = np.isfinite(native_rel) & (native_rel >= float(native_activity_min_rel))
    df["native_activity_feasible"] = native_feasible.astype(int)

    native_reason = np.full(len(df), "", dtype=object)
    missing_abs0 = ~np.isfinite(df["abs_activity_at_0"])
    missing_gox0 = ~np.isfinite(df["gox_abs_activity_at_0"]) | (df["gox_abs_activity_at_0"] <= 0.0)
    below_native = np.isfinite(native_rel) & (native_rel < float(native_activity_min_rel))
    native_reason = np.where(missing_abs0, "missing_abs_activity_at_0", native_reason)
    native_reason = np.where(missing_gox0 & ~missing_abs0, "missing_gox_abs_activity_at_0", native_reason)
    native_reason = np.where(below_native, "native_activity_below_threshold", native_reason)
    df["fog_constraint_reason"] = native_reason

    # Functional activity at 20 min relative to reference polymer.
    # Prefer value pre-computed in t50 CSV; otherwise derive from abs_activity_at_20 / gox_abs_activity_at_20_ref.
    gox_abs20_same_run = np.nan
    if not ref_rows.empty:
        gox_abs20_vals = pd.to_numeric(ref_rows.get("abs_activity_at_20", np.nan), errors="coerce")
        gox_abs20_vals = gox_abs20_vals[np.isfinite(gox_abs20_vals) & (gox_abs20_vals > 0)]
        if not gox_abs20_vals.empty:
            gox_abs20_same_run = float(gox_abs20_vals.iloc[0])

    ref_abs20 = pd.to_numeric(df.get("gox_abs_activity_at_20_ref", np.nan), errors="coerce")
    if not (np.isfinite(ref_abs20).any()):
        ref_abs20 = np.full(len(df), gox_abs20_same_run, dtype=float)
    ref_abs20 = pd.to_numeric(ref_abs20, errors="coerce")
    df["gox_abs_activity_at_20_ref"] = ref_abs20

    fa20 = pd.to_numeric(df.get("functional_activity_at_20_rel", np.nan), errors="coerce")
    fallback_fa20 = np.where(
        np.isfinite(df["abs_activity_at_20"]) & np.isfinite(df["gox_abs_activity_at_20_ref"]) & (df["gox_abs_activity_at_20_ref"] > 0),
        df["abs_activity_at_20"] / df["gox_abs_activity_at_20_ref"],
        np.nan,
    )
    df["functional_activity_at_20_rel"] = np.where(np.isfinite(fa20), fa20, fallback_fa20)
    if "functional_reference_source" in df.columns:
        src = df["functional_reference_source"].astype(str).str.strip()
        default_src_present = (
            "same_run_gox" if _is_reference_polymer_id(reference_polymer_id, DEFAULT_REFERENCE_POLYMER_ID) else "same_run_reference"
        )
        default_src_missing = (
            "missing_gox_reference"
            if _is_reference_polymer_id(reference_polymer_id, DEFAULT_REFERENCE_POLYMER_ID)
            else "missing_reference_polymer"
        )
        src = src.where(src != "", np.where(np.isfinite(df["gox_abs_activity_at_20_ref"]), default_src_present, default_src_missing))
        df["functional_reference_source"] = src

    fa20_rel = pd.to_numeric(df["functional_activity_at_20_rel"], errors="coerce")
    log_fa20 = np.full(len(df), np.nan, dtype=float)
    ok_fa = np.isfinite(fa20_rel) & (fa20_rel > 0)
    log_fa20[ok_fa] = np.log(fa20_rel[ok_fa])
    df["log_functional_activity_at_20_rel"] = log_fa20

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

    # Constrained objective for BO/ranking:
    # optimize FoG only among polymers that keep enough native activity at heat=0.
    fog_native = np.where(native_feasible, fog, np.nan)
    df["fog_native_constrained"] = fog_native
    log_fog_native = np.full_like(fog_native, np.nan, dtype=float)
    ok_native = np.isfinite(fog_native) & (fog_native > 0)
    log_fog_native[ok_native] = np.log(fog_native[ok_native])
    df["log_fog_native_constrained"] = log_fog_native

    # Load use_for_bo from row_map if provided
    if row_map_path is not None and Path(row_map_path).is_file():
        from gox_plate_pipeline.loader import read_row_map_tsv
        row_map = read_row_map_tsv(Path(row_map_path))
        # Merge by polymer_id (t50 CSV aggregates by polymer_id, so we match by polymer_id)
        df["polymer_id_norm"] = df["polymer_id"].astype(str).str.strip()
        row_map["polymer_id_norm"] = row_map["polymer_id"].astype(str).str.strip()
        # Take the first match for each polymer_id (in case of duplicates, use first)
        use_for_bo_map = row_map.groupby("polymer_id_norm")["use_for_bo"].first().to_dict()
        df["use_for_bo"] = df["polymer_id_norm"].map(use_for_bo_map).fillna(True)  # Default True if not found
        df = df.drop(columns=["polymer_id_norm"])
    else:
        df["use_for_bo"] = True  # Default: include in BO

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
        "reference_polymer_id",
        "t50_min",
        "t50_definition",
        "t50_target_rea_percent",
        "rea_at_20_percent",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "native_activity_rel_at_0",
        "native_activity_min_rel_threshold",
        "native_activity_feasible",
        "abs_activity_at_20",
        "gox_abs_activity_at_20_ref",
        "functional_activity_at_20_rel",
        "log_functional_activity_at_20_rel",
        "functional_reference_source",
        "functional_reference_round_id",
        "functional_reference_run_id",
        "t50_censored",
        "gox_t50_same_run_min",
        "fog",
        "log_fog",
        "fog_native_constrained",
        "log_fog_native_constrained",
        "fog_constraint_reason",
        "fog_missing_reason",
        "use_for_bo",
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


def _load_polymer_color_map(color_map_path: Optional[Path]) -> Dict[str, str]:
    """Load polymer_id -> color map from YAML."""
    if color_map_path is None:
        return {}
    path = Path(color_map_path)
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except Exception:
        return {}
    if isinstance(payload, dict) and "polymer_id" in payload and isinstance(payload["polymer_id"], dict):
        payload = payload["polymer_id"]
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in payload.items():
        if isinstance(v, str) and v.strip():
            out[str(k).strip()] = v.strip()
    return out


def _add_scatter_polymer_mapping_legend(
    ax: Any,
    scatter_df: pd.DataFrame,
    *,
    color_map: Optional[Dict[str, str]] = None,
    polymer_col: str = "polymer_id",
    feasible_col: str = "_feasible",
) -> None:
    """
    Add a compact lower-left mapping legend (marker/color -> polymer_id) for scatter plots.
    """
    if scatter_df.empty:
        return
    from matplotlib.lines import Line2D

    df = scatter_df.copy()
    if polymer_col not in df.columns:
        return
    df[polymer_col] = df[polymer_col].astype(str).str.strip()
    df = df[df[polymer_col] != ""].copy()
    if df.empty:
        return
    if "fog" in df.columns:
        df = df.sort_values("fog", ascending=False, kind="mergesort")

    def _polymer_map_order_key(polymer_id: str) -> tuple[int, int, str]:
        """Order legend as: GOx/GOx with*, PMPC, PMTAC, PMBTA-1..N, others."""
        pid = str(polymer_id).strip()
        norm = _normalize_polymer_id_token(pid)
        if norm == "GOX":
            return (0, 0, "")
        if norm.startswith("GOX WITH"):
            suffix = norm[len("GOX WITH") :].strip()
            return (0, 1, suffix)
        if norm.startswith("GOX"):
            return (0, 2, norm)
        if norm == "PMPC":
            return (1, 0, "")
        if norm == "PMTAC":
            return (2, 0, "")
        m = re.match(r"^PMBTA-(\d+)$", norm)
        if m is not None:
            return (3, int(m.group(1)), "")
        return (4, 0, norm)

    cmap = color_map or {}
    default_color = "#4C78A8"
    entries: List[tuple[str, Any]] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        pid = str(row.get(polymer_col, "")).strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        color = cmap.get(pid, default_color)
        feasible = bool(row.get(feasible_col, True))
        if feasible:
            handle = Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=4.2,
                markerfacecolor=color,
                markeredgecolor="0.2",
                markeredgewidth=0.5,
                alpha=0.9,
                label=pid,
            )
        else:
            handle = Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=4.2,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=0.9,
                alpha=0.9,
                label=pid,
            )
        entries.append((pid, handle))
    if not entries:
        return

    entries = sorted(entries, key=lambda x: _polymer_map_order_key(x[0]))
    handles: List[Any] = [h for _, h in entries]
    if not handles:
        return

    ncol = 1 if len(handles) <= 8 else 2 if len(handles) <= 18 else 3
    legend = ax.legend(
        handles=handles,
        title="Polymer map",
        loc="lower left",
        bbox_to_anchor=(0.012, 0.012),
        ncol=ncol,
        fontsize=4.6,
        title_fontsize=5.0,
        frameon=True,
        framealpha=0.90,
        borderpad=0.26,
        labelspacing=0.22,
        handletextpad=0.35,
        columnspacing=0.78,
        borderaxespad=0.14,
    )
    frame = legend.get_frame()
    frame.set_edgecolor("0.65")
    frame.set_linewidth(0.5)


def _plot_run_ranking_bar(
    rank_df: pd.DataFrame,
    *,
    value_col: str,
    error_col: Optional[str] = None,
    rank_col: str,
    title: str,
    xlabel: str,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """
    Plot one run-level ranking bar chart (paper-grade, English).

    Returns True when a figure was written.
    """
    if rank_df.empty:
        return False
    data = rank_df.copy()
    data = data[np.isfinite(pd.to_numeric(data.get(value_col, np.nan), errors="coerce"))].copy()
    data = data[np.isfinite(pd.to_numeric(data.get(rank_col, np.nan), errors="coerce"))].copy()
    if data.empty:
        return False
    data = data.sort_values(rank_col, ascending=True).reset_index(drop=True)
    if error_col is not None and str(error_col).strip():
        if error_col in data.columns:
            data[error_col] = pd.to_numeric(data.get(error_col, np.nan), errors="coerce")
        else:
            error_col = None

    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    labels = []
    values = []
    colors = []
    cmap = color_map or {}
    default_color = "#4C78A8"
    for _, row in data.iterrows():
        pid = str(row.get("polymer_id", "")).strip()
        labels.append(pid)
        values.append(float(row[value_col]))
        if _is_reference_polymer_id(pid, reference_polymer_id=reference_polymer_id):
            colors.append("#808080")
        else:
            colors.append(cmap.get(pid, default_color))
    xerr_values: Optional[np.ndarray] = None
    if error_col is not None:
        err = pd.to_numeric(data.get(error_col, np.nan), errors="coerce").to_numpy(dtype=float)
        err = np.where(np.isfinite(err) & (err > 0.0), err, 0.0)
        if np.any(err > 0.0):
            xerr_values = err

    fig_h = max(2.2, 0.28 * len(labels) + 0.8)
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(5.0, fig_h))
        y = np.arange(len(labels), dtype=float)
        bar_kwargs = dict(
            color=colors,
            edgecolor="0.2",
            linewidth=0.4,
            height=0.62,
        )
        if xerr_values is not None:
            bar_kwargs["xerr"] = xerr_values
            bar_kwargs["error_kw"] = {
                "ecolor": PAPER_ERRORBAR_COLOR,
                "elinewidth": 0.8,
                "capsize": 2.0,
                "capthick": 0.8,
                "alpha": 0.9,
            }
        ax.barh(y, values, **bar_kwargs)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_fog_native_constrained_tradeoff(
    run_df: pd.DataFrame,
    *,
    theta: float,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """
    Plot trade-off scatter: x = native_activity_rel_at_0, y = FoG, with vertical line at theta.
    Paper-grade, English. Excludes the reference polymer row. Returns True when a figure was written.
    """
    df = run_df.copy()
    df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
    ref_norm = _normalize_polymer_id_token(reference_polymer_id)
    ref_mask = df["polymer_id"].map(lambda x: _normalize_polymer_id_token(x) == ref_norm)
    df = df[~ref_mask].copy()
    if df.empty:
        return False
    x = pd.to_numeric(df.get("native_activity_rel_at_0", np.nan), errors="coerce")
    x_sem = pd.to_numeric(df.get("native_activity_rel_at_0_sem", np.nan), errors="coerce")
    y = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    y_sem = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    has_any = np.any(np.isfinite(x)) and np.any(np.isfinite(y))
    if not has_any:
        return False
    fe_col = pd.to_numeric(df.get("native_activity_feasible", np.nan), errors="coerce")
    if np.any(np.isfinite(fe_col)):
        df["_feasible"] = fe_col.fillna(0).astype(int) > 0
    else:
        df["_feasible"] = np.isfinite(x) & (x >= float(theta))
    cmap = color_map or {}
    default_color = "#4C78A8"

    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        plotted: List[tuple[str, float, float, bool]] = []
        for _, row in df.iterrows():
            xi = float(row.get("native_activity_rel_at_0", np.nan))
            yi = float(row.get("fog", np.nan))
            if not (np.isfinite(xi) and np.isfinite(yi) and yi > 0):
                continue
            pid = str(row.get("polymer_id", "")).strip()
            fe = bool(row.get("_feasible", False))
            color = cmap.get(pid, default_color)
            xi_sem = float(row.get("native_activity_rel_at_0_sem", np.nan))
            yi_sem = float(row.get("fog_sem", np.nan))
            if (np.isfinite(xi_sem) and xi_sem > 0.0) or (np.isfinite(yi_sem) and yi_sem > 0.0):
                ax.errorbar(
                    xi,
                    yi,
                    xerr=(xi_sem if np.isfinite(xi_sem) and xi_sem > 0.0 else None),
                    yerr=(yi_sem if np.isfinite(yi_sem) and yi_sem > 0.0 else None),
                    fmt="none",
                    ecolor=PAPER_ERRORBAR_COLOR,
                    elinewidth=0.75,
                    capsize=1.8,
                    alpha=0.8,
                    zorder=16,
                )
            ax.scatter(
                xi,
                yi,
                s=28,
                color=color,
                edgecolors="0.2",
                linewidths=0.5,
                alpha=0.9,
                zorder=20,
                marker="o" if fe else "s",
            )
            plotted.append((pid, xi, yi, fe))
        ax.axvline(
            x=theta,
            color="0.4",
            linestyle="--",
            linewidth=0.8,
            zorder=10,
            label=rf"$\theta$ = {theta:.2f}",
        )
        ax.set_xlabel(rf"$U_{{0}}$ (vs {reference_polymer_id} at 0 min)")
        ax.set_ylabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
        ax.set_title("FoG vs native activity (constrained objective)")
        ax.legend(loc="upper right", fontsize=6)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_xlim(0.0, None)
        ax.set_ylim(0.0, None)
        _add_scatter_polymer_mapping_legend(ax, df, color_map=cmap, feasible_col="_feasible")
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_fog_native_constrained_decision(
    run_df: pd.DataFrame,
    *,
    theta: float,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """
    Plot decision chart for constrained FoG objective (paper main figure style):
      left  = native_0 (vs reference at 0 min) with gate line (theta)
      right = FoG with infeasible rows visually marked.
    Returns True when a figure was written.
    """
    df = run_df.copy()
    df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
    ref_norm = _normalize_polymer_id_token(reference_polymer_id)
    df = df[df["polymer_id"].map(lambda x: _normalize_polymer_id_token(x) != ref_norm)].copy()
    if df.empty:
        return False

    df["native_activity_rel_at_0"] = pd.to_numeric(df.get("native_activity_rel_at_0", np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    df = df[np.isfinite(df["native_activity_rel_at_0"]) & np.isfinite(df["fog"]) & (df["fog"] > 0)].copy()
    if df.empty:
        return False

    if "native_activity_feasible" in df.columns and np.any(pd.notna(df["native_activity_feasible"])):
        feasible = pd.to_numeric(df["native_activity_feasible"], errors="coerce").fillna(0).astype(int) > 0
    else:
        feasible = df["native_activity_rel_at_0"] >= float(theta)
    df["native_activity_feasible"] = feasible.astype(int)

    df = df.sort_values(["native_activity_feasible", "fog"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    if df.empty:
        return False

    cmap = color_map or {}
    default_color = "#4C78A8"
    colors: List[str] = []
    for _, row in df.iterrows():
        pid = str(row.get("polymer_id", "")).strip()
        colors.append(cmap.get(pid, default_color))
    labels = [str(pid) for pid in df["polymer_id"].tolist()]
    y = np.arange(len(df), dtype=float)
    h = max(2.5, 0.30 * len(df) + 0.9)

    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig, (ax_native, ax_fog) = plt.subplots(
            1,
            2,
            figsize=(7.2, h),
            sharey=True,
            gridspec_kw={"width_ratios": [1.0, 1.2]},
        )

        native_vals = df["native_activity_rel_at_0"].to_numpy(dtype=float)
        fog_vals = df["fog"].to_numpy(dtype=float)
        feasible_vals = df["native_activity_feasible"].to_numpy(dtype=int)
        if "native_activity_rel_at_0_sem" in df.columns:
            native_sem_vals = pd.to_numeric(df["native_activity_rel_at_0_sem"], errors="coerce").to_numpy(dtype=float)
        else:
            native_sem_vals = np.full(len(df), np.nan, dtype=float)
        if "fog_sem" in df.columns:
            fog_sem_vals = pd.to_numeric(df["fog_sem"], errors="coerce").to_numpy(dtype=float)
        else:
            fog_sem_vals = np.full(len(df), np.nan, dtype=float)
        if "fog_native_constrained_sem" in df.columns:
            fog_native_sem_vals = pd.to_numeric(df["fog_native_constrained_sem"], errors="coerce").to_numpy(dtype=float)
        else:
            fog_native_sem_vals = np.full(len(df), np.nan, dtype=float)
        fog_err_vals = np.where(np.isfinite(fog_native_sem_vals), fog_native_sem_vals, fog_sem_vals)

        # Left panel: native gate.
        native_xerr = np.where(np.isfinite(native_sem_vals) & (native_sem_vals > 0.0), native_sem_vals, 0.0)
        native_bar_kwargs = {}
        if np.any(native_xerr > 0.0):
            native_bar_kwargs["xerr"] = native_xerr
            native_bar_kwargs["error_kw"] = {
                "ecolor": PAPER_ERRORBAR_COLOR,
                "elinewidth": 0.8,
                "capsize": 2.0,
                "capthick": 0.8,
                "alpha": 0.85,
            }
        native_bars = ax_native.barh(
            y,
            native_vals,
            color=colors,
            edgecolor="0.2",
            linewidth=0.4,
            height=0.62,
            alpha=0.92,
            zorder=15,
            **native_bar_kwargs,
        )
        for bar, fe in zip(native_bars, feasible_vals):
            if int(fe) <= 0:
                bar.set_alpha(0.40)
                bar.set_hatch("//")
                bar.set_edgecolor("0.45")
        ax_native.axvline(
            x=float(theta),
            color="0.25",
            linestyle=(0, (3, 2)),
            linewidth=0.8,
            alpha=0.95,
            zorder=20,
        )
        ax_native.set_xlabel(rf"$U_{{0}}$ (vs {reference_polymer_id} at 0 min)")
        ax_native.set_yticks(y)
        ax_native.set_yticklabels(labels)
        ax_native.invert_yaxis()
        ax_native.grid(axis="x", linestyle=":", alpha=0.28)

        # Right panel: FoG with feasible/infeasible marking.
        fog_xerr = np.where(np.isfinite(fog_err_vals) & (fog_err_vals > 0.0), fog_err_vals, 0.0)
        fog_bar_kwargs = {}
        if np.any(fog_xerr > 0.0):
            fog_bar_kwargs["xerr"] = fog_xerr
            fog_bar_kwargs["error_kw"] = {
                "ecolor": PAPER_ERRORBAR_COLOR,
                "elinewidth": 0.8,
                "capsize": 2.0,
                "capthick": 0.8,
                "alpha": 0.85,
            }
        fog_bars = ax_fog.barh(
            y,
            fog_vals,
            color=colors,
            edgecolor="0.2",
            linewidth=0.4,
            height=0.62,
            alpha=0.92,
            zorder=15,
            **fog_bar_kwargs,
        )
        for bar, fe in zip(fog_bars, feasible_vals):
            if int(fe) <= 0:
                bar.set_alpha(0.40)
                bar.set_hatch("//")
                bar.set_edgecolor("0.45")
        ax_fog.axvline(
            x=1.0,
            color="0.35",
            linestyle=(0, (3, 2)),
            linewidth=0.75,
            alpha=0.85,
            zorder=20,
        )
        ax_fog.set_xlabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
        ax_fog.grid(axis="x", linestyle=":", alpha=0.28)

        # Keep y labels only on the left.
        ax_fog.tick_params(axis="y", which="both", left=False, labelleft=False)

        # Include horizontal error bars in axis limits so caps stay inside the plot area.
        fog_upper_with_err = float(np.nanmax(fog_vals + fog_xerr))
        x_right = max(1.4, float(np.nanmax(fog_vals)) * 1.16, fog_upper_with_err * 1.08)
        ax_fog.set_xlim(0.0, x_right)
        native_upper_with_err = float(np.nanmax(native_vals + native_xerr))
        native_right = max(
            float(theta) * 1.25,
            float(np.nanmax(native_vals)) * 1.15,
            native_upper_with_err * 1.08,
            1.05,
        )
        ax_native.set_xlim(0.0, native_right)

        feasible_n = int(np.sum(feasible_vals > 0))
        fig.suptitle(
            rf"Feasibility-constrained FoG ranking ({feasible_n}/{len(df)} feasible, $U_{{0}}\geq\theta$, $\theta={float(theta):.2f}$)",
            y=0.995,
            fontsize=7.0,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985], pad=0.35)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_maina_native_vs_fog(
    run_df: pd.DataFrame,
    *,
    theta: float,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """
    MainA: native_0 vs FoG scatter with feasibility gate.
    Feasible points: filled circles. Infeasible: open circles.
    """
    df = run_df.copy()
    df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
    ref_norm = _normalize_polymer_id_token(reference_polymer_id)
    df = df[df["polymer_id"].map(lambda x: _normalize_polymer_id_token(x) != ref_norm)].copy()
    if df.empty:
        return False
    x_col = "native_0" if "native_0" in df.columns else "native_activity_rel_at_0"
    x_sem_col = "native_activity_rel_at_0_sem" if "native_activity_rel_at_0_sem" in df.columns else ""
    df[x_col] = pd.to_numeric(df.get(x_col, np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    df["_fog_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    if x_sem_col:
        df["_x_sem"] = pd.to_numeric(df.get(x_sem_col, np.nan), errors="coerce")
    else:
        df["_x_sem"] = np.nan
    fe = pd.to_numeric(df.get("native_activity_feasible", np.nan), errors="coerce")
    if np.isfinite(fe).any():
        df["_feasible"] = fe.fillna(0).astype(int) > 0
    else:
        df["_feasible"] = np.isfinite(df[x_col]) & (df[x_col] >= float(theta))
    df = df[np.isfinite(df[x_col]) & np.isfinite(df["fog"]) & (df["fog"] > 0)].copy()
    if df.empty:
        return False

    cmap = color_map or {}
    default_color = "#4C78A8"
    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt

    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for _, row in df.iterrows():
            pid = str(row.get("polymer_id", "")).strip()
            color = cmap.get(pid, default_color)
            xi = float(row[x_col])
            yi = float(row["fog"])
            xi_sem = float(row.get("_x_sem", np.nan))
            yi_sem = float(row.get("_fog_sem", np.nan))
            if (np.isfinite(xi_sem) and xi_sem > 0.0) or (np.isfinite(yi_sem) and yi_sem > 0.0):
                ax.errorbar(
                    xi,
                    yi,
                    xerr=(xi_sem if np.isfinite(xi_sem) and xi_sem > 0.0 else None),
                    yerr=(yi_sem if np.isfinite(yi_sem) and yi_sem > 0.0 else None),
                    fmt="none",
                    ecolor=PAPER_ERRORBAR_COLOR,
                    elinewidth=0.75,
                    capsize=1.8,
                    alpha=0.8,
                    zorder=16,
                )
            if bool(row["_feasible"]):
                ax.scatter(
                    xi,
                    yi,
                    s=30,
                    marker="o",
                    color=color,
                    edgecolors="0.2",
                    linewidths=0.5,
                    alpha=0.9,
                    zorder=20,
                )
            else:
                ax.scatter(
                    xi,
                    yi,
                    s=30,
                    marker="o",
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.9,
                    alpha=0.8,
                    zorder=18,
                )
        ax.axvline(x=float(theta), color="0.30", linestyle=(0, (3, 2)), linewidth=0.8, zorder=10)
        ax.axhline(y=1.0, color="0.45", linestyle=(0, (3, 2)), linewidth=0.7, zorder=10)
        ax.set_xlabel(rf"$U_{{0}}$ (vs {reference_polymer_id} at 0 min)")
        ax.set_ylabel(r"$\mathrm{FoG}=t_{50}/t_{50,\mathrm{ref}}$")
        ax.set_title("MainA: native gate and FoG")
        ax.grid(True, linestyle=":", alpha=0.30)
        ax.set_xlim(0.0, max(float(theta) * 1.35, float(np.nanmax(df[x_col])) * 1.12, 1.05))
        ax.set_ylim(0.0, max(1.5, float(np.nanmax(df["fog"])) * 1.12))
        _add_scatter_polymer_mapping_legend(ax, df, color_map=cmap, feasible_col="_feasible")
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_mainb_feasible_ranking(
    run_df: pd.DataFrame,
    *,
    theta: float,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """MainB: feasible-only FoG ranking bar chart."""
    df = run_df.copy()
    df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
    ref_norm = _normalize_polymer_id_token(reference_polymer_id)
    df = df[df["polymer_id"].map(lambda x: _normalize_polymer_id_token(x) != ref_norm)].copy()
    if df.empty:
        return False
    x_col = "native_0" if "native_0" in df.columns else "native_activity_rel_at_0"
    df[x_col] = pd.to_numeric(df.get(x_col, np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    df["fog_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    fe = pd.to_numeric(df.get("native_activity_feasible", np.nan), errors="coerce")
    if np.isfinite(fe).any():
        df["_feasible"] = fe.fillna(0).astype(int) > 0
    else:
        df["_feasible"] = np.isfinite(df[x_col]) & (df[x_col] >= float(theta))
    df = df[df["_feasible"] & np.isfinite(df["fog"]) & (df["fog"] > 0)].copy()
    if df.empty:
        return False
    rank = (
        df.groupby("polymer_id", as_index=False)
        .agg(
            fog_median=("fog", "median"),
            fog_sem_median=("fog_sem", "median"),
            native0_median=(x_col, "median"),
            n_rows=("polymer_id", "size"),
        )
        .sort_values("fog_median", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    rank["rank"] = np.arange(1, len(rank) + 1, dtype=int)
    from gox_plate_pipeline.fitting import apply_paper_style, PAPER_ERRORBAR_COLOR
    import matplotlib.pyplot as plt
    cmap = color_map or {}
    default_color = "#4C78A8"
    labels = [str(pid) for pid in rank["polymer_id"]]
    colors = [cmap.get(str(pid), default_color) for pid in rank["polymer_id"]]
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(5.2, max(2.2, 0.30 * len(rank) + 0.9)))
        y = np.arange(len(rank), dtype=float)
        if "fog_sem_median" in rank.columns:
            fog_err = pd.to_numeric(rank["fog_sem_median"], errors="coerce").to_numpy(dtype=float)
        else:
            fog_err = np.full(len(rank), np.nan, dtype=float)
        fog_err = np.where(np.isfinite(fog_err) & (fog_err > 0.0), fog_err, 0.0)
        bar_kwargs = {}
        if np.any(fog_err > 0.0):
            bar_kwargs["xerr"] = fog_err
            bar_kwargs["error_kw"] = {
                "ecolor": PAPER_ERRORBAR_COLOR,
                "elinewidth": 0.8,
                "capsize": 2.0,
                "capthick": 0.8,
                "alpha": 0.85,
            }
        ax.barh(
            y,
            rank["fog_median"].to_numpy(dtype=float),
            color=colors,
            edgecolor="0.2",
            linewidth=0.4,
            height=0.62,
            **bar_kwargs,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.axvline(x=1.0, color="0.45", linestyle=(0, (3, 2)), linewidth=0.75)
        ax.set_xlabel(r"$\mathrm{FoG}$ median among feasible")
        ax.set_title("MainB: feasible-only ranking")
        ax.grid(axis="x", linestyle=":", alpha=0.30)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def _plot_theta_sensitivity_summary(
    run_df: pd.DataFrame,
    *,
    theta_values: List[float],
    out_path: Path,
    csv_out_path: Optional[Path] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> bool:
    """Supplementary: theta sensitivity (feasible n and top-k overlap)."""
    df = run_df.copy()
    df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
    ref_norm = _normalize_polymer_id_token(reference_polymer_id)
    df = df[df["polymer_id"].map(lambda x: _normalize_polymer_id_token(x) != ref_norm)].copy()
    if df.empty:
        return False
    x_col = "native_0" if "native_0" in df.columns else "native_activity_rel_at_0"
    df[x_col] = pd.to_numeric(df.get(x_col, np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    df = df[np.isfinite(df[x_col]) & np.isfinite(df["fog"]) & (df["fog"] > 0)].copy()
    if df.empty:
        return False
    values = sorted(set(float(v) for v in theta_values if np.isfinite(v) and float(v) > 0.0))
    if not values:
        values = [0.60, 0.70, 0.75]
    rows: List[Dict[str, Any]] = []
    top_sets: Dict[float, set[str]] = {}
    for th in values:
        sub = df[df[x_col] >= float(th)].copy()
        if sub.empty:
            rows.append({"theta": float(th), "feasible_polymers": 0})
            top_sets[float(th)] = set()
            continue
        agg = (
            sub.groupby("polymer_id", as_index=False)
            .agg(fog_median=("fog", "median"))
            .sort_values("fog_median", ascending=False, kind="mergesort")
            .reset_index(drop=True)
        )
        top_sets[float(th)] = set(agg["polymer_id"].head(5).astype(str).tolist())
        rows.append({"theta": float(th), "feasible_polymers": int(len(agg))})
    sens = pd.DataFrame(rows)
    base_theta = min(values, key=lambda x: abs(float(x) - 0.70))
    base_set = top_sets.get(float(base_theta), set())
    sens["top5_overlap_vs_theta70"] = [
        float(len(top_sets.get(float(th), set()) & base_set)) / max(1, len(base_set))
        for th in sens["theta"].tolist()
    ]
    # write companion csv
    sens_out = Path(csv_out_path) if csv_out_path is not None else out_path.with_suffix(".csv")
    sens_out.parent.mkdir(parents=True, exist_ok=True)
    sens.to_csv(sens_out, index=False)
    from gox_plate_pipeline.fitting import apply_paper_style
    import matplotlib.pyplot as plt
    with plt.rc_context(apply_paper_style()):
        fig, (ax_n, ax_o) = plt.subplots(1, 2, figsize=(7.0, 2.8))
        ax_n.plot(sens["theta"], sens["feasible_polymers"], marker="o", linewidth=1.0, color="#1f77b4")
        ax_n.set_xlabel("theta")
        ax_n.set_ylabel("Feasible polymers (n)")
        ax_n.set_title("Supp: feasible count")
        ax_n.grid(True, linestyle=":", alpha=0.30)
        ax_o.plot(sens["theta"], sens["top5_overlap_vs_theta70"], marker="o", linewidth=1.0, color="#2ca02c")
        ax_o.set_xlabel("theta")
        ax_o.set_ylabel(r"Top5 overlap vs $\theta=0.70$")
        ax_o.set_ylim(0.0, 1.05)
        ax_o.set_title("Supp: top-k stability")
        ax_o.grid(True, linestyle=":", alpha=0.30)
        fig.tight_layout(pad=0.3)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def write_run_ranking_outputs(
    fog_df: pd.DataFrame,
    run_id: str,
    out_dir: Path,
    *,
    color_map_path: Optional[Path] = None,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> Dict[str, Path]:
    """
    Write per-run ranking CSVs and bar charts for t50/FoG/functional.

    Outputs:
      - CSV: out_dir/csv/
        - t50_ranking__{run_id}.csv
        - fog_ranking__{run_id}.csv
        - fog_native_constrained_ranking__{run_id}.csv
        - functional_ranking__{run_id}.csv
      - t50_ranking__{run_id}.png (if plottable rows exist)
      - fog_ranking__{run_id}.png (if plottable rows exist)
      - fog_native_constrained_ranking__{run_id}.png (if plottable rows exist)
      - fog_native_constrained_decision__{run_id}.png (main: native gate + FoG ranking, if plottable)
      - fog_native_constrained_tradeoff__{run_id}.png (supplementary scatter, if plottable)
      - mainA_native0_vs_fog__{run_id}.png (paper main scatter)
      - mainB_feasible_fog_ranking__{run_id}.png (paper main feasible ranking)
      - supp_theta_sensitivity__{run_id}.png + out_dir/csv/supp_theta_sensitivity__{run_id}.csv
      - out_dir/csv/primary_objective_table__{run_id}.csv (native_0, FoG, t50, t_theta, QC flags)
      - functional_ranking__{run_id}.png (if plottable rows exist)

    Ranking score applies an absolute-activity guard when available:
      activity_weight = clip(abs_activity_at_0 / GOx_abs_activity_at_0, 0, 1)
      t50_activity_adjusted_min = t50_min * activity_weight
      fog_activity_adjusted = fog * activity_weight
    """
    run_id = str(run_id).strip()
    reference_polymer_id = str(reference_polymer_id).strip() or DEFAULT_REFERENCE_POLYMER_ID
    reference_polymer_norm = _normalize_polymer_id_token(reference_polymer_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = _ensure_csv_subdir(out_dir)

    df = fog_df.copy()
    if "run_id" not in df.columns:
        df["run_id"] = run_id
    df["run_id"] = df["run_id"].astype(str)
    df = df[df["run_id"] == run_id].copy()
    if df.empty:
        # Keep reproducible empty outputs when no rows are available.
        df = pd.DataFrame(columns=["run_id", "polymer_id", "t50_min", "t50_censored", "fog", "fog_missing_reason"])
    if "reference_polymer_id" in df.columns:
        ref_col = df["reference_polymer_id"].astype(str).str.strip()
        ref_col = ref_col.where(ref_col != "", reference_polymer_id)
        df["reference_polymer_id"] = ref_col
    else:
        df["reference_polymer_id"] = reference_polymer_id

    df["t50_min"] = pd.to_numeric(df.get("t50_min", np.nan), errors="coerce")
    df["t50_sem_min"] = pd.to_numeric(df.get("t50_sem_min", np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    df["fog_sem"] = pd.to_numeric(df.get("fog_sem", np.nan), errors="coerce")
    df["fog_native_constrained"] = pd.to_numeric(df.get("fog_native_constrained", np.nan), errors="coerce")
    df["fog_native_constrained_sem"] = pd.to_numeric(
        df.get("fog_native_constrained_sem", np.nan), errors="coerce"
    )
    df["abs_activity_at_0"] = pd.to_numeric(df.get("abs_activity_at_0", np.nan), errors="coerce")
    df["gox_abs_activity_at_0_ref"] = pd.to_numeric(df.get("gox_abs_activity_at_0_ref", np.nan), errors="coerce")
    df["gox_abs_activity_at_0"] = pd.to_numeric(df.get("gox_abs_activity_at_0", np.nan), errors="coerce")
    df["native_activity_rel_at_0"] = pd.to_numeric(df.get("native_activity_rel_at_0", np.nan), errors="coerce")
    df["native_activity_min_rel_threshold"] = pd.to_numeric(
        df.get("native_activity_min_rel_threshold", np.nan), errors="coerce"
    )
    if "fog_constraint_reason" in df.columns:
        df["fog_constraint_reason"] = df["fog_constraint_reason"].astype(str)
    else:
        df["fog_constraint_reason"] = ""
    df["abs_activity_at_20"] = pd.to_numeric(df.get("abs_activity_at_20", np.nan), errors="coerce")
    df["gox_abs_activity_at_20_ref"] = pd.to_numeric(df.get("gox_abs_activity_at_20_ref", np.nan), errors="coerce")
    df["functional_activity_at_20_rel"] = pd.to_numeric(df.get("functional_activity_at_20_rel", np.nan), errors="coerce")
    df["functional_activity_at_20_rel_sem"] = pd.to_numeric(
        df.get("functional_activity_at_20_rel_sem", np.nan), errors="coerce"
    )
    if "functional_reference_source" in df.columns:
        df["functional_reference_source"] = df["functional_reference_source"].astype(str)
    if "t50_censored" in df.columns:
        df["t50_censored"] = pd.to_numeric(df["t50_censored"], errors="coerce").fillna(1).astype(int)
    gox_abs_activity_at_0 = np.nan
    ref_mask = df["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) == reference_polymer_norm)
    if np.any(ref_mask):
        ref_vals = df.loc[ref_mask, "abs_activity_at_0"]
        ref_vals = ref_vals[np.isfinite(ref_vals) & (ref_vals > 0)]
        if not ref_vals.empty:
            gox_abs_activity_at_0 = float(ref_vals.iloc[0])
    if np.isfinite(gox_abs_activity_at_0) and gox_abs_activity_at_0 > 0:
        df["gox_abs_activity_at_0"] = np.where(
            np.isfinite(df["gox_abs_activity_at_0"]),
            df["gox_abs_activity_at_0"],
            float(gox_abs_activity_at_0),
        )
    else:
        df["gox_abs_activity_at_0"] = np.where(
            np.isfinite(df["gox_abs_activity_at_0"]),
            df["gox_abs_activity_at_0"],
            df["gox_abs_activity_at_0_ref"],
        )

    abs0_vs_gox = np.where(
        np.isfinite(df["abs_activity_at_0"])
        & np.isfinite(df["gox_abs_activity_at_0"])
        & (df["gox_abs_activity_at_0"] > 0.0),
        df["abs_activity_at_0"] / df["gox_abs_activity_at_0"],
        np.nan,
    )
    abs0_vs_gox = pd.to_numeric(abs0_vs_gox, errors="coerce")
    activity_weight = np.clip(abs0_vs_gox, 0.0, 1.0)
    activity_weight = np.where(np.isfinite(activity_weight), activity_weight, 1.0)
    df["abs0_vs_gox"] = abs0_vs_gox
    df["activity_weight"] = activity_weight
    df["t50_activity_adjusted_min"] = df["t50_min"] * df["activity_weight"]
    df["t50_activity_adjusted_sem_min"] = np.where(
        np.isfinite(df["t50_sem_min"]),
        df["t50_sem_min"] * df["activity_weight"],
        np.nan,
    )
    df["fog_activity_adjusted"] = df["fog"] * df["activity_weight"]
    df["fog_activity_adjusted_sem"] = np.where(
        np.isfinite(df["fog_sem"]),
        df["fog_sem"] * df["activity_weight"],
        np.nan,
    )

    # Native-activity feasibility and constrained FoG objective.
    native_thr = pd.to_numeric(df.get("native_activity_min_rel_threshold", np.nan), errors="coerce")
    if not np.isfinite(native_thr).any():
        native_thr = np.full(len(df), float(NATIVE_ACTIVITY_MIN_REL_DEFAULT), dtype=float)
    else:
        fallback_thr = float(np.nanmedian(native_thr[np.isfinite(native_thr)]))
        native_thr = np.where(np.isfinite(native_thr), native_thr, fallback_thr)
    df["native_activity_min_rel_threshold"] = native_thr
    rel = pd.to_numeric(df.get("native_activity_rel_at_0", np.nan), errors="coerce")
    if not np.isfinite(rel).any():
        rel = pd.to_numeric(df.get("abs0_vs_gox", np.nan), errors="coerce")
    df["native_activity_rel_at_0"] = rel
    native_feasible = np.isfinite(rel) & np.isfinite(native_thr) & (rel >= native_thr)
    df["native_activity_feasible"] = native_feasible.astype(int)
    fog_native = pd.to_numeric(df.get("fog_native_constrained", np.nan), errors="coerce")
    fog_native_fallback = np.where(native_feasible, df["fog"], np.nan)
    df["fog_native_constrained"] = np.where(np.isfinite(fog_native), fog_native, fog_native_fallback)

    # Fill functional metric from abs20/reference if missing.
    fa20 = pd.to_numeric(df.get("functional_activity_at_20_rel", np.nan), errors="coerce")
    fa20_fallback = np.where(
        np.isfinite(df["abs_activity_at_20"]) & np.isfinite(df["gox_abs_activity_at_20_ref"]) & (df["gox_abs_activity_at_20_ref"] > 0),
        df["abs_activity_at_20"] / df["gox_abs_activity_at_20_ref"],
        np.nan,
    )
    df["functional_activity_at_20_rel"] = np.where(np.isfinite(fa20), fa20, fa20_fallback)

    # t50 ranking table
    t50_cols = [
        "run_id",
        "polymer_id",
        "reference_polymer_id",
        "t50_min",
        "t50_sem_min",
        "t50_activity_adjusted_min",
        "t50_activity_adjusted_sem_min",
        "t50_censored",
        "t50_definition",
        "t50_target_rea_percent",
        "rea_at_20_percent",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "abs0_vs_gox",
        "activity_weight",
        "use_for_bo",
    ]
    t50_available = [c for c in t50_cols if c in df.columns]
    t50_tbl = df[t50_available].copy()
    t50_tbl["rank_t50"] = np.nan
    t50_valid = t50_tbl[
        np.isfinite(t50_tbl["t50_min"])
        & (t50_tbl["t50_min"] > 0)
        & np.isfinite(t50_tbl["t50_activity_adjusted_min"])
        & (t50_tbl["t50_activity_adjusted_min"] > 0)
    ].copy()
    if not t50_valid.empty:
        t50_valid = t50_valid.sort_values(
            ["t50_activity_adjusted_min", "t50_min"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        t50_valid["rank_t50"] = np.arange(1, len(t50_valid) + 1, dtype=int)
        t50_tbl = t50_valid
    t50_out = csv_dir / f"t50_ranking__{run_id}.csv"
    t50_tbl.to_csv(t50_out, index=False)
    _remove_legacy_csv(out_dir / f"t50_ranking__{run_id}.csv", t50_out)

    # FoG ranking table
    fog_cols = [
        "run_id",
        "polymer_id",
        "reference_polymer_id",
        "fog",
        "fog_sem",
        "fog_activity_adjusted",
        "fog_activity_adjusted_sem",
        "log_fog",
        "fog_missing_reason",
        "gox_t50_same_run_min",
        "t50_min",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "abs0_vs_gox",
        "activity_weight",
        "t50_censored",
        "use_for_bo",
    ]
    fog_available = [c for c in fog_cols if c in df.columns]
    fog_tbl = df[fog_available].copy()
    fog_tbl["rank_fog"] = np.nan
    fog_valid = fog_tbl[
        np.isfinite(fog_tbl["fog"])
        & (fog_tbl["fog"] > 0)
        & np.isfinite(fog_tbl["fog_activity_adjusted"])
        & (fog_tbl["fog_activity_adjusted"] > 0)
    ].copy()
    if not fog_valid.empty:
        fog_valid = fog_valid.sort_values(
            ["fog_activity_adjusted", "fog"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        fog_valid["rank_fog"] = np.arange(1, len(fog_valid) + 1, dtype=int)
        fog_tbl = fog_valid
    fog_out = csv_dir / f"fog_ranking__{run_id}.csv"
    fog_tbl.to_csv(fog_out, index=False)
    _remove_legacy_csv(out_dir / f"fog_ranking__{run_id}.csv", fog_out)

    # Native-constrained FoG ranking table
    fog_constrained_cols = [
        "run_id",
        "polymer_id",
        "reference_polymer_id",
        "fog_native_constrained",
        "fog_native_constrained_sem",
        "fog",
        "native_activity_rel_at_0",
        "native_activity_min_rel_threshold",
        "native_activity_feasible",
        "fog_constraint_reason",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "abs0_vs_gox",
        "t50_min",
        "use_for_bo",
    ]
    fog_constrained_available = [c for c in fog_constrained_cols if c in df.columns]
    fog_constrained_tbl = df[fog_constrained_available].copy()
    fog_constrained_tbl["rank_fog_native_constrained"] = np.nan
    fog_constrained_valid = fog_constrained_tbl[
        np.isfinite(fog_constrained_tbl["fog_native_constrained"])
        & (fog_constrained_tbl["fog_native_constrained"] > 0)
    ].copy()
    if not fog_constrained_valid.empty:
        fog_constrained_valid = fog_constrained_valid.sort_values(
            ["fog_native_constrained", "fog"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        fog_constrained_valid["rank_fog_native_constrained"] = np.arange(
            1, len(fog_constrained_valid) + 1, dtype=int
        )
        fog_constrained_tbl = fog_constrained_valid
    fog_constrained_out = csv_dir / f"fog_native_constrained_ranking__{run_id}.csv"
    fog_constrained_tbl.to_csv(fog_constrained_out, index=False)
    _remove_legacy_csv(out_dir / f"fog_native_constrained_ranking__{run_id}.csv", fog_constrained_out)

    # Functional (20 min) ranking table
    func_cols = [
        "run_id",
        "polymer_id",
        "reference_polymer_id",
        "functional_activity_at_20_rel",
        "functional_activity_at_20_rel_sem",
        "abs_activity_at_20",
        "gox_abs_activity_at_20_ref",
        "functional_reference_source",
        "functional_reference_round_id",
        "functional_reference_run_id",
        "abs_activity_at_0",
        "gox_abs_activity_at_0_ref",
        "gox_abs_activity_at_0",
        "abs0_vs_gox",
        "use_for_bo",
    ]
    func_available = [c for c in func_cols if c in df.columns]
    func_tbl = df[func_available].copy()
    func_tbl["rank_functional"] = np.nan
    func_valid = func_tbl[
        np.isfinite(func_tbl["functional_activity_at_20_rel"])
        & (func_tbl["functional_activity_at_20_rel"] > 0)
    ].copy()
    if not func_valid.empty:
        func_valid = func_valid.sort_values(
            ["functional_activity_at_20_rel", "abs_activity_at_20"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)
        func_valid["rank_functional"] = np.arange(1, len(func_valid) + 1, dtype=int)
        func_tbl = func_valid
    func_out = csv_dir / f"functional_ranking__{run_id}.csv"
    func_tbl.to_csv(func_out, index=False)
    _remove_legacy_csv(out_dir / f"functional_ranking__{run_id}.csv", func_out)

    cmap = _load_polymer_color_map(color_map_path)
    t50_png = out_dir / f"t50_ranking__{run_id}.png"
    fog_png = out_dir / f"fog_ranking__{run_id}.png"
    fog_constrained_png = out_dir / f"fog_native_constrained_ranking__{run_id}.png"
    fog_decision_png = out_dir / f"fog_native_constrained_decision__{run_id}.png"
    func_png = out_dir / f"functional_ranking__{run_id}.png"
    wrote_t50_png = _plot_run_ranking_bar(
        t50_tbl,
        value_col="t50_activity_adjusted_min",
        error_col="t50_activity_adjusted_sem_min",
        rank_col="rank_t50",
        title=r"$t_{50}$ ranking (activity-adjusted)",
        xlabel=r"Activity-adjusted $t_{50}$ (min)",
        out_path=t50_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    wrote_fog_png = _plot_run_ranking_bar(
        fog_tbl,
        value_col="fog_activity_adjusted",
        error_col="fog_activity_adjusted_sem",
        rank_col="rank_fog",
        title="FoG ranking (activity-adjusted)",
        xlabel="activity-adjusted FoG",
        out_path=fog_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    wrote_fog_constrained_png = _plot_run_ranking_bar(
        fog_constrained_tbl,
        value_col="fog_native_constrained",
        error_col="fog_native_constrained_sem",
        rank_col="rank_fog_native_constrained",
        title=r"FoG ranking ($U_{0}\geq\theta$ constrained)",
        xlabel=r"$\mathrm{FoG}$ ($U_{0}\geq\theta$ constrained)",
        out_path=fog_constrained_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    wrote_func_png = _plot_run_ranking_bar(
        func_tbl,
        value_col="functional_activity_at_20_rel",
        error_col="functional_activity_at_20_rel_sem",
        rank_col="rank_functional",
        title="Functional ranking (20 min)",
        xlabel=f"functional activity ratio at 20 min (vs {reference_polymer_id})",
        out_path=func_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_t50_png and t50_png.exists():
        t50_png.unlink(missing_ok=True)
    if not wrote_fog_png and fog_png.exists():
        fog_png.unlink(missing_ok=True)
    if not wrote_fog_constrained_png and fog_constrained_png.exists():
        fog_constrained_png.unlink(missing_ok=True)
    if not wrote_func_png and func_png.exists():
        func_png.unlink(missing_ok=True)

    # Native-gated decision figure (main) + trade-off scatter (supplementary).
    theta = float(np.nanmedian(df["native_activity_min_rel_threshold"])) if "native_activity_min_rel_threshold" in df.columns and np.any(np.isfinite(df["native_activity_min_rel_threshold"])) else NATIVE_ACTIVITY_MIN_REL_DEFAULT
    wrote_decision = _plot_fog_native_constrained_decision(
        df,
        theta=theta,
        out_path=fog_decision_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_decision and fog_decision_png.exists():
        fog_decision_png.unlink(missing_ok=True)

    tradeoff_png = out_dir / f"fog_native_constrained_tradeoff__{run_id}.png"
    wrote_tradeoff = _plot_fog_native_constrained_tradeoff(
        df,
        theta=theta,
        out_path=tradeoff_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_tradeoff and tradeoff_png.exists():
        tradeoff_png.unlink(missing_ok=True)

    # Additional paper-oriented main figure set (v2).
    maina_png = out_dir / f"mainA_native0_vs_fog__{run_id}.png"
    mainb_png = out_dir / f"mainB_feasible_fog_ranking__{run_id}.png"
    supp_theta_png = out_dir / f"supp_theta_sensitivity__{run_id}.png"
    supp_theta_csv = csv_dir / f"supp_theta_sensitivity__{run_id}.csv"
    wrote_maina = _plot_maina_native_vs_fog(
        df,
        theta=theta,
        out_path=maina_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_maina and maina_png.exists():
        maina_png.unlink(missing_ok=True)
    wrote_mainb = _plot_mainb_feasible_ranking(
        df,
        theta=theta,
        out_path=mainb_png,
        color_map=cmap,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_mainb and mainb_png.exists():
        mainb_png.unlink(missing_ok=True)
    wrote_supp_theta = _plot_theta_sensitivity_summary(
        df,
        theta_values=[0.60, 0.70, 0.75],
        out_path=supp_theta_png,
        csv_out_path=supp_theta_csv,
        reference_polymer_id=reference_polymer_id,
    )
    if not wrote_supp_theta and supp_theta_png.exists():
        supp_theta_png.unlink(missing_ok=True)
    if not wrote_supp_theta and supp_theta_csv.exists():
        supp_theta_csv.unlink(missing_ok=True)
    _remove_legacy_csv(out_dir / f"supp_theta_sensitivity__{run_id}.csv", supp_theta_csv)

    # Primary objective table for quick PI/reviewer reading.
    ref_norm = _normalize_polymer_id_token(reference_polymer_id)
    primary_tbl = df[df["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) != ref_norm)].copy()
    if "native_0" not in primary_tbl.columns and "native_activity_rel_at_0" in primary_tbl.columns:
        primary_tbl["native_0"] = pd.to_numeric(primary_tbl.get("native_activity_rel_at_0", np.nan), errors="coerce")
    keep_cols = [
        "run_id",
        "polymer_id",
        "native_0",
        "native_activity_rel_at_0",
        "native_activity_min_rel_threshold",
        "native_activity_feasible",
        "fog",
        "fog_native_constrained",
        "t50_min",
        "t_theta",
        "t_theta_censor_flag",
        "reference_qc_fail",
        "reference_qc_reason",
    ]
    keep = [c for c in keep_cols if c in primary_tbl.columns]
    primary_tbl = primary_tbl[keep].copy()
    if "fog_native_constrained" in primary_tbl.columns:
        primary_tbl = primary_tbl.sort_values(
            ["native_activity_feasible", "fog_native_constrained", "fog"],
            ascending=[False, False, False],
            kind="mergesort",
        )
    primary_tbl_out = csv_dir / f"primary_objective_table__{run_id}.csv"
    primary_tbl.to_csv(primary_tbl_out, index=False)
    _remove_legacy_csv(out_dir / f"primary_objective_table__{run_id}.csv", primary_tbl_out)

    outputs: Dict[str, Path] = {
        "t50_ranking_csv": t50_out,
        "fog_ranking_csv": fog_out,
        "fog_native_constrained_ranking_csv": fog_constrained_out,
        "functional_ranking_csv": func_out,
        "primary_objective_table_csv": primary_tbl_out,
    }
    if wrote_t50_png:
        outputs["t50_ranking_png"] = t50_png
    if wrote_fog_png:
        outputs["fog_ranking_png"] = fog_png
    if wrote_fog_constrained_png:
        outputs["fog_native_constrained_ranking_png"] = fog_constrained_png
    if wrote_decision:
        outputs["fog_native_constrained_decision_png"] = fog_decision_png
    if wrote_tradeoff:
        outputs["fog_native_constrained_tradeoff_png"] = tradeoff_png
    if wrote_maina:
        outputs["mainA_native0_vs_fog_png"] = maina_png
    if wrote_mainb:
        outputs["mainB_feasible_fog_ranking_png"] = mainb_png
    if wrote_supp_theta:
        outputs["supp_theta_sensitivity_png"] = supp_theta_png
        outputs["supp_theta_sensitivity_csv"] = supp_theta_csv
    if wrote_func_png:
        outputs["functional_ranking_png"] = func_png
    return outputs


def build_round_averaged_fog(
    run_round_map: Dict[str, str],
    processed_dir: Path,
    *,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> pd.DataFrame:
    """
    Build round-averaged FoG from per-run fog_summary CSVs.

    - FoG is currently computed per run (same-run GOx as denominator). This function
      averages FoG by (round_id, polymer_id) across all runs in that round.
    - GOx row is excluded from the output. Only rows with finite fog and fog > 0 are averaged.
    - If a round has no run with GOx (all runs have gox_t50_same_run_min NaN), raises ValueError.
    - Output columns:
      round_id, polymer_id, mean_fog, mean_log_fog, robust_fog, robust_log_fog,
      log_fog_mad, mean_fog_native_constrained, mean_log_fog_native_constrained,
      robust_fog_native_constrained, robust_log_fog_native_constrained,
      log_fog_native_constrained_mad, native_feasible_fraction, n_observations, run_ids.
    """
    processed_dir = Path(processed_dir)
    reference_polymer_id = str(reference_polymer_id).strip() or DEFAULT_REFERENCE_POLYMER_ID
    reference_polymer_norm = _normalize_polymer_id_token(reference_polymer_id)
    # round_id -> list of run_ids
    round_to_runs: Dict[str, List[str]] = {}
    for rid, oid in run_round_map.items():
        rid, oid = str(rid).strip(), str(oid).strip()
        if not oid:
            continue
        round_to_runs.setdefault(oid, []).append(rid)

    if not round_to_runs:
        return pd.DataFrame(
            columns=[
                "round_id",
                "polymer_id",
                "reference_polymer_id",
                "mean_fog",
                "mean_log_fog",
                "robust_fog",
                "robust_log_fog",
                "log_fog_mad",
                "mean_fog_native_constrained",
                "mean_log_fog_native_constrained",
                "robust_fog_native_constrained",
                "robust_log_fog_native_constrained",
                "log_fog_native_constrained_mad",
                "native_feasible_fraction",
                "n_observations",
                "run_ids",
            ]
        )

    rows: List[dict] = []
    for round_id, run_ids in sorted(round_to_runs.items()):
        run_ids = sorted(run_ids)
        # Load fog_summary for each run in this round
        dfs: List[pd.DataFrame] = []
        round_has_reference = False
        for rid in run_ids:
            path = processed_dir / rid / "fit" / f"fog_summary__{rid}.csv"
            if not path.is_file():
                continue
            df = pd.read_csv(path)
            df["run_id"] = df.get("run_id", rid)
            df["polymer_id"] = df["polymer_id"].astype(str).str.strip()
            gox_t50 = df.get("gox_t50_same_run_min", pd.Series(dtype=float))
            if gox_t50.notna().any() and np.isfinite(gox_t50).any():
                round_has_reference = True
            dfs.append(df)

        if not dfs:
            raise ValueError(f"Round {round_id!r} has no fog_summary CSV under processed_dir for runs {run_ids}.")
        if not round_has_reference:
            raise ValueError(
                f"Round {round_id!r} has no reference polymer {reference_polymer_id!r} in any run "
                "(all runs have no bare reference polymer). "
                "FoG cannot be computed for this round."
            )

        combined = pd.concat(dfs, ignore_index=True)
        # Exclude reference-polymer row.
        combined = combined[
            combined["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) != reference_polymer_norm)
        ].copy()
        # Only valid fog
        fog = pd.to_numeric(combined.get("fog", np.nan), errors="coerce")
        log_fog = pd.to_numeric(combined.get("log_fog", np.nan), errors="coerce")
        valid = np.isfinite(fog) & (fog > 0)
        combined = combined.loc[valid].copy()
        combined["_fog"] = fog[valid].values
        combined["_log_fog"] = log_fog[valid].values
        fog_native = pd.to_numeric(combined.get("fog_native_constrained", np.nan), errors="coerce")
        log_fog_native = pd.to_numeric(combined.get("log_fog_native_constrained", np.nan), errors="coerce")
        combined["_fog_native"] = fog_native
        combined["_log_fog_native"] = log_fog_native
        native_feasible = pd.to_numeric(combined.get("native_activity_feasible", np.nan), errors="coerce")
        if not np.isfinite(native_feasible).any():
            native_feasible = np.isfinite(fog_native).astype(float)
        combined["_native_feasible"] = np.where(np.isfinite(native_feasible), native_feasible, 0.0)

        for polymer_id, g in combined.groupby("polymer_id", sort=False):
            pid = str(polymer_id).strip()
            mean_fog = float(g["_fog"].mean())
            mean_log_fog = float(g["_log_fog"].mean())
            robust_fog = float(np.nanmedian(g["_fog"]))
            robust_log_fog = float(np.nanmedian(g["_log_fog"]))
            log_fog_mad = float(np.nanmedian(np.abs(g["_log_fog"] - robust_log_fog)))
            g_native = g[np.isfinite(g["_fog_native"]) & (g["_fog_native"] > 0)].copy()
            if not g_native.empty:
                mean_fog_native = float(g_native["_fog_native"].mean())
                mean_log_fog_native = float(g_native["_log_fog_native"].mean())
                robust_fog_native = float(np.nanmedian(g_native["_fog_native"]))
                robust_log_fog_native = float(np.nanmedian(g_native["_log_fog_native"]))
                log_fog_native_mad = float(
                    np.nanmedian(np.abs(g_native["_log_fog_native"] - robust_log_fog_native))
                )
            else:
                mean_fog_native = np.nan
                mean_log_fog_native = np.nan
                robust_fog_native = np.nan
                robust_log_fog_native = np.nan
                log_fog_native_mad = np.nan
            native_feasible_fraction = float(np.nanmean(np.clip(g["_native_feasible"], 0.0, 1.0)))
            n_obs = int(len(g))
            run_list = sorted(g["run_id"].astype(str).unique().tolist())
            rows.append({
                "round_id": round_id,
                "polymer_id": pid,
                "reference_polymer_id": reference_polymer_id,
                "mean_fog": mean_fog,
                "mean_log_fog": mean_log_fog,
                "robust_fog": robust_fog,
                "robust_log_fog": robust_log_fog,
                "log_fog_mad": log_fog_mad,
                "mean_fog_native_constrained": mean_fog_native,
                "mean_log_fog_native_constrained": mean_log_fog_native,
                "robust_fog_native_constrained": robust_fog_native,
                "robust_log_fog_native_constrained": robust_log_fog_native,
                "log_fog_native_constrained_mad": log_fog_native_mad,
                "native_feasible_fraction": native_feasible_fraction,
                "n_observations": n_obs,
                "run_ids": ",".join(run_list),
            })

    return pd.DataFrame(rows)


def build_round_gox_traceability(
    run_round_map: Dict[str, str],
    processed_dir: Path,
    *,
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
) -> pd.DataFrame:
    """
    Build per-round GOx traceability: abs_activity (and REA_percent) by heat_min for each run,
    listing all pre-averaged well-level values so which GOx was used for that round is auditable.

    - Reads rates_with_rea.csv for each run in each round; keeps only rows with polymer_id == GOx.
    - Output columns: round_id, run_id, heat_min, plate_id, well, abs_activity, REA_percent.
    - One row per (round_id, run_id, heat_min, well); no averaging (all raw values for that run).
    """
    processed_dir = Path(processed_dir)
    reference_polymer_id = str(reference_polymer_id).strip() or DEFAULT_REFERENCE_POLYMER_ID
    reference_polymer_norm = _normalize_polymer_id_token(reference_polymer_id)
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
            gox = df[
                df["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) == reference_polymer_norm)
            ].copy()
            if gox.empty:
                continue
            for _, row in gox.iterrows():
                rows.append({
                    "round_id": round_id,
                    "run_id": run_id,
                    "reference_polymer_id": reference_polymer_id,
                    "heat_min": pd.to_numeric(row.get("heat_min", np.nan), errors="coerce"),
                    "plate_id": str(row.get("plate_id", "")),
                    "well": str(row.get("well", "")),
                    "abs_activity": pd.to_numeric(row.get("abs_activity", np.nan), errors="coerce"),
                    "REA_percent": pd.to_numeric(row.get("REA_percent", np.nan), errors="coerce"),
                })
    out_cols = [
        "round_id",
        "run_id",
        "reference_polymer_id",
        "heat_min",
        "plate_id",
        "well",
        "abs_activity",
        "REA_percent",
    ]
    return pd.DataFrame(rows, columns=out_cols)


def compute_per_plate_t50_from_rates(
    rates_df: pd.DataFrame,
    run_id: str,
    *,
    t50_definition: str = "y0_half",
) -> pd.DataFrame:
    """
    Compute t50 per (run_id, plate_id, polymer_id) from rates_with_rea-style DataFrame.

    - Uses only rows with status == 'ok' and finite REA_percent.
    - Aggregates by (plate_id, polymer_id, heat_min) -> mean REA_percent, then fits t50 per (plate_id, polymer_id).
    - Prefers exp/exp_plateau t50; fallback t50_linear.
    - t50_definition: "y0_half" or "rea50".
    - Output: run_id, plate_id, polymer_id, t50_min, n_points, fit_model, t50_definition.
    """
    from gox_plate_pipeline.polymer_timeseries import (
        T50_DEFINITION_REA50,
        fit_exponential_decay,
        normalize_t50_definition,
        t50_linear,
        value_at_time_linear,
    )  # avoid circular import

    t50_definition = normalize_t50_definition(t50_definition)

    df = rates_df.copy()
    df["polymer_id"] = df.get("polymer_id", pd.Series(dtype=str)).astype(str).str.strip()
    df["plate_id"] = df.get("plate_id", pd.Series(dtype=str)).astype(str)
    df["heat_min"] = pd.to_numeric(df.get("heat_min", np.nan), errors="coerce")
    df["REA_percent"] = pd.to_numeric(df.get("REA_percent", np.nan), errors="coerce")
    df["abs_activity"] = pd.to_numeric(df.get("abs_activity", np.nan), errors="coerce")
    df["status"] = df.get("status", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    ok = df[(df["status"] == "ok") & np.isfinite(df["REA_percent"]) & (df["REA_percent"] > 0)].copy()
    if ok.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "plate_id",
                "polymer_id",
                "t50_min",
                "n_points",
                "fit_model",
                "t50_definition",
                "abs_activity_at_0",
            ]
        )

    # Aggregate by (plate_id, polymer_id, heat_min) -> mean REA/absolute activity
    agg = ok.groupby(["plate_id", "polymer_id", "heat_min"], dropna=False).agg(
        REA_percent=("REA_percent", "mean"),
        abs_activity=("abs_activity", "mean"),
    ).reset_index()

    rows: List[dict] = []
    for (plate_id, polymer_id), g in agg.groupby(["plate_id", "polymer_id"], dropna=False):
        g = g.sort_values("heat_min")
        t = g["heat_min"].to_numpy(dtype=float)
        y = g["REA_percent"].to_numpy(dtype=float)
        aa = g["abs_activity"].to_numpy(dtype=float)
        n_pts = int(len(g))
        if t.size < 3 or np.unique(t).size < 2:
            continue
        abs_activity_at_0 = value_at_time_linear(t, aa, at_time_min=0.0)
        abs_activity_at_0 = (
            float(abs_activity_at_0)
            if abs_activity_at_0 is not None and np.isfinite(float(abs_activity_at_0))
            else np.nan
        )
        y0_init = float(np.nanmax(y)) if np.any(np.isfinite(y)) else None
        fit = fit_exponential_decay(
            t,
            y,
            y0=y0_init,
            fixed_y0=100.0,
            min_points=3,
            t50_definition=t50_definition,
        )
        t50_exp = float(fit.t50) if (fit is not None and fit.t50 is not None and np.isfinite(fit.t50)) else np.nan
        y0_lin = 100.0
        t50_lin_val = t50_linear(
            t,
            y,
            y0=y0_lin,
            target_frac=0.5,
            target_value=(50.0 if t50_definition == T50_DEFINITION_REA50 else None),
        )
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
            "t50_definition": t50_definition,
            "abs_activity_at_0": abs_activity_at_0,
        })
    return pd.DataFrame(rows)


def build_fog_plate_aware(
    run_round_map: Dict[str, str],
    processed_dir: Path,
    *,
    t50_definition: str = "y0_half",
    reference_polymer_id: str = DEFAULT_REFERENCE_POLYMER_ID,
    native_activity_min_rel: float = NATIVE_ACTIVITY_MIN_REL_DEFAULT,
    exclude_outlier_gox: bool = False,
    gox_outlier_low_threshold: float = 0.33,
    gox_outlier_high_threshold: float = 3.0,
    gox_guard_same_plate: bool = True,
    gox_guard_low_threshold: Optional[float] = None,
    gox_guard_high_threshold: Optional[float] = None,
    gox_round_fallback_stat: str = "median",
    gox_round_trimmed_mean_proportion: float = 0.1,
    native_reference_mode: str = "same_plate_then_round",
    ref_agg_method: str = "median",
    ref_trimmed_mean_proportion: float = 0.1,
    reference_qc_mad_rel_threshold: float = 0.25,
    reference_qc_min_abs0: float = 0.0,
    reference_qc_exclude: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FogWarningInfo]:
    """
    Build FoG with denominator rule: same plate GOx → same round GOx.

    - Per-plate t50 is computed from rates_with_rea per run (REA vs heat_min per plate).
    - t50_definition controls how per-plate t50 is computed ("y0_half" or "rea50").
    - For each (run_id, plate_id, polymer_id): gox_t50 = GOx t50 from same (run_id, plate_id) if present,
      else mean of GOx t50 across all (run_id, plate_id) in that round. FoG = t50_polymer / gox_t50.
    - Round average GOx t50 calculation: All (run, plate) GOx t50 values are averaged with equal weight
      (run-level weighting is not applied; each plate contributes equally).
    - Outlier detection: GOx t50 values that are < median * gox_outlier_low_threshold or
      > median * gox_outlier_high_threshold are detected and warned. If exclude_outlier_gox=True,
      outliers are excluded from round average GOx t50 calculation.
    - Denominator guard: when gox_guard_same_plate=True, even if same-plate GOx exists, values outside
      [round_median * gox_guard_low_threshold, round_median * gox_guard_high_threshold] are treated as
      unstable and fall back to round representative GOx t50.
    - Round fallback denominator: median (default), mean, or trimmed_mean of round GOx t50 after optional outlier exclusion.
    - If a round has no GOx in any (run, plate), raises ValueError.
    - Returns (per_row_df, round_averaged_df, gox_traceability_df, warning_info).
      per_row_df columns: run_id, plate_id, polymer_id, t50_min, gox_t50_used_min, denominator_source, fog, log_fog,
                          abs_activity_at_0, gox_abs_activity_at_0_ref, native_activity_rel_at_0,
                          native_activity_feasible, fog_native_constrained, log_fog_native_constrained,
                          native_0, U_*, t_theta, t_theta_censor_flag, reference_qc_*.
      warning_info: FogWarningInfo object containing warning details.

    Args:
        exclude_outlier_gox: If True, exclude outlier GOx t50 values from round average calculation.
        gox_outlier_low_threshold: Lower threshold multiplier for outlier detection (default: 0.33).
        gox_outlier_high_threshold: Upper threshold multiplier for outlier detection (default: 3.0).
        gox_guard_same_plate: If True, guard same_plate denominator and fallback to same_round when extreme.
        gox_guard_low_threshold: Lower multiplier for same_plate guard. If None, uses gox_outlier_low_threshold.
        gox_guard_high_threshold: Upper multiplier for same_plate guard. If None, uses gox_outlier_high_threshold.
        gox_round_fallback_stat: Round representative denominator ("median", "mean", or "trimmed_mean").
        gox_round_trimmed_mean_proportion: Proportion to trim from each tail for "trimmed_mean" (default 0.1).
        native_reference_mode: Native/U(t) reference mode:
            "same_plate_then_round" (legacy) or "same_run_then_round" (policy v2 primary).
        ref_agg_method: Aggregation for same-round or same-run reference abs0: "median" (default),
            "trimmed_mean", or "mean".
        ref_trimmed_mean_proportion: Trim proportion for ref_agg_method="trimmed_mean".
        reference_qc_mad_rel_threshold: Relative MAD threshold for run-level reference abs0 QC fail.
        reference_qc_min_abs0: Absolute minimum for run-level reference abs0 median QC.
        reference_qc_exclude: If True, rows from QC-failed runs become infeasible for constrained objective.
    """
    from gox_plate_pipeline.polymer_timeseries import normalize_t50_definition  # avoid circular import

    t50_definition = normalize_t50_definition(t50_definition)
    reference_polymer_id = str(reference_polymer_id).strip() or DEFAULT_REFERENCE_POLYMER_ID
    reference_polymer_norm = _normalize_polymer_id_token(reference_polymer_id)
    native_activity_min_rel = float(native_activity_min_rel)
    if not np.isfinite(native_activity_min_rel) or native_activity_min_rel <= 0.0:
        raise ValueError(
            f"native_activity_min_rel must be a positive finite number, got {native_activity_min_rel!r}"
        )
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
            "round_id",
            "run_id",
            "plate_id",
            "polymer_id",
            "reference_polymer_id",
            "t50_min",
            "t50_definition",
            "gox_t50_used_min",
            "denominator_source",
            "fog",
            "log_fog",
            "abs_activity_at_0",
            "gox_abs_activity_at_0_ref",
            "native_activity_rel_at_0",
            "native_activity_min_rel_threshold",
            "native_activity_feasible",
            "fog_native_constrained",
            "log_fog_native_constrained",
            "fog_constraint_reason",
            "native_0",
            "native_0_reference_source",
            "t_theta",
            "t_theta_censor_flag",
            "censor_flag",
            "reference_qc_fail",
            "reference_qc_reason",
            "reference_qc_ref_abs0_median",
            "reference_qc_ref_abs0_mad",
            "reference_qc_ref_abs0_rel_mad",
        ])
        return empty, pd.DataFrame(columns=[
            "round_id",
            "polymer_id",
            "reference_polymer_id",
            "mean_fog",
            "mean_log_fog",
            "robust_fog",
            "robust_log_fog",
            "log_fog_mad",
            "mean_fog_native_constrained",
            "mean_log_fog_native_constrained",
            "robust_fog_native_constrained",
            "robust_log_fog_native_constrained",
            "log_fog_native_constrained_mad",
            "native_feasible_fraction",
            "n_observations",
            "run_ids",
        ]), empty, FogWarningInfo()

    guard_low_mult = float(gox_guard_low_threshold if gox_guard_low_threshold is not None else gox_outlier_low_threshold)
    guard_high_mult = float(gox_guard_high_threshold if gox_guard_high_threshold is not None else gox_outlier_high_threshold)
    if guard_low_mult <= 0 or guard_high_mult <= 0:
        raise ValueError("gox_guard thresholds must be positive multipliers.")
    fallback_stat = str(gox_round_fallback_stat).strip().lower()
    if fallback_stat not in {"median", "mean", "trimmed_mean"}:
        raise ValueError(
            f"gox_round_fallback_stat must be 'median', 'mean', or 'trimmed_mean', got {gox_round_fallback_stat!r}"
        )
    trimmed_proportion = float(gox_round_trimmed_mean_proportion)
    if fallback_stat == "trimmed_mean" and not (0.0 <= trimmed_proportion < 0.5):
        raise ValueError("gox_round_trimmed_mean_proportion must be in [0, 0.5).")
    native_ref_mode = str(native_reference_mode).strip().lower()
    if native_ref_mode not in {"same_plate_then_round", "same_run_then_round"}:
        raise ValueError(
            "native_reference_mode must be 'same_plate_then_round' or 'same_run_then_round', "
            f"got {native_reference_mode!r}"
        )
    ref_method = str(ref_agg_method).strip().lower()
    if ref_method not in {"median", "mean", "trimmed_mean"}:
        raise ValueError(
            "ref_agg_method must be 'median', 'mean', or 'trimmed_mean', "
            f"got {ref_agg_method!r}"
        )
    ref_trim_prop = float(ref_trimmed_mean_proportion)
    if ref_method == "trimmed_mean" and not (0.0 <= ref_trim_prop < 0.5):
        raise ValueError("ref_trimmed_mean_proportion must be in [0, 0.5).")
    qc_rel_mad_thr = float(reference_qc_mad_rel_threshold)
    qc_min_abs0 = float(reference_qc_min_abs0)
    if not np.isfinite(qc_rel_mad_thr) or qc_rel_mad_thr < 0.0:
        raise ValueError("reference_qc_mad_rel_threshold must be a finite non-negative number.")
    if not np.isfinite(qc_min_abs0) or qc_min_abs0 < 0.0:
        raise ValueError("reference_qc_min_abs0 must be a finite non-negative number.")

    # Collect per-plate t50 for all runs in all rounds
    run_plate_t50: Dict[str, pd.DataFrame] = {}  # run_id -> DataFrame run_id, plate_id, polymer_id, t50_min
    run_abs_profiles: Dict[str, pd.DataFrame] = {}  # run_id -> DataFrame plate_id, polymer_id, heat_min, abs_activity
    missing_rates_files: List[tuple[str, str]] = []  # (round_id, run_id) pairs
    heat_grid_vals: set[float] = set()
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
            run_plate_t50[run_id] = compute_per_plate_t50_from_rates(
                df,
                run_id,
                t50_definition=t50_definition,
            )
            abs_df = df.copy()
            abs_df["plate_id"] = abs_df.get("plate_id", pd.Series(dtype=str)).astype(str)
            abs_df["polymer_id"] = abs_df.get("polymer_id", pd.Series(dtype=str)).astype(str).str.strip()
            abs_df["heat_min"] = pd.to_numeric(abs_df.get("heat_min", np.nan), errors="coerce")
            abs_df["abs_activity"] = pd.to_numeric(abs_df.get("abs_activity", np.nan), errors="coerce")
            abs_df["status"] = abs_df.get("status", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
            abs_df = abs_df[
                (abs_df["status"] == "ok")
                & np.isfinite(abs_df["heat_min"])
                & np.isfinite(abs_df["abs_activity"])
                & (abs_df["abs_activity"] > 0.0)
            ].copy()
            if abs_df.empty:
                run_abs_profiles[run_id] = pd.DataFrame(columns=["plate_id", "polymer_id", "heat_min", "abs_activity"])
            else:
                abs_agg = (
                    abs_df.groupby(["plate_id", "polymer_id", "heat_min"], dropna=False)
                    .agg(abs_activity=("abs_activity", "mean"))
                    .reset_index()
                )
                run_abs_profiles[run_id] = abs_agg
                for hh in abs_agg["heat_min"].tolist():
                    hv = pd.to_numeric(hh, errors="coerce")
                    if np.isfinite(hv):
                        heat_grid_vals.add(float(hv))

    heat_grid = sorted([float(h) for h in heat_grid_vals if np.isfinite(float(h))])
    per_row_rows: List[dict] = []
    round_gox_t50: Dict[str, List[float]] = {}  # round_id -> list of GOx t50 values (all run,plate in that round)

    for round_id, run_ids in sorted(round_to_runs.items()):
        run_ids = sorted(run_ids)
        gox_in_round: List[float] = []
        gox_plate_info: List[tuple[str, str, float, float]] = []  # (run_id, plate_id, t50, abs0)
        for run_id in run_ids:
            pt50 = run_plate_t50.get(run_id)
            if pt50 is None or pt50.empty:
                continue
            gox_plate = pt50[
                pt50["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) == reference_polymer_norm)
            ]
            for _, r in gox_plate.iterrows():
                plate_id = str(r.get("plate_id", ""))
                t = pd.to_numeric(r.get("t50_min", np.nan), errors="coerce")
                abs0 = pd.to_numeric(r.get("abs_activity_at_0", np.nan), errors="coerce")
                if np.isfinite(t) and t > 0:
                    gox_in_round.append(float(t))
                    gox_plate_info.append((run_id, plate_id, float(t), float(abs0) if np.isfinite(abs0) else np.nan))

        if not gox_in_round:
            raise ValueError(
                f"Round {round_id!r} has no reference polymer {reference_polymer_id!r} t50 in any (run, plate). "
                "FoG with same_plate/same_round denominator cannot be computed."
            )

        # Outlier detection: median-based
        import warnings
        median_gox = float(np.median(gox_in_round))
        low_threshold = median_gox * gox_outlier_low_threshold
        high_threshold = median_gox * gox_outlier_high_threshold
        outliers: List[tuple[str, str, float]] = []
        for run_id, plate_id, t50_val, _abs0 in gox_plate_info:
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
            print(f"⚠️  警告: 異常{reference_polymer_id} t50値が検出されました", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            print(f"Round: {round_id}", file=sys.stderr)
            print(f"{reference_polymer_id} t50 中央値: {median_gox:.3f} min", file=sys.stderr)
            print(f"異常値の閾値: [{low_threshold:.3f}, {high_threshold:.3f}] min", file=sys.stderr)
            print(f"検出された異常値: {outlier_details}", file=sys.stderr)
            if exclude_outlier_gox:
                print(f"→ 異常値はround平均{reference_polymer_id} t50の計算から除外されます。", file=sys.stderr)
            else:
                print(f"→ 異常値はround平均{reference_polymer_id} t50の計算に含まれます。", file=sys.stderr)
                print("→ 除外する場合は、--exclude_outlier_gox フラグを使用してください。", file=sys.stderr)
            print("=" * 80 + "\n", file=sys.stderr)
            # Also issue standard warning for programmatic access
            warnings.warn(
                f"Round {round_id!r} has outlier {reference_polymer_id} t50 values (median={median_gox:.3f}min, "
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
                        f"Round {round_id!r} has no reference-polymer t50 remaining after excluding outliers. "
                        "Cannot compute FoG."
                    )

        round_gox_t50[round_id] = gox_in_round
        outlier_keys = (
            {(rid, pid, float(t50_val)) for rid, pid, t50_val in outliers}
            if exclude_outlier_gox
            else set()
        )
        valid_gox_plate_info = [
            (rid, pid, t50_val, abs0)
            for rid, pid, t50_val, abs0 in gox_plate_info
            if (not outlier_keys) or ((rid, pid, float(t50_val)) not in outlier_keys)
        ]
        gox_abs0_round_vals = [
            float(abs0)
            for _rid, _pid, _t50_val, abs0 in valid_gox_plate_info
            if np.isfinite(abs0) and float(abs0) > 0.0
        ]
        round_gox_abs0_fallback = _aggregate_reference_values(
            gox_abs0_round_vals,
            method=ref_method,
            trimmed_mean_proportion=ref_trim_prop,
        )

        # same-run reference abs0 (policy-v2 primary reference)
        run_gox_abs0_by_run: Dict[str, float] = {}
        run_ref_abs0_vals_map: Dict[str, List[float]] = {}
        for rid, _pid, _t50_val, abs0 in valid_gox_plate_info:
            if np.isfinite(abs0) and float(abs0) > 0.0:
                run_ref_abs0_vals_map.setdefault(str(rid), []).append(float(abs0))
        for rid, vals in run_ref_abs0_vals_map.items():
            run_gox_abs0_by_run[rid] = _aggregate_reference_values(
                vals,
                method=ref_method,
                trimmed_mean_proportion=ref_trim_prop,
            )

        # Run-level QC on same-run reference abs0 spread.
        run_ref_qc: Dict[str, Dict[str, Any]] = {}
        for rid in run_ids:
            vals = run_ref_abs0_vals_map.get(rid, [])
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr) & (arr > 0.0)]
            n_ref = int(arr.size)
            if n_ref == 0:
                run_ref_qc[rid] = {
                    "fail": True,
                    "reason": "missing_same_run_reference_abs0",
                    "median": np.nan,
                    "mad": np.nan,
                    "rel_mad": np.nan,
                    "n_ref": 0,
                }
                continue
            med = float(np.nanmedian(arr))
            mad = float(np.nanmedian(np.abs(arr - med)))
            rel_mad = float(mad / med) if np.isfinite(med) and med > 0 else np.nan
            reason_parts: List[str] = []
            fail = False
            if np.isfinite(med) and med < qc_min_abs0:
                fail = True
                reason_parts.append("ref_abs0_median_below_min")
            if n_ref >= 2 and np.isfinite(rel_mad) and rel_mad > qc_rel_mad_thr:
                fail = True
                reason_parts.append("ref_abs0_rel_mad_above_threshold")
            run_ref_qc[rid] = {
                "fail": bool(fail),
                "reason": ";".join(reason_parts),
                "median": med,
                "mad": mad,
                "rel_mad": rel_mad,
                "n_ref": n_ref,
            }
            if fail:
                warning_info.reference_qc_fail_runs.append(
                    {
                        "round_id": round_id,
                        "run_id": rid,
                        "n_reference_wells": n_ref,
                        "ref_abs0_median": med,
                        "ref_abs0_mad": mad,
                        "ref_abs0_rel_mad": rel_mad,
                        "reason": ";".join(reason_parts),
                    }
                )

        # Round representative GOx t50 (used for same_round fallback and same_plate guard fallback).
        # Values are taken after optional outlier exclusion above.
        round_gox_arr = np.asarray(gox_in_round, dtype=float)
        round_gox_median = float(np.nanmedian(round_gox_arr))
        round_gox_mean = float(np.nanmean(round_gox_arr))
        if fallback_stat == "median":
            round_gox_fallback = round_gox_median
        elif fallback_stat == "trimmed_mean":
            n_gox = len(round_gox_arr)
            if n_gox >= 3:
                round_gox_fallback = float(
                    trim_mean(round_gox_arr, proportiontocut=min(trimmed_proportion, (n_gox - 1) / (2 * n_gox)))
                )
            else:
                round_gox_fallback = round_gox_median
        else:
            round_gox_fallback = round_gox_mean
        guard_low_threshold = round_gox_median * guard_low_mult
        guard_high_threshold = round_gox_median * guard_high_mult
        guard_logged_plates: set[tuple[str, str, str]] = set()

        for run_id in run_ids:
            pt50 = run_plate_t50.get(run_id)
            if pt50 is None or pt50.empty:
                continue
            run_abs_df = run_abs_profiles.get(run_id)
            if run_abs_df is None:
                run_abs_df = pd.DataFrame(columns=["plate_id", "polymer_id", "heat_min", "abs_activity"])
            gox_by_plate: Dict[str, float] = {}
            gox_abs0_by_plate: Dict[str, float] = {}
            gox_plate = pt50[
                pt50["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) == reference_polymer_norm)
            ]
            for _, r in gox_plate.iterrows():
                plate_id = str(r.get("plate_id", ""))
                t = pd.to_numeric(r.get("t50_min", np.nan), errors="coerce")
                abs0 = pd.to_numeric(r.get("abs_activity_at_0", np.nan), errors="coerce")
                if np.isfinite(t) and t > 0:
                    # Check for duplicate GOx t50 for the same plate (should not happen)
                    if plate_id in gox_by_plate:
                        existing_t50 = gox_by_plate[plate_id]
                        raise ValueError(
                            f"Run {run_id!r}, plate {plate_id!r} has multiple reference-polymer t50 values: "
                            f"{existing_t50:.3f}min and {t:.3f}min. "
                            "Each plate should have exactly one reference-polymer t50 value."
                        )
                    gox_by_plate[plate_id] = float(t)
                    if np.isfinite(abs0) and abs0 > 0:
                        gox_abs0_by_plate[plate_id] = float(abs0)
            run_abs0_ref = run_gox_abs0_by_run.get(run_id, np.nan)
            run_qc = run_ref_qc.get(
                run_id,
                {
                    "fail": True,
                    "reason": "missing_same_run_reference_abs0",
                    "median": np.nan,
                    "mad": np.nan,
                    "rel_mad": np.nan,
                    "n_ref": 0,
                },
            )

            polymers = pt50[
                pt50["polymer_id"].astype(str).map(lambda x: _normalize_polymer_id_token(x) != reference_polymer_norm)
            ]
            for _, r in polymers.iterrows():
                plate_id = str(r.get("plate_id", ""))
                polymer_id = str(r.get("polymer_id", "")).strip()
                t50_min = pd.to_numeric(r.get("t50_min", np.nan), errors="coerce")
                abs_activity_at_0 = pd.to_numeric(r.get("abs_activity_at_0", np.nan), errors="coerce")
                if not np.isfinite(t50_min) or t50_min <= 0:
                    continue
                gox_t50 = gox_by_plate.get(plate_id)
                if gox_t50 is not None and gox_t50 > 0:
                    use_same_plate = True
                    if gox_guard_same_plate:
                        if (gox_t50 < guard_low_threshold) or (gox_t50 > guard_high_threshold):
                            use_same_plate = False
                            key = (round_id, run_id, plate_id)
                            if key not in guard_logged_plates:
                                warning_info.guarded_same_plate.append({
                                    "round_id": round_id,
                                    "run_id": run_id,
                                    "plate_id": plate_id,
                                    "same_plate_gox_t50_min": float(gox_t50),
                                    "fallback_gox_t50_min": float(round_gox_fallback),
                                    "guard_low_threshold_min": float(guard_low_threshold),
                                    "guard_high_threshold_min": float(guard_high_threshold),
                                })
                                guard_logged_plates.add(key)
                    if use_same_plate:
                        denominator_source = "same_plate"
                    else:
                        gox_t50 = round_gox_fallback
                        denominator_source = "same_round"
                else:
                    gox_t50 = round_gox_fallback
                    denominator_source = "same_round"
                fog = t50_min / gox_t50
                log_fog = np.log(fog) if fog > 0 else np.nan
                same_plate_abs0 = gox_abs0_by_plate.get(plate_id, np.nan)
                if native_ref_mode == "same_run_then_round":
                    if np.isfinite(run_abs0_ref) and float(run_abs0_ref) > 0.0:
                        gox_abs0_ref = float(run_abs0_ref)
                        native_ref_source = "same_run"
                    else:
                        gox_abs0_ref = float(round_gox_abs0_fallback) if np.isfinite(round_gox_abs0_fallback) and float(round_gox_abs0_fallback) > 0.0 else np.nan
                        native_ref_source = "same_round"
                else:
                    if np.isfinite(same_plate_abs0) and float(same_plate_abs0) > 0.0:
                        gox_abs0_ref = float(same_plate_abs0)
                        native_ref_source = "same_plate"
                    else:
                        gox_abs0_ref = float(round_gox_abs0_fallback) if np.isfinite(round_gox_abs0_fallback) and float(round_gox_abs0_fallback) > 0.0 else np.nan
                        native_ref_source = "same_round"
                native_rel = (
                    float(abs_activity_at_0) / float(gox_abs0_ref)
                    if np.isfinite(abs_activity_at_0) and np.isfinite(gox_abs0_ref) and gox_abs0_ref > 0
                    else np.nan
                )
                native_feasible = bool(np.isfinite(native_rel) and native_rel >= float(native_activity_min_rel))
                qc_fail = bool(run_qc.get("fail", False))
                fog_native = float(fog) if native_feasible else np.nan
                log_fog_native = np.log(fog_native) if np.isfinite(fog_native) and fog_native > 0 else np.nan
                if not np.isfinite(abs_activity_at_0):
                    constraint_reason = "missing_abs_activity_at_0"
                elif not np.isfinite(gox_abs0_ref):
                    constraint_reason = "missing_gox_abs_activity_at_0"
                elif np.isfinite(native_rel) and native_rel < float(native_activity_min_rel):
                    constraint_reason = "native_activity_below_threshold"
                else:
                    constraint_reason = ""
                if reference_qc_exclude and qc_fail:
                    native_feasible = False
                    fog_native = np.nan
                    log_fog_native = np.nan
                    constraint_reason = (
                        f"{constraint_reason};reference_qc_fail"
                        if constraint_reason
                        else "reference_qc_fail"
                    )

                # U(t) series uses reference abs activity at t=0 (same-run first in policy v2 mode).
                u_cols: Dict[str, float] = {}
                t_theta = np.nan
                t_theta_flag = "missing_profile"
                if np.isfinite(gox_abs0_ref) and float(gox_abs0_ref) > 0.0 and (run_abs_df is not None) and (not run_abs_df.empty):
                    prof = run_abs_df[
                        (run_abs_df["plate_id"].astype(str) == str(plate_id))
                        & (run_abs_df["polymer_id"].astype(str).str.strip() == polymer_id)
                    ].copy()
                    if not prof.empty:
                        prof["heat_min"] = pd.to_numeric(prof["heat_min"], errors="coerce")
                        prof["abs_activity"] = pd.to_numeric(prof["abs_activity"], errors="coerce")
                        prof = prof[np.isfinite(prof["heat_min"]) & np.isfinite(prof["abs_activity"])].copy()
                        if not prof.empty:
                            prof = prof.sort_values("heat_min")
                            u_vals = prof["abs_activity"].to_numpy(dtype=float) / float(gox_abs0_ref)
                            t_vals = prof["heat_min"].to_numpy(dtype=float)
                            t_theta, t_theta_flag = _compute_t_theta_from_series(
                                t_vals,
                                u_vals,
                                theta=float(native_activity_min_rel),
                            )
                            for hh, uv in zip(t_vals, u_vals):
                                u_cols[_u_col_name(float(hh))] = float(uv) if np.isfinite(uv) else np.nan
                elif not (np.isfinite(gox_abs0_ref) and float(gox_abs0_ref) > 0.0):
                    t_theta_flag = "missing_reference_abs0"
                for hh in heat_grid:
                    col = _u_col_name(hh)
                    if col not in u_cols:
                        u_cols[col] = np.nan
                per_row_rows.append({
                    "round_id": round_id,
                    "run_id": run_id,
                    "plate_id": plate_id,
                    "polymer_id": polymer_id,
                    "reference_polymer_id": reference_polymer_id,
                    "t50_min": t50_min,
                    "t50_definition": str(r.get("t50_definition", t50_definition)),
                    "gox_t50_used_min": gox_t50,
                    "denominator_source": denominator_source,
                    "fog": fog,
                    "log_fog": log_fog,
                    "abs_activity_at_0": float(abs_activity_at_0) if np.isfinite(abs_activity_at_0) else np.nan,
                    "gox_abs_activity_at_0_ref": gox_abs0_ref,
                    "native_activity_rel_at_0": native_rel,
                    "native_activity_min_rel_threshold": float(native_activity_min_rel),
                    "native_activity_feasible": int(native_feasible),
                    "fog_native_constrained": fog_native,
                    "log_fog_native_constrained": log_fog_native,
                    "fog_constraint_reason": constraint_reason,
                    "native_0": native_rel,
                    "native_0_reference_source": native_ref_source,
                    "t_theta": float(t_theta) if np.isfinite(t_theta) else np.nan,
                    "t_theta_censor_flag": str(t_theta_flag),
                    "censor_flag": str(t_theta_flag),
                    "reference_qc_fail": int(qc_fail),
                    "reference_qc_reason": str(run_qc.get("reason", "")),
                    "reference_qc_ref_abs0_median": float(run_qc.get("median", np.nan)) if np.isfinite(pd.to_numeric(run_qc.get("median", np.nan), errors="coerce")) else np.nan,
                    "reference_qc_ref_abs0_mad": float(run_qc.get("mad", np.nan)) if np.isfinite(pd.to_numeric(run_qc.get("mad", np.nan), errors="coerce")) else np.nan,
                    "reference_qc_ref_abs0_rel_mad": float(run_qc.get("rel_mad", np.nan)) if np.isfinite(pd.to_numeric(run_qc.get("rel_mad", np.nan), errors="coerce")) else np.nan,
                    **u_cols,
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
            robust_fog = float(np.nanmedian(g["fog"]))
            robust_log_fog = float(np.nanmedian(g["log_fog"]))
            log_fog_mad = float(np.nanmedian(np.abs(g["log_fog"] - robust_log_fog)))
            fog_native = pd.to_numeric(g.get("fog_native_constrained", np.nan), errors="coerce")
            log_fog_native = pd.to_numeric(g.get("log_fog_native_constrained", np.nan), errors="coerce")
            native_feasible = pd.to_numeric(g.get("native_activity_feasible", np.nan), errors="coerce")
            native_valid = np.isfinite(fog_native) & (fog_native > 0)
            if np.any(native_valid):
                fog_native_vals = fog_native[native_valid]
                log_fog_native_vals = log_fog_native[native_valid]
                robust_fog_native = float(np.nanmedian(fog_native_vals))
                robust_log_fog_native = float(np.nanmedian(log_fog_native_vals))
                log_fog_native_mad = float(
                    np.nanmedian(np.abs(log_fog_native_vals - robust_log_fog_native))
                )
                mean_fog_native = float(np.nanmean(fog_native_vals))
                mean_log_fog_native = float(np.nanmean(log_fog_native_vals))
            else:
                robust_fog_native = np.nan
                robust_log_fog_native = np.nan
                log_fog_native_mad = np.nan
                mean_fog_native = np.nan
                mean_log_fog_native = np.nan
            if not np.isfinite(native_feasible).any():
                native_feasible_fraction = float(np.nanmean(native_valid.astype(float)))
            else:
                native_feasible_fraction = float(np.nanmean(np.clip(native_feasible, 0.0, 1.0)))
            round_av_rows.append({
                "round_id": round_id,
                "polymer_id": pid,
                "reference_polymer_id": reference_polymer_id,
                "mean_fog": float(g["fog"].mean()),
                "mean_log_fog": float(g["log_fog"].mean()),
                "robust_fog": robust_fog,
                "robust_log_fog": robust_log_fog,
                "log_fog_mad": log_fog_mad,
                "mean_fog_native_constrained": mean_fog_native,
                "mean_log_fog_native_constrained": mean_log_fog_native,
                "robust_fog_native_constrained": robust_fog_native,
                "robust_log_fog_native_constrained": robust_log_fog_native,
                "log_fog_native_constrained_mad": log_fog_native_mad,
                "native_feasible_fraction": native_feasible_fraction,
                "n_observations": int(len(g)),
                "run_ids": ",".join(sorted(g["run_id"].astype(str).unique().tolist())),
            })
    round_averaged_df = pd.DataFrame(round_av_rows)

    gox_trace_df = build_round_gox_traceability(
        run_round_map,
        processed_dir,
        reference_polymer_id=reference_polymer_id,
    )
    return per_row_df, round_averaged_df, gox_trace_df, warning_info
