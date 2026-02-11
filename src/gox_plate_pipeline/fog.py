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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import trim_mean
import yaml


# t50 unit used in outputs (documented for BO and figures)
T50_UNIT = "min"


@dataclass
class FogWarningInfo:
    """Warning information collected during FoG calculation."""
    outlier_gox: List[Dict[str, Any]] = field(default_factory=list)  # List of outlier GOx t50 info
    guarded_same_plate: List[Dict[str, Any]] = field(default_factory=list)  # same_plate guard fallback details
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
    
    if (
        not warning_info.outlier_gox
        and not warning_info.guarded_same_plate
        and not warning_info.missing_rates_files
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
    df["abs_activity_at_0"] = pd.to_numeric(df.get("abs_activity_at_0", np.nan), errors="coerce")
    df["abs_activity_at_20"] = pd.to_numeric(df.get("abs_activity_at_20", np.nan), errors="coerce")
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

    # Functional activity at 20 min relative to GOx reference.
    # Prefer value pre-computed in t50 CSV; otherwise derive from abs_activity_at_20 / gox_abs_activity_at_20_ref.
    gox_abs20_same_run = np.nan
    if not gox.empty:
        gox_abs20_vals = pd.to_numeric(gox.get("abs_activity_at_20", np.nan), errors="coerce")
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
        src = src.where(src != "", np.where(np.isfinite(df["gox_abs_activity_at_20_ref"]), "same_run_gox", "missing_gox_reference"))
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
        "t50_min",
        "t50_definition",
        "t50_target_rea_percent",
        "rea_at_20_percent",
        "abs_activity_at_0",
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


def _plot_run_ranking_bar(
    rank_df: pd.DataFrame,
    *,
    value_col: str,
    rank_col: str,
    title: str,
    xlabel: str,
    out_path: Path,
    color_map: Optional[Dict[str, str]] = None,
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

    from gox_plate_pipeline.fitting import apply_paper_style
    import matplotlib.pyplot as plt

    labels = []
    values = []
    colors = []
    cmap = color_map or {}
    default_color = "#4C78A8"
    for _, row in data.iterrows():
        pid = str(row.get("polymer_id", "")).strip()
        rank_val = int(row[rank_col])
        labels.append(f"{rank_val}. {pid}")
        values.append(float(row[value_col]))
        if pid.upper() == "GOX":
            colors.append("#808080")
        else:
            colors.append(cmap.get(pid, default_color))

    fig_h = max(2.2, 0.28 * len(labels) + 0.8)
    with plt.rc_context(apply_paper_style()):
        fig, ax = plt.subplots(figsize=(5.0, fig_h))
        y = np.arange(len(labels), dtype=float)
        ax.barh(
            y,
            values,
            color=colors,
            edgecolor="0.2",
            linewidth=0.4,
            height=0.62,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    return True


def write_run_ranking_outputs(
    fog_df: pd.DataFrame,
    run_id: str,
    out_dir: Path,
    *,
    color_map_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Write per-run ranking CSVs and bar charts for t50/FoG.

    Outputs (under out_dir):
      - t50_ranking__{run_id}.csv
      - fog_ranking__{run_id}.csv
      - functional_ranking__{run_id}.csv
      - t50_ranking__{run_id}.png (if plottable rows exist)
      - fog_ranking__{run_id}.png (if plottable rows exist)
      - functional_ranking__{run_id}.png (if plottable rows exist)

    Ranking score applies an absolute-activity guard when available:
      activity_weight = clip(abs_activity_at_0 / GOx_abs_activity_at_0, 0, 1)
      t50_activity_adjusted_min = t50_min * activity_weight
      fog_activity_adjusted = fog * activity_weight
    """
    run_id = str(run_id).strip()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fog_df.copy()
    if "run_id" not in df.columns:
        df["run_id"] = run_id
    df["run_id"] = df["run_id"].astype(str)
    df = df[df["run_id"] == run_id].copy()
    if df.empty:
        # Keep reproducible empty outputs when no rows are available.
        df = pd.DataFrame(columns=["run_id", "polymer_id", "t50_min", "t50_censored", "fog", "fog_missing_reason"])

    df["t50_min"] = pd.to_numeric(df.get("t50_min", np.nan), errors="coerce")
    df["fog"] = pd.to_numeric(df.get("fog", np.nan), errors="coerce")
    df["abs_activity_at_0"] = pd.to_numeric(df.get("abs_activity_at_0", np.nan), errors="coerce")
    df["abs_activity_at_20"] = pd.to_numeric(df.get("abs_activity_at_20", np.nan), errors="coerce")
    df["gox_abs_activity_at_20_ref"] = pd.to_numeric(df.get("gox_abs_activity_at_20_ref", np.nan), errors="coerce")
    df["functional_activity_at_20_rel"] = pd.to_numeric(df.get("functional_activity_at_20_rel", np.nan), errors="coerce")
    if "functional_reference_source" in df.columns:
        df["functional_reference_source"] = df["functional_reference_source"].astype(str)
    if "t50_censored" in df.columns:
        df["t50_censored"] = pd.to_numeric(df["t50_censored"], errors="coerce").fillna(1).astype(int)
    gox_abs_activity_at_0 = np.nan
    gox_mask = df["polymer_id"].astype(str).str.strip().str.upper() == "GOX"
    if np.any(gox_mask):
        gox_vals = df.loc[gox_mask, "abs_activity_at_0"]
        gox_vals = gox_vals[np.isfinite(gox_vals) & (gox_vals > 0)]
        if not gox_vals.empty:
            gox_abs_activity_at_0 = float(gox_vals.iloc[0])
    df["gox_abs_activity_at_0"] = gox_abs_activity_at_0
    if np.isfinite(gox_abs_activity_at_0) and gox_abs_activity_at_0 > 0:
        abs0_vs_gox = df["abs_activity_at_0"] / float(gox_abs_activity_at_0)
        abs0_vs_gox = pd.to_numeric(abs0_vs_gox, errors="coerce")
        activity_weight = np.clip(abs0_vs_gox, 0.0, 1.0)
        activity_weight = np.where(np.isfinite(activity_weight), activity_weight, 1.0)
        df["abs0_vs_gox"] = abs0_vs_gox
    else:
        activity_weight = np.ones(len(df), dtype=float)
        df["abs0_vs_gox"] = np.nan
    df["activity_weight"] = activity_weight
    df["t50_activity_adjusted_min"] = df["t50_min"] * df["activity_weight"]
    df["fog_activity_adjusted"] = df["fog"] * df["activity_weight"]

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
        "t50_min",
        "t50_activity_adjusted_min",
        "t50_censored",
        "t50_definition",
        "t50_target_rea_percent",
        "rea_at_20_percent",
        "abs_activity_at_0",
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
    t50_out = out_dir / f"t50_ranking__{run_id}.csv"
    t50_tbl.to_csv(t50_out, index=False)

    # FoG ranking table
    fog_cols = [
        "run_id",
        "polymer_id",
        "fog",
        "fog_activity_adjusted",
        "log_fog",
        "fog_missing_reason",
        "gox_t50_same_run_min",
        "t50_min",
        "abs_activity_at_0",
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
    fog_out = out_dir / f"fog_ranking__{run_id}.csv"
    fog_tbl.to_csv(fog_out, index=False)

    # Functional (20 min) ranking table
    func_cols = [
        "run_id",
        "polymer_id",
        "functional_activity_at_20_rel",
        "abs_activity_at_20",
        "gox_abs_activity_at_20_ref",
        "functional_reference_source",
        "functional_reference_round_id",
        "functional_reference_run_id",
        "abs_activity_at_0",
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
    func_out = out_dir / f"functional_ranking__{run_id}.csv"
    func_tbl.to_csv(func_out, index=False)

    cmap = _load_polymer_color_map(color_map_path)
    t50_png = out_dir / f"t50_ranking__{run_id}.png"
    fog_png = out_dir / f"fog_ranking__{run_id}.png"
    func_png = out_dir / f"functional_ranking__{run_id}.png"
    wrote_t50_png = _plot_run_ranking_bar(
        t50_tbl,
        value_col="t50_activity_adjusted_min",
        rank_col="rank_t50",
        title="t50 ranking (activity-adjusted)",
        xlabel="activity-adjusted t50 (min)",
        out_path=t50_png,
        color_map=cmap,
    )
    wrote_fog_png = _plot_run_ranking_bar(
        fog_tbl,
        value_col="fog_activity_adjusted",
        rank_col="rank_fog",
        title="FoG ranking (activity-adjusted)",
        xlabel="activity-adjusted FoG",
        out_path=fog_png,
        color_map=cmap,
    )
    wrote_func_png = _plot_run_ranking_bar(
        func_tbl,
        value_col="functional_activity_at_20_rel",
        rank_col="rank_functional",
        title="Functional ranking (20 min)",
        xlabel="functional activity ratio at 20 min (vs GOx)",
        out_path=func_png,
        color_map=cmap,
    )
    if not wrote_t50_png and t50_png.exists():
        t50_png.unlink(missing_ok=True)
    if not wrote_fog_png and fog_png.exists():
        fog_png.unlink(missing_ok=True)
    if not wrote_func_png and func_png.exists():
        func_png.unlink(missing_ok=True)

    outputs: Dict[str, Path] = {
        "t50_ranking_csv": t50_out,
        "fog_ranking_csv": fog_out,
        "functional_ranking_csv": func_out,
    }
    if wrote_t50_png:
        outputs["t50_ranking_png"] = t50_png
    if wrote_fog_png:
        outputs["fog_ranking_png"] = fog_png
    if wrote_func_png:
        outputs["functional_ranking_png"] = func_png
    return outputs


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
    - Output columns:
      round_id, polymer_id, mean_fog, mean_log_fog, robust_fog, robust_log_fog,
      log_fog_mad, n_observations, run_ids.
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
        return pd.DataFrame(
            columns=[
                "round_id",
                "polymer_id",
                "mean_fog",
                "mean_log_fog",
                "robust_fog",
                "robust_log_fog",
                "log_fog_mad",
                "n_observations",
                "run_ids",
            ]
        )

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
            robust_fog = float(np.nanmedian(g["_fog"]))
            robust_log_fog = float(np.nanmedian(g["_log_fog"]))
            log_fog_mad = float(np.nanmedian(np.abs(g["_log_fog"] - robust_log_fog)))
            n_obs = int(len(g))
            run_list = sorted(g["run_id"].astype(str).unique().tolist())
            rows.append({
                "round_id": round_id,
                "polymer_id": pid,
                "mean_fog": mean_fog,
                "mean_log_fog": mean_log_fog,
                "robust_fog": robust_fog,
                "robust_log_fog": robust_log_fog,
                "log_fog_mad": log_fog_mad,
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
    )  # avoid circular import

    t50_definition = normalize_t50_definition(t50_definition)

    df = rates_df.copy()
    df["polymer_id"] = df.get("polymer_id", pd.Series(dtype=str)).astype(str).str.strip()
    df["plate_id"] = df.get("plate_id", pd.Series(dtype=str)).astype(str)
    df["heat_min"] = pd.to_numeric(df.get("heat_min", np.nan), errors="coerce")
    df["REA_percent"] = pd.to_numeric(df.get("REA_percent", np.nan), errors="coerce")
    df["status"] = df.get("status", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    ok = df[(df["status"] == "ok") & np.isfinite(df["REA_percent"]) & (df["REA_percent"] > 0)].copy()
    if ok.empty:
        return pd.DataFrame(
            columns=["run_id", "plate_id", "polymer_id", "t50_min", "n_points", "fit_model", "t50_definition"]
        )

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
        })
    return pd.DataFrame(rows)


def build_fog_plate_aware(
    run_round_map: Dict[str, str],
    processed_dir: Path,
    *,
    t50_definition: str = "y0_half",
    exclude_outlier_gox: bool = False,
    gox_outlier_low_threshold: float = 0.33,
    gox_outlier_high_threshold: float = 3.0,
    gox_guard_same_plate: bool = True,
    gox_guard_low_threshold: Optional[float] = None,
    gox_guard_high_threshold: Optional[float] = None,
    gox_round_fallback_stat: str = "median",
    gox_round_trimmed_mean_proportion: float = 0.1,
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
      per_row_df columns: run_id, plate_id, polymer_id, t50_min, gox_t50_used_min, denominator_source, fog, log_fog.
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
    """
    from gox_plate_pipeline.polymer_timeseries import normalize_t50_definition  # avoid circular import

    t50_definition = normalize_t50_definition(t50_definition)
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
            "t50_min",
            "t50_definition",
            "gox_t50_used_min",
            "denominator_source",
            "fog",
            "log_fog",
        ])
        return empty, pd.DataFrame(columns=[
            "round_id",
            "polymer_id",
            "mean_fog",
            "mean_log_fog",
            "robust_fog",
            "robust_log_fog",
            "log_fog_mad",
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
            run_plate_t50[run_id] = compute_per_plate_t50_from_rates(
                df,
                run_id,
                t50_definition=t50_definition,
            )

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
                per_row_rows.append({
                    "round_id": round_id,
                    "run_id": run_id,
                    "plate_id": plate_id,
                    "polymer_id": polymer_id,
                    "t50_min": t50_min,
                    "t50_definition": str(r.get("t50_definition", t50_definition)),
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
            robust_fog = float(np.nanmedian(g["fog"]))
            robust_log_fog = float(np.nanmedian(g["log_fog"]))
            log_fog_mad = float(np.nanmedian(np.abs(g["log_fog"] - robust_log_fog)))
            round_av_rows.append({
                "round_id": round_id,
                "polymer_id": pid,
                "mean_fog": float(g["fog"].mean()),
                "mean_log_fog": float(g["log_fog"].mean()),
                "robust_fog": robust_fog,
                "robust_log_fog": robust_log_fog,
                "log_fog_mad": log_fog_mad,
                "n_observations": int(len(g)),
                "run_ids": ",".join(sorted(g["run_id"].astype(str).unique().tolist())),
            })
    round_averaged_df = pd.DataFrame(round_av_rows)

    gox_trace_df = build_round_gox_traceability(run_round_map, processed_dir)
    return per_row_df, round_averaged_df, gox_trace_df, warning_info
