#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure local src/ is used (avoid importing an older installed package)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Headless plotting backend
import matplotlib

matplotlib.use("Agg")

from gox_plate_pipeline.fog import write_run_ranking_outputs  # noqa: E402
from gox_plate_pipeline.polymer_timeseries import (  # noqa: E402
    extract_measurement_date_from_run_id,
    normalize_t50_definition,
    plot_per_polymer_timeseries_across_runs_with_error_bars,
)
from gox_plate_pipeline.summary import build_run_manifest_dict  # noqa: E402


def _normalize_t50_value(v: object) -> str:
    try:
        return normalize_t50_definition(str(v))
    except Exception:
        return ""


def _parse_bool(v: object, *, default: bool = True) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    if s in {"", "—", "-", "na", "n/a", "none", "null"}:
        return default
    return default


def _safe_group_stem(text: str) -> str:
    s = "" if text is None else str(text).strip()
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return out or "group"


def _load_run_group_table(path: Path | None) -> dict[str, dict[str, object]]:
    """
    Load run group TSV as:
      run_id -> {"group_id": str, "include": bool}
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}

    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    if "run_id" not in df.columns:
        return {}

    group_col = None
    for c in ["group_id", "analysis_group_id", "date_prefix"]:
        if c in df.columns:
            group_col = c
            break
    include_col = None
    for c in ["include_in_group_mean", "include_in_same_date_mean", "include", "enabled", "use"]:
        if c in df.columns:
            include_col = c
            break

    out: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        rid = str(row.get("run_id", "")).strip()
        if not rid:
            continue
        gid = str(row.get(group_col, "")).strip() if group_col is not None else ""
        if not gid:
            gid = extract_measurement_date_from_run_id(rid)
        inc = _parse_bool(row.get(include_col), default=True) if include_col is not None else True
        out[rid] = {"group_id": gid, "include": bool(inc)}
    return out


def _collect_group_runs(
    *,
    run_id: str,
    explicit_runs: list[str] | None,
    run_group_table: dict[str, dict[str, object]] | None,
    group_id_override: str | None = None,
) -> tuple[list[str], str]:
    table = run_group_table or {}

    if explicit_runs:
        runs = sorted({str(x).strip() for x in explicit_runs if str(x).strip()})
        if run_id not in runs:
            runs.append(run_id)
        inferred_group = str(table.get(run_id, {}).get("group_id", "")).strip()
        if not inferred_group:
            inferred_group = extract_measurement_date_from_run_id(run_id)
        if group_id_override:
            inferred_group = str(group_id_override).strip()
        return sorted(set(runs)), inferred_group

    if group_id_override:
        gid = str(group_id_override).strip()
        if not gid:
            gid = extract_measurement_date_from_run_id(run_id)
        runs = [rid for rid, meta in table.items() if str(meta.get("group_id", "")).strip() == gid and bool(meta.get("include", True))]
        if not runs:
            runs = [run_id]
        return sorted(set(runs)), gid

    if table and run_id in table:
        gid = str(table[run_id].get("group_id", "")).strip() or extract_measurement_date_from_run_id(run_id)
        runs = [rid for rid, meta in table.items() if str(meta.get("group_id", "")).strip() == gid and bool(meta.get("include", True))]
        # If anchor run itself is excluded in the table, still keep it to avoid surprising no-op.
        if run_id not in runs:
            runs.append(run_id)
        return sorted(set(runs)), gid

    # Fallback (no run group table or anchor missing in table): anchor run only.
    gid = extract_measurement_date_from_run_id(run_id)
    return [run_id], gid


# Backward-compat helper (used by tests / older references)
def _load_same_date_include_map(path: Path | None) -> dict[str, bool]:
    table = _load_run_group_table(path)
    return {rid: bool(meta.get("include", True)) for rid, meta in table.items()}


# Backward-compat helper (used by tests / older references)
def _collect_same_date_runs(
    *,
    run_id: str,
    processed_dir: Path,
    explicit_runs: list[str] | None,
    same_date_include_map: dict[str, bool] | None = None,
) -> list[str]:
    del processed_dir  # no longer used in group-based collection
    table = {}
    for rid, inc in (same_date_include_map or {}).items():
        table[str(rid).strip()] = {
            "group_id": extract_measurement_date_from_run_id(str(rid)),
            "include": bool(inc),
        }
    runs, _gid = _collect_group_runs(
        run_id=run_id,
        explicit_runs=explicit_runs,
        run_group_table=table,
        group_id_override=None,
    )
    return runs


def _existing_paths_for_runs(processed_dir: Path, run_ids: list[str]) -> tuple[list[Path], list[Path]]:
    summary_paths: list[Path] = []
    fog_paths: list[Path] = []
    for rid in run_ids:
        s = processed_dir / rid / "fit" / "summary_simple.csv"
        f = processed_dir / rid / "fit" / f"fog_summary__{rid}.csv"
        if s.is_file():
            summary_paths.append(s)
        if f.is_file():
            fog_paths.append(f)
    return summary_paths, fog_paths


def _aggregate_same_date_fog(
    *,
    run_ids: list[str],
    processed_dir: Path,
    group_run_id: str,
    t50_definition: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for rid in run_ids:
        p = processed_dir / rid / "fit" / f"fog_summary__{rid}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        df = df.copy()
        df["source_run_id"] = rid
        if "t50_definition" in df.columns:
            df["_t50_norm"] = df["t50_definition"].map(_normalize_t50_value)
            matched = df["_t50_norm"] == t50_definition
            if matched.any():
                df = df.loc[matched].copy()
            else:
                continue
            df["t50_definition"] = df["_t50_norm"]
            df = df.drop(columns=["_t50_norm"])
        else:
            df["t50_definition"] = t50_definition
        rows.append(df)

    base_cols = [
        "run_id",
        "polymer_id",
        "t50_min",
        "t50_censored",
        "fog",
        "log_fog",
        "t50_definition",
        "source_run_ids",
        "n_source_runs",
        "n_t50",
        "n_fog",
        "t50_target_rea_percent",
        "rea_at_20_percent",
        "use_for_bo",
        "fog_missing_reason",
    ]
    if not rows:
        return pd.DataFrame(columns=base_cols)

    combined = pd.concat(rows, ignore_index=True)
    combined["polymer_id"] = combined["polymer_id"].astype(str).str.strip()
    combined = combined[combined["polymer_id"].ne("")].copy()
    if combined.empty:
        return pd.DataFrame(columns=base_cols)

    combined["t50_min"] = pd.to_numeric(combined.get("t50_min", np.nan), errors="coerce")
    combined["fog"] = pd.to_numeric(combined.get("fog", np.nan), errors="coerce")
    combined["rea_at_20_percent"] = pd.to_numeric(combined.get("rea_at_20_percent", np.nan), errors="coerce")
    combined["t50_target_rea_percent"] = pd.to_numeric(combined.get("t50_target_rea_percent", np.nan), errors="coerce")
    if "use_for_bo" not in combined.columns:
        combined["use_for_bo"] = True

    out_rows: list[dict[str, object]] = []
    for pid, g in combined.groupby("polymer_id", sort=True):
        pid = str(pid).strip()
        if not pid:
            continue

        src_runs = sorted({str(x).strip() for x in g["source_run_id"].tolist() if str(x).strip()})
        t50_vals = g["t50_min"]
        t50_vals = t50_vals[np.isfinite(t50_vals) & (t50_vals > 0)]
        fog_vals = g["fog"]
        fog_vals = fog_vals[np.isfinite(fog_vals) & (fog_vals > 0)]
        rea20_vals = g["rea_at_20_percent"]
        rea20_vals = rea20_vals[np.isfinite(rea20_vals)]
        tgt_vals = g["t50_target_rea_percent"]
        tgt_vals = tgt_vals[np.isfinite(tgt_vals)]
        use_flags = g["use_for_bo"].fillna(True).astype(bool)

        mean_t50 = float(t50_vals.mean()) if len(t50_vals) > 0 else np.nan
        mean_fog = float(fog_vals.mean()) if len(fog_vals) > 0 else np.nan
        out_rows.append(
            {
                "run_id": group_run_id,
                "polymer_id": pid,
                "t50_min": mean_t50,
                "t50_censored": int(0 if np.isfinite(mean_t50) else 1),
                "fog": mean_fog,
                "log_fog": float(np.log(mean_fog)) if np.isfinite(mean_fog) and mean_fog > 0 else np.nan,
                "t50_definition": t50_definition,
                "source_run_ids": "|".join(src_runs),
                "n_source_runs": int(len(src_runs)),
                "n_t50": int(len(t50_vals)),
                "n_fog": int(len(fog_vals)),
                "t50_target_rea_percent": float(tgt_vals.mean()) if len(tgt_vals) > 0 else np.nan,
                "rea_at_20_percent": float(rea20_vals.mean()) if len(rea20_vals) > 0 else np.nan,
                "use_for_bo": bool(use_flags.all()) if len(use_flags) > 0 else True,
                "fog_missing_reason": "" if len(fog_vals) > 0 else "missing_in_all_source_runs",
            }
        )
    return pd.DataFrame(out_rows, columns=base_cols)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Across run-group aggregation: per-polymer mean-fit plots with SEM error bars, "
            "and mean t50/FoG ranking bars."
        )
    )
    p.add_argument("--run_id", required=True, help="Anchor run_id.")
    p.add_argument(
        "--processed_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Processed root containing {run_id}/fit outputs.",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "across_runs",
        help="Output root for cross-run grouped outputs.",
    )
    p.add_argument(
        "--run_ids",
        nargs="*",
        default=None,
        help="Optional explicit run_id list. If omitted, runs are selected from run group TSV.",
    )
    p.add_argument(
        "--run_group_tsv",
        type=Path,
        default=REPO_ROOT / "meta" / "run_group_map.tsv",
        help=(
            "TSV to control run grouping and inclusion. "
            "Columns: run_id, group_id, include_in_group_mean."
        ),
    )
    p.add_argument(
        "--ignore_run_group_tsv",
        action="store_true",
        help="Ignore --run_group_tsv (then only --run_ids or anchor run is used).",
    )
    # Backward-compatible aliases (hidden)
    p.add_argument("--same_date_include_tsv", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument("--ignore_same_date_include_tsv", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--group_id",
        default=None,
        help="Optional group_id override. Useful when selecting by group directly.",
    )
    p.add_argument(
        "--group_run_id",
        default=None,
        help="Optional output run_id label (default: {group_id}-group_mean).",
    )
    p.add_argument(
        "--t50_definition",
        type=str,
        default="y0_half",
        choices=["y0_half", "rea50"],
        help="t50 definition to aggregate from fog_summary files.",
    )
    p.add_argument("--dpi", type=int, default=600, help="PNG dpi.")
    p.add_argument("--dry_run", action="store_true", help="Print planned actions without writing files.")
    p.add_argument("--debug", action="store_true", help="Verbose logging.")
    args = p.parse_args()

    run_id = str(args.run_id).strip()
    if not run_id:
        raise ValueError("--run_id must be non-empty.")
    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_dir():
        raise FileNotFoundError(f"processed_dir not found: {processed_dir}")

    t50_definition = normalize_t50_definition(args.t50_definition)

    # Backward compatibility for old args
    ignore_group_tsv = bool(args.ignore_run_group_tsv) or bool(args.ignore_same_date_include_tsv)
    run_group_tsv_path = Path(args.run_group_tsv)
    if args.same_date_include_tsv is not None:
        run_group_tsv_path = Path(args.same_date_include_tsv)
    if ignore_group_tsv:
        run_group_tsv_path = None

    run_group_table = _load_run_group_table(run_group_tsv_path)
    candidate_runs, resolved_group_id = _collect_group_runs(
        run_id=run_id,
        explicit_runs=args.run_ids,
        run_group_table=run_group_table,
        group_id_override=(str(args.group_id).strip() if args.group_id else None),
    )
    if not candidate_runs:
        raise ValueError("No candidate runs selected for grouped aggregation.")

    group_run_id = (
        str(args.group_run_id).strip()
        if args.group_run_id and str(args.group_run_id).strip()
        else f"{_safe_group_stem(resolved_group_id)}-group_mean"
    )
    output_root = Path(args.out_dir) / group_run_id
    ranking_dir = output_root / "ranking"
    plot_root = output_root / "plots"
    plot_group_label = _safe_group_stem(resolved_group_id)

    summary_paths, fog_paths = _existing_paths_for_runs(processed_dir, candidate_runs)
    runs_for_plot = sorted({p.parents[1].name for p in summary_paths})
    runs_for_fog = sorted({p.parents[1].name for p in fog_paths})

    if args.debug or args.dry_run:
        print("anchor run_id:", run_id)
        print("group_id:", resolved_group_id)
        print("output run_id:", group_run_id)
        print("candidate runs:", candidate_runs)
        if run_group_tsv_path is not None:
            print("run-group TSV:", run_group_tsv_path)
            print("run-group entries:", len(run_group_table))
        else:
            print("run-group TSV: ignored")
        print("runs with summary_simple:", runs_for_plot)
        print("runs with fog_summary:", runs_for_fog)
        print("t50 definition:", t50_definition)
        print("output root:", output_root)

    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)

    # 1) Per-polymer plots across grouped runs (mean fit + SEM error bars)
    across_plot_dir = plot_per_polymer_timeseries_across_runs_with_error_bars(
        run_id=run_id,
        processed_dir=processed_dir,
        out_fit_dir=plot_root,
        color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
        same_date_runs=runs_for_plot,
        group_label=plot_group_label,
        dpi=int(args.dpi),
    )
    if across_plot_dir is None:
        print("Skipped grouped plots: need at least 2 runs with summary_simple.csv.")
    else:
        print(f"Saved (grouped plots): {across_plot_dir}")

    # 2) Mean t50/FoG aggregation and ranking outputs
    agg_fog_df = _aggregate_same_date_fog(
        run_ids=runs_for_fog,
        processed_dir=processed_dir,
        group_run_id=group_run_id,
        t50_definition=t50_definition,
    )
    fog_summary_path = output_root / f"fog_summary__{group_run_id}.csv"
    agg_fog_df.to_csv(fog_summary_path, index=False)
    print(f"Saved (grouped mean FoG summary): {fog_summary_path}")

    if not agg_fog_df.empty:
        ranking_outputs = write_run_ranking_outputs(
            fog_df=agg_fog_df,
            run_id=group_run_id,
            out_dir=ranking_dir,
            color_map_path=REPO_ROOT / "meta" / "polymer_colors.yml",
        )
        if "t50_ranking_csv" in ranking_outputs:
            print(f"Saved (mean t50 ranking): {ranking_outputs['t50_ranking_csv']}")
        if "fog_ranking_csv" in ranking_outputs:
            print(f"Saved (mean FoG ranking): {ranking_outputs['fog_ranking_csv']}")
        if "t50_ranking_png" in ranking_outputs:
            print(f"Saved (mean t50 ranking plot): {ranking_outputs['t50_ranking_png']}")
        if "fog_ranking_png" in ranking_outputs:
            print(f"Saved (mean FoG ranking plot): {ranking_outputs['fog_ranking_png']}")
    else:
        print("No rows were aggregated for grouped mean FoG; ranking outputs were skipped.")

    used_runs_path = output_root / f"source_runs__{group_run_id}.txt"
    with open(used_runs_path, "w", encoding="utf-8") as f:
        f.write("group_id\t" + str(resolved_group_id) + "\n")
        f.write("candidate_runs\t" + ",".join(candidate_runs) + "\n")
        f.write("runs_for_plot\t" + ",".join(runs_for_plot) + "\n")
        f.write("runs_for_fog\t" + ",".join(runs_for_fog) + "\n")
        f.write("t50_definition\t" + t50_definition + "\n")
    print(f"Saved (source runs): {used_runs_path}")

    manifest_inputs: list[Path] = []
    manifest_inputs.extend(summary_paths)
    manifest_inputs.extend(fog_paths)
    manifest = build_run_manifest_dict(
        group_run_id,
        manifest_inputs,
        git_root=REPO_ROOT,
        extra={
            "anchor_run_id": run_id,
            "group_id": str(resolved_group_id),
            "candidate_runs": candidate_runs,
            "run_group_tsv": (str(run_group_tsv_path.resolve()) if run_group_tsv_path and run_group_tsv_path.exists() else ""),
            "runs_for_plot": runs_for_plot,
            "runs_for_fog": runs_for_fog,
            "t50_definition": t50_definition,
            "outputs": {
                "fog_summary": str(fog_summary_path.resolve()),
                "ranking_dir": str(ranking_dir.resolve()),
                "plots_dir": str(across_plot_dir.resolve()) if across_plot_dir is not None else "",
            },
        },
    )
    manifest_path = output_root / f"run_manifest__{group_run_id}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved (manifest): {manifest_path}")


if __name__ == "__main__":
    main()
