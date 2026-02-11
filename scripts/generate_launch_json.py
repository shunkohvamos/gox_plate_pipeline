"""
Scan data/raw and data/meta for raw + row_map pairs and update .vscode/launch.json
so each dataset gets:
  - "Extract clean CSV (run_id)"
  - "Fit rates+REA [t50=y0/2] (run_id)"    # default: no per-well fit PNGs
  - "Fit rates+REA [t50=REA50] (run_id)"   # default: no per-well fit PNGs
  - "Well plots only (run_id)"             # on-demand: per-well fit PNGs only
  - "Well plots only (Debug) (run_id)"     # same as above + extra counters
  - "Group mean plots+ranking [t50=y0/2] (run_id)"   # across runs (grouped)
  - "Group mean plots+ranking [t50=REA50] (run_id)"  # across runs (grouped)
  - "Group mean plots+ranking (Debug) [t50=y0/2] (run_id)"
  - "Group mean plots+ranking (Debug) [t50=REA50] (run_id)"

Convention:
  - Raw (recommended): data/raw/{run_id}/*.csv  (folder = one experimental batch)
  - Raw (legacy):      data/raw/*.csv
  - Row map: data/meta/{run_id}.tsv (tab-separated)
  - Run ID = raw folder name (recommended) or raw file stem (legacy)

Run this script after adding new raw data + TSV, or use the VS Code task
"Generate launch.json from data" (Terminal > Run Task).

Static setup entries also include:
  - "全フォルダ–Round対応TSVを出力"
  - "全フォルダ–集計グループTSVを出力"
"""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _with_presentation(cfg: dict, *, group: str, order: int) -> dict:
    """
    Add VS Code presentation metadata for clearer Run and Debug organization.

    group: logical category in dropdown (same group name -> visually grouped)
    order: ordering inside each group
    """
    out = dict(cfg)
    out["presentation"] = {
        "group": str(group),
        "order": int(order),
    }
    return out


def discover_datasets(repo_root: Path) -> list[tuple[str, Path, Path]]:
    """Return list of (run_id, raw_path, row_map_path) for each valid pair."""
    raw_dir = repo_root / "data" / "raw"
    meta_dir = repo_root / "data" / "meta"
    if not raw_dir.is_dir():
        return []

    pairs: list[tuple[str, Path, Path]] = []
    seen_run_ids: set[str] = set()

    # Preferred layout: data/raw/{run_id}/*.csv
    for raw_path in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        csvs = sorted(p for p in raw_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
        if not csvs:
            continue
        run_id = raw_path.name
        row_map = meta_dir / f"{run_id}.tsv"
        if not row_map.is_file():
            row_map = meta_dir / f"{run_id}_row_map.tsv"
        if row_map.is_file():
            pairs.append((run_id, raw_path, row_map))
            seen_run_ids.add(run_id)

    # Legacy layout: data/raw/{run_id}.csv
    for raw_path in sorted(raw_dir.glob("*.csv")):
        run_id = raw_path.stem
        if run_id in seen_run_ids:
            continue
        row_map = meta_dir / f"{run_id}.tsv"
        if not row_map.is_file():
            row_map = meta_dir / f"{run_id}_row_map.tsv"
        if row_map.is_file():
            pairs.append((run_id, raw_path, row_map))

    return pairs


def build_extract_config(run_id: str, raw_path: Path, row_map_path: Path, repo_root: Path) -> dict:
    raw_rel = raw_path.relative_to(repo_root).as_posix()
    row_rel = row_map_path.relative_to(repo_root).as_posix()
    return {
        "name": f"Extract clean CSV ({run_id})",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/extract_clean_csv.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--raw", raw_rel,
            "--row_map", row_rel,
            "--config", "meta/config.yml",
        ],
        "justMyCode": True,
    }


def build_fit_config(
    run_id: str,
    *,
    t50_definition: str = "y0_half",
) -> dict:
    tidy_rel = f"data/processed/{run_id}/extract/tidy.csv"
    if str(t50_definition).strip().lower() == "rea50":
        name = f"Fit rates+REA [t50=REA50] ({run_id})"
    else:
        name = f"Fit rates+REA [t50=y0/2] ({run_id})"
    args = [
        "--tidy", tidy_rel,
        "--config", "meta/config.yml",
        "--out_dir", "data/processed",
        "--write_well_plots", "0",
        "--min_points", "6",
        "--max_points", "30",
        "--r2_min", "0.97",
        "--slope_min", "0.0",
        "--max_t_end", "600",
        "--min_span_s", "0",
        "--min_delta_y", "0",
        "--t50_definition", str(t50_definition),
    ]
    return {
        "name": name,
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/fit_initial_rates.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "python": "${workspaceFolder}/.venv/bin/python",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": args,
        "justMyCode": True,
    }


def build_well_plots_only_config(run_id: str, *, debug: bool = False) -> dict:
    tidy_rel = f"data/processed/{run_id}/extract/tidy.csv"
    name = f"Well plots only (Debug) ({run_id})" if debug else f"Well plots only ({run_id})"
    args = [
        "--tidy", tidy_rel,
        "--config", "meta/config.yml",
        "--out_dir", "data/processed",
        "--min_points", "6",
        "--max_points", "30",
        "--r2_min", "0.97",
        "--slope_min", "0.0",
        "--max_t_end", "600",
        "--min_span_s", "0",
        "--min_delta_y", "0",
        "--plot_mode", "all",
    ]
    if debug:
        args.append("--debug")
    return {
        "name": name,
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/generate_well_plots_only.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "python": "${workspaceFolder}/.venv/bin/python",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": args,
        "justMyCode": True,
    }


def build_group_mean_summary_config(
    run_id: str,
    *,
    t50_definition: str = "y0_half",
    debug: bool = False,
) -> dict:
    if str(t50_definition).strip().lower() == "rea50":
        mode_label = "REA50"
    else:
        mode_label = "y0/2"
    if debug:
        name = f"Group mean plots+ranking (Debug) [t50={mode_label}] ({run_id})"
    else:
        name = f"Group mean plots+ranking [t50={mode_label}] ({run_id})"
    args = [
        "--run_id", str(run_id),
        "--processed_dir", "data/processed",
        "--out_dir", "data/processed/across_runs",
        "--run_group_tsv", "meta/run_group_map.tsv",
        "--t50_definition", str(t50_definition),
    ]
    if debug:
        args.append("--debug")
    return {
        "name": name,
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/run_group_mean_summary.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "python": "${workspaceFolder}/.venv/bin/python",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": args,
        "justMyCode": True,
    }


def is_generated_config(name: str) -> bool:
    return (
        name.startswith("Extract clean CSV (")
        or name.startswith("Fit rates+REA ")
        or name.startswith("Well plots only ")
        or name.startswith("Group mean plots+ranking ")
        or name.startswith("Same-date mean plots+ranking ")
    )


def build_generate_launch_config() -> dict:
    """Launch config to run this script from Run and Debug dropdown (no args)."""
    return {
        "name": "Generate launch.json from data",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/generate_launch_json.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [],
        "justMyCode": True,
    }


def build_generate_tsv_template_config() -> dict:
    """Launch config to generate row-map TSV template from raw (no args = all raw without TSV)."""
    return {
        "name": "Generate TSV template from raw",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/generate_row_map_template.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [],
        "justMyCode": True,
    }


def build_generate_tsv_template_overwrite_config() -> dict:
    """Launch config to regenerate row-map TSV templates for all raw inputs (overwrite existing TSV)."""
    return {
        "name": "Generate TSV template from raw (overwrite)",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/generate_row_map_template.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": ["--overwrite"],
        "justMyCode": True,
    }


def build_run_round_tsv_config() -> dict:
    """Launch config to output TSV of all run folders and their BO round (run_id, round_id)."""
    return {
        "name": "全フォルダ–Round対応TSVを出力",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/generate_run_round_tsv.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [],
        "justMyCode": True,
    }


def build_run_group_tsv_config() -> dict:
    """Launch config to output TSV for cross-run grouping control."""
    return {
        "name": "全フォルダ–集計グループTSVを出力",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/generate_run_group_tsv.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [],
        "justMyCode": True,
    }


def build_extract_all_config(*, dry_run: bool = False) -> dict:
    """Run extract for all discovered runs."""
    name = "Extract clean CSV 全run (Dry run)" if dry_run else "Extract clean CSV 全run"
    args = [
        "--processed_dir", "data/processed",
        "--config", "meta/config.yml",
    ]
    if dry_run:
        args.append("--dry_run")
    return {
        "name": name,
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/run_extract_all.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": args,
        "justMyCode": True,
    }


def build_fit_all_config(*, t50_definition: str = "y0_half", dry_run: bool = False) -> dict:
    """Run fit (and extract-if-missing) for all runs."""
    if dry_run:
        name = "Fit rates+REA 全run (Dry run)"
    elif str(t50_definition).strip().lower() == "rea50":
        name = "Fit rates+REA 全run [t50=REA50]"
    else:
        name = "Fit rates+REA 全run [t50=y0/2]"
    args = [
        "--processed_dir", "data/processed",
        "--config", "meta/config.yml",
        "--t50_definition", str(t50_definition),
    ]
    if dry_run:
        args.append("--dry_run")
    return {
        "name": name,
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/run_fit_all.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": args,
        "justMyCode": True,
    }


def build_run_fit_then_round_fog_config(*, t50_definition: str = "y0_half") -> dict:
    """Run Fit+REA on all round-associated runs, then build round-averaged FoG + GOx traceability."""
    if str(t50_definition).strip().lower() == "rea50":
        name = "Fit+REA 全run → Round平均FoGまとめ [t50=REA50]"
    else:
        name = "Fit+REA 全run → Round平均FoGまとめ [t50=y0/2]"
    return {
        "name": name,
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/run_fit_then_round_fog.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--run_round_map", "meta/bo_run_round_map.tsv",
            "--processed_dir", "data/processed",
            "--out_fog", "data/processed/fog_round_averaged/fog_round_averaged.csv",
            "--t50_definition", str(t50_definition),
        ],
        "justMyCode": True,
    }


def build_rounds_to_bo_config(*, t50_definition: str = "y0_half", dry_run: bool = False) -> dict:
    """One-shot pipeline: round-assigned extract+fit -> plate-aware FoG -> BO."""
    if dry_run:
        name = "Round指定全run → BO一括 (Dry run)"
    elif str(t50_definition).strip().lower() == "rea50":
        name = "Round指定全run → BO一括 [t50=REA50]"
    else:
        name = "Round指定全run → BO一括 [t50=y0/2]"
    args = [
        "--run_round_map", "meta/bo_run_round_map.tsv",
        "--processed_dir", "data/processed",
        "--config", "meta/config.yml",
        "--t50_definition", str(t50_definition),
        "--out_bo_dir", "data/processed/bo_runs",
        "--n_suggestions", "8",
        "--acquisition", "ei",
    ]
    if dry_run:
        args.append("--dry_run")
    return {
        "name": name,
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/run_rounds_to_bo.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": args,
        "justMyCode": True,
    }


def build_run_fit_then_round_fog_dry_config() -> dict:
    """Dry run: only print what would be run (extract/fit per run, then round-averaged FoG)."""
    return {
        "name": "Fit+REA 全run → Round平均FoGまとめ (Dry run)",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/run_fit_then_round_fog.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--run_round_map", "meta/bo_run_round_map.tsv",
            "--dry_run",
        ],
        "justMyCode": True,
    }


def build_run_fit_then_round_fog_debug_config() -> dict:
    """Same as main but with --debug for verbose output."""
    return {
        "name": "Fit+REA 全run → Round平均FoGまとめ (Debug)",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/run_fit_then_round_fog.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--run_round_map", "meta/bo_run_round_map.tsv",
            "--debug",
        ],
        "justMyCode": True,
    }


def build_fog_plate_aware_config(*, t50_definition: str = "y0_half") -> dict:
    """FoG with denominator rule: same plate GOx → same round GOx (no fit run)."""
    if str(t50_definition).strip().lower() == "rea50":
        name = "FoG（同一プレート→同一ラウンド）計算 [t50=REA50]"
    else:
        name = "FoG（同一プレート→同一ラウンド）計算 [t50=y0/2]"
    return {
        "name": name,
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/build_fog_plate_aware.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--run_round_map", "meta/bo_run_round_map.tsv",
            "--processed_dir", "data/processed",
            "--out_dir", "data/processed/fog_plate_aware",
            "--t50_definition", str(t50_definition),
        ],
        "justMyCode": True,
    }


def build_fog_plate_aware_dry_config() -> dict:
    """Dry run: only check inputs and list outputs for FoG (same plate → same round)."""
    return {
        "name": "FoG（同一プレート→同一ラウンド）Dry run",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/build_fog_plate_aware.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--run_round_map", "meta/bo_run_round_map.tsv",
            "--t50_definition", "y0_half",
            "--dry_run",
        ],
        "justMyCode": True,
    }


def build_fog_plate_aware_exclude_outlier_config() -> dict:
    """FoG with denominator rule: same plate GOx → same round GOx, excluding outlier GOx t50."""
    return {
        "name": "FoG（同一プレート→同一ラウンド）計算（異常GOx除外）",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/build_fog_plate_aware.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--run_round_map", "meta/bo_run_round_map.tsv",
            "--processed_dir", "data/processed",
            "--out_dir", "data/processed",
            "--t50_definition", "y0_half",
            "--exclude_outlier_gox",
        ],
        "justMyCode": True,
    }


def build_bo_learning_data_config() -> dict:
    """Build BO learning CSV from round-averaged FoG (same-run GOx denominator)."""
    return {
        "name": "BO学習データ作成（Round平均FoG）",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/build_bo_learning_data.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--catalog", "meta/bo_catalog_bma.csv",
            "--run_round_map", "meta/bo_run_round_map.tsv",
            "--fog_round_averaged", "data/processed/fog_round_averaged/fog_round_averaged.csv",
            "--out", "data/processed/bo_learning/bo_learning.csv",
            "--exclusion_report", "data/processed/bo_learning/bo_learning_excluded.csv",
        ],
        "justMyCode": True,
    }


def build_bo_learning_data_plate_aware_config() -> dict:
    """Build BO learning CSV from plate-aware round-averaged FoG (same-plate → same-round GOx denominator)."""
    return {
        "name": "BO学習データ作成（Plate-aware Round平均FoG）",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/build_bo_learning_data.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--catalog", "meta/bo_catalog_bma.csv",
            "--run_round_map", "meta/bo_run_round_map.tsv",
            "--fog_round_averaged", "data/processed/fog_plate_aware/fog_plate_aware_round_averaged.csv",
            "--out", "data/processed/bo_learning/bo_learning_plate_aware.csv",
            "--exclusion_report", "data/processed/bo_learning/bo_learning_excluded_plate_aware.csv",
        ],
        "justMyCode": True,
    }


def build_bayesian_optimization_config() -> dict:
    """Run pure-regression Bayesian optimization from plate-aware learning data (with rebuild)."""
    return {
        "name": "Bayesian Optimization（Pure Regression / Plate-aware）",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/run_bayesian_optimization.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "python": "${workspaceFolder}/.venv/bin/python",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--rebuild_learning",
            "--catalog", "meta/bo_catalog_bma.csv",
            "--fog_round_averaged", "data/processed/fog_plate_aware/fog_plate_aware_round_averaged.csv",
            "--bo_learning", "data/processed/bo_learning/bo_learning_plate_aware.csv",
            "--exclusion_report", "data/processed/bo_learning/bo_learning_excluded_plate_aware.csv",
            "--fog_plate_aware", "data/processed/fog_plate_aware/fog_plate_aware.csv",
            "--out_dir", "data/processed/bo_runs",
            "--n_suggestions", "8",
            "--acquisition", "ei",
            "--min_component", "0.02",
            "--max_component", "0.95",
            "--min_fraction_distance", "0.06",
            "--objective_column", "log_fog_corrected",
        ],
        "justMyCode": True,
    }


def build_bayesian_optimization_no_rebuild_config() -> dict:
    """Run pure-regression Bayesian optimization from existing bo_learning CSV (fast)."""
    return {
        "name": "Bayesian Optimization（Pure Regression / 既存学習データ）",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/run_bayesian_optimization.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "python": "${workspaceFolder}/.venv/bin/python",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--bo_learning", "data/processed/bo_learning/bo_learning_plate_aware.csv",
            "--fog_plate_aware", "data/processed/fog_plate_aware/fog_plate_aware.csv",
            "--out_dir", "data/processed/bo_runs",
            "--n_suggestions", "8",
            "--acquisition", "ei",
            "--min_component", "0.02",
            "--max_component", "0.95",
            "--min_fraction_distance", "0.06",
            "--objective_column", "log_fog_corrected",
        ],
        "justMyCode": True,
    }


def build_mol_logp_master_config() -> dict:
    """Validate MolLogP master sheet (meta/mol_logp_master.csv). Independent of BO."""
    return {
        "name": "MolLogP マスター確認",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/validate_mol_logp_master.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "python": "${workspaceFolder}/.venv/bin/python",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--master", "meta/mol_logp_master.csv",
        ],
        "justMyCode": True,
    }


def main() -> None:
    repo_root = _repo_root()
    launch_path = repo_root / ".vscode" / "launch.json"

    datasets = discover_datasets(repo_root)

    # Build generated configs (per-run block)
    # Ordering policy:
    #   1) Extract
    #   2) Fit rates+REA [t50=y0/2]
    #   3) Fit rates+REA [t50=REA50]
    #   4) Well plots only
    #   5) Well plots only (Debug)
    #   6) Group mean plots+ranking [t50=y0/2]
    #   7) Group mean plots+ranking [t50=REA50]
    #   8) Group mean plots+ranking (Debug) [t50=y0/2]
    #   9) Group mean plots+ranking (Debug) [t50=REA50]
    generated: list[dict] = []
    for idx, (run_id, raw_path, row_map_path) in enumerate(datasets):
        base = idx * 20
        generated.extend([
            _with_presentation(
                build_extract_config(run_id, raw_path, row_map_path, repo_root),
                group="10 Per-run",
                order=base + 1,
            ),
            _with_presentation(
                build_fit_config(run_id, t50_definition="y0_half"),
                group="10 Per-run",
                order=base + 2,
            ),
            _with_presentation(
                build_fit_config(run_id, t50_definition="rea50"),
                group="10 Per-run",
                order=base + 3,
            ),
            _with_presentation(
                build_well_plots_only_config(run_id, debug=False),
                group="10 Per-run",
                order=base + 4,
            ),
            _with_presentation(
                build_well_plots_only_config(run_id, debug=True),
                group="10 Per-run",
                order=base + 5,
            ),
            _with_presentation(
                build_group_mean_summary_config(run_id, t50_definition="y0_half", debug=False),
                group="12 Group",
                order=base + 1,
            ),
            _with_presentation(
                build_group_mean_summary_config(run_id, t50_definition="rea50", debug=False),
                group="12 Group",
                order=base + 2,
            ),
            _with_presentation(
                build_group_mean_summary_config(run_id, t50_definition="y0_half", debug=True),
                group="12 Group",
                order=base + 3,
            ),
            _with_presentation(
                build_group_mean_summary_config(run_id, t50_definition="rea50", debug=True),
                group="12 Group",
                order=base + 4,
            ),
        ])

    # Merge with existing launch.json: keep non-generated configs, replace generated
    if launch_path.exists():
        with open(launch_path, "r", encoding="utf-8") as f:
            launch = json.load(f)
        configs = launch.get("configurations", [])
        _static_names = (
            "Generate launch.json from data",
            "Generate TSV template from raw",
            "Generate TSV template from raw (overwrite)",
            "全フォルダ–Round対応TSVを出力",
            "全フォルダ–集計グループTSVを出力",
            "全フォルダ–Same-date集計TSVを出力",
            "Extract clean CSV 全run",
            "Extract clean CSV 全run (Dry run)",
            "Fit rates+REA 全run [t50=y0/2]",
            "Fit rates+REA 全run [t50=REA50]",
            "Fit rates+REA 全run (Dry run)",
            "Fit+REA 全run → Round平均FoGまとめ",
            "Fit+REA 全run → Round平均FoGまとめ [t50=y0/2]",
            "Fit+REA 全run → Round平均FoGまとめ [t50=REA50]",
            "Fit+REA 全run → Round平均FoGまとめ (Dry run)",
            "Fit+REA 全run → Round平均FoGまとめ (Debug)",
            "Round指定全run → BO一括 [t50=y0/2]",
            "Round指定全run → BO一括 [t50=REA50]",
            "Round指定全run → BO一括 (Dry run)",
            "FoG（同一プレート→同一ラウンド）計算",
            "FoG（同一プレート→同一ラウンド）計算 [t50=y0/2]",
            "FoG（同一プレート→同一ラウンド）計算 [t50=REA50]",
            "FoG（同一プレート→同一ラウンド）Dry run",
            "FoG（同一プレート→同一ラウンド）計算（異常GOx除外）",
            "BO学習データ作成（Round平均FoG）",
            "BO学習データ作成（Plate-aware Round平均FoG）",
            "Bayesian Optimization（Pure Regression / Plate-aware）",
            "Bayesian Optimization（Pure Regression / 既存学習データ）",
            # Legacy aliases (kept here so generator can clean duplicates)
            "Bayesian Optimization（Plate-aware）",
            "Bayesian Optimization（既存学習データ）",
            "MolLogP マスター確認",
        )
        others = [
            c for c in configs
            if not is_generated_config(c.get("name", ""))
            and c.get("name") not in _static_names
        ]
    else:
        launch = {"version": "0.2.0", "configurations": []}
        others = []

    # Organize static configs by workflow stage
    setup_configs = [
        _with_presentation(build_generate_launch_config(), group="00 Setup", order=1),
        _with_presentation(build_generate_tsv_template_config(), group="00 Setup", order=2),
        _with_presentation(build_generate_tsv_template_overwrite_config(), group="00 Setup", order=3),
        _with_presentation(build_run_round_tsv_config(), group="00 Setup", order=4),
        _with_presentation(build_run_group_tsv_config(), group="00 Setup", order=5),
    ]
    all_run_configs = [
        _with_presentation(build_extract_all_config(dry_run=False), group="15 All-run", order=1),
        _with_presentation(build_extract_all_config(dry_run=True), group="15 All-run", order=2),
        _with_presentation(build_fit_all_config(t50_definition="y0_half", dry_run=False), group="15 All-run", order=3),
        _with_presentation(build_fit_all_config(t50_definition="rea50", dry_run=False), group="15 All-run", order=4),
        _with_presentation(build_fit_all_config(t50_definition="y0_half", dry_run=True), group="15 All-run", order=5),
    ]
    stage_configs = [
        _with_presentation(build_rounds_to_bo_config(t50_definition="y0_half", dry_run=False), group="17 Round→BO", order=1),
        _with_presentation(build_rounds_to_bo_config(t50_definition="rea50", dry_run=False), group="17 Round→BO", order=2),
        _with_presentation(build_rounds_to_bo_config(t50_definition="y0_half", dry_run=True), group="17 Round→BO", order=3),
        _with_presentation(build_run_fit_then_round_fog_config(t50_definition="y0_half"), group="20 Batch Fit", order=1),
        _with_presentation(build_run_fit_then_round_fog_config(t50_definition="rea50"), group="20 Batch Fit", order=2),
        _with_presentation(build_run_fit_then_round_fog_dry_config(), group="20 Batch Fit", order=3),
        _with_presentation(build_run_fit_then_round_fog_debug_config(), group="20 Batch Fit", order=4),
        _with_presentation(build_fog_plate_aware_config(t50_definition="y0_half"), group="30 FoG", order=1),
        _with_presentation(build_fog_plate_aware_config(t50_definition="rea50"), group="30 FoG", order=2),
        _with_presentation(build_fog_plate_aware_dry_config(), group="30 FoG", order=3),
        _with_presentation(build_fog_plate_aware_exclude_outlier_config(), group="30 FoG", order=4),
        _with_presentation(build_bo_learning_data_config(), group="40 BO", order=1),
        _with_presentation(build_bo_learning_data_plate_aware_config(), group="40 BO", order=2),
        _with_presentation(build_bayesian_optimization_config(), group="40 BO", order=3),
        _with_presentation(build_bayesian_optimization_no_rebuild_config(), group="40 BO", order=4),
        _with_presentation(build_mol_logp_master_config(), group="90 Utility", order=1),
    ]

    # So generator configs appear in Run and Debug dropdown
    launch["configurations"] = (
        setup_configs
        + generated
        + all_run_configs
        + stage_configs
        + others
    )
    launch_path.parent.mkdir(parents=True, exist_ok=True)
    with open(launch_path, "w", encoding="utf-8") as f:
        json.dump(launch, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Updated {launch_path}")
    if datasets:
        for run_id, _, _ in datasets:
            print(f"  - {run_id}")
    else:
        print("  (no raw folder/file with matching data/meta/{run_id}.tsv; run 'Generate TSV template from raw' first)")


if __name__ == "__main__":
    main()
