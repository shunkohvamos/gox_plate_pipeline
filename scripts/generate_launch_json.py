"""
Scan data/raw and data/meta for raw + row_map pairs and update .vscode/launch.json
so each dataset gets "Extract clean CSV (run_id)" and "Fit rates+REA (run_id)" configs.

Convention:
  - Raw (recommended): data/raw/{run_id}/*.csv  (folder = one experimental batch)
  - Raw (legacy):      data/raw/*.csv
  - Row map: data/meta/{run_id}.tsv (tab-separated)
  - Run ID = raw folder name (recommended) or raw file stem (legacy)

Run this script after adding new raw data + TSV, or use the VS Code task
"Generate launch.json from data" (Terminal > Run Task).
"""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def build_fit_config(run_id: str) -> dict:
    tidy_rel = f"data/processed/{run_id}/extract/tidy.csv"
    return {
        "name": f"Fit rates+REA ({run_id})",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/fit_initial_rates.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
        "python": "${workspaceFolder}/.venv/bin/python",
        "env": {"PYTHONPATH": "${workspaceFolder}/src"},
        "args": [
            "--tidy", tidy_rel,
            "--config", "meta/config.yml",
            "--out_dir", "data/processed",
            "--plot_dir", "data/processed/plots",
            "--min_points", "6",
            "--max_points", "30",
            "--r2_min", "0.97",
            "--slope_min", "0.0",
            "--max_t_end", "600",
            "--min_span_s", "0",
            "--min_delta_y", "0",
        ],
        "justMyCode": True,
    }


def is_generated_config(name: str) -> bool:
    return name.startswith("Extract clean CSV (") or name.startswith("Fit rates+REA (")


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


def build_run_fit_then_round_fog_config() -> dict:
    """Run Fit+REA on all round-associated runs, then build round-averaged FoG + GOx traceability."""
    return {
        "name": "Fit+REA 全run → Round平均FoGまとめ",
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
        ],
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


def build_fog_plate_aware_config() -> dict:
    """FoG with denominator rule: same plate GOx → same round GOx (no fit run)."""
    return {
        "name": "FoG（同一プレート→同一ラウンド）計算",
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


def main() -> None:
    repo_root = _repo_root()
    launch_path = repo_root / ".vscode" / "launch.json"

    datasets = discover_datasets(repo_root)

    # Build generated configs (Extract + Fit rates+REA per dataset)
    generated: list[dict] = []
    for run_id, raw_path, row_map_path in datasets:
        generated.append(build_extract_config(run_id, raw_path, row_map_path, repo_root))
        generated.append(build_fit_config(run_id))

    # Merge with existing launch.json: keep non-generated configs, replace generated
    if launch_path.exists():
        with open(launch_path, "r", encoding="utf-8") as f:
            launch = json.load(f)
        configs = launch.get("configurations", [])
        _static_names = (
            "Generate launch.json from data",
            "Generate TSV template from raw",
            "全フォルダ–Round対応TSVを出力",
            "Fit+REA 全run → Round平均FoGまとめ",
            "Fit+REA 全run → Round平均FoGまとめ (Dry run)",
            "Fit+REA 全run → Round平均FoGまとめ (Debug)",
            "FoG（同一プレート→同一ラウンド）計算",
            "FoG（同一プレート→同一ラウンド）Dry run",
            "FoG（同一プレート→同一ラウンド）計算（異常GOx除外）",
            "BO学習データ作成（Round平均FoG）",
            "BO学習データ作成（Plate-aware Round平均FoG）",
        )
        others = [
            c for c in configs
            if not is_generated_config(c.get("name", ""))
            and c.get("name") not in _static_names
        ]
    else:
        launch = {"version": "0.2.0", "configurations": []}
        others = []

    # So generator configs appear in Run and Debug dropdown
    launch["configurations"] = (
        generated
        + [
            build_generate_launch_config(),
            build_generate_tsv_template_config(),
            build_run_round_tsv_config(),
            build_run_fit_then_round_fog_config(),
            build_run_fit_then_round_fog_dry_config(),
            build_run_fit_then_round_fog_debug_config(),
            build_fog_plate_aware_config(),
            build_fog_plate_aware_dry_config(),
            build_fog_plate_aware_exclude_outlier_config(),
            build_bo_learning_data_config(),
            build_bo_learning_data_plate_aware_config(),
        ]
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
