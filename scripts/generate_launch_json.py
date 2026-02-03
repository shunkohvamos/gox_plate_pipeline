"""
Scan data/raw and data/meta for raw + row_map pairs and update .vscode/launch.json
so each dataset gets "Extract clean CSV (run_id)" and "Fit rates+REA (run_id)" configs.

Convention:
  - Raw: data/raw/*.csv
  - Row map: data/meta/{stem}.tsv (same stem as raw; tab-separated)
  - Run ID = stem of raw file (e.g. GOx_Concentration_afterheat)

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
    for raw_path in sorted(raw_dir.glob("*.csv")):
        stem = raw_path.name.removesuffix(".csv")
        # Prefer {stem}.tsv; fallback to {stem}_row_map.tsv
        row_map = meta_dir / f"{stem}.tsv"
        if not row_map.is_file():
            row_map = meta_dir / f"{stem}_row_map.tsv"
        if row_map.is_file():
            pairs.append((stem, raw_path, row_map))
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


def build_fit_config(run_id: str, repo_root: Path) -> dict:
    tidy_rel = f"data/processed/{run_id}/extract/tidy.csv"
    return {
        "name": f"Fit rates+REA ({run_id})",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/scripts/fit_initial_rates.py",
        "console": "integratedTerminal",
        "cwd": "${workspaceFolder}",
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


def main() -> None:
    repo_root = _repo_root()
    launch_path = repo_root / ".vscode" / "launch.json"

    datasets = discover_datasets(repo_root)

    # Build generated configs (Extract + Fit per dataset)
    generated: list[dict] = []
    for run_id, raw_path, row_map_path in datasets:
        generated.append(build_extract_config(run_id, raw_path, row_map_path, repo_root))
        generated.append(build_fit_config(run_id, repo_root))

    # Merge with existing launch.json: keep non-generated configs, replace generated
    if launch_path.exists():
        with open(launch_path, "r", encoding="utf-8") as f:
            launch = json.load(f)
        configs = launch.get("configurations", [])
        others = [
            c for c in configs
            if not is_generated_config(c.get("name", ""))
            and c.get("name") not in ("Generate launch.json from data", "Generate TSV template from raw")
        ]
    else:
        launch = {"version": "0.2.0", "configurations": []}
        others = []

    # So "Generate launch.json from data" and "Generate TSV template from raw" appear in Run and Debug dropdown
    launch["configurations"] = (
        generated
        + [build_generate_launch_config(), build_generate_tsv_template_config()]
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
        print("  (no data/raw/*.csv with data/meta/{stem}.tsv; run 'Generate TSV template from raw' first)")


if __name__ == "__main__":
    main()
