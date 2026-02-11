from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_same_date_mean_summary import (  # noqa: E402
    _aggregate_same_date_fog,
    _collect_group_runs,
    _collect_same_date_runs,
    _load_run_group_table,
    _load_same_date_include_map,
)


class TestSameDateMeanSummary(unittest.TestCase):
    def test_aggregate_same_date_fog_means_values(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processed = root / "processed"
            run_ids = ["260205-R1", "260205-R2"]
            for rid in run_ids:
                fit_dir = processed / rid / "fit"
                fit_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {"run_id": "260205-R1", "polymer_id": "P1", "t50_min": 20.0, "fog": 2.0, "t50_definition": "y0_half"},
                    {"run_id": "260205-R1", "polymer_id": "P2", "t50_min": 10.0, "fog": 1.0, "t50_definition": "y0_half"},
                ]
            ).to_csv(processed / "260205-R1" / "fit" / "fog_summary__260205-R1.csv", index=False)
            pd.DataFrame(
                [
                    {"run_id": "260205-R2", "polymer_id": "P1", "t50_min": 30.0, "fog": 4.0, "t50_definition": "y0_half"},
                    {"run_id": "260205-R2", "polymer_id": "P2", "t50_min": 20.0, "fog": 2.0, "t50_definition": "y0_half"},
                ]
            ).to_csv(processed / "260205-R2" / "fit" / "fog_summary__260205-R2.csv", index=False)

            out = _aggregate_same_date_fog(
                run_ids=run_ids,
                processed_dir=processed,
                group_run_id="260205-same_date_mean",
                t50_definition="y0_half",
            )
            self.assertEqual(set(out["polymer_id"].tolist()), {"P1", "P2"})
            p1 = out[out["polymer_id"] == "P1"].iloc[0]
            p2 = out[out["polymer_id"] == "P2"].iloc[0]
            self.assertAlmostEqual(float(p1["t50_min"]), 25.0, places=6)
            self.assertAlmostEqual(float(p1["fog"]), 3.0, places=6)
            self.assertAlmostEqual(float(p2["t50_min"]), 15.0, places=6)
            self.assertAlmostEqual(float(p2["fog"]), 1.5, places=6)
            self.assertTrue((out["run_id"] == "260205-same_date_mean").all())

    def test_aggregate_same_date_fog_skips_mismatched_definition(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processed = root / "processed"
            run_ids = ["260205-R1", "260205-R2"]
            for rid in run_ids:
                fit_dir = processed / rid / "fit"
                fit_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {"run_id": "260205-R1", "polymer_id": "P1", "t50_min": 20.0, "fog": 2.0, "t50_definition": "y0_half"},
                ]
            ).to_csv(processed / "260205-R1" / "fit" / "fog_summary__260205-R1.csv", index=False)
            pd.DataFrame(
                [
                    {"run_id": "260205-R2", "polymer_id": "P1", "t50_min": 100.0, "fog": 10.0, "t50_definition": "rea50"},
                ]
            ).to_csv(processed / "260205-R2" / "fit" / "fog_summary__260205-R2.csv", index=False)

            out = _aggregate_same_date_fog(
                run_ids=run_ids,
                processed_dir=processed,
                group_run_id="260205-same_date_mean",
                t50_definition="y0_half",
            )
            self.assertEqual(len(out), 1)
            p1 = out.iloc[0]
            self.assertAlmostEqual(float(p1["t50_min"]), 20.0, places=6)
            self.assertAlmostEqual(float(p1["fog"]), 2.0, places=6)

    def test_run_group_tsv_can_include_different_dates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processed = root / "processed"
            for rid in ["260205-R1", "260206-R1", "260207-R1"]:
                fit_dir = processed / rid / "fit"
                fit_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    [{"polymer_id": "P1", "heat_min": 0, "abs_activity": 1.0, "REA_percent": 100.0}]
                ).to_csv(fit_dir / "summary_simple.csv", index=False)

            include_tsv = root / "run_group_map.tsv"
            include_tsv.write_text(
                "run_id\tgroup_id\tinclude_in_group_mean\tnotes\n"
                "260205-R1\tcondA\tTrue\t\n"
                "260206-R1\tcondA\tTrue\t\n"
                "260207-R1\tcondB\tTrue\t\n",
                encoding="utf-8",
            )

            run_group_table = _load_run_group_table(include_tsv)
            runs, group_id = _collect_group_runs(
                run_id="260205-R1",
                explicit_runs=None,
                run_group_table=run_group_table,
            )
            self.assertEqual(group_id, "condA")
            self.assertEqual(runs, ["260205-R1", "260206-R1"])

            # Backward-compat helpers still work with the new TSV schema.
            include_map = _load_same_date_include_map(include_tsv)
            runs_compat = _collect_same_date_runs(
                run_id="260205-R1",
                processed_dir=processed,
                explicit_runs=None,
                same_date_include_map=include_map,
            )
            self.assertIn("260205-R1", runs_compat)


if __name__ == "__main__":
    unittest.main()
