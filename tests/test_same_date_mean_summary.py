from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_same_date_mean_summary import (  # noqa: E402
    _aggregate_same_date_fog,
    _collect_all_round_runs,
    _collect_group_runs,
    _collect_run_top_t50,
    _collect_same_date_runs,
    _load_run_group_table,
    _load_same_date_include_map,
    _plot_run_top_t50,
)


class TestSameDateMeanSummary(unittest.TestCase):
    def test_collect_all_round_runs_uses_only_valid_round_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_round_tsv = root / "bo_run_round_map.tsv"
            run_round_tsv.write_text(
                "run_id\tround_id\n"
                "260211-1\tR1\n"
                "260211-2\tR2\n"
                "260211-3\t—\n"
                "260211-4\t\n",
                encoding="utf-8",
            )
            runs = _collect_all_round_runs(run_round_tsv)
            self.assertEqual(runs, ["260211-1", "260211-2"])

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

    def test_aggregate_same_date_fog_excludes_reference_abs0_outlier_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processed = root / "processed"
            run_ids = ["r1", "r2", "r3", "r4"]
            for rid in run_ids:
                fit_dir = processed / rid / "fit"
                fit_dir.mkdir(parents=True, exist_ok=True)

            rows_by_run = {
                "r1": [
                    {"run_id": "r1", "polymer_id": "GOX", "t50_min": 10.0, "fog": 1.0, "t50_definition": "y0_half", "abs_activity_at_0": 10.0},
                    {"run_id": "r1", "polymer_id": "P1", "t50_min": 20.0, "fog": 2.0, "t50_definition": "y0_half", "abs_activity_at_0": 12.0},
                ],
                "r2": [
                    {"run_id": "r2", "polymer_id": "GOX", "t50_min": 10.0, "fog": 1.0, "t50_definition": "y0_half", "abs_activity_at_0": 10.5},
                    {"run_id": "r2", "polymer_id": "P1", "t50_min": 21.0, "fog": 2.1, "t50_definition": "y0_half", "abs_activity_at_0": 11.8},
                ],
                "r3": [
                    {"run_id": "r3", "polymer_id": "GOX", "t50_min": 10.0, "fog": 1.0, "t50_definition": "y0_half", "abs_activity_at_0": 9.5},
                    {"run_id": "r3", "polymer_id": "P1", "t50_min": 19.0, "fog": 1.9, "t50_definition": "y0_half", "abs_activity_at_0": 11.9},
                ],
                "r4": [
                    {"run_id": "r4", "polymer_id": "GOX", "t50_min": 10.0, "fog": 1.0, "t50_definition": "y0_half", "abs_activity_at_0": 80.0},
                    {"run_id": "r4", "polymer_id": "P1", "t50_min": 100.0, "fog": 10.0, "t50_definition": "y0_half", "abs_activity_at_0": 50.0},
                ],
            }
            for rid, rows in rows_by_run.items():
                pd.DataFrame(rows).to_csv(processed / rid / "fit" / f"fog_summary__{rid}.csv", index=False)

            outlier_events: list[dict[str, object]] = []
            out = _aggregate_same_date_fog(
                run_ids=run_ids,
                processed_dir=processed,
                group_run_id="group-x",
                t50_definition="y0_half",
                reference_polymer_id="GOX",
                apply_outlier_filter=True,
                outlier_min_runs=4,
                outlier_z_threshold=3.5,
                outlier_ratio_low=0.33,
                outlier_ratio_high=3.0,
                reference_abs0_outlier_low=0.5,
                reference_abs0_outlier_high=2.0,
                outlier_min_keep=2,
                outlier_events=outlier_events,
            )
            p1 = out[out["polymer_id"] == "P1"].iloc[0]
            self.assertEqual(int(p1["n_source_runs"]), 3)
            self.assertNotIn("r4", str(p1["source_run_ids"]))
            self.assertAlmostEqual(float(p1["fog"]), 2.0, places=6)
            self.assertTrue(any(e.get("event_type") == "reference_run_excluded" for e in outlier_events))

    def test_aggregate_same_date_fog_excludes_polymer_outlier_row(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processed = root / "processed"
            run_ids = ["r1", "r2", "r3", "r4", "r5"]
            for rid in run_ids:
                fit_dir = processed / rid / "fit"
                fit_dir.mkdir(parents=True, exist_ok=True)

            fogs = {"r1": 2.0, "r2": 2.1, "r3": 1.9, "r4": 2.2, "r5": 100.0}
            for rid in run_ids:
                pd.DataFrame(
                    [
                        {"run_id": rid, "polymer_id": "GOX", "t50_min": 10.0, "fog": 1.0, "t50_definition": "y0_half", "abs_activity_at_0": 10.0},
                        {"run_id": rid, "polymer_id": "P1", "t50_min": 20.0, "fog": fogs[rid], "log_fog": float(np.log(fogs[rid])), "t50_definition": "y0_half", "abs_activity_at_0": 12.0},
                    ]
                ).to_csv(processed / rid / "fit" / f"fog_summary__{rid}.csv", index=False)

            outlier_events: list[dict[str, object]] = []
            out = _aggregate_same_date_fog(
                run_ids=run_ids,
                processed_dir=processed,
                group_run_id="group-y",
                t50_definition="y0_half",
                reference_polymer_id="GOX",
                apply_outlier_filter=True,
                outlier_min_runs=4,
                outlier_z_threshold=3.5,
                outlier_ratio_low=0.33,
                outlier_ratio_high=3.0,
                reference_abs0_outlier_low=0.5,
                reference_abs0_outlier_high=2.0,
                outlier_min_keep=2,
                outlier_events=outlier_events,
            )
            p1 = out[out["polymer_id"] == "P1"].iloc[0]
            self.assertEqual(int(p1["n_source_runs"]), 4)
            self.assertNotIn("r5", str(p1["source_run_ids"]))
            self.assertAlmostEqual(float(p1["fog"]), 2.05, places=6)
            self.assertTrue(any(e.get("event_type") == "polymer_metric_outlier_excluded" for e in outlier_events))

    def test_collect_run_top_t50_collects_top3_per_source_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            processed = root / "processed"
            run_ids = ["r1", "r2"]
            for rid in run_ids:
                (processed / rid / "fit").mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {"run_id": "r1", "polymer_id": "GOX", "t50_min": 10.0, "t50_definition": "y0_half"},
                    {"run_id": "r1", "polymer_id": "GOx with DMSO", "t50_min": 99.0, "t50_definition": "y0_half"},
                    {"run_id": "r1", "polymer_id": "GOx with EtOH", "t50_min": 98.0, "t50_definition": "y0_half"},
                    {"run_id": "r1", "polymer_id": "P1", "t50_min": 22.0, "t50_definition": "y0_half"},
                    {"run_id": "r1", "polymer_id": "P2", "t50_min": 20.0, "t50_definition": "y0_half"},
                    {"run_id": "r1", "polymer_id": "P3", "t50_min": 18.0, "t50_definition": "y0_half"},
                    {"run_id": "r1", "polymer_id": "P4", "t50_min": 5.0, "t50_definition": "y0_half"},
                ]
            ).to_csv(processed / "r1" / "fit" / "fog_summary__r1.csv", index=False)
            pd.DataFrame(
                [
                    {"run_id": "r2", "polymer_id": "GOX", "t50_min": 11.0, "t50_definition": "y0_half"},
                    {"run_id": "r2", "polymer_id": "Q1", "t50_min": 30.0, "t50_definition": "y0_half"},
                    {"run_id": "r2", "polymer_id": "Q2", "t50_min": 16.0, "t50_definition": "y0_half"},
                ]
            ).to_csv(processed / "r2" / "fit" / "fog_summary__r2.csv", index=False)

            out = _collect_run_top_t50(
                run_ids=run_ids,
                processed_dir=processed,
                group_run_id="grp-group_mean",
                t50_definition="y0_half",
                reference_polymer_id="GOX",
                top_n=3,
            )
            self.assertTrue((out["run_id"] == "grp-group_mean").all())
            self.assertNotIn("GOX", set(out["polymer_id"].astype(str)))
            self.assertNotIn("GOx with DMSO", set(out["polymer_id"].astype(str)))
            self.assertNotIn("GOx with EtOH", set(out["polymer_id"].astype(str)))
            self.assertEqual(int((out["source_run_id"] == "r1").sum()), 3)
            self.assertEqual(int((out["source_run_id"] == "r2").sum()), 2)

            r1 = out[out["source_run_id"] == "r1"].sort_values("rank_in_source_run")
            self.assertEqual(r1["polymer_id"].tolist(), ["P1", "P2", "P3"])
            self.assertEqual(r1["rank_in_source_run"].tolist(), [1, 2, 3])
            self.assertEqual(r1["t50_min"].tolist(), [22.0, 20.0, 18.0])

    def test_plot_run_top_t50_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plot_path = root / "run_top3_t50__grp-group_mean.png"
            table = pd.DataFrame(
                [
                    {
                        "run_id": "grp-group_mean",
                        "source_run_id": "r1",
                        "rank_in_source_run": 1,
                        "polymer_id": "P1",
                        "t50_min": 20.0,
                        "t50_definition": "y0_half",
                    },
                    {
                        "run_id": "grp-group_mean",
                        "source_run_id": "r1",
                        "rank_in_source_run": 2,
                        "polymer_id": "P2",
                        "t50_min": 18.0,
                        "t50_definition": "y0_half",
                    },
                    {
                        "run_id": "grp-group_mean",
                        "source_run_id": "r2",
                        "rank_in_source_run": 1,
                        "polymer_id": "Q1",
                        "t50_min": 25.0,
                        "t50_definition": "y0_half",
                    },
                    {
                        "run_id": "grp-group_mean",
                        "source_run_id": "r2",
                        "rank_in_source_run": 2,
                        "polymer_id": "Q2",
                        "t50_min": 15.0,
                        "t50_definition": "y0_half",
                    },
                ]
            )

            wrote = _plot_run_top_t50(
                table,
                out_path=plot_path,
                color_map_path=root / "polymer_colors.yml",
                top_n=3,
            )
            self.assertTrue(wrote)
            self.assertTrue(plot_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 0)

if __name__ == "__main__":
    unittest.main()
