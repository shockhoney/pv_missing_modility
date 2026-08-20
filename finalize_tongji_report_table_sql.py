"""Materialize the report table dataset with an auditable SQLite query."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ROOT / "outputs/gipssr/hyperparameter_experiments/tongji/seed_42"
ARTIFACT_PATH = EXPERIMENT_ROOT / "artifact.json"
METRICS_PATH = EXPERIMENT_ROOT / "metrics_fill_table.csv"

SQL = """SELECT
  CASE parameter
    WHEN 'Top-K candidates' THEN '1. Top-K candidates'
    WHEN 'Refinement scale alpha' THEN '2. Refinement scale α'
    WHEN 'Recovery bound wmax' THEN '3. Recovery bound wmax'
  END AS parameter,
  value,
  printf('%.3f', pm_eer) AS pm_eer,
  printf('%.2f', pm_top1) AS pm_top1,
  printf('%.2f', pm_tar_1e3) AS pm_tar_1e3,
  printf('%.2f', pm_tar_1e4) AS pm_tar_1e4,
  printf('%.3f', vm_eer) AS vm_eer,
  printf('%.2f', vm_top1) AS vm_top1,
  printf('%.2f', vm_tar_1e3) AS vm_tar_1e3,
  printf('%.2f', vm_tar_1e4) AS vm_tar_1e4,
  K,
  alpha,
  wmax,
  experiment
FROM fill_metrics
ORDER BY CASE parameter
  WHEN 'Top-K candidates' THEN 1
  WHEN 'Refinement scale alpha' THEN 2
  WHEN 'Recovery bound wmax' THEN 3
END,
CAST(value AS REAL)"""


def main() -> None:
    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE fill_metrics (
        parameter TEXT NOT NULL,
        value TEXT NOT NULL,
        experiment TEXT NOT NULL,
        K INTEGER NOT NULL,
        alpha REAL NOT NULL,
        wmax REAL NOT NULL,
        pm_eer REAL NOT NULL,
        pm_top1 REAL NOT NULL,
        pm_tar_1e3 REAL NOT NULL,
        pm_tar_1e4 REAL NOT NULL,
        vm_eer REAL NOT NULL,
        vm_top1 REAL NOT NULL,
        vm_tar_1e3 REAL NOT NULL,
        vm_tar_1e4 REAL NOT NULL
        )"""
    )
    connection.executemany(
        "INSERT INTO fill_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                row["parameter"],
                row["value"],
                row["experiment"],
                int(row["K"]),
                float(row["alpha"]),
                float(row["wmax"]),
                float(row["PM_EER_percent"]),
                float(row["PM_Top1_percent"]),
                float(row["PM_TAR_1e-3_percent"]),
                float(row["PM_TAR_1e-4_percent"]),
                float(row["VM_EER_percent"]),
                float(row["VM_Top1_percent"]),
                float(row["VM_TAR_1e-3_percent"]),
                float(row["VM_TAR_1e-4_percent"]),
            )
            for row in source_rows
        ],
    )
    table_rows = [dict(row) for row in connection.execute(SQL)]
    connection.close()
    if len(table_rows) != 9:
        raise ValueError("Expected nine SQL result rows")

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    artifact["snapshot"]["datasets"]["fill_table"] = table_rows
    for collection in (artifact["manifest"]["sources"], artifact["sources"]):
        for source in collection:
            if source.get("id") != "metrics-fill-table":
                continue
            source["query"].update(
                {
                    "engine": "sqlite",
                    "language": "sql",
                    "sql": SQL,
                    "description": "Queries the nine reviewed presentation rows after deterministic import into the in-memory fill_metrics table.",
                    "tables_used": ["fill_metrics"],
                }
            )
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(ARTIFACT_PATH)


if __name__ == "__main__":
    main()
