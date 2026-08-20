"""Materialize the report chart dataset with an auditable SQLite query."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ROOT / "outputs/gipssr/hyperparameter_experiments/tongji/seed_42"
ARTIFACT_PATH = EXPERIMENT_ROOT / "artifact.json"
METRICS_PATH = EXPERIMENT_ROOT / "metrics_unique.csv"

SQL = """SELECT
  CASE experiment
    WHEN 'k_3' THEN 'K=3'
    WHEN 'default' THEN 'K=5, α=0.25, wmax=0.75 (default)'
    WHEN 'k_8' THEN 'K=8'
    WHEN 'alpha_0.10' THEN 'α=0.10'
    WHEN 'alpha_0.50' THEN 'α=0.50'
    WHEN 'wmax_0.55' THEN 'wmax=0.55'
    WHEN 'wmax_0.95' THEN 'wmax=0.95'
  END AS setting,
  (pm_eer + vm_eer) / 2.0 AS macro_eer,
  pm_eer,
  vm_eer,
  K,
  alpha,
  wmax,
  experiment
FROM hyperparameter_metrics
ORDER BY CASE experiment
  WHEN 'k_3' THEN 1
  WHEN 'default' THEN 2
  WHEN 'k_8' THEN 3
  WHEN 'alpha_0.10' THEN 4
  WHEN 'alpha_0.50' THEN 5
  WHEN 'wmax_0.55' THEN 6
  WHEN 'wmax_0.95' THEN 7
END"""


def main() -> None:
    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """CREATE TABLE hyperparameter_metrics (
        experiment TEXT PRIMARY KEY,
        pm_eer REAL NOT NULL,
        vm_eer REAL NOT NULL,
        K INTEGER NOT NULL,
        alpha REAL NOT NULL,
        wmax REAL NOT NULL
        )"""
    )
    connection.executemany(
        "INSERT INTO hyperparameter_metrics VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                row["experiment"],
                float(row["PM_EER_percent"]),
                float(row["VM_EER_percent"]),
                int(row["K"]),
                float(row["alpha"]),
                float(row["wmax"]),
            )
            for row in source_rows
        ],
    )
    chart_rows = [dict(row) for row in connection.execute(SQL)]
    connection.close()
    if len(chart_rows) != 7:
        raise ValueError("Expected seven SQL result rows")

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    artifact["snapshot"]["datasets"]["unique_eer"] = chart_rows
    for collection in (artifact["manifest"]["sources"], artifact["sources"]):
        for source in collection:
            if source.get("id") != "metrics-unique":
                continue
            source["query"].update(
                {
                    "engine": "sqlite",
                    "language": "sql",
                    "sql": SQL,
                    "description": "Queries the seven reviewed CSV rows after deterministic import into the in-memory hyperparameter_metrics table.",
                    "tables_used": ["hyperparameter_metrics"],
                }
            )
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(ARTIFACT_PATH)


if __name__ == "__main__":
    main()
