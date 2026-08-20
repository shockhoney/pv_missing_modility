"""Add the required macro-EER comparison chart to the Tongji report artifact."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ROOT / "outputs/gipssr/hyperparameter_experiments/tongji/seed_42"
ARTIFACT_PATH = EXPERIMENT_ROOT / "artifact.json"
METRICS_PATH = EXPERIMENT_ROOT / "metrics_unique.csv"
VALIDATION_PATH = EXPERIMENT_ROOT / "validation_summary.json"


def main() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    with METRICS_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    labels = {
        "k_3": "K=3",
        "default": "K=5, α=0.25, wmax=0.75 (default)",
        "k_8": "K=8",
        "alpha_0.10": "α=0.10",
        "alpha_0.50": "α=0.50",
        "wmax_0.55": "wmax=0.55",
        "wmax_0.95": "wmax=0.95",
    }
    chart_rows = []
    for row in rows:
        pm_eer = float(row["PM_EER_percent"])
        vm_eer = float(row["VM_EER_percent"])
        chart_rows.append(
            {
                "setting": labels[row["experiment"]],
                "macro_eer": (pm_eer + vm_eer) / 2.0,
                "pm_eer": pm_eer,
                "vm_eer": vm_eer,
                "K": int(row["K"]),
                "alpha": float(row["alpha"]),
                "wmax": float(row["wmax"]),
                "experiment": row["experiment"],
            }
        )
    if len(chart_rows) != 7:
        raise ValueError("Expected seven unique configurations")

    source = {
        "id": "metrics-unique",
        "label": "Tongji unique-configuration metrics",
        "path": "outputs/gipssr/hyperparameter_experiments/tongji/seed_42/metrics_unique.csv",
        "query": {
            "engine": "Python",
            "language": "python",
            "description": "One row per unique trained configuration with PM/VM metrics and the mean directional EER used for the comparison chart.",
            "executed_at": artifact["manifest"]["generatedAt"],
            "filters": ["dataset=tongji", "seed=42", "unique_configurations_only=true"],
            "metric_definitions": [
                "Macro EER is the unweighted mean of palmprint-missing EER and palm-vein-missing EER, expressed in percentage points."
            ],
        },
    }
    for key in ("sources",):
        artifact[key] = [item for item in artifact.get(key, []) if item.get("id") != source["id"]]
        artifact[key].append(source)
    artifact["manifest"]["sources"] = [
        item for item in artifact["manifest"].get("sources", []) if item.get("id") != source["id"]
    ]
    artifact["manifest"]["sources"].append(source)

    chart = {
        "id": "macro-eer-chart",
        "title": "双方向平均 EER",
        "subtitle": "7 个唯一 seed-42 配置；单位为 %，越低越好。",
        "showDescription": True,
        "intent": "comparison",
        "question": "哪个唯一超参数配置的 PM/VM 平均 EER 最低？",
        "rationale": "七个离散配置适合用横向条形图比较单一同单位指标，长标签可保持可读；精确数值仍由后续表格提供。",
        "type": "horizontalBar",
        "dataset": "unique_eer",
        "sourceId": "metrics-unique",
        "encodings": {
            "x": {"field": "setting", "type": "nominal", "label": "配置"},
            "y": {
                "field": "macro_eer",
                "type": "quantitative",
                "format": "number",
                "label": "PM/VM 平均 EER",
                "unit": "%",
            },
            "tooltip": [
                {"field": "pm_eer", "type": "quantitative", "label": "PM EER", "unit": "%"},
                {"field": "vm_eer", "type": "quantitative", "label": "VM EER", "unit": "%"},
            ],
        },
        "valueFormat": "number",
        "unit": "%",
        "layout": "full",
        "maxRows": 7,
    }
    artifact["manifest"]["charts"] = [
        item for item in artifact["manifest"].get("charts", []) if item.get("id") != chart["id"]
    ]
    artifact["manifest"]["charts"].append(chart)
    artifact["snapshot"]["datasets"]["unique_eer"] = chart_rows

    block = {
        "id": "macro-eer-visual",
        "type": "chart",
        "chartId": "macro-eer-chart",
        "layout": "full",
    }
    blocks = [item for item in artifact["manifest"]["blocks"] if item.get("id") != block["id"]]
    insert_at = next(
        (index + 1 for index, item in enumerate(blocks) if item.get("id") == "key-findings"),
        3,
    )
    blocks.insert(insert_at, block)
    artifact["manifest"]["blocks"] = blocks

    validation["chart_map"] = [
        {
            "section": "Key findings",
            "question": chart["question"],
            "family": "Comparison & Ranking",
            "type": "horizontalBar",
            "dataset": "unique_eer",
            "fields": ["setting", "macro_eer"],
            "claim": "alpha=0.50 has the lowest single-run macro EER, with a very small margin over default.",
            "palette_policy": "single-root preferred",
        }
    ]
    VALIDATION_PATH.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(ARTIFACT_PATH)


if __name__ == "__main__":
    main()
