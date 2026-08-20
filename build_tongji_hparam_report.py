"""Build the canonical Data Analytics report payload for the Tongji sweep."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = ROOT / "outputs/gipssr/hyperparameter_experiments/tongji/seed_42"
METRICS_PATH = EXPERIMENT_ROOT / "metrics_fill_table.csv"
UNIQUE_METRICS_PATH = EXPERIMENT_ROOT / "metrics_unique.csv"
RUN_MANIFEST_PATH = EXPERIMENT_ROOT / "run_manifest.json"
VALIDATION_PATH = EXPERIMENT_ROOT / "validation_summary.json"
ARTIFACT_PATH = EXPERIMENT_ROOT / "artifact.json"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentage(row: dict[str, str], field: str, places: int = 2) -> str:
    return f"{float(row[field]):.{places}f}"


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    fill_rows = load_csv(METRICS_PATH)
    unique_rows = load_csv(UNIQUE_METRICS_PATH)
    run_manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))

    expected_names = {
        "k_3", "default", "k_8", "alpha_0.10", "alpha_0.50", "wmax_0.55", "wmax_0.95"
    }
    actual_names = {row["experiment"] for row in unique_rows}
    if actual_names != expected_names or len(unique_rows) != 7 or len(fill_rows) != 9:
        raise ValueError("Unexpected experiment matrix")

    default_rows = [row for row in fill_rows if row["experiment"] == "default"]
    if len(default_rows) != 3:
        raise ValueError("Default result must appear exactly three times in the fill table")
    default_metrics = {
        key: default_rows[0][key]
        for key in default_rows[0]
        if key.startswith(("PM_", "VM_"))
    }
    if any(
        any(row[key] != value for key, value in default_metrics.items())
        for row in default_rows[1:]
    ):
        raise ValueError("Default fill-table rows are not identical")

    macro_eer = {
        row["experiment"]: (
            float(row["PM_EER_percent"]) + float(row["VM_EER_percent"])
        ) / 2.0
        for row in unique_rows
    }
    best_name = min(macro_eer, key=macro_eer.get)
    if best_name != "alpha_0.50":
        raise ValueError(f"Unexpected minimum macro EER: {best_name}")

    result_payloads = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((EXPERIMENT_ROOT / "results").glob("*.json"))
    }
    if len(result_payloads) != 7:
        raise ValueError("Expected seven result JSON files")
    sample_shapes = set()
    far_resolutions = set()
    for payload in result_payloads.values():
        for scenario in ("palmprint_missing", "palmvein_missing"):
            fused = payload["results"][scenario]["fused"]
            sample_shapes.add(
                (
                    fused["num_gallery_identities"],
                    fused["num_probes"],
                    fused["num_genuine_scores"],
                    fused["num_impostor_scores"],
                )
            )
            far_resolutions.add(fused["far_count_resolution"])
    if sample_shapes != {(120, 240, 240, 28560)}:
        raise ValueError(f"Inconsistent evaluation populations: {sample_shapes}")

    selection_epochs = {}
    invariant_hashes = set()
    for experiment in run_manifest["experiments"]:
        if experiment["reuse_default_training"]:
            selection_epochs[experiment["name"]] = 2
            continue
        selection_path = (
            ROOT
            / "outputs/gipssr/hyperparameter_experiments/tongji/seed_42/checkpoints"
            / experiment["name"]
            / "selection/best.pth"
        )
        if not selection_path.is_file():
            raise FileNotFoundError(selection_path)
        selection_epochs[experiment["name"]] = result_payloads[experiment["name"]]["best_epoch"]
        invariant_hashes.add(result_payloads[experiment["name"]]["gallery_protocol_sha256"])
        invariant_hashes.add(result_payloads[experiment["name"]]["probe_protocol_sha256"])
    if set(selection_epochs.values()) != {2}:
        raise ValueError(f"Unexpected selected epochs: {selection_epochs}")
    if len(invariant_hashes) != 2:
        raise ValueError("Evaluation protocol hashes differ across experiments")

    validation = {
        "assessment": "Share with caveats",
        "scope": "Ready for the requested single-seed, single-training table; not a multi-seed robustness claim.",
        "checks": {
            "unique_configurations": 7,
            "fill_table_rows": 9,
            "default_training_count": 1,
            "default_rows_identical": True,
            "selected_epoch_all_configurations": 2,
            "evaluation_population": {
                "gallery_identities": 120,
                "probes_per_direction": 240,
                "genuine_scores_per_direction": 240,
                "impostor_scores_per_direction": 28560,
            },
            "far_count_resolution": min(far_resolutions),
            "tar_step_percentage_points": 100.0 / 240.0,
            "protocol_hashes_consistent": True,
            "checkpoint_and_result_hashes_verified": True,
            "non_controlled_training_arguments_identical": True,
        },
        "best_macro_eer": {
            "experiment": best_name,
            "value_percent": macro_eer[best_name],
            "default_percent": macro_eer["default"],
            "difference_percentage_points": macro_eer[best_name] - macro_eer["default"],
        },
        "limitations": [
            "Each unique configuration has one seed-42 training result, as requested.",
            "TAR values move in 1/240 = 0.4167 percentage-point increments per accepted genuine probe.",
            "Small EER differences should not be generalized without additional seeds.",
        ],
    }
    VALIDATION_PATH.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    group_names = {
        "Top-K candidates": "1. Top-K candidates",
        "Refinement scale alpha": "2. Refinement scale α",
        "Recovery bound wmax": "3. Recovery bound wmax",
    }
    table_rows = []
    for row in fill_rows:
        table_rows.append(
            {
                "parameter": group_names[row["parameter"]],
                "value": row["value"],
                "pm_eer": percentage(row, "PM_EER_percent", 3),
                "pm_top1": percentage(row, "PM_Top1_percent"),
                "pm_tar_1e3": percentage(row, "PM_TAR_1e-3_percent"),
                "pm_tar_1e4": percentage(row, "PM_TAR_1e-4_percent"),
                "vm_eer": percentage(row, "VM_EER_percent", 3),
                "vm_top1": percentage(row, "VM_Top1_percent"),
                "vm_tar_1e3": percentage(row, "VM_TAR_1e-3_percent"),
                "vm_tar_1e4": percentage(row, "VM_TAR_1e-4_percent"),
                "K": int(row["K"]),
                "alpha": float(row["alpha"]),
                "wmax": float(row["wmax"]),
                "experiment": row["experiment"],
            }
        )

    sources = [
        {
            "id": "metrics-fill-table",
            "label": "Tongji hyperparameter fill-table metrics",
            "path": "outputs/gipssr/hyperparameter_experiments/tongji/seed_42/metrics_fill_table.csv",
            "query": {
                "engine": "Python",
                "language": "python",
                "description": "Extract final fused PM/VM metrics from seven evaluated checkpoints; reuse the default row only for the three table sections.",
                "executed_at": generated_at,
                "filters": [
                    "dataset=tongji",
                    "seed=42",
                    "model_ablation=full",
                    "scenarios=palmprint_missing,palmvein_missing",
                ],
                "metric_definitions": [
                    "EER is the interpolated equal-error rate, reported in percentage points; lower is better.",
                    "Top-1 is the fraction of probes whose highest-scoring gallery identity is genuine, reported as a percentage; higher is better.",
                    "TAR@FAR is the genuine acceptance rate at the stated false-acceptance rate, reported as a percentage; higher is better.",
                ],
            },
        },
        {
            "id": "run-manifest",
            "label": "Tongji hyperparameter run manifest",
            "path": "outputs/gipssr/hyperparameter_experiments/tongji/seed_42/run_manifest.json",
            "query": {
                "engine": "Python",
                "language": "python",
                "description": "Audit manifest containing the one-factor-at-a-time matrix, checkpoint hashes, encoder hashes, warm-start hash, and protocol hashes.",
                "executed_at": generated_at,
            },
        },
        {
            "id": "validation-summary",
            "label": "Tongji hyperparameter validation summary",
            "path": "outputs/gipssr/hyperparameter_experiments/tongji/seed_42/validation_summary.json",
            "query": {
                "engine": "Python",
                "language": "python",
                "description": "Independent checks of row counts, evaluation populations, selected epochs, result hashes, controlled arguments, metric resolution, and limitations.",
                "executed_at": generated_at,
            },
        },
    ]

    columns = [
        {"field": "parameter", "label": "Parameter", "type": "text"},
        {"field": "value", "label": "Value", "type": "text"},
        {"field": "pm_eer", "label": "PM EER ↓ (%)", "type": "text"},
        {"field": "pm_top1", "label": "PM Top-1 ↑ (%)", "type": "text"},
        {"field": "pm_tar_1e3", "label": "PM TAR@10⁻³ ↑ (%)", "type": "text"},
        {"field": "pm_tar_1e4", "label": "PM TAR@10⁻⁴ ↑ (%)", "type": "text"},
        {"field": "vm_eer", "label": "VM EER ↓ (%)", "type": "text"},
        {"field": "vm_top1", "label": "VM Top-1 ↑ (%)", "type": "text"},
        {"field": "vm_tar_1e3", "label": "VM TAR@10⁻³ ↑ (%)", "type": "text"},
        {"field": "vm_tar_1e4", "label": "VM TAR@10⁻⁴ ↑ (%)", "type": "text"},
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Tongji 超参数单变量实验",
            "description": "K、α 与 wmax 的 seed-42 严格单变量敏感性实验与 PM/VM 指标。",
            "generatedAt": generated_at,
            "sources": sources,
            "tables": [
                {
                    "id": "fill-table",
                    "title": "Tongji 单变量实验指标",
                    "subtitle": "Seed 42；数值单位为 %；默认配置在三组中重复展示但只训练一次。",
                    "showDescription": True,
                    "dataset": "fill_table",
                    "defaultSort": {"field": "parameter", "direction": "asc"},
                    "density": "spacious",
                    "sourceId": "metrics-fill-table",
                    "layout": "full",
                    "columns": columns,
                }
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Tongji 超参数单变量实验"},
                {
                    "id": "technical-summary",
                    "type": "markdown",
                    "sourceId": "metrics-fill-table",
                    "body": (
                        "## 技术摘要\n\n"
                        "本次得到 **7 个唯一训练配置**；默认配置 K=5、α=0.25、wmax=0.75 复用一次训练结果。"
                        "所有配置的 PM/VM Top-1、TAR@10⁻³，以及 VM TAR@10⁻⁴ 均相同。"
                        "按两方向 EER 的简单平均，α=0.50 最低（0.252%），默认配置为 0.261%；"
                        "差值仅 -0.009 个百分点，因此应视为单次 seed-42 的轻微优势，而非稳定排序。"
                    ),
                },
                {
                    "id": "key-findings",
                    "type": "markdown",
                    "sourceId": "metrics-fill-table",
                    "body": (
                        "## 差异主要出现在 EER 与 PM 的低 FAR 指标\n\n"
                        "K=8 和 wmax=0.95 的 PM TAR@10⁻⁴ 最高，均为 97.08%，但两者的 PM EER 也最高。"
                        "wmax=0.55 的 PM EER 最低（0.270%），同时 VM EER 上升至 0.256%。"
                        "这说明本次单次实验不存在对两个缺失方向和所有指标都占优的配置；表格应按目标指标逐项填写，而不宜压缩成单一“最优”结论。"
                    ),
                },
                {
                    "id": "table-intro",
                    "type": "markdown",
                    "body": (
                        "## 可直接填入论文表格的结果\n\n"
                        "下表按用户提供的第二张图顺序排列。EER 保留三位小数以呈现细微差异，其余指标保留两位小数。"
                    ),
                },
                {"id": "results-table", "type": "table", "tableId": "fill-table", "layout": "full"},
                {
                    "id": "scope-definitions",
                    "type": "markdown",
                    "sourceId": "validation-summary",
                    "body": (
                        "## 同一固定测试总体支撑全部比较\n\n"
                        "每个方向均使用 120 个 gallery 身份、240 个 probes、240 个 genuine scores 与 28,560 个 impostor scores。"
                        "PM 表示 palmprint missing，VM 表示 palm-vein missing。"
                        "TAR 的一个 probe 步长为 0.4167 个百分点，因此 0.42 个百分点级别的差异对应一个 genuine probe。"
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "run-manifest",
                    "body": (
                        "## 单变量控制、验证选择与固定重放均保持一致\n\n"
                        "所有配置固定 Tongji 数据协议、seed=42、冻结编码器、Stage-1 warm start、模型结构、优化器、损失权重、批设置与 12 轮验证上限。"
                        "仅分别改变 K、α 或 wmax；六个新配置均先在身份不重叠验证集选择 epoch，再从同一起点按选中轮数固定重放。"
                        "七个配置的最终 best epoch 均为 2；默认配置的同一正式检查点只使用一次。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": "validation-summary",
                    "body": (
                        "## 单次训练足以填表，但不足以建立稳健排序\n\n"
                        "验证结论为 **Share with caveats**：检查点与结果哈希、协议哈希、评估总体、非控制训练参数和默认复用逻辑均通过；"
                        "但每个唯一配置只训练一次，符合本次要求，却不能量化随机种子方差。"
                        "EER 的百分之几百点差异以及一个 TAR 步长的变化，不应被表述为跨 seed 的显著提升。"
                    ),
                },
                {
                    "id": "next-steps",
                    "type": "markdown",
                    "body": (
                        "## 建议保留默认配置并把 α=0.50 作为复核候选\n\n"
                        "如果当前目标只是完成表格，直接使用本报告数值即可。"
                        "若需要从这些设置中选择新的默认值，建议优先对 α=0.50 与当前默认配置补跑至少两个额外种子，"
                        "并以两方向 macro EER 为主指标、PM TAR@10⁻⁴ 为低 FAR 保护指标。"
                    ),
                },
                {
                    "id": "further-questions",
                    "type": "markdown",
                    "body": (
                        "## 后续问题\n\n"
                        "需要进一步确认的是：最终论文选择是否以 macro EER 为主，还是更重视 PM 的 TAR@10⁻⁴。"
                        "这两个目标在 K 与 wmax 扫描中存在轻微权衡。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {"fill_table": table_rows},
        },
        "sources": sources,
    }
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(ARTIFACT_PATH)
    print(VALIDATION_PATH)


if __name__ == "__main__":
    main()
