# 掌纹-掌静脉模态缺失识别

本项目面向 PolyU 掌纹与掌静脉双模态识别，研究重点是**模态缺失场景下的验证任务**。整体流程采用三阶段训练：

1. 联合预训练掌纹与掌静脉 `MobileFaceNet` 编码器，并对齐共享身份空间
2. 在稳定特征空间上训练跨模态特征恢复器
3. 训练 DyMo 风格的 transformer 融合与动态选择模块

最终测试输出完整模态、单模态缺失和随机缺失场景下的 `EER`、`AUC`、`TAR@FAR` 等指标。

## 仓库结构

```text
pv_missing_modility/
|-- build_polyu_missing_protocol.py
|-- train_encoder.py
|-- train_recoverer.py
|-- train_dymo.py
|-- prepare_dymo_stats.py
|-- test_dymo.py
|-- requirements.txt
|-- README.md
|-- data/
|-- data_txt/
|-- models/
|   |-- stage1_mobileFacenet.py
|   `-- dymo.py
`-- utils/
    |-- datasets_txt.py
    |-- dymo_stats.py
    |-- metrics.py
    `-- prototype_loss.py
```

## 数据与协议

默认使用 PolyU 数据集，原始目录为：

```text
data/PolyU/
```

协议文件位于：

```text
data_txt/
```

默认使用四个协议文件：

- `polyu_train_full.txt`
- `polyu_val_full.txt`
- `polyu_val_missing_fixed.txt`
- `polyu_test_missing_protocol.txt`

协议格式：

```text
palm_path vein_path label palm_exists vein_exists split
```

含义：

- `palm_path`：掌纹图像路径，缺失时为 `NA`
- `vein_path`：掌静脉图像路径，缺失时为 `NA`
- `label`：类别标签
- `palm_exists` / `vein_exists`：模态是否存在
- `split`：协议类型，如 `train`、`val`、`full`、`palm_only`、`vein_only`、`random_missing`

如需重新生成协议：

```bash
python build_polyu_missing_protocol.py --root_dir data/PolyU --output_dir data_txt
```

## 模型流程

### 1. 共享身份空间编码器

- `train_encoder.py --modality joint`

联合训练掌纹与掌静脉两路 `MobileFaceNet`，使用共享分类头和跨模态监督对比损失，让两种模态落在同一个身份空间中。训练完成后仍会导出兼容下游脚本的 `palm_best.pth` 和 `vein_best.pth`。

### 2. 特征恢复器

恢复器学习两条映射：

- `palm -> recovered vein`
- `vein -> recovered palm`

恢复器只在特征空间工作，不做图像生成。训练时先冻结编码器，再小学习率微调。

### 3. DyMo 主模型

DyMo 主体包括：

- 双分支 `MobileFaceNet`
- 缺失模态恢复器
- token 化与动态 transformer 融合
- 分类头与验证 embedding 投影头
- DyMo 动态选择器

DyMo 会分别比较：

- 只使用真实可用模态
- 加入恢复模态后再次融合

如果加入恢复模态后质量更高，就接受恢复模态用于最终验证。

## 环境安装

```bash
pip install -r requirements.txt
```

## 训练流程

推荐按以下顺序运行。

### 第一步：联合预训练掌纹/掌静脉编码器

```bash
python train_encoder.py ^
  --modality joint ^
  --train_full_list data_txt/polyu_train_full.txt ^
  --val_full_list data_txt/polyu_val_full.txt ^
  --save_dir outputs_dymo/encoders
```

输出：

```text
outputs_dymo/encoders/palm_best.pth
outputs_dymo/encoders/vein_best.pth
outputs_dymo/encoders/joint_best.pth
```

如需保留旧的单模态独立训练方式，仍可分别运行 `--modality palm` 与 `--modality vein`。

### 第二步：训练特征恢复器

```bash
python train_recoverer.py ^
  --train_full_list data_txt/polyu_train_full.txt ^
  --val_full_list data_txt/polyu_val_full.txt ^
  --palm_ckpt outputs_dymo/encoders/palm_best.pth ^
  --vein_ckpt outputs_dymo/encoders/vein_best.pth ^
  --save_dir outputs_dymo/recoverer
```

输出：

```text
outputs_dymo/recoverer/recoverer_best.pth
```

### 第三步：训练 DyMo 主模型

```bash
python train_dymo.py ^
  --train_full_list data_txt/polyu_train_full.txt ^
  --val_full_list data_txt/polyu_val_full.txt ^
  --val_missing_list data_txt/polyu_val_missing_fixed.txt ^
  --palm_ckpt outputs_dymo/encoders/palm_best.pth ^
  --vein_ckpt outputs_dymo/encoders/vein_best.pth ^
  --recoverer_ckpt outputs_dymo/recoverer/recoverer_best.pth ^
  --save_dir outputs_dymo/dymo
```

默认会：

- 用 palm / vein 预训练 checkpoint 初始化两路编码器
- 用 recoverer checkpoint 初始化恢复器
- 低学习率微调编码器
- 冻结恢复器，仅训练 transformer 与 DyMo 主体

如需让 recoverer 在 DyMo 阶段继续联合训练，可加：

```bash
--train_recoverers
```

输出：

```text
outputs_dymo/dymo/dymo_best.pth
```

### 第四步：生成 DyMo 统计量

```bash
python prepare_dymo_stats.py ^
  --train_full_list data_txt/polyu_train_full.txt ^
  --checkpoint outputs_dymo/dymo/dymo_best.pth
```

输出：

```text
outputs_dymo/dymo/gaussian/subset_gaussian.pt
```

### 第五步：测试

```bash
python test_dymo.py ^
  --train_full_list data_txt/polyu_train_full.txt ^
  --val_protocol_list data_txt/polyu_val_missing_fixed.txt ^
  --protocol_list data_txt/polyu_test_missing_protocol.txt ^
  --palm_ckpt outputs_dymo/encoders/palm_best.pth ^
  --vein_ckpt outputs_dymo/encoders/vein_best.pth ^
  --checkpoint outputs_dymo/dymo/dymo_best.pth ^
  --stats_path outputs_dymo/dymo/gaussian/subset_gaussian.pt ^
  --selection_mode open ^
  --search_tau
```

## 测试输出

测试脚本会输出三组结果：

1. `Single-Modality Baseline`
   - palm encoder baseline
   - vein encoder baseline

2. `DyMo Without Recovered Selection`
   - 只使用真实可用模态，不启用恢复模态

3. `DyMo With Recovered Selection`
   - 启用 log-prob reward 的 DyMo 选择器
   - 可在验证集搜索 `palm_only` / `vein_only` 的固定 `tau` 后再测试

正式测试协议覆盖：

- `full`
- `palm_only`
- `vein_only`
- `random_missing`

指标包括：

- `AUC`
- `EER`
- `ACC@EER threshold`
- `TAR@FAR=1e-5`
- `TAR@FAR=1e-4`
- `TAR@FAR=1e-3`
- `Recovered modality accepted ratio`

## 推荐执行顺序

```bash
python build_polyu_missing_protocol.py --root_dir data/PolyU --output_dir data_txt
python train_encoder.py --modality joint --train_full_list data_txt/polyu_train_full.txt --val_full_list data_txt/polyu_val_full.txt --save_dir outputs_dymo/encoders
python train_recoverer.py --train_full_list data_txt/polyu_train_full.txt --val_full_list data_txt/polyu_val_full.txt --palm_ckpt outputs_dymo/encoders/palm_best.pth --vein_ckpt outputs_dymo/encoders/vein_best.pth --save_dir outputs_dymo/recoverer
python train_dymo.py --train_full_list data_txt/polyu_train_full.txt --val_full_list data_txt/polyu_val_full.txt --val_missing_list data_txt/polyu_val_missing_fixed.txt --palm_ckpt outputs_dymo/encoders/palm_best.pth --vein_ckpt outputs_dymo/encoders/vein_best.pth --recoverer_ckpt outputs_dymo/recoverer/recoverer_best.pth --save_dir outputs_dymo/dymo
python prepare_dymo_stats.py --train_full_list data_txt/polyu_train_full.txt --checkpoint outputs_dymo/dymo/dymo_best.pth
python test_dymo.py --train_full_list data_txt/polyu_train_full.txt --val_protocol_list data_txt/polyu_val_missing_fixed.txt --protocol_list data_txt/polyu_test_missing_protocol.txt --palm_ckpt outputs_dymo/encoders/palm_best.pth --vein_ckpt outputs_dymo/encoders/vein_best.pth --checkpoint outputs_dymo/dymo/dymo_best.pth --stats_path outputs_dymo/dymo/gaussian/subset_gaussian.pt --selection_mode open --search_tau
```
