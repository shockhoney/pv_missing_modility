# Hetero-MMRNet 实验计划

## 1. 总体目标

构建面向掌纹-掌静脉模态缺失识别的 Hetero-MMRNet。

整体思路：

1. 联合训练掌纹编码器和掌静脉编码器，获得语义对齐的双模态特征空间。
2. 基于该特征空间训练模态恢复器。
3. 在真实模态与恢复模态基础上进行融合与动态选择。
4. 在完整模态、单模态缺失、随机缺失场景下进行验证。

训练阶段分类头统一采用 **ArcFace**。  
测试阶段保留分类头，使用 ArcFace 做闭集识别。

## 2. 数据协议

使用闭集协议，训练、验证、测试共享同一批身份，按每类样本划分。

| 文件 | 用途 |
|---|---|
| `closed_train_full.txt` | 训练阶段使用的完整双模态样本 |
| `closed_val_full.txt` | 闭集验证 |
| `closed_test_protocol.txt` | 最终闭集测试 |

协议格式：

```text
palm_path vein_path label palm_exists vein_exists split
```

测试场景：

| 场景 | 含义 |
|---|---|
| `full` | 掌纹 + 掌静脉均存在 |
| `palm_only` | 只有掌纹 |
| `vein_only` | 只有掌静脉 |
| `random_missing` | 随机缺失一个模态 |

## 3. Step 1：联合训练双模态编码器

### 目标

联合训练掌纹编码器和掌静脉编码器，使两种模态进入统一身份判别空间。

### 编码器设计

| 模态 | 编码器方案 |
|---|---|
| 掌纹 | ResNet50 + UAA 图像增强 |
| 掌静脉 | ConvNeXt V2-Tiny 方法 |

特征表示：

```text
f_p = E_p(x_p)
f_v = E_v(x_v)
```

其中：

| 符号 | 含义 |
|---|---|
| `E_p` | 掌纹编码器 |
| `E_v` | 掌静脉编码器 |
| `f_p` | 掌纹特征 |
| `f_v` | 掌静脉特征 |

### 分类头

两个模态的身份监督均采用 **ArcFace**。

### 损失函数

```text
Loss = L_palm_id + L_vein_id + L_align + optional L_joint_id
```

含义：

| 损失 | 作用 |
|---|---|
| `L_palm_id` | 掌纹特征的身份判别损失，ArcFace |
| `L_vein_id` | 掌静脉特征的身份判别损失，ArcFace |
| `L_align` | 约束同一身份的掌纹/掌静脉特征靠近 |
| `L_joint_id` | 可选，联合特征的身份判别损失 |

### 训练目标

1. 掌纹特征具备单模态识别能力。
2. 掌静脉特征具备单模态识别能力。
3. 同一身份的 `f_p` 与 `f_v` 在共享空间中接近。
4. 不同身份之间保持足够区分度。

### 输出权重

```text
palm.pth
vein.pth
joint.pth
```

后续主要使用：

```text
palm.pth
vein.pth
```

## 4. Step 2：训练模态恢复器

### 目标

学习掌纹特征与掌静脉特征之间的双向恢复关系。

```text
f_v_hat = R_p2v(f_p)
f_p_hat = R_v2p(f_v)
```

| 恢复方向 | 含义 |
|---|---|
| `R_p2v` | 由掌纹特征恢复掌静脉特征 |
| `R_v2p` | 由掌静脉特征恢复掌纹特征 |

### 基本损失

```text
L_recovery = L_rec + lambda_cos * L_cos + lambda_id * L_id
```

其中：

| 损失 | 作用 |
|---|---|
| `L_rec` | 特征重建约束 |
| `L_cos` | 方向一致性约束 |
| `L_id` | 恢复特征的身份判别约束，ArcFace |

该阶段建议冻结 `E_p` 和 `E_v`。

## 5. Step 3：训练全模态融合模块

### 目标

在完整双模态输入下获得稳定的融合表示，作为后续缺失模态恢复效果的 baseline。

输入：

```text
f_p, f_v
```

输出：

```text
z_full = F(f_p, f_v)
```

身份监督：

```text
L_full = ArcFace(z_full, y)
```

该阶段只关注完整模态融合能力，不处理模态缺失。

## 6. Step 4：缺失模态恢复与融合

### 目标

在单模态缺失时，利用真实模态恢复缺失模态特征，并完成融合。

掌静脉缺失：

```text
f_v_hat = R_p2v(f_p)
z_palm_only = F(f_p, f_v_hat)
```

掌纹缺失：

```text
f_p_hat = R_v2p(f_v)
z_vein_only = F(f_p_hat, f_v)
```

训练目标：

1. 完整模态下保持强识别能力。
2. 掌静脉缺失时，恢复特征能有效补充掌纹特征。
3. 掌纹缺失时，恢复特征能有效补充掌静脉特征。
4. 恢复特征不能破坏真实模态已有的判别能力。

## 7. Step 5：动态选择机制

### 目标

判断恢复模态是否可靠，避免低质量恢复特征拉低识别效果。

候选表示：

```text
z_real = 使用真实可用模态得到的表示
z_fuse = 使用真实模态 + 恢复模态得到的融合表示
```

最终输出：

```text
z_final = Select(z_real, z_fuse)
```

选择机制需要解决的问题：

1. 恢复特征质量较高时，采用融合表示。
2. 恢复特征质量较低时，保留真实模态表示。
3. 在 `palm_only`、`vein_only`、`random_missing` 下保持稳定。

## 8. Step 6：最终测试

测试时使用 ArcFace 分类头做闭集识别。

测试场景：

| 场景 | 表示 |
|---|---|
| `full` | `z_full = F(f_p, f_v)` |
| `palm_only` | `z_final = Select(z_real, F(f_p, R_p2v(f_p)))` |
| `vein_only` | `z_final = Select(z_real, F(R_v2p(f_v), f_v))` |
| `random_missing` | 按实际缺失模态选择对应流程 |

评价指标：

```text
Recognition Rate (%)
```

## 9. 结果记录表

| 模型 | 场景 | Recognition Rate (%) |
|---|---|---:|
| Palm Encoder | palm_only |  |
| Vein Encoder | vein_only |  |
| Full Fusion Baseline | full |  |
| Hetero-MMRNet | full |  |
| Hetero-MMRNet | palm_only |  |
| Hetero-MMRNet | vein_only |  |
| Hetero-MMRNet | random_missing |  |
