# 计算机视觉课程期末报告

---

## 题目一：语义分割任务实现与评价（代码：`code_sample1`）

### 1. 任务简介与理论背景
语义分割旨在为输入图像的每一个像素分配语义类别标签，本实验聚焦于前景/背景二元分割。该任务通常基于编码器—解码器结构来捕获语义上下文并恢复空间分辨率。设输入图像为 \(x \in \mathbb{R}^{H\times W}\)，模型输出像素概率 \(p_{ij} = \sigma(f_\theta(x)_{ij})\)，其中 \(\sigma\) 为 Sigmoid 函数、\(f_\theta\) 为 UNet。训练目标是最小化像素级分类误差并最大化预测与真值掩码 \(y\) 的区域重叠。理论上，Dice 系数是 Jaccard 指数的调和形式，对类别极度不平衡更鲁棒；而加权 BCE 可通过正类权重 \(w^+\) 缓解前景稀疏带来的梯度消失问题。

### 2. 代码结构与模块关系
| 模块 | 作用 | 关键实现 |
| --- | --- | --- |
| `main.py` | 解析配置、构建数据集、实例化模型与 `Trainer` | `torch.utils.data.random_split` 划分训练/验证；自动选择 CUDA |
| `dataset.py` | 定义 `MyDataset`，实现灰度读取、文件名对齐、同步增强及二值化 | 增强组合包含 `RandomHorizontalFlip`、`RandomRotation`、`RandomAffine`、`ElasticTransform` |
| `model/net.py` | 五层下采样 + 五层上采样的 `UNet` 主干，使用跳连融合特征 | Skip-connection 通过 `torch.cat` 实现，输出 1 通道 logits |
| `model/training.py` | `Trainer` 类封装训练/验证循环、混合精度、梯度裁剪、可视化与 checkpoint | 评价指标函数 `calculate_iou/dice/accuracy` 等集中定义 |
| `test_model.py` | 统一的测试脚本，支持批量指标统计与可视化导出 | 通过 `ModelTester.test_dataset` 返回均值±方差 |

### 3. 模型与算法设计
#### 3.1 模型结构
- **Encoder**：四个阶段的 `Conv-BN-ReLU` 块后接 \(2\times\) MaxPool，以指数级扩展感受野。
- **Bottleneck**：1024 通道的双卷积抑制信息瓶颈。
- **Decoder**：`ConvTranspose2d` 上采样 + 与对应编码层拼接，再经双卷积细化，最后以 \(1\times1\) 卷积输出 logits。
- **跳连机制**：在理论上相当于显式保留低层边缘信息，与上层语义特征互补，能够避免解码阶段出现空间模糊。

#### 3.2 损失函数
- **加权 BCE**：\(\mathcal{L}_{\text{BCE}} = - w^+ y\log p - w^- (1-y)\log(1-p)\)，其中 `pos_weight=16`，用于提高前景梯度。
- **Dice Loss**：\(\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum p y + \epsilon}{\sum p + \sum y + \epsilon}\)，直接最大化区域交集。
- **总损失**：\(\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}\)。该组合理论上兼顾像素精度与轮廓重叠度。

#### 3.3 训练策略
1. **数据增强**：随机抽取 1–3 种操作构成管线，并通过统一随机种子保证图像与掩码变换一致；掩码插值统一使用 `NEAREST`，避免灰度污染。
2. **优化与数值稳定**：采用 Adam（默认 lr=5e-5）+ 混合精度 (`torch.cuda.amp`) + 梯度裁剪（max-norm=1）来抑制梯度爆炸。
3. **度量监控**：每 10 个 batch 输出 IoU、Dice、MAE、梯度范数以及概率分布统计，自动提示梯度消失、输出塌陷等异常。
4. **可视化**：每个 epoch 将 `Input/Prob/Pred/GT` 及差异图保存到 `PredvsGT/`，形成直观的定性评估材料。

### 4. 评价指标与理论说明
- **IoU (Jaccard)**：\(\frac{|P\cap G|}{|P\cup G|}\)，取值 [0,1]，越大表示预测与真值重叠越充分。
- **Dice**：与 IoU 的关系为 \(\text{Dice} = \frac{2\ \text{IoU}}{1 + \text{IoU}}\)，对前景稀疏时更稳定。
- **Acc / Precision / Recall / F1**：采用像素级 TP/FP/FN 统计，可评估模型在不同阈值下的检测能力。
- **MAE/MSE**：衡量概率空间内的偏差，反映模型对不确定性建模的能力。
- **可选指标**：可扩展 Hausdorff Distance、Boundary F1 等以评估边缘质量。

### 5. 实验配置（待补充）
- **数据集**：`data/train`、`data/val`，可注明来源（如医学影像、工业缺陷等）、样本数量及分辨率。
- **硬件环境**：GPU/CPU 型号、显存、PyTorch 版本等。
- **训练超参**：batch size、epoch 数、学习率策略（当前固定，可按需记录）。

### 6. 训练结果（待填写实测数据）
| 指标 | 数值 | 备注 |
| --- | --- | --- |
| 平均训练 Loss |  | BCE + Dice |
| 平均训练 IoU |  | 建议报告阈值 0.2 |
| 平均训练 Dice |  |  |
| 平均训练 Acc |  |  |
| 平均训练 MAE |  |  |

> 建议插入训练曲线（Loss/IoU）或 `PredvsGT/epoch_xxx.png`，形成完整实验记录。

### 7. 测试结果（待填写）
| 指标 | 数值 | 备注 |
| --- | --- | --- |
| IoU / Dice (mean ± std) |  | 可由 `test_model.py` 输出 |
| Acc / Precision / Recall / F1 |  |  |
| MAE / MSE |  |  |
| 推理耗时 |  | 可统计 ms/图 |

### 8. 困难与收获
1. **数据命名不一致**：通过取文件名交集 `common_keys` 并输出 warning，确保图像与掩码严格一一对应。
2. **极端类不平衡**：增加 `pos_weight` 与 Dice Loss，同时在评估阶段使用更低阈值（0.2）提升召回率。
3. **训练稳定性**：混合精度 + 梯度裁剪 + 动态日志三重保障，快速定位梯度消失或概率塌陷问题。
4. **可视化驱动调参**：差异图中红/蓝区域揭示 FP/FN，指导针对性增广与阈值调整。

---

## 题目二：基于 Context Encoders 的图像修复实现（代码：`context_encoders_new`）

### 1. 任务简介与理论背景
图像修复（Image Inpainting）旨在根据可见上下文预测缺损区域，属于典型的条件生成问题。Context Encoder（CE）通过自编码器结构学习全局语义，再借助对抗训练鼓励生成结果更逼真。理论上，CE 的目标可表示为：
\[
\min_G \max_D \ \mathbb{E}_{x \sim p_{\text{data}}}[\log D(\phi(x))] + \mathbb{E}_{\tilde{x}}[\log (1 - D(\phi(\tilde{x})))] + \lambda \|G(\tilde{x}) - \phi(x)\|_2^2
\]
其中 \(\phi(x)\) 表示图像中心块，\(\tilde{x}\) 为被遮挡后的图像。GAN 部分保证生成块的感知真实性，L2 项则确保与原始区域在像素空间一致。为了缓解边界错位，CE 额外对重叠区域加权，强化接缝平滑性。

### 2. 代码结构与模块说明
| 模块 | 功能 | 亮点 |
| --- | --- | --- |
| `main.py` | 自动解析数据目录、缺省时按比例随机划分 train/val/test，并初始化网络与训练器 | `_resolve_dir` 自动适配多路径；`_compute_split_sizes` 确保比例合法 |
| `dataset.py` | `ContextEncoderDataset` 统一读入文件夹/ImageFolder/CIFAR10，生成损坏图与中心块 | `_create_corrupted_image` 依照 ImageNet 均值进行填充，保证分布一致 |
| `model/net.py` | `ContextEncoderGenerator`（DCGAN 编码器+解码器）与 `ContextEncoderDiscriminator` | `weights_init` 采用 \(\mathcal{N}(0, 0.02)\) 初始化，符合 DCGAN 经验 |
| `model/training.py` | `ContextEncoderTrainer` 封装 GAN 训练、智能 L2 权重、日志记录、可视化、测试评估 | `_create_weight_matrix` 对重叠边缘乘 10 倍权重，改善接缝 |
| `metrics.py` | 实现 PSNR、SSIM、MSE、MAE 及批量统计工具 | `calculate_batch_metrics` 输出均值与标准差，供验证/测试使用 |

### 3. 模型与算法设计
#### 3.1 生成器架构
- **编码器**：6 层卷积（步长 2）逐层下采样至 \(1\times1\)，使用 `LeakyReLU` 激活。
- **瓶颈维度**：`n_bottleneck=4000`，兼顾语义表达能力与计算效率。
- **解码器**：镜像式反卷积堆叠，逐步恢复到 \(64 \times 64\) 中心块，`Tanh` 输出 \([-1,1]\) 范围，方便与归一化后的图像对齐。

#### 3.2 判别器架构
- 4 层卷积 + BatchNorm + `LeakyReLU`，最终使用 Sigmoid 输出真实性概率。输入为真实/生成的中心块，尺寸 \(64 \times 64\)。

#### 3.3 损失函数与权重
- **对抗损失**：`BCELoss`，鼓励生成块在感知层面逼真。
- **加权 L2 损失**：通过 `wtl2Matrix` 对重叠像素赋予更高权重（overlapL2Weight=10），理论上可抑制断边现象。
- **总损失**：\(\mathcal{L}_G = (1-\lambda)\mathcal{L}_{\text{adv}} + \lambda \mathcal{L}_{\text{L2}}\)（默认为 \(\lambda=0.998\)）。
- **判别器损失**：标准 GAN 目标 \(\mathcal{L}_D = -\mathbb{E}[\log D(x)] - \mathbb{E}[\log (1-D(G(\tilde{x})))]\)。

#### 3.4 训练策略
1. **优化器**：Adam（lr=2e-4, β₁=0.5），符合 DCGAN 的稳定训练经验。
2. **日志与可视化**：每个 epoch 同时生成 `cropped/real/recon` 样本与 `result/test/comparison_epoch_xxx.png`，便于定性评估。
3. **断点续训**：`--netG/--netD` 参数支持加载历史权重，`ContextEncoderTrainer` 会自动恢复 epoch 计数。
4. **指标统计**：训练/验证/测试全流程统一调用 `calculate_batch_metrics`，并写入 `logs/training_log_*.json`，方便后续绘制曲线或撰写报告。

### 4. 评价指标与理论说明
- **MSE / MAE**：衡量像素级误差；MSE 对离群值更敏感，MAE 更关注整体偏差。
- **PSNR**：\(20\log_{10}\frac{\text{MAX}}{\sqrt{\text{MSE}}}\)，用于度量生成图像与原图的信噪比，通常 >20 dB 表示较好的还原度。
- **SSIM**：结合亮度、对比度、结构三项，体现感知质量。0～1 之间，越接近 1 说明结构越相似。
- **对抗损失**：虽然无法直接反映视觉质量，但可作为训练稳定性的信号，当 `Loss_D` 长期过低或过高时需要调参。

### 5. 实验配置（示例）
- **数据集**：`data/train`（Paris StreetView 等），图像经 `Resize -> CenterCrop -> Normalize` 处理为 `128×128`。
- **遮挡策略**：中心块尺寸 \(64×64\)，重叠像素 `overlap_pred=4`，遮挡颜色按 ImageNet 均值填充，使输入分布与训练集一致。
- **运行环境**：CPU/GPU（示例日志为 CPU 训练），可在报告中补充显卡型号及训练时长。
- **超参**：batch size=4，epochs=25，可根据实际情况补充学习率、随机种子等信息。

### 6. 训练结果（基于 `training_log_20251126_101742.json`）
| Epoch | Loss_D | Loss_G_adv | Loss_G_L2 | 说明 |
| --- | --- | --- | --- | --- |
| 0 | 0.450 | 5.267 | 0.345 | 初期生成器尚未收敛，L2 较大 |
| 10 | 0.214 | 6.323 | 0.182 | 判别器趋稳，重建误差下降 |
| 20 | 0.362 | 4.965 | 0.142 | 对抗损失波动，L2 持续下降 |
| 24 | 0.376 | 4.605 | 0.129 | 训练结束，边界更平滑 |

> 建议在正式报告中附带 `result/train/recon_samples_epoch_024.png` 等图片展示视觉演化。

### 7. 测试结果（Epoch 24）
| 指标 | 数值 | 备注 |
| --- | --- | --- |
| PSNR | 18.95 dB | 来自 `training_log_20251126_101742.json` |
| SSIM | 0.474 | 结构相似度稳步提升 |
| MSE | 0.0165 | 与训练初期相比显著降低 |
| MAE | 0.0895 | 反映整体偏差 |
| Loss_D / Loss_G_adv / Loss_G_L2 | 1.61 / 2.06 / 0.125 | GAN 与 L2 同时收敛 |

可在附录中加入 `result/test/comparison_epoch_024.png`（展示“损坏→真实中心→生成中心→重建图像”）及 `result/test/reconstructed_epoch_024.png`。

### 8. 试验过程中遇到的困难与收获
1. **多种数据组织形式**：通过 `DirectImageDataset` + `ImageFolder` 兼容不同目录结构，并在加载失败时输出清晰错误提示，保证实验可复现性。
2. **GAN 震荡问题**：从原论文经验出发设置 β₁=0.5，并通过较大 `wtl2` 让 L2 主导，使生成器聚焦于结构正确性，对抗项则提供细节纹理。
3. **评估口径统一**：在验证与测试阶段共用 `calculate_batch_metrics`，并将均值±标准差写入 JSON，后续绘图或比较模型版本时更方便。
4. **可视化反馈**：多尺度可视化（损坏图、生成中心块、重建全图）帮助定位失真区域；对比图中按行拼接，可直接观察模型是否在纹理/色彩上存在系统性偏差。

---

### 结语
两项实验分别代表判别式（语义分割）与生成式（图像修复）的典型任务。从实现到评估均遵循“理论推导 → 代码结构 → 训练策略 → 指标/可视化”的完整路径。实际撰写期末报告时，可在上述框架基础上补充：
- 数据集描述与预处理细节；
- 训练/验证曲线及定性图像；
- 可能的改进方向（如引入 Attention UNet、使用 perceptual loss 等）；
- 与 baseline 或其他方法的横向对比。

以上内容已经按照学术报告的逻辑顺序排版，并留出必要的空位以便填写最终实验数据。若后续需要扩展附录或引用文献，可直接在文末追加章节。***

