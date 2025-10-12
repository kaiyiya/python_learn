# Context Encoder - 现代化PyTorch实现

这是一个基于现代化PyTorch实现的Context Encoder，用于图像修复任务。代码从原始的context_encoder_pytorch-master项目重构而来，使用了更现代的PyTorch写法和更好的代码组织。

## 主要改进

### 1. 现代化PyTorch写法
- 移除了过时的`Variable`和`torch.autograd.Variable`
- 使用现代的`torch.device`进行设备管理
- 改进了权重初始化方法
- 使用更清晰的网络结构定义

### 2. 更好的代码组织
- 模块化设计，每个组件职责明确
- 配置文件集中管理所有参数
- 数据集类支持多种数据格式
- 训练器类封装了完整的GAN训练逻辑

### 3. 增强的功能
- 支持多种数据集格式（ImageFolder、CIFAR-10等）
- 自动设备检测和CUDA优化
- 完整的检查点保存和加载
- 详细的训练日志和可视化

## 项目结构

```
context_encoders_new/
├── configs/
│   └── config_loader.py      # 配置文件管理
├── model/
│   ├── net.py               # 网络模型定义
│   └── training.py          # 训练逻辑
├── dataset.py               # 数据集类
├── main.py                  # 主程序入口
└── README.md               # 说明文档
```

## 使用方法

### 1. 基本训练

```bash
# 使用默认参数训练
python main.py

# 使用自定义参数训练
python main.py --batch_size 8 --niter 50 --lr 0.0001 --cuda

# 从检查点继续训练
python main.py --netG model/netG_context_encoder.pth --netD model/netD_context_encoder.pth
```

### 2. 测试模型结构

```bash
python main.py test
```

### 3. 主要参数说明

- `--dataset`: 数据集类型 (streetview/imagenet/folder/cifar10)
- `--dataroot`: 数据路径
- `--batch_size`: 批次大小
- `--image_size`: 图像尺寸
- `--niter`: 训练轮数
- `--lr`: 学习率
- `--wtl2`: L2损失权重
- `--cuda`: 启用CUDA加速

## 网络架构

### 生成器 (ContextEncoderGenerator)
- **编码器**: 128x128 → 1x1 (压缩特征)
- **解码器**: 1x1 → 64x64 (重建图像)
- 使用卷积和转置卷积层
- 包含批归一化和激活函数

### 判别器 (ContextEncoderDiscriminator)
- 64x64 → 1x1 (判断真假)
- 使用卷积层逐步降维
- 输出概率值

## 训练过程

1. **数据预处理**: 从完整图像创建损坏图像和中心区域
2. **判别器训练**: 识别真实和生成的图像
3. **生成器训练**: 结合对抗损失和重建损失
4. **智能权重**: 边缘区域权重更高，确保修复边界自然

## 损失函数

生成器总损失 = (1-wtl2) × 对抗损失 + wtl2 × L2重建损失

- **对抗损失**: 让生成器欺骗判别器
- **L2重建损失**: 确保生成图像与真实图像相似
- **智能权重矩阵**: 边缘区域权重更高

## 输出文件

训练过程中会生成以下文件：

- `result/train/real/`: 真实图像样本
- `result/train/cropped/`: 损坏图像样本  
- `result/train/recon/`: 重建图像样本
- `model/netG_context_encoder.pth`: 生成器模型
- `model/netD_context_encoder.pth`: 判别器模型

## 兼容性

代码保留了原有的UNet类和Trainer类，确保与现有代码的兼容性。你可以继续使用原有的接口，也可以使用新的ContextEncoderTrainer。

## 依赖要求

- PyTorch >= 1.8.0
- torchvision
- PIL
- numpy
- configargparse

## 注意事项

1. 确保数据路径正确
2. 根据GPU内存调整批次大小
3. 监控训练过程中的损失变化
4. 定期保存检查点以防训练中断

## 示例命令

```bash
# 小批次测试
python main.py --batch_size 2 --niter 1 --cuda

# 完整训练
python main.py --batch_size 8 --niter 100 --lr 0.0002 --cuda --wtl2 0.998

# 从检查点继续
python main.py --netG model/netG_context_encoder.pth --netD model/netD_context_encoder.pth --cuda
```
