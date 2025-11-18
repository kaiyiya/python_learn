# 梯度下降与随机梯度下降快速指南

## 场景
- 任务：线性回归 `y = 2x - 1`。
- 目的：比较 **批量梯度下降（Batch Gradient Descent, BGD）** 和 **随机梯度下降（Stochastic Gradient Descent, SGD）** 在 NumPy 下的实现与表现。

## 两种方法一览
| 特性 | BGD | SGD |
| --- | --- | --- |
| 梯度估计 | 使用全量样本，梯度精确 | 每次只用一个样本，梯度噪声大 |
| 收敛速度 | 步伐稳定但可能慢 | 迭代快，但波动大 |
| 计算复杂度 | 单步成本高 | 单步轻量，适合大数据 |
| 收敛轨迹 | 平滑、稳定 | 呈“抖动”式前进，需要调学习率/动量 |

一句话记忆：**BGD 稳定但慢，SGD 快但抖**。

## 代码入口
- Python 脚本：`learning/optimization_basics/gradient_methods.py`
- 运行命令（PowerShell）：`python learning\optimization_basics\gradient_methods.py`
- 输出内容：每隔若干 epoch 打印损失、参数，最后对比两种方法收敛结果。

## 核心实现片段
```23:55:learning/optimization_basics/gradient_methods.py
def batch_gradient_descent(...):
    ...
    grad_w = (2.0 / len(x)) * (x.T @ error)
    grad_b = (2.0 / len(x)) * np.sum(error)
    w -= lr * grad_w
    b -= lr * grad_b
```

```64:98:learning/optimization_basics/gradient_methods.py
def stochastic_gradient_descent(...):
    ...
    for idx in indices:
        ...
        grad_w = 2.0 * error * x_i.T
        grad_b = 2.0 * error
        w -= lr * grad_w
        b -= lr * grad_b
```

## 学习建议
- **学习率**：BGD 可取稍大（如 0.05），SGD 通常更小以减弱震荡。
- **遍历顺序**：SGD 需在每个 epoch 打乱样本，避免次序偏差。
- **改进方向**：可加入 Mini-batch、Momentum、Adam 等，兼顾效率与稳定。

## 快速检查清单
- 数据生成：确认噪声水平是否合适。
- 参数打印：观察损失下降趋势是否符合预期。
- 若出现不收敛：降低学习率或增加 epoch。

