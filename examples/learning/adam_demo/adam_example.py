"""
Adam 优化器示例
----------------
本示例用 NumPy 手动实现 Adam 算法，并在一个简单的线性回归任务上进行演示。
目标函数为 y = 3x + 4，我们使用平方误差作为损失函数，Adam 将自动调整学习率，
帮助模型更快、更稳定地收敛。
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def adam_step(
    grads: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    lr: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据当前梯度执行一次 Adam 更新。

    参数说明：
        grads: 当前梯度。
        m, v: 一阶、二阶动量。
        t: 当前迭代步数（从 1 开始）。
        lr: 基础学习率。
        beta1, beta2: 动量衰减系数。
        eps: 防止除零的小常数。
    """
    m = beta1 * m + (1.0 - beta1) * grads
    v = beta2 * v + (1.0 - beta2) * (grads**2)

    m_hat = m / (1.0 - math.pow(beta1, t))
    v_hat = v / (1.0 - math.pow(beta2, t))

    update = lr * m_hat / (np.sqrt(v_hat) + eps)
    return update, m, v


def generate_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """生成线性回归数据集。"""
    x = np.linspace(-1, 1, n_samples).reshape(-1, 1)
    noise = np.random.normal(0, 0.05, size=(n_samples, 1))
    y = 3.0 * x + 4.0 + noise
    return x, y


def train(
    epochs: int = 200,
    lr: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> Tuple[np.ndarray, np.ndarray, Iterable[float]]:
    """使用 Adam 训练线性模型 y = w * x + b。"""
    rng = np.random.default_rng(20241110)
    x, y = generate_data()

    # 参数初始化
    w = rng.normal(size=(1, 1))
    b = rng.normal(size=(1, 1))

    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_b = np.zeros_like(b)
    v_b = np.zeros_like(b)

    losses = []
    for epoch in range(1, epochs + 1):
        # 模型预测
        y_pred = x @ w + b

        # 均方误差损失
        error = y_pred - y
        loss = float(np.mean(error**2))
        losses.append(loss)

        # 计算梯度
        grad_w = (2.0 / len(x)) * (x.T @ error)
        grad_b = (2.0 / len(x)) * np.sum(error, keepdims=True)

        # Adam 更新
        update_w, m_w, v_w = adam_step(grad_w, m_w, v_w, epoch, lr, beta1, beta2)
        update_b, m_b, v_b = adam_step(grad_b, m_b, v_b, epoch, lr, beta1, beta2)

        w -= update_w
        b -= update_b

        if epoch % 40 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss={loss:.6f} | w={w.item():.3f} | b={b.item():.3f}")

    return w, b, losses


def main() -> None:
    """执行训练并展示 Adam 的效果。"""
    w, b, losses = train()
    print("\n训练完成：")
    print(f"w ≈ {w.item():.3f}, b ≈ {b.item():.3f}")
    print(f"最后三次损失: {[round(l, 6) for l in losses[-3:]]}")


if __name__ == "__main__":
    main()

