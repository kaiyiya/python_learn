"""
梯度下降（Batch Gradient Descent, BGD）与随机梯度下降（Stochastic Gradient Descent, SGD）示例。

我们依旧使用线性回归任务 y = 2x - 1，展示两种算法在 NumPy 环境中的更新过程差异。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class GDResult:
    weights: np.ndarray
    bias: float
    losses: List[float]


def generate_linear_data(n_samples: int = 100, seed: int = 20241110) -> Tuple[np.ndarray, np.ndarray]:
    """生成线性回归数据集。"""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.5, 1.5, n_samples).reshape(-1, 1)
    noise = rng.normal(0, 0.1, size=(n_samples, 1))
    y = 2.0 * x - 1.0 + noise
    return x, y


def batch_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
) -> GDResult:
    """标准批量梯度下降：每个迭代使用全量样本计算梯度。"""
    rng = np.random.default_rng(42)
    w = rng.normal(size=(x.shape[1], 1))
    b = float(rng.normal())
    losses: List[float] = []

    for epoch in range(1, epochs + 1):
        y_pred = x @ w + b
        error = y_pred - y
        loss = float(np.mean(error**2))
        losses.append(loss)

        grad_w = (2.0 / len(x)) * (x.T @ error)
        grad_b = (2.0 / len(x)) * np.sum(error)

        w -= lr * grad_w
        b -= lr * grad_b

        if epoch % 50 == 0 or epoch == 1:
            print(f"[BGD] epoch={epoch:3d}, loss={loss:.5f}, w={w.item():.3f}, b={b:.3f}")

    return GDResult(weights=w, bias=b, losses=losses)


def stochastic_gradient_descent(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 20,
) -> GDResult:
    """
    随机梯度下降：逐样本更新参数。

    为便于观察，默认 epoch 数少一些；一个 epoch 会遍历数据集一次。
    """
    rng = np.random.default_rng(7)
    w = rng.normal(size=(x.shape[1], 1))
    b = float(rng.normal())
    losses: List[float] = []
    indices = np.arange(len(x))

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        epoch_losses: List[float] = []

        for idx in indices:
            x_i = x[idx : idx + 1]
            y_i = y[idx : idx + 1]

            y_pred = float(x_i @ w + b)
            error = y_pred - float(y_i)
            loss = error**2
            epoch_losses.append(loss)

            grad_w = 2.0 * error * x_i.T
            grad_b = 2.0 * error

            w -= lr * grad_w
            b -= lr * grad_b

        epoch_loss = float(np.mean(epoch_losses))
        losses.append(epoch_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[SGD] epoch={epoch:3d}, loss={epoch_loss:.5f}, w={w.item():.3f}, b={b:.3f}")

    return GDResult(weights=w, bias=b, losses=losses)


def main() -> None:
    """比较批量梯度下降与随机梯度下降。"""
    x, y = generate_linear_data()

    print("=== 批量梯度下降（BGD） ===")
    bgd_result = batch_gradient_descent(x, y)

    print("\n=== 随机梯度下降（SGD） ===")
    sgd_result = stochastic_gradient_descent(x, y)

    print("\n最终参数对比：")
    print(f"BGD -> w={bgd_result.weights.item():.3f}, b={bgd_result.bias:.3f}, last loss={bgd_result.losses[-1]:.6f}")
    print(f"SGD -> w={sgd_result.weights.item():.3f}, b={sgd_result.bias:.3f}, last loss={sgd_result.losses[-1]:.6f}")


if __name__ == "__main__":
    main()

