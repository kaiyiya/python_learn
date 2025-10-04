"""
反向传播四个核心公式的详细推导和应用
===============================================

本文件详细推导反向传播算法的四个核心公式，并展示它们在实际代码中的应用。
基于 Michael Nielsen 的《Neural Networks and Deep Learning》第2章内容。

作者: AI Assistant
日期: 2024
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid激活函数"""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Sigmoid函数的导数"""
    return sigmoid(z) * (1 - sigmoid(z))

class BackpropDerivation:
    """
    反向传播公式推导类
    详细展示四个核心公式的数学推导过程
    """
    
    def __init__(self):
        self.network_structure = [784, 30, 10]  # 输入层、隐藏层、输出层
        self.cost_function = "quadratic"  # 使用二次损失函数
        
    def formula_1_output_layer_error(self):
        """
        公式1: 输出层误差项
        δ^L_j = ∂C/∂a^L_j * σ'(z^L_j)
        
        推导过程:
        1. 定义输出层误差: δ^L_j = ∂C/∂z^L_j
        2. 链式法则: ∂C/∂z^L_j = ∂C/∂a^L_j * ∂a^L_j/∂z^L_j
        3. 由于 a^L_j = σ(z^L_j), 所以 ∂a^L_j/∂z^L_j = σ'(z^L_j)
        4. 因此: δ^L_j = ∂C/∂a^L_j * σ'(z^L_j)
        """
        print("=" * 60)
        print("公式1: 输出层误差项 δ^L_j = ∂C/∂a^L_j * σ'(z^L_j)")
        print("=" * 60)
        
        print("\n【数学推导】")
        print("1. 定义输出层误差: δ^L_j = ∂C/∂z^L_j")
        print("2. 应用链式法则: ∂C/∂z^L_j = ∂C/∂a^L_j * ∂a^L_j/∂z^L_j")
        print("3. 由于 a^L_j = σ(z^L_j), 所以 ∂a^L_j/∂z^L_j = σ'(z^L_j)")
        print("4. 因此: δ^L_j = ∂C/∂a^L_j * σ'(z^L_j)")
        
        print("\n【二次损失函数下的具体形式】")
        print("对于二次损失函数 C = 1/2 * ||a^L - y||²:")
        print("∂C/∂a^L_j = a^L_j - y_j")
        print("因此: δ^L_j = (a^L_j - y_j) * σ'(z^L_j)")
        
        print("\n【代码实现】")
        print("在 network.py 的 backprop 方法中:")
        print("```python")
        print("# 计算输出层的误差项")
        print("delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])")
        print("# 其中 cost_derivative 返回 (a^L - y)")
        print("```")
        
        print("\n【矩阵形式】")
        print("δ^L = (a^L - y) ⊙ σ'(z^L)")
        print("其中 ⊙ 表示逐元素乘法")
        
        return "δ^L_j = (a^L_j - y_j) * σ'(z^L_j)"
    
    def formula_2_hidden_layer_error(self):
        """
        公式2: 隐藏层误差项
        δ^l_j = ∑_k w^{l+1}_{kj} * δ^{l+1}_k * σ'(z^l_j)
        
        推导过程:
        1. 定义隐藏层误差: δ^l_j = ∂C/∂z^l_j
        2. 链式法则: ∂C/∂z^l_j = ∑_k ∂C/∂z^{l+1}_k * ∂z^{l+1}_k/∂z^l_j
        3. 由于 z^{l+1}_k = ∑_i w^{l+1}_{ki} * a^l_i + b^{l+1}_k
        4. 所以 ∂z^{l+1}_k/∂z^l_j = w^{l+1}_{kj} * σ'(z^l_j)
        5. 因此: δ^l_j = ∑_k δ^{l+1}_k * w^{l+1}_{kj} * σ'(z^l_j)
        """
        print("\n" + "=" * 60)
        print("公式2: 隐藏层误差项 δ^l_j = ∑_k w^{l+1}_{kj} * δ^{l+1}_k * σ'(z^l_j)")
        print("=" * 60)
        
        print("\n【数学推导】")
        print("1. 定义隐藏层误差: δ^l_j = ∂C/∂z^l_j")
        print("2. 应用链式法则: ∂C/∂z^l_j = ∑_k ∂C/∂z^{l+1}_k * ∂z^{l+1}_k/∂z^l_j")
        print("3. 由于 z^{l+1}_k = ∑_i w^{l+1}_{ki} * a^l_i + b^{l+1}_k")
        print("4. 所以 ∂z^{l+1}_k/∂z^l_j = w^{l+1}_{kj} * σ'(z^l_j)")
        print("5. 因此: δ^l_j = ∑_k δ^{l+1}_k * w^{l+1}_{kj} * σ'(z^l_j)")
        
        print("\n【矩阵形式】")
        print("δ^l = ((W^{l+1})^T * δ^{l+1}) ⊙ σ'(z^l)")
        print("其中 (W^{l+1})^T 是权重矩阵的转置")
        
        print("\n【代码实现】")
        print("在 network.py 的 backprop 方法中:")
        print("```python")
        print("# 反向传播隐藏层")
        print("for l in xrange(2, self.num_layers):")
        print("    z = zs[-l]  # 当前层的加权输入")
        print("    sp = sigmoid_prime(z)  # 当前层激活函数的导数")
        print("    delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp")
        print("```")
        
        print("\n【具体例子 - 3层网络】")
        print("对于网络 [784, 30, 10]:")
        print("- 第2层(隐藏层)误差: δ^2 = (W^3)^T * δ^3 ⊙ σ'(z^2)")
        print("- 形状: (30, 10)^T * (10, 1) ⊙ (30, 1) = (30, 1)")
        
        return "δ^l = ((W^{l+1})^T * δ^{l+1}) ⊙ σ'(z^l)"
    
    def formula_3_bias_gradient(self):
        """
        公式3: 偏置梯度
        ∂C/∂b^l_j = δ^l_j
        
        推导过程:
        1. 由于 z^l_j = ∑_k w^l_{jk} * a^{l-1}_k + b^l_j
        2. 所以 ∂z^l_j/∂b^l_j = 1
        3. 因此: ∂C/∂b^l_j = ∂C/∂z^l_j * ∂z^l_j/∂b^l_j = δ^l_j * 1 = δ^l_j
        """
        print("\n" + "=" * 60)
        print("公式3: 偏置梯度 ∂C/∂b^l_j = δ^l_j")
        print("=" * 60)
        
        print("\n【数学推导】")
        print("1. 由于 z^l_j = ∑_k w^l_{jk} * a^{l-1}_k + b^l_j")
        print("2. 所以 ∂z^l_j/∂b^l_j = 1")
        print("3. 因此: ∂C/∂b^l_j = ∂C/∂z^l_j * ∂z^l_j/∂b^l_j = δ^l_j * 1 = δ^l_j")
        
        print("\n【矩阵形式】")
        print("∇b^l = δ^l")
        print("偏置梯度等于误差项")
        
        print("\n【代码实现】")
        print("在 network.py 的 backprop 方法中:")
        print("```python")
        print("# 设置输出层的梯度")
        print("nabla_b[-1] = delta  # 输出层偏置梯度等于误差项")
        print("")
        print("# 隐藏层梯度")
        print("nabla_b[-l] = delta  # 当前层偏置梯度等于误差项")
        print("```")
        
        print("\n【直观理解】")
        print("偏置的梯度就是该层的误差项，因为偏置直接影响神经元的激活")
        
        return "∇b^l = δ^l"
    
    def formula_4_weight_gradient(self):
        """
        公式4: 权重梯度
        ∂C/∂w^l_{jk} = a^{l-1}_k * δ^l_j
        
        推导过程:
        1. 由于 z^l_j = ∑_i w^l_{ji} * a^{l-1}_i + b^l_j
        2. 所以 ∂z^l_j/∂w^l_{jk} = a^{l-1}_k
        3. 因此: ∂C/∂w^l_{jk} = ∂C/∂z^l_j * ∂z^l_j/∂w^l_{jk} = δ^l_j * a^{l-1}_k
        """
        print("\n" + "=" * 60)
        print("公式4: 权重梯度 ∂C/∂w^l_{jk} = a^{l-1}_k * δ^l_j")
        print("=" * 60)
        
        print("\n【数学推导】")
        print("1. 由于 z^l_j = ∑_i w^l_{ji} * a^{l-1}_i + b^l_j")
        print("2. 所以 ∂z^l_j/∂w^l_{jk} = a^{l-1}_k")
        print("3. 因此: ∂C/∂w^l_{jk} = ∂C/∂z^l_j * ∂z^l_j/∂w^l_{jk} = δ^l_j * a^{l-1}_k")
        
        print("\n【矩阵形式】")
        print("∇W^l = δ^l * (a^{l-1})^T")
        print("权重梯度等于误差项与前一层激活值的外积")
        
        print("\n【代码实现】")
        print("在 network.py 的 backprop 方法中:")
        print("```python")
        print("# 输出层权重梯度")
        print("nabla_w[-1] = np.dot(delta, activations[-2].transpose())")
        print("")
        print("# 隐藏层权重梯度")
        print("nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())")
        print("```")
        
        print("\n【具体例子 - 3层网络】")
        print("对于网络 [784, 30, 10]:")
        print("- 第2层权重梯度: ∇W^2 = δ^2 * (a^1)^T")
        print("- 形状: (30, 1) * (1, 784) = (30, 784)")
        print("- 第3层权重梯度: ∇W^3 = δ^3 * (a^2)^T")
        print("- 形状: (10, 1) * (1, 30) = (10, 30)")
        
        print("\n【直观理解】")
        print("权重的梯度等于:")
        print("- 该层的误差项 (δ^l_j)")
        print("- 前一层对应神经元的激活值 (a^{l-1}_k)")
        print("这反映了权重对误差的贡献程度")
        
        return "∇W^l = δ^l * (a^{l-1})^T"
    
    def complete_backprop_algorithm(self):
        """
        完整的反向传播算法流程
        """
        print("\n" + "=" * 80)
        print("完整的反向传播算法流程")
        print("=" * 80)
        
        print("\n【算法步骤】")
        print("1. 前向传播: 计算所有层的激活值 a^l 和加权输入 z^l")
        print("2. 输出层误差: δ^L = (a^L - y) ⊙ σ'(z^L)")
        print("3. 反向传播误差: 从输出层到输入层，逐层计算 δ^l")
        print("4. 计算梯度: ∇b^l = δ^l, ∇W^l = δ^l * (a^{l-1})^T")
        
        print("\n【伪代码】")
        print("```")
        print("def backprop(x, y):")
        print("    # 前向传播")
        print("    activations = [x]")
        print("    zs = []")
        print("    for b, w in zip(biases, weights):")
        print("        z = np.dot(w, activation) + b")
        print("        zs.append(z)")
        print("        activation = sigmoid(z)")
        print("        activations.append(activation)")
        print("    ")
        print("    # 反向传播")
        print("    delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])")
        print("    nabla_b[-1] = delta")
        print("    nabla_w[-1] = np.dot(delta, activations[-2].transpose())")
        print("    ")
        print("    for l in range(2, num_layers):")
        print("        delta = np.dot(weights[-l+1].transpose(), delta) * sigmoid_prime(zs[-l])")
        print("        nabla_b[-l] = delta")
        print("        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())")
        print("    ")
        print("    return nabla_b, nabla_w")
        print("```")
        
        print("\n【时间复杂度】")
        print("- 前向传播: O(∑_l n_l * n_{l-1})")
        print("- 反向传播: O(∑_l n_l * n_{l-1})")
        print("- 总复杂度: O(∑_l n_l * n_{l-1})")
        print("其中 n_l 是第l层的神经元数量")
    
    def practical_example(self):
        """
        实际例子: 3层网络 [784, 30, 10] 的反向传播
        """
        print("\n" + "=" * 80)
        print("实际例子: 3层网络 [784, 30, 10] 的反向传播")
        print("=" * 80)
        
        print("\n【网络结构】")
        print("输入层: 784个神经元")
        print("隐藏层: 30个神经元")
        print("输出层: 10个神经元")
        
        print("\n【矩阵形状】")
        print("W^2: (30, 784) - 输入层到隐藏层权重")
        print("b^2: (30, 1)  - 隐藏层偏置")
        print("W^3: (10, 30) - 隐藏层到输出层权重")
        print("b^3: (10, 1)  - 输出层偏置")
        
        print("\n【前向传播】")
        print("a^1 = x                    # 输入 (784, 1)")
        print("z^2 = W^2 * a^1 + b^2      # 隐藏层加权输入 (30, 1)")
        print("a^2 = σ(z^2)               # 隐藏层激活 (30, 1)")
        print("z^3 = W^3 * a^2 + b^3      # 输出层加权输入 (10, 1)")
        print("a^3 = σ(z^3)               # 输出层激活 (10, 1)")
        
        print("\n【反向传播】")
        print("δ^3 = (a^3 - y) ⊙ σ'(z^3)  # 输出层误差 (10, 1)")
        print("δ^2 = (W^3)^T * δ^3 ⊙ σ'(z^2)  # 隐藏层误差 (30, 1)")
        
        print("\n【梯度计算】")
        print("∇b^3 = δ^3                # 输出层偏置梯度 (10, 1)")
        print("∇W^3 = δ^3 * (a^2)^T      # 输出层权重梯度 (10, 1) × (1, 30) = (10, 30)")
        print("∇b^2 = δ^2                # 隐藏层偏置梯度 (30, 1)")
        print("∇W^2 = δ^2 * (a^1)^T      # 隐藏层权重梯度 (30, 1) × (1, 784) = (30, 784)")
        
        print("\n【参数更新】")
        print("W^2 = W^2 - η * ∇W^2      # 更新隐藏层权重")
        print("b^2 = b^2 - η * ∇b^2      # 更新隐藏层偏置")
        print("W^3 = W^3 - η * ∇W^3      # 更新输出层权重")
        print("b^3 = b^3 - η * ∇b^3      # 更新输出层偏置")
    
    def run_complete_derivation(self):
        """
        运行完整的推导过程
        """
        print("反向传播四个核心公式的详细推导")
        print("=" * 80)
        
        # 推导四个公式
        formula1 = self.formula_1_output_layer_error()
        formula2 = self.formula_2_hidden_layer_error()
        formula3 = self.formula_3_bias_gradient()
        formula4 = self.formula_4_weight_gradient()
        
        # 完整算法
        self.complete_backprop_algorithm()
        
        # 实际例子
        self.practical_example()
        
        print("\n" + "=" * 80)
        print("总结: 四个核心公式")
        print("=" * 80)
        print(f"1. 输出层误差: {formula1}")
        print(f"2. 隐藏层误差: {formula2}")
        print(f"3. 偏置梯度: {formula3}")
        print(f"4. 权重梯度: {formula4}")
        
        print("\n这些公式构成了反向传播算法的数学基础，")
        print("通过链式法则将误差从输出层反向传播到输入层，")
        print("从而计算每个参数的梯度，实现神经网络的训练。")

def main():
    """
    主函数：运行反向传播公式推导
    """
    derivation = BackpropDerivation()
    derivation.run_complete_derivation()

if __name__ == "__main__":
    main()
