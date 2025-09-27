import random
import numpy as np
from cffi.backend_ctypes import xrange


class Network(object):
    def __init__(self, sizes):
        """初始化神经网络
        sizes: 一个列表，包含每一层的神经元数量
               例如 [2, 3, 1] 表示输入层2个神经元，隐藏层3个神经元，输出层1个神经元"""

        # 网络层数等于sizes列表的长度
        self.num_layers = len(sizes)

        # 保存各层神经元数量
        self.sizes = sizes

        # 初始化偏置值，对除输入层外的每层生成随机偏置
        # 使用高斯分布(均值0, 方差1)初始化
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # 初始化权重矩阵，连接相邻两层神经元
        # 权重矩阵的形状为(后一层神经元数, 前一层神经元数)
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        # 遍历网络中的每一层（从第二层开始，因为第一层是输入层）
        for b, w in zip(self.biases, self.weights):
            # 计算当前层的加权输入: z = w*a + b
            # np.dot(w, a) 计算权重矩阵w与输入向量a的点积
            # 加上偏置项b得到加权输入z
            # 然后通过sigmoid激活函数得到当前层的输出
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """使用小批量随机梯度下降训练神经网络。
        training_data: 训练数据列表，每个元素是(x, y)元组，x是输入，y是期望输出
        epochs: 训练轮数
        mini_batch_size: 小批量数据大小
        eta: 学习率
        test_data: 测试数据（可选），用于跟踪训练进度"""

        # 如果提供了测试数据，获取测试数据集大小
        if test_data:
            n_test = len(test_data)

        # 获取训练数据集大小
        n = len(training_data)

        # 进行epochs轮训练
        for j in xrange(epochs):
            # 每轮开始前随机打乱训练数据
            random.shuffle(training_data)

            # 将训练数据分割成多个小批量
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            # 对每个小批量数据进行训练
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # 如果提供了测试数据，评估当前网络性能
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """通过反向传播算法对单个小批量数据应用梯度下降来更新网络的权重和偏置
        mini_batch: 包含(x, y)元组的列表，表示训练样本和期望输出
        eta: 学习率"""

        # 初始化梯度累计变量，用于存储整个小批量数据的梯度和
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 遍历小批量中的每个训练样本
        for x, y in mini_batch:
            # 使用反向传播计算当前样本的梯度
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 累加所有样本的梯度
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 使用梯度下降更新权重和偏置
        # 权重更新公式: w = w - (eta/mini_batch_size) * ∇w
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        # 偏置更新公式: b = b - (eta/mini_batch_size) * ∇b
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """反向传播算法，计算损失函数对网络参数的梯度"""

        # 初始化梯度数组，用于存储各层偏置和权重的梯度
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播阶段：计算每一层的输出
        activation = x  # 初始激活值为输入x
        activations = [x]  # 存储所有层的激活值，第一层为输入
        zs = []  # 存储所有层的加权输入z值

        # 遍历每一层的权重和偏置
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b  # 计算加权输入：z = w*a + b
            zs.append(z)  # 保存当前层的z值
            activation = sigmoid(z)  # 通过激活函数得到当前层的输出
            activations.append(activation)  # 保存当前层的激活值

        # 反向传播阶段：从输出层开始计算梯度
        # 计算输出层的误差项delta
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])

        # 设置输出层的梯度
        nabla_b[-1] = delta  # 输出层偏置梯度等于误差项
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # 输出层权重梯度等于误差项与前一层激活值的外积

        # 反向传播隐藏层：从倒数第二层开始向前计算梯度
        # l=2表示倒数第二层，l=self.num_layers-1表示第一层（输入层无参数）
        for l in xrange(2, self.num_layers):
            z = zs[-l]  # 当前层的加权输入
            sp = sigmoid_prime(z)  # 当前层激活函数的导数
            # 计算当前层的误差项：delta = ((w^(l+1))^T * delta^(l+1)) ⊙ sigmoid'(z^l)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta  # 当前层偏置梯度等于误差项
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())  # 当前层权重梯度等于误差项与前一层激活值的外积

        # 返回计算得到的偏置和权重梯度
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        # 对测试数据中的每个样本，计算网络预测结果和真实标签
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        # 统计预测正确的样本数量
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        # 计算二次损失函数对输出激活值的偏导数
        # 对于二次损失函数 C = 1/2 * ||output_activations - y||^2
        # 其对 output_activations 的偏导数为 (output_activations - y)
        return (output_activations - y)


def sigmoid(z):
    """Sigmoid激活函数，将任意实数映射到(0,1)区间
    公式: σ(z) = 1 / (1 + e^(-z))
    常用于神经网络的激活函数，使输出有界且可导"""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Sigmoid函数的导数
    公式: σ'(z) = σ(z) * (1 - σ(z))
    用于反向传播算法中计算梯度"""
    return sigmoid(z) * (1 - sigmoid(z))
