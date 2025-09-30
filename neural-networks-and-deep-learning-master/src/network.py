import random
import numpy as np
try:
    from cffi.backend_ctypes import xrange  # 某些环境下可用
except Exception:
    # 兼容没有 cffi 的环境：使用 Python3 的 range 代替
    xrange = range


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
        # 形状说明：对第 l 层(1-based, 输入层为第1层)而言，b^l 的形状为 (sizes[l-1], 1)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # 初始化权重矩阵，连接相邻两层神经元
        # 权重矩阵的形状为(后一层神经元数, 前一层神经元数)
        # 即对第 l 层(从2开始)而言，W^l 的形状为 (sizes[l-1], sizes[l-2])
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # 调试打印选项（默认关闭）
        self.debug = False  # 是否开启调试打印
        self.debug_print_values = True  # 是否打印具体矩阵数值（否则只打印形状）
        self.debug_precision = 4  # 数值打印精度
        self.debug_edgeitems = 3  # 每边打印的元素个数（过大矩阵时会省略中间）
        self.debug_threshold = 1000  # 触发省略的阈值（元素总数超过则省略中间）
        self.debug_linewidth = 120  # 每行最大字符宽度

    # -------- Debug 辅助方法 --------
    def set_debug(self, enabled=True, print_values=True, precision=4, edgeitems=3, threshold=1000, linewidth=120):
        """配置调试打印
        enabled: 是否启用
        print_values: 打印形状之外是否打印矩阵具体数值
        precision/edgeitems/threshold/linewidth: 控制 numpy 打印格式
        """
        self.debug = bool(enabled)
        self.debug_print_values = bool(print_values)
        self.debug_precision = int(precision)
        self.debug_edgeitems = int(edgeitems)
        self.debug_threshold = int(threshold)
        self.debug_linewidth = int(linewidth)

    def _dbg(self, *args):
        if self.debug:
            print(*args)

    def _arr_to_str(self, arr):
        try:
            return np.array2string(
                np.asarray(arr),
                precision=self.debug_precision,
                edgeitems=self.debug_edgeitems,
                threshold=self.debug_threshold,
                max_line_width=self.debug_linewidth,
                suppress_small=False
            )
        except Exception:
            return str(arr)

    def _dbg_arr(self, title, arr):
        if not self.debug:
            return
        try:
            shape_str = getattr(arr, 'shape', None)
        except Exception:
            shape_str = None
        if self.debug_print_values and hasattr(arr, 'shape'):
            print(f"{title}: shape={arr.shape}\n{self._arr_to_str(arr)}")
        elif hasattr(arr, 'shape'):
            print(f"{title}: shape={arr.shape}")
        else:
            print(f"{title}: {arr}")

    def feedforward(self, a):
        # 前向传播(逐层)：输入 a 的形状为 (sizes[0], 1)
        self._dbg_arr("[FF] a^1 (input)", a)
        # 遍历网络中的每一层（从第二层开始，因为第一层是输入层）
        for b, w in zip(self.biases, self.weights):
            # 计算当前层的加权输入: z = W * a + b
            # 形状：w 为 (sizes[l], sizes[l-1])，a 为 (sizes[l-1], 1)
            #      np.dot(w, a) 为 (sizes[l], 1)，与 b (sizes[l], 1) 相加
            # 然后通过 sigmoid 激活函数得到当前层输出 a <- σ(z)，形状仍为 (sizes[l], 1)
            self._dbg_arr("[FF] W (layer)", w)
            self._dbg_arr("[FF] b (layer)", b)
            z = np.dot(w, a) + b
            self._dbg_arr("[FF] z (layer)", z)
            a = sigmoid(z)
            self._dbg_arr("[FF] a (layer output)", a)
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
            self._dbg(f"\n==== Epoch {j} start ====")
            # 每轮开始前随机打乱训练数据
            random.shuffle(training_data)

            # 将训练数据分割成多个小批量
            # mini_batch 的长度为 mini_batch_size，每个元素是 (x, y)
            # 其中 x 的形状为 (sizes[0], 1)，y 的形状通常为 (sizes[-1], 1) 或分类标签
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]

            # 对每个小批量数据进行训练
            for batch_index, mini_batch in enumerate(mini_batches):
                self._dbg(f"-- Mini-batch {batch_index} size={len(mini_batch)} --")
                self.update_mini_batch(mini_batch, eta)

            # 如果提供了测试数据，评估当前网络性能
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
            self._dbg(f"==== Epoch {j} end ====\n")

    def update_mini_batch(self, mini_batch, eta):
        """通过反向传播算法对单个小批量数据应用梯度下降来更新网络的权重和偏置
        mini_batch: 包含(x, y)元组的列表，表示训练样本和期望输出
        eta: 学习率"""

        # 初始化梯度累计变量，用于存储整个小批量数据的梯度和
        # 形状与参数一致：nabla_b[l] 形状为 (sizes[l+1], 1)，nabla_w[l] 为 (sizes[l+1], sizes[l])
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 遍历小批量中的每个训练样本
        for sample_index, (x, y) in enumerate(mini_batch):
            self._dbg(f"[MB] Sample {sample_index}")
            self._dbg_arr("[MB] x", x)
            self._dbg_arr("[MB] y", y)
            # 使用反向传播计算当前样本的梯度
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 累加所有样本的梯度
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # 打印当前样本的梯度
            for l, (dnb, dnw) in enumerate(zip(delta_nabla_b, delta_nabla_w), start=1):
                self._dbg_arr(f"[MB] grad nabla_b layer {l}", dnb)
                self._dbg_arr(f"[MB] grad nabla_w layer {l}", dnw)

        # 使用梯度下降更新权重和偏置
        # 权重更新公式: w = w - (eta/mini_batch_size) * ∇w
        # 形状保持与原权重一致；按元素相减
        self._dbg("[MB] Accumulated gradients (before update):")
        for l, (nb, nw) in enumerate(zip(nabla_b, nabla_w), start=1):
            self._dbg_arr(f"[MB] nabla_b sum layer {l}", nb)
            self._dbg_arr(f"[MB] nabla_w sum layer {l}", nw)

        old_weights = [w.copy() for w in self.weights]
        old_biases = [b.copy() for b in self.biases]

        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        # 偏置更新公式: b = b - (eta/mini_batch_size) * ∇b
        # 形状保持与原偏置一致；按元素相减
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

        # 打印参数更新的前后对比
        self._dbg("[MB] Parameters updated:")
        for l, (w_old, w_new) in enumerate(zip(old_weights, self.weights), start=1):
            self._dbg_arr(f"[MB] W layer {l} before", w_old)
            self._dbg_arr(f"[MB] W layer {l} after ", w_new)
        for l, (b_old, b_new) in enumerate(zip(old_biases, self.biases), start=1):
            self._dbg_arr(f"[MB] b layer {l} before", b_old)
            self._dbg_arr(f"[MB] b layer {l} after ", b_new)

    def backprop(self, x, y):
        """反向传播算法，计算损失函数对网络参数的梯度
        形状约定：
        - 第 l 层激活 a^l 的形状为 (sizes[l-1], 1)
        - 第 l 层加权输入 z^l 的形状为 (sizes[l-1], 1)
        - 第 l 层偏置梯度 nabla_b^l 与 b^l 同形状 (sizes[l-1], 1)
        - 第 l 层权重梯度 nabla_w^l 与 W^l 同形状 (sizes[l-1], sizes[l-2])
        """

        # 初始化梯度数组，用于存储各层偏置和权重的梯度
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播阶段：计算每一层的输出
        activation = x  # 初始激活值为输入x，形状 (sizes[0], 1)
        activations = [x]  # 存储所有层的激活值，第一层为输入
        zs = []  # 存储所有层的加权输入z值
        self._dbg_arr("[BP] a^1 (input)", activation)

        # 遍历每一层的权重和偏置
        for b, w in zip(self.biases, self.weights):
            self._dbg_arr("[BP] W (layer)", w)
            self._dbg_arr("[BP] b (layer)", b)
            z = np.dot(w, activation) + b  # 计算加权输入：z = w*a + b，形状与 b 相同 (sizes[l], 1)
            zs.append(z)  # 保存当前层的z值
            self._dbg_arr("[BP] z (layer)", z)
            activation = sigmoid(z)  # 通过激活函数得到当前层的输出，形状 (sizes[l], 1)
            self._dbg_arr("[BP] a (layer output)", activation)
            activations.append(activation)  # 保存当前层的激活值

        # 反向传播阶段：从输出层开始计算梯度
        # 计算输出层的误差项delta
        # 形状：cost_derivative(...) 与 activations[-1] 同形状 (sizes[-1], 1)
        #       sigmoid_prime(zs[-1]) 同形状；逐元素乘法保持 (sizes[-1], 1)
        delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
        self._dbg_arr("[BP] delta (output layer)", delta)

        # 设置输出层的梯度
        nabla_b[-1] = delta  # 输出层偏置梯度等于误差项，形状 (sizes[-1], 1)
        # 权重梯度为 delta * (a^{L-1})^T，形状 (sizes[-1], sizes[-2])
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        self._dbg_arr("[BP] nabla_b (output layer)", nabla_b[-1])
        self._dbg_arr("[BP] nabla_w (output layer)", nabla_w[-1])

        # 反向传播隐藏层：从倒数第二层开始向前计算梯度
        # l=2表示倒数第二层，l=self.num_layers-1表示第一层（输入层无参数）
        for l in xrange(2, self.num_layers):
            z = zs[-l]  # 当前层的加权输入，形状 (sizes[-l], 1)
            sp = sigmoid_prime(z)  # 当前层激活函数的导数，形状 (sizes[-l], 1)
            # 计算当前层误差：delta^l = (W^{l+1})^T * delta^{l+1} ⊙ σ'(z^l)
            # 形状：(W^{l+1})^T 为 (sizes[-l], sizes[-l+1])，与 delta^{l+1} (sizes[-l+1], 1) 相乘
            #      得到 (sizes[-l], 1)，再与 sp 逐元素相乘，形状保持 (sizes[-l], 1)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta  # 当前层偏置梯度等于误差项，形状 (sizes[-l], 1)
            # 权重梯度：delta^l * (a^{l-1})^T，形状 (sizes[-l], sizes[-l-1])
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
            self._dbg_arr(f"[BP] delta (layer -{l})", delta)
            self._dbg_arr(f"[BP] nabla_b (layer -{l})", nabla_b[-l])
            self._dbg_arr(f"[BP] nabla_w (layer -{l})", nabla_w[-l])

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
