from mnist_loader import load_data_wrapper
from network import Network



def main():
    # 加载 MNIST 数据（包含在 data/mnist.pkl.gz）
    training_data, validation_data, test_data = load_data_wrapper()

    # 构建一个简单的网络 784-30-10
    net = Network([784, 30, 10])

    # 开启调试：打印矩阵形状与具体数值（可根据需要关闭具体数值）
    net.set_debug(enabled=True, print_values=True, precision=4, edgeitems=3, threshold=1000, linewidth=120)

    # 取一个很小的训练集子集，便于观察输出
    small_training = list(training_data)[:10]

    # 训练 1 个 epoch，mini-batch 大小为 5，学习率 3.0
    net.SGD(small_training, epochs=1, mini_batch_size=5, eta=3.0, test_data=list(test_data)[:20])


if __name__ == "__main__":
    main()


