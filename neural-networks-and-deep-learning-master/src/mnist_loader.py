"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    # 从压缩的pickle文件中加载原始MNIST数据
    # 返回三个元组：(训练数据, 验证数据, 测试数据)
    tr_d, va_d, te_d = load_data()

    # 处理训练数据
    # tr_d[0]: 训练图像数据，形状为(50000, 784)的numpy数组
    # tr_d[1]: 训练标签数据，形状为(50000,)的numpy数组，包含0-9的数字标签
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # 将每个784维的扁平图像重塑为(784, 1)的列向量
    # 从形状(784,) → (784, 1)，便于矩阵运算

    training_results = [vectorized_result(y) for y in tr_d[1]]
    # 将数字标签转换为one-hot编码向量
    # 例如：标签2 → [0,0,1,0,0,0,0,0,0,0]^T
    # 每个标签y通过vectorized_result函数转换为10维向量

    training_data = zip(training_inputs, training_results)
    # 将输入和标签配对，创建训练数据
    # 结果：[(x1, y1), (x2, y2), ..., (x50000, y50000)]
    # 其中xi是(784,1)的图像向量，yi是(10,1)的标签向量

    # 处理验证数据
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    # 验证图像：从(10000, 784) → 10000个(784, 1)向量
    validation_data = zip(validation_inputs, va_d[1])
    # 验证数据：[(x1, y1), (x2, y2), ..., (x10000, y10000)]
    # 注意：验证数据的标签保持原始数字形式，不转换为one-hot

    # 处理测试数据
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    # 测试图像：从(10000, 784) → 10000个(784, 1)向量
    test_data = zip(test_inputs, te_d[1])
    # 测试数据：[(x1, y1), (x2, y2), ..., (x10000, y10000)]
    # 注意：测试数据的标签也保持原始数字形式

    return (training_data, validation_data, test_data)
    # 返回处理后的三个数据集
    # training_data: 用于训练，标签为one-hot编码
    # validation_data: 用于验证，标签为原始数字
    # test_data: 用于测试，标签为原始数字


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
