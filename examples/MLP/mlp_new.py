import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

matplotlib.use('TkAgg')  # 或 'Agg' 不弹窗仅保存图像
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))

    ])

    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(train_dataset,
                                                                                      batch_size=batch_size,
                                                                                      shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], output_size=10, dropout_rate=0.2):
        super(MLP, self).__init__()
        #         创建网络层数
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        #         输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)


def main():
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    #     参数
    batch_size = 64  # 每个训练批次包含64个样本,用于控制每次参数更新时使用的数据量
    learning_rate = 0.001  # 学习率为0.001,用于控制模型参数的更新(步长)
    epochs = 10  # 训练轮数
    hidden_sizes = [256, 128, 64]  # 三个隐藏层,每个隐藏层有x个神经元

    # 加载数据
    train_loader, test_loader = load_data(batch_size)
    model = MLP(input_size=28 * 28,
                hidden_sizes=hidden_sizes,
                output_size=10,
                dropout_rate=0.3).to(device)
    print(model)  # 至此拼接了模型
#     训练模型
#     train_model


if __name__ == "__main__":
    main()
