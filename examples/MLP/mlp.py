import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

matplotlib.use('TkAgg')  # 或 'Agg' 不弹窗仅保存图像
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. 数据预处理和加载
def load_data(batch_size=64):
    """
    加载FashionMNIST数据集
    """
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下载训练集和测试集
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

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 设置shuffle为True, 则每次迭代都会打乱数据
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 2. 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[128, 64], output_size=10, dropout_rate=0.2):
        """
        多层感知机模型
        
        参数:
        - input_size: 输入维度 (28x28=784)
        - hidden_sizes: 隐藏层维度列表
        - output_size: 输出类别数 (10个类别)
        - dropout_rate: dropout比率
        """
        super(MLP, self).__init__()

        # 创建网络层
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))  # 输入层到第一个隐藏层
        layers.append(nn.ReLU())  # 激活函数
        layers.append(nn.Dropout(dropout_rate))  # dropout层

        # 添加中间隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)  # 创建网络, 将所有层连接起来

    def forward(self, x):
        # 将图像展平
        x = x.view(x.size(0), -1)
        return self.network(x)


# 3. 训练函数
def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.001):
    """
    训练模型
    """
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # 优化器

    # 记录训练过程
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()  # 清空梯度, 防止梯度累积
            loss.backward()  # 反向传播
            optimizer.step()  # 优化, 更新参数

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # 用于预测模型的准确率
            total += targets.size(0)  # 计算总样本数
            correct += (predicted == targets).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch + 1}/{epochs} | '
                      f'Batch: {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}')

        # 计算训练精度
        train_accuracy = 100 * correct / total
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # 在测试集上评估
        test_accuracy = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'  Train Loss: {train_losses[-1]:.4f}')
        print(f'  Train Accuracy: {train_accuracy:.2f}%')
        print(f'  Test Accuracy: {test_accuracy:.2f}%')
        print('-' * 50)

    return train_losses, train_accuracies, test_accuracies


# 4. 评估函数
def evaluate_model(model, test_loader):
    """
    在测试集上评估模型
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# 5. 可视化结果
def plot_results(train_losses, train_accuracies, test_accuracies):
    """
    绘制训练结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制损失
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 绘制精度
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# 6. 预测示例
def predict_example(model, test_loader, class_names):
    """
    预测并显示一些测试样本
    """
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # 取前6个样本
    images, labels = images[:6].to(device), labels[:6].to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # 显示结果
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for i in range(6):
        axes[i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axes[i].set_title(f'True: {class_names[labels[i]]}\nPred: {class_names[predicted[i]]}')
        axes[i].axis('off')
        axes[i].set_facecolor('lightgreen' if predicted[i] == labels[i] else 'lightcoral')

    plt.tight_layout()
    plt.show()


# 主函数
def main():
    # FashionMNIST类别名称
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 超参数
    batch_size = 64  # 每个训练批次包含64个样本,用于控制每次参数更新时使用的数据量
    learning_rate = 0.001  # 学习率为0.001,用于控制模型参数的更新(步长)
    epochs = 10  # 训练轮数
    hidden_sizes = [256, 128, 64]  # 三个隐藏层,每个隐藏层有x个神经元

    # 加载数据
    train_loader, test_loader = load_data(batch_size)

    # 创建模型
    model = MLP(
        input_size=28 * 28,
        hidden_sizes=hidden_sizes,
        output_size=10,
        dropout_rate=0.3
    ).to(device)

    print("模型结构:")
    print(model)
    print(f"总参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    train_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, epochs, learning_rate
    )

    # 绘制结果
    plot_results(train_losses, train_accuracies, test_accuracies)

    # 最终评估
    final_accuracy = evaluate_model(model, test_loader)
    print(f"最终测试集准确率: {final_accuracy:.2f}%")

    # 显示预测示例
    predict_example(model, test_loader, class_names)

    # 保存模型
    torch.save(model.state_dict(), 'mlp_fashionmnist.pth')
    print("模型已保存为 'mlp_fashionmnist.pth'")


if __name__ == "__main__":
    main()
