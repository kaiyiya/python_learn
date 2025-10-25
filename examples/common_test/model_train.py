import copy
import time
from copy import deepcopy

from matplotlib import pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from model import LeNet
import torch
import pandas as pt


def train_val_data_process():
    train_dataset = FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=224)
        ])
    )
    train_data, val_data = random_split(train_dataset, [round(0.8 * len(train_dataset)),
                                                        len(train_dataset) - round(
                                                            0.8 * len(train_dataset))])
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=128,
                                  shuffle=True,
                                  num_workers=8)

    val_dataloader = DataLoader(dataset=train_data,
                                batch_size=128,
                                shuffle=True,
                                num_workers=8)
    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    model.to(device)
    best_model_wts = deepcopy(model.state_dict())  # 复制当前模型参数
    #     初始化参数
    # 最高准确度
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        train_num = 0
        val_num = 0
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()  # 训练模式
            output = model(b_x)  # 前向传播过程, 得到预测结果
            pre_lab = torch.argmax(output, dim=1)  # 找到最大概率的结果下标
            loss = criterion(output, b_y)  # 通过损失函数计算损失
            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            train_loss += loss.item * b_x.size(0)
            train_acc += torch.sum(pre_lab == b_y)
            train_num += b_x.size(0)
        #     验证
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            # 前向传播
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            val_loss += loss.item * b_x.size(0)
            val_acc += torch.sum(pre_lab == b_y)
            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_acc.item() / train_num)
        val_acc_all.append(val_acc.item() / val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))
        # 寻找最高准确度的权重参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            # 保存当前的最高准确度
            best_model_wts = copy.deepcopy(model.state_dict())
        # 训练耗时
        time_use = time.time() - since
        print("训练耗时:{:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
        # 选择最优参数
        # 加载最高的准确率
        model.load_state_dict(best_model_wts)
        torch.save(model.load_state_dict(best_model_wts), 'lenet_fashionmnist.pth')
        # 保存所有数据
        train_process = pt.DataFrame(data={"epoch": range(num_epochs),
                                           "train_loss_all": train_loss_all,
                                           "train_acc_all": train_acc_all,
                                           "val_loss_all": val_loss_all,
                                           "val_acc_all": val_acc_all,
                                           "best_acc": best_acc,
                                           "time_use": time_use
                                           })
        return train_process


#         画图
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss_all"], 'ro-', label="train_loss")
    plt.plot(train_process["epoch"], train_process["val_loss_all"], 'bo-', label="val_loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc_all"], 'ro-', label="train_acc")
    plt.plot(train_process["epoch"], train_process["val_acc_all"], 'bo-', label="val_acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")


if __name__ == '__main__':
    # 实例化模型
    model = LeNet()
    # 加载数据集
    train_dataloader, val_dataloader = train_val_data_process()
    # 训练模型
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs=20)
    # 画图
    matplot_acc_loss(train_process)
