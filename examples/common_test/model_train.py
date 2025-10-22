import time
from copy import deepcopy

from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib
from model import LeNet
import torch


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
    stime = time.time()

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

