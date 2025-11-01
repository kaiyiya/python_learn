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


def test_data_process():
    test_data = FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=28)
        ])
    )
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=0)

    return test_dataloader


def test_model_process(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    test_corrects = 0.0
    test_num = 0
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0)
    test_acc = test_corrects.double().item() / test_num
    print("测试集准确率:{:.4f}".format(test_acc))


if __name__ == '__main__':
    model = LeNet()
    model.load_state_dict(torch.load('lenet_fashionmnist.pth'))
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)


def newMain(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = device.to(device)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model = model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()
            print("预测值", classes[result], '========================', "真实值", classes[label])
