import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        #     卷积层
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.sig = nn.Sigmoid()
        #     池化层
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        #     平展层
        self.flatten = nn.Flatten()
        # 修改全连接层以适应224x224输入：224->112->108->54，所以是16*54*54=46656
        self.f5 = nn.Linear(16 * 54 * 54, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        # 前向传播
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.sig(self.f5(x))
        x = self.sig(self.f6(x))
        x = self.f7(x)
        return x


if __name__ == '__main__':
    model = LeNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    s = summary(model, (1, 28, 28))
    print(s)
