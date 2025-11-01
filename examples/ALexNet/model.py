import torch
from torch import nn
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.ReLU = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3)
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3)
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3)
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)
        #         平展
        self.flatten = nn.Flatten()
        # 全连接层
        self.f9 = nn.Linear(256 * 6 * 6, 4096)
        self.f10 = nn.Linear(4096, 4096)
        self.f11 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s4(x)
        x = self.ReLU(self.c5(x))
        x = self.ReLU(self.c6(x))
        x = self.ReLU(self.c7(x))
        x = self.s8(x)
        x = self.flatten(x)
        # droupout
        x = self.ReLU(self.f9(x))
        x = nn.Dropout(p=0.5)(x)
        x = self.ReLU(self.f10(x))
        x = nn.Dropout(p=0.5)(x)
        x = self.f11(x)
        return x


if __name__ == '__main__':
    model = AlexNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    s = summary(model, (1, 227, 227))
    print(s)
