import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

train_data = torchvision.datasets.MNIST(root='./mnist', train=True, download=True,
                                        transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='./mnist', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        # 定义第一个全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 定义激活函数
        self.relu = nn.ReLU()
        # 定义第二个全连接层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 定义第三个全连接层
        self.fc3 = nn.Linear(hidden_size, num_classes)

        # 定义forward函数
        # x输入的数据

    def forward(self, x):
        out = self.fc1(x)  # 第一层运算
        out = self.relu(out)  # 将上一步结果送入激活函数
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


input_size = 28 * 28
hidden_size = 512
num_classes = 10  # 输出大小
model = MLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 300 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))
# 测试网络
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

torch.save(model.state_dict(), 'model.ckpt')
