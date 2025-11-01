from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('TkAgg')  # 或者 'Agg', 'Qt5Agg'

train_data = FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.Resize(size=224),
    transforms.ToTensor(),
]))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

# 获取一个批次的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break

batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
class_label = train_data.classes
print(class_label)

# 可视化一个批次数据
# plt.figure(figsize=(12, 5))
# for li in np.arange(len(batch_y)):
#     plt.subplot(4, 16, li + 1)
#     plt.imshow(batch_x[li, :, :], cmap=plt.cm.gray)
#     plt.title(class_label[batch_y[li]], size=10)
#     plt.axis('off')  # 修正：添加括号
# plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 可适当调整间距
# plt.show()
