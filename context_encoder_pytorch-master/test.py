from __future__ import print_function

import argparse

import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from model import _netG

parser = argparse.ArgumentParser()
# 指定数据集类型，默认值为'streetview'，可选值包括cifar10、lsun、imagenet等
parser.add_argument('--dataset', default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
# 指定数据集存放的位置，默认值为'dataset/val'
parser.add_argument('--dataroot', default='dataset/val', help='path to dataset')
# 设置工作进程数，默认值为2
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# 设置输入批次属性大小,默认值为64
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
# 设置图片大小，默认值为128
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
# 设置输入向量的大小，默认值为100
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
# 设置生成器通道数，默认值为64
parser.add_argument('--ngf', type=int, default=64)
# 配置判别器通道数，默认值为64
parser.add_argument('--ndf', type=int, default=64)
# 设置输入通道数，默认值为3（通常用于RGB图像）
parser.add_argument('--nc', type=int, default=3)
# 设置训练轮数，默认值为25，并提供帮助说明
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
# 设置学习率，默认值为0.0002
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
# 设置Adam优化器的beta1参数，默认值为0.5
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# 配置是否使用GPU，默认值为False
parser.add_argument('--cuda', action='store_true', help='enables cuda')
# 配置gpu个数
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# 指定生成器网络模型文件路径，用于继续训练
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
# 配置判别器网络模型文件路径，用于继续训练
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
# 指定输出文件夹路径，默认为当前目录('.')，用于保存图像和模型检查点
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
# 用于接收整数类型的种子值。该参数的作用是让用户可以手动设置随机数生成器的种子，以确保实验结果的可重现性。当用户提供该参数时，程序会使用指定的种子值来初始化随机数生成器。
parser.add_argument('--manualSeed', type=int, help='manual seed')
# 设置编码器瓶颈层的维度，默认值为4000
parser.add_argument('--nBottleneck', type=int, default=4000, help='of dim for bottleneck of encoder')
# 设置重叠边缘的整数值，默认为4
parser.add_argument('--overlapPred', type=int, default=4, help='overlapping edges')
# 设置编码器第一个卷积层的滤波器数量，默认值为64
parser.add_argument('--nef', type=int, default=64, help='of encoder filters in first conv layer')
# 控制是否使用权重衰减，默认值为0.999，0表示不使用
parser.add_argument('--wtl2', type=float, default=0.999, help='0 means do not use else use with this weight')
opt = parser.parse_args()
print(opt)

netG = _netG(opt)
# 加载生成器模型权重：从指定路径opt.netG加载预训练的生成器模型参数
netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
# 将生成器网络模型设置为评估模式
netG.eval()

# 这段代码定义了一个图像预处理流水线：
# transforms.Compose() - 将多个图像变换操作组合在一起
# transforms.Scale() - 将图像缩放到指定尺寸
# transforms.CenterCrop() - 从图像中心裁剪出指定大小的区域
# transforms.ToTensor() - 将PIL图像转换为PyTorch张量
# transforms.Normalize() - 对图像进行标准化处理，均值和标准差都设为0.5
# 整体功能是对输入图像进行标准化预处理，便于神经网络训练。
transform = transforms.Compose([transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 创建一个ImageFolder数据集对象，用于加载指定目录下的图像数据
dataset = dset.ImageFolder(root=opt.dataroot, transform=transform)
assert dataset
# 创建一个DataLoader对象，用于加载数据集
# DataLoader：PyTorch的数据加载工具
# dataset：传入的数据集对象
# batch_size：每个批次的样本数量，从配置参数获取
# shuffle=True：每个epoch随机打乱数据顺序
# num_workers：使用多进程加速数据加载，进程数由配置参数指定
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
# 两个张量的维度都是(批次大小, 3, 图像高度, 图像宽度)，其中3表示RGB三个颜色通道。这些张量作为神经网络的输入数据容器，opt.batchSize和opt.imageSize分别来自配置选项，指定批次大小和图像尺寸。
input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
# 裁剪图像的中心区域
real_center = torch.FloatTensor(opt.batchSize, 3, opt.imageSize / 2, opt.imageSize / 2)
# 这段代码创建了一个均方误差损失函数对象
criterionMSE = nn.MSELoss()

# 当启用CUDA时，将神经网络模型`netG`和相关张量数据（输入数据、损失函数、真实中心数据）都移动到GPU内存中，以便利用GPU进行加速计算。这是深度学习训练中常见的设备迁移操作。
if opt.cuda:
    netG.cuda()
    input_real, input_cropped = input_real.cuda(), input_cropped.cuda()
    criterionMSE.cuda()
    real_center = real_center.cuda()
# 这段代码的功能是将输入数据转换为PyTorch的Variable对象：
# 通过Variable包装，使这些张量能够参与自动梯度计算，便于后续的神经网络训练和反向传播。
input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
real_center = Variable(real_center)
# 将数据加载器转换为迭代器对象
dataiter = iter(dataloader)
# 从迭代器中获取下一个批次的数据，返回真实数据样本和对应的标签（标签用_忽略）
real_cpu, _ = dataiter.next()

# 将input_real张量调整为real_cpu的大小，并将real_cpu的数据复制到input_real中
# 作用：保存完整的原始图像，用于后续对比
input_real.data.resize_(real_cpu.size()).copy_(real_cpu)

# 将input_cropped张量调整为real_cpu的大小，并将real_cpu的数据复制到input_cropped中
# 作用：先复制完整图像，后续会在中心位置进行遮挡处理
input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)

# 从原始图像中提取中心区域（假设imageSize=128，则提取[32:96, 32:96]，即中心64×64的区域）
# 切片范围：从1/4处开始，取1/2的宽度和高度
# 作用：提取真实的中心区域，作为ground truth用于后续评估模型预测效果
real_center_cpu = real_cpu[:, :, opt.imageSize / 4:opt.imageSize / 4 + opt.imageSize / 2,
                  opt.imageSize / 4:opt.imageSize / 4 + opt.imageSize / 2]

# 将提取的中心区域复制到real_center张量中
real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

# ===== 以下代码对图像中心区域进行遮挡，模拟图像损坏场景 =====
# 对input_cropped的R通道(通道0)中心区域填充灰色值
# 遮挡范围：保留overlapPred(默认4)像素的边缘，只遮挡中心的56×56区域
# 填充值：2*117.0/255.0-1.0 ≈ -0.082（归一化到[-1,1]范围的ImageNet均值R通道）
input_cropped.data[:, 0, opt.imageSize / 4 + opt.overlapPred:opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred,
opt.imageSize / 4 + opt.overlapPred:opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred] = 2 * 117.0 / 255.0 - 1.0

# 对input_cropped的G通道(通道1)中心区域填充灰色值
# 填充值：2*104.0/255.0-1.0 ≈ -0.184（归一化到[-1,1]范围的ImageNet均值G通道）
input_cropped.data[:, 1, opt.imageSize / 4 + opt.overlapPred:opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred,
opt.imageSize / 4 + opt.overlapPred:opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred] = 2 * 104.0 / 255.0 - 1.0

# 对input_cropped的B通道(通道2)中心区域填充灰色值
# 填充值：2*123.0/255.0-1.0 ≈ -0.035（归一化到[-1,1]范围的ImageNet均值B通道）
# 完成后，input_cropped中心区域被填充为灰色，用于输入模型进行图像修复
input_cropped.data[:, 2, opt.imageSize / 4 + opt.overlapPred:opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred,
opt.imageSize / 4 + opt.overlapPred:opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred] = 2 * 123.0 / 255.0 - 1.0
# 使用生成器网络netG对裁剪后的输入图像进行前向传播,生成假图像fake
fake = netG(input_cropped)
# 使用均方误差(MSE)准则计算fake（生成的假数据）与real_center（真实数据中心区域）之间的差异，将结果赋值给errG作为生成器的损失。
errG = criterionMSE(fake, real_center)
# 创建输入图像的深拷贝副本
recon_image = input_cropped.clone()
# 将拷贝的图像数据进行切片操作,选择图像中心区域
recon_image.data[:, :, opt.imageSize / 4:opt.imageSize / 4 + opt.imageSize / 2,
opt.imageSize / 4:opt.imageSize / 4 + opt.imageSize / 2] = fake.data
# 保存输入图像、裁剪后的图像和修复后的图像
vutils.save_image(real_cpu, 'val_real_samples.png', normalize=True)
vutils.save_image(input_cropped.data, 'val_cropped_samples.png', normalize=True)
vutils.save_image(recon_image.data, 'val_recon_samples.png', normalize=True)
p = 0
l1 = 0
l2 = 0
# 将生成的假数据转换为NumPy数组
fake = fake.data.numpy()
# 将PyTorch张量real_center转换为NumPy数组格式
real_center = real_center.data.numpy()

from psnr import psnr
import numpy as np
# 计算真是中心点与虚假点坐标的差值
t = real_center - fake
# 用于衡量两个点集之间的平均距离误差。
l2 = np.mean(np.square(t))
l1 = np.mean(np.abs(t))
# 将真实中心区域数据从[-1,1]范围映射回[0,255]范围
real_center = (real_center + 1) * 127.5
fake = (fake + 1) * 127.5

for i in range(opt.batchSize):
    p = p + psnr(real_center[i].transpose(1, 2, 0), fake[i].transpose(1, 2, 0))

print(l2)

print(l1)

print(p / opt.batchSize)
