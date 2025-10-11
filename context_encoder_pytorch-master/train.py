"""
Context Encoder 训练脚本
使用GAN架构训练图像修复模型
- 生成器G：预测被遮挡的中心区域
- 判别器D：判断修复区域的真实性
"""
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from model import _netlocalD, _netG
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')

parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

# Context Encoder 特定参数
parser.add_argument('--nBottleneck', type=int, default=4000, help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred', type=int, default=4, help='overlapping edges')  # 保留边缘像素数
parser.add_argument('--nef', type=int, default=64, help='of encoder filters in first conv layer')
parser.add_argument('--wtl2', type=float, default=0.998,
                    help='0 means do not use else use with this weight')  # L2损失权重（99.8%）
parser.add_argument('--wtlD', type=float, default=0.001,
                    help='0 means do not use else use with this weight')  # 判别器损失权重（0.2%）

opt = parser.parse_args()
print(opt)

# 创建输出目录，用于保存训练过程中的图像和模型
try:
    os.makedirs("result/train/cropped")  # 保存遮挡后的图像
    os.makedirs("result/train/real")  # 保存原始图像
    os.makedirs("result/train/recon")  # 保存重建后的图像
    os.makedirs("model")  # 保存模型权重
except OSError:
    pass

# 设置随机种子，确保实验可重现
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
# 启动CuDNN的自动调优机制,会给每个卷积操作添加最优的配置
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# 当数据集类型为imagenet、folder或lfw时，创建一个图像文件夹数据集。
# 通过ImageFolder类加载指定路径的图片数据，并应用一系列图像变换处理：
# 缩放图片到指定尺寸、中心裁剪、转换为张量格式，最后进行归一化处理（均值和标准差都设为0.5）。
if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
                           )
elif opt.dataset == 'streetview':
    transform = transforms.Compose([transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dset.ImageFolder(root=opt.dataroot, transform=transform)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

# 网络参数配置
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3  # RGB三通道
nef = int(opt.nef)
nBottleneck = int(opt.nBottleneck)
wtl2 = float(opt.wtl2)
overlapL2Weight = 10  # 边缘区域的L2损失权重（是中心区域的10倍）


# 权重初始化函数：使用正态分布初始化网络权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)  # 卷积层：均值0，标准差0.02
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # BN层：均值1，标准差0.02
        m.bias.data.fill_(0)  # 偏置初始化为0


resume_epoch = 0  # 用于从checkpoint恢复训练时记录起始epoch

# ========== 创建生成器网络 ==========
netG = _netG(opt)  # 生成器：用于预测被遮挡的中心区域
netG.apply(weights_init)  # 应用权重初始化
if opt.netG != '':  # 如果提供了预训练模型路径，则加载
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
print(netG)

# ========== 创建判别器网络 ==========
netD = _netlocalD(opt)  # 局部判别器：只判别中心64×64区域的真伪
netD.apply(weights_init)  # 应用权重初始化
if opt.netD != '':  # 如果提供了预训练模型路径，则加载
    netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']
print(netD)

# ========== 定义损失函数 ==========
criterion = nn.BCELoss()  # 二元交叉熵损失：用于判别器的真/假分类
criterionMSE = nn.MSELoss()  # 均方误差损失：用于生成器的重建损失

# ========== 初始化数据张量 ==========
input_real = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)  # 原始完整图像
input_cropped = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)  # 中心被遮挡的图像
label = torch.FloatTensor(opt.batchSize)  # 判别器的标签
real_label = 1  # 真实样本标签
fake_label = 0  # 生成样本标签

real_center = torch.FloatTensor(opt.batchSize, 3, opt.imageSize / 2, opt.imageSize / 2)  # 真实的中心区域

# 如果使用CUDA，将所有模型和数据移到GPU
if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    criterionMSE.cuda()
    input_real, input_cropped, label = input_real.cuda(), input_cropped.cuda(), label.cuda()
    real_center = real_center.cuda()

# 将张量包装为Variable，以便进行自动求导
input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)
real_center = Variable(real_center)

# ========== 设置优化器 ==========
# 使用Adam优化器，学习率0.0002，beta1=0.5（GAN训练的常用设置）
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# ========================================
# 开始训练循环
# ========================================
for epoch in range(resume_epoch, opt.niter):
    for i, data in enumerate(dataloader, 0):
        # ===== 1. 数据准备 =====
        real_cpu, _ = data  # 获取一批真实图像

        # 提取图像中心区域（假设imageSize=128，提取[32:96, 32:96]，即中心64×64区域）
        real_center_cpu = real_cpu[:, :, int(opt.imageSize / 4):int(opt.imageSize / 4) + int(opt.imageSize / 2),
                          int(opt.imageSize / 4):int(opt.imageSize / 4) + int(opt.imageSize / 2)]

        batch_size = real_cpu.size(0)

        # 复制数据到GPU（如果使用CUDA）
        input_real.data.resize_(real_cpu.size()).copy_(real_cpu)  # 保存完整图像
        input_cropped.data.resize_(real_cpu.size()).copy_(real_cpu)  # 将被遮挡的图像
        real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)  # ground truth中心区域

        # 在中心区域填充灰色，模拟图像损坏（保留overlapPred=4像素的边缘）
        # 填充ImageNet均值颜色（归一化到[-1,1]范围）
        input_cropped.data[:, 0,
        int(opt.imageSize / 4 + opt.overlapPred):int(opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred),
        int(opt.imageSize / 4 + opt.overlapPred):int(
            opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred)] = 2 * 117.0 / 255.0 - 1.0  # R通道
        input_cropped.data[:, 1,
        int(opt.imageSize / 4 + opt.overlapPred):int(opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred),
        int(opt.imageSize / 4 + opt.overlapPred):int(
            opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred)] = 2 * 104.0 / 255.0 - 1.0  # G通道
        input_cropped.data[:, 2,
        int(opt.imageSize / 4 + opt.overlapPred):int(opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred),
        int(opt.imageSize / 4 + opt.overlapPred):int(
            opt.imageSize / 4 + opt.imageSize / 2 - opt.overlapPred)] = 2 * 123.0 / 255.0 - 1.0  # B通道

        # ===== 2. 训练判别器D =====
        # 目标：让判别器能够区分真实中心区域和生成的中心区域

        netD.zero_grad()  # 清空判别器梯度

        # (2.1) 训练判别器识别真实样本
        label.data.resize_(batch_size).fill_(real_label)  # 标签设为1（真）
        output = netD(real_center)  # 判别器判断真实中心区域
        errD_real = criterion(output, label)  # 计算损失：应该接近1
        errD_real.backward()  # 反向传播
        D_x = output.data.mean()  # 记录判别器对真实样本的平均输出

        # (2.2) 训练判别器识别生成样本
        fake = netG(input_cropped)  # 生成器预测被遮挡的中心区域
        label.data.fill_(fake_label)  # 标签设为0（假）
        output = netD(fake.detach())  # 判别生成的区域（detach()阻止梯度传到G）
        errD_fake = criterion(output, label)  # 计算损失：应该接近0
        errD_fake.backward()  # 反向传播
        D_G_z1 = output.data.mean()  # 记录判别器对生成样本的平均输出

        # (2.3) 更新判别器
        errD = errD_real + errD_fake  # 总损失
        optimizerD.step()  # 更新判别器参数

        # ===== 3. 训练生成器G =====
        # 目标：生成逼真的中心区域，既要骗过判别器，又要接近真实图像

        netG.zero_grad()  # 清空生成器梯度

        # (3.1) 对抗损失：让判别器认为生成的是真实的
        label.data.fill_(real_label)  # 标签设为1（真），希望判别器被骗
        output = netD(fake)  # 判别器判断生成的中心区域
        errG_D = criterion(output, label)  # 对抗损失：希望接近1

        # (3.2) L2重建损失（加权）- Context Encoder的关键创新
        # 创建权重矩阵：边缘区域权重更高，确保与原图自然融合
        wtl2Matrix = real_center.clone()
        wtl2Matrix.data.fill_(wtl2 * overlapL2Weight)  # 边缘权重 = 0.998 × 10 = 9.98
        # 中心区域权重较低
        wtl2Matrix.data[:, :, int(opt.overlapPred):int(opt.imageSize / 2 - opt.overlapPred),
        int(opt.overlapPred):int(opt.imageSize / 2 - opt.overlapPred)] = wtl2  # 中心权重 = 0.998

        # 计算加权的L2损失
        errG_l2 = (fake - real_center).pow(2)  # 逐像素平方差
        errG_l2 = errG_l2 * wtl2Matrix  # 应用权重矩阵
        errG_l2 = errG_l2.mean()  # 求平均

        # (3.3) 总损失 = 对抗损失 + L2重建损失
        # 权重配比：0.2% 对抗 + 99.8% L2
        errG = (1 - wtl2) * errG_D + wtl2 * errG_l2

        errG.backward()  # 反向传播

        D_G_z2 = output.data.mean()  # 记录判别器对生成样本的输出
        optimizerG.step()  # 更新生成器参数

        # ===== 4. 打印训练信息 =====
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG_D.data[0], errG_l2.data[0], D_x, D_G_z1,))

        # ===== 5. 定期保存可视化结果 =====
        if i % 100 == 0:  # 每100个batch保存一次
            # 保存原始图像
            vutils.save_image(real_cpu,
                              'result/train/real/real_samples_epoch_%03d.png' % (epoch))
            # 保存遮挡后的图像
            vutils.save_image(input_cropped.data,
                              'result/train/cropped/cropped_samples_epoch_%03d.png' % (epoch))
            # 保存重建后的完整图像
            recon_image = input_cropped.clone()
            recon_image.data[:, :, int(opt.imageSize / 4):int(opt.imageSize / 4 + opt.imageSize / 2),
            int(opt.imageSize / 4):int(opt.imageSize / 4 + opt.imageSize / 2)] = fake.data
            vutils.save_image(recon_image.data,
                              'result/train/recon/recon_center_samples_epoch_%03d.png' % (epoch))

    # ===== 6. 每个epoch结束后保存模型 =====
    torch.save({'epoch': epoch + 1,
                'state_dict': netG.state_dict()},
               'model/netG_streetview.pth')
    torch.save({'epoch': epoch + 1,
                'state_dict': netD.state_dict()},
               'model/netlocalD.pth')
