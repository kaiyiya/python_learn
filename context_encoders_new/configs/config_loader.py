import sys
# sys.path.append('../lightNDF')
import numpy as np
import os
import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    """
    Parse input arguments
    """
    parser = configargparse.ArgumentParser(description='Context Encoder for Image Inpainting')

    # 数据集相关参数
    # 默认是街景数据集
    parser.add_argument('--dataset', default='streetview', help='dataset type: streetview | imagenet | folder')
    # 设置数据集的根目录
    parser.add_argument('--dataroot', default='data/train', help='path to dataset')
    # 设置数据加载的进程数
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')  # 批处理大小,每次训练的批次
    parser.add_argument('--image_size', type=int, default=128,
                        help='the height / width of the input image')  # 图像会被调整到128*128像素
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')  # 整个数据集被训练的次数
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')  # 学习率learning_rate,也就是模型更新的步长
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam optimizer')  # Adam优化器的beta1参数,代表了对历史值的重视程度

    # 网络结构参数
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')  # 潜在向量z的维度
    parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')  # 生成器特征图数量,也就是滤波器
    parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters')  # 判别器特征图数量,也就是滤波器
    parser.add_argument('--nc', type=int, default=3, help='number of input channels')  # 输入的通道数
    parser.add_argument('--nef', type=int, default=64, help='number of encoder filters')  # 设置编码器滤波器的数量
    parser.add_argument('--n_bottleneck', type=int, default=4000, help='bottleneck dimension')  # 瓶颈层维度

    # 损失权重参数
    parser.add_argument('--wtl2', type=float, default=0.998, help='L2 loss weight')  # L2损失
    parser.add_argument('--wtlD', type=float, default=0.001, help='discriminator loss weight')  # 判别器损失
    parser.add_argument('--overlap_pred', type=int, default=4, help='overlapping edges')  # 边缘重叠像素

    # 设备相关
    parser.add_argument('--cuda', action='store_true', help='enables cuda')  # 是否使用显卡
    parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')  # 设置使用的GPU数量
    parser.add_argument('--manual_seed', type=int, help='manual seed')  # 随机数种子

    # 模型保存和加载
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")  # 生成器存储
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()  # 解析参数

    # 设置随机种子
    if cfg.manual_seed is None:
        import random
        cfg.manual_seed = random.randint(1, 10000)

    return cfg
