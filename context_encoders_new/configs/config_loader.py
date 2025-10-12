import sys
#sys.path.append('../lightNDF')
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
    parser.add_argument('--dataset', default='streetview', help='dataset type: streetview | imagenet | folder')
    parser.add_argument('--dataroot', default='data/train', help='path to dataset')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--image_size', type=int, default=128, help='the height / width of the input image')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam optimizer')
    
    # 网络结构参数
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')
    parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters')
    parser.add_argument('--nc', type=int, default=3, help='number of input channels')
    parser.add_argument('--nef', type=int, default=64, help='number of encoder filters')
    parser.add_argument('--n_bottleneck', type=int, default=4000, help='bottleneck dimension')
    
    # 损失权重参数
    parser.add_argument('--wtl2', type=float, default=0.998, help='L2 loss weight')
    parser.add_argument('--wtlD', type=float, default=0.001, help='discriminator loss weight')
    parser.add_argument('--overlap_pred', type=int, default=4, help='overlapping edges')
    
    # 设备相关
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--manual_seed', type=int, help='manual seed')
    
    # 模型保存和加载
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()
    
    # 设置随机种子
    if cfg.manual_seed is None:
        import random
        cfg.manual_seed = random.randint(1, 10000)
    
    return cfg

