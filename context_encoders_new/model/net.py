import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random


def weights_init(m):
    """自定义权重初始化函数"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ContextEncoderGenerator(nn.Module):
    """
    现代化的上下文编码器生成器
    使用更现代的PyTorch写法
    """
    def __init__(self, opt):
        super(ContextEncoderGenerator, self).__init__()
        self.ngpu = opt.ngpu
        self.nc = opt.nc
        self.nef = opt.nef
        self.ngf = opt.ngf
        self.n_bottleneck = opt.n_bottleneck
        
        # 编码器部分
        self.encoder = nn.Sequential(
            # 输入: (nc) x 128 x 128
            nn.Conv2d(self.nc, self.nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (nef) x 64 x 64
            
            nn.Conv2d(self.nef, self.nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace=True),
            # (nef) x 32 x 32
            
            nn.Conv2d(self.nef, self.nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (nef*2) x 16 x 16
            
            nn.Conv2d(self.nef * 2, self.nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (nef*4) x 8 x 8
            
            nn.Conv2d(self.nef * 4, self.nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nef * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (nef*8) x 4 x 4
            
            nn.Conv2d(self.nef * 8, self.n_bottleneck, 4, bias=False),
            nn.BatchNorm2d(self.n_bottleneck),
            nn.LeakyReLU(0.2, inplace=True),
            # (n_bottleneck) x 1 x 1
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            # 输入: (n_bottleneck) x 1 x 1
            nn.ConvTranspose2d(self.n_bottleneck, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32
            
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.decoder, 
                                             nn.parallel.data_parallel(self.encoder, input, range(self.ngpu)), 
                                             range(self.ngpu))
        else:
            encoded = self.encoder(input)
            output = self.decoder(encoded)
        return output


class ContextEncoderDiscriminator(nn.Module):
    """
    现代化的上下文编码器判别器
    用于判断修复区域是否真实
    """
    def __init__(self, opt):
        super(ContextEncoderDiscriminator, self).__init__()
        self.ngpu = opt.ngpu
        self.nc = opt.nc
        self.ndf = opt.ndf
        
        self.main = nn.Sequential(
            # 输入: (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32
            
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
            
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8
            
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4
            
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 1 x 1 x 1
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1)


# 为了兼容性，保留原有的UNet类
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(kernel_size=2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(kernel_size=2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(kernel_size=2)(enc3))

        bottleneck = self.bottleneck(nn.MaxPool2d(kernel_size=2)(enc4))

        dec4 = self.dec4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.upconv1(dec2), enc1), dim=1))

        return torch.sigmoid(self.out_conv(dec1))


if __name__ == "__main__":
    # 测试Context Encoder模型
    class Args:
        def __init__(self):
            self.ngpu = 1
            self.nc = 3
            self.nef = 64
            self.ngf = 64
            self.ndf = 64
            self.n_bottleneck = 4000
    
    opt = Args()
    
    # 测试生成器
    generator = ContextEncoderGenerator(opt)
    generator.apply(weights_init)
    
    # 测试判别器
    discriminator = ContextEncoderDiscriminator(opt)
    discriminator.apply(weights_init)
    
    # 测试前向传播
    x = torch.randn(2, 3, 128, 128)
    fake_center = torch.randn(2, 3, 64, 64)
    
    print("输入图像形状:", x.shape)
    generated = generator(x)
    print("生成器输出形状:", generated.shape)
    
    d_real = discriminator(fake_center)
    d_fake = discriminator(generated)
    print("判别器真实输出形状:", d_real.shape)
    print("判别器假图像输出形状:", d_fake.shape)
