from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
import random

from context_encoders_new.model.net import ContextEncoderGenerator, ContextEncoderDiscriminator, weights_init


class ContextEncoderTrainer(object):
    """
    现代化的Context Encoder训练器
    实现GAN训练逻辑，包含生成器和判别器的对抗训练
    """

    def __init__(self, train_loader, opt):
        self.opt = opt
        self.train_loader = train_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() and opt.cuda else "cpu")

        # 设置随机种子
        self._set_seed()

        # 创建模型
        self.netG = ContextEncoderGenerator(opt).to(self.device)
        self.netD = ContextEncoderDiscriminator(opt).to(self.device)

        # 应用权重初始化
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # 加载预训练模型（如果存在）
        self._load_checkpoints()

        # 定义损失函数
        self.criterion = nn.BCELoss()
        self.criterionMSE = nn.MSELoss()

        # 定义优化器
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # 训练参数
        self.epochs = opt.niter
        self.wtl2 = opt.wtl2
        self.overlapL2Weight = 10  # 用于控制重叠区域权重

        # 创建输出目录
        self._create_directories()

        print(f"训练器初始化完成，使用设备: {self.device}")
        print(f"生成器参数数量: {sum(p.numel() for p in self.netG.parameters())}")
        print(f"判别器参数数量: {sum(p.numel() for p in self.netD.parameters())}")

    def _set_seed(self):
        """设置随机种子"""
        random.seed(self.opt.manual_seed)
        torch.manual_seed(self.opt.manual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.opt.manual_seed)
        print(f"随机种子设置为: {self.opt.manual_seed}")

    def _load_checkpoints(self):
        """加载检查点"""
        self.resume_epoch = 0

        if self.opt.netG != '':
            checkpoint = torch.load(self.opt.netG, map_location=self.device)
            self.netG.load_state_dict(checkpoint['state_dict'])
            self.resume_epoch = checkpoint['epoch']
            print(f"加载生成器检查点: {self.opt.netG}")

        if self.opt.netD != '':
            checkpoint = torch.load(self.opt.netD, map_location=self.device)
            self.netD.load_state_dict(checkpoint['state_dict'])
            self.resume_epoch = checkpoint['epoch']
            print(f"加载判别器检查点: {self.opt.netD}")

    def _create_directories(self):
        """创建输出目录"""
        directories = [
            "result/train/cropped",
            "result/train/real",
            "result/train/recon",
            "model"
        ]

        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError:
                pass

    def _create_weight_matrix(self, real_center):
        """
        创建智能权重矩阵
        边缘区域权重更高，确保修复边界自然
        """
        wtl2Matrix = real_center.clone()
        wtl2Matrix.data.fill_(self.wtl2 * self.overlapL2Weight)

        # 中心区域权重较低
        center_size = self.opt.image_size // 2
        wtl2Matrix.data[:, :,
        self.opt.overlap_pred:center_size - self.opt.overlap_pred,
        self.opt.overlap_pred:center_size - self.opt.overlap_pred] = self.wtl2

        return wtl2Matrix

    def train_model(self):
        """主训练循环"""
        print("开始训练...")

        for epoch in range(self.resume_epoch, self.epochs):
            epoch_losses = {'D': [], 'G_D': [], 'G_L2': []}

            for i, (corrupted_images, real_centers) in enumerate(self.train_loader):
                # 移动数据到设备
                corrupted_images = corrupted_images.to(self.device)
                real_centers = real_centers.to(self.device)
                batch_size = corrupted_images.size(0)

                # 创建标签 - 确保形状与判别器输出匹配
                real_label = torch.full((batch_size, 1), 1.0, device=self.device)
                fake_label = torch.full((batch_size, 1), 0.0, device=self.device)

                ############################
                # (1) 更新判别器网络
                ###########################
                self.netD.zero_grad()

                # 训练判别器识别真实图像
                output_real = self.netD(real_centers)#清除判别器网络的所有参数梯度
                # 调试信息：打印形状
                if i == 0 and epoch == 0:
                    print(f"Debug - output_real shape: {output_real.shape}")
                    print(f"Debug - real_label shape: {real_label.shape}")
                errD_real = self.criterion(output_real, real_label)
                errD_real.backward()
                D_x = output_real.mean().item()

                # 训练判别器识别生成图像
                fake_centers = self.netG(corrupted_images)
                output_fake = self.netD(fake_centers.detach())
                errD_fake = self.criterion(output_fake, fake_label)
                errD_fake.backward()
                D_G_z1 = output_fake.mean().item()

                errD = errD_real + errD_fake
                self.optimizerD.step()

                ############################
                # (2) 更新生成器网络
                ###########################
                self.netG.zero_grad()

                # 对抗损失：让生成器欺骗判别器
                output_fake = self.netD(fake_centers)
                errG_D = self.criterion(output_fake, real_label)

                # L2重建损失：确保生成图像与真实图像相似
                wtl2Matrix = self._create_weight_matrix(real_centers)
                errG_l2 = (fake_centers - real_centers).pow(2)
                errG_l2 = errG_l2 * wtl2Matrix
                errG_l2 = errG_l2.mean()

                # 总损失
                errG = (1 - self.wtl2) * errG_D + self.wtl2 * errG_l2
                errG.backward()

                D_G_z2 = output_fake.mean().item()
                self.optimizerG.step()

                # 记录损失
                epoch_losses['D'].append(errD.item())
                epoch_losses['G_D'].append(errG_D.item())
                epoch_losses['G_L2'].append(errG_l2.item())

                # 打印训练信息
                if i % 100 == 0:
                    print(f'[{epoch}/{self.epochs}][{i}/{len(self.train_loader)}] '
                          f'Loss_D: {errD.item():.4f} '
                          f'Loss_G: {errG_D.item():.4f}/{errG_l2.item():.4f} '
                          f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')

                # 保存样本图像
                if i % 100 == 0:
                    self._save_sample_images(epoch, corrupted_images, real_centers, fake_centers)

            # 打印epoch统计信息
            avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
            print(f'Epoch [{epoch}/{self.epochs}] - '
                  f'Avg Loss_D: {avg_losses["D"]:.4f} '
                  f'Avg Loss_G_D: {avg_losses["G_D"]:.4f} '
                  f'Avg Loss_G_L2: {avg_losses["G_L2"]:.4f}')

            # 保存模型检查点
            self._save_checkpoints(epoch)

    def _save_sample_images(self, epoch, corrupted_images, real_centers, fake_centers):
        """保存样本图像"""
        # 保存真实图像
        vutils.save_image(real_centers,
                          f'result/train/real/real_samples_epoch_{epoch:03d}.png',
                          normalize=True, nrow=4)

        # 保存损坏图像
        vutils.save_image(corrupted_images,
                          f'result/train/cropped/cropped_samples_epoch_{epoch:03d}.png',
                          normalize=True, nrow=4)

        # 保存重建图像
        recon_images = corrupted_images.clone()
        center_size = self.opt.image_size // 2
        center_start = self.opt.image_size // 4

        recon_images[:, :,
        center_start:center_start + center_size,
        center_start:center_start + center_size] = fake_centers

        vutils.save_image(recon_images,
                          f'result/train/recon/recon_samples_epoch_{epoch:03d}.png',
                          normalize=True, nrow=4)

    def _save_checkpoints(self, epoch):
        """保存模型检查点"""
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.netG.state_dict()
        }, 'model/netG_context_encoder.pth')

        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.netD.state_dict()
        }, 'model/netD_context_encoder.pth')


class Trainer(object):
    """
    保留原有的训练器类以保持兼容性
    """

    def __init__(self, train_loader, model, opt):
        self.args = opt
        self.train_loader = train_loader
        self.model = model
        self.criterion = torch.nn.functional.binary_cross_entropy
        self.optimizer = torch.optim.Adam(lr=0.003, params=model.parameters())
        self.epochs = 200
        self.model.cuda()

    def train_model(self):
        for epoch in range(self.epochs):
            losses = []
            for i, (img, mask) in enumerate(self.train_loader):
                img, mask = img.cuda(), mask.float().cuda()

                self.optimizer.zero_grad()
                output = self.model(img)
                loss = self.criterion(output, mask)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                if i % 10 == 0:
                    print(f'setp{i}---train_loss, ', loss.item())

            print(f'epoch{epoch}-----------loss:{sum(losses) / len(losses)}')
            if epoch % 10 == 0:
                torch.save(self.model, f'unet-{epoch}.pth.tar')

        torch.save(self.model, f'Unet-epochs{self.epochs}.pth')
