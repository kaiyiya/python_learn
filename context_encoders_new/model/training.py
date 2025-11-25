from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
from datetime import datetime
import json


class ContextEncoderTrainer(object):
    """
    现代化的Context Encoder训练器
    实现GAN训练逻辑，包含生成器和判别器的对抗训练
    """

    def __init__(self, train_loader, generator, discriminator, opt, device, val_loader=None, test_loader=None):
        self.opt = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        # 使用传入的模型
        self.netG = generator.to(self.device)
        self.netD = discriminator.to(self.device)

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
        
        # 初始化日志记录
        self._init_logging()

        print(f"训练器初始化完成，使用设备: {self.device}")
        print(f"生成器参数数量: {sum(p.numel() for p in self.netG.parameters())}")
        print(f"判别器参数数量: {sum(p.numel() for p in self.netD.parameters())}")

    def _load_checkpoints(self):
        """加载检查点"""
        self.resume_epoch = 0

        if hasattr(self.opt, 'netG') and self.opt.netG != '':
            if os.path.exists(self.opt.netG):
                checkpoint = torch.load(self.opt.netG, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.netG.load_state_dict(checkpoint['state_dict'])
                    self.resume_epoch = checkpoint.get('epoch', 0)
                else:
                    self.netG.load_state_dict(checkpoint)
                print(f"加载生成器检查点: {self.opt.netG}, epoch: {self.resume_epoch}")

        if hasattr(self.opt, 'netD') and self.opt.netD != '':
            if os.path.exists(self.opt.netD):
                checkpoint = torch.load(self.opt.netD, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.netD.load_state_dict(checkpoint['state_dict'])
                    resume_epoch_d = checkpoint.get('epoch', 0)
                else:
                    self.netD.load_state_dict(checkpoint)
                    resume_epoch_d = 0
                if resume_epoch_d > self.resume_epoch:
                    self.resume_epoch = resume_epoch_d
                print(f"加载判别器检查点: {self.opt.netD}, epoch: {resume_epoch_d}")

    def _create_directories(self):
        """创建输出目录"""
        directories = [
            "result/train/cropped",
            "result/train/real",
            "result/train/recon",
            "result/test",
            "logs",
            "model"
        ]

        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError:
                pass
    
    def _init_logging(self):
        """初始化日志记录"""
        # 创建带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/training_log_{timestamp}.txt"
        self.json_log_file = f"logs/training_log_{timestamp}.json"
        
        # 初始化JSON日志数据结构
        self.training_log = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "epochs": self.epochs,
                "batch_size": self.opt.batch_size,
                "learning_rate": self.opt.lr,
                "wtl2": self.wtl2,
                "image_size": self.opt.image_size,
                "device": str(self.device)
            },
            "epochs": []
        }
        
        # 写入初始信息
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"训练日志 - 开始时间: {self.training_log['start_time']}\n")
            f.write("=" * 80 + "\n")
            f.write(f"配置信息:\n")
            for key, value in self.training_log['config'].items():
                f.write(f"  {key}: {value}\n")
            f.write("=" * 80 + "\n\n")
        
        print(f"✓ 日志文件已创建: {self.log_file}")
    
    def _log(self, message, to_console=True):
        """记录日志信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message)
        
        if to_console:
            print(message)

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
        log_msg = f'开始训练，共 {self.epochs} 个epoch\n{"=" * 80}'
        self._log(log_msg)

        for epoch in range(self.resume_epoch, self.epochs):
            self.netG.train()
            self.netD.train()
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
                output_real = self.netD(real_centers)  # 清除判别器网络的所有参数梯度
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
                    log_msg = (f'[{epoch}/{self.epochs}][{i}/{len(self.train_loader)}] '
                              f'Loss_D: {errD.item():.4f} '
                              f'Loss_G: {errG_D.item():.4f}/{errG_l2.item():.4f} '
                              f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
                    self._log(log_msg)

                # 保存样本图像
                if i % 100 == 0:
                    self._save_sample_images(epoch, corrupted_images, real_centers, fake_centers)

            # 打印epoch统计信息
            avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
            
            # 创建当前epoch的日志记录
            self.current_epoch_log = {
                "epoch": epoch,
                "train": {
                    "Loss_D": avg_losses["D"],
                    "Loss_G_D": avg_losses["G_D"],
                    "Loss_G_L2": avg_losses["G_L2"]
                }
            }
            
            log_msg = (f'\n{"=" * 80}\n'
                      f'Epoch [{epoch}/{self.epochs}] 总结:\n'
                      f'  平均 Loss_D: {avg_losses["D"]:.6f}\n'
                      f'  平均 Loss_G_D: {avg_losses["G_D"]:.6f}\n'
                      f'  平均 Loss_G_L2: {avg_losses["G_L2"]:.6f}\n'
                      f'{"=" * 80}')
            self._log(log_msg)

            # 验证集评估
            if self.val_loader is not None:
                self._validate(epoch)

            # 测试集评估（每个epoch或特定epoch）
            if self.test_loader is not None:
                # 可以选择每个epoch都测试，或者只在特定epoch测试
                # 这里设置为每个epoch都测试，你也可以改为 epoch % 5 == 0 等
                self._test(epoch)
            
            # 将当前epoch的日志添加到总日志中
            if hasattr(self, 'current_epoch_log'):
                self.training_log['epochs'].append(self.current_epoch_log.copy())
                # 保存JSON日志（每个epoch都保存，方便随时查看）
                with open(self.json_log_file, 'w', encoding='utf-8') as f:
                    json.dump(self.training_log, f, indent=2, ensure_ascii=False)

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

    def _validate(self, epoch):
        """在验证集上评估模型"""
        self.netG.eval()
        self.netD.eval()

        val_losses = {'D': [], 'G_D': [], 'G_L2': []}

        with torch.no_grad():
            for corrupted_images, real_centers in self.val_loader:
                corrupted_images = corrupted_images.to(self.device)
                real_centers = real_centers.to(self.device)
                batch_size = corrupted_images.size(0)

                real_label = torch.full((batch_size, 1), 1.0, device=self.device)
                fake_label = torch.full((batch_size, 1), 0.0, device=self.device)

                # 判别器评估
                output_real = self.netD(real_centers)
                fake_centers = self.netG(corrupted_images)
                output_fake = self.netD(fake_centers)

                errD_real = self.criterion(output_real, real_label)
                errD_fake = self.criterion(output_fake, fake_label)
                errD = errD_real + errD_fake

                # 生成器评估
                errG_D = self.criterion(output_fake, real_label)
                wtl2Matrix = self._create_weight_matrix(real_centers)
                errG_l2 = (fake_centers - real_centers).pow(2)
                errG_l2 = errG_l2 * wtl2Matrix
                errG_l2 = errG_l2.mean()

                val_losses['D'].append(errD.item())
                val_losses['G_D'].append(errG_D.item())
                val_losses['G_L2'].append(errG_l2.item())

        avg_val_losses = {k: sum(v) / len(v) for k, v in val_losses.items()}
        log_msg = (f'验证集:\n'
                  f'  平均 Val Loss_D: {avg_val_losses["D"]:.6f}\n'
                  f'  平均 Val Loss_G_D: {avg_val_losses["G_D"]:.6f}\n'
                  f'  平均 Val Loss_G_L2: {avg_val_losses["G_L2"]:.6f}\n'
                  f'{"=" * 80}\n')
        self._log(log_msg)
        
        # 更新JSON日志
        if hasattr(self, 'current_epoch_log'):
            self.current_epoch_log["val"] = {
                "Loss_D": avg_val_losses["D"],
                "Loss_G_D": avg_val_losses["G_D"],
                "Loss_G_L2": avg_val_losses["G_L2"]
            }

        self.netG.train()
        self.netD.train()

    def _test(self, epoch):
        """在测试集上评估模型"""
        self.netG.eval()
        self.netD.eval()

        test_losses = {'D': [], 'G_D': [], 'G_L2': []}
        
        # 收集所有测试图片用于生成大图
        all_corrupted = []
        all_real_centers = []
        all_fake_centers = []
        all_recon_images = []

        with torch.no_grad():
            for corrupted_images, real_centers in self.test_loader:
                corrupted_images = corrupted_images.to(self.device)
                real_centers = real_centers.to(self.device)
                batch_size = corrupted_images.size(0)

                real_label = torch.full((batch_size, 1), 1.0, device=self.device)
                fake_label = torch.full((batch_size, 1), 0.0, device=self.device)

                # 判别器评估
                output_real = self.netD(real_centers)
                fake_centers = self.netG(corrupted_images)
                output_fake = self.netD(fake_centers)

                errD_real = self.criterion(output_real, real_label)
                errD_fake = self.criterion(output_fake, fake_label)
                errD = errD_real + errD_fake

                # 生成器评估
                errG_D = self.criterion(output_fake, real_label)
                wtl2Matrix = self._create_weight_matrix(real_centers)
                errG_l2 = (fake_centers - real_centers).pow(2)
                errG_l2 = errG_l2 * wtl2Matrix
                errG_l2 = errG_l2.mean()

                test_losses['D'].append(errD.item())
                test_losses['G_D'].append(errG_D.item())
                test_losses['G_L2'].append(errG_l2.item())
                
                # 收集图片用于生成大图
                all_corrupted.append(corrupted_images.cpu())
                all_real_centers.append(real_centers.cpu())
                all_fake_centers.append(fake_centers.cpu())
                
                # 创建重建图像（将生成的中心区域放回损坏图像中）
                recon_images = corrupted_images.clone()
                center_size = self.opt.image_size // 2
                center_start = self.opt.image_size // 4
                recon_images[:, :,
                    center_start:center_start + center_size,
                    center_start:center_start + center_size] = fake_centers
                all_recon_images.append(recon_images.cpu())

        # 合并所有批次
        all_corrupted = torch.cat(all_corrupted, dim=0)
        all_real_centers = torch.cat(all_real_centers, dim=0)
        all_fake_centers = torch.cat(all_fake_centers, dim=0)
        all_recon_images = torch.cat(all_recon_images, dim=0)
        
        # 生成测试结果大图
        self._save_test_montage(all_corrupted, all_real_centers, all_fake_centers, 
                                all_recon_images, epoch)

        avg_test_losses = {k: sum(v) / len(v) for k, v in test_losses.items()}
        log_msg = (f'测试集:\n'
                  f'  平均 Test Loss_D: {avg_test_losses["D"]:.6f}\n'
                  f'  平均 Test Loss_G_D: {avg_test_losses["G_D"]:.6f}\n'
                  f'  平均 Test Loss_G_L2: {avg_test_losses["G_L2"]:.6f}\n'
                  f'{"=" * 80}\n')
        self._log(log_msg)
        
        # 更新JSON日志
        if hasattr(self, 'current_epoch_log'):
            self.current_epoch_log["test"] = {
                "Loss_D": avg_test_losses["D"],
                "Loss_G_D": avg_test_losses["G_D"],
                "Loss_G_L2": avg_test_losses["G_L2"]
            }

        self.netG.train()
        self.netD.train()
    
    def _save_test_montage(self, corrupted_images, real_centers, fake_centers, 
                          recon_images, epoch):
        """保存测试集所有图片排列成的大图"""
        num_images = corrupted_images.size(0)
        
        # 计算网格大小（尽量接近正方形）
        nrow = int(num_images ** 0.5) + 1
        if nrow * (nrow - 1) >= num_images:
            nrow = nrow - 1
        
        # 保存损坏图像大图
        vutils.save_image(
            corrupted_images,
            f'result/test/corrupted_epoch_{epoch:03d}.png',
            normalize=True,
            nrow=nrow
        )
        
        # 保存真实中心区域大图
        vutils.save_image(
            real_centers,
            f'result/test/real_centers_epoch_{epoch:03d}.png',
            normalize=True,
            nrow=nrow
        )
        
        # 保存生成的中心区域大图
        vutils.save_image(
            fake_centers,
            f'result/test/fake_centers_epoch_{epoch:03d}.png',
            normalize=True,
            nrow=nrow
        )
        
        # 保存重建图像大图（最重要的结果）
        vutils.save_image(
            recon_images,
            f'result/test/reconstructed_epoch_{epoch:03d}.png',
            normalize=True,
            nrow=nrow
        )
        
        # 保存对比图：每行显示 [损坏图像 | 真实中心 | 生成中心 | 重建图像]
        # 只保存前16个样本的对比图（如果样本数少于16，则全部保存）
        num_compare = min(16, num_images)
        compare_images = []
        
        for i in range(num_compare):
            # 创建一行对比图：损坏图像 | 真实中心 | 生成中心 | 重建图像
            row = torch.cat([
                corrupted_images[i:i+1],
                real_centers[i:i+1],
                fake_centers[i:i+1],
                recon_images[i:i+1]
            ], dim=3)  # 在宽度维度拼接
            compare_images.append(row)
        
        if compare_images:
            compare_grid = torch.cat(compare_images, dim=0)
            vutils.save_image(
                compare_grid,
                f'result/test/comparison_epoch_{epoch:03d}.png',
                normalize=True,
                nrow=1  # 每行一个样本的对比
            )
        
        self._log(f'✓ 测试集图片已保存到 result/test/ (共 {num_images} 个样本)', to_console=False)

    def _save_checkpoints(self, epoch):
        """保存模型检查点"""
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.netG.state_dict()
            }, f'model/netG_epoch_{epoch}.pth')

            torch.save({
                'epoch': epoch + 1,
                'state_dict': self.netD.state_dict()
            }, f'model/netD_epoch_{epoch}.pth')

        # 保存最终模型
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.netG.state_dict()
        }, 'model/netG_context_encoder_final.pth')

        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.netD.state_dict()
        }, 'model/netD_context_encoder_final.pth')
        
        # 训练结束，保存最终日志
        if epoch == self.epochs - 1:
            self.training_log["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.json_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_log, f, indent=2, ensure_ascii=False)
            
            final_log = (f'\n{"=" * 80}\n'
                        f'训练完成！\n'
                        f'开始时间: {self.training_log["start_time"]}\n'
                        f'结束时间: {self.training_log["end_time"]}\n'
                        f'日志文件: {self.log_file}\n'
                        f'JSON日志: {self.json_log_file}\n'
                        f'{"=" * 80}\n')
            self._log(final_log)


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
