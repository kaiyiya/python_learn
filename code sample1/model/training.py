from __future__ import division
import torch
import time
import numpy as np
import os
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, train_loader, model, opt, device, val_loader=None):
        self.args = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.device = device
        # 使用BCEWithLogits以适配logits输出；改为全局pos_weight
        import torch.nn.functional as F
        self.F = F
        self.criterion = None  # 用F.binary_cross_entropy_with_logits按步构造
        self.optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
        self.epochs = 200
        self.model.to(device)
        # 全局pos_weight（基于数据集前景约3%估算，限制上限50）
        self.pos_weight = torch.tensor(32.0, device=self.device)
        self.pos_weight = torch.clamp(self.pos_weight, max=50.0)

    def calculate_iou(self, pred, target, threshold=0.5):
        """计算IoU (Intersection over Union)"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()

        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection

        if union == 0:
            return torch.tensor(1.0).to(pred.device)
        return intersection / union

    def calculate_dice(self, pred, target, threshold=0.5, smooth=1e-6):
        """计算Dice系数"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()

        intersection = (pred_binary * target_binary).sum()
        dice = (2.0 * intersection + smooth) / (pred_binary.sum() + target_binary.sum() + smooth)
        return dice

    def dice_loss(self, logits, target, smooth=1e-6):
        """Dice Loss 基于概率（对logits做sigmoid）"""
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum()
        union = probs.sum() + target.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice

    def calculate_accuracy(self, pred, target, threshold=0.5):
        """计算准确率"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()

        correct = (pred_binary == target_binary).float().sum()
        total = pred_binary.numel()
        return correct / total

    def calculate_mae(self, pred, target):
        """计算平均绝对误差 (MAE)"""
        return torch.abs(pred - target).mean()

    def get_gradient_norm(self, model):
        """计算梯度范数"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def train_model(self):
        print(f'开始训练，共 {self.epochs} 个epoch')
        print('=' * 80)

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            losses = []
            ious = []
            dices = []
            accuracies = []
            maes = []
            gradient_norms = []

            self.model.train()

            for i, (img, mask) in enumerate(self.train_loader):
                step_start_time = time.time()
                img, mask = img.to(self.device), mask.float().to(self.device)

                self.optimizer.zero_grad()
                output = self.model(img)  # logits
                
                # 使用全局pos_weight
                bce = self.F.binary_cross_entropy_with_logits(output, mask, pos_weight=self.pos_weight)
                dice = self.dice_loss(output, mask)
                loss = bce + 1.5 * dice
                loss.backward()

                # 计算梯度范数（在step之前）
                grad_norm = self.get_gradient_norm(self.model)
                gradient_norms.append(grad_norm)

                self.optimizer.step()

                # 计算各种指标
                with torch.no_grad():
                    # 检查数据是否正常
                    mask_sum = mask.sum().item()
                    logits_sum = output.sum().item()
                    probs = torch.sigmoid(output)
                    output_sum = probs.sum().item()
                    mask_nonzero = (mask > 0.01).sum().item()  # 统计非零像素
                    
                    # 使用更低阈值以缓解前景稀少（0.2）
                    thr = 0.2
                    iou = self.calculate_iou(probs, mask, threshold=thr).item()
                    dice = self.calculate_dice(probs, mask, threshold=thr).item()
                    accuracy = self.calculate_accuracy(probs, mask, threshold=thr).item()
                    mae = self.calculate_mae(probs, mask).item()

                losses.append(loss.item())
                ious.append(iou)
                dices.append(dice)
                accuracies.append(accuracy)
                maes.append(mae)

                step_time = time.time() - step_start_time
                samples_per_sec = img.size(0) / step_time if step_time > 0 else 0

                # 获取当前学习率
                current_lr = self.optimizer.param_groups[0]['lr']

                # 每10个step打印一次详细信息
                if i % 10 == 0:
                    print(f'\n[Epoch {epoch}/{self.epochs}] [Step {i}/{len(self.train_loader)}]')
                    print(f'  Loss: {loss.item():.6f}')
                    print(f'  IoU: {iou:.4f} | Dice: {dice:.4f} | Acc: {accuracy:.4f} | MAE: {mae:.6f}')
                    print(f'  LR: {current_lr:.6f} | Grad Norm: {grad_norm:.6f}')
                    print(f'  Speed: {samples_per_sec:.2f} samples/sec | Step Time: {step_time:.3f}s')
                    print(f'  Image range: [{img.min():.6f}, {img.max():.6f}]')
                    print(f'  Logits range: [{output.min():.6f}, {output.max():.6f}] | Probs range: [{probs.min():.6f}, {probs.max():.6f}] | Prob sum: {output_sum:.2f}')
                    print(f'  Mask range: [{mask.min():.6f}, {mask.max():.6f}] | Mask sum: {mask_sum:.2f} | Non-zero pixels: {mask_nonzero}')
                    
                    # 警告检查
                    if grad_norm < 1e-6:
                        print(f'  ⚠️  警告: 梯度范数接近0，模型可能已停止更新！')
                    if probs.max() < 1e-3:
                        print(f'  ⚠️  警告: 概率输出几乎全为0，可能模型塌陷！')
                    if mask_nonzero < mask.numel() * 0.01:  # 非零像素少于1%
                        print(f'  ⚠️  警告: 掩码几乎全为背景（非零像素<1%），数据可能有问题！')

            # 训练集Epoch总结
            epoch_time = time.time() - epoch_start_time
            avg_loss = np.mean(losses)
            avg_iou = np.mean(ious)
            avg_dice = np.mean(dices)
            avg_accuracy = np.mean(accuracies)
            avg_mae = np.mean(maes)
            avg_grad_norm = np.mean(gradient_norms)
            total_samples = len(self.train_loader.dataset)
            epoch_speed = total_samples / epoch_time if epoch_time > 0 else 0

            print(f'\n{"=" * 80}')
            print(f'Epoch {epoch}/{self.epochs} 总结:')
            print(f'  平均 Loss: {avg_loss:.6f}')
            print(f'  平均 IoU: {avg_iou:.4f}')
            print(f'  平均 Dice: {avg_dice:.4f}')
            print(f'  平均准确率: {avg_accuracy:.4f}')
            print(f'  平均 MAE: {avg_mae:.6f}')
            print(f'  平均梯度范数: {avg_grad_norm:.4f}')
            print(f'  当前学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'  训练时间: {epoch_time:.2f}s | 训练速度: {epoch_speed:.2f} samples/sec')
            print(f'{"=" * 80}')

            # 可视化：保存一组对比图（输入/概率/二值/GT）
            try:
                os.makedirs('PredvsGT', exist_ok=True)
                with torch.no_grad():
                    for vis_img, vis_mask in self.train_loader:
                        vis_img = vis_img.to(self.device)
                        vis_mask = vis_mask.to(self.device)
                        vis_logits = self.model(vis_img)
                        vis_probs = torch.sigmoid(vis_logits)
                        thr = 0.2
                        vis_pred = (vis_probs > thr).float()

                        # 仅保存第一个batch的第一个样本
                        img_np = vis_img[0, 0].detach().cpu().numpy()
                        prob_np = vis_probs[0, 0].detach().cpu().numpy()
                        pred_np = vis_pred[0, 0].detach().cpu().numpy()
                        mask_np = vis_mask[0, 0].detach().cpu().numpy()

                        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                        axes[0].imshow(img_np, cmap='gray'); axes[0].set_title('Input'); axes[0].axis('off')
                        im1 = axes[1].imshow(prob_np, cmap='viridis'); axes[1].set_title('Prob'); axes[1].axis('off'); plt.colorbar(im1, ax=axes[1])
                        axes[2].imshow(pred_np, cmap='gray'); axes[2].set_title(f'Pred(th={thr})'); axes[2].axis('off')
                        axes[3].imshow(mask_np, cmap='gray'); axes[3].set_title('GT'); axes[3].axis('off')
                        plt.tight_layout()
                        plt.savefig(os.path.join('PredvsGT', f'epoch_{epoch:03d}.png'), dpi=150, bbox_inches='tight')
                        plt.close()
                        break
            except Exception:
                pass

            # 验证集评估
            if self.val_loader is not None:
                self.model.eval()
                val_losses = []
                val_ious = []
                val_dices = []
                val_accuracies = []
                val_maes = []
                with torch.no_grad():
                    for img, mask in self.val_loader:
                        img, mask = img.to(self.device), mask.float().to(self.device)
                        output = self.model(img)  # logits
                        # 与训练相同的全局pos_weight
                        bce_v = self.F.binary_cross_entropy_with_logits(output, mask, pos_weight=self.pos_weight)
                        dice_v = self.dice_loss(output, mask)
                        vloss = bce_v + 1.5 * dice_v
                        val_losses.append(vloss.item())
                        probs = torch.sigmoid(output)
                        thr = 0.2
                        val_ious.append(self.calculate_iou(probs, mask, threshold=thr).item())
                        val_dices.append(self.calculate_dice(probs, mask, threshold=thr).item())
                        val_accuracies.append(self.calculate_accuracy(probs, mask, threshold=thr).item())
                        val_maes.append(self.calculate_mae(probs, mask).item())

                print(f'验证集:')
                print(f'  平均 Val Loss: {np.mean(val_losses):.6f}')
                print(f'  平均 Val IoU: {np.mean(val_ious):.4f}')
                print(f'  平均 Val Dice: {np.mean(val_dices):.4f}')
                print(f'  平均 Val Acc: {np.mean(val_accuracies):.4f}')
                print(f'  平均 Val MAE: {np.mean(val_maes):.6f}')
                print(f'{"=" * 80}\n')
            else:
                print()

            # 保存模型
            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f'unet-{epoch}.pth')
                print(f'✓ 模型已保存: unet-{epoch}.pth\n')

        torch.save(self.model.state_dict(), f'Unet-epochs{self.epochs}.pth')
        print(f'\n训练完成！最终模型已保存: Unet-epochs{self.epochs}.pth')
