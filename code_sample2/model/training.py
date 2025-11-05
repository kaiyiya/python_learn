from __future__ import division
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import BCEWithLogitsLoss

def calculate_iou(pred, target, threshold=0.5):
    # 二值化
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    # 计算交集和并集
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection

    # 避免除零
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.item()

class Trainer(object):
    def __init__(self,train_loader,val_loader,model, opt):
        self.args = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        # self.criterion = torch.nn.functional.binary_cross_entropy
        self.criterion = BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(lr=0.00005, params=model.parameters())
        self.epochs = 200

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def plot_comparison(self, img, output, mask, epoch, batch_idx=0):

        output=torch.sigmoid(output)
        # 选择第一个样本进行可视化
        img_np = img[batch_idx].cpu().detach().permute(1, 2, 0).numpy()
        output_np = output[batch_idx, 0].cpu().detach().numpy()
        mask_np = mask[batch_idx, 0].cpu().detach().numpy()

        # 如果图像是归一化的，反归一化显示
        if img_np.min() < 0 or img_np.max() > 1:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        # 创建对比图
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # 输入图像
        axes[0].imshow(img_np)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        # 模型输出（概率图）
        im1 = axes[1].imshow(output_np, cmap='viridis')
        axes[1].set_title('Model Output (Probability)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])

        # 二值化预测
        pred_binary = (output_np > 0.4).astype(np.float32)
        axes[2].imshow(pred_binary, cmap='gray')
        axes[2].set_title('Binary Prediction')
        axes[2].axis('off')

        # 真实mask
        axes[3].imshow(mask_np, cmap='gray')
        axes[3].set_title('Ground Truth Mask')
        axes[3].axis('off')

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'PredvsGT/comparison_epoch.png', dpi=150, bbox_inches='tight')
        plt.close()

        # # 保存一个只有二值预测和真实mask的简洁对比图
        # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        #
        # axes[0].imshow(img_np)
        # axes[0].set_title('Input Image')
        # axes[0].axis('off')
        #
        # axes[1].imshow(pred_binary, cmap='gray')
        # axes[1].set_title('Prediction')
        # axes[1].axis('off')
        #
        # axes[2].imshow(mask_np, cmap='gray')
        # axes[2].set_title('Mask')
        # axes[2].axis('off')
        #
        # plt.tight_layout()
        # plt.savefig(f'PredvsGT/simple_comparison_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        # plt.close()

    def train_model(self):
        print("Starting model training ...")
        for epoch in range(self.epochs):
            losses = []
            ious = []
            for i, (img, mask) in enumerate(self.train_loader):
                img, mask = img.to(self.device), mask.float().to(self.device)

                self.optimizer.zero_grad()
                output = self.model(img)

                '''print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
                print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
                print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")'''

                loss = self.criterion(output, mask)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                iou = calculate_iou(torch.sigmoid(output), mask, threshold=0.5)
                ious.append(iou)

                if (i+1)%10==0:
                    print(f'epoch{epoch+1} step{i+1}---train_loss, ', loss.item())

                if i == 0:
                    self.plot_comparison(img, output, mask, epoch)

            avg_loss = sum(losses) / len(losses)
            avg_iou = sum(ious) / len(ious)

            if (epoch+1) % 20 == 0:
                val_iou = self.val_model()
                print(f'epoch{epoch + 1}---Train loss: {avg_loss:.4f}, '
                      f'Train IoU: {avg_iou:.4f}, Val IoU: {val_iou}')
                torch.save(self.model, f'model_pth/unet-{epoch+1}.pth.tar')
            else:
                print(f'epoch{epoch + 1}---Train loss: {avg_loss:.4f}, '
                      f'Train IoU: {avg_iou:.4f}')

        torch.save(self.model, f'model_pth/Unet-epochs{self.epochs}.pth')

    def val_model(self):
        self.model.eval()
        ious = []

        with torch.no_grad():
            for i, (img, mask) in enumerate(self.val_loader):
                img, mask = img.to(self.device), mask.float().to(self.device)

                output = self.model(img)
                output_sigmoid = torch.sigmoid(output)

                iou = calculate_iou(output_sigmoid, mask, threshold=0.5)
                ious.append(iou)

        avg_iou = np.mean(ious)
        self.model.train()
        return avg_iou