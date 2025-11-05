import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset import *
import matplotlib.pyplot as plt


# U-Net架构 - 医学图像分割的黄金标准
class MedicalUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(MedicalUNet, self).__init__()

        # 编码器 (下采样路径)
        self.encoder1 = self._block(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features[0], features[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features[1], features[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 瓶颈层
        self.bottleneck = self._block(features[2], features[3])

        # 解码器 (上采样路径)
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = self._block(features[2] * 2, features[2])
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = self._block(features[1] * 2, features[1])
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = self._block(features[0] * 2, features[0])

        # 输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 编码路径
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        # 瓶颈
        bottleneck = self.bottleneck(self.pool3(enc3))

        # 解码路径 + 跳跃连接
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))


# 针对医学图像的复合损失函数
class MedicalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.1, delta=0.1, pos_weight=2.0):
        super(MedicalLoss, self).__init__()
        self.alpha = alpha  # Dice损失权重
        self.beta = beta  # BCE损失权重
        self.gamma = gamma  # 边界损失权重
        self.delta = delta
        self.pos_weight = pos_weight

    def dice_loss(self, pred, target):
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        return 1.0 - (2.0 * intersection + smooth) / (union + smooth)

    def boundary_loss(self, pred, target):
        # 计算梯度，关注边界区域
        pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        boundary_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        boundary_loss_y = F.mse_loss(pred_grad_y, target_grad_y)

        return (boundary_loss_x + boundary_loss_y) / 2


    def area_loss(self, pred, target):
        """计算预测区域与真实区域的面积差异（用L1损失）"""
        # 计算真实掩码的面积（每个样本的目标像素总和）
        target_area = target.sum(dim=(1, 2, 3), keepdim=True)
        # 计算预测概率图的“面积”（概率和，近似二值化后的像素和）
        pred_area = pred.sum(dim=(1, 2, 3), keepdim=True)
        # 用L1损失约束面积差异（也可尝试MSE损失）
        return F.l1_loss(pred_area, target_area)

    def forward(self, pred, target):
        pos_weight = torch.tensor(([self.pos_weight]))
        bce_loss = F.binary_cross_entropy(pred, target,pos_weight)
        dice_loss = self.dice_loss(pred, target)
        bound_loss = self.boundary_loss(pred, target)
        area_loss = self.area_loss(pred, target)

        return (self.alpha * dice_loss +
                self.beta * bce_loss +
                self.gamma * bound_loss+
                self.delta * area_loss
                )


# 训练函数
def train_medical_model(model, dataloader, criterion, optimizer, num_epochs=50):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (images, masks) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
            print(f"Mask range: [{masks.min():.3f}, {masks.max():.3f}]")
            print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.6f}')

            if batch_idx == 0:
                    plot_comparison(images, outputs, masks, epoch)
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}] completed, Average Loss: {avg_loss:.6f}')

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'model_pth/medical_unet_epoch_{epoch + 1}.pth')

    return model

def plot_comparison(img, output, mask, epoch, batch_idx=0):

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
    plt.show()
    plt.savefig(f'PredvsGT/comparison_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    # plt.close()

    # 另外保存一个只有二值预测和真实mask的简洁对比图
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


# 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in dataloader:
            outputs = model(images)
            predictions = (outputs > 0.5).float()

            # 计算Dice系数
            smooth = 1.0
            intersection = (predictions * masks).sum()
            union = predictions.sum() + masks.sum()
            dice = (2.0 * intersection + smooth) / (union + smooth)

            total_dice += dice.item() * images.size(0)
            total_samples += images.size(0)

    return total_dice / total_samples



# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MyDataset('data/val/img', 'data/val/gt')
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    # 初始化模型
    model = MedicalUNet().to(device)
    # model.load_state_dict(torch.load('model_pth/medical_unet_epoch_50.pth'))
    criterion = MedicalLoss(alpha=0.7, beta=0.2, gamma=0.1, delta=0)  # 调整权重
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)

    # 训练
    print("Starting medical lesion segmentation training...")
    trained_model = train_medical_model(model, dataloader, criterion, optimizer, num_epochs=50)


