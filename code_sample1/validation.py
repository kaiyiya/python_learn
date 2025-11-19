import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import DataLoader
from dataset import MyDataset
from model.net import UNet


def comprehensive_validation(model_path, val_img_dir, val_gt_dir, output_dir='./'):
    # 设置设
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = torch.load(model_path, map_location=device)
    model.eval()
    print(f"Loaded model from {model_path}")

    # 创建验证数据集和数据加载器
    val_dataset = MyDataset(val_img_dir, val_gt_dir,augment=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 存储所有结果
    all_images = []
    all_predictions = []
    all_masks = []
    all_ious = []

    print("Starting comprehensive validation...")

    with torch.no_grad():
        for i, (img, mask) in enumerate(val_loader):
            img, mask = img.to(device), mask.float().to(device)

            output = model(img)
            output_sigmoid = torch.sigmoid(output)

            iou = calculate_iou(output_sigmoid, mask, threshold=0.5)
            all_ious.append(iou)

            img_np = img[0].cpu().detach().permute(1, 2, 0).numpy()
            pred_np = (output_sigmoid[0, 0].cpu().detach().numpy() > 0.5).astype(np.float32)
            mask_np = mask[0, 0].cpu().detach().numpy()

            if img_np.min() < 0 or img_np.max() > 1:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

            all_images.append(img_np)
            all_predictions.append(pred_np)
            all_masks.append(mask_np)

            print(f"Processed image {i + 1}/{len(val_loader)} - IoU: {iou:.4f}")

    # 计算平均IoU
    avg_iou = np.mean(all_ious)
    print(f"\nAverage IoU: {avg_iou:.4f}")

    # 绘制综合对比图
    plot_comprehensive_comparison(all_images, all_predictions, all_masks, all_ious,
                                  output_dir, avg_iou)

    return avg_iou


def calculate_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def plot_comprehensive_comparison(images, predictions, masks, ious, output_dir, avg_iou):

    n_images = len(images)

    n_cols = 9
    n_rows = (n_images + 2) // 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 3 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_images):
        row = i // 3
        col_start = (i % 3) * 3

        # 原图
        axes[row, col_start].imshow(images[i])
        axes[row, col_start].set_title(f'Sample {i + 1}\nInput', fontsize=8)
        axes[row, col_start].axis('off')

        # 预测二值图
        axes[row, col_start + 1].imshow(predictions[i], cmap='gray')
        axes[row, col_start + 1].set_title(f'Prediction\nIoU: {ious[i]:.3f}', fontsize=8)
        axes[row, col_start + 1].axis('off')

        # 真实掩码
        axes[row, col_start + 2].imshow(masks[i], cmap='gray')
        axes[row, col_start + 2].set_title('Ground Truth', fontsize=8)
        axes[row, col_start + 2].axis('off')

    # 隐藏多余的子图
    for i in range(n_images * 3, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    # 添加总标题
    plt.suptitle(f'Comprehensive Validation Results (Average IoU: {avg_iou:.4f})',
                 fontsize=16, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # 保存图片
    output_path = os.path.join(output_dir, 'comprehensive_validation_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comprehensive validation results saved to: {output_path}")


def quick_validation(model_path, val_img_dir, val_gt_dir):
    """
    快速验证函数：只计算指标，不生成可视化
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = torch.load(model_path, map_location=device)
    model.eval()

    # 创建验证数据集和数据加载器
    val_dataset = MyDataset(val_img_dir, val_gt_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    ious = []

    with torch.no_grad():
        for i, (img, mask) in enumerate(val_loader):
            img, mask = img.to(device), mask.float().to(device)

            output = model(img)
            output_sigmoid = torch.sigmoid(output)

            iou = calculate_iou(output_sigmoid, mask, threshold=0.5)
            ious.append(iou)

    avg_iou = np.mean(ious)
    std_iou = np.std(ious)

    print(f"Quick Validation Results:")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Std IoU: {std_iou:.4f}")
    print(f"Min IoU: {min(ious):.4f}")
    print(f"Max IoU: {max(ious):.4f}")

    return avg_iou


if __name__ == '__main__':
    # 使用示例
    model_path = 'Unet-epochs200.pth'  # 修改为你的模型路径
    val_img_dir = 'data/val/img'
    val_gt_dir = 'data/val/gt'

    # 运行综合验证（生成大图）
    avg_iou = comprehensive_validation(model_path, val_img_dir, val_gt_dir)

    # 或者运行快速验证（只计算指标）
    # avg_iou = quick_validation(model_path, val_img_dir, val_gt_dir)
