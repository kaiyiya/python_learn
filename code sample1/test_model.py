import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import MyDataset
from model.net import UNet
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse


class ModelTester:
    def __init__(self, model_path, device='cuda'):
        """
        初始化模型测试器
        
        Args:
            model_path: 训练好的模型权重路径
            device: 使用的设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {self.device}')
        
        # 加载模型
        self.model = UNet(in_channels=1, out_channels=1)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f'✓ 成功加载模型: {model_path}')
        else:
            print(f'⚠ 警告: 模型文件不存在: {model_path}')
            print('将使用随机初始化的模型进行测试')
        
        self.model.to(self.device)
        self.model.eval()
    
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
    
    def calculate_accuracy(self, pred, target, threshold=0.5):
        """计算准确率"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()
        
        correct = (pred_binary == target_binary).float().sum()
        total = pred_binary.numel()
        return correct / total
    
    def calculate_precision(self, pred, target, threshold=0.5, smooth=1e-6):
        """计算精确率"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()
        
        tp = (pred_binary * target_binary).sum()
        fp = (pred_binary * (1 - target_binary)).sum()
        
        precision = (tp + smooth) / (tp + fp + smooth)
        return precision
    
    def calculate_recall(self, pred, target, threshold=0.5, smooth=1e-6):
        """计算召回率"""
        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()
        
        tp = (pred_binary * target_binary).sum()
        fn = ((1 - pred_binary) * target_binary).sum()
        
        recall = (tp + smooth) / (tp + fn + smooth)
        return recall
    
    def calculate_f1_score(self, pred, target, threshold=0.5):
        """计算F1分数"""
        precision = self.calculate_precision(pred, target, threshold)
        recall = self.calculate_recall(pred, target, threshold)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        return f1
    
    def calculate_mae(self, pred, target):
        """计算平均绝对误差 (MAE)"""
        return torch.abs(pred - target).mean()
    
    def calculate_mse(self, pred, target):
        """计算均方误差 (MSE)"""
        return ((pred - target) ** 2).mean()
    
    def test_dataset(self, test_loader, threshold=0.5, save_results=False, output_dir='test_results'):
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
            threshold: 二值化阈值
            save_results: 是否保存可视化结果
            output_dir: 结果保存目录
        """
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            print(f'结果将保存到: {output_dir}')
        
        all_metrics = {
            'loss': [],
            'iou': [],
            'dice': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'mae': [],
            'mse': []
        }
        
        print('\n开始测试...')
        print('=' * 80)
        
        with torch.no_grad():
            for batch_idx, (img, mask) in enumerate(test_loader):
                img = img.to(self.device)
                mask = mask.float().to(self.device)
                
                # 前向传播
                output = self.model(img)
                
                # 计算损失
                loss = F.binary_cross_entropy(output, mask)
                
                # 计算各种指标
                iou = self.calculate_iou(output, mask, threshold).item()
                dice = self.calculate_dice(output, mask, threshold).item()
                accuracy = self.calculate_accuracy(output, mask, threshold).item()
                precision = self.calculate_precision(output, mask, threshold).item()
                recall = self.calculate_recall(output, mask, threshold).item()
                f1 = self.calculate_f1_score(output, mask, threshold).item()
                mae = self.calculate_mae(output, mask).item()
                mse = self.calculate_mse(output, mask).item()
                
                # 记录指标
                all_metrics['loss'].append(loss.item())
                all_metrics['iou'].append(iou)
                all_metrics['dice'].append(dice)
                all_metrics['accuracy'].append(accuracy)
                all_metrics['precision'].append(precision)
                all_metrics['recall'].append(recall)
                all_metrics['f1'].append(f1)
                all_metrics['mae'].append(mae)
                all_metrics['mse'].append(mse)
                
                # 打印批次结果
                print(f'\n[Batch {batch_idx + 1}/{len(test_loader)}]')
                print(f'  Loss: {loss.item():.6f}')
                print(f'  IoU: {iou:.4f} | Dice: {dice:.4f} | Acc: {accuracy:.4f}')
                print(f'  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')
                print(f'  MAE: {mae:.6f} | MSE: {mse:.6f}')
                
                # 保存可视化结果
                if save_results and batch_idx < 5:  # 只保存前5个批次的可视化结果
                    self.visualize_results(img, mask, output, batch_idx, output_dir, threshold)
        
        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        std_metrics = {k: np.std(v) for k, v in all_metrics.items()}
        
        # 打印最终统计结果
        print(f'\n{"="*80}')
        print('测试结果统计:')
        print(f'  平均 Loss: {avg_metrics["loss"]:.6f} ± {std_metrics["loss"]:.6f}')
        print(f'  平均 IoU: {avg_metrics["iou"]:.4f} ± {std_metrics["iou"]:.4f}')
        print(f'  平均 Dice: {avg_metrics["dice"]:.4f} ± {std_metrics["dice"]:.4f}')
        print(f'  平均准确率: {avg_metrics["accuracy"]:.4f} ± {std_metrics["accuracy"]:.4f}')
        print(f'  平均精确率: {avg_metrics["precision"]:.4f} ± {std_metrics["precision"]:.4f}')
        print(f'  平均召回率: {avg_metrics["recall"]:.4f} ± {std_metrics["recall"]:.4f}')
        print(f'  平均 F1分数: {avg_metrics["f1"]:.4f} ± {std_metrics["f1"]:.4f}')
        print(f'  平均 MAE: {avg_metrics["mae"]:.6f} ± {std_metrics["mae"]:.6f}')
        print(f'  平均 MSE: {avg_metrics["mse"]:.6f} ± {std_metrics["mse"]:.6f}')
        print(f'{"="*80}\n')
        
        return avg_metrics, std_metrics
    
    def visualize_results(self, img, mask, output, batch_idx, output_dir, threshold=0.5):
        """可视化测试结果"""
        # 转换回numpy格式用于可视化
        img_np = img[0, 0].cpu().numpy()
        mask_np = mask[0, 0].cpu().numpy()
        output_np = output[0, 0].cpu().numpy()
        pred_binary = (output_np > threshold).astype(np.float32)
        
        # 创建可视化图像
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(output_np, cmap='gray')
        axes[2].set_title('Prediction (Probability)')
        axes[2].axis('off')
        
        axes[3].imshow(pred_binary, cmap='gray')
        axes[3].set_title(f'Prediction (Binary, threshold={threshold})')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'batch_{batch_idx}_result.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'  ✓ 可视化结果已保存: batch_{batch_idx}_result.png')


def main():
    parser = argparse.ArgumentParser(description='测试训练好的UNet模型')
    parser.add_argument('--model_path', type=str, default='Unet-epochs200.pth',
                        help='模型权重文件路径')
    parser.add_argument('--test_img_dir', type=str, default='data/val/img',
                        help='测试图像目录')
    parser.add_argument('--test_mask_dir', type=str, default='data/val/gt',
                        help='测试掩码目录')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='测试批次大小')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值')
    parser.add_argument('--save_results', action='store_true',
                        help='是否保存可视化结果')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='使用的设备 (cuda 或 cpu)')
    
    args = parser.parse_args()
    
    # 创建测试数据集和数据加载器
    print(f'加载测试数据集...')
    print(f'  图像目录: {args.test_img_dir}')
    print(f'  掩码目录: {args.test_mask_dir}')
    
    try:
        test_dataset = MyDataset(args.test_img_dir, args.test_mask_dir)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        print(f'✓ 测试数据集加载成功，共 {len(test_dataset)} 个样本')
    except Exception as e:
        print(f'✗ 加载测试数据集失败: {e}')
        return
    
    # 创建测试器并测试
    tester = ModelTester(args.model_path, device=args.device)
    
    avg_metrics, std_metrics = tester.test_dataset(
        test_loader,
        threshold=args.threshold,
        save_results=args.save_results,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
