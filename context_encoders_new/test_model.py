import torch
from torch.utils.data import DataLoader
from dataset import ContextEncoderDataset
from model.net import ContextEncoderGenerator, ContextEncoderDiscriminator
import os
import numpy as np
import argparse
from psnr import psnr
import torchvision.utils as vutils
import configs.config_loader as cfg_loader


class ModelTester:
    def __init__(self, generator_path, discriminator_path=None, device='cuda'):
        """
        初始化模型测试器
        
        Args:
            generator_path: 训练好的生成器模型权重路径
            discriminator_path: 训练好的判别器模型权重路径（可选）
            device: 使用的设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {self.device}')

        # 加载配置以获取模型参数
        args = cfg_loader.get_config()

        # 创建模型
        self.generator = ContextEncoderGenerator(args)
        if discriminator_path and os.path.exists(discriminator_path):
            self.discriminator = ContextEncoderDiscriminator(args)

        # 加载生成器权重
        if os.path.exists(generator_path):
            checkpoint = torch.load(generator_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['state_dict'])
            else:
                self.generator.load_state_dict(checkpoint)
            print(f'✓ 成功加载生成器模型: {generator_path}')
        else:
            print(f'⚠ 警告: 模型文件不存在: {generator_path}')
            print('将使用随机初始化的模型进行测试')

        self.generator.to(self.device)
        self.generator.eval()

        if discriminator_path and os.path.exists(discriminator_path):
            checkpoint = torch.load(discriminator_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.discriminator.load_state_dict(checkpoint['state_dict'])
            else:
                self.discriminator.load_state_dict(checkpoint)
            self.discriminator.to(self.device)
            self.discriminator.eval()
            print(f'✓ 成功加载判别器模型: {discriminator_path}')

    def calculate_psnr(self, pred, target):
        """计算PSNR"""
        # 将tensor转换为numpy数组，并调整到[0,255]范围
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        # 如果是归一化到[-1,1]，转换到[0,255]
        if pred.min() < 0:
            pred = (pred + 1) * 127.5
        if target.min() < 0:
            target = (target + 1) * 127.5

        # 确保在[0,255]范围内
        pred = np.clip(pred, 0, 255)
        target = np.clip(target, 0, 255)

        # 如果是CHW格式，转换为HWC
        if len(pred.shape) == 3 and pred.shape[0] == 3:
            pred = pred.transpose(1, 2, 0)
        if len(target.shape) == 3 and target.shape[0] == 3:
            target = target.transpose(1, 2, 0)

        return psnr(target, pred)

    def calculate_mse(self, pred, target):
        """计算均方误差 (MSE)"""
        return ((pred - target) ** 2).mean().item()

    def calculate_mae(self, pred, target):
        """计算平均绝对误差 (MAE)"""
        return torch.abs(pred - target).mean().item()

    def test_dataset(self, test_loader, save_results=False, output_dir='test_results'):
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
            save_results: 是否保存可视化结果
            output_dir: 结果保存目录
        """
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'cropped'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'real'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'recon'), exist_ok=True)
            print(f'结果将保存到: {output_dir}')

        all_metrics = {
            'mse': [],
            'mae': [],
            'psnr': []
        }

        print('\n开始测试...')
        print('=' * 80)

        with torch.no_grad():
            for batch_idx, (corrupted_images, real_centers) in enumerate(test_loader):
                corrupted_images = corrupted_images.to(self.device)
                real_centers = real_centers.to(self.device)

                # 生成修复的中心区域
                fake_centers = self.generator(corrupted_images)

                # 计算指标
                mse = self.calculate_mse(fake_centers, real_centers)
                mae = self.calculate_mae(fake_centers, real_centers)

                # 计算PSNR（逐个样本计算）
                batch_psnr = 0.0
                for i in range(fake_centers.size(0)):
                    batch_psnr += self.calculate_psnr(fake_centers[i], real_centers[i])
                batch_psnr /= fake_centers.size(0)

                # 记录指标
                all_metrics['mse'].append(mse)
                all_metrics['mae'].append(mae)
                all_metrics['psnr'].append(batch_psnr)

                # 打印批次结果
                print(f'\n[Batch {batch_idx + 1}/{len(test_loader)}]')
                print(f'  MSE: {mse:.6f} | MAE: {mae:.6f} | PSNR: {batch_psnr:.2f} dB')

                # 保存可视化结果
                if save_results:
                    self._save_batch_results(
                        corrupted_images, real_centers, fake_centers, batch_idx, output_dir
                    )

        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        std_metrics = {k: np.std(v) for k, v in all_metrics.items()}

        # 打印最终统计结果
        print(f'\n{"=" * 80}')
        print('测试结果统计:')
        print(f'  平均 MSE: {avg_metrics["mse"]:.6f} ± {std_metrics["mse"]:.6f}')
        print(f'  平均 MAE: {avg_metrics["mae"]:.6f} ± {std_metrics["mae"]:.6f}')
        print(f'  平均 PSNR: {avg_metrics["psnr"]:.2f} ± {std_metrics["psnr"]:.2f} dB')
        print(f'{"=" * 80}\n')

        return avg_metrics, std_metrics

    def _save_batch_results(self, corrupted_images, real_centers, fake_centers, batch_idx, output_dir):
        """保存批次结果"""
        # 重建完整图像
        recon_images = corrupted_images.clone()
        center_size = real_centers.size(2)
        center_start = corrupted_images.size(2) // 4

        recon_images[:, :,
        center_start:center_start + center_size,
        center_start:center_start + center_size] = fake_centers

        # 保存图像
        vutils.save_image(
            corrupted_images,
            os.path.join(output_dir, 'cropped', f'batch_{batch_idx:03d}.png'),
            normalize=True, nrow=4
        )

        vutils.save_image(
            real_centers,
            os.path.join(output_dir, 'real', f'batch_{batch_idx:03d}.png'),
            normalize=True, nrow=4
        )

        vutils.save_image(
            recon_images,
            os.path.join(output_dir, 'recon', f'batch_{batch_idx:03d}.png'),
            normalize=True, nrow=4
        )


def main():
    parser = argparse.ArgumentParser(description='测试训练好的Context Encoder模型')
    parser.add_argument('--generator_path', type=str, default='model/netG_context_encoder_final.pth',
                        help='生成器模型权重文件路径')
    parser.add_argument('--discriminator_path', type=str, default='',
                        help='判别器模型权重文件路径（可选）')
    parser.add_argument('--test_dataroot', type=str, default='data/val',
                        help='测试数据集路径')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='测试批次大小')
    parser.add_argument('--save_results', action='store_true',
                        help='是否保存可视化结果')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='结果保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='使用的设备 (cuda 或 cpu)')

    args = parser.parse_args()

    # 加载配置
    cfg = cfg_loader.get_config()
    cfg.dataroot = args.test_dataroot
    cfg.batch_size = args.batch_size

    # 创建测试数据集和数据加载器
    print(f'加载测试数据集...')
    print(f'  数据路径: {args.test_dataroot}')

    try:
        test_dataset = ContextEncoderDataset(cfg, is_train=False)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=cfg.workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f'✓ 测试数据集加载成功，共 {len(test_dataset)} 个样本')
    except Exception as e:
        print(f'✗ 加载测试数据集失败: {e}')
        import traceback
        traceback.print_exc()
        return

    # 创建测试器并测试
    tester = ModelTester(args.generator_path, args.discriminator_path, device=args.device)

    avg_metrics, std_metrics = tester.test_dataset(
        test_loader,
        save_results=args.save_results,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
