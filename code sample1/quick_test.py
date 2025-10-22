#!/usr/bin/env python3
"""
快速测试脚本 - 使用少量数据快速验证代码是否能运行
"""

import torch
from torch.utils.data import DataLoader
from dataset import MyDataset
from model.net import UNet
from model.training import Trainer
import configs.config_loader as cfg_loader

def quick_test():
    """快速测试训练流程"""
    print("=== 快速测试开始 ===")
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    try:
        # 获取配置
        args = cfg_loader.get_config()
        print(f"批次大小: {args.batch_size}")
        
        # 创建数据集
        print("创建数据集...")
        dataset = MyDataset('data/train/img', 'data/train/gt')
        print(f"数据集大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = DataLoader(dataset=dataset, batch_size=min(args.batch_size, 2), shuffle=True)
        
        # 创建模型
        print("创建模型...")
        model = UNet(in_channels=1, out_channels=1)
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 测试一个批次
        print("测试数据加载...")
        for i, (img, mask) in enumerate(dataloader):
            print(f"批次 {i}: 图像形状 {img.shape}, 掩码形状 {mask.shape}")
            if i >= 2:  # 只测试前3个批次
                break
        
        # 创建训练器（使用较少的epoch进行测试）
        print("创建训练器...")
        trainer = Trainer(dataloader, model, args, device)
        trainer.epochs = 2  # 只训练2个epoch进行测试
        
        # 开始训练
        print("开始快速训练测试...")
        trainer.train_model()
        
        print("✅ 快速测试完成！代码可以正常运行。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    quick_test()
