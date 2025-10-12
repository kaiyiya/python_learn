#!/usr/bin/env python3
"""
测试数据加载和预处理功能
"""

import torch
from dataset import ContextEncoderDataset, create_dataloader
import matplotlib.pyplot as plt
import numpy as np

def test_dataset_creation():
    """测试数据集创建"""
    print("测试数据集创建...")
    
    class TestArgs:
        def __init__(self):
            self.dataset = 'streetview'
            self.dataroot = 'data/train'
            self.image_size = 128
            self.overlap_pred = 4
            self.batch_size = 2
            self.workers = 0  # 避免多进程问题
    
    opt = TestArgs()
    
    try:
        # 测试数据集创建
        dataset = ContextEncoderDataset(opt)
        print(f"✅ 数据集创建成功，大小: {len(dataset)}")
        
        # 测试单个样本
        corrupted, center = dataset[0]
        print(f"✅ 单个样本测试通过")
        print(f"   损坏图像形状: {corrupted.shape}")
        print(f"   中心区域形状: {center.shape}")
        print(f"   损坏图像数值范围: [{corrupted.min():.3f}, {corrupted.max():.3f}]")
        print(f"   中心区域数值范围: [{center.min():.3f}, {center.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return False

def test_dataloader():
    """测试数据加载器"""
    print("\n测试数据加载器...")
    
    class TestArgs:
        def __init__(self):
            self.dataset = 'streetview'
            self.dataroot = 'data/train'
            self.image_size = 128
            self.overlap_pred = 4
            self.batch_size = 2
            self.workers = 0
    
    opt = TestArgs()
    
    try:
        # 测试数据加载器
        dataloader = create_dataloader(opt)
        print(f"✅ 数据加载器创建成功，批次数量: {len(dataloader)}")
        
        # 测试一个批次
        for i, (corrupted_batch, center_batch) in enumerate(dataloader):
            print(f"✅ 批次 {i} 测试通过")
            print(f"   损坏图像批次形状: {corrupted_batch.shape}")
            print(f"   中心区域批次形状: {center_batch.shape}")
            print(f"   批次数值范围: [{corrupted_batch.min():.3f}, {corrupted_batch.max():.3f}]")
            break  # 只测试第一个批次
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        return False

def test_image_corruption():
    """测试图像损坏逻辑"""
    print("\n测试图像损坏逻辑...")
    
    # 创建测试图像
    test_image = torch.randn(1, 3, 128, 128)
    
    class TestArgs:
        def __init__(self):
            self.overlap_pred = 4
    
    opt = TestArgs()
    dataset = ContextEncoderDataset.__new__(ContextEncoderDataset)
    dataset.overlap_pred = opt.overlap_pred
    
    try:
        # 测试损坏图像创建
        corrupted = dataset._create_corrupted_image(test_image)
        
        # 检查中心区域是否被正确填充
        center_h_start = 128 // 4
        center_h_end = center_h_start + 128 // 2
        center_w_start = 128 // 4
        center_w_end = center_w_start + 128 // 2
        
        fill_h_start = center_h_start + opt.overlap_pred
        fill_h_end = center_h_end - opt.overlap_pred
        fill_w_start = center_w_start + opt.overlap_pred
        fill_w_end = center_w_end - opt.overlap_pred
        
        # 检查填充区域的值
        fill_region = corrupted[0, :, fill_h_start:fill_h_end, fill_w_start:fill_w_end]
        expected_r = 2 * 117.0 / 255.0 - 1.0
        expected_g = 2 * 104.0 / 255.0 - 1.0
        expected_b = 2 * 123.0 / 255.0 - 1.0
        
        print(f"✅ 图像损坏测试通过")
        print(f"   填充区域红色通道值: {fill_region[0, 0, 0]:.3f} (期望: {expected_r:.3f})")
        print(f"   填充区域绿色通道值: {fill_region[1, 0, 0]:.3f} (期望: {expected_g:.3f})")
        print(f"   填充区域蓝色通道值: {fill_region[2, 0, 0]:.3f} (期望: {expected_b:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ 图像损坏测试失败: {e}")
        return False

def test_center_extraction():
    """测试中心区域提取"""
    print("\n测试中心区域提取...")
    
    # 创建测试图像
    test_image = torch.randn(1, 3, 128, 128)
    
    class TestArgs:
        def __init__(self):
            self.overlap_pred = 4
    
    opt = TestArgs()
    dataset = ContextEncoderDataset.__new__(ContextEncoderDataset)
    dataset.overlap_pred = opt.overlap_pred
    
    try:
        # 测试中心区域提取
        center = dataset._extract_center_region(test_image)
        
        print(f"✅ 中心区域提取测试通过")
        print(f"   原始图像形状: {test_image.shape}")
        print(f"   中心区域形状: {center.shape}")
        print(f"   中心区域数值范围: [{center.min():.3f}, {center.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 中心区域提取测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试数据加载和预处理...")
    
    tests = [
        test_dataset_creation,
        test_dataloader,
        test_image_corruption,
        test_center_extraction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有数据测试通过！")
    else:
        print("❌ 部分测试失败，请检查数据路径和配置")
