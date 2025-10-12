#!/usr/bin/env python3
"""
测试脚本 - 检查代码是否能正常运行
"""

def test_imports():
    """测试所有必要的导入"""
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch 导入失败: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ TorchVision 版本: {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ TorchVision 导入失败: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL 导入成功")
    except ImportError as e:
        print(f"✗ PIL 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy 版本: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy 导入失败: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib 导入成功")
    except ImportError as e:
        print(f"✗ Matplotlib 导入失败: {e}")
        return False
    
    try:
        import configargparse
        print("✓ ConfigArgParse 导入成功")
    except ImportError as e:
        print(f"✗ ConfigArgParse 导入失败: {e}")
        return False
    
    return True

def test_cuda():
    """测试CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA 可用，设备数量: {torch.cuda.device_count()}")
            print(f"✓ 当前设备: {torch.cuda.get_device_name(0)}")
        else:
            print("! CUDA 不可用，将使用CPU")
    except Exception as e:
        print(f"✗ CUDA 测试失败: {e}")

def test_data_structure():
    """测试数据结构"""
    import os
    
    # 检查数据目录
    data_dirs = [
        'data/train/img',
        'data/train/gt',
        'data/val/img', 
        'data/val/gt'
    ]
    
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"✓ {dir_path}: {len(files)} 个文件")
        else:
            print(f"✗ {dir_path}: 目录不存在")

def main():
    print("=== Code Sample1 环境测试 ===\n")
    
    print("1. 测试导入...")
    if not test_imports():
        print("\n❌ 导入测试失败，请安装必要的依赖包")
        print("运行: pip install -r requirements.txt")
        return
    
    print("\n2. 测试CUDA...")
    test_cuda()
    
    print("\n3. 测试数据结构...")
    test_data_structure()
    
    print("\n✅ 环境测试完成！")

if __name__ == "__main__":
    main()
