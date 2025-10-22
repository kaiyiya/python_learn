#!/usr/bin/env python3
"""
测试训练步骤和损失函数
"""

import torch
import torch.nn as nn
from model.net import ContextEncoderGenerator, ContextEncoderDiscriminator, weights_init

def test_loss_functions():
    """测试损失函数计算"""
    print("测试损失函数计算...")
    
    # 创建测试数据
    batch_size = 4
    real_centers = torch.randn(batch_size, 3, 64, 64)
    fake_centers = torch.randn(batch_size, 3, 64, 64)
    
    # 创建标签
    real_label = torch.full((batch_size, 1), 1.0)
    fake_label = torch.full((batch_size, 1), 0.0)
    
    # 定义损失函数
    criterion = nn.BCELoss()
    criterionMSE = nn.MSELoss()
    
    try:
        # 测试BCE损失
        bce_loss = criterion(torch.sigmoid(torch.randn(batch_size, 1)), real_label)
        print(f"✅ BCE损失计算: {bce_loss.item():.4f}")
        
        # 测试MSE损失
        mse_loss = criterionMSE(fake_centers, real_centers)
        print(f"✅ MSE损失计算: {mse_loss.item():.4f}")
        
        # 测试L2损失（带权重矩阵）
        wtl2 = 0.998
        overlapL2Weight = 10
        overlap_pred = 4
        
        wtl2Matrix = real_centers.clone()
        wtl2Matrix.data.fill_(wtl2 * overlapL2Weight)
        center_size = 64
        wtl2Matrix.data[:, :, 
                       overlap_pred:center_size - overlap_pred,
                       overlap_pred:center_size - overlap_pred] = wtl2
        
        l2_loss = (fake_centers - real_centers).pow(2)
        l2_loss = l2_loss * wtl2Matrix
        l2_loss = l2_loss.mean()
        
        print(f"✅ 加权L2损失计算: {l2_loss.item():.4f}")
        print(f"   权重矩阵形状: {wtl2Matrix.shape}")
        print(f"   权重矩阵数值范围: [{wtl2Matrix.min():.3f}, {wtl2Matrix.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")
        return False

def test_training_step():
    """测试完整的训练步骤"""
    print("\n测试完整训练步骤...")
    
    class TestArgs:
        def __init__(self):
            self.ngpu = 1
            self.nc = 3
            self.nef = 64
            self.ngf = 64
            self.ndf = 64
            self.n_bottleneck = 4000
            self.image_size = 128
            self.overlap_pred = 4
            self.wtl2 = 0.998
            self.overlapL2Weight = 10
    
    opt = TestArgs()
    
    # 创建模型
    generator = ContextEncoderGenerator(opt)
    discriminator = ContextEncoderDiscriminator(opt)
    
    # 应用权重初始化
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # 创建优化器
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 创建测试数据
    batch_size = 4
    corrupted_images = torch.randn(batch_size, 3, 128, 128)
    real_centers = torch.randn(batch_size, 3, 64, 64)
    
    # 创建标签
    real_label = torch.full((batch_size, 1), 1.0)
    fake_label = torch.full((batch_size, 1), 0.0)
    
    # 定义损失函数
    criterion = nn.BCELoss()
    
    try:
        # 判别器训练步骤
        print("  测试判别器训练步骤...")
        discriminator.zero_grad()
        
        # 真实图像
        output_real = discriminator(real_centers)
        errD_real = criterion(output_real, real_label)
        errD_real.backward()
        D_x = output_real.mean().item()
        
        # 生成图像
        fake_centers = generator(corrupted_images)
        output_fake = discriminator(fake_centers.detach())
        errD_fake = criterion(output_fake, fake_label)
        errD_fake.backward()
        D_G_z1 = output_fake.mean().item()
        
        errD = errD_real + errD_fake
        optimizerD.step()
        
        print(f"  ✅ 判别器损失: {errD.item():.4f}")
        print(f"     D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}")
        
        # 生成器训练步骤
        print("  测试生成器训练步骤...")
        generator.zero_grad()
        
        output_fake = discriminator(fake_centers)
        errG_D = criterion(output_fake, real_label)
        
        # L2损失
        wtl2Matrix = real_centers.clone()
        wtl2Matrix.data.fill_(opt.wtl2 * opt.overlapL2Weight)
        wtl2Matrix.data[:, :, opt.overlap_pred:opt.image_size//2 - opt.overlap_pred,
                       opt.overlap_pred:opt.image_size//2 - opt.overlap_pred] = opt.wtl2
        
        errG_l2 = (fake_centers - real_centers).pow(2)
        errG_l2 = errG_l2 * wtl2Matrix
        errG_l2 = errG_l2.mean()
        
        errG = (1 - opt.wtl2) * errG_D + opt.wtl2 * errG_l2
        errG.backward()
        
        D_G_z2 = output_fake.mean().item()
        optimizerG.step()
        
        print(f"  ✅ 生成器对抗损失: {errG_D.item():.4f}")
        print(f"  ✅ 生成器L2损失: {errG_l2.item():.4f}")
        print(f"  ✅ 生成器总损失: {errG.item():.4f}")
        print(f"     D(G(z)): {D_G_z2:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradient_flow():
    """测试梯度流"""
    print("\n测试梯度流...")
    
    class TestArgs:
        def __init__(self):
            self.ngpu = 1
            self.nc = 3
            self.nef = 64
            self.ngf = 64
            self.ndf = 64
            self.n_bottleneck = 4000
            self.image_size = 128
            self.overlap_pred = 4
    
    opt = TestArgs()
    
    # 创建模型
    generator = ContextEncoderGenerator(opt)
    discriminator = ContextEncoderDiscriminator(opt)
    
    # 创建测试数据
    corrupted_images = torch.randn(2, 3, 128, 128, requires_grad=True)
    real_centers = torch.randn(2, 3, 64, 64)
    
    try:
        # 测试生成器梯度
        fake_centers = generator(corrupted_images)
        loss = fake_centers.mean()
        loss.backward()
        
        # 检查梯度
        has_grad = corrupted_images.grad is not None
        grad_norm = corrupted_images.grad.norm().item() if has_grad else 0
        
        print(f"✅ 生成器梯度测试通过")
        print(f"   输入梯度存在: {has_grad}")
        print(f"   梯度范数: {grad_norm:.6f}")
        
        # 测试判别器梯度
        discriminator.zero_grad()
        output = discriminator(real_centers)
        loss = output.mean()
        loss.backward()
        
        # 检查判别器参数梯度
        total_grad_norm = 0
        param_count = 0
        for param in discriminator.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
                param_count += 1
        
        print(f"✅ 判别器梯度测试通过")
        print(f"   有梯度的参数数量: {param_count}")
        print(f"   总梯度范数: {total_grad_norm:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 梯度流测试失败: {e}")
        return False

def test_memory_usage():
    """测试内存使用情况"""
    print("\n测试内存使用情况...")
    
    import psutil
    import os
    
    try:
        # 获取当前进程
        process = psutil.Process(os.getpid())
        
        # 记录初始内存
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"初始内存使用: {initial_memory:.2f} MB")
        
        # 创建模型
        class TestArgs:
            def __init__(self):
                self.ngpu = 1
                self.nc = 3
                self.nef = 64
                self.ngf = 64
                self.ndf = 64
                self.n_bottleneck = 4000
                self.image_size = 128
                self.overlap_pred = 4
        
        opt = TestArgs()
        generator = ContextEncoderGenerator(opt)
        discriminator = ContextEncoderDiscriminator(opt)
        
        # 记录模型创建后内存
        model_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"模型创建后内存: {model_memory:.2f} MB")
        print(f"模型内存占用: {model_memory - initial_memory:.2f} MB")
        
        # 测试前向传播内存
        corrupted_images = torch.randn(4, 3, 128, 128)
        real_centers = torch.randn(4, 3, 64, 64)
        
        with torch.no_grad():
            fake_centers = generator(corrupted_images)
            d_real = discriminator(real_centers)
            d_fake = discriminator(fake_centers)
        
        forward_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"前向传播后内存: {forward_memory:.2f} MB")
        print(f"前向传播内存增加: {forward_memory - model_memory:.2f} MB")
        
        # 计算参数数量
        gen_params = sum(p.numel() for p in generator.parameters())
        disc_params = sum(p.numel() for p in discriminator.parameters())
        total_params = gen_params + disc_params
        
        print(f"✅ 内存使用测试通过")
        print(f"   生成器参数: {gen_params:,}")
        print(f"   判别器参数: {disc_params:,}")
        print(f"   总参数数量: {total_params:,}")
        print(f"   参数内存估算: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        
        return True
        
    except Exception as e:
        print(f"❌ 内存使用测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试训练相关功能...")
    
    tests = [
        test_loss_functions,
        test_training_step,
        test_gradient_flow,
        test_memory_usage
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有训练测试通过！")
    else:
        print("❌ 部分测试失败，请检查实现")
