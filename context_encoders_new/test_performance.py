#!/usr/bin/env python3
"""
性能基准测试
"""

import torch
import time
from model.net import ContextEncoderGenerator, ContextEncoderDiscriminator, weights_init

def benchmark_forward_pass():
    """测试前向传播性能"""
    print("测试前向传播性能...")
    
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
    
    # 应用权重初始化
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # 测试不同批次大小
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        print(f"\n测试批次大小: {batch_size}")
        
        # 创建测试数据
        corrupted_images = torch.randn(batch_size, 3, 128, 128)
        real_centers = torch.randn(batch_size, 3, 64, 64)
        
        # 预热
        with torch.no_grad():
            for _ in range(5):
                fake_centers = generator(corrupted_images)
                d_real = discriminator(real_centers)
                d_fake = discriminator(fake_centers)
        
        # 测试生成器性能
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                fake_centers = generator(corrupted_images)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        gen_time = time.time() - start_time
        
        # 测试判别器性能
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                d_real = discriminator(real_centers)
                d_fake = discriminator(fake_centers)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        disc_time = time.time() - start_time
        
        print(f"  生成器: {gen_time/100*1000:.2f} ms/batch")
        print(f"  判别器: {disc_time/100*1000:.2f} ms/batch")
        print(f"  总时间: {(gen_time+disc_time)/100*1000:.2f} ms/batch")

def benchmark_training_step():
    """测试训练步骤性能"""
    print("\n测试训练步骤性能...")
    
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
    
    # 预热
    for _ in range(5):
        # 判别器训练
        discriminator.zero_grad()
        fake_centers = generator(corrupted_images)
        output_real = discriminator(real_centers)
        errD_real = criterion(output_real, real_label)
        errD_real.backward()
        output_fake = discriminator(fake_centers.detach())
        errD_fake = criterion(output_fake, fake_label)
        errD_fake.backward()
        optimizerD.step()
        
        # 生成器训练
        generator.zero_grad()
        output_fake = discriminator(fake_centers)
        errG_D = criterion(output_fake, real_label)
        errG_D.backward()
        optimizerG.step()
    
    # 测试训练步骤性能
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(50):
        # 判别器训练
        discriminator.zero_grad()
        fake_centers = generator(corrupted_images)
        output_real = discriminator(real_centers)
        errD_real = criterion(output_real, real_label)
        errD_real.backward()
        output_fake = discriminator(fake_centers.detach())
        errD_fake = criterion(output_fake, fake_label)
        errD_fake.backward()
        optimizerD.step()
        
        # 生成器训练
        generator.zero_grad()
        output_fake = discriminator(fake_centers)
        errG_D = criterion(output_fake, real_label)
        errG_D.backward()
        optimizerG.step()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    training_time = time.time() - start_time
    
    print(f"训练步骤性能: {training_time/50*1000:.2f} ms/step")
    print(f"理论训练速度: {50/training_time:.2f} steps/second")

def benchmark_memory_efficiency():
    """测试内存效率"""
    print("\n测试内存效率...")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
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
    
    # 测试不同批次大小的内存使用
    batch_sizes = [1, 2, 4, 8, 16]
    
    for batch_size in batch_sizes:
        # 清理内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建模型
        generator = ContextEncoderGenerator(opt)
        discriminator = ContextEncoderDiscriminator(opt)
        
        # 创建数据
        corrupted_images = torch.randn(batch_size, 3, 128, 128)
        real_centers = torch.randn(batch_size, 3, 64, 64)
        
        # 前向传播
        with torch.no_grad():
            fake_centers = generator(corrupted_images)
            d_real = discriminator(real_centers)
            d_fake = discriminator(fake_centers)
        
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = current_memory - initial_memory
        
        print(f"批次大小 {batch_size:2d}: {memory_usage:6.2f} MB")
        
        # 清理
        del generator, discriminator, corrupted_images, real_centers, fake_centers, d_real, d_fake
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def benchmark_model_size():
    """测试模型大小"""
    print("\n测试模型大小...")
    
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
    
    # 计算参数数量
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    total_params = gen_params + disc_params
    
    # 计算模型大小（假设float32）
    gen_size_mb = gen_params * 4 / 1024 / 1024
    disc_size_mb = disc_params * 4 / 1024 / 1024
    total_size_mb = total_params * 4 / 1024 / 1024
    
    print(f"生成器参数: {gen_params:,} ({gen_size_mb:.2f} MB)")
    print(f"判别器参数: {disc_params:,} ({disc_size_mb:.2f} MB)")
    print(f"总参数数量: {total_params:,} ({total_size_mb:.2f} MB)")
    
    # 测试不同配置的模型大小
    print("\n不同配置的模型大小对比:")
    
    configs = [
        (32, 32, 32, 2000, "小模型"),
        (64, 64, 64, 4000, "标准模型"),
        (128, 128, 128, 8000, "大模型")
    ]
    
    for nef, ngf, ndf, n_bottleneck, name in configs:
        class ConfigArgs:
            def __init__(self):
                self.ngpu = 1
                self.nc = 3
                self.nef = nef
                self.ngf = ngf
                self.ndf = ndf
                self.n_bottleneck = n_bottleneck
                self.image_size = 128
                self.overlap_pred = 4
        
        config_opt = ConfigArgs()
        
        gen = ContextEncoderGenerator(config_opt)
        disc = ContextEncoderDiscriminator(config_opt)
        
        gen_params = sum(p.numel() for p in gen.parameters())
        disc_params = sum(p.numel() for p in disc.parameters())
        total_params = gen_params + disc_params
        total_size_mb = total_params * 4 / 1024 / 1024
        
        print(f"{name:8s}: {total_params:8,} 参数 ({total_size_mb:6.2f} MB)")

if __name__ == "__main__":
    print("开始性能基准测试...")
    
    import torch.nn as nn
    
    tests = [
        benchmark_forward_pass,
        benchmark_training_step,
        benchmark_memory_efficiency,
        benchmark_model_size
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print("\n🎉 性能基准测试完成！")
