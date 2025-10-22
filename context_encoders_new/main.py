
import torch
import torch.backends.cudnn as cudnn
from dataset import ContextEncoderDataset, create_dataloader
from model.net import ContextEncoderGenerator, ContextEncoderDiscriminator
from model.training import ContextEncoderTrainer
import configs.config_loader as cfg_loader
import os


def main():
    """主函数"""
    # 加载配置
    opt = cfg_loader.get_config()
    
    # 打印配置信息
    print("=" * 50)
    print("Context Encoder 训练配置")
    print("=" * 50)
    print(f"数据集: {opt.dataset}")
    print(f"数据路径: {opt.dataroot}")
    print(f"批次大小: {opt.batch_size}")
    print(f"图像尺寸: {opt.image_size}")
    print(f"训练轮数: {opt.niter}")
    print(f"学习率: {opt.lr}")
    print(f"L2损失权重: {opt.wtl2}")
    print(f"使用CUDA: {opt.cuda}")
    print(f"随机种子: {opt.manual_seed}")
    print("=" * 50)
    
    # 检查CUDA可用性
    if torch.cuda.is_available() and not opt.cuda:
        print("警告: 检测到CUDA设备，但未启用CUDA。建议使用 --cuda 参数。")
    
    # 设置CUDA优化
    if opt.cuda:
        cudnn.benchmark = True
    
    # 创建数据加载器
    print("创建数据加载器...")
    try:
        train_loader = create_dataloader(opt, is_train=True)
        print(f"训练数据加载器创建成功，批次数量: {len(train_loader)}")
    except Exception as e:
        print(f"创建数据加载器失败: {e}")
        print("请检查数据路径是否正确")
        return
    
    # 创建训练器
    print("初始化训练器...")
    try:
        trainer = ContextEncoderTrainer(train_loader, opt)
        print("训练器初始化成功")
    except Exception as e:
        print(f"训练器初始化失败: {e}")
        return
    
    # 开始训练
    print("开始训练...")
    try:
        trainer.train_model()
        print("训练完成！")
    except KeyboardInterrupt:
        print("训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def test_models():
    """测试模型结构"""
    print("测试模型结构...")
    
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
    
    # 测试生成器
    generator = ContextEncoderGenerator(opt)
    print(f"生成器参数数量: {sum(p.numel() for p in generator.parameters())}")
    
    # 测试判别器
    discriminator = ContextEncoderDiscriminator(opt)
    print(f"判别器参数数量: {sum(p.numel() for p in discriminator.parameters())}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 128, 128)
    fake_center = torch.randn(2, 3, 64, 64)
    
    with torch.no_grad():
        generated = generator(x)
        d_real = discriminator(fake_center)
        d_fake = discriminator(generated)
    
    print(f"输入图像形状: {x.shape}")
    print(f"生成器输出形状: {generated.shape}")
    print(f"判别器真实输出形状: {d_real.shape}")
    print(f"判别器假图像输出形状: {d_fake.shape}")
    print("模型测试完成！")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # 测试模式
        test_models()
    else:
        # 正常训练模式
        main()

