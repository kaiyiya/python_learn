import torch
from torch.utils.data import DataLoader, random_split
from dataset import ContextEncoderDataset
from model.net import ContextEncoderGenerator, ContextEncoderDiscriminator
from model.training import ContextEncoderTrainer
import configs.config_loader as cfg_loader


if __name__ == '__main__':
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    args = cfg_loader.get_config()

    # cuDNN benchmark（固定分辨率有益）
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 数据集与划分
    # 方式1: 从训练数据中划分出训练集、验证集、测试集
    full_dataset = ContextEncoderDataset(args, is_train=True)
    train_ratio = 0.7  # 训练集占70%
    val_ratio = 0.15   # 验证集占15%
    test_ratio = 0.15  # 测试集占15%
    
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size  # 剩余部分作为测试集
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 方式2: 如果有独立的测试数据目录，可以这样加载（取消注释使用）
    # test_dataset = ContextEncoderDataset(args, is_train=False)
    # 注意：需要在config中设置 test_dataroot 参数，或直接修改 args.dataroot

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False
    )

    print(f'训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}')

    # 创建模型
    from model.net import weights_init
    generator = ContextEncoderGenerator(args)
    discriminator = ContextEncoderDiscriminator(args)
    
    # 应用权重初始化
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # 训练
    trainer = ContextEncoderTrainer(
        train_loader, generator, discriminator, args, device, 
        val_loader=val_loader, test_loader=test_loader
    )
    trainer.train_model()

