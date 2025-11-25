import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from dataset import ContextEncoderDataset
from model.net import ContextEncoderGenerator, ContextEncoderDiscriminator
from model.training import ContextEncoderTrainer
import configs.config_loader as cfg_loader


def _resolve_dir(*candidates):
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


def _compute_split_sizes(total, ratios):
    normalized = [max(0.0, r) for r in ratios]
    if total <= 0:
        raise ValueError("数据集大小必须大于 0")
    if sum(normalized) == 0:
        raise ValueError("拆分比例之和不能为 0")
    ratio_sum = sum(normalized)
    normalized = [r / ratio_sum for r in normalized]

    sizes = []
    remaining = total
    for idx, frac in enumerate(normalized):
        if idx == len(normalized) - 1:
            size = remaining
        else:
            size = int(round(total * frac))
            min_remaining = len(normalized) - idx - 1
            size = max(1, min(size, remaining - min_remaining))
        sizes.append(size)
        remaining -= size
    return sizes


if __name__ == '__main__':
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    args = cfg_loader.get_config()

    # cuDNN benchmark（固定分辨率有益）
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 自动检测数据目录
    default_train_dir = _resolve_dir(args.train_dir, args.dataroot, 'data/train')
    if not default_train_dir:
        raise RuntimeError("未找到可用的训练数据目录，请检查 --train_dir 或 --dataroot 设置。")
    default_val_dir = _resolve_dir(args.val_dir, 'data/val')
    default_test_dir = _resolve_dir(args.test_dir, 'data/test')

    # 构建训练集
    train_dataset = ContextEncoderDataset(args, is_train=True, dataroot=default_train_dir)

    # 构建验证/测试集（如有）
    val_dataset = ContextEncoderDataset(args, is_train=False, dataroot=default_val_dir) if default_val_dir else None
    test_dataset = ContextEncoderDataset(args, is_train=False, dataroot=default_test_dir) if default_test_dir else None

    # 如果没有单独的验证集，则对训练数据进行划分
    if val_dataset is None:
        print("⚠ 未提供独立验证集，将从训练集中划分。")
        base_dataset = train_dataset

        if test_dataset is None:
            train_size, val_size, test_size = _compute_split_sizes(
                len(base_dataset),
                [args.train_ratio, args.val_ratio, args.test_ratio]
            )
            train_dataset, val_dataset, test_dataset = random_split(
                base_dataset, [train_size, val_size, test_size]
            )
        else:
            train_size, val_size = _compute_split_sizes(
                len(base_dataset),
                [1.0 - args.val_ratio, args.val_ratio]
            )
            train_dataset, val_dataset = random_split(
                base_dataset, [train_size, val_size]
            )

    # 如果没有单独测试集，则复用验证集
    if test_dataset is None and val_dataset is not None:
        print("⚠ 未提供独立测试集，测试阶段将复用验证集。")
        test_dataset = ContextEncoderDataset(args, is_train=False,
                                             dataroot=default_val_dir) if default_val_dir else val_dataset

    if val_dataset is None or test_dataset is None:
        raise RuntimeError("无法构建验证集或测试集，请检查数据路径配置。")

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

    print(f'训练集路径: {default_train_dir}, 样本数: {len(train_dataset)}')
    if default_val_dir:
        print(f'验证集路径: {default_val_dir}, 样本数: {len(val_dataset)}')
    else:
        print(f'验证集来自训练集划分，样本数: {len(val_dataset)}')
    if default_test_dir:
        print(f'测试集路径: {default_test_dir}, 样本数: {len(test_dataset)}')
    else:
        print(f'测试集来自验证集复用，样本数: {len(test_dataset)}')

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
