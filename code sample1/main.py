
import torch
from torch.utils.data import DataLoader, random_split
from dataset import MyDataset
from model.net import UNet
from model.training import Trainer
import configs.config_loader as cfg_loader



if __name__ == '__main__':
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    args = cfg_loader.get_config()

    # 创建数据集（使用训练集目录）并划分训练/验证
    full_dataset = MyDataset('data/train/img', 'data/train/gt')
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    model = UNet(in_channels=1, out_channels=1)

    # 创建训练器并开始训练（含验证集）
    trainer = Trainer(train_loader, model, args, device, val_loader=val_loader)
    trainer.train_model()

