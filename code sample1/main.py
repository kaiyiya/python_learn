
import torch
from torch.utils.data import DataLoader
from dataset import MyDataset
from model.net import UNet
from model.training import Trainer
import configs.config_loader as cfg_loader



if __name__ == '__main__':
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    args = cfg_loader.get_config()

    # 创建数据集和数据加载器
    dataset = MyDataset('data/val/img', 'data/val/gt')
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    # 创建模型
    model = UNet(in_channels=1, out_channels=1)

    # 创建训练器并开始训练
    trainer = Trainer(dataloader, model, args, device)
    trainer.train_model()

