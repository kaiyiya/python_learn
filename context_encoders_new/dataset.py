import torch
import torchvision
from torch.utils.data import Dataset, random_split, DataLoader
import os
from PIL import Image
from torchvision import transforms
import random
import numpy as np


class ContextEncoderDataset(Dataset):
    """
    上下文编码器数据集类
    用于图像修复任务，支持多种数据集格式
    """
    def __init__(self, opt, is_train=True):
        self.opt = opt
        self.is_train = is_train
        self.image_size = opt.image_size
        self.overlap_pred = opt.overlap_pred
        
        # 根据数据集类型加载数据
        if opt.dataset in ['imagenet', 'folder', 'streetview']:
            self.dataset = self._load_folder_dataset()
        elif opt.dataset == 'cifar10':
            self.dataset = self._load_cifar10_dataset()
        else:
            raise ValueError(f"不支持的数据集类型: {opt.dataset}")
    
    def _load_folder_dataset(self):
        """加载文件夹格式的数据集"""
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        return torchvision.datasets.ImageFolder(
            root=self.opt.dataroot,
            transform=transform
        )
    
    def _load_cifar10_dataset(self):
        """加载CIFAR-10数据集"""
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        return torchvision.datasets.CIFAR10(
            root=self.opt.dataroot,
            download=True,
            transform=transform
        )
    
    def _create_corrupted_image(self, image):
        """
        创建损坏的图像（中心区域用固定颜色填充）
        这是Context Encoder的核心：从完整图像创建需要修复的损坏图像
        """
        corrupted = image.clone()
        batch_size, channels, height, width = image.shape
        
        # 计算中心区域的位置
        center_h_start = height // 4
        center_h_end = center_h_start + height // 2
        center_w_start = width // 4
        center_w_end = center_w_start + width // 2
        
        # 计算需要填充的区域（考虑重叠）
        fill_h_start = center_h_start + self.overlap_pred
        fill_h_end = center_h_end - self.overlap_pred
        fill_w_start = center_w_start + self.overlap_pred
        fill_w_end = center_w_end - self.overlap_pred
        
        # 用固定颜色填充中心区域
        # 这些颜色值对应ImageNet的均值（归一化后）
        corrupted[:, 0, fill_h_start:fill_h_end, fill_w_start:fill_w_end] = 2 * 117.0 / 255.0 - 1.0  # 红色通道
        corrupted[:, 1, fill_h_start:fill_h_end, fill_w_start:fill_w_end] = 2 * 104.0 / 255.0 - 1.0  # 绿色通道
        corrupted[:, 2, fill_h_start:fill_h_end, fill_w_start:fill_w_end] = 2 * 123.0 / 255.0 - 1.0  # 蓝色通道
        
        return corrupted
    
    def _extract_center_region(self, image):
        """
        提取图像的中心区域作为修复目标
        """
        batch_size, channels, height, width = image.shape
        center_h_start = height // 4
        center_h_end = center_h_start + height // 2
        center_w_start = width // 4
        center_w_end = center_w_start + width // 2
        
        return image[:, :, center_h_start:center_h_end, center_w_start:center_w_end]
    
    def __getitem__(self, index):
        # 获取原始图像
        image, _ = self.dataset[index]
        
        # 如果是单张图像，需要添加batch维度
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 创建损坏的图像
        corrupted_image = self._create_corrupted_image(image)
        
        # 提取中心区域作为修复目标
        center_region = self._extract_center_region(image)
        
        # 移除batch维度（DataLoader会自动添加）
        corrupted_image = corrupted_image.squeeze(0)
        center_region = center_region.squeeze(0)
        
        return corrupted_image, center_region
    
    def __len__(self):
        return len(self.dataset)


class MyDataset(Dataset):
    """
    保留原有的数据集类以保持兼容性
    """
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = os.listdir(img_dir)
        self.mask_files = self.img_files
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.ToTensor()
            ]
        )

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        mask_path = os.path.join(self.mask_dir, self.img_files[index])
        img_, mask_ = Image.open(img_path), Image.open(mask_path)
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        img = self.transform(img_)
        torch.random.manual_seed(seed)
        mask = self.transform(mask_)
        mask *= 255.

        return img, mask

    def __len__(self):
        return len(self.img_files)


def create_dataloader(opt, is_train=True):
    """
    创建数据加载器的工厂函数
    """
    dataset = ContextEncoderDataset(opt, is_train)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=is_train,
        num_workers=opt.workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    
    return dataloader


if __name__ == '__main__':
    # 测试Context Encoder数据集
    class Args:
        def __init__(self):
            self.dataset = 'streetview'
            self.dataroot = 'data/train'
            self.image_size = 128
            self.overlap_pred = 4
            self.batch_size = 4
            self.workers = 2
    
    opt = Args()
    
    # 测试数据集
    try:
        dataset = ContextEncoderDataset(opt)
        print(f"数据集大小: {len(dataset)}")
        
        # 测试单个样本
        corrupted, center = dataset[0]
        print(f"损坏图像形状: {corrupted.shape}")
        print(f"中心区域形状: {center.shape}")
        
        # 测试数据加载器
        dataloader = create_dataloader(opt)
        for i, (corrupted_batch, center_batch) in enumerate(dataloader):
            print(f"批次 {i}: 损坏图像 {corrupted_batch.shape}, 中心区域 {center_batch.shape}")
            if i >= 2:  # 只测试前几个批次
                break
                
    except Exception as e:
        print(f"测试数据集时出错: {e}")
        print("请确保数据路径正确")

