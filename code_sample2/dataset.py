import torch
import torchvision
from torch.utils.data import Dataset, random_split, DataLoader
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random


# shape = (512, 512)
class MyDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = os.listdir(img_dir)
        self.mask_files = self.img_files
        self.aug_transform = self.get_augmentation_transform()
        self.base_transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        self.augment=augment

    def get_augmentation_transform(self):
        return [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),  # 平移
                scale=(0.9, 1.1),  # 缩放
                shear=5  # 剪切
            ),
            # transforms.RandomPerspective(
            #     distortion_scale=0.1,  # 透视变形程度
            #     p=0.3
            # ),
            transforms.ElasticTransform(
                alpha=10.0,  # 弹性变形的幅度
                sigma=3.0  # 弹性变形的平滑度
            )
        ]
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        mask_path = os.path.join(self.mask_dir, self.img_files[index])
        img_, mask_ = Image.open(img_path), Image.open(mask_path)


        if self.augment:
            seed = index+int(torch.randint(0,1000000,(1,)).item())

            # seed = torch.random.seed()
            torch.random.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            num_transforms=random.randint(1,min(3,len(self.aug_transform)))
            selected_transforms=random.sample(self.aug_transform,num_transforms)
            transform_pipeline=transforms.Compose(selected_transforms+[transforms.ToTensor()])

            img=transform_pipeline(img_)
            torch.random.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            mask=transform_pipeline(mask_)
            # img = self.aug_transform(img_)
            # img = (img-img.min())/(img.max()-img.min())
            # torch.random.manual_seed(seed)
            # mask = self.aug_transform(mask_)
        else:
            img=self.base_transform(img_)
            mask=self.base_transform(mask_)

        # mask = mask * 255.
        mask = (mask > 0).float()
        # print(f"Mask1 unique values: {torch.unique(mask1)}")
        # print(f"Mask2 unique values: {torch.unique(mask2)}")
        # print(torch.all(mask1==mask2))

        # print(f"Image shape: {img.shape}")
        # print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
        # print(f"Mask shape: {mask.shape}")
        # print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")

        return img, mask

    def __len__(self):
        return len(self.img_files)


def plot_all_images_and_masks(dataset):
    """绘制所有的img和mask对比图"""
    num_images = len(dataset)

    # 计算子图的行数和列数
    cols = 2  # 两列：原图和mask
    rows = num_images

    # 创建大图
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * num_images))

    # 如果只有一张图片，axes的shape需要调整
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        img, mask = dataset[i]

        # 将张量转换为numpy数组用于显示
        img_np = img.numpy().transpose(1, 2, 0)  # 从(C, H, W)转为(H, W, C)
        mask_np = mask.numpy().transpose(1, 2, 0)  # 从(C, H, W)转为(H, W, C)

        # 如果图像是单通道，去掉通道维度
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)
        if mask_np.shape[2] == 1:
            mask_np = mask_np.squeeze(2)

        # 显示原图
        axes[i, 0].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
        axes[i, 0].set_title(f'Image {i + 1}: {dataset.img_files[i]}')
        axes[i, 0].axis('off')

        # 显示mask
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f'Mask {i + 1}')
        axes[i, 1].axis('off')

        print(f"Processed image {i + 1}/{num_images}: {dataset.img_files[i]}")

    plt.tight_layout()
    plt.show()


def plot_images_in_grid(dataset, cols=4):
    """以网格形式绘制img和mask对比图"""
    num_images = len(dataset)
    rows = (num_images + cols - 1) // cols * 2  # 每张图片占两行（原图和mask）

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

    for i in range(num_images):
        img, mask = dataset[i]

        # 计算行索引：每张图片占用两行
        row_img = (i // cols) * 2
        row_mask = row_img + 1
        col = i % cols

        # 将张量转换为numpy数组用于显示
        img_np = img.numpy().transpose(1, 2, 0)
        mask_np = mask.numpy().transpose(1, 2, 0)

        # 如果图像是单通道，去掉通道维度
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)
        if mask_np.shape[2] == 1:
            mask_np = mask_np.squeeze(2)

        # 显示原图
        axes[row_img, col].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
        axes[row_img, col].set_title(f'Img {i + 1}')
        axes[row_img, col].axis('off')

        # 显示mask
        axes[row_mask, col].imshow(mask_np, cmap='gray')
        axes[row_mask, col].set_title(f'Mask {i + 1}')
        axes[row_mask, col].axis('off')

    # 隐藏多余的子图
    for i in range(num_images, cols * (rows // 2)):
        row_img = (i // cols) * 2
        row_mask = row_img + 1
        col = i % cols
        axes[row_img, col].axis('off')
        axes[row_mask, col].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 初始化数据集
    dataset = MyDataset('data/val/img', 'data/val/gt',augment=False)

    print(f"数据集共有 {len(dataset)} 张图片")
    # print("图片文件名列表:", dataset.img_files)

    # # 方法1：绘制所有图片的垂直排列对比图
    # print("正在绘制垂直排列的对比图...")
    # plot_all_images_and_masks(dataset)

    # 方法2：绘制网格排列的对比图（更紧凑）
    print("正在绘制网格排列的对比图...")
    plot_images_in_grid(dataset, cols=4)  # 可以调整cols参数来控制每行显示的图片数量

    # 额外：显示第一张图片的详细信息
    if len(dataset) > 0:
        img, mask = dataset[8]
        print(f"\n第9张图片的详细信息:")
        print(f"Image shape: {img.shape}")
        print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        print(f"Mask unique values: {torch.unique(mask)}")
