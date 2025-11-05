import torch
import torchvision
from torch.utils.data import Dataset, random_split, DataLoader
import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
# shape = (512, 512)
class MyDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.base_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # 高级增强列表（随机选择1-3个组合）
        # 注意：对mask需使用NEAREST插值以保持二值
        self.aug_transforms = [
            'hflip',  # RandomHorizontalFlip
            'rot',    # RandomRotation(15)
            'affine', # RandomAffine
            'elastic' # ElasticTransform
        ]
        
        # 检查目录是否存在
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"图像目录不存在: {img_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"掩码目录不存在: {mask_dir}")
            
        # 获取文件列表并过滤（统一小写比较），按交集对齐
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        img_set = {f.lower(): f for f in img_files}
        mask_set = {f.lower(): f for f in mask_files}
        common_keys = sorted(list(set(img_set.keys()) & set(mask_set.keys())))

        if len(common_keys) == 0:
            raise RuntimeError("未在图像与掩码目录中找到同名文件，请检查文件命名是否一致")

        if len(common_keys) != len(img_files) or len(common_keys) != len(mask_files):
            print(f"警告: 图像与掩码未完全一一对应，按文件名交集对齐，样本数={len(common_keys)}")

        # 保存对齐后的原始文件名（保持原大小写）
        self.files = [(img_set[k], mask_set[k]) for k in common_keys]

    def __getitem__(self, index):
        img_name, mask_name = self.files[index]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 确保文件存在
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"文件不存在: {img_path} 或 {mask_path}")
            
        img_, mask_ = Image.open(img_path), Image.open(mask_path)
        
        # 转换为灰度图
        if img_.mode != 'L':
            img_ = img_.convert('L')
        if mask_.mode != 'L':
            mask_ = mask_.convert('L')
        
        # 检查原始掩码的像素值范围（在转换为tensor之前）
        mask_array = np.array(mask_)
        mask_min = mask_array.min()
        mask_max = mask_array.max()
        mask_unique = np.unique(mask_array)
        
        # 增强：从增强列表中随机取1-3个组合，并与ToTensor串联；用相同随机种子保证img/mask一致
        if self.augment and len(self.aug_transforms) > 0:
            seed = int(torch.randint(0, 1_000_000, (1,)).item())
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            num_transforms = random.randint(1, min(3, len(self.aug_transforms)))
            selected = random.sample(self.aug_transforms, num_transforms)

            # 为img与mask分别构建pipeline，确保插值方式不同但随机性一致
            img_ops = []
            mask_ops = []
            for name in selected:
                if name == 'hflip':
                    img_ops.append(transforms.RandomHorizontalFlip(p=0.5))
                    mask_ops.append(transforms.RandomHorizontalFlip(p=0.5))
                elif name == 'rot':
                    img_ops.append(transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR))
                    mask_ops.append(transforms.RandomRotation(15, interpolation=InterpolationMode.NEAREST))
                elif name == 'affine':
                    img_ops.append(transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.9,1.1), shear=5, interpolation=InterpolationMode.BILINEAR))
                    mask_ops.append(transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.9,1.1), shear=5, interpolation=InterpolationMode.NEAREST))
                elif name == 'elastic':
                    try:
                        img_ops.append(transforms.ElasticTransform(alpha=10.0, sigma=3.0, interpolation=InterpolationMode.BILINEAR))
                        mask_ops.append(transforms.ElasticTransform(alpha=10.0, sigma=3.0, interpolation=InterpolationMode.NEAREST))
                    except TypeError:
                        # 旧版本无 interpolation 参数，则退化为对img使用默认，对mask仍用默认，后续再二值化
                        img_ops.append(transforms.ElasticTransform(alpha=10.0, sigma=3.0))
                        mask_ops.append(transforms.ElasticTransform(alpha=10.0, sigma=3.0))

            img_pipeline = transforms.Compose(img_ops + [transforms.ToTensor()])
            mask_pipeline = transforms.Compose(mask_ops + [transforms.ToTensor()])

            img = img_pipeline(img_)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            mask = mask_pipeline(mask_)
        else:
            img = self.base_transform(img_)
            mask = self.base_transform(mask_)
        
        # 最终统一二值化（兼容0/1或0/255及插值后的灰度）：
        mask = (mask > 0).float()
        
        return img, mask

    def __len__(self):
        return len(self.files)



if __name__ == '__main__':
    dataset = MyDataset('train/img', 'train/gt')
    print(dataset[0])

