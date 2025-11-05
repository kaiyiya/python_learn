import torch
import torchvision
from torch.utils.data import Dataset, random_split, DataLoader
import os
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
# shape = (512, 512)
class MyDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
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
        
        # 同步数据增强：水平翻转 + 旋转，与原逻辑等价（随机角度±30°）
        # 对掩码使用最近邻插值，避免插值产生非0/1灰度
        rng = random.Random()
        seed = torch.randint(0, 2**32, (1,)).item()
        rng.seed(seed)

        do_flip = rng.random() < 0.5
        angle = rng.uniform(-30, 30)

        if do_flip:
            img_ = TF.hflip(img_)
            mask_ = TF.hflip(mask_)

        img_ = TF.rotate(img_, angle, interpolation=InterpolationMode.BILINEAR, expand=False)
        mask_ = TF.rotate(mask_, angle, interpolation=InterpolationMode.NEAREST, expand=False)

        # 转张量：img -> [0,1]，mask需要归一化
        img = TF.to_tensor(img_)
        mask = TF.to_tensor(mask_)
        
        # 处理mask：TF.to_tensor会将[0,255]缩放到[0,1]
        # 如果mask最大值>1，说明可能是0-255范围，需要归一化
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        # 二值化mask：如果mask值范围很小（可能是灰度图），需要二值化
        # 否则保持原样（假设已经是0-1范围的二值图）
        if mask.max() > 0.5:
            # 如果最大值>0.5，说明前景像素存在，进行二值化
            mask = (mask > 0.5).float()
        else:
            # 如果最大值很小，可能是数据本身几乎全为背景，保持原样
            # 但输出警告信息（在训练时会显示）
            pass
        
        return img, mask

    def __len__(self):
        return len(self.files)



if __name__ == '__main__':
    dataset = MyDataset('train/img', 'train/gt')
    print(dataset[0])

