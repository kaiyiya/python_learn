import torch
import torchvision
from torch.utils.data import Dataset, random_split, DataLoader
import os
from PIL import Image
from torchvision import transforms
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
            
        # 获取文件列表并过滤
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 确保图像和掩码文件数量一致
        if len(self.img_files) != len(self.mask_files):
            print(f"警告: 图像文件数量({len(self.img_files)})与掩码文件数量({len(self.mask_files)})不一致")
            
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
        
        # 确保文件存在
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"文件不存在: {img_path} 或 {mask_path}")
            
        img_, mask_ = Image.open(img_path), Image.open(mask_path)
        
        # 转换为灰度图
        if img_.mode != 'L':
            img_ = img_.convert('L')
        if mask_.mode != 'L':
            mask_ = mask_.convert('L')
        
        # 使用相同的随机种子确保数据增强的一致性
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        img = self.transform(img_)
        torch.manual_seed(seed)
        mask = self.transform(mask_)
        
        # 归一化mask到0-1范围
        mask = mask * 255.0

        return img, mask

    def __len__(self):
        return len(self.img_files)



if __name__ == '__main__':
    dataset = MyDataset('train/img', 'train/gt')
    print(dataset[0])

