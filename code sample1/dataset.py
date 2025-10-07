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



if __name__ == '__main__':
    dataset = MyDataset('train/img', 'train/gt')
    print(dataset[0])

