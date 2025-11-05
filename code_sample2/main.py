from dataset import *
from model.net import *
from model.training import *
import configs.config_loader as cfg_loader


def plot_batch_comparison(images, masks, batch_idx, cols=4):
    """
    绘制一个batch的img和mask对比图
    images: [B, C, H, W]
    masks: [B, C, H, W]
    """
    batch_size = images.shape[0]
    rows = (batch_size + cols - 1) // cols * 2  # 每张图片占两行

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size):
        # 计算行索引
        row_img = (i // cols) * 2
        row_mask = row_img + 1
        col = i % cols

        # 转换为numpy并调整维度
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
        mask_np = masks[i].cpu().numpy().transpose(1, 2, 0)

        # 处理单通道
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)
        if mask_np.shape[2] == 1:
            mask_np = mask_np.squeeze(2)

        # 显示原图
        axes[row_img, col].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
        axes[row_img, col].set_title(f'Img {i}')
        axes[row_img, col].axis('off')

        # 显示mask
        axes[row_mask, col].imshow(mask_np, cmap='gray')
        axes[row_mask, col].set_title(f'Mask {i}')
        axes[row_mask, col].axis('off')

    # 隐藏多余的子图
    for i in range(batch_size, cols * (rows // 2)):
        row_img = (i // cols) * 2
        row_mask = row_img + 1
        col = i % cols
        if rows > 1:
            axes[row_img, col].axis('off')
            axes[row_mask, col].axis('off')

    plt.suptitle(f'Batch {batch_idx}')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    args = cfg_loader.get_config()


    train_set = MyDataset('data/train/img', 'data/train/gt',augment=True)
    train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=False)
    val_set = MyDataset('data/val/img', 'data/val/gt',augment=False)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

    model = UNet(in_channels=1, out_channels=1)
    trainer = Trainer(train_loader,val_loader, model, args)
    trainer.train_model()

