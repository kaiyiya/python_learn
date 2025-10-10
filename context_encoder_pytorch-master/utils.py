import torch
from PIL import Image
from torch.autograd import Variable


# 加载和调整图片大小的函数
def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


# 将张量数据保存为图像文件
def save_image(filename, data):
    # 这种变换常见于深度学习图像处理中，因为神经网络通常使用归一化数据[-1,1]进行训练，而保存图像时需要转换回标准的[0,255]像素值范围。
    img = data.clone().add(1).div(2).mul(255).clamp(0, 255).numpy()
    # 将PyTorch的通道优先格式 (C, H, W) 转换为PIL库要求的通道后置格式 (H, W, C)
    img = img.transpose(1, 2, 0).astype("uint8")
    # 创建一个Image对象
    img = Image.fromarray(img)
    # 保存图片
    img.save(filename)


def gram_matrix(y):
    # PyTorch: (batch_size, channels, height, width) - 通常称为 NCHW 格式
    (b, ch, h, w) = y.size()
    # 将输入张量y从4D形状(batch_size, channels, height, width)重新整形为3D形状(batch_size, channels, width*height)。
    features = y.view(b, ch, w * h)
    #  features_t = features.transpose(1, 2) - 将特征矩阵的通道维度和空间维度进行转置
    features_t = features.transpose(1, 2)
    # 计算 Gram 矩阵，即特征矩阵的乘积
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # 创建一个与输入batch数据形状相同的新张量，用于存储均值
    mean = batch.data.new(batch.data.size())
    # 创建一个与输入batch数据形状相同的新张量，用于存储标准差
    std = batch.data.new(batch.data.size())
    # 这些参数是用于图像标准化的ImageNet数据集的均值和标准差
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    # 将输入的像素值从[0,255]范围缩放到[0,1]范围
    batch = torch.div(batch, 255.0)
    # 使用ImageNet数据集的均值对批次数据进行中心化处理，减去对应的RGB通道均值
    batch -= Variable(mean)
    # 使用ImageNet数据集的方差对批次数据进行标准化处理，除以对应的RGB通道标准差
    batch = torch.div(batch, Variable(std))
    return batch
