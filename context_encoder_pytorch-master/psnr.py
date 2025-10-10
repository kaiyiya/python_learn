import numpy
import math
# import scipy.misc

# PSNR用于衡量图像质量，值越大表示图像失真越小。
def psnr(img1, img2):
    # 计算均方误差
    mse = numpy.mean((img1 - img2) ** 2)
    # 如果两个图像完全一致，则返回100
    if mse == 0:
        return 100
    # 这段代码计算PSNR（峰值信噪比）值。首先定义像素最大值为255.0，然后使用公式20 * log10(PIXEL_MAX / sqrt(mse))计算PSNR值并返回。该公式是图像质量评估的标准方法，用于衡量两张图像的相似度。
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# real = scipy.misc.imread('dataset/test/test/006_im.png').astype(numpy.float32)[32:32+64,32:32+64,:]
# recon = scipy.misc.imread('out.png').astype(numpy.float32)[32:32+64,32:32+64,:]


# print(psnr(real,recon))
