"""
图像修复任务的评估指标模块
包含PSNR、SSIM、MSE、MAE等多种评估指标
"""
import torch
import torch.nn.functional as F
import numpy as np
from math import log10


def denormalize_tensor(tensor, mean=0.5, std=0.5):
    """
    将归一化的tensor反归一化到[0, 1]范围
    Args:
        tensor: 归一化的tensor，范围通常在[-1, 1]
        mean: 归一化时使用的均值
        std: 归一化时使用的标准差
    Returns:
        反归一化后的tensor，范围[0, 1]
    """
    return (tensor * std) + mean


def tensor_to_numpy(tensor):
    """
    将PyTorch tensor转换为numpy数组，并调整维度顺序
    Args:
        tensor: PyTorch tensor，形状为 [C, H, W] 或 [B, C, H, W]
    Returns:
        numpy数组，形状为 [H, W, C] 或 [B, H, W, C]
    """
    if tensor.dim() == 3:
        # [C, H, W] -> [H, W, C]
        return tensor.cpu().detach().numpy().transpose(1, 2, 0)
    elif tensor.dim() == 4:
        # [B, C, H, W] -> [B, H, W, C]
        return tensor.cpu().detach().numpy().transpose(0, 2, 3, 1)
    else:
        return tensor.cpu().detach().numpy()


def calculate_mse(pred, target):
    """
    计算均方误差 (Mean Squared Error)
    Args:
        pred: 预测图像 tensor [B, C, H, W] 或 [C, H, W]
        target: 真实图像 tensor [B, C, H, W] 或 [C, H, W]
    Returns:
        MSE值
    """
    return F.mse_loss(pred, target).item()


def calculate_mae(pred, target):
    """
    计算平均绝对误差 (Mean Absolute Error)
    Args:
        pred: 预测图像 tensor
        target: 真实图像 tensor
    Returns:
        MAE值
    """
    return F.l1_loss(pred, target).item()


def calculate_psnr(pred, target, max_val=1.0):
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)
    Args:
        pred: 预测图像 tensor，范围[0, 1]或[-1, 1]
        target: 真实图像 tensor，范围[0, 1]或[-1, 1]
        max_val: 像素最大值，默认1.0（归一化图像）
    Returns:
        PSNR值（dB）
    """
    # 确保tensor在[0, 1]范围
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0  # 完全一致
    
    psnr = 20 * log10(max_val / torch.sqrt(mse))
    # 如果psnr是tensor，转换为float；如果已经是float，直接返回
    return psnr.item() if isinstance(psnr, torch.Tensor) else float(psnr)


def calculate_ssim(pred, target, window_size=11, size_average=True):
    """
    计算结构相似性指数 (Structural Similarity Index)
    简化版本，使用平均池化窗口
    Args:
        pred: 预测图像 tensor [B, C, H, W] 或 [C, H, W]
        target: 真实图像 tensor [B, C, H, W] 或 [C, H, W]
        window_size: 滑动窗口大小
        size_average: 是否对batch求平均
    Returns:
        SSIM值，范围[0, 1]，越接近1越好
    """
    
    # 确保tensor在[0, 1]范围
    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)
    
    # 如果是单张图像，添加batch维度
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        remove_batch = True
    else:
        remove_batch = False
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 计算均值（使用平均池化）
    mu1 = F.avg_pool2d(pred, window_size, 1, window_size // 2)
    mu2 = F.avg_pool2d(target, window_size, 1, window_size // 2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.avg_pool2d(pred * pred, window_size, 1, window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, 1, window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu1_mu2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # 对空间维度求平均
    if size_average:
        ssim = ssim_map.mean()
    else:
        # 对每个样本分别求平均
        ssim = ssim_map.view(ssim_map.size(0), -1).mean(dim=1)
    
    if remove_batch:
        ssim = ssim.squeeze(0) if ssim.dim() > 0 else ssim
    
    return ssim.item() if isinstance(ssim, torch.Tensor) else float(ssim)


def calculate_batch_metrics(pred, target):
    """
    批量计算所有指标
    Args:
        pred: 预测图像 tensor [B, C, H, W]
        target: 真实图像 tensor [B, C, H, W]
    Returns:
        dict: 包含所有指标的字典
    """
    metrics = {}
    
    # 基本指标
    metrics['mse'] = calculate_mse(pred, target)
    metrics['mae'] = calculate_mae(pred, target)
    
    # PSNR（逐个样本计算后平均）
    batch_psnr = []
    for i in range(pred.size(0)):
        psnr_val = calculate_psnr(pred[i], target[i])
        batch_psnr.append(psnr_val)
    metrics['psnr'] = np.mean(batch_psnr)
    metrics['psnr_std'] = np.std(batch_psnr)
    
    # SSIM（逐个样本计算后平均）
    batch_ssim = []
    for i in range(pred.size(0)):
        ssim_val = calculate_ssim(pred[i], target[i], size_average=False)
        batch_ssim.append(ssim_val)
    metrics['ssim'] = np.mean(batch_ssim)
    metrics['ssim_std'] = np.std(batch_ssim)
    
    return metrics


def calculate_metrics_summary(all_metrics):
    """
    计算所有批次的指标汇总
    Args:
        all_metrics: 字典列表，每个字典包含一个批次的指标
    Returns:
        dict: 包含平均值和标准差的汇总
    """
    summary = {}
    
    # 收集所有指标值
    metric_names = all_metrics[0].keys()
    
    for name in metric_names:
        values = [m[name] for m in all_metrics]
        summary[f'{name}_mean'] = np.mean(values)
        summary[f'{name}_std'] = np.std(values)
        summary[f'{name}_min'] = np.min(values)
        summary[f'{name}_max'] = np.max(values)
    
    return summary

