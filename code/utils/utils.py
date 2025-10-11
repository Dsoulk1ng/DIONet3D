import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_pred(pred, gt=None, title="Prediction"):
    """
    可视化预测结果和真实标签（如果有）
    Args:
        pred (torch.Tensor): 预测结果 [H, W]
        gt (torch.Tensor, optional): 真实标签 [H, W]
        title (str): 标题
    """
    pred_np = pred.cpu().numpy()
    plt.imshow(pred_np, cmap="hot")
    plt.colorbar()
    if gt is not None:
        gt_np = gt.cpu().numpy()
        plt.contour(gt_np, colors='blue', linewidths=0.5)
    plt.title(title)
    plt.show()

def calculate_metrics(pred, gt):
    """
    计算MSE和MAE
    Args:
        pred (torch.Tensor): 预测结果 [B, 1, H, W]
        gt (torch.Tensor): 真实标签 [B, 1, H, W]
    Returns:
        mse (float), mae (float)
    """
    mse = torch.mean((pred - gt) ** 2).item()
    mae = torch.mean(torch.abs(pred - gt)).item()
    return mse, mae
