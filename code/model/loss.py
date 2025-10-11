import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
import torchvision.models as models
import numpy as np


class DynamicLossWeight:
    def __init__(self, momentum=0.65, clip=10.0):
        self.avg_top = None
        self.avg_bot = None
        self.momentum = momentum
        self.clip = clip

    def update(self, loss_top_val, loss_bot_val):
        if self.avg_top is None:
            self.avg_top = loss_top_val
            self.avg_bot = loss_bot_val
        else:
            self.avg_top = self.momentum * self.avg_top + (1 - self.momentum) * loss_top_val
            self.avg_bot = self.momentum * self.avg_bot + (1 - self.momentum) * loss_bot_val

        wt = self.avg_top / (self.avg_top + self.avg_bot + 1e-6) * 2
        wb = self.avg_bot / (self.avg_top + self.avg_bot + 1e-6) * 2

        wt = min(max(wt, 1/self.clip), self.clip)
        wb = min(max(wb, 1/self.clip), self.clip)
        return wt, wb

loss_balancer = DynamicLossWeight()

class Hybrid_loss_v2(nn.Module):
    def __init__(self, vmin=0.7, vmax=0.9):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.mse = nn.MSELoss(reduction='none')
        self.smoothl1 = nn.SmoothL1Loss(beta=0.01, reduction='mean')
        self.l1 = nn.L1Loss()
        
    def forward(self, pred_top, pred_bot, label_top, label_bot, mask_density_top, mask_density_bot, epoch):
        device = pred_top.device
        label_top = label_top.to(device)
        label_bot = label_bot.to(device)
        mask_density_top = mask_density_top.to(device)
        mask_density_bot = mask_density_bot.to(device)
        
        mask_top = (label_top > 0.5).float()
        mask_bot = (label_bot > 0.5).float()

        cell_mask_top = (mask_density_top > 1e-3).float()
        cell_mask_bot = (mask_density_bot > 1e-3).float()

        label_top_norm = label_top
        label_bot_norm = label_bot
        
        # MSE loss
        mask_density_bot = ((mask_density_bot > 0.4) & (mask_density_bot < 1.0)).float()
        mask_density_top = ((mask_density_top > 0.4) & (mask_density_top < 1.0)).float()
        alpha = 3.0
        weight_top = cell_mask_top * (1.0 + alpha * mask_density_top) * mask_top
        weight_bot = cell_mask_bot * (1.0 + alpha * mask_density_bot) * mask_bot
        
    
        loss_top = (self.mse(pred_top, label_top_norm) * weight_top).sum() / (weight_top.sum() + 1e-6)
        loss_bot = (self.mse(pred_bot, label_bot_norm) * weight_bot).sum() / (weight_bot.sum() + 1e-6)

        grad_k = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=device).view(1,1,3,3)/8.
        grad_loss_top = self.l1(F.conv2d(pred_top, grad_k, padding=1), F.conv2d(label_top_norm, grad_k, padding=1))
        grad_loss_bot = self.l1(F.conv2d(pred_bot, grad_k, padding=1), F.conv2d(label_bot_norm, grad_k, padding=1))
        edge_loss = grad_loss_top + grad_loss_bot

        
        # SSIM
        ssim_loss_top = 1 - ms_ssim(pred_top, label_top_norm, data_range=1.0, size_average=True)
        ssim_loss_bot = 1 - ms_ssim(pred_bot, label_bot_norm, data_range=1.0, size_average=True)

        ssim_loss = ssim_loss_top + ssim_loss_bot
    
        wt, wb = loss_balancer.update(loss_top.item(), loss_bot.item())
        we = min(0.05 + epoch * 0.001, 0.5)
        ws = min(0.025 + epoch * 0.001, 0.25)
        
        total_loss = wt * loss_top + wb * loss_bot + we * edge_loss  + ws * ssim_loss
        loss_parts = {
            'mse_top': loss_top.item(),
            'mse_bot': loss_bot.item(),
            'edge': edge_loss.item(),
            'ssim': ssim_loss.item(),
        }
        
        return total_loss, loss_parts
