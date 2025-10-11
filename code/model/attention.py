import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelGatedFusion(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, 2),
            nn.Softmax(dim=1)  # output 2 weights
        )

    def forward(self, top_feat, bot_feat):
        B, C, H, W = top_feat.shape
        top_stat = self.avg_pool(top_feat).view(B, -1)
        bot_stat = self.avg_pool(bot_feat).view(B, -1)
        fuse_input = torch.cat([top_stat, bot_stat], dim=1)  # [B, 2C]
        weights = self.fc(fuse_input).unsqueeze(-1).unsqueeze(-1)  # [B, 2, 1, 1]

        w_top = weights[:, 0:1]
        w_bot = weights[:, 1:2]
        fused = w_top * top_feat + w_bot * bot_feat
        return fused

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_channels = max(1, in_channels // reduction)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(a))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out