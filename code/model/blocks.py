import torch.nn as nn
import torch.nn.functional as F
from .attention import CBAM
import torch

class EnhancedInputHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.attn1 = CBAM(out_channels)

        self.res2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.attn2 = CBAM(out_channels)       

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.initial(x)
       
        residual = out
        out1 = self.res1(out)
        out1 += residual
        out1 = self.relu(out1)
        out1 = self.attn1(out1)

        residual1 = out1
        out2 = self.res2(out1)
        out2 += residual1
        out2 = self.relu(out2)
        out2 = self.attn2(out2)
       
        return out2


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class OutputHeadTop(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(OutputHeadTop, self).__init__()

        self.predictor2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            CBAM(in_channels//2),
            UNetBlock(in_channels//2, in_channels // 4),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        out = self.predictor2(x)
        return out  