import torch
import torch.nn as nn
from .blocks import OutputHeadTop, EnhancedInputHead
from .attention import CBAM, ChannelGatedFusion
from .backbone import U_Net


class IRDropNetMultiDie(nn.Module):
    def __init__(self, global_in_c = 5, base_channels=64):
        
        super(IRDropNetMultiDie, self).__init__()

        self.input_head_top = EnhancedInputHead(global_in_c, base_channels)
        self.input_head_bottom = EnhancedInputHead(global_in_c, base_channels)

        
        self.fusionlayer1 = ChannelGatedFusion(in_channels=base_channels, reduction=16)
        
        self.net = U_Net(in_ch=base_channels, out_ch=64)
        
        
        self.adapt_top = nn.Sequential(nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels),
                               nn.ReLU(inplace=True), CBAM(base_channels))
        self.adapt_bot = nn.Sequential(nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels),
                                    nn.ReLU(inplace=True), CBAM(base_channels))
        
        self.output_head_top = OutputHeadTop(in_channels=base_channels, out_channels=1)
        self.output_head_bot = OutputHeadTop(in_channels=base_channels, out_channels=1)

    def forward(self, feat_top, feat_bottom):
        #top_main = feat_top[:, [1, 2, 3, 4], :, :]      # [B, 4, H, W]
        #bot_main = feat_bottom[:, [1, 2, 3, 4], :, :]

        top_main = feat_top
        bot_main = feat_bottom

        
        out_top = self.input_head_top(top_main)
        out_bottom = self.input_head_bottom(bot_main)
        
        combined = self.fusionlayer1(out_top, out_bottom)

        
        dec_out = self.net(combined)

        top_dec = self.adapt_top(dec_out)
        bot_dec = self.adapt_bot(dec_out)
        pred_top = self.output_head_top(top_dec)
        pred_bot = self.output_head_bot(bot_dec)

        return pred_top, pred_bot
