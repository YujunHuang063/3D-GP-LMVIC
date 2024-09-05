import pdb
import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F

from compressai.models.utils import conv
from compressai.layers import ResidualBlock, GDN, MaskedConv2d, conv3x3, ResidualBlockWithStride

from utils_main.general_utils import pad_to_window_size
    
class ICT(nn.Module):
    def __init__(self, channels, window_size, num_heads, args=None):
        super(ICT, self).__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.args = args

        self.main_net1 = conv(channels, channels, kernel_size=3, stride=1)
        self.ref_net1 = conv(channels, channels, kernel_size=3, stride=1)
        self.main_net2 = ResidualBlock(channels*2, channels)

    def forward(self, x_main, x_ref, x_mask=None):
        x_main, pad_h, pad_w = pad_to_window_size(x_main, self.window_size)
        x_ref, _, _ = pad_to_window_size(x_ref, self.window_size)
        x_mask, _, _ = pad_to_window_size(x_mask, self.window_size) if x_mask is not None else (None, None, None)

        x_main_init = x_main
        x_main = self.main_net1(x_main)
        x_ref = self.ref_net1(x_ref)
        x_concat = torch.cat((x_main, x_ref), 1)

        x_concat = (x_concat * (torch.sigmoid(x_mask) if (self.args and self.args.sigmoid_mask) else x_mask)) if (x_mask is not None and (not self.args or not self.args.wo_img_mask)) else x_concat
        x_main = self.main_net2(x_concat)
        x_main = x_main_init + x_main

        x_main = x_main[:, :, :x_main.shape[2] - pad_h, :x_main.shape[3] - pad_w]
        return x_main