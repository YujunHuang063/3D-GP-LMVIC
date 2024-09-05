import pdb

import torch
from torch import nn

class Entropy_Parameters(nn.Module):
    def __init__(self, M = 192, mode=0):
        super().__init__()
        if mode == 0:
            self.entropy_parameters = nn.Sequential(
                nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
            )
        elif mode==1:
            self.entropy_parameters = nn.Sequential(
                nn.Conv2d(M * 18 // 3, M * 14 // 3, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(M * 14 // 3, M * 10 // 3, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(M * 10 // 3, M * 6 // 3, 1),
            )

    def forward(self, x):
        return self.entropy_parameters(x)