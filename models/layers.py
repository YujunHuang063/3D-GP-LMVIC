import torch
from torch import nn

class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, slope=0.01, inplace=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope, inplace=inplace),
        )
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 1)

        self.adaptor = None
        if in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity
    
class ConvFFN3(nn.Module):
    def __init__(self, in_ch, inplace=False):
        super().__init__()
        expansion_factor = 2
        internal_ch = in_ch * expansion_factor
        self.conv = nn.Conv2d(in_ch, internal_ch * 2, 1)
        self.conv_out = nn.Conv2d(internal_ch, in_ch, 1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=inplace)

    def forward(self, x):
        identity = x
        x1, x2 = self.conv(x).chunk(2, 1)
        out = self.relu1(x1) + self.relu2(x2)
        return identity + self.conv_out(out)
    
class DepthConvBlock4(nn.Module):
    def __init__(self, in_ch, out_ch, slope_depth_conv=0.01, inplace=False):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, slope=slope_depth_conv, inplace=inplace),
            ConvFFN3(out_ch, inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)
    
def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )
    
class UNet2(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, inplace=False):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DepthConvBlock4(in_ch, 32, inplace=inplace)
        self.conv2 = DepthConvBlock4(32, 64, inplace=inplace)
        self.conv3 = DepthConvBlock4(64, 128, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock4(128, 128, inplace=inplace),
            DepthConvBlock4(128, 128, inplace=inplace),
            DepthConvBlock4(128, 128, inplace=inplace),
            DepthConvBlock4(128, 128, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = DepthConvBlock4(128, 64, inplace=inplace)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = DepthConvBlock4(64, out_ch, inplace=inplace)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2