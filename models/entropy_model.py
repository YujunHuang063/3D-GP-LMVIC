import pdb
import torch
import math
from torch import nn
from compressai.models.utils import update_registered_buffers, conv, deconv
from compressai.entropy_models import GaussianConditional
from compressai.models import CompressionModel, get_scale_table
from compressai.ops import quantize_ste
from compressai.layers import ResidualBlock, GDN, MaskedConv2d, conv3x3, ResidualBlockWithStride
import torch.nn.functional as F
import copy

from utils_main.general_utils import pad_to_window_size

class CheckMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask: A
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """
    def __init__(self, *args, mask_type: str = "A", **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')
        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        if mask_type == "A":
            self.mask[:, :, 0::2, 1::2] = 1
            self.mask[:, :, 1::2, 0::2] = 1
        else:
            self.mask[:, :, 0::2, 0::2] = 1
            self.mask[:, :, 1::2, 1::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)
        return out

class Hyperprior(CompressionModel):
    def __init__(self, in_planes: int = 192, mid_planes: int = 192, out_planes: int=192):
        super().__init__(entropy_bottleneck_channels=mid_planes)
        self.hyper_encoder = nn.Sequential(
            conv(in_planes, mid_planes, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(mid_planes, mid_planes, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(mid_planes, mid_planes, stride=2, kernel_size=5),
        )
        if out_planes == 2 * in_planes:
            self.hyper_decoder = nn.Sequential(
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                deconv(mid_planes, in_planes * 3 // 2, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(in_planes * 3 // 2, out_planes, stride=1, kernel_size=3),
            )
        else:
            self.hyper_decoder = nn.Sequential(
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                deconv(mid_planes, mid_planes, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(mid_planes, out_planes, stride=1, kernel_size=3),
            )

    def forward(self, y, out_z=False):
        y, pad_h, pad_w  = pad_to_window_size(y, 4)
        z = self.hyper_encoder(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset
        params = self.hyper_decoder(z_hat)
        params = params[:, :, :params.shape[2] - pad_h, :params.shape[3] - pad_w]
        if out_z:
            return params, z_likelihoods, z_hat
        else:
            return params, z_likelihoods

    def compress(self, y):
        y, pad_h, pad_w  = pad_to_window_size(y, 4)
        z = self.hyper_encoder(y)
        z_strings = self.entropy_bottleneck.compress(z)

        medians = self.entropy_bottleneck._get_medians().detach()
        spatial_dims = len(z.size()) - 2
        medians = self.entropy_bottleneck._extend_ndims(medians, spatial_dims)
        medians = medians.expand(z.size(0), *([-1] * (spatial_dims + 1)))
        z_hat = torch.round(z - medians) + medians

        # z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.hyper_decoder(z_hat)
        params = params[:, :, :params.shape[2] - pad_h, :params.shape[3] - pad_w]
        return params, z_hat, z_strings #{"strings": z_string, "shape": z.size()[-2:]}

    def decompress(self, strings, z_shape, y_shape=None):
        #assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings, z_shape)
        params = self.hyper_decoder(z_hat)
        if y_shape:
            params = params[:, :, :y_shape[0], :y_shape[1]]
        return params, z_hat
