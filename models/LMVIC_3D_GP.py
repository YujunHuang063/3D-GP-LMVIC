import os
import pdb
import time
import math

import numpy as np
from math import exp

from utils_main.general_utils import pad_to_window_size
from utils_main.graphics_utils import depth_projection_batch, match, downsample_offsets, torch_warp_simple
from models.entropy_model import Hyperprior, CheckMaskedConv2d
from models.ICT import ICT
from .sub_models import Entropy_Parameters
from .layers import UNet2

import torch
from torch import nn
from compressai.models.utils import update_registered_buffers, conv, deconv
from compressai.entropy_models import GaussianConditional
from compressai.models import CompressionModel, get_scale_table
from compressai.ops import quantize_ste
from compressai.layers import ResidualBlock, GDN, MaskedConv2d, conv3x3, ResidualBlockWithStride
from deepspeed.profiling.flops_profiler import get_model_profile
import torch.nn.functional as F
from torch.autograd import Variable

from compressai.ans import BufferedRansEncoder, RansDecoder
from pytorch_msssim import ssim, ms_ssim

class LMVIC_3D_GP(nn.Module):
    def __init__(self, N = 128, M = 192, args=None):
        super().__init__()
        self.N = N
        self.M = M
        self.args = args
        self.coder_layer_num = 4
        self.encoder_scale = 16
        self.hyper_encoder_scale = 4

        if not args.image_domain_align:
            img_input_channel = 3
        else:
            img_input_channel = 9

        if not args.downsample_scale1_fusion:
            img_encoder_start_net = nn.ModuleList([
                                        nn.Sequential(
                                            nn.Sequential(
                                                conv(img_input_channel, N, kernel_size=5, stride=2),
                                                GDN(N)
                                            ),
                                            ICT(channels=N, window_size=args.window_size, num_heads=args.num_heads, args=args)
                                        ),
                                    ])
        else:
            img_encoder_start_net = nn.ModuleList([
                                        nn.Sequential(
                                            nn.Sequential(
                                                conv(img_input_channel, N, kernel_size=5, stride=1),
                                                GDN(N)
                                            ),
                                            ICT(channels=N, window_size=args.window_size, num_heads=args.num_heads, args=args)
                                        ),
                                        nn.Sequential(
                                            nn.Sequential(
                                                conv(N, N, kernel_size=5, stride=2),
                                                GDN(N)
                                            ),
                                            ICT(channels=N, window_size=args.window_size, num_heads=args.num_heads, args=args)
                                        ),
                                    ])

        self.img_encoder = nn.Sequential(
            *img_encoder_start_net,
            nn.Sequential(
                nn.Sequential(
                    conv(N, N, kernel_size=5, stride=2),
                    GDN(N)
                ),
                ICT(channels=N, window_size=args.window_size, num_heads=args.num_heads, args=args)
            ),
            nn.Sequential(
                nn.Sequential(
                    conv(N, N, kernel_size=5, stride=2),
                    GDN(N)
                ),
                ICT(channels=N, window_size=args.window_size, num_heads=args.num_heads, args=args)
            ),
            conv(N, M, kernel_size=5, stride=2)
        )
        self.img_hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.img_context_prediction = conv(
            M, M*2, kernel_size=5, stride=1
        )
        
        self.img_entropy_parameters = Entropy_Parameters(M, 1)
        
        self.img_gaussian_conditional = GaussianConditional(None)
        self.M = M

        if not args.downsample_scale1_fusion:
            img_decoder_end_net = nn.ModuleList([
                                        nn.Sequential(
                                            GDN(N, inverse=True),
                                            deconv(N, 3, kernel_size=5, stride=2)
                                        )
                                    ])
        else:
            img_decoder_end_net = nn.ModuleList([
                                        nn.Sequential(
                                            nn.Sequential(
                                                GDN(N, inverse=True),
                                                deconv(N, N, kernel_size=5, stride=2)
                                            ),
                                            ICT(channels=N, window_size=args.window_size, num_heads=args.num_heads, args=args)
                                        ),
                                        nn.Sequential(
                                            GDN(N, inverse=True),
                                            deconv(N, 3, kernel_size=5, stride=1)
                                        )
                                    ])
        
        self.img_decoder = nn.Sequential(
            nn.Sequential(
                deconv(M, N, kernel_size=5, stride=2),
                ICT(channels=N, window_size=args.window_size, num_heads=args.num_heads, args=args)
            ),
            nn.Sequential(
                nn.Sequential(
                    GDN(N, inverse=True),
                    deconv(N, N, kernel_size=5, stride=2)
                ),
                ICT(channels=N, window_size=args.window_size, num_heads=args.num_heads, args=args)
            ),
            nn.Sequential(
                nn.Sequential(
                    GDN(N, inverse=True),
                    deconv(N, N, kernel_size=5, stride=2)
                ),
                ICT(channels=N, window_size=args.window_size, num_heads=args.num_heads, args=args)
            ),
            *img_decoder_end_net
        )

        if args.Unet_at_decoder:
            self.unet = UNet2(N, N)

        if not args.downsample_scale1_fusion:
            img_ref_encoder_start_net = nn.ModuleList([
                                            nn.Sequential(
                                                conv(3, N, kernel_size=5, stride=2),
                                                GDN(N)
                                            ),
                                        ])
        else:
            img_ref_encoder_start_net = nn.ModuleList([
                                            nn.Sequential(
                                                conv(3, N, kernel_size=5, stride=1),
                                                GDN(N)
                                            ),
                                            nn.Sequential(
                                                conv(N, N, kernel_size=5, stride=2),
                                                GDN(N)
                                            ),
                                        ])

        self.img_ref_encoder = nn.Sequential(
            *img_ref_encoder_start_net,
            nn.Sequential(
                conv(N, N, kernel_size=5, stride=2),
                GDN(N)
            ),
            nn.Sequential(
                conv(N, N, kernel_size=5, stride=2),
                GDN(N)
            ),
            conv(N, 2*M, kernel_size=5, stride=2)
        )

        if not self.args.mask_by_radius:
            img_mask_input_channel = 1
        elif not self.args.mask_by_radius_a_lot:
            img_mask_input_channel = 2
        else:
            img_mask_input_channel = 6

        mask_mid_feats = 2*N

        if not args.downsample_scale1_fusion:
            img_mask_encoder_start_net = nn.ModuleList([
                                            nn.Sequential(
                                                conv(img_mask_input_channel, mask_mid_feats, kernel_size=3, stride=2),
                                                nn.LeakyReLU(inplace=True)
                                            ),
                                        ])
        else:
            img_mask_encoder_start_net = nn.ModuleList([
                                            nn.Sequential(
                                                conv(img_mask_input_channel, mask_mid_feats, kernel_size=3, stride=1),
                                                nn.LeakyReLU(inplace=True)
                                            ),
                                            nn.Sequential(
                                                conv(mask_mid_feats, mask_mid_feats, kernel_size=3, stride=2),
                                                nn.LeakyReLU(inplace=True)
                                            ),
                                        ])

        self.img_mask_encoder = nn.Sequential(
            *img_mask_encoder_start_net,
            nn.Sequential(
                conv(mask_mid_feats, mask_mid_feats, kernel_size=3, stride=2),
                nn.LeakyReLU(inplace=True)
            ),
            nn.Sequential(
                conv(mask_mid_feats, mask_mid_feats, kernel_size=3, stride=2),
                nn.LeakyReLU(inplace=True)
            ),
            conv(mask_mid_feats, 6*M, kernel_size=3, stride=2)
        )

        self.dep_encoder = nn.Sequential(
            nn.Sequential(
                conv(1, N, kernel_size=5, stride=2),
                GDN(N)
            ),
            nn.Sequential(
                conv(2*N, N, kernel_size=5, stride=2),
                GDN(N)
            ),
            nn.Sequential(
                conv(2*N, N, kernel_size=5, stride=2),
                GDN(N)
            ),
            conv(2*N, M, kernel_size=5, stride=2)
        )

        self.dep_hyperprior = Hyperprior(in_planes=M, mid_planes=N, out_planes=M*2)
        self.dep_context_prediction = conv(
            M, M*2, kernel_size=5, stride=1
        )

        self.dep_entropy_parameters = Entropy_Parameters(M, 1)

        self.dep_gaussian_conditional = GaussianConditional(None)
        self.M = M
        
        self.dep_decoder = nn.Sequential(
            nn.Sequential(
                deconv(M, N, kernel_size=5, stride=2),
                GDN(N, inverse=True)
            ),
            nn.Sequential(
                deconv(2*N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True)
            ),
            nn.Sequential(
                deconv(2*N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True)
            ),
                deconv(2*N, 1, kernel_size=5, stride=2)
        )

        if args.dep_decoder_last_layer_double:
            self.dep_decoder[-1] = self.dep_decoder[-1].double()

        self.dep_pred_encoder = nn.Sequential(
            nn.Sequential(
                conv(1, N, kernel_size=5, stride=2),
                GDN(N)
            ),
            nn.Sequential(
                conv(N, N, kernel_size=5, stride=2),
                GDN(N)
            ),
            nn.Sequential(
                conv(N, N, kernel_size=5, stride=2),
                GDN(N)
            ),
            conv(N, 2*M, kernel_size=5, stride=2)
        )

        if not self.args.mask_by_radius:
            dep_mask_input_channel = 1
        elif not self.args.mask_by_radius_a_lot:
            dep_mask_input_channel = 2
        else:
            dep_mask_input_channel = 6

        self.dep_mask_encoder = nn.Sequential(
            nn.Sequential(
                conv(dep_mask_input_channel, 2*N, kernel_size=5, stride=2),
                nn.LeakyReLU(inplace=True)
            ),
            nn.Sequential(
                conv(2*N, 2*N, kernel_size=5, stride=2),
                nn.LeakyReLU(inplace=True)
            ),
            nn.Sequential(
                conv(2*N, 2*N, kernel_size=5, stride=2),
                nn.LeakyReLU(inplace=True)
            ),
            conv(2*N, 6*M, kernel_size=5, stride=2)
        )

        self.masks = {}

    def forward(self, view_main, view_ref):
        x_main, depth_main, K_main, w2vt_main = view_main['x_main'], view_main['depth_main'], view_main['K_main'], view_main['w2vt_main']
        x_main, pad_h, pad_w = pad_to_window_size(x_main, self.encoder_scale)
        depth_main, _, _ = pad_to_window_size(depth_main, self.encoder_scale)
        radius = 7.0

        if self.args.wo_reference_view:
            view_ref = {}

        if self.args.mask_by_radius or self.args.norm_by_radius:
            radius = view_main['radius']
            
        if view_ref:
            x_ref, x_ref_feats, depth_ref, K_ref, w2vt_ref = view_ref['x_ref'], view_ref['x_ref_feats'], view_ref['depth_ref'], view_ref['K_ref'], view_ref['w2vt_ref']
            x_ref, _, _ = pad_to_window_size(x_ref, self.encoder_scale)
            depth_ref, _, _ = pad_to_window_size(depth_ref, self.encoder_scale)

            if not self.args.wo_depth_pred:
                # depth coding preprocess
                depth_pred, depth_mask = depth_projection_batch(depth_ref, K_ref, K_main, w2vt_ref, w2vt_main
                                                                , mask_by_radius = self.args.mask_by_radius, radius = radius, args = self.args)
                
        if (not view_ref) or self.args.wo_depth_pred:
            depth_pred = torch.zeros_like(depth_main)
            if not self.args.mask_by_radius:
                depth_mask = torch.zeros_like(depth_main)
            elif not self.args.mask_by_radius_a_lot:
                depth_mask = torch.zeros_like(x_main[:, 0:2])
            else:
                depth_mask = torch.zeros_like(F.pad(x_main, (0, 0, 0, 0, 0, 3)))
            
        # depth coding
        depth_main_hat, depth_main_y_likelihoods, depth_main_z_likelihoods = self.depth_main_forward(depth_main, depth_pred, depth_mask)

        # image coding preprocess
        if view_ref:
            offset, image_mask = match(K_ref, w2vt_ref, depth_ref, K_main, w2vt_main, depth_main_hat
                                       , mask_by_radius = self.args.mask_by_radius, radius = radius, args = self.args)
            if self.args.detach_offset:
                offset = offset.detach()
            downsample_scales = [1, 2, 4, 8]
            offset_downsamples = downsample_offsets(offset, downsample_scales)
        else:
            x_ref = torch.zeros_like(x_main)
            if not self.args.mask_by_radius:
                image_mask = torch.zeros_like(x_main[:, 0:1])
            elif not self.args.mask_by_radius_a_lot:
                image_mask = torch.zeros_like(x_main[:, 0:2])
            else:
                image_mask = torch.zeros_like(F.pad(x_main, (0, 0, 0, 0, 0, 3)))
            x_ref_feats, offset_downsamples = None, None

        if self.args.wo_offset:
            offset_downsamples = None

        # image coding
        x_main_hat, x_dec_feats, x_main_y_likelihoods, x_main_z_likelihoods = self.x_main_forward(x_main, x_ref, x_ref_feats, offset_downsamples, image_mask)
        
        x_main_hat = x_main_hat[:, :, :x_main_hat.shape[2] - pad_h, :x_main_hat.shape[3] - pad_w]
        depth_main_hat = depth_main_hat[:, :, :depth_main_hat.shape[2] - pad_h, :depth_main_hat.shape[3] - pad_w]
        
        return {
            "x_hat": x_main_hat,
            "depth_hat": depth_main_hat,
            "x_likelihoods": {"y": x_main_y_likelihoods, "z": x_main_z_likelihoods},
            "depth_likelihoods": {"y": depth_main_y_likelihoods, "z": depth_main_z_likelihoods},
            "x_dec_feats": x_dec_feats,
        }
    
    def depth_main_forward(self, depth_main, depth_pred, depth_mask):
        depth_main_enc_mid, depth_pred_mid, depth_mask_mid = depth_main, depth_pred, depth_mask.to(depth_main.dtype)
        depth_pred_mid_before_gdn_list, depth_pred_mid_after_gdn_list, depth_mask_mid_list = [], [], []

        # depth encode
        for i in range(self.coder_layer_num):
            depth_main_enc_mid = self.dep_encoder[i](depth_main_enc_mid)

            if i < self.coder_layer_num - 1:
                depth_pred_mid = self.dep_pred_encoder[i][0](depth_pred_mid)
                depth_pred_mid_before_gdn_list.append(depth_pred_mid)
                depth_pred_mid = self.dep_pred_encoder[i][1](depth_pred_mid)
            else:
                depth_pred_mid = self.dep_pred_encoder[i](depth_pred_mid)

            depth_mask_mid = self.dep_mask_encoder[i](depth_mask_mid)
            if i < self.coder_layer_num - 1:
                depth_main_enc_mid = (torch.cat((depth_main_enc_mid, depth_pred_mid), 1) * (torch.sigmoid(depth_mask_mid) if self.args.sigmoid_mask else depth_mask_mid)) if not self.args.wo_dep_mask else torch.cat((depth_main_enc_mid, depth_pred_mid), 1)
            depth_pred_mid_after_gdn_list.append(depth_pred_mid)
            depth_mask_mid_list.append(depth_mask_mid)

        # depth entropy model
        depth_main_y = depth_main_enc_mid
        depth_main_y_hyper_params, depth_main_z_likelihoods, depth_main_z_hat = self.dep_hyperprior(depth_main_y, out_z=True)
        depth_main_y_hat = self.dep_gaussian_conditional.quantize(
            depth_main_y, "noise" if self.training else "dequantize"
        )

        depth_main_y_hyper_params_mask, depth_pred_params_mask, ctx_depth_main_y_hat_params_mask = (torch.sigmoid(depth_mask_mid_list[-1]) if self.args.sigmoid_mask else depth_mask_mid_list[-1]).chunk(3, 1)
        depth_main_y_hyper_params = (depth_main_y_hyper_params * depth_main_y_hyper_params_mask) if not self.args.wo_dep_mask else depth_main_y_hyper_params
        depth_pred_params = (depth_pred_mid_after_gdn_list[-1] * depth_pred_params_mask) if not self.args.wo_dep_mask else depth_pred_mid_after_gdn_list[-1]
        common_params = torch.cat((depth_main_y_hyper_params, depth_pred_params), 1)
        depth_main_y_means_hat, depth_main_y_scales_hat = self.forward_four_part_prior(depth_main_y_hat, common_params
                                                                                       , self.dep_context_prediction, self.dep_entropy_parameters
                                                                                       , curr_ctx_mask = ctx_depth_main_y_hat_params_mask, mode = 'depth')
        _, depth_main_y_likelihoods = self.dep_gaussian_conditional(depth_main_y, depth_main_y_scales_hat, means=depth_main_y_means_hat)
        depth_main_y_ste = quantize_ste(depth_main_y - depth_main_y_means_hat) + depth_main_y_means_hat
        
        # depth decode
        depth_main_dec_mid = depth_main_y_ste
        for i in range(self.coder_layer_num):
            depth_main_dec_mid = self.dep_decoder[i](depth_main_dec_mid)
            if i < self.coder_layer_num - 1:
                depth_pred_mid = depth_pred_mid_before_gdn_list[-i-1]
                depth_mask_mid = depth_mask_mid_list[-i-2]
                depth_main_dec_mid = (torch.cat((depth_main_dec_mid, depth_pred_mid), 1) * (torch.sigmoid(depth_mask_mid) if self.args.sigmoid_mask else depth_mask_mid)) if not self.args.wo_dep_mask else torch.cat((depth_main_dec_mid, depth_pred_mid), 1)
        depth_main_hat = depth_main_dec_mid
        return depth_main_hat, depth_main_y_likelihoods, depth_main_z_likelihoods
    
    def x_main_forward(self, x_main, x_ref, x_ref_feats, offset_downsamples, image_mask):
        # image encode
        downsample_scales = [2, 4, 8]
        coder_layer_num = self.coder_layer_num
        x_main_enc_mid, x_ref_mid, image_mask_mid = x_main, x_ref, image_mask.to(x_main.dtype)
        x_ref_aligned_feats, image_masks_mid = {}, {}

        if self.args.image_domain_align:
            offset = offset_downsamples['downsample_scale_1'] if offset_downsamples is not None else None
            x_align = torch_warp_simple(x_ref, offset) if offset is not None else x_ref
            x_main_enc_mid = torch.cat((x_main_enc_mid, x_align, x_main_enc_mid - x_align), 1)

        if self.args.downsample_scale1_fusion:
            downsample_scales = [1, 2, 4, 8]
            coder_layer_num = coder_layer_num + 1

        for i in range(coder_layer_num):
            image_mask_mid = self.img_mask_encoder[i](image_mask_mid)

            if i < coder_layer_num - 1:
                downsample_scale = downsample_scales[i]
                key = f'downsample_scale_{downsample_scale}'
                offset_downsample = offset_downsamples[key] if offset_downsamples is not None else None
                feature_downsample = x_ref_feats[key] if x_ref_feats is not None else None
                
                x_ref_mid = self.img_ref_encoder[i](x_ref_mid)
                x_ref_mid = (feature_downsample + x_ref_mid) if feature_downsample is not None else x_ref_mid
                x_ref_aligned_feats[key] = torch_warp_simple(x_ref_mid, offset_downsample) if offset_downsample is not None else x_ref_mid

                x_main_enc_mid = self.img_encoder[i][0](x_main_enc_mid)
                x_main_enc_mid = self.img_encoder[i][1](x_main_enc_mid, x_ref_aligned_feats[key], image_mask_mid)
            else:
                downsample_scale = 16
                key = f'downsample_scale_{downsample_scale}'
                
                x_ref_aligned_feats[key] = self.img_ref_encoder[i](x_ref_aligned_feats['downsample_scale_8'])
                x_main_enc_mid = self.img_encoder[i](x_main_enc_mid)

            image_masks_mid[key] = image_mask_mid

        # image entropy model
        x_main_y = x_main_enc_mid
        x_main_y_hyper_params, x_main_z_likelihoods, x_main_z_hat = self.img_hyperprior(x_main_y, out_z=True)
        x_main_y_hat = self.img_gaussian_conditional.quantize(
            x_main_y, "noise" if self.training else "dequantize"
        )

        x_main_y_hyper_params_mask, x_ref_params_mask, ctx_x_main_y_hat_params_mask = (torch.sigmoid(image_masks_mid[key]) if self.args.sigmoid_mask else image_masks_mid[key]).chunk(3, 1)
        x_main_y_hyper_params = (x_main_y_hyper_params * x_main_y_hyper_params_mask) if not self.args.wo_img_mask else x_main_y_hyper_params
        x_ref_params = (x_ref_aligned_feats[key] * x_ref_params_mask) if not self.args.wo_img_mask else x_ref_aligned_feats[key]
        common_params = torch.cat((x_main_y_hyper_params, x_ref_params), 1)
        x_main_y_means_hat, x_main_y_scales_hat = self.forward_four_part_prior(x_main_y_hat, common_params
                                                                                , self.img_context_prediction, self.img_entropy_parameters
                                                                                , curr_ctx_mask = ctx_x_main_y_hat_params_mask, mode='image')
        _, x_main_y_likelihoods = self.img_gaussian_conditional(x_main_y, x_main_y_scales_hat, means=x_main_y_means_hat)
        x_main_y_ste = quantize_ste(x_main_y - x_main_y_means_hat) + x_main_y_means_hat

        # image decode
        x_dec_feats = {}
        x_main_dec_mid = x_main_y_ste

        for i in range(coder_layer_num):
            if i < coder_layer_num - 1:
                x_main_dec_mid = self.img_decoder[i][0](x_main_dec_mid)
                downsample_scale = downsample_scales[-i-1]
                key = f'downsample_scale_{downsample_scale}'
                x_main_dec_mid = self.img_decoder[i][1](x_main_dec_mid, x_ref_aligned_feats[key], image_masks_mid[key])

                if i==2 and self.args.Unet_at_decoder:
                    x_main_dec_mid = self.unet(x_main_dec_mid)

                x_dec_feats[key] = x_main_dec_mid
            else:
                x_main_dec_mid = self.img_decoder[i](x_main_dec_mid)

        x_main_hat = x_main_dec_mid

        return x_main_hat, x_dec_feats, x_main_y_likelihoods, x_main_z_likelihoods
    
    @staticmethod
    def get_one_channel_four_parts_mask(height, width, dtype, device):
        micro_masks = [
            torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device),
            torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device),
            torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device),
            torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
        ]

        masks = []
        for micro_mask in micro_masks:
            mask = micro_mask.repeat((height + 1) // 2, (width + 1) // 2)
            mask = mask[:height, :width]
            mask = torch.unsqueeze(mask, 0)
            mask = torch.unsqueeze(mask, 0)
            masks.append(mask)

        return tuple(masks)
    
    def get_mask_four_parts(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}x{width}x{height}"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0, m1, m2, m3 = self.get_one_channel_four_parts_mask(height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]
    
    def forward_four_part_prior(self, y_hat, common_params, context_prediction, entropy_parameters, curr_ctx_mask=None, mode='image'):
        dtype = y_hat.dtype
        device = y_hat.device
        B, C, H, W = y_hat.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)
        common_params_padded = F.pad(common_params, (0, 0, 0, 0, 0, 2*self.M), "constant", 0)

        gaussian_params = entropy_parameters(common_params_padded)
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        y_hat_0 = y_hat * mask_0
        means_hat_0 = means_hat * mask_0
        scales_hat_0 = scales_hat * mask_0

        y_hat_so_far = y_hat_0
        curr_ctx_params = context_prediction(y_hat_so_far)
        curr_ctx_params = (curr_ctx_params * curr_ctx_mask) if (curr_ctx_mask is not None and ((not self.args.wo_img_mask and mode=='image') or (not self.args.wo_dep_mask and mode=='depth'))) else curr_ctx_params
        common_params_padded1 = common_params_padded.clone()
        common_params_padded1[:, -2*self.M:] = curr_ctx_params
        gaussian_params = entropy_parameters(common_params_padded1)
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        y_hat_1 = y_hat * mask_1
        means_hat_1 = means_hat * mask_1
        scales_hat_1 = scales_hat * mask_1

        y_hat_so_far = y_hat_so_far + y_hat_1
        curr_ctx_params = context_prediction(y_hat_so_far)
        curr_ctx_params = (curr_ctx_params * curr_ctx_mask) if (curr_ctx_mask is not None and ((not self.args.wo_img_mask and mode=='image') or (not self.args.wo_dep_mask and mode=='depth'))) else curr_ctx_params
        common_params_padded2 = common_params_padded.clone()
        common_params_padded2[:, -2*self.M:] = curr_ctx_params
        gaussian_params = entropy_parameters(common_params_padded2)
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        y_hat_2 = y_hat * mask_2
        means_hat_2 = means_hat * mask_2
        scales_hat_2 = scales_hat * mask_2

        y_hat_so_far = y_hat_so_far + y_hat_2
        curr_ctx_params = context_prediction(y_hat_so_far)
        curr_ctx_params = (curr_ctx_params * curr_ctx_mask) if (curr_ctx_mask is not None and ((not self.args.wo_img_mask and mode=='image') or (not self.args.wo_dep_mask and mode=='depth'))) else curr_ctx_params
        common_params_padded3 = common_params_padded.clone()
        common_params_padded3[:, -2*self.M:] = curr_ctx_params
        gaussian_params = entropy_parameters(common_params_padded3)
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        y_hat_3 = y_hat * mask_3
        means_hat_3 = means_hat * mask_3
        scales_hat_3 = scales_hat * mask_3

        means_hat = means_hat_0 + means_hat_1 + means_hat_2 + means_hat_3
        scales_hat = scales_hat_0 + scales_hat_1 + scales_hat_2 + scales_hat_3

        return means_hat, scales_hat
    
    def compress(self, view_main, view_ref):
        x_main, depth_main, K_main, w2vt_main = view_main['x_main'], view_main['depth_main'], view_main['K_main'], view_main['w2vt_main']
        shape_dict = {}
        shape_dict['x_shape'] = x_main.size()[-2:]
        x_main, pad_h, pad_w = pad_to_window_size(x_main, self.encoder_scale)
        depth_main, _, _ = pad_to_window_size(depth_main, self.encoder_scale)
        radius = 7.0

        if self.args.wo_reference_view:
            view_ref = {}

        if self.args.mask_by_radius or self.args.norm_by_radius:
            radius = view_main['radius']

        if view_ref:
            x_ref, x_ref_feats, depth_ref, K_ref, w2vt_ref = view_ref['x_ref'], view_ref['x_ref_feats'], view_ref['depth_ref'], view_ref['K_ref'], view_ref['w2vt_ref']
            x_ref, _, _ = pad_to_window_size(x_ref, self.encoder_scale)
            depth_ref, _, _ = pad_to_window_size(depth_ref, self.encoder_scale)

            if not self.args.wo_depth_pred:
                # depth coding preprocess
                depth_pred, depth_mask = depth_projection_batch(depth_ref, K_ref, K_main, w2vt_ref, w2vt_main
                                                                , mask_by_radius = self.args.mask_by_radius, radius = radius, args = self.args)

        if (not view_ref) or self.args.wo_depth_pred:
            depth_pred = torch.zeros_like(depth_main)

            if not self.args.mask_by_radius:
                depth_mask = torch.zeros_like(depth_main)
            elif not self.args.mask_by_radius_a_lot:
                depth_mask = torch.zeros_like(x_main[:, 0:2])
            else:
                depth_mask = torch.zeros_like(F.pad(x_main, (0, 0, 0, 0, 0, 3)))

        #depth encode
        depth_main_strings, depth_main_y_hat, depth_pred_mid_before_gdn_list, depth_pred_mid_after_gdn_list, depth_mask_mid_list = self.depth_encode(depth_main, depth_pred, depth_mask)
        
        #depth decode
        depth_main_hat = self.depth_decode(depth_main_y_hat = depth_main_y_hat, depth_pred_mid_before_gdn_list = depth_pred_mid_before_gdn_list
                                           , depth_mask_mid_list = depth_mask_mid_list, shape_dict = shape_dict)

        # image coding preprocess
        if view_ref:
            offset, image_mask = match(K_ref, w2vt_ref, depth_ref, K_main, w2vt_main, depth_main_hat
                                       , mask_by_radius = self.args.mask_by_radius, radius = radius, args = self.args)
            downsample_scales = [1, 2, 4, 8]
            offset_downsamples = downsample_offsets(offset, downsample_scales)
        else:
            x_ref = torch.zeros_like(x_main)

            if not self.args.mask_by_radius:
                image_mask = torch.zeros_like(x_main[:, 0:1])
            elif not self.args.mask_by_radius_a_lot:
                image_mask = torch.zeros_like(x_main[:, 0:2])
            else:
                image_mask = torch.zeros_like(F.pad(x_main, (0, 0, 0, 0, 0, 3)))

            x_ref_feats, offset_downsamples = None, None

        if self.args.wo_offset:
            offset_downsamples = None

        # image coding
        x_main_strings, x_main_y_hat, x_ref_aligned_feats, image_mask_mid = self.image_encode(x_main, x_ref, x_ref_feats, offset_downsamples, image_mask)

        strings = {"strings": [x_main_strings, depth_main_strings]}
        x_main_y_hat = {"x_main_y_hat": x_main_y_hat}
        coding_results = {**strings, **shape_dict, **x_main_y_hat}
        return coding_results
    
    def decompress(self, view_main, view_ref, coding_results):
        K_main, w2vt_main = view_main['K_main'], view_main['w2vt_main']
        shape_dict = {}
        shape_dict['x_shape'] = coding_results['x_shape']
        x_main_strings, depth_main_strings = coding_results['strings']
        batch_size = len(x_main_strings[0][0])
        h, w = shape_dict['x_shape']
        x_pad_shape = self.get_padded_shape(h, w, self.encoder_scale)
        radius = 7.0

        if self.args.wo_reference_view:
            view_ref = {}

        if self.args.mask_by_radius or self.args.norm_by_radius:
            radius = view_main['radius']

        if view_ref:
            x_ref, x_ref_feats, depth_ref, K_ref, w2vt_ref = view_ref['x_ref'], view_ref['x_ref_feats'], view_ref['depth_ref'], view_ref['K_ref'], view_ref['w2vt_ref']
            x_ref, _, _ = pad_to_window_size(x_ref, self.encoder_scale)
            depth_ref, _, _ = pad_to_window_size(depth_ref, self.encoder_scale)

            if not self.args.wo_depth_pred:
                # depth coding preprocess
                depth_pred, depth_mask = depth_projection_batch(depth_ref, K_ref, K_main, w2vt_ref, w2vt_main
                                                                , mask_by_radius = self.args.mask_by_radius, radius = radius, args = self.args)

        if (not view_ref) or self.args.wo_depth_pred:
            depth_pred = torch.zeros((batch_size, 1, *x_pad_shape), dtype=K_main.dtype, device=K_main.device)
            if not self.args.mask_by_radius:
                depth_mask = torch.zeros((batch_size, 1, *x_pad_shape), dtype=K_main.dtype, device=K_main.device)
            elif not self.args.mask_by_radius_a_lot:
                depth_mask = torch.zeros((batch_size, 2, *x_pad_shape), dtype=K_main.dtype, device=K_main.device)
            else:
                depth_mask = torch.zeros((batch_size, 6, *x_pad_shape), dtype=K_main.dtype, device=K_main.device)

        #depth decode
        depth_main_hat = self.depth_decode(depth_main_strings = depth_main_strings, depth_pred = depth_pred
                                           , depth_mask = depth_mask, shape_dict = shape_dict)
        
        # image coding preprocess
        if view_ref:
            offset, image_mask = match(K_ref, w2vt_ref, depth_ref, K_main, w2vt_main, depth_main_hat
                                       , mask_by_radius = self.args.mask_by_radius, radius = radius, args = self.args)
            downsample_scales = [1, 2, 4, 8]
            offset_downsamples = downsample_offsets(offset, downsample_scales)
        else:
            x_ref = torch.zeros((batch_size, 3, *x_pad_shape), dtype=K_main.dtype, device=K_main.device)
            if not self.args.mask_by_radius:
                image_mask = torch.zeros((batch_size, 1, *x_pad_shape), dtype=K_main.dtype, device=K_main.device)
            elif not self.args.mask_by_radius_a_lot:
                image_mask = torch.zeros((batch_size, 2, *x_pad_shape), dtype=K_main.dtype, device=K_main.device)
            else:
                image_mask = torch.zeros((batch_size, 6, *x_pad_shape), dtype=K_main.dtype, device=K_main.device)
            x_ref_feats, offset_downsamples = None, None

        if self.args.wo_offset:
            offset_downsamples = None

        # image decode
        x_main_hat, x_dec_feats = self.image_decode(x_main_strings, x_ref, x_ref_feats, offset_downsamples, image_mask, shape_dict, coding_results)
        x_main_hat = x_main_hat[:, :, :h, :w]
        depth_main_hat = depth_main_hat[:, :, :h, :w]

        out = {
            "x_hat": x_main_hat,
            "depth_hat": depth_main_hat,
            "x_dec_feats": x_dec_feats,
        }
        return out
    
    def depth_encode(self, depth_main, depth_pred, depth_mask):
        depth_main_enc_mid, depth_pred_mid, depth_mask_mid = depth_main, depth_pred, depth_mask.to(depth_main.dtype)
        depth_pred_mid_before_gdn_list, depth_pred_mid_after_gdn_list, depth_mask_mid_list = [], [], []

        # depth encode
        for i in range(self.coder_layer_num):
            depth_main_enc_mid = self.dep_encoder[i](depth_main_enc_mid)

            if i < self.coder_layer_num - 1:
                depth_pred_mid = self.dep_pred_encoder[i][0](depth_pred_mid)
                depth_pred_mid_before_gdn_list.append(depth_pred_mid)
                depth_pred_mid = self.dep_pred_encoder[i][1](depth_pred_mid)
            else:
                depth_pred_mid = self.dep_pred_encoder[i](depth_pred_mid)

            depth_mask_mid = self.dep_mask_encoder[i](depth_mask_mid)
            if i < self.coder_layer_num - 1:
                depth_main_enc_mid = (torch.cat((depth_main_enc_mid, depth_pred_mid), 1) * (torch.sigmoid(depth_mask_mid) if self.args.sigmoid_mask else depth_mask_mid)) if not self.args.wo_dep_mask else torch.cat((depth_main_enc_mid, depth_pred_mid), 1)
            depth_pred_mid_after_gdn_list.append(depth_pred_mid)
            depth_mask_mid_list.append(depth_mask_mid)

        # depth entropy model
        depth_main_y = depth_main_enc_mid
        depth_main_y_hyper_params, depth_main_z_hat, depth_main_z_strings = self.dep_hyperprior.compress(depth_main_y)
        
        depth_main_y_hyper_params_mask, depth_pred_params_mask, ctx_depth_main_y_hat_params_mask = (torch.sigmoid(depth_mask_mid_list[-1]) if self.args.sigmoid_mask else depth_mask_mid_list[-1]).chunk(3, 1)
        depth_main_y_hyper_params = (depth_main_y_hyper_params * depth_main_y_hyper_params_mask) if not self.args.wo_dep_mask else depth_main_y_hyper_params
        depth_pred_params = (depth_pred_mid_after_gdn_list[-1] * depth_pred_params_mask) if not self.args.wo_dep_mask else depth_pred_mid_after_gdn_list[-1]
        common_params = torch.cat((depth_main_y_hyper_params, depth_pred_params), 1)
        depth_main_y_strings, depth_main_y_hat = self.compress_four_part_prior(depth_main_y, common_params
                                                                                       , self.dep_context_prediction, self.dep_entropy_parameters
                                                                                       , self.dep_gaussian_conditional, curr_ctx_mask = ctx_depth_main_y_hat_params_mask
                                                                                       , mode = 'depth')
        
        return [depth_main_y_strings, depth_main_z_strings], depth_main_y_hat, depth_pred_mid_before_gdn_list, depth_pred_mid_after_gdn_list, depth_mask_mid_list

    def depth_decode(self, depth_main_strings = None, depth_main_y_hat = None, depth_pred_mid_before_gdn_list = None
                     , depth_pred_mid_after_gdn_list = None, depth_mask_mid_list = None, shape_dict = None
                     , depth_pred = None, depth_mask = None):
        if (depth_pred_mid_after_gdn_list is None and depth_main_strings is not None) or depth_pred_mid_before_gdn_list is None:
            depth_pred_mid = depth_pred
            if depth_pred_mid_before_gdn_list is None:
                record_after = True
                depth_pred_mid_after_gdn_list = []
            if depth_pred_mid_before_gdn_list is None:
                record_before = True
                depth_pred_mid_before_gdn_list = []

            for i in range(self.coder_layer_num):
                if i < self.coder_layer_num - 1:
                    depth_pred_mid = self.dep_pred_encoder[i][0](depth_pred_mid)
                    if record_before: 
                        depth_pred_mid_before_gdn_list.append(depth_pred_mid)
                    depth_pred_mid = self.dep_pred_encoder[i][1](depth_pred_mid)
                else:
                    depth_pred_mid = self.dep_pred_encoder[i](depth_pred_mid)
                if (i == self.coder_layer_num - 1) and record_after:
                    depth_pred_mid_after_gdn_list.append(depth_pred_mid)

        if depth_mask_mid_list is None:
            depth_mask_mid = depth_mask.to(depth_pred.dtype)
            depth_mask_mid_list = []
            for i in range(self.coder_layer_num):
                depth_mask_mid = self.dep_mask_encoder[i](depth_mask_mid)
                depth_mask_mid_list.append(depth_mask_mid)

        if depth_main_strings is not None:
            assert isinstance(depth_main_strings, list) and len(depth_main_strings) == 2

            # FIXME: we don't respect the default entropy coder and directly call the
            # range ANS decoder
            x_shape = shape_dict['x_shape']
            y_shape, z_shape = self.calculate_shapes(x_shape)
            depth_main_y_hyper_params, depth_main_z_hat = self.dep_hyperprior.decompress(depth_main_strings[1], z_shape, y_shape = y_shape)

            depth_main_y_hyper_params_mask, depth_pred_params_mask, ctx_depth_main_y_hat_params_mask = (torch.sigmoid(depth_mask_mid_list[-1]) if self.args.sigmoid_mask else depth_mask_mid_list[-1]).chunk(3, 1)
            depth_main_y_hyper_params = (depth_main_y_hyper_params * depth_main_y_hyper_params_mask) if not self.args.wo_dep_mask else depth_main_y_hyper_params
            depth_pred_params = (depth_pred_mid_after_gdn_list[-1] * depth_pred_params_mask) if not self.args.wo_dep_mask else depth_pred_mid_after_gdn_list[-1]
            common_params = torch.cat((depth_main_y_hyper_params, depth_pred_params), 1)
            depth_main_y_hat = self.decompress_four_part_prior(depth_main_strings[0], common_params
                                                                , self.dep_context_prediction, self.dep_entropy_parameters
                                                                , self.dep_gaussian_conditional, y_shape
                                                                , curr_ctx_mask = ctx_depth_main_y_hat_params_mask
                                                                , mode = 'depth')
        
        # depth decode
        depth_main_dec_mid = depth_main_y_hat
        for i in range(self.coder_layer_num):
            if i < self.coder_layer_num - 1 or not self.args.dep_decoder_last_layer_double:
                depth_main_dec_mid = self.dep_decoder[i](depth_main_dec_mid)
            else:
                depth_main_dec_mid = depth_main_dec_mid.double()
                depth_main_dec_mid = self.dep_decoder[i](depth_main_dec_mid)
                depth_main_dec_mid = depth_main_dec_mid.float()
            if i < self.coder_layer_num - 1:
                depth_pred_mid = depth_pred_mid_before_gdn_list[-i-1]
                depth_mask_mid = depth_mask_mid_list[-i-2]
                depth_main_dec_mid = (torch.cat((depth_main_dec_mid, depth_pred_mid), 1) * (torch.sigmoid(depth_mask_mid) if self.args.sigmoid_mask else depth_mask_mid)) if not self.args.wo_dep_mask else torch.cat((depth_main_dec_mid, depth_pred_mid), 1)
        depth_main_hat = depth_main_dec_mid

        return depth_main_hat

    def image_encode(self, x_main, x_ref, x_ref_feats, offset_downsamples, image_mask):
        # image encode
        downsample_scales = [2, 4, 8]
        x_main_enc_mid, x_ref_mid, image_mask_mid = x_main, x_ref, image_mask.to(x_main.dtype)
        x_ref_aligned_feats, image_masks_mid = {}, {}

        if self.args.image_domain_align:
            offset = offset_downsamples['downsample_scale_1'] if offset_downsamples is not None else None
            x_align = torch_warp_simple(x_ref, offset) if offset is not None else x_ref
            x_main_enc_mid = torch.cat((x_main_enc_mid, x_align, x_main_enc_mid - x_align), 1)

        for i in range(self.coder_layer_num):
            image_mask_mid = self.img_mask_encoder[i](image_mask_mid)

            if i < self.coder_layer_num - 1:
                downsample_scale = downsample_scales[i]
                key = f'downsample_scale_{downsample_scale}'
                offset_downsample = offset_downsamples[key] if offset_downsamples is not None else None
                feature_downsample = x_ref_feats[key] if x_ref_feats is not None else None
                
                x_ref_mid = self.img_ref_encoder[i](x_ref_mid)
                x_ref_mid = (feature_downsample + x_ref_mid) if feature_downsample is not None else x_ref_mid
                x_ref_aligned_feats[key] = torch_warp_simple(x_ref_mid, offset_downsample) if offset_downsample is not None else x_ref_mid

                x_main_enc_mid = self.img_encoder[i][0](x_main_enc_mid)
                x_main_enc_mid = self.img_encoder[i][1](x_main_enc_mid, x_ref_aligned_feats[key], image_mask_mid)
            else:
                downsample_scale = 16
                key = f'downsample_scale_{downsample_scale}'
                
                x_ref_aligned_feats[key] = self.img_ref_encoder[i](x_ref_aligned_feats['downsample_scale_8'])
                x_main_enc_mid = self.img_encoder[i](x_main_enc_mid)

            image_masks_mid[key] = image_mask_mid
        
        x_main_y = x_main_enc_mid
        x_main_y_hyper_params, x_main_z_hat, x_main_z_strings = self.img_hyperprior.compress(x_main_y)

        x_main_y_hyper_params_mask, x_ref_params_mask, ctx_x_main_y_hat_params_mask = (torch.sigmoid(image_masks_mid[key]) if self.args.sigmoid_mask else image_masks_mid[key]).chunk(3, 1)
        x_main_y_hyper_params = (x_main_y_hyper_params * x_main_y_hyper_params_mask) if not self.args.wo_img_mask else x_main_y_hyper_params
        x_ref_params = (x_ref_aligned_feats[key] * x_ref_params_mask) if not self.args.wo_img_mask else x_ref_aligned_feats[key]
        common_params = torch.cat((x_main_y_hyper_params, x_ref_params), 1)
        x_main_y_strings, x_main_y_hat = self.compress_four_part_prior(x_main_y, common_params
                                                                                , self.img_context_prediction, self.img_entropy_parameters
                                                                                , self.img_gaussian_conditional, curr_ctx_mask = ctx_x_main_y_hat_params_mask
                                                                                , mode = 'image')
        
        return [x_main_y_strings, x_main_z_strings], x_main_y_hat, x_ref_aligned_feats, image_mask_mid

    def image_decode(self, x_main_strings, x_ref, x_ref_feats, offset_downsamples, image_mask, shape_dict, coding_results):
        # preprocess
        downsample_scales = [2, 4, 8]
        x_ref_mid, image_mask_mid = x_ref, image_mask.to(x_ref.dtype)
        x_ref_aligned_feats, image_masks_mid = {}, {}

        for i in range(self.coder_layer_num):
            image_mask_mid = self.img_mask_encoder[i](image_mask_mid)

            if i < self.coder_layer_num - 1:
                downsample_scale = downsample_scales[i]
                key = f'downsample_scale_{downsample_scale}'
                offset_downsample = offset_downsamples[key] if offset_downsamples is not None else None
                feature_downsample = x_ref_feats[key] if x_ref_feats is not None else None
                
                x_ref_mid = self.img_ref_encoder[i](x_ref_mid)
                x_ref_mid = (feature_downsample + x_ref_mid) if feature_downsample is not None else x_ref_mid
                x_ref_aligned_feats[key] = torch_warp_simple(x_ref_mid, offset_downsample) if offset_downsample is not None else x_ref_mid
            else:
                downsample_scale = 16
                key = f'downsample_scale_{downsample_scale}'
                
                x_ref_aligned_feats[key] = self.img_ref_encoder[i](x_ref_aligned_feats['downsample_scale_8'])

            image_masks_mid[key] = image_mask_mid
        
        # image entropy decode
        assert isinstance(x_main_strings, list) and len(x_main_strings) == 2

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder
        x_shape = shape_dict['x_shape']
        y_shape, z_shape = self.calculate_shapes(x_shape)
        x_main_y_hyper_params, x_main_z_hat = self.img_hyperprior.decompress(x_main_strings[1], z_shape, y_shape = y_shape)
        
        x_main_y_hyper_params_mask, x_ref_params_mask, ctx_x_main_y_hat_params_mask = (torch.sigmoid(image_masks_mid[key]) if self.args.sigmoid_mask else image_masks_mid[key]).chunk(3, 1)
        x_main_y_hyper_params = (x_main_y_hyper_params * x_main_y_hyper_params_mask) if not self.args.wo_img_mask else x_main_y_hyper_params
        x_ref_params = (x_ref_aligned_feats[key] * x_ref_params_mask) if not self.args.wo_img_mask else x_ref_aligned_feats[key]
        common_params = torch.cat((x_main_y_hyper_params, x_ref_params), 1)
        x_main_y_hat = self.decompress_four_part_prior(x_main_strings[0], common_params
                                                        , self.img_context_prediction, self.img_entropy_parameters
                                                        , self.img_gaussian_conditional, y_shape
                                                        , curr_ctx_mask = ctx_x_main_y_hat_params_mask, mode='image')
        # x_main_y_hat = coding_results["x_main_y_hat"]

        # image decode
        x_dec_feats = {}
        x_main_dec_mid = x_main_y_hat

        for i in range(self.coder_layer_num):
            if i < self.coder_layer_num - 1:
                x_main_dec_mid = self.img_decoder[i][0](x_main_dec_mid)
                downsample_scale = downsample_scales[-i-1]
                key = f'downsample_scale_{downsample_scale}'
                x_main_dec_mid = self.img_decoder[i][1](x_main_dec_mid, x_ref_aligned_feats[key], image_masks_mid[key])

                if i==2 and self.args.Unet_at_decoder:
                    x_main_dec_mid = self.unet(x_main_dec_mid)
                    
                x_dec_feats[key] = x_main_dec_mid
            else:
                x_main_dec_mid = self.img_decoder[i](x_main_dec_mid)

        x_main_hat = x_main_dec_mid.clamp_(0, 1)

        return x_main_hat, x_dec_feats
    
    @staticmethod
    def combine_for_writing(x):
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)
    
    def compress_four_part_prior(self, y, common_params, context_prediction, entropy_parameters, gaussian_conditional, curr_ctx_mask=None, mode='image'):
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)
        common_params = F.pad(common_params, (0, 0, 0, 0, 0, 2*self.M), "constant", 0)

        gaussian_params = entropy_parameters(common_params)
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        y_c_0 = self.combine_for_writing(y * mask_0)
        means_hat_c_0 = self.combine_for_writing(means_hat * mask_0)
        scales_hat_c_0 = self.combine_for_writing(scales_hat * mask_0)
        indexes_c_0 = gaussian_conditional.build_indexes(scales_hat_c_0)
        y_c_0_strings = gaussian_conditional.compress(y_c_0, indexes_c_0, means_hat_c_0)
        
        y_0_hat = (torch.round(y - means_hat) + means_hat) * mask_0
        
        # y_c_0_hat = gaussian_conditional.decompress(y_c_0_strings, indexes_c_0, means=means_hat_c_0)
        # y_0_hat = torch.cat((y_c_0_hat, y_c_0_hat, y_c_0_hat, y_c_0_hat), dim=1) * mask_0

        y_hat_so_far = y_0_hat
        curr_ctx_params = context_prediction(y_hat_so_far)
        curr_ctx_params = (curr_ctx_params * curr_ctx_mask) if (curr_ctx_mask is not None and ((not self.args.wo_img_mask and mode=='image') or (not self.args.wo_dep_mask and mode=='depth'))) else curr_ctx_params
        common_params[:, -2*self.M:] = curr_ctx_params
        gaussian_params = entropy_parameters(common_params)
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        y_c_1 = self.combine_for_writing(y * mask_1)
        means_hat_c_1 = self.combine_for_writing(means_hat * mask_1)
        scales_hat_c_1 = self.combine_for_writing(scales_hat * mask_1)

        indexes_c_1 = gaussian_conditional.build_indexes(scales_hat_c_1)
        y_c_1_strings = gaussian_conditional.compress(y_c_1, indexes_c_1, means_hat_c_1)
        y_1_hat = (torch.round(y - means_hat) + means_hat) * mask_1

        # y_c_1_hat = gaussian_conditional.decompress(y_c_1_strings, indexes_c_1, means=means_hat_c_1)
        # y_1_hat = torch.cat((y_c_1_hat, y_c_1_hat, y_c_1_hat, y_c_1_hat), dim=1) * mask_1

        y_hat_so_far = y_hat_so_far + y_1_hat
        curr_ctx_params = context_prediction(y_hat_so_far)
        curr_ctx_params = (curr_ctx_params * curr_ctx_mask) if (curr_ctx_mask is not None and ((not self.args.wo_img_mask and mode=='image') or (not self.args.wo_dep_mask and mode=='depth'))) else curr_ctx_params
        common_params[:, -2*self.M:] = curr_ctx_params
        gaussian_params = entropy_parameters(common_params)
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        y_c_2 = self.combine_for_writing(y * mask_2)
        means_hat_c_2 = self.combine_for_writing(means_hat * mask_2)
        scales_hat_c_2 = self.combine_for_writing(scales_hat * mask_2)

        indexes_c_2 = gaussian_conditional.build_indexes(scales_hat_c_2)
        y_c_2_strings = gaussian_conditional.compress(y_c_2, indexes_c_2, means_hat_c_2)
        y_2_hat = (torch.round(y - means_hat) + means_hat) * mask_2

        # y_c_2_hat = gaussian_conditional.decompress(y_c_2_strings, indexes_c_2, means=means_hat_c_2)
        # y_2_hat = torch.cat((y_c_2_hat, y_c_2_hat, y_c_2_hat, y_c_2_hat), dim=1) * mask_2

        y_hat_so_far = y_hat_so_far + y_2_hat
        curr_ctx_params = context_prediction(y_hat_so_far)
        curr_ctx_params = (curr_ctx_params * curr_ctx_mask) if (curr_ctx_mask is not None and ((not self.args.wo_img_mask and mode=='image') or (not self.args.wo_dep_mask and mode=='depth'))) else curr_ctx_params
        common_params[:, -2*self.M:] = curr_ctx_params
        gaussian_params = entropy_parameters(common_params)
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        y_c_3 = self.combine_for_writing(y * mask_3)
        means_hat_c_3 = self.combine_for_writing(means_hat * mask_3)
        scales_hat_c_3 = self.combine_for_writing(scales_hat * mask_3)

        indexes_c_3 = gaussian_conditional.build_indexes(scales_hat_c_3)
        y_c_3_strings = gaussian_conditional.compress(y_c_3, indexes_c_3, means_hat_c_3)
        y_3_hat = (torch.round(y - means_hat) + means_hat) * mask_3

        # y_c_3_hat = gaussian_conditional.decompress(y_c_3_strings, indexes_c_3, means=means_hat_c_3)
        # y_3_hat = torch.cat((y_c_3_hat, y_c_3_hat, y_c_3_hat, y_c_3_hat), dim=1) * mask_3

        y_hat = y_0_hat + y_1_hat + y_2_hat + y_3_hat

        return [y_c_0_strings, y_c_1_strings, y_c_2_strings, y_c_3_strings], y_hat
    
    def decompress_four_part_prior(self, y_c_strings, common_params, context_prediction, entropy_parameters, gaussian_conditional, y_shape, curr_ctx_mask=None, mode='image'):
        dtype = common_params.dtype
        device = common_params.device
        B, C, H, W = len(y_c_strings[0]), self.M, y_shape[0], y_shape[1]
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(B, C, H, W, dtype, device)
        common_params = F.pad(common_params, (0, 0, 0, 0, 0, 2*self.M), "constant", 0)

        gaussian_params = entropy_parameters(common_params)
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        means_hat_c_0 = self.combine_for_writing(means_hat * mask_0)
        scales_hat_c_0 = self.combine_for_writing(scales_hat * mask_0)
        
        indexes_c_0 = gaussian_conditional.build_indexes(scales_hat_c_0)
        y_c_0_hat = gaussian_conditional.decompress(y_c_strings[0], indexes_c_0, means=means_hat_c_0)

        # y_c_0_hat[torch.abs(y_c_0_hat) > 1000] = 0.0
        y_0_hat = torch.cat((y_c_0_hat, y_c_0_hat, y_c_0_hat, y_c_0_hat), dim=1) * mask_0

        y_hat_so_far = y_0_hat
        curr_ctx_params = context_prediction(y_hat_so_far)
        curr_ctx_params = (curr_ctx_params * curr_ctx_mask) if (curr_ctx_mask is not None and ((not self.args.wo_img_mask and mode=='image') or (not self.args.wo_dep_mask and mode=='depth'))) else curr_ctx_params
        common_params[:, -2*self.M:] = curr_ctx_params
        gaussian_params = entropy_parameters(common_params)
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        means_hat_c_1 = self.combine_for_writing(means_hat * mask_1)
        scales_hat_c_1 = self.combine_for_writing(scales_hat * mask_1)

        indexes_c_1 = gaussian_conditional.build_indexes(scales_hat_c_1)
        y_c_1_hat = gaussian_conditional.decompress(y_c_strings[1], indexes_c_1, means=means_hat_c_1)
        # y_c_1_hat[torch.abs(y_c_1_hat) > 1000] = 0.0
        y_1_hat = torch.cat((y_c_1_hat, y_c_1_hat, y_c_1_hat, y_c_1_hat), dim=1) * mask_1

        y_hat_so_far = y_hat_so_far + y_1_hat
        curr_ctx_params = context_prediction(y_hat_so_far)
        curr_ctx_params = (curr_ctx_params * curr_ctx_mask) if (curr_ctx_mask is not None and ((not self.args.wo_img_mask and mode=='image') or (not self.args.wo_dep_mask and mode=='depth'))) else curr_ctx_params
        common_params[:, -2*self.M:] = curr_ctx_params
        gaussian_params = entropy_parameters(common_params)
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        means_hat_c_2 = self.combine_for_writing(means_hat * mask_2)
        scales_hat_c_2 = self.combine_for_writing(scales_hat * mask_2)

        indexes_c_2 = gaussian_conditional.build_indexes(scales_hat_c_2)
        y_c_2_hat = gaussian_conditional.decompress(y_c_strings[2], indexes_c_2, means=means_hat_c_2)
        # y_c_2_hat[torch.abs(y_c_2_hat) > 1000] = 0.0
        y_2_hat = torch.cat((y_c_2_hat, y_c_2_hat, y_c_2_hat, y_c_2_hat), dim=1) * mask_2

        y_hat_so_far = y_hat_so_far + y_2_hat
        curr_ctx_params = context_prediction(y_hat_so_far)
        curr_ctx_params = (curr_ctx_params * curr_ctx_mask) if (curr_ctx_mask is not None and ((not self.args.wo_img_mask and mode=='image') or (not self.args.wo_dep_mask and mode=='depth'))) else curr_ctx_params
        common_params[:, -2*self.M:] = curr_ctx_params
        gaussian_params = entropy_parameters(common_params)
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        means_hat_c_3 = self.combine_for_writing(means_hat * mask_3)
        scales_hat_c_3 = self.combine_for_writing(scales_hat * mask_3)

        indexes_c_3 = gaussian_conditional.build_indexes(scales_hat_c_3)
        y_c_3_hat = gaussian_conditional.decompress(y_c_strings[3], indexes_c_3, means=means_hat_c_3)
        # y_c_3_hat[torch.abs(y_c_3_hat) > 1000] = 0.0
        y_3_hat = torch.cat((y_c_3_hat, y_c_3_hat, y_c_3_hat, y_c_3_hat), dim=1) * mask_3

        y_hat = y_0_hat + y_1_hat + y_2_hat + y_3_hat

        return y_hat

    def aux_loss(self):
        """Return a list of the auxiliary entropy bottleneck over module(s)."""
        aux_loss_list = []
        for m in self.modules():
            if isinstance(m, CompressionModel):
                aux_loss_list.append(m.aux_loss())
        return aux_loss_list

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.img_gaussian_conditional,
            "img_gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.img_hyperprior.entropy_bottleneck,
            "img_hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.dep_gaussian_conditional,
            "dep_gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        update_registered_buffers(
            self.dep_hyperprior.entropy_bottleneck,
            "dep_hyperprior.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.img_gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.img_hyperprior.entropy_bottleneck.update(force=force)
        updated |= self.dep_gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= self.dep_hyperprior.entropy_bottleneck.update(force=force)
        return updated

    def fix_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.hyperprior.parameters():
            p.requires_grad = False
        for p in self.context_prediction.parameters():
            p.requires_grad = False
        for p in self.entropy_parameters.parameters():
            p.requires_grad = False
        for p in self.gaussian_conditional.parameters():
            p.requires_grad = False
    
    def load_encoder(self, current_state_dict, checkpoint):
        encoder_dict = {k.replace("g_a", "encoder"): v for k, v in checkpoint.items() if "g_a" in k}
        context_prediction_dict = {k: v for k, v in checkpoint.items() if "context_prediction" in k}
        entropy_parameters_dict = {k: v for k, v in checkpoint.items() if "entropy_parameters" in k}
        gaussian_conditional_dict = {k: v for k, v in checkpoint.items() if "gaussian_conditional" in k}
        hyperprior_dict = {}
        for k, v in checkpoint.items():
            if "h_a" in k:
                hyperprior_dict[k.replace("h_a", "hyperprior.hyper_encoder")] = v
            elif "h_s" in k:
                hyperprior_dict[k.replace("h_s", "hyperprior.hyper_decoder")] = v
            elif "entropy_bottleneck" in k:
                hyperprior_dict[k.replace("entropy_bottleneck", "hyperprior.entropy_bottleneck")] = v

        current_state_dict.update(encoder_dict)
        current_state_dict.update(hyperprior_dict)
        current_state_dict.update(context_prediction_dict)
        current_state_dict.update(entropy_parameters_dict)
        current_state_dict.update(gaussian_conditional_dict)
        #print(current_state_dict.keys())
        #input()
        return current_state_dict
    
    def calculate_shapes(self, x_shape):
        def ceil_div(a, b):
            return math.ceil(a / b)

        h, w = x_shape
        y_height = ceil_div(h, self.encoder_scale)
        y_width = ceil_div(w, self.encoder_scale)
        
        z_height = ceil_div(y_height, self.hyper_encoder_scale)
        z_width = ceil_div(y_width, self.hyper_encoder_scale)
        
        return (y_height, y_width), (z_height, z_width)

    def get_padded_shape(self, h, w, scale):
        def get_next_divisible(value, divisor):
            return math.ceil(value / divisor) * divisor
        return get_next_divisible(h, scale), get_next_divisible(w, scale)