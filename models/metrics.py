import pdb
import math
from math import exp

from piq import psnr, multi_scale_ssim
from pytorch_msssim import ssim, ms_ssim

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def recursive_likelihoods_sum(likelihoods, num_pixels):
    if isinstance(likelihoods, torch.Tensor):
        return torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
    elif isinstance(likelihoods, list):
        return sum(recursive_likelihoods_sum(item, num_pixels) for item in likelihoods)
    elif isinstance(likelihoods, dict):
        return sum(recursive_likelihoods_sum(item, num_pixels) for item in likelihoods.values())
    else:
        return torch.tensor(0.0)


class MSE_Loss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, view_main, args, is_test = False, view_index = 0, is_only_decode_test = False):
        x_main, depth_main = view_main['x_main'], view_main['depth_main']
        x_main_hat, depth_main_hat = output['x_hat'], output['depth_hat']
        view_index = view_index % args.training_sample_length

        N, _, H, W = x_main.size()
        out = {}
        num_pixels = N * H * W

        if not is_only_decode_test:
            out['bpp_img_y'] = recursive_likelihoods_sum(output['x_likelihoods']['y'], num_pixels)
            out['bpp_img_z'] = recursive_likelihoods_sum(output['x_likelihoods']['z'], num_pixels)
            out['bpp_dep_y'] = recursive_likelihoods_sum(output['depth_likelihoods']['y'], num_pixels)
            out['bpp_dep_z'] = recursive_likelihoods_sum(output['depth_likelihoods']['z'], num_pixels)
            out["bpp_img_loss"] = out['bpp_img_y'] + out['bpp_img_z']
            out["bpp_dep_loss"] = out['bpp_dep_y'] + out['bpp_dep_z']
            out["bpp_loss"] = out["bpp_img_loss"] + out["bpp_dep_loss"]

            out["mse_img"] = self.mse(x_main_hat, x_main)

            if depth_main_hat is not None:
                out["mse_dep"] = self.mse(depth_main_hat, depth_main)
            else:
                out["mse_dep"] = torch.tensor(0.0)

            out['mse_loss'] = args.lmbda * out["mse_img"] + args.dep_lmbda * out["mse_dep"]

            if not is_test:
                out['loss'] = args.view_weights[view_index] * out['mse_loss'] + out['bpp_loss']
            else:
                out['loss'] = out['mse_loss'] + out['bpp_loss']

        if is_test or is_only_decode_test:
            x_main, x_main_hat = x_main.clamp(0, 1), x_main_hat.clamp(0, 1)
            out["psnr"] = psnr(x_main_hat, x_main, data_range=1.)
            out["ms-ssim"] = multi_scale_ssim(x_main_hat, x_main, data_range=1.)

        return out

class MS_SSIM_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, view_main, args, is_test = False, view_index = 0, is_only_decode_test = False):
        x_main, depth_main = view_main['x_main'], view_main['depth_main']
        x_main_hat, depth_main_hat = output['x_hat'], output['depth_hat']
        x_main, x_main_hat = x_main.clamp(0, 1), x_main_hat.clamp(0, 1)
        view_index = view_index % args.training_sample_length
        
        N, _, H, W = x_main.size()
        out = {}
        num_pixels = N * H * W

        if not is_only_decode_test:
            out['bpp_img_y'] = recursive_likelihoods_sum(output['x_likelihoods']['y'], num_pixels)
            out['bpp_img_z'] = recursive_likelihoods_sum(output['x_likelihoods']['z'], num_pixels)
            out['bpp_dep_y'] = recursive_likelihoods_sum(output['depth_likelihoods']['y'], num_pixels)
            out['bpp_dep_z'] = recursive_likelihoods_sum(output['depth_likelihoods']['z'], num_pixels)
            out["bpp_img_loss"] = out['bpp_img_y'] + out['bpp_img_z']
            out["bpp_dep_loss"] = out['bpp_dep_y'] + out['bpp_dep_z']
            out["bpp_loss"] = out["bpp_img_loss"] + out["bpp_dep_loss"]

            out["ms_ssim_loss_img"] = 1 - multi_scale_ssim(x_main_hat, x_main, data_range=1.)

            if depth_main_hat is not None:
                out["mse_dep"] = self.mse(depth_main_hat, depth_main)
            else:
                out["mse_dep"] = torch.tensor(0.0)

            if not is_test:
                out['loss'] = args.view_weights[view_index] * (args.lmbda * out["ms_ssim_loss_img"] + args.dep_lmbda * out["mse_dep"]) + out["bpp_loss"]
            else:
                out['loss'] = args.lmbda * out["ms_ssim_loss_img"] + args.dep_lmbda * out["mse_dep"] + out["bpp_loss"]

        if is_test or is_only_decode_test:
            out["psnr"] = psnr(x_main_hat, x_main, data_range=1.)
            out["ms-ssim"] = multi_scale_ssim(x_main_hat, x_main, data_range=1.)

        return out
    
class MSE_Loss_wo_depth(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, x_main, args, is_test = False, is_only_decode_test = False):
        x_main_hat = output['x_hat']

        N, _, H, W = x_main.size()
        out = {}
        num_pixels = N * H * W

        if not is_only_decode_test:

            if args.model_name != "SASIC":
                out['bpp_img_y'] = recursive_likelihoods_sum(output['likelihoods']['y'], num_pixels)
                out['bpp_img_z'] = recursive_likelihoods_sum(output['likelihoods']['z'], num_pixels)
            else:
                out['bpp_img_y'] = output['rate'].y.sum() / num_pixels
                out['bpp_img_z'] = output['rate'].z.sum() / num_pixels

            out["bpp_loss"] = out['bpp_img_y'] + out['bpp_img_z']

            out["mse_img"] = self.mse(x_main_hat, x_main)

            out['loss'] = args.lmbda * out['mse_img'] + out['bpp_loss']

        if is_test or is_only_decode_test:
            x_main, x_main_hat = x_main.clamp(0, 1), x_main_hat.clamp(0, 1)
            out["psnr"] = psnr(x_main_hat, x_main, data_range=1.)
            out["ms-ssim"] = multi_scale_ssim(x_main_hat, x_main, data_range=1.)

        return out
    
class MS_SSIM_Loss_wo_depth(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, x_main, args, is_test = False, is_only_decode_test = False):
        x_main_hat = output['x_hat']
        x_main, x_main_hat = x_main.clamp(0, 1), x_main_hat.clamp(0, 1)
        
        N, _, H, W = x_main.size()
        out = {}
        num_pixels = N * H * W

        if not is_only_decode_test:

            if args.model_name != "SASIC":
                out['bpp_img_y'] = recursive_likelihoods_sum(output['likelihoods']['y'], num_pixels)
                out['bpp_img_z'] = recursive_likelihoods_sum(output['likelihoods']['z'], num_pixels)
            else:
                out['bpp_img_y'] = output['rate'].y.sum() / num_pixels
                out['bpp_img_z'] = output['rate'].z.sum() / num_pixels

            out["bpp_loss"] = out['bpp_img_y'] + out['bpp_img_z']

            out["ms_ssim_loss_img"] = 1 - multi_scale_ssim(x_main_hat, x_main, data_range=1.)

            out['loss'] = args.lmbda * out["ms_ssim_loss_img"] + out["bpp_loss"]

        if is_test or is_only_decode_test:
            out["psnr"] = psnr(x_main_hat, x_main, data_range=1.)
            out["ms-ssim"] = multi_scale_ssim(x_main_hat, x_main, data_range=1.)

        return out