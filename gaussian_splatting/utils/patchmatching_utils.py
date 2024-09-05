import pdb
import time
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from typing import NamedTuple

masks = {}

def create_gaussian_masks(img_h, img_w, patch_h, patch_w):
    #pdb.set_trace()
    """ Creates a set of gaussian maps, each gaussian centered in patch_x center """
    patch_area = patch_h * patch_w
    img_area = img_h * img_w
    num_patches = np.arange(0, img_area // patch_area)
    patch_img_w = img_w / patch_w
    w = np.arange(1, img_w+1, 1, float) - (patch_w % 2)/2
    h = np.arange(1, img_h+1, 1, float) - (patch_h % 2)/2
    h = h[:, np.newaxis]

    # mu = there is a gaussian map centered in each x_patch center
    center_h = (num_patches // patch_img_w + 0.5) * patch_h
    center_w = ((num_patches % patch_img_w) + 0.5) * patch_w

    # gaussian std
    sigma_h = 0.5 * img_h
    sigma_w = 0.5 * img_w

    # create the gaussian maps
    cols_gauss = (w - center_w[:, np.newaxis])[:, np.newaxis, :] ** 2 / sigma_w ** 2
    rows_gauss = np.transpose(h - center_h)[:,:, np.newaxis] ** 2 / sigma_h ** 2
    g = np.exp(-4 * np.log(2) * (rows_gauss + cols_gauss))

    # crop the masks to fit correlation map
    gauss_mask = g[:, (patch_h+1) // 2 - 1:img_h - patch_h // 2,
                (patch_w+1) // 2 - 1:img_w - patch_w // 2]

    return torch.from_numpy(gauss_mask.astype(np.float32)[np.newaxis,:,:,:]).cuda()  

def L2_or_pearson_corr(x, y, patch_h, patch_w, is_cpu=False):
    #pdb.set_trace()
    """This func calculate the Pearson Correlation Coefficient/L2 between a patch x and all patches in image y
    Formula: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    R =  numerator/ denominator.
    where:
    numerator = sum_i(xi*yi - y_mean*xi - x_mean*yi + x_mean*y_mean)
    denominator = sqrt( sum_i(xi^2 - 2xi*x_mean + x_mean^2)*sum_i(yi^2 - 2yi*y_mean + y_mean^2) )

    Input: tensor of patchs x and img y
    Output: map that each pixel in it, is Pearson correlation/L2 correlative for a patch between x and y
    """
    N, C, H, W = x.shape
    patch_size = int(H * W * C)
    
    weights = nn.Parameter(data=x.data, requires_grad=False)
    xy = F.conv2d(y, weights, padding=0, stride=1)

    kernel_mean = torch.ones(1, C, H, W).cuda()/patch_size
    weights = nn.Parameter(data=kernel_mean.data, requires_grad=False)
    y_mean = F.conv2d(y, weights, padding=0, stride=1)
    if is_cpu:
        x = x.cpu()
        y_mean = y_mean.cpu()
        xy = xy.cpu()

    x_sum = torch.sum(x, dim = [1, 2 , 3])
    y_mean_x = y_mean * x_sum.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    numerator = xy - y_mean_x

    sum_x_square = torch.sum(torch.square(x), axis=[1, 2, 3])
    x_mean = torch.mean(x, dim = [1, 2 , 3])
    x_mean_x_sum = x_mean*x_sum
    denominator_x = sum_x_square - x_mean_x_sum

    kernel_sum = torch.ones(1, C, H, W).cuda()
    weights = nn.Parameter(data=kernel_sum.data, requires_grad=False)
    sum_y_square = F.conv2d(torch.square(y), weights, padding=0, stride=1)
    if is_cpu:
        sum_y_square = sum_y_square.cpu()
    y_mean_y_sum = y_mean*y_mean*patch_size
    denominator_y = sum_y_square - y_mean_y_sum
    time.sleep(1e-4)
    del sum_y_square, y_mean_y_sum, sum_x_square, x_mean_x_sum, xy, y_mean_x, y_mean, kernel_sum, weights, x_mean, x_sum, kernel_mean
    torch.cuda.empty_cache() 
    denominator = denominator_y*denominator_x.unsqueeze(0).unsqueeze(2).unsqueeze(2)
    time.sleep(1e-4)
    del denominator_y, denominator_x
    torch.cuda.empty_cache() 
    time.sleep(1e-4)
    torch.cuda.empty_cache() 
    denominator = torch.clamp(denominator, min=0.0)
    out = numerator/(torch.sqrt(denominator) + 1e-4)

    if is_cpu:
        return out.cuda()
    else:
        return out
    
def SI_Wraper(cross_corr, patch_h, patch_w, patchs_num, y, k = 1, temperature=15, is_stack=False):

    _, _, corr_h, corr_w = cross_corr.shape
    _, C, feature_h, feature_w = y.shape
    cross_corr = cross_corr.reshape(1, -1, corr_h*corr_w)

    value, index = torch.topk(cross_corr, k, dim=2)
    value, index = value.squeeze(0), index.squeeze(0)
    value_stable = value - value.max(dim=1, keepdim=True)[0]
    weight = F.softmax(value_stable*temperature, dim=1)

    index_h, index_w = torch.div(index, corr_w, rounding_mode='floor'), index % corr_w
    patch_h_index, patch_w_index = torch.meshgrid(torch.arange(0, patch_h).cuda(), torch.arange(0, patch_w).cuda())
    index_h_to_patch, index_w_to_patch = index_h.unsqueeze(2).unsqueeze(2) + patch_h_index, index_w.unsqueeze(2).unsqueeze(2) + patch_w_index
    pixel_index = (index_h_to_patch * feature_w + index_w_to_patch).reshape(-1)
    y_patches = torch.index_select(y.reshape(-1, C, feature_h*feature_w), 2, pixel_index).reshape(-1, C, patchs_num, k, patch_h, patch_w)

    if is_stack:
        y_reference = y_patches.reshape(-1, C, feature_h//patch_h, feature_w//patch_w, k, patch_h, patch_w).permute(0, 4, 1, 2, 5, 3, 6).reshape(-1, k*C, feature_h, feature_w)
    else:
        y_patches = torch.sum(y_patches * weight.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1), 3)
        y_reference = y_patches.reshape(-1, C, feature_h//patch_h, feature_w//patch_w, patch_h, patch_w).permute(0, 1, 2, 4, 3, 5).reshape(-1, C, feature_h, feature_w)

    return y_reference 

def SI_Finder_at_Image_Domain(x_decs, y_decs, ys, patch_h=32, patch_w=32, args=None, is_pearson_corr_cpu=False):
    if x_decs.shape != y_decs.shape or x_decs.shape != ys.shape:
        raise ValueError("x_decs, y_decs, and ys must have the same shape.")
    N, C, H, W = x_decs.shape
    pad_h = (patch_h - (H % patch_h)) % patch_h
    pad_w = (patch_w - (W % patch_w)) % patch_w
    x_decs = F.pad(x_decs, (0, pad_w, 0, pad_h), mode='replicate')
    y_decs = F.pad(y_decs, (0, pad_w, 0, pad_h), mode='replicate')
    ys = F.pad(ys, (0, pad_w, 0, pad_h), mode='replicate')
    H_pad = H + pad_h
    W_pad = W + pad_w

    if (H_pad, W_pad) not in masks.keys():
        masks[(H_pad, W_pad)] = create_gaussian_masks(H_pad, W_pad, patch_h, patch_w)
    mask = masks[(H_pad, W_pad)]

    for n in range(N):
        y_reference = {}
        x_dec = x_decs[n].unsqueeze(0)
        y = ys[n].unsqueeze(0)
        y_dec = y_decs[n].unsqueeze(0)

        x_dec_patches = x_dec.reshape(1, C, H_pad//patch_h, patch_h, W_pad//patch_w, patch_w).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, patch_h, patch_w)
        patchs_num = x_dec_patches.shape[0]
        cross_corr = L2_or_pearson_corr(x_dec_patches, y_dec, patch_h, patch_w, is_cpu=is_pearson_corr_cpu)
        cross_corr = cross_corr * mask
        y_reference = SI_Wraper(cross_corr, patch_h, patch_w, patchs_num, y)
        if n == 0:
            y_references = y_reference
        else:
            y_references = torch.concat([y_references, y_reference], axis=0)

    y_references = y_references[:, :, :H, :W]
    return y_references