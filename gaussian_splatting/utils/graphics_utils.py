#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from typing import Tuple, Union

import pdb
import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import NamedTuple

YCBCR_WEIGHTS = {
    # Spec: (K_r, K_g, K_b) with K_g = 1 - K_r - K_b
    "ITU-R_BT.709": (0.2126, 0.7152, 0.0722)
}

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left) * (-z_sign)
    P[1, 2] = (top + bottom) / (top - bottom) * (-z_sign)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def build_K(focal_x, focal_y, width, height):
     K = np.zeros((3, 3), dtype=np.float32)
     K[0, 0] = focal_x
     K[0, 2] = width/2
     K[1, 1] = focal_y
     K[1, 2] = height/2
     K[2, 2] = 1
     return K

def depth_reshape(depth, mode="HWC"):
    if depth.dim() == 3:
        depth = depth.unsqueeze(0)
        if mode=="HWC":
            depth = depth.permute(0, 2, 3, 1)
    return depth

def match(K1, w2c_transform1, depth1, K2, w2c_transform2, depth2, width, height, record_occlusion=True):
    pixs = torch.tensor(np.concatenate(np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5, 1), -1)[None, :]).cuda().float()
    invK2 = torch.inverse(torch.tensor(K2).cuda().float())

    pixs_flat = pixs.reshape(-1, 3)
    rays = torch.matmul(pixs_flat, invK2.transpose(0, 1))
    rays = rays.reshape(1, height, width, 3)

    depth2 = depth_reshape(depth2)
    points3DC2 = rays * depth2
    points4DC2 = F.pad(points3DC2, pad=(0, 1), mode='constant', value=1)

    c2w_transform2 = torch.inverse(w2c_transform2)
    points4DC2_flat = points4DC2.reshape(-1, 4)
    points4DW_flat = torch.matmul(points4DC2_flat, c2w_transform2.transpose(0, 1))
    points4DC1_flat = torch.matmul(points4DW_flat, w2c_transform1.transpose(0, 1))
    points3DC1_flat = points4DC1_flat[:, 0:3] / points4DC1_flat[:, 3:4]

    #generate mask
    points3DC1 = points3DC1_flat.reshape(-1, height, width, 3)
    mask = 0.01 <= points3DC1[:, :, :, 2]

    K1 = torch.tensor(K1).cuda().float()
    corresponding_pixs_flat = torch.matmul(points3DC1_flat, K1.transpose(0, 1))
    corresponding_pixs_flat = corresponding_pixs_flat / corresponding_pixs_flat[:, 2:3]
    corresponding_pixs = corresponding_pixs_flat.reshape(-1, height, width, 3)
    mask *= (corresponding_pixs[:, :, :, 0] < (width + 1)) * (corresponding_pixs[:, :, :, 0] > -1)
    mask *= (corresponding_pixs[:, :, :, 1] < (height + 1)) * (corresponding_pixs[:, :, :, 1] > -1)
    offsets = (corresponding_pixs - pixs)[:, :, :, :2]
    
    if record_occlusion:
        depth1 = depth_reshape(depth1, mode="CHW")
        depth1_warp_to_view2 = torch_warp_simple(depth1, offsets)
        mask *= points3DC1[:, :, :, 2] < depth1_warp_to_view2[:, 0, :, :] * 1.05
    
    return offsets, mask

Backward_tensorGrid = {}
Backward_tensorGrid_cpu = {}

def torch_warp(tensorInput, tensorFlow):
    if tensorInput.device == torch.device('cpu'):
        if str(tensorFlow.size()) not in Backward_tensorGrid_cpu:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid_cpu[str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).cpu()

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid_cpu[str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=grid.permute(0, 2, 3, 1),
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=True)
    else:
        device_id = tensorInput.device.index
        if str(tensorFlow.size()) not in Backward_tensorGrid[device_id]:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid[device_id][str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).cuda().to(device_id)

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid[device_id][str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=grid.permute(0, 2, 3, 1),
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=True)
    
def torch_warp_simple(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
            1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), tensorFlow.size(1), -1, -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(1)).view(
            1, tensorFlow.size(1), 1, 1).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical], -1).cuda()

    tensorFlow = torch.cat([tensorFlow[:, :, :, 0:1] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, :, :, 1:2] / ((tensorInput.size(2) - 1.0) / 2.0)], -1)

    grid = (Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow)
    return torch.nn.functional.grid_sample(input=tensorInput,
                                            grid=grid,
                                            mode='bilinear',
                                            padding_mode='border',
                                            align_corners=True)


def flow_warp(im, flow):
    warp = torch_warp(im, flow)
    return warp

def rgb_to_ycbcr444(rgb):
    '''
    input is 3xhxw RGB torch.Tensor, in the range of [0, 1]
    output is y: 1xhxw, uv: 2xhxw, in the range of [0, 1]
    '''
    c, _, _ = rgb.shape
    assert c == 3

    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r, g, b = rgb[0:1, :, :], rgb[1:2, :, :], rgb[2:3, :, :]

    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5

    yuv = torch.cat([y, cb, cr], dim=0)
    yuv = torch.clamp(yuv, 0., 1.)
    return yuv

def ycbcr444_to_rgb(yuv):
    '''
    y is 1xhxw Y torch.Tensor, in the range of [0, 1]
    uv is 2xhxw UV torch.Tensor, in the range of [0, 1]
    return value is 3xhxw RGB torch.Tensor, in the range of [0, 1]
    '''
    y = yuv[0:1, :, :]
    cb = yuv[1:2, :, :]
    cr = yuv[2:3, :, :]

    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg

    rgb = torch.cat([r, g, b], dim=0)
    rgb = torch.clamp(rgb, 0., 1.)
    return rgb