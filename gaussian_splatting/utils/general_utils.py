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
import pdb
import cv2
import math
import numpy as np

import torch
from piq import psnr, multi_scale_ssim
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()
    if silent:
        sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def result_analsis(img1, img2, img1_name, img2_name, result_file_path, psnrs=None, ms_ssims=None, time=None):
    if img1.dim()==3:
        img1 = img1.unsqueeze(0)
    if img2.dim()==3:
        img2 = img2.unsqueeze(0)
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    with open(result_file_path, 'a') as file:
        psnr_value = psnr(img1, img2, data_range=1.0)
        ms_ssim_value = multi_scale_ssim(img1, img2, data_range=1.0)
        file.write(img1_name + " and " + img2_name + f" PSNR: {psnr_value.item():.2f} dB\n")
        file.write(img1_name + " and " + img2_name + f" MS-SSIM: {ms_ssim_value.item():.4f}\n")
        if time:
            file.write("generate " + img1_name + f" Time: {time:.4f}\n")
    if psnrs is not None:
        psnrs.append(psnr_value.item())
    if ms_ssims is not None:
        ms_ssims.append(ms_ssim_value.item())
    return psnr_value.item(), ms_ssim_value.item()

'''def show_offset(view1, view2, offsets, output_file):
    view1 = view1.permute(1, 2, 0).cpu().numpy()
    view2 = view2.permute(1, 2, 0).cpu().numpy()
    height, width = view1.shape[:2]

    offsets_np = offsets.cpu().numpy()  # (H, W, 2)
    offsets_x, offsets_y = offsets_np[:, :, 0], offsets_np[:, :, 1]

    aspect_ratio = width / height
    fig_width = 15  # 指定宽度
    fig_height = fig_width / aspect_ratio

    fig, ax = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # 显示两张图片
    ax[0].imshow(view1)
    ax[0].set_title("View 1")
    ax[1].imshow(view2)
    ax[1].set_title("View 2")

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # 在两张图片之间绘制对应关系的连线
    for i in range(0, height, 100):  # 可以调整步长，选择需要绘制的像素
        for j in range(0, width, 100):
            dx = offsets_x[i, j]
            dy = offsets_y[i, j]
            x1, y1 = j + dx, i + dy
            x2, y2 = j, i
            ax[0].plot(x1, y1, 'ro')
            ax[1].plot(x2, y2, 'ro')
            con = plt.Line2D((x1, x2 + width), (y1, y2), color='cyan')
            fig.add_artist(con)

    plt.savefig(output_file)'''

def show_offset(view1, view2, offsets, output_file):
    # 转换 PyTorch 张量为 NumPy 数组并转换为 BGR 格式
    view1_np = (view1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
    view2_np = (view2.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
    height, width = view1_np.shape[:2]

    # 拼接两幅图像
    combined = np.concatenate((view1_np, view2_np), axis=1)

    # 偏移
    offsets_np = offsets.cpu().numpy()
    offsets_x, offsets_y = offsets_np[:, :, 0], offsets_np[:, :, 1]

    # 颜色列表
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    color_index = 0

    for i in range(0, height, 50):
        for j in range(0, width, 50):
            dx = round(offsets_x[i, j])
            dy = round(offsets_y[i, j])
            x1, y1 = j + width + dx, i + dy
            x2, y2 = j, i

            # 循环选择颜色
            color = colors[color_index]
            color_index = (color_index + 1) % len(colors)

            # 画线条，并设置透明度
            overlay = combined.copy()
            cv2.line(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, combined, 1 - alpha, 0, combined)

            # 在起点和终点添加小点
            cv2.circle(combined, (x1, y1), 3, color, -1)
            cv2.circle(combined, (x2, y2), 3, color, -1)

    # 保存图像
    cv2.imwrite(output_file, combined)

def generate_colors(num_colors):
    colors = []
    hsv_values = [(x / num_colors, 1.0, 1.0) for x in range(num_colors)]
    for hsv in hsv_values:
        hsv_array = np.uint8([[[int(hsv[0] * 180), int(hsv[1] * 255), int(hsv[2] * 255)]]])  # 0-180 for hue
        rgb = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in rgb))
    return colors

colors = [(255, 255, 0), (153, 0, 255), (255, 0, 204), (255, 0, 43), (0, 0, 255), (229, 0, 255), (255, 178, 0), (0, 153, 255)
          , (0, 230, 255), (179, 255, 0), (0, 255, 127), (17, 255, 0), (0, 255, 204), (76, 0, 255), (0, 255, 51), (255, 0, 128)
          , (102, 255, 0), (0, 76, 255), (255, 25, 0), (255, 102, 0)]

def show_offset2(view1, view2, offsets, output_file):
    # 将 PyTorch 张量转换为 NumPy 数组并转换为 BGR 格式
    view1_np = (view1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
    view2_np = (view2.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
    height, width = view1_np.shape[:2]

    # 添加高度差
    height_diff = 150  # 可以根据需求调整
    empty_space = np.full((height_diff, width, 3), 255, dtype=np.uint8)  # 用白色填充

    # 将两幅图像添加高度差并拼接
    view1_np = np.concatenate((view1_np, empty_space), axis=0)
    view2_np = np.concatenate((empty_space, view2_np), axis=0)
    combined = np.concatenate((view1_np, view2_np), axis=1)

    # 偏移
    offsets_np = offsets.cpu().numpy()
    offsets_x, offsets_y = offsets_np[:, :, 0], offsets_np[:, :, 1]

    # 颜色列表
    # colors = generate_colors(20)
    color_index = 0

    for i in range(0, height, 50):
        for j in range(0, width, 100):
            dx = round(offsets_x[i, j])
            dy = round(offsets_y[i, j])
            x1, y1 = j + dx, i + dy
            x2, y2 = j + width, i + height_diff

            # 循环选择颜色
            color = colors[color_index]
            color_index = (color_index + 1) % len(colors)

            # 画线条，并设置透明度
            overlay = combined.copy()
            cv2.line(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, combined, 1 - alpha, 0, combined)

            # 在起点和终点添加小点
            cv2.circle(combined, (x1, y1), 3, color, -1)
            cv2.circle(combined, (x2, y2), 3, color, -1)

    # 保存图像
    cv2.imwrite(output_file, combined)

# 示例用法
# show_offset(view1, view2, offsets, 'output.png')

    '''ords = [(301, 461), (323, 439), (336, 461), (370, 480), (398, 499), (439, 449), (432, 483)]

    for i, j in ords:
        dx = round(offsets_x[i, j])
        dy = round(offsets_y[i, j])
        x1, y1 = j + dx, i + dy
        x2, y2 = j + width, i + height_diff

        # 循环选择颜色
        color = colors[color_index]
        color_index = (color_index + 1) % len(colors)

        # 画线条，并设置透明度
        overlay = combined.copy()
        cv2.line(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, combined, 1 - alpha, 0, combined)

        # 在起点和终点添加小点
        cv2.circle(combined, (x1, y1), 3, color, -1)
        cv2.circle(combined, (x2, y2), 3, color, -1)'''
    
def process_keys(ckpt, prefix="optic_flow."):
    processed_ckpt = {}

    for key, value in ckpt.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            processed_ckpt[new_key] = value

    return processed_ckpt

def summary_result(psnrs, ms_ssims, name, summary_result_file_path, times=None):
     with open(summary_result_file_path, 'a') as file:
        psnr = np.mean(psnrs)
        ms_ssim = np.mean(ms_ssims)
        file.write(name + f" PSNR: {psnr:.2f} dB\n")
        file.write(name + f" MS-SSIM: {ms_ssim:.4f}\n")
        if times:
            time = np.mean(times)
            file.write(name + f" Time: {time:.4f}\n")

def image_tensor2numpy(img_tensor):
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_np

def image_numpy2tensor(img_np):
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).cuda().float() / 255
    return img_tensor