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
import math
import time
import numpy as np
import torch
from piq import psnr, multi_scale_ssim
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, result_analsis, show_offset, show_offset2, process_keys, summary_result, image_tensor2numpy, image_numpy2tensor
from utils.graphics_utils import fov2focal, build_K, match, torch_warp_simple, rgb_to_ycbcr444
from utils.patchmatching_utils import SI_Finder_at_Image_Domain
from utils.opticalflow_utils import ME_Spynet, SpyNet, PWC_Net, flowformerpp_forward
from utils.homography_utils import homography_align
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def align_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    if len(views)==0:
        return
    align_path = os.path.join(model_path, name, "ours_{}".format(iteration), "align")
    makedirs(align_path, exist_ok=True)
    iter_num = len(views) - 1
    cache = {}
    psnrs_wo_align = []
    ms_ssims_wo_align = []
    psnrs_w_align = []
    ms_ssims_w_align = []
    times_w_align = []
    if args.patch_matching:
        psnrs_w_align_w_pm = []
        ms_ssims_w_align_w_pm = []
        times_w_align_w_pm = []
    if args.optical_flow:
        psnrs_w_align_w_opf = []
        ms_ssims_w_align_w_opf = []
        times_w_align_w_opf = []
    if args.optical_flow_pretrained:
        psnrs_w_align_w_opf_pf = []
        ms_ssims_w_align_w_opf_pf = []
        times_w_align_w_opf_pf = []
    if args.optical_flow_pwc_pretrained:
        psnrs_w_align_w_opf_pwc_pf = []
        ms_ssims_w_align_w_opf_pwc_pf = []
        times_w_align_w_opf_pwc_pf = []
    if args.optical_flow_flowformerpp_pretrained:
        psnrs_w_align_w_opf_flowformerpp_pf = []
        ms_ssims_w_align_w_opf_flowformerpp_pf = []
        times_w_align_w_opf_flowformerpp_pf = []
    if args.gaussian_w_optical_flow_pwc_pretrained:
        psnrs_w_align_gaus_w_opf_pwc_pf = []
        ms_ssims_w_align_gaus_w_opf_pwc_pf = []
        times_w_align_gaus_w_opf_pwc_pf = []
    if args.patch_matching_w_optical_flow_pwc_pretrained:
        psnrs_w_align_pm_w_opf_pwc_pf = []
        ms_ssims_w_align_pm_w_opf_pwc_pf = []
        times_w_align_pm_w_opf_pwc_pf = []
    if args.homography:
        psnrs_w_align_w_homo = []
        ms_ssims_w_align_w_homo = []
        times_w_align_w_homo = []
    if args.homography_w_optical_flow_pwc_pretrained:
        psnrs_w_align_homo_w_opf_pwc_pf = []
        ms_ssims_w_align_homo_w_opf_pwc_pf = []
        times_w_align_homo_w_opf_pwc_pf = []

    for id in tqdm(range(iter_num)):
        start_align_time = time.time()
        # preprocess
        if len(cache) == 0:
            view1 = views[id]
            render_result1 = render(view1, gaussians, pipeline, background)
            depth1 = render_result1['depth']
            focal_x1 = fov2focal(view1.FoVx, view1.image_width)
            focal_y1 = fov2focal(view1.FoVy, view1.image_height)
            K1 = build_K(focal_x1, focal_y1, view1.image_width, view1.image_height)
            world_view_transform1 = view1.world_view_transform.transpose(0, 1)
        else:
            view1 = cache['view']
            depth1 = cache['depth']
            K1 = cache['K']
            world_view_transform1 = cache['world_view_transform']

        view2 = views[id+1]
        render_result2 = render(view2, gaussians, pipeline, background)
        depth2 = render_result2['depth']
        focal_x2 = fov2focal(view2.FoVx, view2.image_width)
        focal_y2 = fov2focal(view2.FoVy, view2.image_height)
        K2 = build_K(focal_x2, focal_y2, view2.image_width, view2.image_height)
        world_view_transform2 = view2.world_view_transform.transpose(0, 1)

        # match
        assert view1.image_width == view2.image_width and view1.image_height == view2.image_height
        offsets, mask = match(K1, world_view_transform1, depth1, K2, world_view_transform2, depth2, view2.image_width, view2.image_height)
        
        # align
        gt_img1 = view1.original_image[0:3, :, :].unsqueeze(0)
        gt_img1_warp2img2 = torch_warp_simple(gt_img1, offsets)
        times_w_align.append(time.time() - start_align_time)

        # save
        output_path = os.path.join(align_path, view1.image_name + "to" + view2.image_name)
        makedirs(output_path, exist_ok=True)
        gt_img1 = gt_img1[0]
        gt_img2 = view2.original_image[0:3, :, :]
        gt_img1_warp2img2 = gt_img1_warp2img2[0]
        mask = mask[0].unsqueeze(0).float()
        torchvision.utils.save_image(gt_img1, os.path.join(output_path, view1.image_name + "_gt.png"))
        torchvision.utils.save_image(gt_img2, os.path.join(output_path, view2.image_name + "_gt.png"))
        torchvision.utils.save_image(gt_img1_warp2img2, os.path.join(output_path, view1.image_name + "to" + view2.image_name + "_warp_image.png"))
        torchvision.utils.save_image(mask, os.path.join(output_path, view1.image_name + "to" + view2.image_name + "_warp_mask.png"))
        
        result_file_path = os.path.join(output_path, "result.txt")
        # if os.path.exists(result_file_path):
        #     os.remove(result_file_path)
        result_analsis(gt_img1, gt_img2, "gt view1", "gt view2", result_file_path, psnrs_wo_align, ms_ssims_wo_align)
        result_analsis(gt_img1_warp2img2, gt_img2, "gt view1 warp to view2", "gt view2", result_file_path
                                                    , psnrs_w_align, ms_ssims_w_align, times_w_align[-1])
        
        if args.offset_visualization:
            visualization_path = os.path.join(output_path, view2.image_name + "to" + view1.image_name + "_offset.png")
            offsets = offsets[0]
            show_offset2(gt_img1, gt_img2, offsets, visualization_path)

        if args.patch_matching:
            start_align_w_pm_time = time.time()
            gt_img1_warp2img2_w_pm = SI_Finder_at_Image_Domain(gt_img2.unsqueeze(0), gt_img1.unsqueeze(0), gt_img1.unsqueeze(0))
            times_w_align_w_pm.append(time.time() - start_align_w_pm_time)
            gt_img1_warp2img2_w_pm = gt_img1_warp2img2_w_pm[0]
            torchvision.utils.save_image(gt_img1_warp2img2_w_pm, os.path.join(output_path, view1.image_name + "to" + view2.image_name + "_warp_image_w_pm.png"))
            result_analsis(gt_img1_warp2img2_w_pm, gt_img2, "gt view1 warp to view2 with patching matching", "gt view2", result_file_path
                                                    , psnrs_w_align_w_pm, ms_ssims_w_align_w_pm, times_w_align_w_pm[-1])
    
        torch.cuda.empty_cache()
        if args.optical_flow:
            start_align_w_opf_time = time.time()
            optical_flow_net = args.optical_flow_net
            gt_img1_yuv, gt_img2_yuv = rgb_to_ycbcr444(gt_img1), rgb_to_ycbcr444(gt_img2)
            offsets = optical_flow_net(gt_img2_yuv.unsqueeze(0), gt_img1_yuv.unsqueeze(0))
            offsets = offsets.permute(0, 2, 3, 1)
            gt_img1_warp2img2_w_opf = torch_warp_simple(gt_img1.unsqueeze(0), offsets)
            times_w_align_w_opf.append(time.time() - start_align_w_opf_time)
            torchvision.utils.save_image(gt_img1_warp2img2_w_opf, os.path.join(output_path, view1.image_name + "to" + view2.image_name + "_warp_image_w_opf.png"))
            result_analsis(gt_img1_warp2img2_w_opf, gt_img2, "gt view1 warp to view2 with optical flow", "gt view2", result_file_path
                                                    , psnrs_w_align_w_opf, ms_ssims_w_align_w_opf, times_w_align_w_opf[-1])
            
        torch.cuda.empty_cache()
        if args.optical_flow_pretrained:
            start_align_w_opf_pf_time = time.time()
            optical_flow_pretrained_net = args.optical_flow_pretrained_net
            offsets = optical_flow_pretrained_net(gt_img2.unsqueeze(0), gt_img1.unsqueeze(0))
            offsets = offsets.permute(0, 2, 3, 1)
            gt_img1_warp2img2_w_opf_pt = torch_warp_simple(gt_img1.unsqueeze(0), offsets)
            times_w_align_w_opf_pf.append(time.time() - start_align_w_opf_pf_time)
            torchvision.utils.save_image(gt_img1_warp2img2_w_opf_pt, os.path.join(output_path, view1.image_name + "to" + view2.image_name + "_warp_image_w_opf_pt.png"))
            result_analsis(gt_img1_warp2img2_w_opf_pt, gt_img2, "gt view1 warp to view2 with optical flow pretrained", "gt view2", result_file_path
                                                    , psnrs_w_align_w_opf_pf, ms_ssims_w_align_w_opf_pf, times_w_align_w_opf_pf[-1])
        
        torch.cuda.empty_cache()
        if args.gaussian_w_optical_flow_pwc_pretrained:
            start_align_gaus_w_opf_pwc_pf_time = time.time()
            optical_flow_pwc_pretrained_net = args.optical_flow_pwc_pretrained_net
            offsets = optical_flow_pwc_pretrained_net(gt_img2.unsqueeze(0), gt_img1_warp2img2.unsqueeze(0))
            offsets = offsets.permute(0, 2, 3, 1)
            gt_img1_warp2img2_gaus_w_opf_pwc_pt = torch_warp_simple(gt_img1_warp2img2.unsqueeze(0), offsets)
            times_w_align_gaus_w_opf_pwc_pf.append(time.time() - start_align_gaus_w_opf_pwc_pf_time + times_w_align[-1])
            torchvision.utils.save_image(gt_img1_warp2img2_gaus_w_opf_pwc_pt
                                         , os.path.join(output_path, view1.image_name + "to" + view2.image_name + "_warp_image_gaus_w_opf_pwc_pt.png"))
            result_analsis(gt_img1_warp2img2_gaus_w_opf_pwc_pt, gt_img2, "gt view1 warp to view2 gaussian splatting with optical flow PWC-Net pretrained"
                           , "gt view2", result_file_path, psnrs_w_align_gaus_w_opf_pwc_pf, ms_ssims_w_align_gaus_w_opf_pwc_pf, times_w_align_gaus_w_opf_pwc_pf[-1])
        
        torch.cuda.empty_cache() 
        if args.patch_matching_w_optical_flow_pwc_pretrained:
            start_align_pm_w_opf_pwc_pf_time = time.time()
            optical_flow_pwc_pretrained_net = args.optical_flow_pwc_pretrained_net
            offsets = optical_flow_pwc_pretrained_net(gt_img2.unsqueeze(0), gt_img1_warp2img2_w_pm.unsqueeze(0))
            offsets = offsets.permute(0, 2, 3, 1)
            gt_img1_warp2img2_pm_w_opf_pwc_pt = torch_warp_simple(gt_img1_warp2img2_w_pm.unsqueeze(0), offsets)
            times_w_align_pm_w_opf_pwc_pf.append(time.time() - start_align_pm_w_opf_pwc_pf_time + times_w_align_w_pm[-1])
            torchvision.utils.save_image(gt_img1_warp2img2_pm_w_opf_pwc_pt
                                         , os.path.join(output_path, view1.image_name + "to" + view2.image_name + "_warp_image_pm_w_opf_pwc_pt.png"))
            result_analsis(gt_img1_warp2img2_pm_w_opf_pwc_pt, gt_img2, "gt view1 warp to view2 patch matching with optical flow PWC-Net pretrained"
                           , "gt view2", result_file_path, psnrs_w_align_pm_w_opf_pwc_pf, ms_ssims_w_align_pm_w_opf_pwc_pf, times_w_align_pm_w_opf_pwc_pf[-1])
        
        torch.cuda.empty_cache()
        if args.optical_flow_pwc_pretrained:
            start_align_w_opf_pwc_pf_time = time.time()
            optical_flow_pwc_pretrained_net = args.optical_flow_pwc_pretrained_net
            offsets = optical_flow_pwc_pretrained_net(gt_img2.unsqueeze(0), gt_img1.unsqueeze(0))
            offsets = offsets.permute(0, 2, 3, 1)
            gt_img1_warp2img2_w_opf_pwc_pt = torch_warp_simple(gt_img1.unsqueeze(0), offsets)
            times_w_align_w_opf_pwc_pf.append(time.time() - start_align_w_opf_pwc_pf_time)
            torchvision.utils.save_image(gt_img1_warp2img2_w_opf_pwc_pt
                                         , os.path.join(output_path, view1.image_name + "to" + view2.image_name + "_warp_image_w_opf_pwc_pt.png"))
            result_analsis(gt_img1_warp2img2_w_opf_pwc_pt, gt_img2, "gt view1 warp to view2 with optical flow PWC-Net pretrained", "gt view2", result_file_path
                                                    , psnrs_w_align_w_opf_pwc_pf, ms_ssims_w_align_w_opf_pwc_pf, times_w_align_w_opf_pwc_pf[-1])

        torch.cuda.empty_cache()
        if args.optical_flow_flowformerpp_pretrained:
            start_align_w_opf_flowformerpp_pf_time = time.time()
            optical_flow_flowformerpp_pretrained_net = args.optical_flow_flowformerpp_pretrained_net
            offsets = flowformerpp_forward(optical_flow_flowformerpp_pretrained_net, gt_img2.unsqueeze(0), gt_img1.unsqueeze(0))
            offsets = offsets.permute(0, 2, 3, 1)
            gt_img1_warp2img2_w_opf_flowformerpp_pt = torch_warp_simple(gt_img1.unsqueeze(0), offsets)
            times_w_align_w_opf_flowformerpp_pf.append(time.time() - start_align_w_opf_flowformerpp_pf_time)
            torchvision.utils.save_image(gt_img1_warp2img2_w_opf_flowformerpp_pt
                                         , os.path.join(output_path, view1.image_name + "to" + view2.image_name + "_warp_image_w_opf_flowformerpp_pt.png"))
            result_analsis(gt_img1_warp2img2_w_opf_flowformerpp_pt, gt_img2, "gt view1 warp to view2 with optical flow FlowFormer++ pretrained", "gt view2"
                            , result_file_path, psnrs_w_align_w_opf_flowformerpp_pf, ms_ssims_w_align_w_opf_flowformerpp_pf, times_w_align_w_opf_flowformerpp_pf[-1])
        
        if args.homography:
            start_align_w_homo_time = time.time()
            gt_img1_np, gt_img2_np = image_tensor2numpy(gt_img1), image_tensor2numpy(gt_img2)
            gt_img1_warp2img2_w_homo_np = homography_align(gt_img1_np, gt_img2_np)
            gt_img1_warp2img2_w_homo = image_numpy2tensor(gt_img1_warp2img2_w_homo_np)
            times_w_align_w_homo.append(time.time() - start_align_w_homo_time)
            torchvision.utils.save_image(gt_img1_warp2img2_w_homo
                                         , os.path.join(output_path, view1.image_name + "to" + view2.image_name + "_warp_image_w_homo.png"))
            result_analsis(gt_img1_warp2img2_w_homo, gt_img2, "gt view1 warp to view2 with homography transformation", "gt view2"
                            , result_file_path, psnrs_w_align_w_homo, ms_ssims_w_align_w_homo, times_w_align_w_homo[-1])
        
        torch.cuda.empty_cache()
        if args.homography_w_optical_flow_pwc_pretrained:
            start_align_homo_w_opf_pwc_pf_time = time.time()
            optical_flow_pwc_pretrained_net = args.optical_flow_pwc_pretrained_net
            offsets = optical_flow_pwc_pretrained_net(gt_img2.unsqueeze(0), gt_img1_warp2img2_w_homo.unsqueeze(0))
            offsets = offsets.permute(0, 2, 3, 1)
            gt_img1_warp2img2_homo_w_opf_pwc_pt = torch_warp_simple(gt_img1_warp2img2_w_homo.unsqueeze(0), offsets)
            times_w_align_homo_w_opf_pwc_pf.append(time.time() - start_align_homo_w_opf_pwc_pf_time + times_w_align_w_homo[-1])
            torchvision.utils.save_image(gt_img1_warp2img2_homo_w_opf_pwc_pt
                                         , os.path.join(output_path, view1.image_name + "to" + view2.image_name + "_warp_image_homo_w_opf_pwc_pt.png"))
            result_analsis(gt_img1_warp2img2_homo_w_opf_pwc_pt, gt_img2, "gt view1 warp to view2 homography transformation with optical flow PWC-Net pretrained"
                           , "gt view2", result_file_path, psnrs_w_align_homo_w_opf_pwc_pf, ms_ssims_w_align_homo_w_opf_pwc_pf, times_w_align_homo_w_opf_pwc_pf[-1])

        #postprocess
        cache['view'] = view2
        cache['depth'] = depth2
        cache['K'] = K2
        cache['world_view_transform'] = world_view_transform2
    
    summary_result_file_path = os.path.join(align_path, 'result.txt')
    # if os.path.exists(summary_result_file_path):
    #     os.remove(summary_result_file_path)
    summary_result(psnrs_wo_align, ms_ssims_wo_align, "wo align", summary_result_file_path)
    summary_result(psnrs_w_align, ms_ssims_w_align, "w align", summary_result_file_path, times_w_align)
    if args.patch_matching:
        summary_result(psnrs_w_align_w_pm, ms_ssims_w_align_w_pm, "w align w pm", summary_result_file_path, times_w_align_w_pm)
    if args.optical_flow:
        summary_result(psnrs_w_align_w_opf, ms_ssims_w_align_w_opf, "w align w opf", summary_result_file_path, times_w_align_w_opf)
    if args.optical_flow_pretrained:
        summary_result(psnrs_w_align_w_opf_pf, ms_ssims_w_align_w_opf_pf, "w align w opf pf", summary_result_file_path, times_w_align_w_opf_pf)
    if args.optical_flow_pwc_pretrained:
        summary_result(psnrs_w_align_w_opf_pwc_pf, ms_ssims_w_align_w_opf_pwc_pf, "w align w opf pwc pf", summary_result_file_path, times_w_align_w_opf_pwc_pf)
    if args.optical_flow_flowformerpp_pretrained:
        summary_result(psnrs_w_align_w_opf_flowformerpp_pf, ms_ssims_w_align_w_opf_flowformerpp_pf, "w align w opf flowformerpp pf"
                       , summary_result_file_path, times_w_align_w_opf_flowformerpp_pf)
    if args.gaussian_w_optical_flow_pwc_pretrained:
        summary_result(psnrs_w_align_gaus_w_opf_pwc_pf, ms_ssims_w_align_gaus_w_opf_pwc_pf, "w align gaus w opf pwc pf", summary_result_file_path
                       , times_w_align_gaus_w_opf_pwc_pf)
    if args.patch_matching_w_optical_flow_pwc_pretrained:
        summary_result(psnrs_w_align_pm_w_opf_pwc_pf, ms_ssims_w_align_pm_w_opf_pwc_pf, "w align pm w opf pwc pf", summary_result_file_path
                       , times_w_align_pm_w_opf_pwc_pf)
    if args.homography:
        summary_result(psnrs_w_align_w_homo, ms_ssims_w_align_w_homo, "w align w homo", summary_result_file_path
                       , times_w_align_w_homo)
    if args.homography_w_optical_flow_pwc_pretrained:
        summary_result(psnrs_w_align_homo_w_opf_pwc_pf, ms_ssims_w_align_homo_w_opf_pwc_pf, "w align homo w opf pwc pf", summary_result_file_path
                       , times_w_align_homo_w_opf_pwc_pf)

def align_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if args.optical_flow:
            optical_flow_net = ME_Spynet()
            ckpt = torch.load(args.optical_flow_model_path)
            ckpt = process_keys(ckpt, "optic_flow.")
            optical_flow_net.load_state_dict(ckpt)
            optical_flow_net.cuda()
            optical_flow_net.eval()
            args.optical_flow_net = optical_flow_net

        if args.optical_flow_pretrained:
            optical_flow_pretrained_net = SpyNet()
            ckpt = torch.load(args.optical_flow_pretrained_model_path)
            optical_flow_pretrained_net.load_state_dict(ckpt)
            optical_flow_pretrained_net.cuda()
            optical_flow_pretrained_net.eval()
            args.optical_flow_pretrained_net = optical_flow_pretrained_net

        if args.optical_flow_pwc_pretrained or args.gaussian_w_optical_flow_pwc_pretrained or args.patch_matching_w_optical_flow_pwc_pretrained or args.homography_w_optical_flow_pwc_pretrained:
            optical_flow_pwc_pretrained_net = PWC_Net()
            ckpt = torch.load(args.optical_flow_pwc_pretrained_model_path)
            ckpt = ckpt.items()
            ckpt = {strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in ckpt}
            optical_flow_pwc_pretrained_net.load_state_dict(ckpt)
            optical_flow_pwc_pretrained_net.cuda()
            optical_flow_pwc_pretrained_net.eval()
            args.optical_flow_pwc_pretrained_net = optical_flow_pwc_pretrained_net

        if args.optical_flow_flowformerpp_pretrained:
            from FlowFormerPlusPlus.core.FlowFormer import build_flowformer
            from FlowFormerPlusPlus.configs.submissions import get_cfg as get_submission_cfg
            cfg = get_submission_cfg()
            optical_flow_flowformerpp_pretrained_net = torch.nn.DataParallel(build_flowformer(cfg))
            ckpt = torch.load(args.optical_flow_flowformerpp_pretrained_model_path)
            optical_flow_flowformerpp_pretrained_net.load_state_dict(ckpt)
            optical_flow_flowformerpp_pretrained_net.cuda()
            optical_flow_flowformerpp_pretrained_net.eval()
            args.optical_flow_flowformerpp_pretrained_net = optical_flow_flowformerpp_pretrained_net.module
            
        if not skip_train:
            name = "train_" + args.extra_name if args.extra_name else "train"
            align_set(dataset.model_path, name, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)

        if not skip_test:
            name = "test_" + args.extra_name if args.extra_name else "test"
            align_set(dataset.model_path, name, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--patch_matching", action="store_true")
    parser.add_argument("--optical_flow", action="store_true")
    parser.add_argument("--optical_flow_model_path", default=None, type=str)
    parser.add_argument("--optical_flow_pretrained", action="store_true")
    parser.add_argument("--optical_flow_pretrained_model_path", default=None, type=str)
    parser.add_argument("--optical_flow_pwc_pretrained", action="store_true")
    parser.add_argument("--optical_flow_pwc_pretrained_model_path", default=None, type=str)
    parser.add_argument("--optical_flow_flowformerpp_pretrained", action="store_true")
    parser.add_argument("--optical_flow_flowformerpp_pretrained_model_path", default=None, type=str)
    parser.add_argument("--gaussian_w_optical_flow_pwc_pretrained", action="store_true")
    parser.add_argument("--patch_matching_w_optical_flow_pwc_pretrained", action="store_true")
    parser.add_argument("--homography", action="store_true")
    parser.add_argument("--homography_w_optical_flow_pwc_pretrained", action="store_true")
    parser.add_argument("--offset_visualization", type=bool, default=True)
    parser.add_argument("--extra_name", default=None, type=str)
    pdb.set_trace()
    args = get_combined_args(parser)

    print("Aligning " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    align_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)