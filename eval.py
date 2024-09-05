import os
import pdb
import sys
import time
import random
import deepspeed
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from arguments_main.args import get_dataset_args, get_main_args
from datasets.datasets import MultiViewImageDataset
from utils_main.graphics_utils import depth_projection_batch, match
from utils_main.general_utils import get_output_folder, setup_logger, AverageMeter, clear_dict_and_empty_cache, detach_dict_values, save_checkpoint, calculate_coding_bpp, rename_keys, convert_to_double
from utils_main.training_utils import configure_optimizers, compute_aux_loss
from models.LMVIC_3D_GP import LMVIC_3D_GP
from models.metrics import MSE_Loss, MS_SSIM_Loss

@torch.no_grad()
def eval_model_entropy_coder(model, test_dataset, test_dataloader, criterion, logger):
    device = next(model.parameters()).device

    psnr = AverageMeter('psnr', ':.4e')
    msssim = AverageMeter('ms-ssim', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    enc_time = AverageMeter('enc_time', ':.4e')
    dec_time = AverageMeter('dec_time', ':.4e')
    if args.test_entropy_coding_time:
        enc_y_time = AverageMeter('enc_y_time', ':.4e')
        dec_y_time = AverageMeter('dec_y_time', ':.4e')

    loop = tqdm(test_dataloader)

    view_main, view_ref = {}, {}
    curr_num_in_dataset = 0
    # flops_profiler = deepspeed.profiling.flops_profiler.FlopsProfiler(model)

    for i, batch in enumerate(loop):
        curr_num_in_dataset += 1
        datas = [frame.to(device) for frame in batch]
        
        clear_dict_and_empty_cache(view_main)
        view_main = {}

        view_main['x_main'], view_main['depth_main'], view_main['K_main'], view_main['w2vt_main'], view_main['dataset_id'] = datas[0][:, 0], datas[1][:, 0], datas[2][:, 0], datas[3][:, 0], datas[4][:]
        if args.mask_by_radius or args.norm_by_radius:
            view_main['radius'] = datas[5][:]

        num_pixels = view_main['x_main'].size(0) * view_main['x_main'].size(2) * view_main['x_main'].size(3)
        dataset_name = test_dataset.dataset_list[view_main['dataset_id']]

        if view_ref and (not torch.equal(view_main['dataset_id'], view_ref['dataset_id'])):
            clear_dict_and_empty_cache(view_ref)
            view_ref = {}
            curr_num_in_dataset = 1

        image_name = test_dataset.datasets_images_name[dataset_name][curr_num_in_dataset - 1]

        start_time = time.time()
        coding_results = model.compress(view_main, view_ref)
        enc_time.update(time.time() - start_time)

        if args.test_entropy_coding_time:
            enc_y_time.update(coding_results["encode_y_time"])

        start_time = time.time()
        out_net = model.decompress(view_main, view_ref, coding_results)
        dec_time.update(time.time() - start_time)

        if args.test_entropy_coding_time:
            dec_y_time.update(out_net["decode_y_time"])

        clear_dict_and_empty_cache(view_ref)
        view_ref = {}
        view_ref['x_ref'], view_ref['x_ref_feats'], view_ref['depth_ref'] = out_net["x_hat"].clamp(0, 1), out_net["x_dec_feats"], out_net["depth_hat"]
        view_ref['K_ref'], view_ref['w2vt_ref'], view_ref['dataset_id'] = view_main['K_main'], view_main['w2vt_main'], view_main['dataset_id']
        if curr_num_in_dataset % args.reset_period == 0:
            view_ref['x_ref_feats'] = None

        if not args.ignore_rate_distortion_metrics:
            out_criterion = criterion(out_net, view_main, args, is_only_decode_test=True)

            psnr.update(out_criterion["psnr"].item())
            msssim.update(out_criterion["ms-ssim"].item())
            bpp = calculate_coding_bpp(coding_results["strings"], num_pixels)
            bpp_loss.update(bpp)

        loop.set_description('[{}/{}]'.format(i, len(test_dataloader)))
        tmp_out = {'Bpp':f'{bpp_loss.avg:.4g}', "PSNR": f'{psnr.avg:.4g}', "MS-SSIM": f'{msssim.avg:.4g}', "Enc Time": f'{enc_time.avg:.4g}', "Dec Time": f'{dec_time.avg:.4g}'}
        if args.test_entropy_coding_time:
            tmp_out_aux = {"Enc y Time": f'{enc_y_time.avg:.4g}', "Dec y Time": f'{dec_y_time.avg:.4g}'}
            tmp_out = {**tmp_out, **tmp_out_aux}
        loop.set_postfix(tmp_out)

        if not args.ignore_rate_distortion_metrics:
            logger.info(f'Dataset {dataset_name} Image {image_name} Bpp: {bpp:.4f} PSNR: {out_criterion["psnr"].item():.2f} MS-SSIM: {out_criterion["ms-ssim"].item():.4f} Enc Time: {enc_time.avg:.4f} Dec Time: {dec_time.avg:.4f}')

    out = {"bpp_loss": bpp_loss.avg, "PSNR": psnr.avg, "MS-SSIM": msssim.avg, "Enc Time": enc_time.avg, "Dec Time": dec_time.avg}
    if args.test_entropy_coding_time:
        out_aux = {"Enc y Time": enc_y_time.avg, "Dec y Time": dec_y_time.avg}
        out = {**out, **out_aux}
    
    logger.info(f"test_over ___ BPP: {out['bpp_loss']:.4f} ___ PSNR: {out['PSNR']:.2f} ___ MS-SSIM: {out['MS-SSIM']:.4f} ___ Enc Time: {out['Enc Time']:.4f} ___ Dec Time: {out['Dec Time']:.4f}")
    if args.test_entropy_coding_time:
        logger.info(f"Enc y Time: {out['Enc y Time']:.4f} ___ Dec y Time: {out['Dec y Time']:.4f}")

    return out

@torch.no_grad()
def eval_model_entropy_estimation(model, test_dataset, test_dataloader, criterion, logger):
    device = next(model.parameters()).device

    psnr = AverageMeter('psnr', ':.4e')
    msssim = AverageMeter('ms-ssim', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')

    loop = tqdm(test_dataloader)

    view_main, view_ref = {}, {}
    curr_num_in_dataset = 0

    for i, batch in enumerate(loop):
        curr_num_in_dataset += 1
        datas = [frame.to(device) for frame in batch]
        
        clear_dict_and_empty_cache(view_main)
        view_main = {}

        view_main['x_main'], view_main['depth_main'], view_main['K_main'], view_main['w2vt_main'], view_main['dataset_id'] = datas[0][:, 0], datas[1][:, 0], datas[2][:, 0], datas[3][:, 0], datas[4][:]
        if args.mask_by_radius or args.norm_by_radius:
            view_main['radius'] = datas[5][:]

        dataset_name = test_dataset.dataset_list[view_main['dataset_id']]

        if view_ref and (not torch.equal(view_main['dataset_id'], view_ref['dataset_id'])):
            clear_dict_and_empty_cache(view_ref)
            view_ref = {}
            curr_num_in_dataset = 1

        image_name = test_dataset.datasets_images_name[dataset_name][curr_num_in_dataset - 1]

        out_net = model(view_main, view_ref)

        clear_dict_and_empty_cache(view_ref)
        view_ref = {}
        view_ref['x_ref'], view_ref['x_ref_feats'], view_ref['depth_ref'] = out_net["x_hat"].clamp(0, 1), out_net["x_dec_feats"], out_net["depth_hat"]
        view_ref['K_ref'], view_ref['w2vt_ref'], view_ref['dataset_id'] = view_main['K_main'], view_main['w2vt_main'], view_main['dataset_id']
        if curr_num_in_dataset % args.reset_period == 0:
            view_ref['x_ref_feats'] = None
        
        out_criterion = criterion(out_net, view_main, args, is_test=True)

        psnr.update(out_criterion["psnr"].item())
        msssim.update(out_criterion["ms-ssim"].item())
        bpp_loss.update((out_criterion["bpp_loss"]).item())

        loop.set_description('[{}/{}]'.format(i, len(test_dataloader)))
        loop.set_postfix({'Bpp':bpp_loss.avg, "PSNR": psnr.avg, "MS-SSIM": msssim.avg})

        logger.info(f'Dataset {dataset_name} Image {image_name} Bpp: {(out_criterion["bpp_loss"]).item():.4f} PSNR: {out_criterion["psnr"].item():.2f} MS-SSIM: {out_criterion["ms-ssim"].item():.4f}')

    out = {"bpp_loss": bpp_loss.avg, "PSNR": psnr.avg, "MS-SSIM": msssim.avg}
    
    logger.info(f"test_over ___ BPP: {out['bpp_loss']:.4f} ___ PSNR: {out['PSNR']:.2f} ___ MS-SSIM: {out['MS-SSIM']:.4f}")

    return out

if __name__ == "__main__":
    dataset_args = get_dataset_args()
    args = get_main_args()

    if not args.save_dir:
        save_dir, experiment_id = get_output_folder(os.path.join(args.save_root, '{}/{}/{}/lamda{}/'.format(args.datasets_name, args.metric, args.model_name, int(args.lmbda))), 'train')
    else:
        save_dir = args.save_dir

    log_file_path = os.path.join(save_dir, args.test_log_file_name)
    logger = setup_logger(log_file_path)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    test_dataset = MultiViewImageDataset(dataset_args.data_root, dataset_args, is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"))

    if args.model_name == "LMVIC_3D_GP":
        net = LMVIC_3D_GP(args=args)

    net = net.to(device)

    if args.metric == "mse":
        criterion = MSE_Loss()
    else:
        criterion = MS_SSIM_Loss()

    if args.load_compress_model_path:
        print("Loading model:", args.load_compress_model_path)
        checkpoint = torch.load(args.load_compress_model_path, map_location=device)
        if args.rename_ckpt_key1:
            checkpoint["state_dict"] = rename_keys(checkpoint["state_dict"])
        net.load_state_dict(checkpoint["state_dict"])
        net.update(force=True)
        net.eval()

    if args.entropy_estimation:
        logger.info('Entropy Estimation')
        metrics = eval_model_entropy_estimation(net, test_dataset, test_dataloader, criterion, logger)
    elif args.entropy_coder:
        logger.info('Entropy Coder')
        metrics = eval_model_entropy_coder(net, test_dataset, test_dataloader, criterion, logger)