import os
import pdb
import sys
import random
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from arguments_main.args import get_dataset_args, get_main_args
from datasets.datasets import MultiViewImageDataset
from utils_main.graphics_utils import depth_projection_batch, match
from utils_main.general_utils import get_output_folder, setup_logger, AverageMeter, clear_dict_and_empty_cache, detach_dict_values, save_checkpoint
from utils_main.training_utils import configure_optimizers, compute_aux_loss
from models.LMVIC_3D_GP import LMVIC_3D_GP
from models.metrics import MSE_Loss, MS_SSIM_Loss

# torch.autograd.set_detect_anomaly(True)

def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, args, logger):
    model.train()
    device = next(model.parameters()).device
    if args.metric == "mse":
        metric_name = "mse_img" 
    else:
        metric_name = "ms_ssim_loss_img"

    metric_loss = AverageMeter(args.metric, ':.4e')
    dep_metric_loss = AverageMeter("Depth MSE", ':.4e') 

    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    img_bpp_loss = AverageMeter('Image BppLoss', ':.4e')
    dep_bpp_loss = AverageMeter('Depth BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')

    train_dataloader = tqdm(train_dataloader)
    logger.info(f'Train epoch: {epoch}')
    
    view_main, view_ref = {}, {}
    train_loss = 0
    train_loss_record = 0

    for i, batch in enumerate(train_dataloader):
        datas = [element.to(device) for element in batch]

        seq_length = datas[0].shape[1]
        for j in range(seq_length):
            # clear_dict_and_empty_cache(view_main)
            view_main = {}
            view_main['x_main'], view_main['depth_main'], view_main['K_main'], view_main['w2vt_main'] = datas[0][:, j], datas[1][:, j], datas[2][:, j], datas[3][:, j]

            if args.mask_by_radius or args.norm_by_radius:
                view_main['radius'] = datas[5][:]
                
            out_net = model(view_main, view_ref)

            # clear_dict_and_empty_cache(view_ref)
            view_ref = {}
            view_ref['x_ref'], view_ref['x_ref_feats'], view_ref['depth_ref'] = out_net["x_hat"].clamp(0, 1), out_net["x_dec_feats"], out_net["depth_hat"]
            view_ref['K_ref'], view_ref['w2vt_ref'] = view_main['K_main'], view_main['w2vt_main']
            if (j + 1)  % args.reset_period == 0:
                view_ref['x_ref_feats'] = None
            
            out_criterion = criterion(out_net, view_main, args, view_index = j)
            
            if aux_optimizer is not None:
                out_aux_loss = compute_aux_loss(model.aux_loss(), backward=True)
                aux_optimizer.step()
                aux_optimizer.zero_grad(set_to_none=True)
            else:
                out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)
            
            bpp_loss.update((out_criterion["bpp_loss"]).item())
            img_bpp_loss.update((out_criterion["bpp_img_loss"]).item())
            dep_bpp_loss.update((out_criterion["bpp_dep_loss"]).item())
            aux_loss.update(out_aux_loss.item())
            metric_loss.update(out_criterion[metric_name].item())
            dep_metric_loss.update(out_criterion["mse_dep"].item())

            train_loss = out_criterion["loss"] / args.training_sample_length
            if (j + 1) % args.training_sample_length != 0:
                train_loss.backward(retain_graph=True)
            else:
                train_loss.backward(retain_graph=True)

            train_loss_record += train_loss.item()

            if (j + 1) % args.training_sample_length == 0:
                detach_dict_values(view_ref)

                # train_loss = train_loss / args.training_sample_length
                # train_loss.backward(retain_graph=True)

                if clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                loss.update(train_loss_record)
                train_loss_record = 0

        train_dataloader.set_description('[{}/{}]'.format(i, len(train_dataloader)))
        train_dataloader.set_postfix({"Loss":loss.avg, "Bpp":bpp_loss.avg, args.metric: metric_loss.avg, "Aux":aux_loss.avg
                                      , "mse_dep":dep_metric_loss.avg, "Image Bpp":img_bpp_loss.avg, "Depth Bpp":dep_bpp_loss.avg})
        
        # clear_dict_and_empty_cache(view_main)
        # clear_dict_and_empty_cache(view_ref)
        view_main, view_ref = {}, {}

    out = {"loss": loss.avg, metric_name: metric_loss.avg, "mse_dep": dep_metric_loss.avg
           , "bpp_loss": bpp_loss.avg, "img_bpp_loss": img_bpp_loss.avg, "dep_bpp_loss": dep_bpp_loss.avg
           , "aux_loss":aux_loss.avg}

    return out

def test_epoch(epoch, val_dataloader, model, criterion, args, logger):
    model.eval()
    device = next(model.parameters()).device

    if args.metric == "mse":
        metric_dB_name = 'PSNR'
        metric_name = "mse_img" 
    else:
        metric_dB_name = "MS-SSIM"
        metric_name = "ms_ssim_loss_img"

    metric_dB = AverageMeter(metric_dB_name, ':.4e')
    metric_loss = AverageMeter(args.metric, ':.4e')  
    dep_metric_loss = AverageMeter("Depth MSE", ':.4e')
    psnr = AverageMeter('psnr', ':.4e')
    msssim = AverageMeter('ms-ssim', ':.4e')
    loss = AverageMeter('Loss', ':.4e')
    bpp_loss = AverageMeter('BppLoss', ':.4e')
    img_bpp_loss = AverageMeter('Image BppLoss', ':.4e')
    dep_bpp_loss = AverageMeter('Depth BppLoss', ':.4e')
    aux_loss = AverageMeter('AuxLoss', ':.4e')

    loop = tqdm(val_dataloader)

    view_main, view_ref = {}, {}
    curr_num_in_dataset = 0

    for i, batch in enumerate(loop):
        curr_num_in_dataset += 1
        datas = [frame.to(device) for frame in batch]
        
        # clear_dict_and_empty_cache(view_main)
        view_main = {}

        view_main['x_main'], view_main['depth_main'], view_main['K_main'], view_main['w2vt_main'], view_main['dataset_id'] = datas[0][:, 0], datas[1][:, 0], datas[2][:, 0], datas[3][:, 0], datas[4][:]
        
        if args.mask_by_radius or args.norm_by_radius:
            view_main['radius'] = datas[5][:]

        if view_ref and (not torch.equal(view_main['dataset_id'], view_ref['dataset_id'])):
            # clear_dict_and_empty_cache(view_ref)
            view_ref = {}
            curr_num_in_dataset = 1

        out_net = model(view_main, view_ref)

        # clear_dict_and_empty_cache(view_ref)
        view_ref = {}
        view_ref['x_ref'], view_ref['x_ref_feats'], view_ref['depth_ref'] = out_net["x_hat"].clamp(0, 1), out_net["x_dec_feats"], out_net["depth_hat"]
        view_ref['K_ref'], view_ref['w2vt_ref'], view_ref['dataset_id'] = view_main['K_main'], view_main['w2vt_main'], view_main['dataset_id']
        if curr_num_in_dataset % args.reset_period == 0:
            view_ref['x_ref_feats'] = None
        
        out_criterion = criterion(out_net, view_main, args, is_test=True)

        out_aux_loss = compute_aux_loss(model.aux_loss(), backward=False)
        
        loss.update(out_criterion["loss"].item())
        psnr.update(out_criterion["psnr"].item())
        msssim.update(out_criterion["ms-ssim"].item())
        bpp_loss.update((out_criterion["bpp_loss"]).item())
        img_bpp_loss.update((out_criterion["bpp_img_loss"]).item())
        dep_bpp_loss.update((out_criterion["bpp_dep_loss"]).item())
        aux_loss.update(out_aux_loss.item())
        metric_loss.update(out_criterion[metric_name].item())
        dep_metric_loss.update(out_criterion["mse_dep"].item())

        loop.set_description('[{}/{}]'.format(i, len(val_dataloader)))
        loop.set_postfix({"Loss":loss.avg, 'Bpp':bpp_loss.avg, args.metric: metric_loss.avg, 'Aux':aux_loss.avg
                        , "mse_dep":dep_metric_loss.avg, "Image Bpp":img_bpp_loss.avg, "Depth Bpp":dep_bpp_loss.avg
                        , "PSNR": psnr.avg, "MS-SSIM": msssim.avg})

    out = {"loss": loss.avg, metric_name: metric_loss.avg, "mse_dep": dep_metric_loss.avg
           , "bpp_loss": bpp_loss.avg, "img_bpp_loss": img_bpp_loss.avg, "dep_bpp_loss": dep_bpp_loss.avg
           , "aux_loss":aux_loss.avg, "PSNR": psnr.avg, "MS-SSIM": msssim.avg}
    
    logger.info(f"test_over ___ BPP: {out['bpp_loss']:.4f} ___ PSNR: {out['PSNR']:.2f} ___ MS-SSIM: {out['MS-SSIM']:.4f}")

    return out

if __name__ == "__main__":
    dataset_args = get_dataset_args()
    args = get_main_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if not args.save_dir:
        save_dir, experiment_id = get_output_folder(os.path.join(args.save_root, '{}/{}/{}/lamda{}_dep_lambda{}/'.format(args.datasets_name, args.metric, args.model_name + args.aux_name, int(args.lmbda), int(args.dep_lmbda))), 'train')
    else:
        save_dir = args.save_dir
    log_file_path = os.path.join(save_dir, args.train_log_file_name)
    logger = setup_logger(log_file_path)
    
    train_short_dataset = MultiViewImageDataset(dataset_args.data_root, dataset_args, is_train=True, sample_mode="small")
    train_large_dataset = MultiViewImageDataset(dataset_args.data_root, dataset_args, is_train=True, sample_mode="large")
    test_dataset = MultiViewImageDataset(dataset_args.data_root, dataset_args, is_train=False)
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    train_short_dataloader = DataLoader(train_short_dataset, batch_size=args.batch_size, num_workers=args.num_workers
                                        ,shuffle=True, pin_memory=(device == "cuda"))
    train_large_dataloader = DataLoader(train_large_dataset, batch_size=args.batch_size, num_workers=args.num_workers
                                        ,shuffle=True, pin_memory=(device == "cuda"))
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=(device == "cuda"))

    if args.model_name == "LMVIC_3D_GP":
        net = LMVIC_3D_GP(args=args)

    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.reduce_lr_epochs, 0.5)
    if args.metric == "mse":
        criterion = MSE_Loss()
    else:
        criterion = MS_SSIM_Loss()

    last_epoch = 0
    best_loss = float("inf")

    if args.load_compress_model_path:
        logger.info(f"Loading model: {args.load_compress_model_path}")
        checkpoint = torch.load(args.load_compress_model_path, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])   
        last_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        best_b_model_path = os.path.join(os.path.split(args.load_compress_model_path)[0], 'ckpt.best.pth.tar')
        if os.path.exists(best_b_model_path):
            best_loss = torch.load(best_b_model_path)["loss"]
        else:
            best_loss = checkpoint["loss"]

    if args.metric == "mse":
        metric_dB_name = 'PSNR'
        metric_name = "mse_loss" 
    else:
        metric_dB_name = "MS-SSIM"
        metric_name = "ms_ssim_loss"

    for epoch in range(last_epoch, args.epochs):
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        if epoch < args.large_sample_start_epoch:
            train_dataloader = train_short_dataloader
        else:
            train_dataloader = train_large_dataloader
        train_records = train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm, args, logger)

        logger.info(f'Trained Epoch: {epoch}')
        for key, value in train_records.items():
            logger.info(f'{key}: {value}')

        lr_scheduler.step()

        if (epoch + 1) % args.test_period == 0 or epoch == 0:
            with torch.no_grad():
                test_records = test_epoch(epoch, test_dataloader, net, criterion, args, logger)
                logger.info(f'Tested Epoch: {epoch}')
                for key, value in test_records.items():
                    logger.info(f'{key}: {value}')

            loss = test_records["loss"]
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
        else:
            loss = best_loss
            is_best = False

        if args.save and ((epoch + 1) % args.save_period == 0 or epoch == 0):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                    'lr_scheduler': lr_scheduler.state_dict(),
                },
                epoch, logger, is_best, save_dir
            )