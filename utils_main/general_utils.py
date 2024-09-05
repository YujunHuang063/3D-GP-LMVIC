import os
import pdb
import bisect
import logging
import numpy as np

import torch
import torch.nn.functional as F

def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines

def cuda_to_numpy(cuda_tensor):
    return cuda_tensor.cpu().numpy()

def numpy_to_cuda(numpy_array):
    return torch.tensor(numpy_array).cuda()

def find_min_index_greater_than(cumsum_list, index):
    pos = bisect.bisect_right(cumsum_list, index)
    return pos

def pad_to_window_size(x, window_size):
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_padded, pad_h, pad_w

def setup_logger(log_path):
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_output_folder(parent_dir, env_name, output_current_folder=False):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    if not output_current_folder: 
        experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir, experiment_id

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
def clear_dict_and_empty_cache(data_dict):
    for key in list(data_dict.keys()):
        del data_dict[key]

    torch.cuda.empty_cache()

def detach_dict_values(data_dict):
    detached_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            detached_dict[key] = value.detach().data
        elif isinstance(value, dict):
            detached_dict[key] = detach_dict_values(value)  # Recursive call for nested dictionaries
        else:
            detached_dict[key] = value
    clear_dict_and_empty_cache(data_dict)
    return detached_dict

def save_checkpoint(state, epoch, logger, is_best=False, save_dir=None, filename="ckpt.pth.tar"):
    filename = f"ckpt_epoch_{epoch}.pth.tar"
    save_file = os.path.join(save_dir, filename)

    logger.info(f"Saving model to: {save_file}")

    torch.save(state, save_file)

    keep_latest_checkpoints(save_dir, num_keep=2, logger=logger)

    '''
    if is_best:
        best_save_file = os.path.join(save_dir, f"ckpt.best.pth.tar")
        torch.save(state, best_save_file)
        logger.info(f"Saving best model to: {best_save_file}")
    '''

def keep_latest_checkpoints(save_dir, num_keep=2, logger=None):
    ckpt_files = [f for f in os.listdir(save_dir) if f.startswith("ckpt_epoch_") and f.endswith(".pth.tar")]
    if len(ckpt_files) <= num_keep:
        return

    ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('_')[2].split('.')[0]))

    for old_ckpt in ckpt_files[:-num_keep]:
        old_ckpt_path = os.path.join(save_dir, old_ckpt)
        os.remove(old_ckpt_path)
        if logger:
            logger.info(f"Deleted old checkpoint: {old_ckpt_path}")

def calculate_coding_bpp(coding_results, num_pixels):
    total_length = 0

    def recursive_length(node):
        nonlocal total_length
        if isinstance(node, list):
            for child in node:
                recursive_length(child)
        else:
            total_length += len(node) * 8

    recursive_length(coding_results)
    return total_length / num_pixels

def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)

    return outfeature

def bilineardownsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear', align_corners=False)
    return outfeature

def rename_keys(state_dict):
    new_state_dict = {}
    
    for key in state_dict.keys():
        if "entropy_bottleneck" in key:
            key_new = key
            # 重命名 matrices
            if '.matrices.' in key:
                key_new = key.replace('.matrices.', '._matrix')
            # 重命名 biases
            if '.biases.' in key:
                key_new = key.replace('.biases.', '._bias')
            # 重命名 factors
            if '.factors.' in key:
                key_new = key.replace('.factors.', '._factor')
        
            new_state_dict[key_new] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    
    return new_state_dict

def convert_to_double(data):
    if isinstance(data, torch.Tensor) and data.dtype == torch.float32:
        return data.double()
    elif isinstance(data, dict):
        return {key: convert_to_double(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_double(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_double(item) for item in data)
    else:
        return data