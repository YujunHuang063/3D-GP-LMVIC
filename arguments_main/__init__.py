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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 

            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, "-" + key[0:1], default=value, action="store_true")
                elif t == tuple or t == list:
                    t = type(value[0])
                    group.add_argument("--" + key, "-" + key[0:1], default=value, type=t, nargs='+')
                else:
                    group.add_argument("--" + key, "-" + key[0:1], default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == tuple or t == list:
                    t = type(value[0])
                    group.add_argument("--" + key, default=value, type=t, nargs='+')
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class GaussianModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Gaussian Model Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class GaussianPipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Gaussian Pipeline Parameters")

class GaussianOptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.view_m_lr = 0.0001
        self.view_c_lr = 0.0001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        super().__init__(parser, "Gaussian Optimization Parameters")

class DatasetParams(ParamGroup):
    def __init__(self, parser):
        self.data_root = ''
        self.num_workers = 2
        self.split_ratio = 0.9
        self.sort_split_dataset = False
        self.sort_by_prediction = False
        self.sort_by_prediction_and_inertance = False
        self.split_order_to_disorder = False
        self.sort_views = False
        self.each_interval_sample_test = False
        self.sample_test_interval_length = 60
        self.radius_weight = 1.0
        self.datasets_name = "Tanks_and_Temples"
        self.data_type = 'colmap'
        super().__init__(parser, "Dataset Parameters")

class TrainingParams(ParamGroup):
    def __init__(self, parser):
        self.small_sample_length = 4
        self.small_sample_step = 2
        self.large_sample_length = 32
        self.large_sample_step = 16
        self.large_sample_start_epoch = 20
        self.test_sample_length = 1
        self.test_sample_step = 1
        self.random_crop = False
        self.crop_size = [384, 192]
        self.lmbda = 2048
        self.dep_lmbda = self.lmbda / 4
        self.view_weights = (0.5,1.2,0.5,0.9)
        self.training_sample_length = 4
        self.batch_size = 2
        self.epochs = 30
        self.clip_max_norm = 1.0
        self.metric = "mse"
        self.learning_rate = 1e-4
        self.aux_learning_rate = 1e-3
        self.reduce_lr_epochs = (30, 60, 90, 120)
        super().__init__(parser, "Training Parameters")

class TestingParams(ParamGroup):
    def __init__(self, parser):
        self.test_batch_size = 1
        self.test_period = 10
        self.entropy_estimation = False
        self.entropy_coder = False
        self.test_log_file_name = 'test_result.log'
        self.test_entropy_coding_time = False
        self.test_entropy_estimation_enc_dec_time = False
        self.ignore_rate_distortion_metrics = False
        super().__init__(parser, "Testing Parameters")

class ModelParams(ParamGroup):
    def __init__(self, parser):
        self.model_name = "LMVIC"
        self.aux_name = ''
        self.window_size = 10
        self.cross_attention_hidden_dim = 192
        self.num_heads = 4
        self.reset_period = 10000
        self.mask_by_radius = False
        self.mask_by_radius_a_lot = False
        self.norm_by_radius = False
        self.wo_depth_pred = False
        self.detach_offset = False
        self.skip_attention = False
        self.skip_attention_simple = False
        self.Unet_at_decoder = False
        self.coder_mode = 'light'
        self.sigmoid_mask = False
        self.radius_scale = 3.0
        self.image_domain_align = False
        self.downsample_scale1_fusion = False
        self.wo_img_mask = False
        self.wo_dep_mask = False
        self.dep_decoder_last_layer_double = False
        self.wo_offset =False
        self.wo_reference_view =False
        super().__init__(parser, "Model Parameters")

class GeneralParams(ParamGroup):
    def __init__(self, parser):
        self.seed = 1
        self.cuda = False
        self.save = False
        self.load_compress_model_path = ""
        self.save_root = "save_model/"
        self.save_dir = ""
        self.train_log_file_name = "train_result.log"
        self.save_period = 10
        self.load_homo_model_path = ""
        self.save_compression_result_path = ""
        self.rename_ckpt_key1 = False
        super().__init__(parser, "General Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
