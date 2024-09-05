# entropy estimation
CUDA_VISIBLE_DEVICES=0 python eval.py --data_root gaussian_splatting/data/Tanks\&Temples --cuda \
--model_name LMVIC_3D_GP --norm_by_radius --aux_name _radius_norm \
--load_compress_model_path path/to/checkpoint \
--save_dir path/to/save/result \
--entropy_estimation

# entropy coding
CUDA_VISIBLE_DEVICES=0 python eval.py --data_root gaussian_splatting/data/Tanks\&Temples --cuda \
--model_name LMVIC_3D_GP --norm_by_radius --aux_name _radius_norm --dep_decoder_last_layer_double \
--load_compress_model_path path/to/checkpoint \
--save_dir path/to/save/result \
--entropy_coder