CUDA_VISIBLE_DEVICES=0 python train.py \
--data_root gaussian_splatting/data/Tanks\&Temples --datasets_name Tanks_and_Temples --random_crop --crop_size 256 256 \
--cuda --model_name LMVIC_3D_GP --norm_by_radius --aux_name _radius_norm \
--metric ms_ssim --lmbda 8 --dep_lmbda 64 --epochs 300 --reduce_lr_epochs 60 120 180 240 300 --large_sample_start_epoch 300 \
--save

CUDA_VISIBLE_DEVICES=0 python train.py \
--data_root gaussian_splatting/data/Mip-NeRF_360 --datasets_name Mip-NeRF_360 --random_crop --crop_size 256 256 \
--cuda --model_name LMVIC_3D_GP --norm_by_radius --aux_name _radius_norm \
--metric mse --lmbda 256 --dep_lmbda 64 --epochs 300 --reduce_lr_epochs 60 120 180 240 300 --large_sample_start_epoch 300 \
--save

CUDA_VISIBLE_DEVICES=0 python train.py \
--data_root gaussian_splatting/data/DeepBlending --datasets_name DeepBlending --random_crop --crop_size 256 256 --sort_views \
--cuda --model_name LMVIC_3D_GP --norm_by_radius --aux_name _radius_norm_sort_views \
--metric mse --lmbda 4096 --dep_lmbda 128 --epochs 300 --reduce_lr_epochs 60 120 180 240 300 --large_sample_start_epoch 300 \
--save
