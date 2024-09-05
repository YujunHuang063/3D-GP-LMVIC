CUDA_VISIBLE_DEVICES=1 python train.py -s data/Tanks\&Temples/Train --view_base_image_params --ip 127.0.0.2 --port 6010 \
--model_path output/Train_new_3D_w_view_base_image_params

CUDA_VISIBLE_DEVICES=0 python train.py -s data/Mip-NeRF_360/bonsai --view_base_image_params --ip 127.0.0.2 --port 6010 \
--model_path output/bonsai_new_3D_w_view_base_image_params

CUDA_VISIBLE_DEVICES=0 python train.py -s data/DeepBlending/Ponche --view_base_image_params --ip 127.0.0.2 --port 6011 \
--model_path output/Ponche_new_3D_w_view_base_image_params