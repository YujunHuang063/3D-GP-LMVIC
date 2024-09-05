#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

data_dir="data/Tanks&Temples"
output_dir="output"

folders=$(ls -d $data_dir/*/)

for folder in $folders; do
  folder_name=$(basename "$folder")

  model_path="$data_dir/${folder_name}/scene_params"

  python train.py -s "$folder" --view_base_image_params --ip 127.0.0.3 --port 6010 --model_path "$model_path"
done