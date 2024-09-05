import os
import pdb
import cv2
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path

from utils_main.general_utils import read_file, cuda_to_numpy, find_min_index_greater_than
from utils_main.graphics_utils import sort_cameras_by_overlap, sort_cameras_by_distance, sort_cameras_by_distance_with_prediction, sort_cameras_by_distance_with_prediction_and_inertance

from gaussian_splatting.bridge import load_scene, calculate_view_params_and_depths

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

class MultiViewImageDataset(Dataset):
    def __init__(self, data_root, args, is_train=True, sample_mode="small"):
        self.data_root = data_root
        self.args = args
        self.is_train = is_train
        self.small_sample_length = args.small_sample_length
        self.large_sample_length = args.large_sample_length
        self.small_sample_step = args.small_sample_step
        self.large_sample_step = args.large_sample_step
        self.test_sample_length = args.test_sample_length
        self.test_sample_step = args.test_sample_step
        self.sample_mode = sample_mode
        self.sort_views = args.sort_views
        
        self.dataset_list = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        self.datasets_images_name, self.datasets_images_order, self.datasets_images_name_order = {}, {}, {}

        if is_train:
            self.small_sample_num_list, self.large_sample_num_list = [], []
        else:
            self.test_sample_num_list = []

        for i in range(len(self.dataset_list)):
            data_path = os.path.join(self.data_root, self.dataset_list[i])

            # load images name
            if args.data_type == 'colmap':
                images_path = os.path.join(data_path, "images")
                dataset_images_name = []
                for file_name in os.listdir(images_path):
                    if file_name.endswith('.jpg') or file_name.endswith('.JPG') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
                        dataset_images_name.append(file_name)
                dataset_images_name.sort()
            elif args.data_type == 'blender':
                dataset_images_name = self.readCamerasFromTransforms(data_path, "transforms_train.json") + self.readCamerasFromTransforms(data_path, "transforms_test.json")

            # save view params and depth maps
            with torch.no_grad():
                view_world_transforms, FoVxs, FoVys, radius = self._check_and_handle_view_params_and_depths(data_path, args, dataset_images_name)

            # save order
            order = self._check_and_handle_order(data_path, view_world_transforms, FoVxs, FoVys, radius)

            if args.split_order_to_disorder:
                split_index = int(args.split_ratio * len(dataset_images_name))
                order_first, order_second = np.sort(order[:split_index]), np.sort(order[split_index:])
                order = np.concatenate((order_first, order_second))

            dataset_images_name_order = [dataset_images_name[i] for i in order]
            self.datasets_images_order[self.dataset_list[i]] = order

            assert len(dataset_images_name_order)==len(dataset_images_name)

            if not args.each_interval_sample_test:
                split_index = int(args.split_ratio * len(dataset_images_name_order))
                
                if is_train:
                    dataset_images_name_order = dataset_images_name_order[:split_index]
                    dataset_images_name = dataset_images_name[:split_index]
                else:
                    dataset_images_name_order = dataset_images_name_order[split_index:]
                    dataset_images_name = dataset_images_name[split_index:]

            else:
                def split_data(data, split_ratio, is_train):
                    length = len(data)
                    split_index = int(split_ratio * length)
                    if is_train:
                        return data[:split_index]
                    else:
                        return data[split_index:]
                    
                sample_test_interval_length = args.sample_test_interval_length
                dataset_images_name_order_intervals = []
                dataset_images_name_intervals = []
                
                for j in range(0, len(dataset_images_name_order), sample_test_interval_length):
                    interval_order = dataset_images_name_order[j:j + sample_test_interval_length]
                    interval = dataset_images_name[j:j + sample_test_interval_length]
                    
                    dataset_images_name_order_intervals.extend(split_data(interval_order, args.split_ratio, is_train))
                    dataset_images_name_intervals.extend(split_data(interval, args.split_ratio, is_train))
                
                dataset_images_name_order = dataset_images_name_order_intervals
                dataset_images_name = dataset_images_name_intervals
                
            self.datasets_images_name_order[self.dataset_list[i]] = dataset_images_name_order
            self.datasets_images_name[self.dataset_list[i]] = dataset_images_name
            
            if is_train:
                self.small_sample_num_list.append((len(dataset_images_name) - args.small_sample_length + args.small_sample_step - 1) // args.small_sample_step + 1)
                self.large_sample_num_list.append((len(dataset_images_name) - args.large_sample_length + args.large_sample_step - 1) // args.large_sample_step + 1)
            else:
                self.test_sample_num_list.append((len(dataset_images_name) - args.test_sample_length + args.test_sample_step - 1) // args.test_sample_step + 1)
        
        if is_train:
            self.small_sample_num_cumsum = np.cumsum(self.small_sample_num_list)
            self.large_sample_num_cumsum = np.cumsum(self.large_sample_num_list)
        else:
            self.test_sample_num_cumsum = np.cumsum(self.test_sample_num_list)

    def readCamerasFromTransforms(self, path, transformsfile, extension=".png"):
        dataset_images_name = []

        with open(os.path.join(path, transformsfile)) as json_file:
            contents = json.load(json_file)

            frames = contents["frames"]
            for frame in frames:
                file_name = frame["file_path"] + extension
                dataset_images_name.append(file_name)
            
        return dataset_images_name

    def __getitem__(self, index):
        if self.is_train:
            if self.sample_mode == "small":
                sample_num_cumsum = self.small_sample_num_cumsum
                sample_length = self.small_sample_length
                sample_step = self.small_sample_step
            elif self.sample_mode == "large":
                sample_num_cumsum = self.small_sample_num_cumsum
                sample_length = self.large_sample_length
                sample_step = self.large_sample_step
        else:
            sample_num_cumsum = self.test_sample_num_cumsum
            sample_length = self.test_sample_length
            sample_step = self.test_sample_step

        dataset_index = find_min_index_greater_than(sample_num_cumsum, index)

        if self.sort_views:
            dataset_images_name = self.datasets_images_name_order[self.dataset_list[dataset_index]]
        else:
            dataset_images_name = self.datasets_images_name[self.dataset_list[dataset_index]]
            
        img_start_index = (index - (sample_num_cumsum[dataset_index-1] if dataset_index>0 else 0))*sample_step
        image_paths = self.get_sample_elements(dataset_images_name, img_start_index, sample_length)
        data_path = os.path.join(self.data_root, self.dataset_list[dataset_index])
        view_params_and_depths_path = os.path.join(data_path, 'view_params_and_depths')
        if self.args.mask_by_radius or self.args.norm_by_radius:
            radius_path = os.path.join(data_path, 'radius')
            radius = np.load(os.path.join(radius_path, 'radius.npy'))
        # print(self.dataset_list[dataset_index], image_paths)
        
        images = []
        depths = []
        Ks = []
        world_view_transforms = []

        for image_name in image_paths:
            # load images
            if self.args.data_type == 'colmap':
                image_path = os.path.join(data_path, 'images', image_name)
            elif self.args.data_type == 'blender':
                image_path = os.path.join(data_path, image_name)
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img_array = np.array(img) / 255.0
                images.append(np.transpose(img_array, (2, 0, 1)))

            # extract basename
            basename = os.path.splitext(image_name)[0]
            # load .npz file
            npz_path = os.path.join(view_params_and_depths_path, basename + '.npz')
            data = np.load(npz_path)
            depths.append(data['depth'])
            Ks.append(data['K'])
            world_view_transforms.append(data['world_view_transform'])
            
        # Find the maximum height and width
        max_height = max(image.shape[1] for image in images)
        max_width = max(image.shape[2] for image in images)

        # Pad images and depths to the same size
        padded_images = [self.pad_to_match(image, (image.shape[0], max_height, max_width)) for image in images]
        padded_depths = [self.pad_to_match(depth, (1, max_height, max_width)) for depth in depths]

        # Convert lists to numpy arrays
        images = np.stack(padded_images, axis=0)
        depths = np.stack(padded_depths, axis=0)

        # Ensure images and depths have the same shape
        assert images.shape[2:] == depths.shape[2:], "Images and depths must have the same spatial dimensions."

        Ks = np.stack(Ks, axis=0)
        world_view_transforms = np.stack(world_view_transforms, axis=0)

        # random crop
        if self.is_train and self.args.random_crop:
            height, width = images.shape[2:]
            h_begin, w_begin = random.randint(0, height-self.args.crop_size[1]), random.randint(0, width-self.args.crop_size[0])
            images = images[:, :, h_begin:h_begin+self.args.crop_size[1], w_begin:w_begin+self.args.crop_size[0]]
            depths = depths[:, :, h_begin:h_begin+self.args.crop_size[1], w_begin:w_begin+self.args.crop_size[0]]
            Ks[:, 0, 2] -= w_begin
            Ks[:, 1, 2] -= h_begin
        
        if self.args.mask_by_radius or self.args.norm_by_radius:
            depths = depths / (2 * self.args.radius_scale * radius)

        common_tensors = (
            torch.from_numpy(images).float(),
            torch.from_numpy(depths).float(),
            torch.from_numpy(Ks).float(),
            torch.from_numpy(world_view_transforms).float(),
            torch.tensor(dataset_index, dtype=torch.long)
        )

        if self.args.mask_by_radius or self.args.norm_by_radius:
            common_tensors += (torch.tensor(radius).float(), )

        return common_tensors

    def pad_to_match(self, array, target_shape):
        padding = [(0, 0)] * len(array.shape)
        for i in range(len(array.shape)):
            padding[i] = (0, max(0, target_shape[i] - array.shape[i]))
        return np.pad(array, padding, mode='edge')
 
    def __len__(self):
        if self.is_train:
            if self.sample_mode=="small":
                return np.sum(self.small_sample_num_list)
            elif self.sample_mode=="large":
                return np.sum(self.large_sample_num_list)
        else:
            return np.sum(self.test_sample_num_list)
        
    def _check_and_handle_view_params_and_depths(self, data_path, args, dataset_images_name = None):
        view_params_and_depths_path = os.path.join(data_path, 'view_params_and_depths')
        view_world_transforms, FoVxs, FoVys = [], [], []

        if not os.path.exists(view_params_and_depths_path):
            # load scene params
            args.source_path = data_path
            args.model_path = os.path.join(data_path, "scene_params")
            scene, gaussians = load_scene(args)
            views = scene.getTrainCameras()
            bg_color = [1,1,1] if args.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            os.makedirs(view_params_and_depths_path)
            for i, view in enumerate(views):
                depth, K, world_view_transform, FoVx, FoVy = calculate_view_params_and_depths(view, gaussians, args, background)
                depth, world_view_transform = cuda_to_numpy(depth), cuda_to_numpy(world_view_transform)
                view_world_transform = np.linalg.inv(world_view_transform)
                view_params_and_depth = {
                    'depth': depth,
                    'K': K,
                    'world_view_transform': world_view_transform,
                    'view_world_transform': view_world_transform,
                    'FoVx': FoVx,
                    'FoVy': FoVy,
                }

                if args.data_type == 'colmap':
                    save_path = os.path.join(view_params_and_depths_path, f'{view.image_name}.npz')
                elif args.data_type == 'blender':
                    assert Path(dataset_images_name[i]).stem == view.image_name
                    save_path = os.path.join(view_params_and_depths_path, f'{os.path.splitext(dataset_images_name[i])[0]}.npz')
                    save_dir = os.path.dirname(save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                np.savez_compressed(save_path, **view_params_and_depth)
                view_world_transforms.append(view_world_transform)
                FoVxs.append(FoVx)
                FoVys.append(FoVy)
            radius = scene.cameras_extent
            radius_path = os.path.join(data_path, 'radius')
            os.makedirs(radius_path)
            np.save(os.path.join(radius_path, 'radius.npy'), radius)
        else:
            if args.data_type == 'colmap':
                npz_files = sorted(f for f in os.listdir(view_params_and_depths_path) if f.endswith('.npz'))
            elif args.data_type == 'blender':
                npz_files = [os.path.splitext(image_name)[0] + '.npz' for image_name in dataset_images_name]
            
            for npz_file in npz_files:
                save_path = os.path.join(view_params_and_depths_path, npz_file)
                # read data from .npz file
                data = np.load(save_path)
                view_world_transform, FoVx, FoVy = data['view_world_transform'], data['FoVx'], data['FoVy']
                view_world_transforms.append(view_world_transform)
                FoVxs.append(FoVx)
                FoVys.append(FoVy)

            radius_path = os.path.join(data_path, 'radius')
            radius = np.load(os.path.join(radius_path, 'radius.npy'))

        return view_world_transforms, FoVxs, FoVys, radius

    def _check_and_handle_order(self, data_path, view_world_transforms, FoVxs, FoVys, radius=10):
        if not self.args.sort_split_dataset:
            if not self.args.sort_by_prediction and not self.args.sort_by_prediction_and_inertance:
                order_path = os.path.join(data_path, 'order')
            elif not self.args.sort_by_prediction_and_inertance:
                order_path = os.path.join(data_path, 'order_sort_by_pred')
            else:
                order_path = os.path.join(data_path, 'order_sort_by_pred_and_inert')
        else:
            if not self.args.sort_by_prediction and not self.args.sort_by_prediction_and_inertance:
                order_path = os.path.join(data_path, f'order_split_sort_w_ratio_{self.args.split_ratio:.2f}')
            elif not self.args.sort_by_prediction_and_inertance:
                order_path = os.path.join(data_path, f'order_split_sort_by_pred_w_ratio_{self.args.split_ratio:.2f}')
            else:
                order_path = os.path.join(data_path, f'order_split_sort_by_pred_and_inert_w_ratio_{self.args.split_ratio:.2f}')

        if not os.path.exists(order_path):
            os.makedirs(order_path)
            if not self.args.sort_split_dataset:
                if not self.args.sort_by_prediction and not self.args.sort_by_prediction_and_inertance:
                    order = sort_cameras_by_distance(view_world_transforms, FoVxs, FoVys, radius / 10) # sort_cameras_by_overlap
                elif not self.args.sort_by_prediction_and_inertance:
                    order = sort_cameras_by_distance_with_prediction(view_world_transforms, FoVxs, FoVys, radius / 10) # sort_cameras_by_overlap
                else:
                    order = sort_cameras_by_distance_with_prediction_and_inertance(view_world_transforms, FoVxs, FoVys, radius / 10) # sort_cameras_by_overlap
            else:
                split_index = int(self.args.split_ratio * len(view_world_transforms))
                if not self.args.sort_by_prediction and not self.args.sort_by_prediction_and_inertance:
                    split_first_part_order = sort_cameras_by_distance(view_world_transforms[:split_index], FoVxs, FoVys, radius / 10)
                    split_second_part_order = sort_cameras_by_distance(view_world_transforms[split_index:], FoVxs, FoVys, radius / 10)
                elif not self.args.sort_by_prediction_and_inertance:
                    split_first_part_order = sort_cameras_by_distance_with_prediction(view_world_transforms[:split_index], FoVxs, FoVys, radius / 10)
                    split_second_part_order = sort_cameras_by_distance_with_prediction(view_world_transforms[split_index:], FoVxs, FoVys, radius / 10)
                else:
                    split_first_part_order = sort_cameras_by_distance_with_prediction_and_inertance(view_world_transforms[:split_index], FoVxs, FoVys, radius / 10)
                    split_second_part_order = sort_cameras_by_distance_with_prediction_and_inertance(view_world_transforms[split_index:], FoVxs, FoVys, radius / 10)
                split_second_part_order = [x + split_index for x in split_second_part_order]
                order = split_first_part_order + split_second_part_order

            np.save(os.path.join(order_path, "order.npy"), order)
        else:
            order = np.load(os.path.join(order_path, "order.npy"))

        return order
    
    def get_sample_elements(self, dataset_images_name, img_start_index, sample_length):
        total_length = len(dataset_images_name)
        if img_start_index + sample_length > total_length:
            image_paths = dataset_images_name[-sample_length:]
        else:
            image_paths = dataset_images_name[img_start_index:img_start_index + sample_length]
        return image_paths
        
    def change_to_large_sample_mode(self, ):
        self.sample_mode = "large"