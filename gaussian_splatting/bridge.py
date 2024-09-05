import os
import sys
import pdb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene import Scene
from gaussian_renderer import GaussianModel, render
from utils.graphics_utils import fov2focal, build_K

def load_scene(dataset):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    return scene, gaussians

def calculate_view_params_and_depths(view, gaussians, args, background):
    render_result = render(view, gaussians, args, background)
    depth = render_result['depth']
    focal_x = fov2focal(view.FoVx, view.image_width)
    focal_y = fov2focal(view.FoVy, view.image_height)
    K = build_K(focal_x, focal_y, view.image_width, view.image_height)
    world_view_transform = view.world_view_transform.transpose(0, 1)
    return depth, K, world_view_transform, view.FoVx, view.FoVy