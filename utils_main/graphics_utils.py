import os
import pdb
import sys
import torch
import torch.nn.functional as F
import numpy as np

from utils_main.general_utils import AverageMeter

current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_directory)

from utils_main.tsp_utils import *

def rotation_matrix_to_quaternion(R):
    q = np.empty((4,))
    t = np.trace(R)
    if t > 0.0:
        t = np.sqrt(t + 1.0)
        q[0] = 0.5 * t
        t = 0.5 / t
        q[1] = (R[2, 1] - R[1, 2]) * t
        q[2] = (R[0, 2] - R[2, 0]) * t
        q[3] = (R[1, 0] - R[0, 1]) * t
    else:
        i = 0
        if R[1, 1] > R[0, 0]:
            i = 1
        if R[2, 2] > R[i, i]:
            i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3
        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        q[i + 1] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[k, j] - R[j, k]) * t
        q[j + 1] = (R[j, i] + R[i, j]) * t
        q[k + 1] = (R[k, i] + R[i, k]) * t
    return normalize_quaternion(q)

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def inverse_quaternion(q):
    is_tuple = isinstance(q, tuple)
    
    if is_tuple:
        q = np.array(q)

    q_inv = q.copy()
    q_inv[1:] = -q[1:]
    q_inv = q_inv / np.dot(q, q)

    if is_tuple:
        q_inv = tuple(q_inv)
    
    return q_inv

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def calculate_overlap(matrix1, matrix2, fovx, fovy, cone_length):
    def get_view_cone_verts(fovx, fovy, cone_length):
        h = np.tan(fovy / 2) * cone_length
        w = np.tan(fovx / 2) * cone_length
        return np.array([
            [0, 0, 0, 1], 
            [-w, -h, cone_length, 1], 
            [w, -h, cone_length, 1], 
            [-w, h, cone_length, 1], 
            [w, h, cone_length, 1]
        ])

    view_cone_verts1 = get_view_cone_verts(fovx[0], fovy[0], cone_length)
    view_cone_verts2 = get_view_cone_verts(fovx[1], fovy[1], cone_length)
    
    view_cone_verts1_world = (matrix1 @ view_cone_verts1.T).T[:, :3]
    view_cone_verts2_world = (matrix2 @ view_cone_verts2.T).T[:, :3]

    overlap = np.sum([np.sum((v1 - v2) ** 6) for v1, v2 in zip(view_cone_verts1_world, view_cone_verts2_world)]) # np.linalg.norm(v1 - v2)
    return 1 / (1 + overlap)

def build_overlap_matrix(view_world_transforms, Fovxs, Fovys, cone_length):
    num_cameras = len(view_world_transforms)
    overlap_matrix = np.eye(num_cameras)
    
    for i in range(num_cameras):
        for j in range(i + 1, num_cameras):
            overlap = calculate_overlap(
                view_world_transforms[i], view_world_transforms[j], 
                [Fovxs[i], Fovxs[j]], [Fovys[i], Fovys[j]], cone_length
            )
            overlap_matrix[i, j] = overlap
            overlap_matrix[j, i] = overlap
    
    return overlap_matrix

def spectral_ordering(similarity_matrix):
    degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    laplacian_matrix = degree_matrix - similarity_matrix
    eigvals, eigvecs = np.linalg.eigh(laplacian_matrix)
    fiedler_vector = eigvecs[:, 1]
    order = np.argsort(fiedler_vector)
    return order

def sort_by_similarity_using_mst(similarity_matrix):
    # Step 1: Convert similarity matrix to distance matrix
    distance_matrix = 1 / similarity_matrix - 1

    # Ensure the diagonal elements are zero
    np.fill_diagonal(distance_matrix, 0)
    
    # Step 2: Build a graph from the distance matrix
    graph = nx.from_numpy_matrix(distance_matrix)
    
    # Step 3: Compute the Minimum Spanning Tree (MST)
    mst = nx.minimum_spanning_tree(graph)
    
    # Step 4: Perform DFS on the MST to get the order
    start_node = 0
    order = list(nx.dfs_preorder_nodes(mst, source=start_node))
    
    return order

def sort_cameras_by_overlap(view_world_transforms, Fovxs, Fovys, cone_length):
    overlap_matrix = build_overlap_matrix(view_world_transforms, Fovxs, Fovys, cone_length)
    order = sort_by_similarity_using_mst(overlap_matrix)
    return order

def calculate_distance_matrix(view_world_transforms):
    n = len(view_world_transforms)
    distance_matrix = np.zeros((n, n))

    # 预先计算所有逆矩阵
    inverses = [np.linalg.inv(transform) for transform in view_world_transforms]

    # 计算距离矩阵
    for i in range(n):
        for j in range(i+1, n):
            A = view_world_transforms[i]
            B_inv = inverses[j]
            
            # 计算 AB^-1 和 AB^-1 - I
            AB_inv = np.dot(A, B_inv)
            identity_matrix = np.eye(4)
            difference_AB_inv = AB_inv - identity_matrix

            # 计算 Frobenius 范数
            frobenius_norm_AB_inv = np.linalg.norm(difference_AB_inv, 'fro')

            # 填入距离矩阵
            distance_matrix[i, j] = frobenius_norm_AB_inv
            distance_matrix[j, i] = frobenius_norm_AB_inv

    return distance_matrix

def sort_cameras_by_distance(view_world_transforms, Fovxs, Fovys, cone_length):
    distance_matrix = calculate_distance_matrix(view_world_transforms)
    order = nearest_neighbor_tsp_distance(distance_matrix)
    return order

def sort_cameras_by_distance_with_prediction(view_world_transforms, Fovxs, Fovys, cone_length):
    num_elements = len(view_world_transforms)
    visited = [False] * num_elements
    order = []

    current_index = 0
    order.append(current_index)
    visited[current_index] = True

    while len(order) < num_elements:
        # 预测下一个 view_to_world 矩阵
        if len(order) < 3:
            predicted_transform = predict_next_transform([view_world_transforms[i] for i in order])
        else:
            predicted_transform = predict_next_transform([view_world_transforms[i] for i in order[-3:]])

        min_distance = np.inf
        next_index = -1
        for i in range(num_elements):
            if not visited[i]:
                A = predicted_transform
                B_inv = np.linalg.inv(view_world_transforms[i])
                AB_inv = np.dot(A, B_inv)
                identity_matrix = np.eye(4)
                difference_AB_inv = AB_inv - identity_matrix
                frobenius_norm_AB_inv = np.linalg.norm(difference_AB_inv, 'fro')

                if frobenius_norm_AB_inv < min_distance:
                    min_distance = frobenius_norm_AB_inv
                    next_index = i
        
        order.append(next_index)
        visited[next_index] = True
        current_index = next_index

    return order

def sort_cameras_by_distance_with_prediction_and_inertance(view_world_transforms, Fovxs, Fovys, cone_length):
    num_elements = len(view_world_transforms)
    visited = [False] * num_elements
    order = []

    current_index = 0
    order.append(current_index)
    visited[current_index] = True
    state = "init"
    start_index = 0
    inertance = AverageMeter('inertance', ':.4e')

    while len(order) < num_elements:

        pred_order_ind = max(max(start_index, len(order) - 3), 0)
        predicted_transform = predict_next_transform([view_world_transforms[i] for i in order[pred_order_ind:]])

        if state == "mid":
            curr_index = order[-1]

            next_index = curr_index + 1
            if next_index < num_elements:
                distance = frobenius_norm_difference(predicted_transform, view_world_transforms[next_index])
            
                if not visited[next_index] and (inertance.avg == 0 or distance < 1.8*inertance.avg):
                    order.append(next_index)
                    visited[next_index] = True
                    current_index = next_index
                    inertance.update(distance)
                    continue

            prev_index = curr_index - 1
            if prev_index >= 0:
                distance = frobenius_norm_difference(predicted_transform, view_world_transforms[prev_index])

                if not visited[prev_index] and (inertance.avg == 0 or distance < 1.8*inertance.avg):
                    order.append(prev_index)
                    visited[prev_index] = True
                    current_index = prev_index
                    inertance.update(distance)
                    continue

            min_distance = np.inf
            next_index = -1
            for i in range(curr_index-2, curr_index+3):
                if i >= 0 and i < num_elements and not visited[i]:
                    frobenius_norm_AB_inv = frobenius_norm_difference(predicted_transform, view_world_transforms[i])
                    if frobenius_norm_AB_inv < min_distance:
                        min_distance = frobenius_norm_AB_inv
                        next_index = i

            if inertance.avg == 0 or min_distance < 1.8*inertance.avg:
                order.append(next_index)
                visited[next_index] = True
                current_index = next_index
                inertance.update(min_distance)
                continue

            min_distance = np.inf
            next_index = -1
            for i in range(num_elements):
                if not visited[i]:
                    frobenius_norm_AB_inv = frobenius_norm_difference(predicted_transform, view_world_transforms[i])
                    if frobenius_norm_AB_inv < min_distance:
                        min_distance = frobenius_norm_AB_inv
                        next_index = i
    
            order.append(next_index)
            visited[next_index] = True
            current_index = next_index

            if inertance.avg > 0 and min_distance > 1.8*inertance.avg:
                state = "init"
                start_index = len(order) - 1
                inertance.reset()
            else:
                inertance_i = min(min_distance, 1.8*inertance.avg) if inertance.avg > 0 else min_distance
                inertance.update(inertance_i)
            continue

        if state == "init":
            min_distance = np.inf
            next_index = -1
            for i in range(num_elements):
                if not visited[i]:
                    frobenius_norm_AB_inv = frobenius_norm_difference(predicted_transform, view_world_transforms[i])
                    if frobenius_norm_AB_inv < min_distance:
                        min_distance = frobenius_norm_AB_inv
                        next_index = i
    
            order.append(next_index)
            visited[next_index] = True
            current_index = next_index

            if inertance.avg > 0 and min_distance > 1.8*inertance.avg:
                state = "init"
                start_index = len(order) - 1
                inertance.reset()
            else:
                state = "mid"
                inertance_i = min(min_distance, 1.8*inertance.avg) if inertance.avg > 0 else min_distance
                inertance.update(inertance_i)
            continue

    return order

Pixel_Coords_Dict = {}

@torch.no_grad()
def depth_projection_batch(depth_map, K_src, K_tgt, world_to_view_src, world_to_view_tgt, mask_by_radius = False, radius = None, args = None):
    """
    Project depth map from source view to target view for a batch of images and generate a mask.
    
    Parameters:
    depth_map (torch.Tensor): Depth map of the source view (batch_size, 1, H, W).
    K (torch.Tensor): Intrinsic matrix (batch_size, 3, 3).
    world_to_view_src (torch.Tensor): World to view transformation matrix of source view (batch_size, 4, 4).
    world_to_view_tgt (torch.Tensor): World to view transformation matrix of target view (batch_size, 4, 4).

    Returns:
    torch.Tensor: Depth map of the target view (batch_size, 1, H, W).
    torch.Tensor: Mask of the target view (batch_size, 1, H, W).
    """
    assert args is not None
    if args is not None:
        mask_by_radius = args.mask_by_radius
        mask_by_radius_a_lot = args.mask_by_radius_a_lot

    batch_size, _, H, W = depth_map.shape
    device = depth_map.device

    if mask_by_radius or args.norm_by_radius:
        depth_map = depth_map * 2 * radius.view(-1, 1, 1, 1)
        if args:
            depth_map = depth_map * args.radius_scale

    # Create a mesh grid of pixel coordinates with 0.5 offset
    key = f"{str(depth_map.size())}_{depth_map.device}"
    
    if key not in Pixel_Coords_Dict:
        i, j = torch.meshgrid(torch.arange(H, device=device) + 0.5, torch.arange(W, device=device) + 0.5)
        i = i.flatten()
        j = j.flatten()
        
        # Convert pixel coordinates to homogeneous coordinates
        ones = torch.ones_like(i, device=device)
        pixel_coords = torch.stack((j, i, ones), dim=0)  # (3, H*W)
        pixel_coords = pixel_coords.to(depth_map.dtype)

        # Prepare for batch processing
        pixel_coords = pixel_coords.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 3, H*W)
        Pixel_Coords_Dict[key] = pixel_coords
    else:
        pixel_coords = Pixel_Coords_Dict[key]

    depth_map_flat = depth_map.view(batch_size, -1)  # (batch_size, H*W)

    # Convert depth map to 3D coordinates in the camera frame
    cam_coords = torch.bmm(torch.inverse(K_src), pixel_coords) * depth_map_flat.unsqueeze(1)  # (batch_size, 3, H*W)
    cam_coords_homogeneous = F.pad(cam_coords, (0, 0, 0, 1), value=1)  # (batch_size, 4, H*W)
    
    # Transform 3D coordinates to the target view
    T_src_to_tgt = torch.bmm(world_to_view_tgt, torch.inverse(world_to_view_src))  # (batch_size, 4, 4)
    tgt_coords_homogeneous = torch.bmm(T_src_to_tgt, cam_coords_homogeneous)  # (batch_size, 4, H*W)
    tgt_coords = tgt_coords_homogeneous[:, :3, :] / tgt_coords_homogeneous[:, 3, :].unsqueeze(1)  # Normalize by the last row
    
    # Project 3D coordinates back to the image plane of the target view
    pixel_coords_tgt3D = torch.bmm(K_tgt, tgt_coords)
    eps = 1e-4
    pixel_coords_tgt = pixel_coords_tgt3D[:, :2, :] / torch.abs(pixel_coords_tgt3D[:, 2, :].unsqueeze(1) + eps)  # Normalize by the last row
    
    # Adjust coordinates for offset
    pixel_coords_tgt -= 0.5
    
    # Round pixel coordinates to get valid indices
    pixel_coords_tgt = torch.round(pixel_coords_tgt).long()
    
    # Mask out invalid coordinates
    valid_mask = (pixel_coords_tgt[:, 0, :] >= 0) & (pixel_coords_tgt[:, 0, :] < W) & (pixel_coords_tgt[:, 1, :] >= 0) & (pixel_coords_tgt[:, 1, :] < H) & (pixel_coords_tgt3D[:, 2, :] > 0)
    
    # Create the target depth map and mask with initial values
    tgt_depth_map = torch.full_like(depth_map, float('inf'))
    mask = torch.zeros_like(depth_map, dtype=torch.bool)

    # Flatten the depth map and mask for scatter operation
    tgt_depth_map_flat = tgt_depth_map.view(-1)
    mask_flat = mask.view(-1)

    indices = pixel_coords_tgt[:, 1, :] * W + pixel_coords_tgt[:, 0, :] + torch.arange(batch_size, device=device).long().view(batch_size, -1) * H * W
    flat_indices = indices[valid_mask]
    valid_tgt_depths = tgt_coords[:, 2][valid_mask]

    # Scatter reduce to get the minimum depth value at each pixel location
    tgt_depth_map_flat_result = torch.scatter_reduce(tgt_depth_map_flat, 0, flat_indices, valid_tgt_depths, reduce="amin")

    # Scatter to create mask
    mask_flat.scatter_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.bool))
    
    # Reshape the depth map and mask back to their original shape
    tgt_depth_map = tgt_depth_map_flat_result.view(batch_size, 1, H, W)
    mask = mask_flat.view(batch_size, 1, H, W)
    
    # Replace inf values with zeros or any other default value
    tgt_depth_map[tgt_depth_map == float('inf')] = 0

    if mask_by_radius:

        def create_radius_mask(range):
            radius_valid_mask = (depth_map_flat < (range * radius.view(-1, 1))) * valid_mask
            radius_mask = torch.zeros_like(depth_map, dtype=torch.bool)
            radius_mask_flat = radius_mask.view(-1)
            radius_mask_flat_indices = indices[radius_valid_mask]
            radius_mask_flat.scatter_(0, radius_mask_flat_indices, torch.ones_like(radius_mask_flat_indices, dtype=torch.bool))
            radius_mask = radius_mask_flat.view(batch_size, 1, H, W)

            return radius_mask

        radius_mask = create_radius_mask(2.0)
        mask = torch.cat((mask, radius_mask), 1)
        if args and mask_by_radius_a_lot:
            radius_mask1 = create_radius_mask(1.0)
            radius_mask2 = create_radius_mask(3.0)
            radius_mask3 = create_radius_mask(4.0)
            radius_mask4 = create_radius_mask(5.0)
            mask = torch.cat((mask, radius_mask1, radius_mask2, radius_mask3, radius_mask4), 1)

    if mask_by_radius or args.norm_by_radius:
        tgt_depth_map = tgt_depth_map / (2 * radius.view(-1, 1, 1, 1) + eps)

        if args:
            tgt_depth_map = tgt_depth_map / (args.radius_scale + eps)

    return tgt_depth_map, mask

def depth_reshape(depth, mode="HWC"):
    if depth.dim() == 3:
        depth = depth.unsqueeze(0)
    if mode=="HWC":
        depth = depth.permute(0, 2, 3, 1)
    return depth

Pixs_Flat_Dict = {}

def match(K1, w2c_transform1, depth1, K2, w2c_transform2, depth2, record_occlusion=True, mask_by_radius = False, radius = None, args = None):
    assert args is not None
    if args is not None:
        mask_by_radius = args.mask_by_radius
        mask_by_radius_a_lot = args.mask_by_radius_a_lot

    batch_size = K1.shape[0]
    device = K1.device
    dtype = K1.dtype
    height, width = depth2.shape[2:]

    if mask_by_radius or args.norm_by_radius:
        if depth1 is not None:
            depth1 = depth1 * 2 * radius.view(-1, 1, 1, 1)
            if args:
                depth1 = depth1 * args.radius_scale
        depth2 = depth2 * 2 * radius.view(-1, 1, 1, 1)
        if args:
            depth2 = depth2 * args.radius_scale
    
    # Generate pixel grid
    key = f"{str(depth2.size())}_{depth2.device}"

    if key not in Pixs_Flat_Dict:
        pixs = torch.tensor(np.concatenate(np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5, 1), -1)[None, :], device=device, dtype=dtype)
        pixs = pixs.expand(batch_size, -1, -1, -1)
        pixs_flat = pixs.reshape(batch_size, -1, 3)
        Pixs_Flat_Dict[key] = pixs_flat
    else:
        pixs_flat = Pixs_Flat_Dict[key]
        pixs = pixs_flat.reshape(batch_size, height, width, 3)

    invK2 = torch.inverse(K2)
    
    rays = torch.matmul(pixs_flat, invK2.transpose(1, 2))
    rays = rays.reshape(batch_size, height, width, 3)

    depth2_HWC = depth_reshape(depth2)
    points3DC2 = rays * depth2_HWC
    points4DC2 = F.pad(points3DC2, pad=(0, 1), mode='constant', value=1)

    c2w_transform2 = torch.inverse(w2c_transform2)
    points4DC2_flat = points4DC2.reshape(batch_size, -1, 4)
    points4DW_flat = torch.matmul(points4DC2_flat, c2w_transform2.transpose(1, 2))
    points4DC1_flat = torch.matmul(points4DW_flat, w2c_transform1.transpose(1, 2))
    points3DC1_flat = points4DC1_flat[:, :, 0:3] / points4DC1_flat[:, :, 3:4]

    # Generate mask
    points3DC1 = points3DC1_flat.reshape(batch_size, height, width, 3)
    mask = 0.01 <= points3DC1[:, :, :, 2]

    corresponding_pixs_flat = torch.matmul(points3DC1_flat, K1.transpose(1, 2))
    eps = 1e-4
    corresponding_pixs_flat = corresponding_pixs_flat / torch.abs(corresponding_pixs_flat[:, :, 2:3] + eps)
    corresponding_pixs = corresponding_pixs_flat.reshape(batch_size, height, width, 3)
    mask *= (corresponding_pixs[:, :, :, 0] < (width + 1)) * (corresponding_pixs[:, :, :, 0] > -1)
    mask *= (corresponding_pixs[:, :, :, 1] < (height + 1)) * (corresponding_pixs[:, :, :, 1] > -1)
    offsets = (corresponding_pixs - pixs)[:, :, :, :2]

    if record_occlusion and depth1 is not None:
        depth1_CHW = depth_reshape(depth1, mode="CHW")
        depth1_warp_to_view2 = torch_warp_simple(depth1_CHW, offsets)
        mask *= points3DC1[:, :, :, 2] < depth1_warp_to_view2[:, 0, :, :] * 1.05

    mask = mask.unsqueeze(1)

    if mask_by_radius:
        radius_mask = depth2 < (2 * radius.view(-1, 1, 1, 1))
        mask = torch.cat((mask, radius_mask), 1)

        if mask_by_radius_a_lot:
            radius_mask1 = depth2 < (1 * radius.view(-1, 1, 1, 1))
            radius_mask2 = depth2 < (3 * radius.view(-1, 1, 1, 1))
            radius_mask3 = depth2 < (4 * radius.view(-1, 1, 1, 1))
            radius_mask4 = depth2 < (5 * radius.view(-1, 1, 1, 1))
            mask = torch.cat((mask, radius_mask1, radius_mask2, radius_mask3, radius_mask4), 1)
    
    return offsets, mask

Backward_tensorGrid = {}
Backward_tensorGrid_cpu = {}

def torch_warp_simple(tensorInput, tensorFlow):
    key = f"{str(tensorFlow.size())}_{tensorInput.device}"

    if key not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(2), device=tensorInput.device, dtype=tensorInput.dtype).view(
            1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), tensorFlow.size(1), -1, -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(1), device=tensorInput.device, dtype=tensorInput.dtype).view(
            1, tensorFlow.size(1), 1, 1).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        Backward_tensorGrid[key] = torch.cat([tensorHorizontal, tensorVertical], -1)

    tensorFlow = torch.cat([tensorFlow[:, :, :, 0:1] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, :, :, 1:2] / ((tensorInput.size(2) - 1.0) / 2.0)], -1)

    grid = (Backward_tensorGrid[key] + tensorFlow)
    return torch.nn.functional.grid_sample(input=tensorInput,
                                           grid=grid,
                                           mode='bilinear',
                                           padding_mode='border',
                                           align_corners=True)

def downsample_offsets(offset, downsample_scales):
    downsampled_offsets = {}
    current_offset = offset
    
    for i, scale in enumerate(downsample_scales):
        # Calculate the new size
        batch_size, H, W, _ = current_offset.shape
        current_scale = downsample_scales[i] / downsample_scales[i-1] if i>0 else scale
        new_H = int(H // current_scale)
        new_W = int(W // current_scale)
        
        # Reshape the offset to (batch_size, 2, H, W) for interpolation
        current_offset_reshaped = current_offset.permute(0, 3, 1, 2)
        
        # Perform interpolation (downsampling)
        downsampled_offset = F.interpolate(current_offset_reshaped, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        # Adjust the offset values by the scale
        downsampled_offset = downsampled_offset / current_scale
        
        # Reshape back to (batch_size, new_H, new_W, 2)
        downsampled_offset = downsampled_offset.permute(0, 2, 3, 1)
        
        # Append the downsampled offset to the dict
        key = f'downsample_scale_{downsample_scales[i]}'
        downsampled_offsets[key] = downsampled_offset
        
        # Update the current offset for the next iteration
        current_offset = downsampled_offset
    
    return downsampled_offsets