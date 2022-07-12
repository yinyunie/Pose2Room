#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

import torch
import torch.nn as nn
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if hasattr(m, 'weight') and hasattr(m.weight, 'data'):
            torch.nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and hasattr(m.weight, 'data'):
            torch.nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            torch.nn.init.constant_(m.bias.data, 0.0)

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    return dct_m

def crops2mesh(poses, batch_voxel_crops, dataset_config, batch_crop_bnds=None):
    '''merge voxel crops to scene volume.'''
    voxel_size = dataset_config.voxel_size
    n_batch = poses.shape[0]

    if batch_crop_bnds is None:
        crop_size = dataset_config.crop_size
        sk_bbox_centers = poses[:, :, dataset_config.origin_joint_id].clone()
        sk_bbox_centers = sk_bbox_centers / voxel_size
        sk_bbox_centers = sk_bbox_centers.long()
        x_centers = sk_bbox_centers[..., 0]
        y_centers = sk_bbox_centers[..., 1]
        z_centers = sk_bbox_centers[..., 2]
        x_lbs, x_ubs = x_centers - crop_size[0] // 2, x_centers + crop_size[0] // 2
        y_lbs, y_ubs = y_centers - crop_size[1] // 2, y_centers + crop_size[1] // 2
        z_lbs, z_ubs = z_centers - crop_size[2] // 2, z_centers + crop_size[2] // 2
        batch_crop_bnds = torch.cat(
            [x_lbs.unsqueeze(-1), x_ubs.unsqueeze(-1), y_lbs.unsqueeze(-1), y_ubs.unsqueeze(-1), z_lbs.unsqueeze(-1),
             z_ubs.unsqueeze(-1)], dim=-1)

    '''merge volume crops together'''
    volume_origins = []
    scene_volumes = []
    for b_id in range(n_batch):
        crop_bnds = batch_crop_bnds[b_id]
        voxel_crops = batch_voxel_crops[b_id]
        volume_lb = torch.cat([crop_bnds[:, :2].min().unsqueeze(0), crop_bnds[:, 2:4].min().unsqueeze(0),
                               crop_bnds[:, 4:6].min().unsqueeze(0)], dim=0)
        new_volume_origin = volume_lb * dataset_config.voxel_size
        voxel_template = torch.zeros(size=(crop_bnds[:, :2].max() - crop_bnds[:, :2].min(),
                                           crop_bnds[:, 2:4].max() - crop_bnds[:, 2:4].min(),
                                           crop_bnds[:, 4:6].max() - crop_bnds[:, 4:6].min()))
        for crop_bnd, voxel_crop in zip(crop_bnds, voxel_crops):
            x_lb, x_ub, y_lb, y_ub, z_lb, z_ub = crop_bnd

            voxel_template[x_lb - volume_lb[0]: x_ub - volume_lb[0],
            y_lb - volume_lb[1]: y_ub - volume_lb[1],
            z_lb - volume_lb[2]: z_ub - volume_lb[2]] = voxel_crop[0]

        volume_origins.append(new_volume_origin)
        scene_volumes.append(voxel_template)

    return scene_volumes, volume_origins, batch_crop_bnds

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[...,2] *= -1
    return pc2

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def get_kmeans_mu(x, n_centers, init_times=50, min_delta=1e-3):
    """
    Find an initial value for the mean. Requires a threshold min_delta for the k-means algorithm to stop iterating.
    The algorithm is repeated init_times often, after which the best centerpoint is returned.
    args:
        x:            torch.FloatTensor (n, d) or (n, 1, d)
        init_times:   init
        min_delta:    int
    """
    if len(x.size()) == 3:
        x = x.squeeze(1)
    x_min, x_max = x.min(), x.max()
    x = (x - x_min) / (x_max - x_min)

    min_cost = np.inf

    for i in range(init_times):
        tmp_center = x[np.random.choice(np.arange(x.shape[0]), size=n_centers, replace=False), ...]
        l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - tmp_center), p=2, dim=2)
        l2_cls = torch.argmin(l2_dis, dim=1)

        cost = 0
        for c in range(n_centers):
            cost += torch.norm(x[l2_cls == c] - tmp_center[c], p=2, dim=1).mean()

        if cost < min_cost:
            min_cost = cost
            center = tmp_center

    delta = np.inf

    while delta > min_delta:
        l2_dis = torch.norm((x.unsqueeze(1).repeat(1, n_centers, 1) - center), p=2, dim=2)
        l2_cls = torch.argmin(l2_dis, dim=1)
        center_old = center.clone()

        for c in range(n_centers):
            center[c] = x[l2_cls == c].mean(dim=0)

        delta = torch.norm((center_old - center), dim=1).max()

    return (center * (x_max - x_min) + x_min)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N, dtype=xyz.dtype).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
