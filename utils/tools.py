#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT
import json
import os
import numpy as np
import h5py

voxel_faces = np.array([(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 4, 7, 3)], dtype=np.uint64)

def read_json(file):
    '''
    read json file
    @param file: file path.
    @return:
    '''
    with open(file, 'r') as f:
        output = json.load(f)
    return output


def write_json(file, data):
    '''
    read json file
    @param file: file path.
    @param data: dict content
    @return:
    '''
    assert os.path.exists(os.path.dirname(file))

    with open(file, 'w') as f:
        json.dump(data, f)

def get_box_corners(center, vectors):
    '''
    Convert box center and vectors to the corner-form
    :param center:
    :param vectors:
    :return: corner points related to the box
    '''
    corner_pnts = [None] * 8
    corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
    corner_pnts[1] = tuple(center + vectors[0] - vectors[1] - vectors[2])
    corner_pnts[2] = tuple(center + vectors[0] + vectors[1] - vectors[2])
    corner_pnts[3] = tuple(center - vectors[0] + vectors[1] - vectors[2])

    corner_pnts[4] = tuple(center - vectors[0] - vectors[1] + vectors[2])
    corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
    corner_pnts[6] = tuple(center + vectors[0] + vectors[1] + vectors[2])
    corner_pnts[7] = tuple(center - vectors[0] + vectors[1] + vectors[2])

    return corner_pnts

def grids_from_points(points, voxel_size, point_colors=None):
    '''
    create voxel grids from points
    @param points: N x 3 matrix;
    @param voxel_size: voxel size in meters;
    @param point_colors: color of each point.
    @return:
    '''
    origin = points.min(0)
    world2grid = np.eye(4)
    world2grid[:3, :3] = np.array([[1 / voxel_size, 0, 0], [0, 1 / voxel_size, 0], [0, 0, 1 / voxel_size]])
    world2grid[:3, 3] = - origin / voxel_size

    points_v = np.pad(points, ((0, 0), (0, 1)), 'constant', constant_values=((None, None), (None, 1)))
    points_v = np.uint16(points_v.dot(world2grid.T)[..., :3])

    dims = points_v.max(0) + 1
    voxel_grids = np.zeros(shape=dims, dtype=np.bool)
    voxel_grids[points_v[:, 0], points_v[:, 1], points_v[:, 2]] = True

    if point_colors is not None:
        color_grids = np.zeros(shape=dims, dtype=np.uint32)
        point_colors = point_colors.astype(np.uint32)
        color_grids[points_v[:, 0], points_v[:, 1], points_v[:, 2]] = np.left_shift(point_colors[:, 0], 16) +\
                                                                      np.left_shift(point_colors[:, 1], 8) +\
                                                                      point_colors[:, 2]
        return voxel_grids, world2grid, color_grids
    else:
        return voxel_grids, world2grid

def points_from_grids(voxel_grids, world2grid):
    '''
    return points in world system from voxel grids.
    @param voxel_grids:
    @param world2grid:
    @return:
    '''
    points_v = np.array(np.nonzero(voxel_grids)).T + 0.5 # centroid of a voxel
    points_v = np.pad(points_v, ((0, 0), (0, 1)), 'constant', constant_values=((None, None), (None, 1)))
    points_v = points_v.dot(np.linalg.inv(world2grid.T))
    return points_v[...,:3]

def ndarray2list(nodes):
    if isinstance(nodes, list) or isinstance(nodes, tuple):
        return [ndarray2list(node) for node in nodes]
    elif isinstance(nodes, dict):
        for key in nodes.keys():
            nodes[key] = ndarray2list(nodes[key])
        return nodes
    elif isinstance(nodes, np.ndarray):
        return nodes.tolist()
    elif isinstance(nodes, int) or isinstance(nodes, float) or isinstance(nodes, str):
        return nodes
    else:
        raise NotImplementedError

def write_data_to_hdf5(file_handle, name, data):
    if isinstance(data, list):
        if not len(data):
            file_handle.create_dataset(name,  data=h5py.Empty("i"))
        elif isinstance(data[0], int):
            file_handle.create_dataset(name, shape=(len(data),), dtype=np.int32, data=np.array(data))
        elif isinstance(data[0], float):
            file_handle.create_dataset(name, shape=(len(data),), dtype=np.float32, data=np.array(data))
        elif isinstance(data[0], str):
            asciiList = [item.encode("ascii", "ignore") for item in data]
            file_handle.create_dataset(name, shape=(len(asciiList),), dtype='S10', data=asciiList)
        elif isinstance(data[0], dict):
            group_data = file_handle.create_group(name)
            for node_idx, node in enumerate(data):
                write_data_to_hdf5(group_data, str(node_idx), node)
        else:
            raise NotImplementedError
    elif isinstance(data, int):
        file_handle.create_dataset(name, shape=(1,), dtype='i', data=data)
    elif isinstance(data, float):
        file_handle.create_dataset(name, shape=(1,), dtype='f', data=data)
    elif isinstance(data, str):
        dt = h5py.special_dtype(vlen=str)
        file_handle.create_dataset(name, shape=(1,), dtype=dt, data=data)
    elif isinstance(data, np.ndarray):
        file_handle.create_dataset(name, shape=data.shape, dtype=np.float32, data=data)
    elif isinstance(data, dict):
        group_data = file_handle.create_group(name)
        for key, value in data.items():
            write_data_to_hdf5(group_data, key, value)
    return

def normalize(a, axis=-1, order=2):
    '''
    Normalize any kinds of tensor data along a specific axis
    :param a: source tensor data.
    :param axis: data on this axis will be normalized.
    :param order: Norm order, L0, L1 or L2.
    :return:
    '''
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1

    if len(a.shape) == 1:
        return a / l2
    else:
        return a / np.expand_dims(l2, axis)

class Struct(object):
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
