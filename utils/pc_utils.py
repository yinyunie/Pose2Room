#  Copyright (c) 7.2021. Yinyu Nie
#  License: MIT

import numpy as np
from plyfile import PlyData, PlyElement
import trimesh
import seaborn as sns
from utils.virtualhome import LIMBS, valid_joint_ids
from utils.tools import get_box_corners


box_edge_ids = [[0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]]

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def get_3d_box(box_size, heading_angle, center):
    '''Get the bounding boxes from params'''
    R_mat = head2rot(heading_angle)
    vectors = np.diag(box_size / 2.).dot(R_mat)
    box3d_pts_3d = np.array(get_box_corners(center, vectors))
    return box3d_pts_3d

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def rot2head(R_mat):
    '''Rot_mat is consist with np.array([vx, vy, vz])
    vx is the heading vector, vy is the up-forward vector, vz is the right-forward vector.'''
    R_mat = np.array(R_mat)

    batch_flag = True
    if len(R_mat.shape) == 2:
        R_mat = R_mat[np.newaxis]
        batch_flag = False

    heading = np.arctan2(-R_mat[:, 0, 2], R_mat[:, 0, 0])

    if not batch_flag:
        heading = heading[0]
    return heading

def head2rot(heading):
    batch_flag = True
    if isinstance(heading, float) or isinstance(heading, int):
        heading = np.array([heading])
        batch_flag = False
    assert isinstance(heading, np.ndarray)
    assert len(heading.shape)==1

    R_mat = np.zeros(shape=(len(heading), 3, 3))
    R_mat[:, 0, 0] = np.cos(heading)
    R_mat[:, 0, 2] = -np.sin(heading)
    R_mat[:, 1, 1] = 1
    R_mat[:, 2, 0] = np.sin(heading)
    R_mat[:, 2, 2] = np.cos(heading)

    if not batch_flag:
        R_mat = R_mat[0]
    return R_mat

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_points(points, radius=0.05, color=(0.2, 0.2, 0.2), output_file=None):
    """
    input points: np array (n, 3)
    output_file: string
    """
    '''Initiate a Scene'''
    scene = trimesh.Scene()
    '''Add geometry'''
    for point in points:
        point_mesh = trimesh.creation.icosphere(subdivisions=1, radius=radius, color=color)
        point_mesh.vertices += point
        scene.add_geometry(point_mesh)
        del point_mesh
    '''Export geometry'''
    scene.export(output_file)

def write_oriented_bbox(box_params, sem_classes, all_class_labels, output_file=None, line_width=0.02):
    # set palette
    palette_cls = np.array([*sns.color_palette("hls", len(all_class_labels))])
    '''Initiate a Scene'''
    scene = trimesh.Scene()
    '''Add geometry'''
    for box_param, sem_cls in zip(box_params, sem_classes):
        corners_3d = get_3d_box(box_param[3:6], box_param[6], box_param[0:3])
        for edge_ids in box_edge_ids:
            p0 = corners_3d[edge_ids[0]]
            p1 = corners_3d[edge_ids[1]]
            line_mesh = trimesh.creation.cylinder(line_width, segment=np.array([p0, p1]), face_colors=palette_cls[sem_cls])
            scene.add_geometry(line_mesh)
            del line_mesh
        # orientation
        centroid = box_param[0:3]
        forward_point = (corners_3d[1]-corners_3d[0])/2. + centroid
        line_mesh = trimesh.creation.cylinder(line_width, segment=np.array([centroid, forward_point]), face_colors=[0.8, 0.1, 0.1])
        scene.add_geometry(line_mesh)
        del line_mesh

    '''Export geometry'''
    scene.export(output_file)

def write_joints(skeleton_joints, skip_rates=1, output_file=None, resolution=1):
    """
    input joints: np array (n, s, 3)
    output_file: string
    """
    '''Initiate a Scene'''
    scene = trimesh.Scene()
    '''Add geometry'''
    move_traj = skeleton_joints[:, 0]
    frame_num = skeleton_joints.shape[0] // skip_rates + 1
    movement_dist = np.linalg.norm(np.diff(move_traj, axis=0), axis=1)
    cum_dist = np.cumsum(np.hstack([[0], movement_dist]))
    target_cum_dist = np.linspace(0, sum(movement_dist), frame_num)
    selected_sk_ids = np.argmin(np.abs(cum_dist[:, np.newaxis] - target_cum_dist), axis=0)
    skeleton_joints = skeleton_joints[selected_sk_ids]
    ex_skeleton_joints = np.zeros([skeleton_joints.shape[0], max(valid_joint_ids) + 1, 3])
    ex_skeleton_joints[:, valid_joint_ids] = skeleton_joints

    skeleton_colors = sns.color_palette("Spectral_r", n_colors=frame_num)
    traj_palette = np.array(sns.color_palette("Spectral_r", n_colors=move_traj.shape[0]))
    for sk_idx, skeleton in enumerate(ex_skeleton_joints):
        # draw joints
        for jt_idx, joint in enumerate(skeleton):
            if jt_idx not in valid_joint_ids:
                continue
            if jt_idx == 10:
                radius = 0.1
            else:
                radius = 0.05
            joint_mesh = trimesh.creation.icosphere(subdivisions=resolution, radius=radius, color=skeleton_colors[sk_idx])
            joint_mesh.vertices += joint
            scene.add_geometry(joint_mesh)
            del joint_mesh

        # draw lines
        for line_idx, line in enumerate(LIMBS):
            p0 = skeleton[line[0]]
            p1 = skeleton[line[1]]
            line_mesh = trimesh.creation.cylinder(0.02, segment=np.array([p0, p1]), face_colors=skeleton_colors[sk_idx])
            scene.add_geometry(line_mesh)
            del line_mesh

    # draw directions
    for traj_id in range(move_traj.shape[0] - 1):
        if np.linalg.norm(move_traj[traj_id + 1] - move_traj[traj_id]) == 0.:
            continue
        line_mesh = trimesh.creation.cylinder(0.02, segment=np.array([move_traj[traj_id], move_traj[traj_id+1]]), face_colors=traj_palette[traj_id])
        scene.add_geometry(line_mesh)
        del line_mesh
    '''Export geometry'''
    scene.export(output_file)
