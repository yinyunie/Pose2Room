#  Copyright (c) 7.2021. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
import argparse
import h5py
from utils.virtualhome import dataset_config, LIMBS, valid_joint_ids
from utils.vis_base import VIS_BASE
import seaborn as sns
import numpy as np
import random
import vtk

def dist_node2bbox(nodes, joint_coordinates, joint_num):
    sk_ids = []
    for node in nodes:
        vecs = joint_coordinates - node['centroid']
        dist_offset = np.abs(vecs.dot(node['R_mat'].T)) - np.array(node['size']) / 2.
        dists = dist_offset.max(axis=-1)
        dists = np.max(dists.reshape(-1, joint_num), axis=-1)
        sk_ids.append(dists.argmin())
    return np.sort(sk_ids)

def get_even_dist_joints(skeleton_joints, skip_rates):
    # Downsampling by 1-d distance interpolation
    frame_num = skeleton_joints.shape[0] // skip_rates + 1
    movement_dist = np.linalg.norm(np.diff(skeleton_joints[:, 0], axis=0), axis=1)
    cum_dist = np.cumsum(np.hstack([[0], movement_dist]))
    target_cum_dist = np.linspace(0, sum(movement_dist), frame_num)
    selected_sk_ids = np.argmin(np.abs(cum_dist[:, np.newaxis] - target_cum_dist), axis=0)
    return selected_sk_ids

class VIS_GT(VIS_BASE):
    def __init__(self, nodes=(), cam_locs=(), points_world=(), room_bbox=None, skeleton_joints=None,
                 skeleton_joint_votes=None, skeleton_mask=None, keep_interact_skeleton=False, skip_rates=1):
        super(VIS_GT, self).__init__()
        self.nodes = nodes
        self.cam_locs = cam_locs
        self.room_bbox = room_bbox
        if len(nodes):
            self.class_ids = [node['class_id'][0] for node in nodes]
            # set palette
            self.palette_cls = np.array([*sns.color_palette("hls", len(dataset_config.class_labels))])
        if isinstance(skeleton_joints, np.ndarray):
            self.move_traj = skeleton_joints[:, 0]
            selected_sk_ids = range(skeleton_joints.shape[0])
            if skip_rates > 1 and not keep_interact_skeleton:
                selected_sk_ids = get_even_dist_joints(skeleton_joints, skip_rates)
                skeleton_joints = skeleton_joints[selected_sk_ids]
                skeleton_mask = skeleton_mask[selected_sk_ids]
                skeleton_joint_votes = skeleton_joint_votes[selected_sk_ids]
            elif keep_interact_skeleton:
                joint_coordinates = skeleton_joints.reshape(-1, 3)
                # get distance between joint to nodes
                selected_sk_ids = dist_node2bbox(self.nodes, joint_coordinates, dataset_config.joint_num)
                # add more frames close to it.
                if skip_rates == 1:
                    local_sk_ids = np.arange(-50, 50, skip_rates)[np.newaxis]
                    selected_sk_ids = selected_sk_ids[:, np.newaxis] + local_sk_ids
                    selected_sk_ids = selected_sk_ids.flatten()
                    selected_sk_ids = selected_sk_ids[selected_sk_ids < skeleton_joints.shape[0]]
                    selected_sk_ids = np.sort(selected_sk_ids)
                else:
                    local_sk_ids = np.arange(-50, 50)[np.newaxis]
                    piece_sk_ids = selected_sk_ids[:, np.newaxis] + local_sk_ids
                    even_dist_sk_ids = [selected_sk_ids]
                    for per_piece_sk_ids in piece_sk_ids:
                        per_piece_sk_ids = per_piece_sk_ids[per_piece_sk_ids < skeleton_joints.shape[0]]
                        picked_ids = get_even_dist_joints(skeleton_joints[per_piece_sk_ids], skip_rates)
                        even_dist_sk_ids.append(per_piece_sk_ids[picked_ids])
                    selected_sk_ids = np.sort(np.hstack(even_dist_sk_ids))
                skeleton_joints = skeleton_joints[selected_sk_ids]
                skeleton_mask = skeleton_mask[selected_sk_ids]
                skeleton_joint_votes = skeleton_joint_votes[selected_sk_ids]
            self.skeleton_joints = np.zeros([skeleton_joints.shape[0], max(valid_joint_ids) + 1, 3])
            self.skeleton_joints[:, valid_joint_ids] = skeleton_joints
            self.skeleton_joint_votes = np.zeros([skeleton_joint_votes.shape[0], max(valid_joint_ids) + 1, 10])
            self.skeleton_joint_votes[:, valid_joint_ids] = skeleton_joint_votes
            self.skeleton_mask = skeleton_mask
            self.frame_num = skeleton_joints.shape[0]
            self.traj_palette = np.array(sns.color_palette("Spectral_r", n_colors=self.move_traj.shape[0]))
            self.skeleton_colors = self.traj_palette[selected_sk_ids]

        self.point_cloud = points_world

    def set_render(self, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        # '''draw world system'''
        # renderer.AddActor(self.set_axes_actor())

        cam_fp = (self.move_traj.max(0) + self.move_traj.min(0))/2.
        cam_loc = cam_fp + kwargs.get('cam_centroid', [7,7,7])
        cam_up = [0, sum((cam_loc - cam_fp) ** 2) / (cam_loc[1] - cam_fp[1]), 0] + cam_fp - cam_loc
        camera = self.set_camera(cam_loc, cam_fp, cam_up, self.cam_K)
        renderer.SetActiveCamera(camera)

        '''draw 3D boxes'''
        if 'room_bbox' in kwargs['type']:
            # draw room bbox
            centroid = self.room_bbox['centroid']
            vectors = np.diag(np.array(self.room_bbox['size']) / 2.).dot(self.room_bbox['R_mat'])
            color = [125, 125, 125]
            box_actor = self.get_bbox_cube_actor(centroid, vectors, color, 0.05)
            box_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(box_actor)

        if 'bboxes' in kwargs['type']:
            # draw instance bboxes
            for node_idx, node in enumerate(self.nodes):
                centroid = node['centroid']
                vectors = np.diag(np.array(node['size']) / 2.).dot(node['R_mat'])
                color = self.palette_cls[self.class_ids[node_idx]] * 255
                box_actor = self.get_bbox_line_actor(centroid, vectors, color, 1., 6)
                box_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(box_actor)

                # draw orientations
                color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]

                for index in range(vectors.shape[0]):
                    arrow_actor = self.set_arrow_actor(centroid, vectors[index])
                    arrow_actor.GetProperty().SetColor(color[index])
                    renderer.AddActor(arrow_actor)

        '''draw camera locations'''
        if 'cam_locs' in kwargs['type']:
            for cam_loc in self.cam_locs:
                sphere_actor = self.set_actor(
                    self.set_mapper(self.set_sphere_property(cam_loc[0], 0.1), mode='model'))
                sphere_actor.GetProperty().SetColor([0.8, 0.1, 0.1])
                sphere_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(sphere_actor)

        '''draw scene mesh'''
        if 'mesh' in kwargs['type']:
            ply_actor = self.set_actor(self.set_mapper(self.set_ply_property('./temp/tsdf_mesh_vhome.ply'), 'model'))
            ply_actor.GetProperty().SetOpacity(1)
            ply_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(ply_actor)

        if 'voxels' in kwargs['type']:
            ply_actor = self.set_actor(self.set_mapper(self.set_ply_property('./temp/voxel_mesh_vhome.ply'), 'model'))
            ply_actor.GetProperty().SetOpacity(1)
            ply_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(ply_actor)

        if 'points' in kwargs['type']:
            '''render points'''
            points = np.vstack(self.point_cloud['pc'])
            colors = np.vstack(self.point_cloud['color'])
            point_size = 3

            point_actor = self.set_actor(self.set_mapper(self.set_points_property(points, colors), 'box'))
            point_actor.GetProperty().SetPointSize(point_size)
            point_actor.GetProperty().SetOpacity(1)
            point_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(point_actor)

        if 'skeleton' in kwargs['type']:
            '''render skeleton joints'''
            for sk_idx, skeleton in enumerate(self.skeleton_joints):
                if self.skeleton_mask[sk_idx] == 0:
                    continue
                else:
                    opacity = 1
                # draw joints
                for jt_idx, joint in enumerate(skeleton):
                    if jt_idx not in valid_joint_ids:
                        continue
                    if jt_idx == 10:
                        radius = 0.1
                    else:
                        radius = 0.05

                    sphere_actor = self.set_actor(
                        self.set_mapper(self.set_sphere_property(joint, radius), mode='model'))
                    sphere_actor.GetProperty().SetColor(self.skeleton_colors[sk_idx])
                    sphere_actor.GetProperty().SetOpacity(opacity)
                    sphere_actor.GetProperty().SetInterpolationToPBR()
                    renderer.AddActor(sphere_actor)

                # draw lines
                for line_idx, line in enumerate(LIMBS):
                    p0 = skeleton[line[0]]
                    p1 = skeleton[line[1]]
                    line_actor = self.set_actor(self.set_mapper(self.set_line_property(p0, p1), mode='model'))
                    line_actor.GetProperty().SetLineWidth(6)
                    line_actor.GetProperty().SetColor(self.skeleton_colors[sk_idx])
                    line_actor.GetProperty().SetOpacity(opacity)
                    line_actor.GetProperty().SetInterpolationToPBR()
                    renderer.AddActor(line_actor)

        # draw joint votes
        if 'joint_votes' in kwargs['type']:
            for sk_idx, skeleton in enumerate(self.skeleton_joints):
                if self.skeleton_mask[sk_idx] == 0:
                    continue
                for jt_idx, joint in enumerate(skeleton):
                    if jt_idx != dataset_config.origin_joint_id:
                        continue
                    if not self.skeleton_joint_votes[sk_idx][jt_idx, 0]:
                        continue  # pass those non-vote joint

                    votes = self.skeleton_joint_votes[sk_idx][jt_idx, 1:]
                    votes = np.unique(votes.reshape((3, 3)), axis=0)
                    for vote in votes:
                        # draw vote directions
                        arrow_actor = self.set_arrow_actor(joint, joint + vote, mode='endpoint', tip_len_ratio=0.1,
                                                           tip_r_ratio=0.015, shaft_r_ratio=0.008)
                        arrow_actor.GetProperty().SetColor(self.skeleton_colors[sk_idx])
                        arrow_actor.GetProperty().SetOpacity(0.1)
                        arrow_actor.GetProperty().SetInterpolationToPBR()
                        renderer.AddActor(arrow_actor)

        # draw directions
        for traj_id in range(self.move_traj.shape[0] - 1):
            if np.linalg.norm(self.move_traj[traj_id + 1] - self.move_traj[traj_id]) == 0.:
                continue
            line_actor = self.set_actor(
                self.set_mapper(self.set_line_property(self.move_traj[traj_id], self.move_traj[traj_id + 1]),
                                mode='model'))
            line_actor.GetProperty().SetLineWidth(5)
            line_actor.GetProperty().SetColor(self.traj_palette[traj_id])
            line_actor.GetProperty().SetOpacity(1)
            line_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(line_actor)

        '''light'''
        focal_point = np.array([0., 0., 0.])
        positions = focal_point + np.array([(100, 100, 100), (100, 100, -100), (-100, 100, 100), (-100, 100, -100)])
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(0.8)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(*focal_point)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)
        return renderer

def augment_data(room_bbox, skeleton_joints, object_nodes, skeleton_joint_votes, flip_matrix, rot_func, offset_func):
    '''Augment samples with flip, rotation and offset'''
    if_flip = random.randint(0, 1)
    rot_angle = random.uniform(-np.pi, np.pi)
    offset_scale = random.uniform(-1., 1.)
    rot_mat = rot_func(rot_angle)
    offset = offset_func(offset_scale)

    n_frames, n_joints = skeleton_joint_votes.shape[:2]
    '''begin to augment'''
    '''begin to flip'''
    if if_flip:
        # flip skeleton
        skeleton_joints = np.dot(skeleton_joints, flip_matrix)
        # flip votes
        votes = skeleton_joint_votes[..., 1:].reshape(n_frames, n_joints, 3, 3)
        votes = np.dot(votes, flip_matrix)
        votes = votes.reshape(n_frames, n_joints, 9)
        skeleton_joint_votes[..., 1:] = votes
        # flip room bbox
        room_bbox['centroid'] = np.matmul(np.array(room_bbox['centroid']), flip_matrix)
        R_mat = np.array(room_bbox['R_mat']).dot(flip_matrix)
        R_mat[2] = np.cross(R_mat[0], R_mat[1])
        room_bbox['R_mat'] = R_mat
        # flip object bboxes
        for node in object_nodes:
            node['centroid'] = np.dot(np.array(node['centroid']), flip_matrix)
            R_mat = np.array(node['R_mat']).dot(flip_matrix)
            R_mat[2] = np.cross(R_mat[0], R_mat[1])
            node['R_mat'] = R_mat

    '''begin to rotate'''
    # rotate votes
    point_votes_end = np.zeros_like(skeleton_joint_votes)
    point_votes_end[..., 1:4] = np.dot(skeleton_joints[..., 0:3] + skeleton_joint_votes[..., 1:4], rot_mat)
    point_votes_end[..., 4:7] = np.dot(skeleton_joints[..., 0:3] + skeleton_joint_votes[..., 4:7], rot_mat)
    point_votes_end[..., 7:10] = np.dot(skeleton_joints[..., 0:3] + skeleton_joint_votes[..., 7:10], rot_mat)

    # rotate skeleton
    skeleton_joints = np.dot(skeleton_joints, rot_mat)
    skeleton_joint_votes[..., 1:4] = point_votes_end[..., 1:4] - skeleton_joints[..., 0:3]
    skeleton_joint_votes[..., 4:7] = point_votes_end[..., 4:7] - skeleton_joints[..., 0:3]
    skeleton_joint_votes[..., 7:10] = point_votes_end[..., 7:10] - skeleton_joints[..., 0:3]

    # rotate room bbox
    room_bbox['centroid'] = np.dot(np.array(room_bbox['centroid']), rot_mat)
    room_bbox['R_mat'] = np.array(room_bbox['R_mat']).dot(rot_mat)
    # rotate object bboxes
    for node in object_nodes:
        node['centroid'] = np.dot(np.array(node['centroid']), rot_mat)
        node['R_mat'] = np.array(node['R_mat']).dot(rot_mat)

    '''begin to translate'''
    # translate skeleton
    skeleton_joints += offset

    # translate room bbox
    room_bbox['centroid'] += offset

    # translate object nodes
    for node in object_nodes:
        node['centroid'] += offset

    return room_bbox, object_nodes, skeleton_joints, skeleton_joint_votes

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Virtual Room Visualization.')
    parser.add_argument('--scene-id', type=int, default=3,
                        help='Give a scene id in [0-7].')
    parser.add_argument('--room-id', type=int, default=0,
                        help='Give a scene id in [0-N].')
    parser.add_argument('--script-id', type=int, default=0,
                        help='Give a script id.')
    parser.add_argument('--char-id', type=int, default=1,
                        help='Give a character id in [0-5].')
    parser.add_argument('--augment', action='store_true',
                        help='If implement data augmentation.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    char_name = dataset_config.character_names[args.char_id].split('/')[1]
    sample_file = '%d_%d_%d_%s_0.hdf5' % (args.scene_id, args.room_id, args.script_id, char_name)
    sample_file = dataset_config.sample_path.joinpath(sample_file)

    '''read data'''
    sample_data = h5py.File(sample_file, "r")
    room_bbox = {}
    for key in sample_data['room_bbox'].keys():
        room_bbox[key] = sample_data['room_bbox'][key][:]

    skeleton_joints = sample_data['skeleton_joints'][:]
    skeleton_joint_votes = sample_data['skeleton_joint_votes'][:]

    object_nodes = []
    for idx in range(len(sample_data['object_nodes'])):
        object_node = {}
        node_data = sample_data['object_nodes'][str(idx)]
        for key in node_data.keys():
            if node_data[key].shape is None:
                continue
            object_node[key] = node_data[key][:]
        object_nodes.append(object_node)

    '''augment data'''
    if args.augment:
        flip_matrix = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        rot_func = lambda theta: np.array([[np.cos(theta), 0., -np.sin(theta)],
                                        [0., 1., 0.],
                                        [np.sin(theta), 0, np.cos(theta)]])
        offset_func = lambda scale: np.array([1., 0., 1.]) * scale
        room_bbox, object_nodes, skeleton_joints, skeleton_joint_votes = augment_data(room_bbox, skeleton_joints,
                                                                                      object_nodes, skeleton_joint_votes,
                                                                                      flip_matrix, rot_func, offset_func)
    vote_mask = skeleton_joint_votes[..., 0, 0]

    '''visualize bboxes'''
    viser = VIS_GT(nodes=object_nodes, room_bbox=room_bbox, skeleton_joints=skeleton_joints,
                   skeleton_joint_votes=skeleton_joint_votes, skeleton_mask=vote_mask, keep_interact_skeleton=True, skip_rates=10)
    viser.visualize(type=['bboxes', 'room_bbox', 'skeleton'])


