#  Copyright (c) 7.2021. Yinyu Nie
#  License: MIT

import argparse
import numpy as np
from utils.virtualhome import dataset_config
from external.virtualhome.simulation.unity_simulator.utils_viz import get_skeleton
from utils.tools import read_json
from utils.virtualhome.vis_vhome import VIS_HOME

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Virtual Room Visualization.')
    parser.add_argument('--scene-id', type=int, default=4,
                        help='Give a scene id in [0-7].')
    parser.add_argument('--room-id', type=int, default=1,
                        help='Give a scene id in [0-N].')
    parser.add_argument('--char-id', type=int, default=1,
                        help='Give a character id in [0-5].')
    parser.add_argument('--script-id', type=int, default=0,
                        help='Give a script id.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # np.random.seed(dataset_config.random_seed)
    # random.seed(dataset_config.random_seed)

    '''read skeletons'''
    char_name = dataset_config.character_names[args.char_id]
    sk_input_path = dataset_config.recording_path.joinpath(str(args.scene_id), str(args.room_id), str(args.script_id),
                                                           char_name.split('/')[1])
    skeleton_joints, frames_ids = get_skeleton(sk_input_path, 'script')
    instance_ids = read_json(sk_input_path.parent.joinpath('instance_ids.json'))

    '''read bboxes'''
    # read object bbox
    output_scene_path = dataset_config.script_bbox_path.joinpath(str(args.scene_id))
    bbox_file = output_scene_path.joinpath('bbox_' + str(args.room_id) + '.json')
    nodes_for_det = read_json(bbox_file)
    nodes_for_det = [nodes_for_det[idx] for idx in instance_ids]

    # read room bbox
    room_bbox_file = output_scene_path.joinpath('room_bbox_' + str(args.room_id) + '.json')
    room_bbox = read_json(room_bbox_file)['room_bbox']

    '''Augment data with 2 flips and 4 rotations'''
    if_augment_data = True

    if if_augment_data:
        if np.random.randint(2):
            flip_matrix = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
            flip_matrix = np.repeat(flip_matrix[np.newaxis], skeleton_joints.shape[0], axis=0)
            # flip skeleton
            skeleton_joints = np.matmul(skeleton_joints, flip_matrix)
            # flip room bbox
            room_bbox['centroid'] = np.matmul(np.array(room_bbox['centroid']), flip_matrix[0])
            R_mat = np.array(room_bbox['R_mat']).dot(flip_matrix[0])
            R_mat[2] = np.cross(R_mat[0], R_mat[1])
            room_bbox['R_mat'] = R_mat
            # flip object bboxes
            for node in nodes_for_det:
                node['centroid'] = np.matmul(np.array(node['centroid']), flip_matrix[0])
                R_mat = np.array(node['R_mat']).dot(flip_matrix[0])
                R_mat[2] = np.cross(R_mat[0], R_mat[1])
                node['R_mat'] = R_mat

        # rotation
        rot_num = np.random.randint(4)
        rot90 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        rot90 = np.linalg.matrix_power(rot90, rot_num)  # counter clockwise
        rot90 = np.repeat(rot90[np.newaxis], skeleton_joints.shape[0], axis=0)
        # rotate skeleton
        skeleton_joints = np.matmul(skeleton_joints, rot90)
        # rotate room bbox
        room_bbox['centroid'] = np.matmul(np.array(room_bbox['centroid']), rot90[0])
        room_bbox['R_mat'] = np.array(room_bbox['R_mat']).dot(rot90[0])
        # rotate object bboxes
        for node in nodes_for_det:
            node['centroid'] = np.matmul(np.array(node['centroid']), rot90[0])
            node['R_mat'] = np.array(node['R_mat']).dot(rot90[0])

    '''visualize bboxes'''
    viser = VIS_HOME(nodes=nodes_for_det, room_bbox=room_bbox, skeleton_joints=skeleton_joints)
    viser.visualize(type=['bboxes', 'room_bbox', 'skeleton'])
