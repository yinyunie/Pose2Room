#  generate samples with 8 augmentations for each room
#  Copyright (c) 7.2021. Yinyu Nie
#  License: MIT
import sys
sys.path.append('.')
from utils.virtualhome import dataset_config
from external.virtualhome.simulation.unity_simulator.utils_viz import get_skeleton
from utils.tools import read_json, write_data_to_hdf5
import numpy as np
import h5py
from copy import deepcopy
from utils.virtualhome.vhome_utils import class_mapping, category_mapping
from utils.tools import get_box_corners
from utils.pc_utils import extract_pc_in_box3d
from multiprocessing import Pool
import argparse
from utils.virtualhome.vhome_utils import check_in_box

def augment_sample(room_bbox, object_nodes, skeleton_joints, flip_matrix, rot90, aug_idx):
    '''Augment samples with flip and rotations'''
    if_flip = False if aug_idx <= 3 else True
    rot_num = aug_idx % 4

    # begin to augment
    if if_flip:
        # flip skeleton
        skeleton_joints = np.matmul(skeleton_joints, flip_matrix)
        # flip room bbox
        room_bbox['centroid'] = np.matmul(np.array(room_bbox['centroid']), flip_matrix[0])
        R_mat = np.array(room_bbox['R_mat']).dot(flip_matrix[0])
        R_mat[2] = np.cross(R_mat[0], R_mat[1])
        room_bbox['R_mat'] = R_mat

        # flip object bboxes
        for node in object_nodes:
            node['centroid'] = np.matmul(np.array(node['centroid']), flip_matrix[0])
            R_mat = np.array(node['R_mat']).dot(flip_matrix[0])
            R_mat[2] = np.cross(R_mat[0], R_mat[1])
            node['R_mat'] = R_mat

    rot90 = np.linalg.matrix_power(rot90, rot_num)  # counter clockwise
    rot90 = np.repeat(rot90[np.newaxis], skeleton_joints.shape[0], axis=0)
    # rotate skeleton
    skeleton_joints = np.matmul(skeleton_joints, rot90)
    # rotate room bbox
    room_bbox['centroid'] = np.matmul(np.array(room_bbox['centroid']), rot90[0])
    room_bbox['R_mat'] = np.array(room_bbox['R_mat']).dot(rot90[0])
    # rotate object bboxes
    for node in object_nodes:
        node['centroid'] = np.matmul(np.array(node['centroid']), rot90[0])
        node['R_mat'] = np.array(node['R_mat']).dot(rot90[0])

    return room_bbox, object_nodes, skeleton_joints


def get_votes(object_node, all_joints, joint_votes, indices, joint_vote_idx):
    '''Get votes for each joint respect to a bbox'''
    # get the contact space of the object bbox
    centroid = object_node['centroid']
    vectors = np.diag(np.array(object_node['size']) / 2. + dataset_config.contact_dist_thresh).dot(object_node['R_mat'])
    box3d_pts_3d = np.array(get_box_corners(centroid, vectors))

    # Find all points close to this object's bbox
    pc_in_box3d, inds = extract_pc_in_box3d(all_joints[..., :3], box3d_pts_3d)
    # Assign first dimension to indicate it is in an object box
    joint_votes[inds, 0] = 1
    # Add the votes (all 0 if the point is not in any object's OBB)
    votes = np.expand_dims(centroid, 0) - pc_in_box3d[:, 0:3]
    sparse_inds = indices[inds]  # turn dense True,False inds to sparse number-wise inds
    for i in range(len(sparse_inds)):
        j = sparse_inds[i]
        joint_votes[j, int(joint_vote_idx[j] * 3 + 1):int((joint_vote_idx[j] + 1) * 3 + 1)] = votes[i, :]
        # Populate votes with the fisrt vote
        if joint_vote_idx[j] == 0:
            joint_votes[j, 4:7] = votes[i, :]
            joint_votes[j, 7:10] = votes[i, :]
    joint_vote_idx[inds] = np.minimum(2, joint_vote_idx[inds] + 1)

    return joint_votes, joint_vote_idx


def run(sk_file):
    print('Processing %s.' % (sk_file))
    # read skeletons
    sk_input_path = sk_file.parent.parent.parent
    skeleton_joints, frames_ids = get_skeleton(sk_input_path, 'script')

    # read node id list
    instance_ids = read_json(sk_input_path.parent.joinpath('instance_ids.json'))

    # read room bbox
    scene_id, room_id, script_id, char_name = str(sk_file).split('/')[3:7]
    room_bbox_file = dataset_config.script_bbox_path.joinpath(scene_id, 'room_bbox_%s.json' % (room_id))
    room_bbox = read_json(room_bbox_file)['room_bbox']

    # In some cases, the initial pose are not in the same room with objects. we cut off these initial poses.
    in_box_flags = check_in_box(skeleton_joints[:, dataset_config.origin_joint_id], room_bbox)
    if True not in in_box_flags:
        print('=' * 50)
        print('Animation in: %s does not make sense.' % (sk_file))
        print('=' * 50)
        return
    skeleton_joints = skeleton_joints[in_box_flags.tolist().index(True):]

    # object bboxes
    object_nodes_file = dataset_config.script_bbox_path.joinpath(scene_id, 'bbox_%s.json' % (room_id))
    object_nodes = read_json(object_nodes_file)
    object_nodes = [object_nodes[idx] for idx in instance_ids]

    # In some cases the animation are wrong, no poses will pass by any objects
    in_any_instance_box_flags = False
    for object_node in object_nodes:
        dummy_node = object_node.copy()
        dummy_node['size'] = np.array(dummy_node['size']) + 2 * dataset_config.contact_dist_thresh
        in_instance_flag = check_in_box(skeleton_joints[:, dataset_config.origin_joint_id], dummy_node)
        if True in in_instance_flag :
            in_any_instance_box_flags = True
            break

    if in_any_instance_box_flags == False:
        print('=' * 50)
        print('Animation in: %s does not make sense.' % (sk_file))
        print('=' * 50)
        return

    '''Place the center of world system at the bottom centroid of room bbox'''
    room_centroid = np.array(room_bbox['centroid'])
    room_centroid[1] = room_centroid[1] - room_bbox['size'][1]/2
    room_bbox['centroid'] = (room_bbox['centroid'] - room_centroid)

    for node in object_nodes:
        node['centroid'] = (node['centroid'] - room_centroid)

    skeleton_joints = skeleton_joints - room_centroid

    '''Class mapping'''
    for node in object_nodes:
        class_id, class_name = class_mapping([node['class_name']], return_class_names=True)
        class_id = class_id[0]
        class_name = class_name[0]
        category_id, category_name = category_mapping([node['class_name']], return_cateory_names=True)
        category_id = category_id[0]
        category_name = category_name[0]
        node['class_id'] = class_id
        node['class_name'] = class_name
        node['category_id'] = category_id
        node['category'] = category_name

    '''Augment and Save data'''
    base_name = '_'.join([scene_id, room_id, script_id, char_name])
    flip_matrix = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    flip_matrix = np.repeat(flip_matrix[np.newaxis], skeleton_joints.shape[0], axis=0)
    rot90 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    # Augment data with flip and rotate (0 denotes no augmentation)
    if args.no_augment:
        aug_indices = [0]
    else:
        aug_indices = range(8)

    for aug_idx in aug_indices:
        save_file = dataset_config.sample_path.joinpath(base_name + '_%d.hdf5' % (aug_idx))
        if save_file.is_file():
            print('File exists: %s.' % (save_file))
            continue

        room_bbox_sample = deepcopy(room_bbox)
        object_nodes_sample = deepcopy(object_nodes)
        skeleton_joints_sample = deepcopy(skeleton_joints)

        room_bbox_sample, object_nodes_sample, skeleton_joints_sample = augment_sample(room_bbox_sample,
                                                                                       object_nodes_sample,
                                                                                       skeleton_joints_sample,
                                                                                       flip_matrix, rot90, aug_idx)

        '''get votes for each skeleton joint'''
        n_frames, n_joints = skeleton_joints_sample.shape[:2]
        all_joints = skeleton_joints_sample.reshape(n_frames * n_joints, 3)
        N = n_frames * n_joints  # all joint number
        joint_votes = np.zeros((N, 10))  # 3 votes and 1 vote mask
        joint_vote_idx = np.zeros((N)).astype(np.int32)  # in the range of [0,2]
        indices = np.arange(N)

        for object_node in object_nodes_sample:
            joint_votes, joint_vote_idx = get_votes(object_node, all_joints, joint_votes, indices, joint_vote_idx)

        skeleton_joint_votes = joint_votes.reshape(n_frames, n_joints, joint_votes.shape[-1])
        file_handle = h5py.File(save_file, "w")
        write_data_to_hdf5(file_handle, name='skeleton_joints', data=skeleton_joints_sample)
        write_data_to_hdf5(file_handle, name='skeleton_joint_votes', data=skeleton_joint_votes)
        write_data_to_hdf5(file_handle, name='room_bbox', data=room_bbox_sample)
        write_data_to_hdf5(file_handle, name='object_nodes', data=object_nodes_sample)
        file_handle.close()

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Generate programs in a room.')
    parser.add_argument('--no-augment', action='store_true',
                        help='if not augment data in data preparation phase.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    all_sk_files = dataset_config.recording_path.rglob('pd_script.txt')
    pool = Pool(processes=32)
    pool.map(run, all_sk_files)
    pool.close()
    pool.join()
