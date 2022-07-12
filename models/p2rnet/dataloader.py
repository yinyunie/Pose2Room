#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from models.datasets import Base_Dataset
import h5py
import os
from utils.pc_utils import rot2head
import random

default_collate = torch.utils.data.dataloader.default_collate


class P2RNet_VirtualHome(Base_Dataset):
    def __init__(self,cfg, mode):
        super(P2RNet_VirtualHome, self).__init__(cfg, mode)
        self.aug = mode == 'train'
        self.num_frames = cfg.config['data']['num_frames']
        self.use_height = not cfg.config['data']['no_height']
        self.max_num_obj = cfg.config['data']['max_gt_boxes']
        if self.aug:
            self.flip_matrix = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
            self.rot_func = lambda theta: np.array([[np.cos(theta), 0., -np.sin(theta)],
                                                    [0., 1., 0.],
                                                    [np.sin(theta), 0, np.cos(theta)]])
            self.offset_func = lambda scale: np.array([1., 0., 1.]) * scale

    def augment_data(self, skeleton_joints, object_nodes, skeleton_joint_votes):
        '''Augment training data'''
        if_flip = random.randint(0, 1)
        rot_angle = np.random.choice([-np.pi, -0.5 * np.pi, 0, 0.5 * np.pi])
        offset_scale = random.uniform(-1., 1.)
        rot_mat = self.rot_func(rot_angle)
        offset = self.offset_func(offset_scale)

        n_frames, n_joints = skeleton_joint_votes.shape[:2]
        '''begin to augment'''
        if if_flip:
            '''begin to flip'''
            # flip skeleton
            skeleton_joints = np.dot(skeleton_joints, self.flip_matrix)
            # flip votes
            votes = skeleton_joint_votes[..., 1:].reshape(n_frames, n_joints, 3, 3)
            votes = np.dot(votes, self.flip_matrix)
            votes = votes.reshape(n_frames, n_joints, 9)
            skeleton_joint_votes[..., 1:] = votes
            # flip object bboxes
            for node in object_nodes:
                node['centroid'] = np.dot(np.array(node['centroid']), self.flip_matrix)
                R_mat = np.array(node['R_mat']).dot(self.flip_matrix)
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
        # rotate object bboxes
        for node in object_nodes:
            node['centroid'] = np.dot(np.array(node['centroid']), rot_mat)
            node['R_mat'] = np.array(node['R_mat']).dot(rot_mat)

        '''begin to translate'''
        # translate skeleton
        skeleton_joints += offset
        # translate object nodes
        for node in object_nodes:
            node['centroid'] += offset

        return skeleton_joints, object_nodes, skeleton_joint_votes

    def __getitem__(self, idx):
        '''Get each sample'''
        '''Load data'''
        data_path = self.split[idx]
        sample_data = h5py.File(data_path, "r")
        skeleton_joints = sample_data['skeleton_joints'][:]
        object_nodes = sample_data['object_nodes']
        skeleton_joint_votes = sample_data['skeleton_joint_votes'][:]
        instances = []
        for instance_id in object_nodes.keys():
            object_node = object_nodes[instance_id]
            instance = {'class_id': object_node['class_id'][0], 'centroid': object_node['centroid'][:],
                        'R_mat': object_node['R_mat'][:], 'size': object_node['size'][:]}
            instances.append(instance)
        sample_data.close()

        '''Augment data'''
        if self.aug:
            skeleton_joints, instances, skeleton_joint_votes = self.augment_data(skeleton_joints, instances,
                                                                                 skeleton_joint_votes)
        boxes3D = []
        classes = []
        for instance in instances:
            heading = rot2head(instance['R_mat'])
            box3D = np.hstack([instance['centroid'], np.log(instance['size']), np.sin(heading), np.cos(heading)])
            boxes3D.append(box3D)
            classes.append(instance['class_id'])

        boxes3D = np.array(boxes3D)

        if self.use_height:
            floor_height = np.percentile(skeleton_joints[..., 1], 0.99)
            height = skeleton_joints[..., 1] - floor_height
            skeleton_joints = np.concatenate([skeleton_joints, np.expand_dims(height, -1)], -1)

        target_bboxes_mask = np.zeros((self.max_num_obj))
        target_bboxes_semcls = np.zeros((self.max_num_obj))
        centers = np.zeros((self.max_num_obj, 3))
        sizes = np.zeros((self.max_num_obj, 3))
        headings = np.zeros((self.max_num_obj, 2))

        # store GT in containers
        target_bboxes_mask[0:boxes3D.shape[0]] = 1
        target_bboxes_semcls[0:boxes3D.shape[0]] = classes
        centers[0:boxes3D.shape[0], :] = boxes3D[:, 0:3]
        sizes[0:boxes3D.shape[0], :] = boxes3D[:, 3:6]
        headings[0:boxes3D.shape[0], :] = boxes3D[:, 6:8]

        # Process input frames
        joint_ids = np.linspace(0, skeleton_joints.shape[0]-1, self.num_frames).round().astype(np.uint16)
        input_joints = skeleton_joints[joint_ids]
        input_joint_votes = skeleton_joint_votes[joint_ids, :, 1:]
        joint_votes_mask = skeleton_joint_votes[joint_ids, :, 0]

        # deliver to network
        ret_dict = {}
        ret_dict['input_joints'] = input_joints.astype(np.float32)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['center_label'] = centers.astype(np.float32)
        ret_dict['size'] = sizes.astype(np.float32)
        ret_dict['heading'] = headings.astype(np.float32)
        ret_dict['vote_label'] = input_joint_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = joint_votes_mask.astype(np.int64)
        ret_dict['sample_idx'] = '.'.join(os.path.basename(data_path).split('.')[:-1])
        return ret_dict

def collate_fn(batch):
    '''
    data collater
    :param batch:
    :return:
    '''
    collated_batch = {}
    for key in batch[0]:
        if key not in ['sample_idx']:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
        else:
            collated_batch[key] = [elem[key] for elem in batch]
    return collated_batch

class Custom_Dataloader(object):
    def __init__(self, dataloader, sampler):
        self.dataloader = dataloader
        self.sampler = sampler

# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Init datasets and dataloaders
def P2RNet_dataloader(cfg, mode='train'):
    if cfg.config['data']['dataset'] == 'virtualhome':
        dataset = P2RNet_VirtualHome(cfg, mode)
    else:
        raise NotImplementedError

    if cfg.config['device']['distributed']:
        sampler = DistributedSampler(dataset, shuffle=(mode == 'train'))
    else:
        if mode=='train':
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=cfg.config[mode]['batch_size'],
                                                  drop_last=False)

    dataloader = DataLoader(dataset=dataset,
                            batch_sampler=batch_sampler,
                            num_workers=cfg.config['device']['num_workers'],
                            collate_fn=collate_fn,
                            worker_init_fn=my_worker_init_fn)

    dataloader = Custom_Dataloader(dataloader, sampler)
    return dataloader
