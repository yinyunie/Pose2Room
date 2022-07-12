#  Paths of data used in this projects.
#  Copyright (c) 4.2021. Yinyu Nie
#  License: MIT
from pathlib import Path

import numpy as np
from utils.tools import read_json

class Dataset_Config(object):
    def __init__(self, dataset):
        if dataset == 'virtualhome':
            '''Data generation'''
            self.root_path = Path('datasets/virtualhome_22_classes')
            self.scene_num = 7
            self.joint_num = 53
            self.origin_joint_id = 0 # the idx of hip joint
            self.script_bbox_path = self.root_path.joinpath('script_bbox')
            self.failed_script_log = self.root_path.joinpath('failed_script_log.txt')
            self.recording_path = self.root_path.joinpath('recording')
            self.scene_geo_path = self.root_path.joinpath('scenes')
            self.sample_path = self.root_path.joinpath('samples')
            self.split_path = self.root_path.joinpath('splits')
            self.split_ratio = {'script_level':
                                    {'train': 0.8,
                                     'val': 0.2},
                                'char_level':
                                    {'train':4./5.,
                                     'val': 1./5.},
                                'room_level':
                                    {'train': 14./15.,
                                     'val': 1./15.}}
            self.split_level = 'room_level'
            self.frame_rate = 5
            self.im_size = [640, 480]
            self.pixel_sample_rate = 5
            self.far_clip = 15.
            self.voxel_size = 0.0625
            self.crop_size = np.array([32, 32, 32])
            self.keep_point_cloud = False
            self.category_labels = ['Furniture', 'Windows', 'Electronics', 'Appliances', 'Lamps']
            self.object_props = {'CAN_OPEN', 'HAS_SWITCH', 'SITTABLE', 'SURFACES'}
            self.class_labels_raw = ['bathtub', 'bench', 'nightstand', 'desk', 'closet',
                                     'bathroomcabinet', 'toilet', 'kitchencabinet', 'sofa', 'cabinet',
                                     'garbagecan', 'bookshelf', 'chair', 'bed', 'faucet',
                                     'window', 'tv', 'computer', 'washingmachine', 'fridge',
                                     'dishwasher', 'stove', 'microwave', 'tablelamp']
            self.class_labels = ['bathtub', 'bed', 'bench', 'bookshelf', 'cabinet',
                                 'chair', 'closet', 'desk', 'dishwasher', 'faucet',
                                 'fridge', 'garbagecan', 'lamp', 'microwave', 'monitor',
                                 'nightstand', 'sofa', 'stove', 'toilet', 'washingmachine',
                                 'window', 'computer']
            self.category_not_render = {'Ceiling', 'Walls', 'Doors'}
            self.class_mapping = [0, 2, 15, 7, 6, 4, 18, 4, 16, 4, 11, 3, 5, 1, 9, 20, 14, 21, 19, 10, 8, 17, 13, 12]
            self.category_mapping = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 3, 4]
            self.prior_path = self.split_path.joinpath(self.split_level).joinpath('avg_data.json')
            self.contact_dist_thresh = 1.0 # maximal distance between person and contacted object.
            self.cam_loc_sample_step = 1.5
            self.cam_range_padding = 1.
            self.cam_angle_sample_step = 90
            self.n_seq_per_room = 100
            self.n_inst_per_room = 10
            self.character_names = ['Chars/Male1', 'Chars/Female2', 'Chars/Female4', 'Chars/Male10', 'Chars/Male2']
            self.random_seed = 2
            self.unity_lauch_cmd = ["external/virtualhome/simulation/unity_simulator/linux_exec.v2.2.3.x86_64",
                                    "-screen-fullscreen", "0", "-screen-quality", "3", "-screen-width", "640",
                                    "-screen-height", "480"]
            if not self.root_path.is_dir():
                self.root_path.mkdir()
            if not self.script_bbox_path.is_dir():
                self.script_bbox_path.mkdir()
            if not self.recording_path.is_dir():
                self.recording_path.mkdir()
            if not self.sample_path.is_dir():
                self.sample_path.mkdir()
            if not self.split_path.is_dir():
                self.split_path.mkdir()

            '''For training'''
            self.num_class = len(self.class_labels)
            self.num_heading_bin = 12
            self.num_size_cluster = len(self.class_labels)
            self.type2class = {cls: index for index, cls in enumerate(self.class_labels)}
            self.class2type = {self.type2class[t]: t for t in self.type2class}
            if self.prior_path.is_file():
                prior_data = read_json(self.prior_path)
                self.mean_size_arr = np.zeros(shape=(self.num_class, 3))
                for cls_label, avg_value in prior_data['obj_size_cls_avg'].items():
                    self.mean_size_arr[int(cls_label)] = avg_value
                self.type_mean_size = {}
                for i in range(self.num_size_cluster):
                    self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i, :]
        else:
            raise NotImplementedError

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle % (2 * np.pi)
        assert False not in (angle >= 0) * (angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = np.int16(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr[pred_cls, :] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle
        return obb



if __name__ == '__main__':
    dataset_config = Dataset_Config('GTA_IM')
