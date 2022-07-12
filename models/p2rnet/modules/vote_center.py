#  Copyright (c) 8.2021. Yinyu Nie
#  License: MIT

import torch
import torch.nn as nn
from models.registers import MODULES
from models.p2rnet.modules.sub_modules import SingleConv


@MODULES.register_module
class CenterVoteModule(nn.Module):
    def __init__(self, cfg, optim_spec = None):
        '''
        Skeleton Extraction Net to obtain partial skeleton from a partial scan (refer to PointNet++).
        :param config: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(CenterVoteModule, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Modules'''
        self.origin_joint_id = cfg.dataset_config.origin_joint_id
        self.vote_factor = cfg.config['data']['vote_factor']
        in_dim = 256
        self.out_dim = in_dim # due to residual feature, in_dim has to be == out_dim
        self.conv_input = nn.Sequential(
            SingleConv(in_dim, 256, kernel_size=1, order='cbr', num_groups=8, padding=0, ndim=1),
            SingleConv(256, 256, kernel_size=1, order='cbr', num_groups=8, padding=0, ndim=1),
            SingleConv(256, (3 + self.out_dim) * self.vote_factor, kernel_size=1, order='c', num_groups=8, padding=0,
                       ndim=1))

    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        seed_xyz = seed_xyz[:, :, self.origin_joint_id]
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed * self.vote_factor
        net = self.conv_input(seed_features.transpose(1, 2)) # (batch_size, (3+out_dim)*vote_factor, num_seed)

        net = net.transpose(2, 1).view(batch_size, num_seed, self.vote_factor, 3 + self.out_dim)
        offset = net[:, :, :, 0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)

        residual_features = net[:, :, :, 3:]  # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)

        return vote_xyz, vote_features.contiguous()
