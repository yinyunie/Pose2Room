# Proposal module.
# author: ynie
# date: March, 2020
# cite: VoteNet
import numpy as np
import torch
import torch.nn as nn
from models.registers import MODULES
from models.p2rnet.modules.sub_modules import SingleConv
from external.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModuleVotes
from utils.tools import Struct
from models.p2rnet.modules.mdn import CategoryEmbeddingMDN
from net_utils.libs import farthest_point_sample, index_points

def decode_scores(pred_center, pred_size, pred_heading, sem_obj_feature, end_points):
    sem_obj_feature_transposed = sem_obj_feature.transpose(2, 1)  # (batch_size, 1024, ..)

    # center
    base_xyz = end_points['aggregated_vote_xyz']  # (batch_size, num_proposal, 3)
    center = base_xyz + pred_center.transpose(2, 1)  # (batch_size, num_proposal, 3)
    end_points['center'] = center

    # size
    end_points['size'] = pred_size.transpose(2, 1)

    # heading (sin, cos)
    end_points['heading'] = pred_heading.transpose(2, 1)

    # objectness
    end_points['objectness_scores'] = sem_obj_feature_transposed[..., 0:2]

    # classification scores
    end_points['sem_cls_scores'] = sem_obj_feature_transposed[..., 2:]
    return end_points


@MODULES.register_module
class ProposalNet(nn.Module):
    def __init__(self, cfg, optim_spec = None):
        '''
        Skeleton Extraction Net to obtain partial skeleton from a partial scan (refer to PointNet++).
        :param config: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(ProposalNet, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.cfg = cfg
        '''Parameters'''
        self.num_class = cfg.dataset_config.num_class
        self.num_proposals = cfg.config['data']['num_target']
        self.sampling = cfg.config['data']['cluster_sampling']
        seed_feature_dim = 256
        vote_dim = 256
        if cfg.config['mode'] != 'train':
            self.multi_mode = cfg.eval_config['multi_mode']
            n_ranges = np.arange(1, 100)
            self.n_samples = np.random.choice(n_ranges, 1)[0]

        '''Network modules'''
        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
                npoint=self.num_proposals,
                radius=0.3,
                nsample=16,
                mlp=[seed_feature_dim, 256, vote_dim],
                use_xyz=False,
                normalize_xyz=True,
                bn=False
            )

        '''Prepare for Gaussian Mixture Models'''
        # Object proposal/detection
        # objectness (2), heading center (3), size (3), heading (2), class num
        sem_obj_dim = 2 + self.num_class
        gmm_dim = 128
        self.conv_center = nn.Sequential(SingleConv(vote_dim, 128, kernel_size=1, order='cbr', num_groups=8, padding=0, ndim=1),
                                         SingleConv(128, gmm_dim, kernel_size=1, order='cbr', num_groups=8, padding=0, ndim=1))
        self.conv_heading = nn.Sequential(SingleConv(vote_dim, 128, kernel_size=1, order='cbr', num_groups=8, padding=0, ndim=1),
                                         SingleConv(128, gmm_dim, kernel_size=1, order='cbr', num_groups=8, padding=0, ndim=1))
        self.conv_size = nn.Sequential(SingleConv(vote_dim, 128, kernel_size=1, order='cbr', num_groups=8, padding=0, ndim=1),
                                       SingleConv(128, gmm_dim, kernel_size=1, order='cbr', num_groups=8, padding=0, ndim=1))
        self.conv_sem_obj = nn.Sequential(SingleConv(vote_dim, 128, kernel_size=1, order='cbr', num_groups=8, padding=0, ndim=1),
                                          SingleConv(128, 128, kernel_size=1, order='cbr', num_groups=8, padding=0, ndim=1),
                                          SingleConv(128, sem_obj_dim, kernel_size=1, order='c', num_groups=8, padding=0, ndim=1))

        # load gaussian mixture model for center, size, heading
        self.gmm_center = self.load_gmm(num_gaussian=cfg.config['data']['num_gaussian'], in_dim=gmm_dim, out_dim=3,
                                        type='center')
        self.gmm_size = self.load_gmm(num_gaussian=cfg.config['data']['num_gaussian'], in_dim=gmm_dim, out_dim=3,
                                      type='size')
        self.gmm_heading = self.load_gmm(num_gaussian=cfg.config['data']['num_gaussian'], in_dim=gmm_dim, out_dim=2,
                                         type='heading')

    def init_mu(self, num_gaussian, type):
        init_mu = None
        if type == 'center':
            n_bins_theta = np.ceil(np.sqrt(num_gaussian/2)).astype(np.uint16)
            n_bins_phi = 2 * n_bins_theta
            bin_width = np.pi / n_bins_theta
            phi_grids = [bin_width * idx - np.pi for idx in range(0, n_bins_phi)]
            theta_grids = np.linspace(0, np.pi, n_bins_theta + 2)[1:-1]
            grids = np.array(np.meshgrid(phi_grids, theta_grids)).reshape(2, -1).T

            init_center = np.hstack([0.1 * np.sin(grids[:,[1]]) * np.cos(grids[:,[0]]),
                                     0.1 * np.sin(grids[:,[1]]) * np.sin(grids[:,[0]]),
                                     0.1 * np.cos(grids[:,[1]])])
            init_mu = torch.from_numpy(init_center)
            if num_gaussian < init_mu.size(0):
                init_mu = self.get_farthest_points(init_mu, npoint=num_gaussian)
        elif type == 'size':
            bins_per_dim = np.ceil(num_gaussian ** (1/3)).astype(np.uint32)
            grids_per_dim = np.linspace(0.05, 3, bins_per_dim)
            size_grids = np.log(np.array(np.meshgrid(grids_per_dim,grids_per_dim,grids_per_dim)).reshape(3, -1).T)
            size_grids = torch.from_numpy(size_grids)
            init_mu = self.get_farthest_points(size_grids, npoint=num_gaussian)
        elif type == 'heading':
            bin_width = 2 * np.pi / num_gaussian
            thetas = [bin_width * idx - np.pi for idx in range(0, num_gaussian)]
            init_mu = np.array([[np.sin(theta), np.cos(theta)] for theta in thetas])
            init_mu = torch.from_numpy(init_mu)
        return init_mu

    def get_farthest_points(self, xyz, npoint):
        if len(xyz.size()) == 2:
            xyz = xyz.unsqueeze(0)
        xyz = xyz.float()
        inds = farthest_point_sample(xyz, npoint)
        inds = torch.sort(inds, dim=-1)[0]
        new_xyz = index_points(xyz, inds)
        if new_xyz.size(0) == 1:
            new_xyz = new_xyz.squeeze(0)
        return new_xyz

    def load_gmm(self, num_gaussian, in_dim, out_dim, type):
        '''load a gmm model'''
        # initialize centers (mu) for each single gaussian dist.
        init_mu = self.init_mu(num_gaussian, type)

        # set config file
        mdn_config = Struct(num_gaussian=num_gaussian, out_dim=out_dim, mu_bias_init=init_mu, n_samples=1,
                            central_tendency='mean')
        config = Struct(embedding_dims=[], out_dim=3, continuous_dim=in_dim, batch_norm_continuous_input=False,
                        hidden_dim=128, mdn_config=mdn_config)

        return CategoryEmbeddingMDN(config)


    def forward(self, xyz, features, end_points, export_proposal_feature=False):
        """
        Args:
            xyz: (B,K,3)
            features: (B,K,C)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
        n_batch, num_seeds, _ = xyz.shape
        device = xyz.device
        features = features.transpose(1, 2).contiguous()
        if self.sampling == 'vote_fps':
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds, arg_ids = torch.sort(fps_inds, dim=-1)
            xyz = torch.gather(xyz, 1, arg_ids.unsqueeze(-1).repeat(1, 1, xyz.size(2)))
            features = torch.gather(features, 2, arg_ids.unsqueeze(1).repeat(1, features.size(1), 1))
        elif self.sampling == 'seed_fps':
            seed_xyz = end_points['seed_xyz']
            movement_dist = torch.norm(torch.diff(seed_xyz, dim=1), dim=2)
            cum_dist = torch.cumsum(torch.cat([torch.zeros(size=(n_batch, 1)).to(device), movement_dist], dim=1), dim=1)
            step_len = cum_dist[:, -1] / (self.num_proposals - 1)
            target_cum_dist = step_len.unsqueeze(-1) * torch.arange(self.num_proposals, dtype=torch.float).to(device)
            sample_inds = torch.argmin(torch.abs(cum_dist.unsqueeze(-1) - target_cum_dist.unsqueeze(1)), dim=1)
            sample_inds = sample_inds.type(torch.int32)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            raise NotImplementedError('Undefined sampling strategy.')

        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds.type(torch.int64) # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- Gaussian Mixture Prediction ---------
        # get backbone features from different metrics
        center_feature = self.conv_center(features)
        size_feature = self.conv_size(features)
        heading_feature = self.conv_heading(features)
        sem_obj_feature = self.conv_sem_obj(features)

        # gaussian mixture models
        pred_center = self.gmm_center.predict(center_feature)
        pred_size = self.gmm_size.predict(size_feature)
        pred_heading = self.gmm_heading.predict(heading_feature)

        end_points = decode_scores(pred_center, pred_size, pred_heading, sem_obj_feature, end_points)

        if export_proposal_feature:
            return end_points, features.transpose(1,2).contiguous()
        else:
            return end_points, None

    def generate(self, xyz, features, end_points, export_proposal_feature=False):
        """
        Args:
            xyz: (B,K,3)
            features: (B,K,C)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """
        n_batch, num_seeds, _ = xyz.shape
        device = xyz.device
        features = features.transpose(1, 2).contiguous()
        if self.sampling == 'vote_fps':
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds, arg_ids = torch.sort(fps_inds, dim=-1)
            xyz = torch.gather(xyz, 1, arg_ids.unsqueeze(-1).repeat(1, 1, xyz.size(2)))
            features = torch.gather(features, 2, arg_ids.unsqueeze(1).repeat(1, features.size(1), 1))
        elif self.sampling == 'seed_fps':
            seed_xyz = end_points['seed_xyz']
            movement_dist = torch.norm(torch.diff(seed_xyz, dim=1), dim=2)
            cum_dist = torch.cumsum(torch.cat([torch.zeros(size=(n_batch, 1)).to(device), movement_dist], dim=1), dim=1)
            step_len = cum_dist[:, -1] / (self.num_proposals - 1)
            target_cum_dist = step_len.unsqueeze(-1) * torch.arange(self.num_proposals, dtype=torch.float).to(device)
            sample_inds = torch.argmin(torch.abs(cum_dist.unsqueeze(-1) - target_cum_dist.unsqueeze(1)), dim=1)
            sample_inds = sample_inds.type(torch.int32)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            raise NotImplementedError('Undefined sampling strategy.')

        end_points['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points['aggregated_vote_inds'] = sample_inds.type(torch.int64) # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- Gaussian Mixture Prediction ---------
        # get backbone features from different metrics
        center_feature = self.conv_center(features)
        size_feature = self.conv_size(features)
        heading_feature = self.conv_heading(features)
        sem_obj_feature = self.conv_sem_obj(features)

        # gaussian mixture models
        pred_center, pi_center = self.gmm_center.generate(center_feature, return_pi=True, multi_modes=self.multi_mode, n_samples=self.n_samples)
        pred_size, pi_size = self.gmm_size.generate(size_feature, return_pi=True, multi_modes=self.multi_mode, n_samples=self.n_samples)
        pred_heading, pi_heading = self.gmm_heading.generate(heading_feature, return_pi=True, multi_modes=self.multi_mode, n_samples=self.n_samples)

        end_points = decode_scores(pred_center, pred_size, pred_heading, sem_obj_feature, end_points)

        end_points['pi'] = {'center': pi_center,
                            'size': pi_size,
                            'heading': pi_heading}

        if export_proposal_feature:
            return end_points, features.transpose(1, 2).contiguous()
        else:
            return end_points, None
