#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT
import torch
from torch import nn
from models.registers import LOSSES
from net_utils.nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.1, 0.9] # put larger weights on positive objectness

criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
criterion_size_class = nn.CrossEntropyLoss(reduction='none')
criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
bce_loss = nn.BCEWithLogitsLoss(reduction='none')


class BaseLoss(object):
    '''base loss class'''
    def __init__(self, weight=1, device=0, cfg=None):
        '''initialize loss module'''
        super(BaseLoss, self).__init__()
        self.weight = weight
        self.device = device
        self.origin_joint_id = cfg.dataset_config.origin_joint_id

@LOSSES.register_module
class Null(BaseLoss):
    '''This loss function is for modules where a loss preliminary calculated.'''
    def __call__(self, loss):
        return self.weight * torch.mean(loss)

@LOSSES.register_module
class BoxNetDetectionLoss(BaseLoss):
    '''base loss class'''
    def __init__(self, weight, device, cfg=None):
        '''initialize loss module'''
        super(BoxNetDetectionLoss, self).__init__(weight, device, cfg)
        self.objectness_criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).to(self.device), reduction='none')

    def compute_box_and_sem_cls_loss(self, est_data, gt_data, meta_data, config):
        """ Compute 3D bounding box and semantic classification loss.

        Args:
            est_data, gt_data, meta_data: dict (read-only)

        Returns:
            center_loss
            heading_cls_loss
            heading_reg_loss
            size_cls_loss
            size_reg_loss
            sem_cls_loss
        """
        # get assignment
        object_assignment = meta_data['object_assignment']
        objectness_label = meta_data['objectness_label'].float()

        # Compute center loss
        pred_center = est_data['center']
        gt_center = gt_data['center_label']
        box_label_mask = gt_data['box_label_mask']
        dist1, _, dist2, _ = nn_distance(pred_center, gt_center)  # dist1: BxK, dist2: BxK2
        centroid_reg_loss1 = torch.sum(dist1 * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        centroid_reg_loss2 = torch.sum(dist2 * box_label_mask) / (torch.sum(box_label_mask) + 1e-6)
        center_loss = (centroid_reg_loss1 + centroid_reg_loss2) / 2.

        # Compute size loss
        gt_size = torch.gather(gt_data['size'], 1,
                               object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
        pred_size = est_data['size']
        size_loss = torch.mean(huber_loss(pred_size - gt_size, delta=1.0), -1)
        size_loss = torch.sum(size_loss * objectness_label) / (torch.sum(objectness_label)+1e-6)

        # Computer heading loss
        gt_heading = torch.gather(gt_data['heading'], 1,
                                  object_assignment.unsqueeze(-1).repeat(1, 1, 2))  # select (B,K) from (B,K2)
        pred_heading = est_data['heading']
        heading_loss = torch.mean(huber_loss(pred_heading - gt_heading, delta=1.0), -1)
        heading_loss = torch.sum(heading_loss * objectness_label) / (torch.sum(objectness_label)+1e-6)

        # Computer class loss
        gt_cls_label = torch.gather(gt_data['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        pred_cls_scores = est_data['sem_cls_scores']
        sem_cls_loss = criterion_sem_cls(pred_cls_scores.transpose(2,1), gt_cls_label)
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
        return center_loss, size_loss, heading_loss, sem_cls_loss

    def compute_vote_loss(self, est_data, gt_data):
        '''Compute the vote loss for centroid proposals'''
        '''Vote loss is to make sure the vote xyz close to each gt centroid.'''
        batch_size, num_seed, num_joints = est_data['seed_skeleton'].shape[:3]
        vote_xyz = est_data['vote_xyz']  # B,num_seed*vote_factor,3
        seed_inds = est_data['seed_inds'].long()  # B,num_seed in [0,num_points-1]

        seed_gt_votes_mask = torch.gather(gt_data['vote_label_mask'][..., self.origin_joint_id], 1, seed_inds)
        seed_inds_expand = seed_inds.view(batch_size, num_seed, 1).repeat(1, 1, 3 * GT_VOTE_FACTOR)
        seed_gt_votes = torch.gather(gt_data['vote_label'][:,:,self.origin_joint_id], 1, seed_inds_expand)

        seed_gt_votes = seed_gt_votes.view(batch_size, num_seed, GT_VOTE_FACTOR, 3)
        seed_gt_votes = est_data['seed_skeleton'][:, :, [self.origin_joint_id]] + seed_gt_votes
        seed_gt_votes = seed_gt_votes.view(batch_size * num_seed, GT_VOTE_FACTOR, 3)
        seed_skeleton = est_data['seed_skeleton'].view(batch_size * num_seed, num_joints, 3)
        dist1, ind1, dist2, ind2 = nn_distance(seed_gt_votes, seed_skeleton)
        vote_argmin_inds = torch.gather(ind2, dim=1, index=dist2.argmin(-1).unsqueeze(-1)).view(batch_size, num_seed, 1)

        seed_gt_votes = seed_gt_votes.view(batch_size, num_seed, GT_VOTE_FACTOR, 3)
        seed_gt_votes = torch.gather(seed_gt_votes, 2, vote_argmin_inds.unsqueeze(-1).repeat(1, 1, 1, 3))
        seed_gt_votes = seed_gt_votes.squeeze(2)

        # Compute the min of min of distance
        vote_loss = torch.mean(huber_loss(vote_xyz - seed_gt_votes, delta=1.0), -1)
        vote_loss = torch.sum(vote_loss * seed_gt_votes_mask.float()) / (torch.sum(seed_gt_votes_mask.float()) + 1e-6)
        return vote_loss

    def compute_correspondence(self, est_data, gt_data):
        """
        Associate proposal and GT objects by point-to-point distances
        """
        '''Get assignment'''
        aggregated_vote_xyz = est_data['aggregated_vote_xyz']
        gt_center = gt_data['center_label'][:, :, 0:3]
        box_label_mask = gt_data['box_label_mask']
        dist1 = []
        object_assignment = []
        for per_aggregated_vote_xyz, per_gt_center, per_mask in zip(aggregated_vote_xyz, gt_center, box_label_mask):
            per_dist1, per_ind1, _, _ = nn_distance(per_aggregated_vote_xyz.unsqueeze(0),
                                                    per_gt_center[per_mask > 0].unsqueeze(0))  # dist1: BxK, dist2: BxK2
            dist1.append(per_dist1)
            object_assignment.append(per_ind1)
        dist1 = torch.cat(dist1, dim=0)
        object_assignment = torch.cat(object_assignment, dim=0)

        '''Set GT objectness label'''
        B = gt_center.shape[0]
        K = aggregated_vote_xyz.shape[1]
        euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
        objectness_label = torch.zeros((B, K), dtype=torch.long).to(self.device)
        objectness_mask = torch.zeros((B, K)).to(self.device)
        objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
        objectness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
        objectness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1

        '''Compute objectness loss'''
        objectness_scores = est_data['objectness_scores']
        objectness_loss = self.objectness_criterion(objectness_scores.transpose(2, 1), objectness_label)
        objectness_loss = torch.sum(objectness_loss * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)

        return object_assignment, objectness_loss, objectness_label, objectness_mask

    def __call__(self, est_data, gt_data, dataset_config):
        '''Loss functions'''
        '''Vote loss'''
        vote_loss = self.compute_vote_loss(est_data, gt_data)

        '''For each pred, compute objectness loss and assign the gt'''
        object_assignment, objectness_loss, objectness_label, objectness_mask = self.compute_correspondence(est_data,
                                                                                                            gt_data)
        meta_data = {'object_assignment':object_assignment,
                     'objectness_label': objectness_label}

        '''Box loss and sem cls loss'''
        center_loss, size_loss, heading_loss, sem_cls_loss = self.compute_box_and_sem_cls_loss(est_data, gt_data,
                                                                                               meta_data,
                                                                                               dataset_config)
        loss = 10 * vote_loss + 5 * objectness_loss + 10 * center_loss + 10 * size_loss + 10 * heading_loss + sem_cls_loss


        '''Other statistics'''
        total_num_proposal = objectness_label.shape[0] * objectness_label.shape[1]
        pos_ratio = \
            torch.sum(objectness_label.float().to(self.device)) / float(total_num_proposal)
        neg_ratio = \
            torch.sum(objectness_mask.float()) / float(total_num_proposal) - pos_ratio
        obj_pred_val = torch.argmax(est_data['objectness_scores'], 2)  # B,K
        obj_acc = torch.sum((obj_pred_val == objectness_label.long()).float() * objectness_mask) / (
                    torch.sum(objectness_mask) + 1e-6)

        return {'total':loss,
                'vote_loss': vote_loss,
                'objectness_loss': objectness_loss,
                'center_loss': center_loss,
                'size_loss': size_loss,
                'heading_loss': heading_loss,
                'sem_cls_loss': sem_cls_loss,
                'pos_ratio': pos_ratio,
                'neg_ratio': neg_ratio,
                'obj_acc': obj_acc}
