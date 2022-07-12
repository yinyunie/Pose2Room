#  Trainer for P2RNet.
#  Copyright (c) 5.2021. Yinyu Nie
#  License: MIT

from models.training import BaseTrainer
import torch
from torch import nn
from net_utils.nn_distance import nn_distance, huber_loss
from net_utils import utils
criterion_box_sem_cls = nn.BCEWithLogitsLoss(reduction='none')
criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')

class Trainer(BaseTrainer):
    '''
    Trainer object for total3d.
    '''

    def eval_step(self, data):
        '''
        performs a step in evaluation
        :param data (dict): data dictionary
        :return:
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net(data)

        '''compute losses'''
        loss = self.net.module.loss(est_data, data)

        # for logging
        loss_reduced = utils.reduce_dict(loss)
        loss_dict = {k: v.item() for k, v in loss_reduced.items()}
        return loss_dict

    def eval_nn_loss(self, pred_data, gt_data):
        if isinstance(pred_data, tuple):
            pred_data = pred_data[0]

        # Compute center loss
        pred_center = pred_data['center']
        gt_center = gt_data['center_label']
        box_label_mask = gt_data['box_label_mask']
        centroid_reg_loss1 = 0.
        centroid_reg_loss2 = 0.
        object_assignment = []
        for per_pred_center, per_gt_center, per_box_label_mask in zip(pred_center, gt_center, box_label_mask):
            dist1, ind1, dist2, _ = nn_distance(per_pred_center.unsqueeze(0),
                                                per_gt_center[per_box_label_mask > 0].unsqueeze(0),
                                                l1smooth=True)  # dist1: 1xK, dist2: 1xK2
            centroid_reg_loss1 += torch.sum(dist1)
            centroid_reg_loss2 += torch.sum(dist2)
            object_assignment.append(ind1)
        # Set assignment
        object_assignment = torch.cat(object_assignment, dim=0)  # (B,K) with values in 0,1,...,K2-1
        n_batch = object_assignment.size(0)
        n_proposals = object_assignment.size(1)
        centroid_reg_loss1 = centroid_reg_loss1 / (n_batch * n_proposals + 1e-6)
        centroid_reg_loss2 = centroid_reg_loss2 / (torch.sum(box_label_mask) + 1e-6)
        center_loss = (centroid_reg_loss1 + centroid_reg_loss2)/2.

        # Compute size loss
        gt_size = torch.gather(gt_data['size'], 1,
                               object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
        pred_size = pred_data['size']
        size_loss = torch.mean(huber_loss(pred_size - gt_size, delta=1.0), -1)
        size_loss = torch.mean(size_loss)

        # Computer heading loss
        gt_heading = torch.gather(gt_data['heading'], 1,
                                  object_assignment.unsqueeze(-1).repeat(1, 1, 2))  # select (B,K) from (B,K2)
        pred_heading = pred_data['heading']
        heading_loss = torch.mean(huber_loss(pred_heading - gt_heading, delta=1.0), -1)
        heading_loss = torch.mean(heading_loss)

        # Computer class loss
        gt_cls_label = torch.gather(gt_data['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        pred_cls_scores = pred_data['sem_cls_scores']
        sem_cls_loss = criterion_sem_cls(pred_cls_scores.transpose(2,1), gt_cls_label)
        sem_cls_loss = torch.mean(sem_cls_loss)
        loss = 10 * center_loss + 10 * size_loss + 10 * heading_loss + sem_cls_loss

        return {'total':loss,
                'center_loss': center_loss.item(),
                'size_loss': size_loss.item(),
                'heading_loss': heading_loss.item(),
                'sem_cls_loss': sem_cls_loss.item(),
                'centroid_reg_loss1': centroid_reg_loss1.item(),
                'centroid_reg_loss2': centroid_reg_loss2.item()}

    def visualize_step(self, epoch, phase, iter, data):
        ''' Performs a visualization step.
        '''
        if not self.cfg.config['device']['is_main_process']:
            return
        pass

    def to_device(self, data):
        device = self.device
        for key in data:
            if key in ['sample_idx']: continue
            data[key] = data[key].to(device)
        return data

    def compute_loss(self, data):
        '''
        compute the overall loss.
        :param data (dict): data dictionary
        :return:
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net(data)

        '''compute losses'''
        loss = self.net.module.loss(est_data, data)
        return loss
